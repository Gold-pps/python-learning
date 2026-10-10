"""编排层：把“检索 → 盘点 → 组织 → 校验”包装成一次**可展示**的问答。

为什么单开一层，而不是把 Web 需要的东西塞进 `rag/review_qa.py`：

1. `review_qa.py` 是**算法层**，服务对象是 CLI 与离线评测，产物是四段式 payload。
   Web 额外要的全是"给界面看的"——每步耗时、每次调用的 usage、候选原文、
   引用与候选的对应关系。塞进算法层，CLI 的 `_render` 就得跟着改，而且离线评测
   会平白多背一堆展示逻辑；
2. **注入点已经现成**：`answer_review(plan_call=..., answer_call=...)` 允许替换模型调用，
   于是"计时"可以写成两个包装函数（见 `_wrap_call`），**一行都不用改 review_qa**——
   被包装的 usage 字典多一个 `elapsed_s` 键，`ReviewResult.calls` 里那句
   `dict(usage, stage=...)` 会原样带过去（这是打开源码确认过的，不是猜的）；
3. 检索侧同理：`TimingRetriever` 是纯代理，只补"第几轮、命中几条、花了多久"。
   顺带解决一个显示问题：**初次检索与补检要分得开**，否则界面上先显示
   "命中 5 条"、一秒后候选变成 8 条，用户会以为哪里出错了。

关于"多轮问答"的诚实说明：`answer_review()` 是**单轮无状态**的（一次提问 = 一次独立检索）。
本层不做"查询改写"那种额外模型调用（那会让成本不设上限，与规划"防成本失控"冲突），
而是在判定为**追问**时把上一问拼进检索词——**零额外调用**，代价是检索词变长。
拼了没有、拼成什么，都会回传给界面显示（`retrieval_query` / `carried_from`），
所以这不是暗箱操作：用户看得见，也能一键关掉。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# `_canon` 是 review_qa 的私有函数，这里**故意直接复用**：
# "两条引用算不算同一条"的规则必须与 `validate_citations` 完全同源，
# 见 `_link_citations` 的说明。
from rag.review_qa import (
    REVIEW_SECTIONS,
    REVIEW_TOP_K,
    _canon,
    answer_review,
    append_log,
    build_log_record,
    citation_of,
    default_answer_call,
    default_plan_call,
)

from web.pricing import estimate_cost, is_peak

# 与 main.py 的 `MAX_QUESTION_LEN` 是同一条产品规则（"问题太长就别问了"），
# 但**两个入口各自声明**：那条属于"命令行会话怎么接输入"，这条属于"Web 服务怎么接输入"。
# 值刻意保持一致，并由 `tests/test_web.py` 钉住相等——抄成两份不可怕，可怕的是没人发现它们漂了。
MAX_QUESTION_CHARS = 500

# 追问判定：**指代词 / 追问标记词**，外加"短到不可能是独立问题"的极短句。
#
# 第一版是"长度 ≤ 12 或含指代词"，**第一次真跑就被误伤**（台账 T38）：
# 用户问完“AI 辅助工艺编制怎么做？”（正好 12 字），接着问“异星工厂是什么”，
# 后者被当成追问把上一问粘了过来——检索词被带偏，候选整批跑到数字孪生那边。
# 教训和第 31、36 周那两条是同一条：**判据的特征要挑"目标里几乎不会出现、噪声里才有的"**，
# 而"句子短"在追问里会出现、在独立小问题里也会出现，拿它当特征就是给自己造噪声。
#
# 现在的口径：含指代词 / 追问标记词，或者短到 6 字以内（"还有呢？""为什么？"）。
# 仍然会误伤（"那优化呢？"里的"那"没进表），但界面上**拼了什么、开关在哪都是明摆着的**，
# 用户看得见，点一下就能关——判错的成本被界面兜住了。
_ANAPHORA = (
    "它",
    "他们",
    "这个",
    "这些",
    "那个",
    "那些",
    "上述",
    "上面",
    "之前",
    "刚才",
    "前者",
    "后者",
)
_CONTINUATION = ("还有", "继续", "接着", "展开说", "详细说", "详细讲")
FOLLOW_UP_MAX_CHARS = 6

# 进度行的文案：**只写这一步在做什么，不带注解**。
# 早先写成"检索资料库（本地，不花钱）""组织四段式答案（最慢的一步）"，
# 那些话属于使用说明，塞进进度行只会把"走到哪了"这件事本身冲淡。
_STAGE_LABELS = {
    "retrieve": "检索资料库",
    "plan": "盘点信息缺口",
    "answer": "组织四段式答案",
}


# ---- 纯函数：多轮拼接 ----


def looks_like_follow_up(question: str) -> bool:
    """这句话离开上文还能不能检索？带指代词/追问标记词、或短到 6 字以内的判为"不能"。"""
    text = (question or "").strip()
    if len(text) <= FOLLOW_UP_MAX_CHARS:
        return True
    return any(hint in text for hint in _ANAPHORA + _CONTINUATION)


def compose_query(
    question: str, previous: str | None, *, carry: bool
) -> tuple[str, str | None]:
    """返回 `(真正用于检索与提问的文本, 被带上的上一问或 None)`。"""
    text = (question or "").strip()
    previous = (previous or "").strip()
    if not carry or not previous or not looks_like_follow_up(text):
        return text, None
    return f"{previous} {text}", previous


# ---- 计时与进度：三个包装，全部通过注入生效 ----


@dataclass
class RunTrace:
    """一次运行的进度与计时。`on_event` 由服务层注入（SSE 就靠它推事件）。"""

    on_event: Callable[[dict], None] = lambda _event: None
    retrieval_rounds: list[dict] = field(default_factory=list)

    def emit(self, event: dict) -> None:
        """**回调异常不许打断主流程**：进度条坏了不该让一次已经付过钱的回答丢掉。"""
        try:
            self.on_event(event)
        # 这里**故意**宽catch + 静默：进度是"观测"，不是"逻辑"。
        # 回调可能抛任何东西（网络断了、事件循环关了、前端跑了），
        # 而那都不该让一次已经付过钱的回答丢掉。
        except Exception:  # noqa: BLE001, S110
            pass


class TimingRetriever:
    """检索器的计时代理。只加观测，不改行为——所以除 `search` 外一律转发给被包对象。"""

    def __init__(self, inner, trace: RunTrace):
        self._inner = inner
        self._trace = trace
        self.rounds: list[dict] = trace.retrieval_rounds

    def search(self, query: str, top_k: int = 5, **kwargs) -> list:
        index = len(self.rounds) + 1
        label = _STAGE_LABELS["retrieve"]
        self._trace.emit(
            {
                "type": "stage",
                "stage": "retrieve",
                "status": "start",
                "label": label,
                "round": index,
                "query": query,
            }
        )
        started = time.perf_counter()
        hits = self._inner.search(query, top_k, **kwargs)
        elapsed = time.perf_counter() - started
        self.rounds.append(
            {
                "round": index,
                "query": query,
                "hits": len(hits),
                "elapsed_s": round(elapsed, 3),
            }
        )
        self._trace.emit(
            {
                "type": "stage",
                "stage": "retrieve",
                "status": "done",
                "label": label,
                "round": index,
                "query": query,
                "hits": len(hits),
                "elapsed_s": round(elapsed, 3),
            }
        )
        return hits

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _wrap_call(base: Callable, stage: str, trace: RunTrace) -> Callable:
    """给一次模型调用加计时与进度事件，并把 `elapsed_s` 塞进 usage。

    `elapsed_s` 能活着进入 `ReviewResult.calls`，靠的是 review_qa 里
    `result.calls.append(dict(usage, stage=stage))` 这句浅拷贝——多出来的键不会被丢掉。
    """
    label = _STAGE_LABELS[stage]

    def call(prompt: str, **kwargs):
        trace.emit({"type": "stage", "stage": stage, "status": "start", "label": label})
        started = time.perf_counter()
        try:
            text, usage = base(prompt, **kwargs)
        except Exception as exc:
            trace.emit(
                {
                    "type": "stage",
                    "stage": stage,
                    "status": "error",
                    "label": label,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        elapsed = time.perf_counter() - started
        trace.emit(
            {
                "type": "stage",
                "stage": stage,
                "status": "done",
                "label": label,
                "elapsed_s": round(elapsed, 3),
                "finish_reason": usage.get("finish_reason"),
            }
        )
        return text, dict(usage, elapsed_s=round(elapsed, 3))

    return call


# ---- 视图：把 ReviewResult 翻译成前端要的结构 ----


def _link_citations(sections: dict, candidates: list) -> dict:
    """把每条引用换成"能点开"的对象：指向候选下标，或标记为只能到文件 / 编造。

    归一化直接复用 `review_qa._canon`，**故意不只抄一份**："什么算同一条引用"
    必须与 `validate_citations` 完全一致。若两边规则漂移，就会出现
    "校验说这条没问题、界面却点不开"这种最难查的不一致（第 18 周判据不一致的教训）。
    """
    by_citation: dict[str, int] = {}
    by_file: dict[str, list[int]] = {}
    for index, hit in enumerate(candidates):
        by_citation.setdefault(_canon(citation_of(hit)), index)
        by_file.setdefault(_canon(hit.chunk.source_path), []).append(index)

    def resolve(raw: str) -> dict:
        norm = _canon(raw)
        if norm in by_citation:
            return {"text": raw, "kind": "exact", "hit_index": by_citation[norm]}
        if norm in by_file:
            return {
                "text": raw,
                "kind": "partial",
                "hit_index": None,
                "file_hits": by_file[norm],
            }
        # 走到这里说明校验那一步就该把它挑出来了。留着而不是丢掉：
        # 界面要能显示"这条引用回查不了"，这比静默消失诚实（T28 的教训）。
        return {"text": raw, "kind": "unresolved", "hit_index": None}

    linked: dict = {}
    for key, rows in (sections or {}).items():
        if not isinstance(rows, list):
            linked[key] = rows
            continue
        out = []
        for item in rows:
            if not isinstance(item, dict):
                out.append(item)
                continue
            cites = item.get("citations")
            if not isinstance(cites, list):
                out.append(item)
                continue
            out.append(
                {**item, "citations": [resolve(c) for c in cites if isinstance(c, str)]}
            )
        linked[key] = out
    return linked


def _call_view(call: dict, *, peak: bool) -> dict:
    cost = estimate_cost(
        call.get("prompt_tokens"),
        call.get("cached_tokens"),
        call.get("completion_tokens"),
        peak=peak,
    )
    return {
        "stage": call.get("stage"),
        "prompt_tokens": int(call.get("prompt_tokens") or 0),
        "cached_tokens": int(call.get("cached_tokens") or 0),
        "completion_tokens": int(call.get("completion_tokens") or 0),
        "finish_reason": call.get("finish_reason"),
        "elapsed_s": call.get("elapsed_s"),
        "cost": cost,
    }


def build_view(
    result,
    *,
    question: str,
    retrieval_query: str,
    carried_from: str | None,
    trace: RunTrace,
    total_s: float,
    peak: bool,
) -> dict:
    """纯函数：`ReviewResult` + 计时 → 前端可直接渲染的字典。"""
    candidates = [
        dict(hit.to_dict(), citation=citation_of(hit)) for hit in result.candidates
    ]
    calls = [_call_view(call, peak=peak) for call in result.calls]
    prompt_tokens = sum(c["prompt_tokens"] for c in calls)
    cached_tokens = sum(c["cached_tokens"] for c in calls)
    completion_tokens = sum(c["completion_tokens"] for c in calls)
    model_s = sum(c["elapsed_s"] or 0.0 for c in calls)
    retrieval_s = sum(r.get("elapsed_s") or 0.0 for r in trace.retrieval_rounds)
    payload = result.payload or {}

    return {
        "question": question.strip(),
        "retrieval_query": retrieval_query,
        "carried_from": carried_from,
        "sections": _link_citations(
            {key: payload.get(key, []) for key in REVIEW_SECTIONS}, result.candidates
        ),
        "insufficient": bool(payload.get("insufficient", False)),
        "note": payload.get("note", "") or "",
        "degraded": bool(result.degraded),
        "reason": result.reason,
        "plan_reason": result.plan_reason,
        "thinking_recommended": bool(result.thinking_recommended),
        "citations": {
            "exact": result.citation_exact,
            "partial": result.citation_partial,
            "invalid": list(result.citation_invalid),
            "uncited": result.uncited_points,
        },
        "candidates": candidates,
        "retrieval": list(trace.retrieval_rounds),
        "calls": calls,
        "tokens": {
            "prompt": prompt_tokens,
            "cached": cached_tokens,
            "completion": completion_tokens,
        },
        "cost": {
            "peak": peak,
            "usd": sum(c["cost"]["usd"] for c in calls),
            "cny": sum(c["cost"]["cny"] for c in calls),
        },
        "elapsed": {
            "total_s": round(total_s, 2),
            "retrieval_s": round(retrieval_s, 2),
            "model_s": round(model_s, 2),
        },
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


# ---- 主入口 ----


def ask(
    question: str,
    retriever,
    *,
    previous_question: str | None = None,
    carry_context: bool = True,
    multihop: bool = True,
    thinking_model: str | None = None,
    top_k: int = REVIEW_TOP_K,
    plan_call: Callable | None = None,
    answer_call: Callable | None = None,
    on_event: Callable[[dict], None] | None = None,
    log_path: str | Path | None = None,
    index_label: str = "",
) -> dict:
    """跑一次问答，返回给界面用的视图（**会花 token**；离线测试请注入假的 call）。"""
    trace = RunTrace(on_event=on_event or (lambda _event: None))
    query, carried_from = compose_query(
        question, previous_question, carry=carry_context
    )
    peak = is_peak()
    timed = TimingRetriever(retriever, trace)
    started = time.perf_counter()
    result = answer_review(
        query,
        timed,
        top_k=top_k,
        multihop=multihop,
        thinking_model=thinking_model,
        plan_call=_wrap_call(plan_call or default_plan_call, "plan", trace),
        answer_call=_wrap_call(answer_call or default_answer_call, "answer", trace),
    )
    total_s = time.perf_counter() - started
    view = build_view(
        result,
        question=question,
        retrieval_query=query,
        carried_from=carried_from,
        trace=trace,
        total_s=total_s,
        peak=peak,
    )
    view["top_k"] = top_k
    view["thinking_model"] = thinking_model
    if log_path:
        # 复用 CLI 那套记账口径（字段、留空规则都一致），只多一个来源标记。
        # adopted / note 两格照旧留空：那是**只有人知道**的答案。
        record = build_log_record(
            result, index=index_label, top_k=top_k, elapsed_s=total_s
        )
        record["source"] = "web"
        append_log(log_path, record)
        view["logged_to"] = str(log_path)
    return view


# ---- 会话（多轮的"记事本"）----


@dataclass
class SessionStore:
    """内存里的会话表：只记"每个会话最近问了什么"，用来给追问补上下文。

    **故意不持久化**：这是给"浏览器里连着问几轮"用的临时状态，
    重启服务就没了；落盘留给 `data/usage_log.jsonl`（那是给成本与验收看的）。

    加锁是因为 `ask()` 跑在 worker 线程里（见 server.py 的 SSE 实现），
    而请求可能并发——本项目只有单机单用户，但"以为不会并发"不值得赌。
    """

    max_turns: int = 50
    _turns: dict[str, list[dict]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def history(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._turns.get(session_id, []))

    def last_question(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        history = self.history(session_id)
        return history[-1]["question"] if history else None

    def add(
        self, session_id: str | None, *, question: str, retrieval_query: str
    ) -> int:
        """记一轮，返回轮次（1 起算）。没有 session_id 就不记（无状态单次提问）。"""
        if not session_id:
            return 0
        with self._lock:
            turns = self._turns.setdefault(session_id, [])
            turns.append(
                {
                    "question": question.strip(),
                    "retrieval_query": retrieval_query,
                    "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
            )
            del turns[: max(len(turns) - self.max_turns, 0)]
            return len(turns)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._turns.pop(session_id, None)


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _cite_markdown(citation: dict) -> str:
    """引用在报告里的写法：精确的只写 `[文件:页]`，其余必须把"回查不到"标出来。"""
    suffix = {
        "exact": "",
        "partial": "（缺定位，只能回到文件）",
        "unresolved": "（回查不到）",
    }
    return f"[{citation.get('text', '')}]{suffix.get(citation.get('kind'), '')}"


def render_view_markdown(view: dict) -> str:
    """把视图渲染成 Markdown（离线验证报告与人工核对用；界面走 HTML，不走这里）。

    与 `review_qa._render` 的分工：那份渲染的是"模型看到了什么"（评测视角），
    这份渲染的是"用户看到了什么"（界面视角）——耗时与成本只在这份里。
    """
    lines = [
        f"# 问答视图（{view.get('generated_at', '')}）",
        "",
        f"> 提问：{view.get('question', '')}",
        f"> 检索词：{view.get('retrieval_query', '')}"
        + (
            f"（带上上一问：{view.get('carried_from')}）"
            if view.get("carried_from")
            else ""
        ),
        (
            f"> 候选：{len(view.get('candidates', []))} 条"
            f"｜模型调用：{len(view.get('calls', []))} 次"
            f"｜耗时：合计 {view['elapsed']['total_s']}s"
            f"（检索 {view['elapsed']['retrieval_s']}s + 模型 {view['elapsed']['model_s']}s）"
        ),
        (
            f"> 成本：${view['cost']['usd']:.4f}（约 {view['cost']['cny']:.3f} 元，"
            f"{'高峰' if view['cost']['peak'] else '低谷'}价，估算）"
        ),
        f"> 降级：{view.get('degraded')}｜原因：{view.get('reason', '')}",
        "",
    ]
    for key in ("研究现状", "方法对比", "结论", "可复用点"):
        lines.append(f"## {key}")
        lines.append("")
        rows = view.get("sections", {}).get(key) or []
        if not rows:
            lines.append("- （候选不足以支撑这一段）")
        for item in rows:
            cites = " ".join(_cite_markdown(c) for c in item.get("citations", []))
            detail = ""
            if item.get("pros") or item.get("cons"):
                detail = (
                    f"（优点：{item.get('pros', '-')}；局限：{item.get('cons', '-')}）"
                )
            lines.append(f"- {item.get('point', '')}{detail} {cites}")
        lines.append("")
    if view.get("note") or view.get("insufficient"):
        lines.append(f"**资料缺口声明**：{view.get('note') or '（insufficient=true）'}")
        lines.append("")
    counts = view.get("citations", {})
    lines.extend(
        [
            "## 引用与候选的对账",
            "",
            (
                f"- 精确可回查：{counts.get('exact', 0)}｜只到文件：{counts.get('partial', 0)}"
                f"｜编造：{len(counts.get('invalid', []))}"
                f"｜无引用条目：{counts.get('uncited', 0)}"
            ),
            f"- 候选（可点开看原文的条目）：{len(view.get('candidates', []))} 条",
            "",
            "```json",
            json.dumps(view.get("tokens", {}), ensure_ascii=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
