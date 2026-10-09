"""综述型问答（第 21 周，G3 的核心）：把检索到的片段组织成**四段式、可回查**的答案。

与第一阶段 `notes-qa` 的区别：那边回答"某个事实是什么"（答案落在某一段里），
这边回答"这个方向有哪些方法、各自优缺点"（答案要**跨多篇**组织）。所以判据也换了：
不是"有没有找到那一段"，而是"**每条结论挂的引用，能不能回到原文**"。

一次问答最多 **2 次模型调用**（多跳上限，防成本失控）：

```text
① 混合检索 1        ── 离线，不花钱（top_k，每篇上限、排除不可引用源，见 retriever）
② 调用 1「盘点」    ── 给模型候选，问"够不够、缺什么、下一轮检索什么"
      ↑ 失败/解析不了 → 跳过 ③，直接进 ④（降级为单轮，不因为盘点失败丢掉答案）
③ 混合检索 2        ── 用模型给的 follow_up_queries 再检一轮，与第一批合并去重
④ 调用 2「组织」    ── 要它回四段式 JSON，每条结论挂 [文件:位置] 引用
⑤ 引用校验          ── **引用必须来自给过它的候选**：编造的挑出来单独记账
```

三个刻意的设计：

1. **引用校验放在代码里，不交给提示词**。"要挂引用"写进提示词只是请求，
   模型完全可以编一个看起来很像的页码；`validate_citations()` 拿候选集合去对，
   编造/缺定位分别计数。没有这一步，"引用可回查"就只是自评。
2. **候选为空时一次模型调用都不发**。资料里没有就是没有，没必要花钱让模型说"我不知道"
   （第 19 周的无答案题也是同一个道理：拒答的价值在于**代价低且诚实**）。
3. **降级是显式分支**：`payload=None` + `degraded=True` + 原因（含 `finish_reason` 与回复开头），
   与 `rerank.py` 同一套规矩——被 `max_tokens` 截断和"模型乱答"必须能分开（台账 T26）。

成本口径：两次调用的 usage 都记在 `ReviewResult.calls` 里，N 题一跑就能算出每题均价。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .retriever import DEFAULT_TOP_K, Retriever, SearchHit

# ---- 参数 ----
REVIEW_SECTIONS = ("研究现状", "方法对比", "结论", "可复用点")
REVIEW_TOP_K = DEFAULT_TOP_K  # 每轮检索条数
REVIEW_MAX_PER_SOURCE = 2  # 同一篇最多几条（治"一篇综述霸榜"，第 20 周实测）
REVIEW_SNIPPET_CHARS = 600  # 单条候选截断长度
REVIEW_MAX_CANDIDATES = 10  # 两轮合并后的候选上限（决定 prompt 大小，也就决定成本）
# 输出预算：**要按"思考 + 答案"估，而答案长度由我们要求的结构决定**。
#   4096 → 首次真跑 3/3 全部被截断（2026-10-09）：四段式 JSON 本身约 1.2k token，
#          而本项目的模型会先内部推理，思考动辄 2.5k~4k token——其中 1 题回复为空，
#          即思考把 4096 全吃光了。这是 T30 那条教训在"更长答案"上的第二次现形。
#   8192 → 给思考与答案都留出余量。仍被截断时：**先让每段写短，再考虑继续加大**
#          （写不完 = 这次调用 100% 白花钱，比写得少贵得多）。
# 盘点只回一个小 JSON，但它会顺带把"缺什么"写得很细：实测 522~2043 token，
# 其中一题的 2043 曾经贴着我设的 2048 上限（差 5 个 token 就白跑一次），
# 所以留到 4096 —— 上限只在真生成时付费，**贴脸才是风险**。
REVIEW_PLAN_MAX_TOKENS = 4096
REVIEW_ANSWER_MAX_TOKENS = 8192
REVIEW_REASON_CHARS = 120
REVIEW_PLAN_MAX_FOLLOW_UPS = 3  # 一轮补检最多用几条查询，防止检索轮次失控

# 思考模式判据（规划 2.4 要求"判据可执行化"）：问题里出现这些词，或显式 `/deep` 前缀。
# 关键词故意取"半步"而不是完整短语：第一版写的是 `"有哪些方法"`，而真实提问是
# **"有哪些主流方法"**——差一个"主流"就命不中（第 21 周写测试时才发现）。
# 判据宁可宽一点：多开一次思考的代价是钱，漏判的代价是答案质量。
_THINKING_HINTS = (
    "有哪些",
    "哪些方法",
    "对比",
    "现状",
    "趋势",
    "优缺点",
    "综述",
    "进展",
)
DEEP_PREFIX = "/deep"

# 指令放最前面且保持稳定：DeepSeek 的 Context Caching 按前缀命中，
# 变化的部分（问题与候选）一律放最后，让这部分指令只付一次钱（第 7 周结论）。
PLAN_PROMPT = """你是资料检索的"缺口盘点员"。只做一件事：判断下面这些候选片段够不够回答用户问题；
不够就指出缺什么，并提出下一轮检索用的查询词。

规则：
1. 只依据候选片段判断，不要回答问题、不要补充片段以外的事实；
2. `enough` 为 true 表示"这些片段已足以覆盖问题的各个子方面"；
3. 需要补检时 `follow_up_queries` 最多 {max_follow_ups} 条，每条是一个**可直接拿去检索的短查询**
   （用资料里的术语，不要写成疑问句）；
4. 只输出一个 json 对象，不要解释、不要输出思考过程：
   {{"enough": true, "gaps": ["..."], "follow_up_queries": ["..."]}}
5. 候选与问题完全无关时：enough=false，gaps 写明"候选与问题无关"，follow_up_queries 给 1 条。

用户问题：
{query}

候选片段（共 {n} 条）：
{candidates}
"""

ANSWER_PROMPT = """你是科研资料综述助手。依据给定的候选片段，把用户问题的答案组织成四段。

规则：
1. **只使用候选片段里的内容**，不许引入片段以外的知识；片段之间结论冲突时如实并列，不要替它们调和；
2. 四段的键名固定：`研究现状`、`方法对比`、`结论`、`可复用点`，每段是一个数组：
   - `研究现状` / `结论` / `可复用点`：元素为 {{"point": "一句话结论", "citations": ["文件:位置", ...]}}；
   - `方法对比`：元素为 {{"method": "方法名", "pros": "优点", "cons": "局限", "citations": [...]}}；
3. **控制篇幅，保证 json 能写完**：每段最多 4 条；`point` 不超过 60 字；`pros` / `cons` 各不超过 30 字；
   每条结论最多挂 2 个引用。**宁可少写几条，也不要写到一半断掉**——写不完等于这次调用全部白花钱；
4. **每条结论至少挂一个引用**，引用的"文件:位置"必须**原样取自候选里给出的**，
   例如 {{"citations": ["刀具磨损监测/某篇.pdf:第 4 页"]}}——不许自己编页码，也不许引用候选之外的文献；
5. 候选不足以支撑某一段时，那一段给空数组，把 `insufficient` 置为 true，
   并在 `note` 里写清"资料中没有……"；**禁止跨主题拼凑**；
6. 只输出一个 json 对象，不要解释、不要输出思考过程。

输出格式：
{{"研究现状": [], "方法对比": [], "结论": [], "可复用点": [], "insufficient": false, "note": ""}}

用户问题：
{query}

候选片段（共 {n} 条）：
{candidates}
"""


@dataclass(frozen=True)
class Plan:
    """调用 1 的结果：候选够不够、缺什么、下一轮查什么。"""

    enough: bool
    gaps: tuple[str, ...] = ()
    follow_up_queries: tuple[str, ...] = ()


@dataclass
class ReviewResult:
    """一次综述问答的完整结果。`payload` 为 None 表示最终回答失败（已降级）。"""

    question: str
    payload: dict | None
    candidates: list[SearchHit] = field(default_factory=list)
    plan: Plan | None = None
    calls: list[dict] = field(default_factory=list)  # 每次调用的 usage
    degraded: bool = False
    reason: str = ""
    plan_reason: str = ""  # 盘点环节的情况（失败不拖垮回答，所以与 reason 分开）
    citation_exact: int = 0
    citation_partial: int = 0
    citation_invalid: list[str] = field(default_factory=list)
    uncited_points: int = 0
    thinking_recommended: bool = False


# ---- 提示词 ----


def citation_of(hit: SearchHit) -> str:
    """候选项的标准引用写法：`文件:位置`（PDF 是页码，笔记是行号区间）。"""
    return f"{hit.chunk.source_path}:{hit.chunk.locator}"


def build_candidates_block(
    hits: list[SearchHit], *, snippet_chars: int = REVIEW_SNIPPET_CHARS
) -> str:
    """把候选拼成提示词里的片段块。编号从 1 起算，`文件`/`位置` 与原引用格式一致。"""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        text = hit.chunk.text.strip()
        if len(text) > snippet_chars:
            text = text[:snippet_chars] + "…"
        blocks.append(
            f"[{i}] 文件：{hit.chunk.source_path}｜位置：{hit.chunk.locator}\n{text}"
        )
    return "\n\n".join(blocks)


def should_use_thinking(question: str) -> bool:
    """综述型问题才值得开思考模式（规划 2.4 的判据）。纯函数，可离线测。"""
    return any(
        hint in question for hint in _THINKING_HINTS
    ) or question.strip().startswith(DEEP_PREFIX)


def strip_deep_prefix(question: str) -> str:
    """去掉 `/deep` 前缀（它是给我们的开关，不该进检索词）。"""
    text = question.strip()
    if text.startswith(DEEP_PREFIX):
        return text[len(DEEP_PREFIX) :].strip()
    return text


# ---- 解析与校验 ----


def _loads_lenient(text: str):
    """去掉 ``` 围栏后解析 JSON；失败返回 None（不做"猜 JSON"式修补）。

    容忍代码围栏是因为它是模型最常见的输出习惯；除此之外一律判失败——
    修补会把"模型没听懂"伪装成"结构合规"（`rerank.parse_ranking` 的同一条规矩）。
    """
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def parse_plan(text: str) -> Plan | None:
    """解析"盘点"回复；结构不合规返回 None（调用方按"盘点失败"处理）。"""
    data = _loads_lenient(text)
    if not isinstance(data, dict) or not isinstance(data.get("enough"), bool):
        return None
    gaps = data.get("gaps")
    follows = data.get("follow_up_queries")
    if not isinstance(gaps, list) or not isinstance(follows, list):
        return None
    if not all(isinstance(g, str) for g in gaps):
        return None
    if not all(isinstance(q, str) for q in follows):
        return None
    return Plan(
        enough=data["enough"],
        gaps=tuple(g.strip() for g in gaps if g.strip()),
        follow_up_queries=tuple(q.strip() for q in follows if q.strip())[
            :REVIEW_PLAN_MAX_FOLLOW_UPS
        ],
    )


def parse_answer(text: str) -> dict | None:
    """解析四段式回复；任何一段结构不合规就整体判失败（宁缺毋滥，交给降级）。"""
    data = _loads_lenient(text)
    if not isinstance(data, dict):
        return None
    out: dict = {}
    for key in REVIEW_SECTIONS:
        items = data.get(key)
        if not isinstance(items, list):
            return None
        cleaned = []
        for item in items:
            if not isinstance(item, dict):
                return None
            point = item.get("point") or item.get("method")
            if not isinstance(point, str) or not point.strip():
                return None
            citations = item.get("citations")
            if not isinstance(citations, list) or not all(
                isinstance(c, str) for c in citations
            ):
                return None
            entry: dict = {"point": point.strip()}
            for extra_key in ("method", "pros", "cons"):
                value = item.get(extra_key)
                if isinstance(value, str) and value.strip():
                    entry[extra_key] = value.strip()
            entry["citations"] = [c.strip() for c in citations if c.strip()]
            cleaned.append(entry)
        out[key] = cleaned
    out["insufficient"] = bool(data.get("insufficient", False))
    note = data.get("note")
    out["note"] = note.strip() if isinstance(note, str) else ""
    return out


def _canon(citation: str) -> str:
    """归一化引用：去方括号、去空白。让"多一个空格"不算编造。"""
    return citation.strip().strip("[]").replace(" ", "")


def validate_citations(
    payload: dict, candidates: list[SearchHit]
) -> tuple[dict, int, int, list[str], int]:
    """把答案里的引用与候选集合对齐。返回 `(清洗后的 payload, 精确数, 缺定位数, 编造列表, 无引用条目数)`。

    三档：

    - **精确**：`文件:位置` 与某个候选完全一致 → 可回查到具体页/段；
    - **缺定位**：只写了文件名（或位置对不上）→ 能回查到文件，但还没到页；
    - **编造**：文件根本不在候选里 → 单独记账，**不静默删掉**（删掉就看不见问题了）。

    条目保留原样（只把编造引用摘出来），因为"模型没给引用"是**要报告的事实**，
    不是我们要替它抹平的瑕疵（T28 的教训：别把模型明确排除的东西又悄悄塞回去）。
    """
    exact_strings = {_canon(citation_of(hit)) for hit in candidates}
    file_strings = {_canon(hit.chunk.source_path) for hit in candidates}
    cleaned = dict(payload)
    exact = partial = 0
    invalid: list[str] = []
    uncited = 0
    for key in REVIEW_SECTIONS:
        rows = []
        for item in payload.get(key, []):
            kept: list[str] = []
            seen: set[str] = set()
            for raw in item.get("citations", []):
                norm = _canon(raw)
                if norm in seen:
                    # 同一条引用写两遍：去重。真跑 R21-1 的"电流传感器"那条就是
                    # 同一个 `文件:页` 连写了两次（提示词说"最多 2 个引用"，
                    # 它当成"可以写两个"了）。同一出处算一次。
                    continue
                if norm in exact_strings:
                    seen.add(norm)
                    kept.append(raw)
                    exact += 1
                elif norm in file_strings:
                    seen.add(norm)
                    kept.append(raw)
                    partial += 1
                else:
                    invalid.append(raw)
            if not kept:
                uncited += 1
            rows.append(dict(item, citations=kept))
        cleaned[key] = rows
    return cleaned, exact, partial, list(dict.fromkeys(invalid)), uncited


# ---- 主流程 ----


def default_plan_call(
    prompt: str, *, model: str | None = None, max_tokens: int = REVIEW_PLAN_MAX_TOKENS
) -> tuple[str, dict]:
    from .rerank import chat

    return chat(prompt, max_tokens=max_tokens, json_mode=True, model=model)


def default_answer_call(
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = REVIEW_ANSWER_MAX_TOKENS,
) -> tuple[str, dict]:
    from .rerank import chat

    return chat(prompt, max_tokens=max_tokens, json_mode=True, model=model)


def _failure_reason(text: str, usage: dict) -> str:
    """降级原因：必须能区分"被截断"与"模型乱答"，并带上回复开头（T26）。

    截断时额外给一句**可执行的处置**：报告是给使用者看的，
    "被截断"这三个字本身不告诉他下一步该动哪个旋钮。
    """
    truncated = usage.get("finish_reason") == "length"
    if truncated and not text.strip():
        what = (
            "回复为空且被 max_tokens 截断（completion_tokens="
            f"{usage.get('completion_tokens')}，思考占满了预算）"
        )
    elif truncated:
        what = f"回复被 max_tokens 截断（completion_tokens={usage.get('completion_tokens')}）"
    else:
        what = "回复不是合法 json 或字段不合规"
    hint = (
        "；处置：先让每段写短（见提示词里的篇幅上限），再调大 --max-tokens"
        if truncated
        else ""
    )
    head = text.strip().replace("\n", " ")[:REVIEW_REASON_CHARS]
    return f"{what}{hint}；回复开头：{head!r}"


def _merge_hits(primary: list[SearchHit], extra: list[SearchHit]) -> list[SearchHit]:
    """按 chunk.id 合并去重，先来先得（顺序即优先级），总条数受 `REVIEW_MAX_CANDIDATES` 限制。"""
    merged = list(primary)
    seen = {hit.chunk.id for hit in merged}
    for hit in extra:
        if hit.chunk.id in seen:
            continue
        merged.append(hit)
        seen.add(hit.chunk.id)
        if len(merged) >= REVIEW_MAX_CANDIDATES:
            break
    return merged[:REVIEW_MAX_CANDIDATES]


def answer_review(
    question: str,
    retriever: Retriever,
    *,
    top_k: int = REVIEW_TOP_K,
    max_per_source: int = REVIEW_MAX_PER_SOURCE,
    multihop: bool = True,
    thinking_model: str | None = None,
    plan_call=default_plan_call,
    answer_call=default_answer_call,
) -> ReviewResult:
    """跑一次综述问答（**会花 token**；离线测试请注入假的 `plan_call` / `answer_call`）。

    `thinking_model` 只在判据命中时使用，且必须由调用方给出确切模型名
    （**不猜模型名**，见 `rerank.chat` 的说明）。
    """
    question = strip_deep_prefix(question)
    thinking = should_use_thinking(question)
    model = thinking_model if (thinking and thinking_model) else None
    result = ReviewResult(
        question=question, payload=None, thinking_recommended=thinking
    )

    hits = retriever.search(question, top_k, max_per_source=max_per_source)
    result.candidates = hits
    if not hits:
        # 没有候选就别花钱问模型"你知道吗"——"资料中没有"是事实，不是要推理的结论。
        result.payload = {
            **{key: [] for key in REVIEW_SECTIONS},
            "insufficient": True,
            "note": "资料中没有找到与该问题相关的片段。",
        }
        result.reason = "检索无候选，未调用模型"
        return result

    # ② 盘点：够不够、缺什么（失败只影响是否补检，不影响最终回答）
    if multihop:
        plan_prompt = PLAN_PROMPT.format(
            max_follow_ups=REVIEW_PLAN_MAX_FOLLOW_UPS,
            query=question,
            n=len(hits),
            candidates=build_candidates_block(hits),
        )
        try:
            text, usage = plan_call(plan_prompt, model=model)
            result.calls.append(dict(usage, stage="plan"))
            plan = parse_plan(text)
            if plan is None:
                result.plan_reason = _failure_reason(text, usage)
            else:
                result.plan = plan
                result.plan_reason = (
                    "候选已足够"
                    if plan.enough
                    else f"认为有缺口：{'；'.join(plan.gaps)}"
                )
        except Exception as exc:  # noqa: BLE001 —— 盘点失败不该拖垮回答
            result.plan_reason = f"盘点调用失败：{type(exc).__name__}: {exc}"

        # ③ 补检：只在盘点明确说"不够"且给了查询词时才做（最多一轮）
        if result.plan is not None and not result.plan.enough:
            extra: list[SearchHit] = []
            for follow_up in result.plan.follow_up_queries:
                extra.extend(
                    retriever.search(follow_up, top_k, max_per_source=max_per_source)
                )
            if extra:
                result.candidates = _merge_hits(hits, extra)

    # ④ 组织：四段式 + 引用
    answer_prompt = ANSWER_PROMPT.format(
        query=question,
        n=len(result.candidates),
        candidates=build_candidates_block(result.candidates),
    )
    try:
        text, usage = answer_call(answer_prompt, model=model)
        result.calls.append(dict(usage, stage="answer"))
    except Exception as exc:  # noqa: BLE001
        result.degraded = True
        result.reason = f"回答调用失败：{type(exc).__name__}: {exc}"
        return result

    payload = parse_answer(text)
    if payload is None:
        result.degraded = True
        result.reason = _failure_reason(text, usage)
        return result

    payload, exact, partial, invalid, uncited = validate_citations(
        payload, result.candidates
    )
    result.payload = payload
    result.citation_exact = exact
    result.citation_partial = partial
    result.citation_invalid = invalid
    result.uncited_points = uncited
    result.reason = (
        f"四段式回答已产出；引用：精确 {exact}、缺定位 {partial}、"
        f"编造 {len(invalid)}；无引用条目 {uncited}"
    )
    return result


# ---------------------------------------------------------------------------
# CLI（**会花 token**，由使用者在低谷时段跑）：
#   uv run python -m rag.review_qa --query "刀具磨损监测有哪些主流方法？" --out eval\第21周综述问答.md
# ---------------------------------------------------------------------------


def _render(result: ReviewResult, index: str, top_k: int) -> str:
    calls = len(result.calls)
    prompt_tokens = sum(c.get("prompt_tokens") or 0 for c in result.calls)
    completion_tokens = sum(c.get("completion_tokens") or 0 for c in result.calls)
    cached = sum(c.get("cached_tokens") or 0 for c in result.calls)
    lines = [
        "# 第 21 周综述型问答",
        "",
        f"> 问题：{result.question}",
        (
            f"> 索引：`{index}`｜top_k={top_k}｜候选 {len(result.candidates)} 条"
            f"｜模型调用 {calls} 次｜prompt {prompt_tokens} / completion {completion_tokens}"
            f" / 缓存命中 {cached}"
        ),
        f"> 降级：{result.degraded}｜原因：{result.reason}",
        (
            f"> 盘点：{result.plan_reason or '（未启用多跳）'}"
            f"｜思考模式判据命中：{result.thinking_recommended}"
        ),
        "",
        "## 四段式答案",
        "",
    ]
    payload = result.payload or {}
    for key in REVIEW_SECTIONS:
        lines.append(f"### {key}")
        lines.append("")
        rows = payload.get(key) or []
        if not rows:
            lines.append("- （候选不足以支撑这一段）")
        for item in rows:
            cites = " ".join(f"[{c}]" for c in item.get("citations", []))
            detail = ""
            if item.get("pros") or item.get("cons"):
                detail = (
                    f"（优点：{item.get('pros', '-')}；局限：{item.get('cons', '-')}）"
                )
            lines.append(f"- {item.get('point', '')}{detail} {cites}")
        lines.append("")
    if payload.get("insufficient") or payload.get("note"):
        lines.append(
            f"**资料缺口声明**：{payload.get('note', '') or '（insufficient=true）'}"
        )
        lines.append("")
    lines.extend(
        [
            "## 引用校验（拿候选集合去对，不看模型自评）",
            "",
            f"- 精确可回查：{result.citation_exact}",
            f"- 只到文件、缺定位：{result.citation_partial}",
            f"- **编造（候选里没有）：{len(result.citation_invalid)}**"
            + (f" → {result.citation_invalid}" if result.citation_invalid else ""),
            f"- 一条引用都没挂的条目：{result.uncited_points}",
            "",
            "## 模型看到的候选（人工核对用）",
            "",
        ]
    )
    for i, hit in enumerate(result.candidates, start=1):
        text = hit.chunk.text.replace("\n", " ").strip()[:60]
        lines.append(
            f"{i}. `{hit.chunk.source_path}`（{hit.chunk.locator}）{hit.channels} | {text}"
        )
    lines.append("")
    if result.calls:
        lines.append("## 每次调用")
        lines.append("")
        for i, call in enumerate(result.calls, start=1):
            lines.append(f"- 第 {i} 次（{call.get('stage')}）：{call}")
        lines.append("")
    return "\n".join(lines)


def _cli() -> int:
    import argparse
    from functools import partial
    from pathlib import Path

    parser = argparse.ArgumentParser(description="综述型问答（四段式 + 可回查引用）")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--top-k", type=int, default=REVIEW_TOP_K)
    parser.add_argument("--max-per-source", type=int, default=REVIEW_MAX_PER_SOURCE)
    parser.add_argument(
        "--no-multihop", action="store_true", help="只跑一轮检索（省一次模型调用）"
    )
    parser.add_argument(
        "--thinking-model",
        default=None,
        help="命中思考判据时改用这个模型名（**以控制台为准**，不许猜）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=REVIEW_ANSWER_MAX_TOKENS,
        help="回答阶段的输出上限（**思考 + 答案一起算**；被截断 = 这次调用白花钱）",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--json", action="store_true", help="额外打印 JSON 结果")
    args = parser.parse_args()

    from .embedder import Embedder
    from .retriever import make_stdout_forgiving

    make_stdout_forgiving()
    retriever = Retriever.from_index(args.index, Embedder(backend="fastembed"))
    result = answer_review(
        args.query,
        retriever,
        top_k=args.top_k,
        max_per_source=args.max_per_source,
        multihop=not args.no_multihop,
        thinking_model=args.thinking_model,
        answer_call=partial(default_answer_call, max_tokens=args.max_tokens),
    )
    report = _render(result, args.index, args.top_k)
    print(report)
    if args.json:
        print(json.dumps(result.payload, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"[综述] 报告写入 {out}")
    return 0 if not result.degraded else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
