"""LLM 重排：用 `deepseek-flash` 把初筛候选（默认 top20）压成 top5。

为什么需要它：RRF 只用"排名"这一个信号（谁在几路里排第几），而模型能读懂
"这个片段到底答没答这个问题"。代价是一次额外的 LLM 调用，所以本模块的重点
其实不是"怎么调模型"，而是**什么时候不调**、以及**调砸了怎么办**：

1. **触发条件落到参数上**：相邻候选的融合分差 < `RERANK_GAP` 才算"接近"。
   分得很开说明融合结果本身就有把握，没必要花钱再问一次模型。
   `RERANK_GAP = 0.01` 是**初值**，第 18 周实测的相邻分差分布见同周笔记，
   调参依据在那边，不在这里拍脑袋。
2. **失败一律降级为"不重排"**：调用异常、回复不是合法 JSON、编号越界——
   任何一种都退回 RRF 顺序，并把 `degraded=True` 与原因带回调用方。
   检索主链路不因为一次重排失败而失败（这是显式分支，不是 try/except 糊一层）。
3. **每次都要记账**：返回 usage（prompt / completion / 缓存命中）。
   第 19 周做"混合+重排"那一列的成本对比时直接用这个数，不另算。

依赖链：本模块自己构造 `openai` 客户端，**不 import `config.py`**
（`config.py` 依赖 `openai-agents`，那会把 T18 的依赖链重新拉回 .venv-rag）。
接入点从 `constants.py` 取。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .retriever import DEFAULT_TOP_K, SearchHit

# ---- 成本与触发参数 ----
RERANK_GAP = 0.01  # 相邻候选分差小于它才触发重排（初值，待实测校准）
RERANK_CANDIDATES = 20  # 送进模型的候选上限
RERANK_TOP_N = 5  # 重排后保留条数
RERANK_SNIPPET_CHARS = 600  # 单条候选截断长度，用来兜住长片段
RERANK_TIMEOUT = 60.0
RERANK_MAX_RETRIES = 1

# 单次重排的输出上限。**这个值被真实数据纠正过两次**：
#   256  → 首跑即被截断（completion_tokens 正好 256，JSON 中途断掉）；
#   1024 → 20 题里仍有 5 题被截断，而且**回复是空的**（`content=''`、`finish_reason=length`）。
# 空的回复说明一件事：**这个模型会先在内部推理再输出**，那部分思考同样计入
# `completion_tokens`。所以预算不能按"我要的 JSON 有多长"（约 30 token）来估，
# 要按"思考 + 答案"来估——实测成功那 15 次的平均 completion 是 ~490 token，
# 截断那 5 次直接顶到 1024。4096 是给足余量：**被截断 = 这次调用 100% 白花钱**，
# 而上限只在真的生成时才付费。
RERANK_MAX_TOKENS = 4096
RERANK_REASON_CHARS = 120  # 降级原因里回显的回复长度（够看出"被截断"还是"答非所问"）

# 指令放最前面且保持稳定：DeepSeek 的 Context Caching 按前缀命中，
# 变化的部分（问题与候选）一律放到后面，才能让这部分指令"只付一次钱"。
PROMPT_TEMPLATE = """你是检索结果重排器。任务：按"与用户问题的相关程度"给候选片段排序。

规则：
1. 只依据片段内容判断，不要回答问题、不要补充片段以外的事实；
2. 与问题无关的片段不要放进结果；
3. 最多保留 {top_n} 条，最相关的在前；
4. 只输出一个 json 对象，不要任何解释、不要输出思考过程，也不要列全部候选。
   格式：{{"ranking": [3, 1, 8]}}，数字是片段编号，最多列 {top_n} 个；
5. 若没有任何片段与问题相关，输出 {{"ranking": []}}。

用户问题：
{query}

候选片段（共 {n} 条）：
{candidates}
"""


@dataclass(frozen=True)
class RerankResult:
    """重排结果。`hits` 永远是"可以直接用"的列表，降级时就是 RRF 顺序的前 top_n。"""

    hits: list[SearchHit]
    triggered: bool  # 是否真的调了模型
    degraded: bool  # 调了但失败，已退回不重排
    reason: str = ""
    usage: dict | None = None


def adjacent_gaps(hits: list[SearchHit]) -> list[float]:
    """相邻候选的分数差（已假设按分数降序）。"""
    return [hits[i].score - hits[i + 1].score for i in range(len(hits) - 1)]


def should_rerank(
    hits: list[SearchHit],
    gap: float = RERANK_GAP,
    *,
    candidates: int = RERANK_CANDIDATES,
) -> tuple[bool, float | None]:
    """是否需要重排。返回 `(是否触发, 窗口内最小的相邻分差)`。

    只看前 `candidates` 条：后面那些本来也进不了最终 top_n，
    为它们花钱重排没有意义。
    """
    window = list(hits)[:candidates]
    if len(window) < 2:
        return False, None
    smallest = min(adjacent_gaps(window))
    return smallest < gap, smallest


def build_prompt(
    query: str,
    hits: list[SearchHit],
    top_n: int = RERANK_TOP_N,
    *,
    snippet_chars: int = RERANK_SNIPPET_CHARS,
) -> str:
    """把问题与候选片段拼成重排提示词（编号从 1 开始，与让模型输出的编号一致）。"""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        text = hit.chunk.text.strip()
        if len(text) > snippet_chars:
            text = text[:snippet_chars] + "…"
        blocks.append(
            f"[{i}] 来源：{hit.chunk.source_path}｜位置：{hit.chunk.locator}\n{text}"
        )
    return PROMPT_TEMPLATE.format(
        top_n=top_n,
        query=query.strip(),
        n=len(hits),
        candidates="\n\n".join(blocks),
    )


def parse_ranking(text: str, n_candidates: int) -> list[int] | None:
    """把模型回复解析成 **0 起**的候选下标；任何不合规都返回 None（交给降级）。

    严格到什么程度：只认 `{"ranking": [...]}`，且每个元素都必须是范围内的整数
    （`bool` 不算整数）。重复编号按"首次出现"去重；出现越界或非整数就整条判失败——
    **不做"猜 JSON"式的修补**，因为那种修补会把"模型没听懂"伪装成"重排成功"。
    唯一容忍的包装是 ``` 代码围栏，它是模型最常见的输出习惯。
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
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    ranking = data.get("ranking")
    if not isinstance(ranking, list):
        return None

    order: list[int] = []
    for item in ranking:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        idx = item - 1
        if idx < 0 or idx >= n_candidates:
            return None
        if idx not in order:
            order.append(idx)
    return order


def _usage_info(usage, finish_reason: str | None = None) -> dict:
    """把 SDK 的 usage 摊平成普通 dict（字段用 getattr 取，避免猜结构）。

    `finish_reason` 一并带回来：`"length"` 表示被 `max_tokens` 截断，
    这是"回复不完整"与"模型乱答"之间唯一的分辨依据。
    """
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    if cached is None:
        # DeepSeek 自己的兼容字段，部分 SDK 版本不会落进 prompt_tokens_details
        cached = getattr(usage, "prompt_cache_hit_tokens", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "cached_tokens": cached,
        "finish_reason": finish_reason,
    }


_client = None


def _get_client():
    """延迟构造 openai 客户端（首次真实调用时才需要 openai / python-dotenv）。"""
    global _client
    if _client is None:
        from constants import DEEPSEEK_BASE_URL
        from dotenv import load_dotenv
        from openai import OpenAI

        # rag/rerank.py -> parents[0]=rag, [1]=notes-qa, [2]=agent-learning
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "没有找到 DEEPSEEK_API_KEY（应在 agent-learning/.env 里）"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=RERANK_TIMEOUT,
            max_retries=RERANK_MAX_RETRIES,
        )
    return _client


def chat(
    prompt: str,
    *,
    max_tokens: int = RERANK_MAX_TOKENS,
    json_mode: bool = False,
) -> tuple[str, dict]:
    """一次最小化调用：返回 `(回复文本, usage)`。

    重排（本模块）与回答级评测（`eval/rag_answer_eval.py`）都走这里——
    客户端构造与 usage 摊平只写一份，`finish_reason` 也一定带回来
    （它是区分"被 max_tokens 截断"与"模型没按格式回"的唯一证据，见台账 T26）。
    """
    from constants import DEEPSEEK_MODEL

    extra: dict = {}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    response = _get_client().chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        **extra,
    )
    choice = response.choices[0]
    return choice.message.content or "", _usage_info(
        response.usage, choice.finish_reason
    )


def default_llm_call(prompt: str) -> tuple[str, dict]:
    """重排用的调用：要求严格 JSON。"""
    return chat(prompt, json_mode=True)


def rerank(
    query: str,
    hits: list[SearchHit],
    *,
    top_n: int = RERANK_TOP_N,
    gap: float = RERANK_GAP,
    candidates: int = RERANK_CANDIDATES,
    call=None,
) -> RerankResult:
    """对初筛结果做一次重排。`call` 可注入（测试用假函数，生产用 default_llm_call）。"""
    hits = list(hits)
    if not hits:
        return RerankResult(hits=[], triggered=False, degraded=False, reason="无候选")

    need, smallest = should_rerank(hits, gap, candidates=candidates)
    if not need:
        why = (
            "候选少于 2 条"
            if smallest is None
            else f"相邻最小分差 {smallest:.5f} ≥ gap {gap}"
        )
        return RerankResult(
            hits=hits[:top_n], triggered=False, degraded=False, reason=why
        )

    pool = hits[:candidates]
    prompt = build_prompt(query, pool, top_n)
    try:
        text, usage = (default_llm_call if call is None else call)(prompt)
    except Exception as exc:  # noqa: BLE001 —— 重排失败绝不能拖垮检索主链路
        return RerankResult(
            hits=pool[:top_n],
            triggered=True,
            degraded=True,
            reason=f"LLM 调用失败：{type(exc).__name__}: {exc}",
        )

    order = parse_ranking(text, len(pool))
    # `None` = 解析失败；`[]` = **合法的**"都不相关"（提示词规则 5 就是这么要求的）。
    # 第一版写成 `if not order`，把后者也当成解析失败——无答案题上拿不到"都不相关"这个信号。
    if order is None:
        # 降级原因必须能区分"被截断"和"模型没按格式回"，并且带上回复开头——
        # 否则 FAIL 之后只能靠重跑一次才知道发生了什么（第 14 周 T10 的教训，
        # 第 18 周真跑第一遍就撞上：completion_tokens 正好等于上限 256）。
        truncated = usage.get("finish_reason") == "length"
        if truncated and not text.strip():
            # 空回复 + 撞上限：几乎所有 token 都花在内部推理上了（真实跑 20 题里命中 5 次）
            what = (
                f"回复为空且被 max_tokens 截断（completion_tokens="
                f"{usage.get('completion_tokens')}，思考占满了预算）"
            )
        elif truncated:
            what = f"回复被 max_tokens 截断（completion_tokens={usage.get('completion_tokens')}）"
        else:
            what = "回复不是合法 json 或编号非法"
        head = text.strip().replace("\n", " ")[:RERANK_REASON_CHARS]
        return RerankResult(
            hits=pool[:top_n],
            triggered=True,
            degraded=True,
            reason=f"{what}，已退回 RRF 顺序；回复开头：{head!r}",
            usage=usage,
        )

    if not order:
        # 模型说"没有一条相关"。这不是失败，是**结论**：提示词规则 2/5 明确要求
        # "无关的不要放进结果""都没有就输出 []"。返回空列表，把判断权交回调用方。
        return RerankResult(
            hits=[],
            triggered=True,
            degraded=False,
            reason="模型判定所有候选都不相关（返回 0 条）",
            usage=usage,
        )

    # 只返回**模型点过名的**候选。
    # 第一版把"未点名"的按原顺序补在后面凑满 top_n，理由是怕模型少答导致召回变少；
    # 真实跑了一次才发现这跟提示词自相矛盾：规则 2 说了"无关的不要放进来"，
    # 代码又把模型明确排除掉的片段塞回最终结果（那次它把"留给你填"的空白小节
    # 排在最后又送了回来）。调用方本来就拿得到完整候选池，不需要这里替它兜底。
    selected = [pool[i] for i in order][:top_n]
    dropped = len(pool) - len(order)
    return RerankResult(
        hits=selected,
        triggered=True,
        degraded=False,
        reason=f"模型点中 {len(order)} 条（未点名的 {dropped} 条按不相关丢弃）",
        usage=usage,
    )


# ---------------------------------------------------------------------------
# CLI（会花 token，由使用者在自己时段跑）：
#   & "..\..\.venv-rag\Scripts\python.exe" -m rag.rerank --query "..." --backend fastembed
# ---------------------------------------------------------------------------


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="混合检索 + LLM 重排")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidates", type=int, default=RERANK_CANDIDATES)
    parser.add_argument("--gap", type=float, default=RERANK_GAP)
    parser.add_argument("--backend", default="fastembed", choices=["bge", "fastembed"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-rerank", action="store_true", help="只跑混合检索，不调模型"
    )
    args = parser.parse_args()

    # 延迟导入：只有真跑稠密检索才需要 torch / onnxruntime
    from .embedder import Embedder
    from .retriever import Retriever, format_hits, make_stdout_forgiving

    make_stdout_forgiving()
    embedder = Embedder(backend=args.backend, device=args.device)

    retriever = Retriever.from_index(args.index, embedder)
    pool = retriever.search(args.query, args.candidates, candidates=args.candidates)
    print(f"[混合检索候选 {len(pool)} 条] {args.query}")
    print(format_hits(pool))

    if args.no_rerank:
        return 0

    result = rerank(
        args.query, pool, top_n=args.top_k, gap=args.gap, candidates=args.candidates
    )
    print(
        f"\n[重排] triggered={result.triggered} degraded={result.degraded} "
        f"原因={result.reason}"
    )
    if result.usage:
        print(f"[重排用量] {result.usage}")
    print(format_hits(result.hits))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
