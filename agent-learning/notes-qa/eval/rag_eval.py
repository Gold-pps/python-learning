"""第 19 周检索评测：四组对比 + 分类别 Recall@5（G2 验收用）。

四组（前三组离线、不花钱）：
    关键词基线（第一阶段做法）/ BM25 / 纯向量 / 混合(RRF)　＋　[可选] 混合 + LLM 重排

指标口径（**必须先说清，否则数字会被误读**）：
- **Recall@5**：某个方法的 `top_k` 里，是否把该题的期望来源全部覆盖（严格：来源文件 + 片段含关键词；
  宽松：只要来源文件出现）。跨文档 / 多跳题要求**每个期望来源都被覆盖**，另报覆盖率。
- **重排组的返回条数可能少于 `top_k`**：模型只点中 4 条就返回 4 条（台账 T28）。
  Recall 按"该方法实际返回的列表"算，不拿候选池充数。
- **引用准确率 / 无答案拒答率不在这里**：那是 Agent 级指标（要模型答完才判），
  见 `rag/评测说明.md` 的"待补口径"与 `eval/rag_answer_eval.py`。

用法（在 `agent-learning/notes-qa` 下）：
    # 只跑三组离线对比（不花钱，主 .venv 也行：--no-dense）
    & "..\\..\\.venv-rag\\Scripts\\python.exe" -m eval.rag_eval --out eval/第19周对比表.md

    # 加上重排组（**会花 token**，25 题约 25 次调用，由使用者低谷时段跑）
    & "..\\..\\.venv-rag\\Scripts\\python.exe" -m eval.rag_eval --with-rerank --out eval/第19周对比表.md

    # 重排结果会存成 JSON；之后重算表格直接读缓存，不再调用模型
    & "..\\..\\.venv-rag\\Scripts\\python.exe" -m eval.rag_eval --rerank-cache eval/rag_eval_rerank.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

from rag.retriever import Retriever, SearchHit, keyword_baseline_search

from .retrieval_cases import CASES, CATEGORIES, CATEGORY_NO_ANSWER, Case

OFFLINE_METHODS = ("关键词基线", "BM25", "纯向量", "混合(RRF)")
RERANK_METHOD = "混合+重排"
RERANK_CANDIDATES = 20


def _covered(hits: Sequence[SearchHit], case: Case, *, strict: bool) -> set[str]:
    """返回被覆盖的"期望来源"集合。严格模式还要求片段正文里出现该来源的关键词。"""
    covered: set[str] = set()
    for exp in case.expects:
        for hit in hits:
            if hit.chunk.source_path != exp.file:
                continue
            if not strict or any(
                kw.casefold() in hit.chunk.text.casefold() for kw in exp.keywords
            ):
                covered.add(exp.file)
                break
    return covered


def _score(hits: Sequence[SearchHit], case: Case) -> tuple[bool, bool, float]:
    """返回 `(严格命中, 宽松命中, 覆盖率)`。"""
    need = len(case.expects)
    strict = _covered(hits, case, strict=True)
    loose = _covered(hits, case, strict=False)
    return len(strict) == need, bool(loose), len(loose) / need


def _describe_top(hits: Sequence[SearchHit], limit: int = 2) -> str:
    if not hits:
        return "（无命中）"
    parts = []
    for hit in list(hits)[:limit]:
        parts.append(f"{hit.chunk.source_path}（{hit.chunk.locator}）")
    return "、".join(parts)


def _ground_truth_report(retriever: Retriever) -> tuple[list[str], list[str]]:
    """自检：每个期望来源 + 关键词真的可达吗？无答案题的关键词真的 0 命中吗？

    这两条对应台账 T8 的两条纪律：**一道无解的题会被读成"能力差"**，
    而**把负样本关键词写进资料库等于把答案卡发下去**。
    """
    unreachable: list[str] = []
    polluted: list[str] = []
    for case in CASES:
        if case.category == CATEGORY_NO_ANSWER:
            for kw in case.negative_keywords:
                hit_files = sorted(
                    {
                        c.source_path
                        for c in retriever.chunks
                        if kw.casefold() in c.text.casefold()
                    }
                )
                if hit_files:
                    polluted.append(f"{case.id} 的负样本词 {kw!r} 出现在 {hit_files}")
            continue
        for exp in case.expects:
            texts = [c.text for c in retriever.chunks if c.source_path == exp.file]
            if not texts:
                unreachable.append(f"{case.id}: 语料里没有 {exp.file}")
            elif not any(
                kw.casefold() in t.casefold() for kw in exp.keywords for t in texts
            ):
                unreachable.append(
                    f"{case.id}: {exp.file} 里找不到 {list(exp.keywords)}"
                )
    return unreachable, polluted


def _run_offline(retriever: Retriever, queries: dict[str, str], top_k: int):
    """跑三/四组离线方法，返回 `{方法: {题号: hits}}`。"""
    results: dict[str, dict[str, list[SearchHit]]] = {m: {} for m in OFFLINE_METHODS}
    for case in CASES:
        if case.category == CATEGORY_NO_ANSWER:
            continue
        question = queries[case.id]
        results["关键词基线"][case.id] = keyword_baseline_search(
            question, retriever.chunks, top_k
        )
        results["BM25"][case.id] = retriever.search(question, top_k, use_dense=False)
        results["纯向量"][case.id] = retriever.search(question, top_k, use_sparse=False)
        results["混合(RRF)"][case.id] = retriever.search(question, top_k)
    return results


def _run_rerank(retriever: Retriever, queries: dict[str, str], top_k: int):
    """跑重排组（**会花 token**）。返回 `(结果, 用量合计, 失败题号)`。"""
    from rag.rerank import rerank

    results: dict[str, list[SearchHit]] = {}
    usages: list[dict] = []
    degraded: list[str] = []
    payload: dict[str, dict] = {}
    for case in CASES:
        if case.category == CATEGORY_NO_ANSWER:
            continue
        question = queries[case.id]
        pool = retriever.search(
            question, RERANK_CANDIDATES, candidates=RERANK_CANDIDATES
        )
        outcome = rerank(question, pool, top_n=top_k, candidates=RERANK_CANDIDATES)
        results[case.id] = outcome.hits
        if outcome.usage:
            usages.append(outcome.usage)
        if outcome.degraded:
            degraded.append(f"{case.id}（{outcome.reason[:60]}）")
        payload[case.id] = {
            "hits": [h.to_dict() for h in outcome.hits],
            "triggered": outcome.triggered,
            "degraded": outcome.degraded,
            "reason": outcome.reason,
            "usage": outcome.usage,
        }
        print(
            f"  {case.id}: triggered={outcome.triggered} degraded={outcome.degraded} "
            f"返回 {len(outcome.hits)} 条"
        )
    return results, usages, degraded, payload


def _load_rerank_cache(path: Path, top_k: int):
    """从缓存 JSON 读回重排结果（不再调模型）。"""
    from rag.chunk import Chunk

    data = json.loads(path.read_text(encoding="utf-8"))
    results: dict[str, list[SearchHit]] = {}
    usages: list[dict] = []
    degraded: list[str] = []
    for case_id, item in data.items():
        hits = []
        for raw in item["hits"]:
            chunk = Chunk(
                id=raw["id"],
                source_path=raw["source_path"],
                doc_type=raw.get("doc_type", "md"),
                locator=raw["locator"],
                start=raw["start"],
                end=raw["end"],
                text=raw["text"],
                hash=raw.get("hash", ""),
            )
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=raw["score"],
                    dense_rank=raw.get("dense_rank"),
                    dense_score=raw.get("dense_score"),
                    sparse_rank=raw.get("sparse_rank"),
                    sparse_score=raw.get("sparse_score"),
                )
            )
        results[case_id] = hits[:top_k]
        if item.get("usage"):
            usages.append(item["usage"])
        if item.get("degraded"):
            degraded.append(f"{case_id}（{item.get('reason', '')[:60]}）")
    return results, usages, degraded


def _usage_summary(usages: list[dict]) -> dict:
    if not usages:
        return {}
    return {
        "调用次数": len(usages),
        "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usages),
        "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usages),
        "cached_tokens": sum(u.get("cached_tokens") or 0 for u in usages),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="第 19 周 RAG 检索评测")
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", default="fastembed", choices=["bge", "fastembed"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-dense", action="store_true", help="只用关键词基线与 BM25")
    parser.add_argument(
        "--with-rerank", action="store_true", help="**会花 token**：跑 LLM 重排组"
    )
    parser.add_argument(
        "--rerank-cache", default="", help="读已有的重排结果 JSON，不再调模型"
    )
    parser.add_argument(
        "--save-rerank", default="eval/rag_eval_rerank.json", help="重排结果存这里"
    )
    parser.add_argument("--out", default="", help="报告写成 UTF-8 文件")
    parser.add_argument(
        "--show", default="", help="逗号分隔的题号：打印各方法的 top-k 明细"
    )
    args = parser.parse_args()
    show_ids = {s.strip() for s in args.show.split(",") if s.strip()}

    # 报告里只用 ASCII 标记，但片段/文件名是中文、随时可能夹带 emoji——
    # 统一走这个容错（第 18 周为了 ✔/✗ 和 ✅ 各崩过一次，T23）。
    from rag.retriever import make_stdout_forgiving

    make_stdout_forgiving()

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    embedder = None
    if not args.no_dense:
        from rag.embedder import Embedder

        embedder = Embedder(backend=args.backend, device=args.device)
    retriever = Retriever.from_index(args.index, embedder)

    queries = {c.id: c.question for c in CASES}
    scored = [c for c in CASES if c.category != CATEGORY_NO_ANSWER]
    categories = []
    for case in scored:
        if case.category not in categories:
            categories.append(case.category)
    english = [c.id for c in CASES if "英文" in c.tags]

    emit(f"索引：{args.index}｜片段 {len(retriever.chunks)}｜后端 {args.backend}")
    dist = " + ".join(
        f"{cat} {sum(1 for c in CASES if c.category == cat)}" for cat in CATEGORIES
    )
    emit(f"评测集：{len(CASES)} 条 = {dist}")
    emit(f"英文题：{len(english)} 条（{', '.join(english)}）")

    unreachable, polluted = _ground_truth_report(retriever)
    if unreachable:
        emit()
        emit("!! ground truth 不可达（这些题的分数无效）：")
        for line in unreachable:
            emit(f"   - {line}")
    if polluted:
        emit()
        emit("!! 无答案题的负样本词出现在语料里（T8：等于把答案卡发下去）：")
        for line in polluted:
            emit(f"   - {line}")
    if not unreachable and not polluted:
        emit("自检：ground truth 全部可达、负样本词全部 0 命中 [OK]")

    results = _run_offline(retriever, queries, args.top_k)
    usages: list[dict] = []
    degraded: list[str] = []
    if args.rerank_cache:
        rerank_hits, usages, degraded = _load_rerank_cache(
            Path(args.rerank_cache), args.top_k
        )
        results[RERANK_METHOD] = rerank_hits
        emit(f"\n重排组：读缓存 {args.rerank_cache}")
    elif args.with_rerank:
        emit(f"\n重排组：调用 deepseek-flash（{len(scored)} 题，会花 token）")
        rerank_hits, usages, degraded, payload = _run_rerank(
            retriever, queries, args.top_k
        )
        results[RERANK_METHOD] = rerank_hits
        Path(args.save_rerank).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        emit(f"重排结果已存：{args.save_rerank}")

    methods = [m for m in (*OFFLINE_METHODS, RERANK_METHOD) if m in results]

    # ---- 逐题表 ----
    emit()
    emit(f"逐题（Top-{args.top_k}；严格判据：来源文件 + 片段含关键词）")
    emit(f"{'id':<7}{'类别':<5}" + "".join(f"{m:<11}" for m in methods) + "覆盖")
    for case in scored:
        marks = []
        for method in methods:
            hits = results[method].get(case.id, [])
            strict, _, _ = _score(hits, case)
            marks.append("+" if strict else "-")
        _, _, cover = _score(results["混合(RRF)"].get(case.id, []), case)
        emit(
            f"{case.id:<7}{case.category:<5}"
            + "".join(f"{m:<11}" for m in marks)
            + f"{cover:.0%}"
        )
        if case.id in show_ids:
            for method in methods:
                emit(
                    f"    [{method}] {_describe_top(results[method].get(case.id, []))}"
                )

    # ---- 汇总 ----
    emit()
    emit("Recall@5（严格判据）——按类别：")
    header = f"{'类别':<7}{'题数':<5}" + "".join(f"{m:<11}" for m in methods)
    emit(header)
    summary: dict[str, dict[str, tuple[int, int]]] = {}
    for cat in categories:
        cases_in = [c for c in scored if c.category == cat]
        row = f"{cat:<7}{len(cases_in):<5}"
        summary[cat] = {}
        for method in methods:
            ok = sum(1 for c in cases_in if _score(results[method].get(c.id, []), c)[0])
            summary[cat][method] = (ok, len(cases_in))
            row += f"{ok}/{len(cases_in):<9}"
        emit(row)
    total_row = f"{'合计':<7}{len(scored):<5}"
    for method in methods:
        ok = sum(summary[cat][method][0] for cat in categories)
        total_row += f"{ok}/{len(scored):<9}"
    emit(total_row)

    emit()
    emit("Recall@5（宽松判据：只看来源文件）——按类别：")
    emit(header)
    for cat in categories:
        cases_in = [c for c in scored if c.category == cat]
        row = f"{cat:<7}{len(cases_in):<5}"
        for method in methods:
            ok = sum(1 for c in cases_in if _score(results[method].get(c.id, []), c)[1])
            row += f"{ok}/{len(cases_in):<9}"
        emit(row)
    total_row = f"{'合计':<7}{len(scored):<5}"
    for method in methods:
        ok = sum(1 for c in scored if _score(results[method].get(c.id, []), c)[1])
        total_row += f"{ok}/{len(scored):<9}"
    emit(total_row)

    # ---- 与基线的差值（G2 达标线） ----
    emit()
    emit("与关键词基线的差值（严格判据，百分点；G2 达标线：有答案类 ≥ +15pp）")
    for method in methods:
        if method == "关键词基线":
            continue
        row = f"  {method:<11}"
        for cat in categories:
            base = summary[cat]["关键词基线"][0]
            cur = summary[cat][method][0]
            n = summary[cat][method][1]
            row += f"{cat} {(cur - base) / n * 100:+.1f}pp   "
        emit(row)

    # ---- 覆盖率（跨文档 / 多跳） ----
    multi = [c for c in scored if len(c.expects) > 1]
    if multi:
        emit()
        emit(f"多来源题（{len(multi)} 条）的覆盖率：")
        for method in methods:
            covs = [_score(results[method].get(c.id, []), c)[2] for c in multi]
            emit(f"  {method:<11} 平均 {statistics.mean(covs):.0%}")

    # ---- 无答案组 ----
    no_answer = [c for c in CASES if c.category == CATEGORY_NO_ANSWER]
    emit()
    emit(
        f"无答案组（{len(no_answer)} 条）：检索侧只能做负样本自检"
        "（拒答率要模型答完才判，见 rag/评测说明.md）"
    )
    for case in no_answer:
        emit(f"  {case.id}: {case.question}")

    # ---- 重排用量 ----
    if usages:
        emit()
        emit(f"重排用量：{_usage_summary(usages)}")
        if degraded:
            emit(f"重排失败（已降级为不重排）{len(degraded)} 条：")
            for line in degraded:
                emit(f"  - {line}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "# 第 19 周检索对比表（脚本生成）\n\n"
            "> 生成命令见 `eval/rag_eval.py` 文件头；本文件由脚本生成，改完脚本请重跑。\n"
            "> 评测集与 ground truth 在 `eval/retrieval_cases.py`（含四类与负样本词）。\n"
            "> 口径：Recall@5 按各方法**实际返回的列表**算；重排后可能不足 5 条（台账 T28）。\n\n"
            "```text\n" + "\n".join(lines) + "\n```\n",
            encoding="utf-8",
        )
        print(f"\n[已写入 {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
