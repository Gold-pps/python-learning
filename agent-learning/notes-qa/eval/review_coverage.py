"""第 21 周检索冒烟：综述型问题在 44 篇论文语料上的**检索侧**覆盖（离线，不花 token）。

它只回答一个问题：**在花钱调模型之前，检索有没有把该方向的几篇论文捞上来？**

为什么值得先跑这个便宜的：

- 生成侧（`rag/review_qa.py`）每次都要花 token，而它拿到的候选全部来自检索；
  检索跑偏时，模型再强也只能组织出一篇"看起来很像"的废话，而且查不出问题在哪；
- 第 20 周实测已有两个具体症状：**同一篇综述霸榜**（top-5 里 3 条来自同一篇）与
  **跨子主题串台**（"研究现状"把相邻方向拉进来）。所以这里同时报"不限上限"与
  "每篇上限 N"两栏，看这两条约束实际买到多少东西。

用法（在 `agent-learning/notes-qa` 下；第 21 周起主 .venv 也能跑，fastembed 是可选依赖）：

    uv run python -m eval.review_coverage --out eval/第21周检索冒烟.md

控制台只打印 ASCII 标记的摘要（Windows 控制台是 GBK，见台账 T23），细节进报告文件。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.embedder import Embedder
from rag.retriever import Retriever, SearchHit, make_stdout_forgiving, subtopic_of

from .review_cases import CASES, ReviewCase

# 文件名里出现这些词 → 更像"综述型文献"，用来解释"为什么这篇会排到第一"
_REVIEW_HINTS = ("进展", "综述", "现状", "趋势")


def _is_review(source_path: str) -> bool:
    return any(hint in source_path for hint in _REVIEW_HINTS)


def _stats(hits: list[SearchHit], case: ReviewCase) -> dict:
    papers = {hit.chunk.source_path for hit in hits}
    in_subtopic = sum(
        1 for hit in hits if subtopic_of(hit.chunk.source_path) == case.subtopic
    )
    return {
        "n": len(hits),
        "papers": len(papers),
        "in_subtopic": in_subtopic,
        "has_review": any(_is_review(hit.chunk.source_path) for hit in hits),
    }


def _case_block(case: ReviewCase, results: dict[str, list[SearchHit]]) -> list[str]:
    lines = [f"### {case.id} {case.subtopic}", "", f"问题：{case.question}", ""]
    for label, hits in results.items():
        lines.append(f"**{label}**")
        lines.append("")
        if not hits:
            lines.append("- （无命中）")
        for rank, hit in enumerate(hits, start=1):
            same = (
                "OK"
                if subtopic_of(hit.chunk.source_path) == case.subtopic
                else "!! 跨子主题"
            )
            review = " 综述类" if _is_review(hit.chunk.source_path) else ""
            text = hit.chunk.text.replace("\n", " ").strip()[:40]
            lines.append(
                f"{rank}. [{hit.score:.5f}] `{hit.chunk.source_path}`"
                f"（{hit.chunk.locator}）{hit.channels} | {same}{review} | {text}"
            )
        lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="第 21 周综述型问题的检索侧覆盖（离线）"
    )
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-per-source", type=int, default=2)
    parser.add_argument(
        "--include-unreadable",
        action="store_true",
        help="不排除那 2 篇 /G 乱码论文（默认排除，与检索层默认一致）",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    make_stdout_forgiving()
    embedder = Embedder(backend="fastembed")
    retriever = Retriever.from_index(args.index, embedder)

    excluded = len(retriever.unreadable_rows())
    emit_lines: list[str] = []
    emit = emit_lines.append

    emit("# 第 21 周检索冒烟：综述型问题的检索侧覆盖（离线，不花 token）")
    emit("")
    emit(
        f"> 索引：`{args.index}`｜片段 {len(retriever.chunks)}｜top_k={args.top_k}"
        f"｜上限 {args.max_per_source} 条/篇｜排除「仅存在性可检索」{excluded} 条片段"
    )
    emit("> 生成侧要花 token，见 `rag/review_qa.py`；本文件只测检索。")
    emit("")
    emit("## 语料子主题（第一段路径）")
    emit("")
    for name, count in retriever.subtopics().items():
        emit(f"- {name}：{count} 片段")
    emit("")

    summary: dict[str, tuple[dict, dict]] = {}
    for case in CASES:
        plain = retriever.search(
            case.question,
            args.top_k,
            max_per_source=None,
            exclude_unreadable=not args.include_unreadable,
        )
        capped = retriever.search(
            case.question,
            args.top_k,
            max_per_source=args.max_per_source,
            exclude_unreadable=not args.include_unreadable,
        )
        summary[case.id] = (_stats(plain, case), _stats(capped, case))
        emit_lines.extend(
            _case_block(
                case,
                {
                    "不加上限（第 18/19 周口径）": plain,
                    f"每篇上限 {args.max_per_source}": capped,
                },
            )
        )

    emit("## 汇总")
    emit("")
    emit(
        "| 题 | 子主题 | 不加上限：论文数 / 子主题命中 | 加上限：论文数 / 子主题命中 | 含综述类 |"
    )
    emit(
        "| -- | ------ | ----------------------------- | --------------------------- | -------- |"
    )
    for case in CASES:
        plain, capped = summary[case.id]
        emit(
            f"| {case.id} | {case.subtopic} | "
            f"{plain['papers']} 篇 / {plain['in_subtopic']}/{plain['n']} | "
            f"{capped['papers']} 篇 / {capped['in_subtopic']}/{capped['n']} | "
            f"{'是' if capped['has_review'] else '否'} |"
        )
    emit("")

    print(f"[检索冒烟] 片段 {len(retriever.chunks)}，排除 {excluded} 条不可引用片段")
    print(f"{'题':<7}{'子主题':<22}{'不加上限':<16}{'加上限'}")
    for case in CASES:
        plain, capped = summary[case.id]
        p = f"{plain['papers']}篇 {plain['in_subtopic']}/{plain['n']}"
        c = f"{capped['papers']}篇 {capped['in_subtopic']}/{capped['n']}"
        print(f"{case.id:<7}{case.subtopic:<22}{p:<16}{c}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(emit_lines) + "\n", encoding="utf-8")
        print(f"[检索冒烟] 报告写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
