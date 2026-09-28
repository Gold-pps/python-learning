"""回答级评测：检索 → 一次回答 → 判"引用准确率 / 无答案拒答率"（**会花 token**）。

G2 第 4 条要三个数字，其中两个必须"答完才判"，所以单列一个脚本：

| 指标             | 在哪算                                | 花不花钱 |
| ---------------- | ------------------------------------- | -------- |
| Recall@5         | `eval/rag_eval.py`（检索级）          | 否       |
| 引用准确率       | 本脚本                                | **是**   |
| 无答案拒答率     | 本脚本                                | **是**   |

流程（每题**一次**调用）：混合检索 top_k → 片段按 `[文件名:起始行-结束行]` 标注拼进提示词
→ `temperature=0` 回答 → 用 `eval/answer_judge.py` 判引用与拒答。

口径（与第 14 周的教训一致，见台账 T10）：
- **内容与格式分开判**：`引用准确率` 看"给出的引用是否全部落在期望来源里"；
  另单独报 `格式合规率`（有没有用约定的方括号）与 `零引用率`。混在一起会把"格式漂移"
  误报成"没给出处"；那次 C1/D2 就是这么被误判的。
- **无答案题给引用 = 编造**：所以除"拒答率"外另报"干净拒答率"（拒答 **且** 没编造出处）。
- 回答全文会存进 JSON（`--save`）——FAIL 之后能复查，不用重跑一次花钱的实验。

用法（在 `agent-learning/notes-qa` 下）：
    # 先拿一道题试水（1 次调用）
    & "..\\..\\.venv-rag\\Scripts\\python.exe" -m eval.rag_answer_eval --only W18-3

    # 全套（25 题 ≈ 25 次调用）
    & "..\\..\\.venv-rag\\Scripts\\python.exe" -m eval.rag_answer_eval --out eval/第19周回答级评测.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.retriever import Retriever, SearchHit

from .answer_judge import judge_answer
from .retrieval_cases import CASES, CATEGORY_NO_ANSWER, Case

# 固定指令放最前面（与重排同一考虑）：DeepSeek 的 Context Caching 按前缀命中
ANSWER_PROMPT = """你是资料问答助手。只能依据下面给出的检索片段回答，不要使用片段以外的知识。

规则：
1. 每条事实后面必须标出来源，格式为 [文件名:起始行-结束行]，文件名用片段标题里给的那个；
2. 片段里没有足够信息回答时，直接说"资料里没有相关内容"，不要编造、不要推测；
3. 不要复述片段全文，直接给答案，简洁一点。

用户问题：
{query}

检索片段（共 {n} 条）：
{context}
"""


def build_context(hits: list[SearchHit], *, snippet_chars: int = 700) -> str:
    blocks = []
    for hit in hits:
        c = hit.chunk
        text = c.text.strip()
        if len(text) > snippet_chars:
            text = text[:snippet_chars] + "…"
        blocks.append(
            f"[{c.source_path}:{c.start}-{c.end}]（位置：{c.locator}）\n{text}"
        )
    return "\n\n".join(blocks)


ANSWER_MAX_TOKENS = (
    4096  # 与重排同理：模型会先内部推理，预算按"思考 + 答案"估，见台账 T30
)
INVALID_NOTE = "无效（撞 max_tokens、回复为空）"


def is_invalid_answer(text: str, usage: dict | None) -> bool:
    """这次调用是否**无效**（不能拿去算指标）。

    两种情形：回复为空、或 `finish_reason == "length"`（撞上限）。
    实测 25 题里有 10 题是空回复——若不单独拎出来，"零引用率"会把它们算成
    "没给出处"，指标就撒谎了（第 19 周第一版就是这么错的：12/20 里有一大半是空回复）。
    """
    if not text.strip():
        return True
    return (usage or {}).get("finish_reason") == "length"


def build_prompt(case: Case, hits: list[SearchHit]) -> str:
    return ANSWER_PROMPT.format(
        query=case.question.strip(),
        n=len(hits),
        context=build_context(hits),
    )


def rescore_saved(path: Path) -> list[dict]:
    """读回已保存的回答，用**当前判据**重新判分（不调用模型）。

    为什么要有这个开关：判据也是代码，也会写错。第 19 周第一版把"空回复"算成
    "零引用"，如果改正判据必须重跑 25 次调用才能看到新数字，那改判据的代价就太高了
    ——保存回答全文（`--save`）就是为了这一刻（第 14 周 T10 的教训）。
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        negative = row.get("category") == CATEGORY_NO_ANSWER
        verdict = judge_answer(
            row["answer"], expect_files=row.get("expect_files", []), negative=negative
        )
        row.update(
            {
                "citations": sorted(verdict.citations),
                "has_bracket": verdict.has_bracket,
                "refusal": verdict.refusal,
                "cited_expected": verdict.cited_expected,
                "fabricated": verdict.fabricated,
                "invalid": is_invalid_answer(row["answer"], row.get("usage")),
            }
        )
    return rows


def _emit_metrics(per_case: list[dict], emit) -> None:
    """指标块。**只统计有效调用**；空回复/撞上限的单独列出来，绝不混进比率里。"""
    usages = [item["usage"] for item in per_case if item["usage"]]
    # 指标从"实际跑出来的结果"里取，而不是从用例列表取——`--dry-run` 时 per_case 是空的，
    # 从用例列表取会把 Case 当 dict 用（冒烟测试时真踩到）。
    valid = [r for r in per_case if not r["invalid"]]
    invalid_rows = [r for r in per_case if r["invalid"]]
    answered = [r for r in valid if r["category"] != CATEGORY_NO_ANSWER]
    refused = [r for r in valid if r["category"] == CATEGORY_NO_ANSWER]

    def _rate(rows: list[dict], key: str) -> str:
        if not rows:
            return "n/a"
        ok = sum(1 for r in rows if r[key])
        return f"{ok}/{len(rows)}"

    emit()
    emit("回答级指标（**只统计有效调用**）：")
    emit(f"  有效调用                              {len(valid)}/{len(per_case)}")
    if invalid_rows:
        emit(
            f"  无效（撞 max_tokens、回复为空）        {len(invalid_rows)} 条："
            + "、".join(r["id"] for r in invalid_rows)
        )
        emit("  ↑ 空回复不能算作零引用，否则指标会撒谎（第 19 周第一版就是这么错的）")
    if answered:
        emit(
            f"  引用准确率（引用全部落在期望来源里）  {_rate(answered, 'cited_expected')}"
        )
        emit(
            f"  格式合规率（用了 [文件名:行号]）      {_rate(answered, 'has_bracket')}"
        )
        emit(
            f"  零引用率（完全没给出处）              "
            f"{sum(1 for r in answered if not r['citations'])}/{len(answered)}"
        )
    if refused:
        emit(f"  无答案拒答率                          {_rate(refused, 'refusal')}")
        clean = sum(1 for r in refused if r["refusal"] and not r["citations"])
        emit(f"  干净拒答率（拒答且没编造引用）        {clean}/{len(refused)}")

    if usages:
        emit()
        emit("用量（记账口径）：")
        emit(f"  调用次数 {len(usages)}")
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            emit(f"  {key:<18}{sum(u.get(key) or 0 for u in usages)}")


def _write_report(args, lines: list[str]) -> int:
    """把本次输出写成 UTF-8 报告（命令行入口与 --rescore 共用）。"""
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "# 第 19 周回答级评测（脚本生成）\n\n"
            "> 生成命令见 `eval/rag_answer_eval.py` 文件头；改判据后用 --rescore 重判即可，不必重跑。\n"
            "> 判据在 `eval/answer_judge.py`：内容（引用是否可回查）与格式（是否用方括号）分开统计。\n"
            "> 回答全文在 `eval/rag_answer_last_run.json`（含每题 usage）。\n"
            "> **只统计有效调用**：撞 max_tokens 的空回复单独列出，它们不算零引用。\n\n"
            "```text\n" + "\n".join(lines) + "\n```\n",
            encoding="utf-8",
        )
        print(f"\n[已写入 {out}]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="第 19 周回答级评测（引用准确率 / 拒答率）"
    )
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", default="fastembed", choices=["bge", "fastembed"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", default="", help="逗号分隔的题号：先拿 1~2 道试水")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=ANSWER_MAX_TOKENS,
        help="单次回答的 token 上限（含模型的内部思考，别按答案长度估）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只拼提示词、打印长度，**不调用模型**"
    )
    parser.add_argument(
        "--rescore",
        default="",
        help="用已保存的 JSON 重新判分（**不调用模型**）：判据改了就别重跑、重判即可",
    )
    parser.add_argument("--out", default="", help="报告写成 UTF-8 文件")
    parser.add_argument(
        "--save",
        default="eval/rag_answer_last_run.json",
        help="回答全文存这里（便于复查）",
    )
    args = parser.parse_args()

    from rag.embedder import Embedder
    from rag.rerank import chat
    from rag.retriever import make_stdout_forgiving

    make_stdout_forgiving()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    cases = [c for c in CASES if not only or c.id in only]

    # `--rescore` 不需要检索、也不需要模型：只读保存下来的回答重新判分
    retriever = None
    if not args.rescore:
        embedder = Embedder(backend=args.backend, device=args.device)
        retriever = Retriever.from_index(args.index, embedder)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    if retriever is not None:
        emit(
            f"索引：{args.index}｜片段 {len(retriever.chunks)}｜后端 {args.backend}｜"
            f"Top-{args.top_k}"
        )
    emit(f"题量：{len(cases)}（{args.only or '全部'}）")
    emit()

    per_case: list[dict] = []

    if args.rescore:
        per_case = rescore_saved(Path(args.rescore))
        emit(f"[重新判分] {args.rescore}：{len(per_case)} 条（**不调用模型**）")
        emit(f"{'id':<7}{'类别':<5}{'有效':<5}引用")
        for row in per_case:
            emit(
                f"{row['id']:<7}{row['category']:<5}"
                f"{'Y' if not row['invalid'] else '-':<5}"
                f"{'、'.join(row['citations']) or '-'}"
            )
        _emit_metrics(per_case, emit)
        return _write_report(args, lines)

    emit(f"{'id':<7}{'类别':<5}{'引用':<22}{'格式':<5}{'拒答':<5}备注")
    for case in cases:
        hits = retriever.search(case.question, args.top_k)
        prompt = build_prompt(case, hits)
        if args.dry_run:
            emit(
                f"{case.id:<7}{case.category:<5}检索 {len(hits)} 条｜提示词 {len(prompt)} 字符"
            )
            continue
        text, usage = chat(prompt, max_tokens=args.max_tokens)
        negative = case.category == CATEGORY_NO_ANSWER
        verdict = judge_answer(text, expect_files=case.expect_files, negative=negative)
        invalid = is_invalid_answer(text, usage)

        if invalid:
            note = INVALID_NOTE
        elif negative:
            note = "干净拒答" if (verdict.refusal and verdict.no_citation) else ""
            if verdict.fabricated:
                note = f"编造引用 {sorted(verdict.citations)}"
        else:
            note = (
                ""
                if verdict.cited_expected
                else (f"引用={sorted(verdict.citations) or '无'}")
            )
        cites = ",".join(sorted(verdict.citations)) or "-"
        emit(
            f"{case.id:<7}{case.category:<5}{cites[:20]:<22}"
            f"{'Y' if verdict.has_bracket else '-':<5}"
            f"{'Y' if verdict.refusal else '-':<5}{note}"
        )
        per_case.append(
            {
                "id": case.id,
                "category": case.category,
                "question": case.question,
                "expect_files": list(case.expect_files),
                "hits": [h.to_dict() for h in hits],
                "answer": text,
                "usage": usage,
                "citations": sorted(verdict.citations),
                "has_bracket": verdict.has_bracket,
                "refusal": verdict.refusal,
                "cited_expected": verdict.cited_expected,
                "fabricated": verdict.fabricated,
                "invalid": invalid,
            }
        )

    _emit_metrics(per_case, emit)

    if args.save and not args.dry_run:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(per_case, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        emit(f"\n回答全文已存：{args.save}")

    return _write_report(args, lines)


if __name__ == "__main__":
    raise SystemExit(main())
