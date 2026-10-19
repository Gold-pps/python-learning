"""核对清单生成器（第三阶段 H3）：把一次问答的产物转成**人能逐条核对**的清单。

它解决的是一条硬限制：语料是机械 / 数控方向的论文，而**使用者不是这个领域的**，
于是"结论对不对"没人能判断。清单绕开这件事——它不问"专业上对不对"，
只问一件事：**这条结论能不能在它引用的原文里找到依据。**

"原文有没有这么说"不需要专业背景（把方法 A 的数值安到方法 B 上、把限定条件抹掉、
写了原文没写的，这些一眼能看出来）——所以**作者自己换个时间重读也能做这件事**。

三条它刻意做的事：

1. **把原文摆在结论旁边**：核对着不用去翻索引、也不用懂检索——
   看到的就是**模型当时看到的那段**（600 字片段），这正是"它有没有依据"的直接证据；
2. **把"一条引用都没挂"的条目单独标出来**：那是风险最高的一类（结论完全没有出处）；
3. **留出"核对者是谁 / 属于哪一档证据"**：清单本身不许声称"已核对"——
   谁核的、什么背景、算哪一档证据，必须写下来（规划里就是这么要求的）。

用法（在 `agent-learning/notes-qa` 下）：

```powershell
# 从已有的视图 JSON 生成（**不花钱**；视图 JSON 可以从夹具、也可以从一次真问答导出）
uv run python -m web.audit --view tests/fixtures/sample_view.json --out eval/核对清单-示例.md

# 现场跑一次问答再生成（**会花钱**，2~7 分；写完视图 JSON 与清单两份产物）
uv run python -m web.audit --ask "刀具磨损监测有哪些主流方法？" --out eval/核对清单-刀具.md
```
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag.review_qa import REVIEW_SECTIONS

CHOICES = "☐ 支持　☐ 不支持　☐ 无法判断"

HEADER = """# 核对清单（原文支不支持这条结论）

> **这份清单不判断“专业上对不对”**——它只问一件事：**这条结论，能不能在它引用的原文里找到依据。**
> 这个判断不需要专业背景（只看原文有没有这么说），但能抓出最危险的那类错：
> 结论写了原文没写的、把方法 A 的数值安到方法 B 上、把限定条件抹掉。
>
> 怎么填：每条结论后面勾一个（**支持 / 不支持 / 无法判断**），不支持的写出理由。
> 填完在文末写上**核对者是谁**（含背景）与**这属于哪一档证据**，连同清单一起交回（`eval/` 下留档）。
"""


def _rows(view: dict) -> list[tuple[str, dict]]:
    """把视图摊成 `(段名, 条目)` 列表，**按固定段序**（研究现状 / 方法对比 / 结论 / 可复用点）。"""
    sections = view.get("sections") or {}
    out: list[tuple[str, dict]] = []
    for key in REVIEW_SECTIONS:
        for item in sections.get(key) or []:
            out.append((key, item))
    return out


def _quote_lines(cite: dict, candidates: list) -> list[str]:
    """一条引用 → 清单里那几行（含原文）。**三种 kind 的处理不同**，因为风险不同。"""
    text = cite.get("text") or ""
    kind = cite.get("kind")
    lines: list[str] = []

    if kind == "exact":
        index = cite.get("hit_index")
        candidate = (
            candidates[index]
            if isinstance(index, int) and index < len(candidates)
            else None
        )
        lines.append(f"- 引用：`{text}`（精确可回查）")
        lines.append(
            f"  > {(candidate or {}).get('text', '（视图里没有带出这段原文）')}"
        )
    elif kind == "partial":
        lines.append(f"- 引用：`{text}`（只到文件）")
        lines.append("  - ⚠️ **只到文件、没给到页**——片段是该文件本次命中的，请自行定位")
        for index in cite.get("file_hits") or []:
            if isinstance(index, int) and index < len(candidates):
                lines.append(f"  > {candidates[index].get('text', '')}")
    else:  # unresolved：正常路径走不到（校验那一步就把编造引用从条目上摘掉了）
        lines.append(f"- 引用：`{text}`（**回查不到**）")
    return lines


# "硬事实" = 数字（含小数）与英文标识符（型号 / 缩写）。它们**能被精确核对**，
# 也最容易出错（把方法 A 的数值安到方法 B 上）。
#
# 为什么不用"实词是否出现"：第一版就是拿实词做的，**42/46 条全被标** = 噪声——
# 结论是对原文的**压缩改写**，用词本来就不一样，重合率天然低（台账 T52）。
_HARD_FACT = re.compile(r"[A-Za-z]{2,}[A-Za-z0-9]*|\d+(?:\.\d+)?")
_CITATION_MARK = re.compile(r"\[\d+\]")  # 结论里「候选[5]」这类标记，别当成数字抓


def _cited_text(item: dict, candidates: list) -> str:
    """该条目引用的原文（拼起来）。只认 exact 的 hit 与 partial 的 file_hits。"""
    texts: list[str] = []
    for cite in item.get("citations") or []:
        index = cite.get("hit_index")
        if (
            cite.get("kind") == "exact"
            and isinstance(index, int)
            and index < len(candidates)
        ):
            texts.append(candidates[index].get("text") or "")
        for file_index in cite.get("file_hits") or []:
            if isinstance(file_index, int) and file_index < len(candidates):
                texts.append(candidates[file_index].get("text") or "")
    return "\n".join(texts)


def hard_fact_gaps(view: dict) -> list[dict]:
    """逐条结论核**硬事实**：结论里的数字 / 英文型号，是否出现在它引用的原文里。

    两档提示，都是"该看什么"，不是判断：

    - **出现在别的候选里** → 数值对得上、但**引用挂错了**（把 B 的出处挂在 A 上）；
    - **这次问答的所有候选里都没有** → **依据不在原文里**（模型可能凭记忆补的），
      核对着只能记"无法判断"——它不等于错，但**没法从原文核**。

    `AI` 这类词也会被列出来，属于第二种情况的特例：缺口题里结论在讨论"AI 辅助工艺编制"，
    而引用的都是数控论文——**它恰好暴露"答的不是这个问题"**，不是缺陷。

    返回与 `_rows()` **同序**的列表：`{section, point, missing: [{fact, elsewhere}], no_source}`。
    """
    candidates = view.get("candidates") or []
    report: list[dict] = []
    for section, item in _rows(view):
        cited = _cited_text(item, candidates)
        claim = _CITATION_MARK.sub(
            " ",
            " ".join(
                filter(
                    None,
                    [
                        item.get("point") or "",
                        item.get("pros") or "",
                        item.get("cons") or "",
                    ],
                )
            ),
        )
        # 该条目引用的 `文件:位置`。**这一档很关键**：引用是页级的，
        # 同一页常常被切成好几段（T49），所以"数在另一个候选里"未必是挂错——
        # 只要那个候选与引用**同一文件同一位置**，读者照着引用照样查得到（T53）。
        cited_texts = {
            str(cite.get("text") or "").strip()
            for cite in item.get("citations") or []
            if isinstance(cite, dict)
        }
        cited_places = {
            f"{candidate.get('source_path')}:{candidate.get('locator')}"
            for candidate in candidates
            if f"{candidate.get('source_path')}:{candidate.get('locator')}"
            in cited_texts
        }
        facts = list(dict.fromkeys(_HARD_FACT.findall(claim)))
        missing: list[dict] = []
        for fact in facts:
            needle = fact.casefold()
            if needle in cited.casefold():
                continue
            elsewhere = [
                index
                for index, candidate in enumerate(candidates)
                if needle in (candidate.get("text") or "").casefold()
            ]
            same_place = [
                index
                for index in elsewhere
                if f"{candidates[index].get('source_path')}:{candidates[index].get('locator')}"
                in cited_places
            ]
            missing.append(
                {
                    "fact": fact,
                    "elsewhere": elsewhere,
                    "same_place": same_place,
                }
            )
        report.append(
            {
                "section": section,
                "point": item.get("point") or "",
                "missing": missing,
                "no_source": not cited.strip(),
            }
        )
    return report


def build_checklist(view: dict, *, source: str = "") -> str:
    """把视图转成核对清单（Markdown）。纯函数：不联网、不花钱、可离线测。"""
    rows = _rows(view)
    gaps = hard_fact_gaps(view)
    candidates = view.get("candidates") or []
    counts = view.get("citations") or {}
    elapsed = view.get("elapsed") or {}
    cost = view.get("cost") or {}

    out: list[str] = [HEADER]
    out.append(f"- 问题：**{view.get('question') or '（未记录）'}**")
    out.append(f"- 检索词：`{view.get('retrieval_query') or ''}`")
    if view.get("carried_from"):
        out.append(f"- 追问带上了上一问：`{view['carried_from']}`")
    if source:
        out.append(f"- 产物来源：`{source}`")
    out.append(
        "- 引用校验（代码对账）：精确 {exact} / 只到文件 {partial} / 编造 {invalid} / 无引用条目 {uncited}".format(
            exact=counts.get("exact", 0),
            partial=counts.get("partial", 0),
            invalid=len(counts.get("invalid") or []),
            uncited=counts.get("uncited", 0),
        )
    )
    if elapsed or cost:
        out.append(
            "- 本次耗时 {total}s、估算成本 ${usd}（{peak}价）".format(
                total=elapsed.get("total_s", "?"),
                usd=cost.get("usd", "?"),
                peak="高峰" if cost.get("peak") else "低谷",
            )
        )
    if view.get("degraded"):
        out.append(f"- ⚠️ **这一次是降级回答**：{view.get('reason') or ''}")
    if view.get("insufficient"):
        out.append(
            "- ⚠️ **模型自己声明资料不足以回答**（下面给的是相邻证据）："
            f"{view.get('note') or ''}"
        )
    out.append(f"- 待核对结论数：**{len(rows)}**")
    suspects = sorted(
        (
            (len(gap["missing"]), number)
            for number, gap in enumerate(gaps, start=1)
            if gap.get("missing")
        ),
        key=lambda pair: (-pair[0], pair[1]),
    )[:5]
    if suspects:
        listed = "、".join(
            f"第 {number} 条（{count} 项）" for count, number in suspects
        )
        out.append(
            f"- **最值得先看**：{listed}——这些条目的结论里有**数字或型号**在它引用的原文里找不到"
            "（“把 A 的数值安到 B 上”那类错最容易在这里露出来）。"
            "**只挑这几条对着原文看一眼**，比几十条全过一遍有效"
        )
    unsourced = [
        number for number, gap in enumerate(gaps, start=1) if gap.get("no_source")
    ]
    if unsourced:
        out.append(f"- ⚠️ 没有可核原文的条目：第 {'、'.join(map(str, unsourced))} 条")
    out.append("")

    invalid = counts.get("invalid") or []
    if invalid:
        out.append("## 编造的引用（不在正文里，单独列出）")
        out.append("")
        out.append(
            "这些引用**没有挂进任何结论**（校验那一步就摘掉了），列在这里是为了让人看见问题："
        )
        out.extend(f"- `{item}`" for item in invalid)
        out.append("")

    out.append("---")
    out.append("")
    for number, (section, item) in enumerate(rows, start=1):
        title = item.get("point") or "（这条没写结论）"
        extra = ""
        if item.get("pros") or item.get("cons"):
            extra = f"（优点：{item.get('pros', '-')}；局限：{item.get('cons', '-')}）"
        out.append(f"## {number}. 〔{section}〕{title}{extra}")
        out.append("")
        citations = item.get("citations") or []
        if not citations:
            out.append("- 引用：（这条结论**一条引用都没挂**——无引用条目，重点看）")
        for cite in citations:
            out.extend(_quote_lines(cite, candidates))
        gap = gaps[number - 1] if number - 1 < len(gaps) else {}
        for flagged in gap.get("missing") or []:
            fact = flagged.get("fact", "")
            elsewhere = flagged.get("elsewhere") or []
            same_place = flagged.get("same_place") or []
            if same_place:
                out.append(
                    f"- ⚠️ 硬事实核对：`{fact}` **在它引用的同一页的另一段里**"
                    f"（第 {'、'.join(map(str, same_place))} 条）——引用仍可回查，"
                    "只是模型没把这一段挂准（T53）"
                )
            elif elsewhere:
                out.append(
                    f"- ⚠️ 硬事实核对：`{fact}` **出现在别的文件/页的候选里**"
                    f"（第 {'、'.join(map(str, elsewhere))} 条）——**引用挂错了**"
                )
            else:
                out.append(
                    f"- ⚠️ 硬事实核对：`{fact}` **模型这次看到的候选片段里都没有**"
                    "——**不等于整页没有**（片段只送 600 字，整页原文可能有，需你自己去 PDF 核）；"
                    "但对“依据是否在模型看到的原文里”这个问题，只能记“无法判断”"
                )
        if gap.get("no_source"):
            out.append("- ⚠️ 这条没有可核的原文——**先看这一条**")
        out.append("")
        out.append(f"- 判断：{CHOICES}")
        out.append("- 理由（不支持的必填）：")
        out.append("")

    out.append("---")
    out.append("")
    out.append("## 结论表（填完交回）")
    out.append("")
    out.append("| # | 段 | 结论（前 30 字） | 判断 | 理由 |")
    out.append("| - | -- | ---------------- | ---- | ---- |")
    for number, (section, item) in enumerate(rows, start=1):
        title = (item.get("point") or "")[:30]
        out.append(f"| {number} | {section} | {title} |  |  |")
    out.append("")
    out.append("## 核对者（必填，清单不代替它）")
    out.append("")
    out.append("- 核对者：＿＿＿＿＿　背景：＿＿＿＿＿（**与本领域的关系要写清**）")
    out.append("- 日期：＿＿＿＿＿")
    out.append(
        "- 这属于哪一档证据：☐ 非本领域读者逐条看原文（**能判“原文有没有这么说”，"
        "判不了“专业上对不对”**）　☐ 本领域读者复核　☐ 其他：＿＿＿＿＿"
    )
    out.append("- 总体备注：")
    out.append("")
    return "\n".join(out)


def _load_view(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """命令行：`--view <json>`（离线）或 `--ask <问题>`（花钱）→ 核对清单。"""
    import argparse

    parser = argparse.ArgumentParser(description="把一次问答的产物转成核对清单（H3）")
    parser.add_argument("--view", help="已有的视图 JSON（离线、不花钱）")
    parser.add_argument("--ask", help="现场跑一次问答再生成（**会花钱**，2~7 分）")
    parser.add_argument(
        "--out", help="输出的 Markdown 路径（默认为 eval/核对清单-<时间戳>.md）"
    )
    parser.add_argument("--index", help="现场问答用的索引目录（默认与 web/MCP 一致）")
    parser.add_argument(
        "--no-multihop", action="store_true", help="现场问答只跑一轮（省一次模型调用）"
    )
    args = parser.parse_args(argv)

    if bool(args.view) == bool(args.ask):
        parser.error("--view 与 --ask 必须二选一")

    if args.view:
        view_path = Path(args.view)
        view = _load_view(view_path)
        source = str(view_path)
    else:
        # 复用 web 层那一套（含它的懒加载检索器），**不写第二份**
        from web.server import create_app
        from web.service import ask

        app = create_app(
            index_dir=Path(args.index) if args.index else None, log_path=None
        )
        view = ask(
            args.ask,
            app.state.get_retriever(),
            multihop=not args.no_multihop,
        )
        view_path = None
        source = "现场问答（deepseek-flash）"

    text = build_checklist(view, source=source)

    if args.out:
        out_path = Path(args.out)
    elif view_path is not None:
        out_path = Path("eval") / f"核对清单-{view_path.stem}.md"
    else:
        out_path = Path("eval") / "核对清单-现场问答.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8", newline="\n")

    if view_path is None:
        # 现场问答顺手把视图 JSON 也留一份：以后重生成清单不必再花钱
        view_json = out_path.with_suffix(".json")
        view_json.write_text(
            json.dumps(view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[视图 JSON] {view_json}（下次可用 --view 直接生成，不必再花钱）")
    print(f"[核对清单] {out_path}")
    print(f"  待核对结论 {len(_rows(view))} 条；引用校验：{view.get('citations')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
