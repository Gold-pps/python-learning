"""对照评测：把**已发表的综述**与系统产出的四段式并排放，让人填"覆盖了没"（第四阶段延伸，离线 0 元）。

**为什么需要它**：这个项目所有指标都是自证的（检索覆盖、引用 0 编造），从来没有**外部材料**来判断
"答得够不够全"。而现成的外部材料就在语料里——那些综述型论文，它们自己的目录就是一份"应该覆盖什么"的清单。

**为什么机器不打分**：台账 T52 已经证明"词面重合度"这类信号不携带"覆盖了没"的信息
（结论是对原文的压缩改写，用词天然不重合）。所以这里只做两件事：

1. 从综述 PDF 里抽出**章节标题**（"应该覆盖什么"）；
2. 把系统产出的每条结论按段列在旁边，**留一列空白给人填**。

判断永远由人做——但**数覆盖面这件事不需要专业背景**（漏了哪一节，翻目录就能看出来）。

用法（在 `agent-learning/notes-qa` 下）：

    uv run python -m eval.review_contrast \
        --pdf "data/corpus/刀具磨损监测/刀具磨损状态监测综述_制造技术与机床2026.pdf" \
        --view eval/综述包-01-刀具磨损监测.json \
        --out eval/对照-刀具磨损监测.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SECTIONS = ("研究现状", "方法对比", "结论", "可复用点")

# 章节标题的形态：`1 引言` / `2.1 刀具磨损监测方法` / `3 结论与展望`（编号 + 短标题）
_HEADING = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2})[\s、.．]+\s*(\S.{0,38})$")
_NOISE = re.compile(r"(图|表|式|第\s*\d+\s*期|Vol|No\.|doi|DOI)")


def headings_from_text(text: str, limit: int = 40) -> list[str]:
    """从正文里抽章节标题（编号 + 短标题），去重保序。抽不到就返回空——**不硬猜**。"""
    found: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        candidate = line.strip()
        if not candidate or len(candidate) > 42 or _NOISE.search(candidate):
            continue
        match = _HEADING.match(candidate)
        if not match:
            continue
        title = match.group(2).strip()
        # 标题里不该有大段标点、满篇数字，也不该有百分号或小数——
        # 第一版漏了后两条，"2 96%、2.53%和1.34%。"这种正文行被当成了章节标题。
        if (
            len(title) < 2
            or title.count("，") > 1
            or _NOISE.search(title)
            or "%" in title
            or re.search(r"\d\.\d", title)
        ):
            continue
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(f"{match.group(1)} {title}")
        if len(found) >= limit:
            break
    return found


def points_from_view(view: dict) -> list[tuple[str, str]]:
    """系统产出：按段列出每条结论（只取要点，不带原文，便于并排看）。"""
    rows: list[tuple[str, str]] = []
    for section in SECTIONS:
        for item in view.get("sections", {}).get(section) or []:
            point = (item.get("point") or "").strip()
            if not point:
                continue
            detail = ""
            if item.get("pros") or item.get("cons"):
                detail = f"（优点：{item.get('pros') or '-'}；局限：{item.get('cons') or '-'}）"
            rows.append((section, point + detail))
    return rows


def build_report(headings: list[str], rows: list[tuple[str, str]], *, source: str, asked: str) -> str:
    """并排报告：左列是"综述说该讲什么"，右列是"系统讲了什么"，中间留白给人填。"""
    lines = [
        "# 对照评测：已发表综述 vs 系统产出",
        "",
        f"> 综述：`{source}`",
        f"> 系统问题：{asked}",
        "> **机器只并排、不打分**：判断「覆盖了没」由人做（数覆盖面不需要专业背景）。",
        "",
        f"## A. 综述的章节（**应该覆盖什么**，共 {len(headings)} 节）",
        "",
    ]
    if headings:
        lines += ["| # | 综述章节 | 系统覆盖了吗？ | 对应系统结论 |", "| - | -------- | -------------- | ------------ |"]
        for index, heading in enumerate(headings, start=1):
            lines.append(f"| {index} | {heading} | ☐ 覆盖　☐ 部分　☐ 漏 |  |")
    else:
        lines.append("（没能从 PDF 里抽出章节标题——请手工看目录，或换一份文字层清晰的版本）")

    lines += [
        "",
        f"## B. 系统产出的结论（**它讲了什么**，共 {len(rows)} 条）",
        "",
    ]
    current = ""
    for section, point in rows:
        if section != current:
            current = section
            lines += ["", f"### {section}", ""]
        lines.append(f"- {point}")

    lines += [
        "",
        "## C. 结论（人填）",
        "",
        "- 综述分了 ___ 节，系统覆盖 ___ 节，**漏了**：",
        "- 系统有、综述没有的（可能是新文献或系统幻觉，需回原文核）：",
        "- 一句话判断：这份产出能不能直接用于 ___（组会 / 开题 / 内部备忘）",
        "",
        "> 提醒：如果某条结论的「对应系统结论」填不出来，说明系统没覆盖它；",
        "> 反之若系统有某条而综述没有，**先别当优点**——回原文看那条引用在不在。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对照评测：已发表综述 vs 系统产出（离线）")
    parser.add_argument("--pdf", required=True, help="综述论文的 PDF 路径")
    parser.add_argument("--view", required=True, help="系统产出：综述包-0N-*.json")
    parser.add_argument("--out", default="", help="报告写到哪（默认打印到 stdout）")
    args = parser.parse_args(argv)

    from rag.parsers import parse_file  # 局部导入：只有这条路径用得到

    blocks = parse_file(Path(args.pdf))
    text = "\n".join(getattr(block, "text", "") or "" for block in blocks)
    headings = headings_from_text(text)
    view = json.loads(Path(args.view).read_text(encoding="utf-8"))
    report = build_report(
        headings,
        points_from_view(view),
        source=args.pdf,
        asked=view.get("question") or "",
    )
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"written: {args.out} (headings={len(headings)})")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
