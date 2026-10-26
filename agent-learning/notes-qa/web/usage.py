"""使用日志统计（第四阶段 H1）：**一条命令给出文档里要引用的那些数字**。

为什么需要它：README 与汇报材料里的"26 次提问 / 0.71 元 / 累计 N 条引用"都是**手抄**的，
一抄就会过期——台账 **T56** 已经咬了三次（`523/3073`、`657/2501`、`574 条`），
每次都是"语料或使用变了、文档没变"，而且**没有任何测试会发现**。

有了这个命令，文档就有第三条路可选：不给数字、或者给"跑 `python -m web.usage --summary` 看"。

用法（在 `agent-learning/notes-qa` 下）：

    uv run python -m web.usage --summary
    uv run python -m web.usage --summary --json     # 给脚本/测试用

**口径**（写清楚，否则又是"数对不上"）：
- 一条记录 = 一次问答（日志是追加写的，`source` 区分 web / cli / mcp）；
- `精确 / 只到文件 / 编造` 直接取记录里的 `citation_exact` / `citation_partial` / `citation_invalid`；
- 花费取 `cost.usd`（缺失则记 0 并单独报"无成本记录的条数"）——**不猜**。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parents[1] / "data" / "usage_log.jsonl"


def summarize(log_path: str | Path = DEFAULT_LOG) -> dict:
    """统计一份 JSONL 使用日志。坏行**跳过并计数**，不让一行脏数据毁掉全部统计。"""
    path = Path(log_path)
    summary = {
        "path": str(path),
        "exists": path.exists(),
        "questions": 0,
        "exact": 0,
        "partial": 0,
        "invalid": 0,
        "uncited": 0,
        "unsupported_facts": 0,
        "degraded": 0,
        "usd": 0.0,
        "no_cost": 0,
        "bad_lines": 0,
        "first_day": "",
        "last_day": "",
    }
    if not path.exists():
        return summary

    days: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            summary["bad_lines"] += 1
            continue
        if not isinstance(record, dict):
            summary["bad_lines"] += 1
            continue

        summary["questions"] += 1
        summary["exact"] += int(record.get("citation_exact") or 0)
        summary["partial"] += int(record.get("citation_partial") or 0)
        summary["invalid"] += len(record.get("citation_invalid") or [])
        summary["uncited"] += int(record.get("uncited_points") or 0)
        summary["unsupported_facts"] += len(record.get("unsupported_facts") or [])
        summary["degraded"] += 1 if record.get("degraded") else 0

        cost = record.get("cost")
        usd = None
        if isinstance(cost, dict):
            usd = cost.get("usd")
        elif isinstance(cost, (int, float)):
            usd = cost
        if isinstance(usd, (int, float)):
            summary["usd"] += float(usd)
        else:
            summary["no_cost"] += 1

        day = str(record.get("at") or record.get("ts") or "")[:10]
        if day:
            days.append(day)

    if days:
        summary["first_day"] = min(days)
        summary["last_day"] = max(days)
    return summary


def format_summary(summary: dict) -> str:
    """给人看的一段（口径写在里面，省得下次再问"这个数怎么来的"）。"""
    if not summary["exists"]:
        return f"使用日志不存在：{summary['path']}（还没跑过 web/server 或 cli？）"
    lines = [
        f"日志：{summary['path']}",
        f"问答次数：{summary['questions']}"
        + (
            f"（{summary['first_day']} ~ {summary['last_day']}）"
            if summary["first_day"]
            else ""
        ),
        (
            f"引用：精确 {summary['exact']} / 只到文件 {summary['partial']}"
            f" / 编造 {summary['invalid']} / 无引用条目 {summary['uncited']}"
        ),
        f"硬事实待核：{summary['unsupported_facts']} 项；降级回答：{summary['degraded']} 次",
        f"花费：${summary['usd']:.4f} ≈ {summary['usd'] * 7.1:.2f} 元"
        + (
            f"（另有 {summary['no_cost']} 条无成本记录，未计入）"
            if summary["no_cost"]
            else ""
        ),
    ]
    if summary["bad_lines"]:
        lines.append(f"⚠️ 跳过坏行 {summary['bad_lines']} 条（不是 JSON，或不是对象）")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="使用日志统计（第四阶段 H1）")
    parser.add_argument("--summary", action="store_true", help="打印一段人读的摘要")
    parser.add_argument(
        "--json", action="store_true", help="输出 JSON（给脚本/测试用）"
    )
    parser.add_argument(
        "--log", default=str(DEFAULT_LOG), help="日志路径（默认 data/usage_log.jsonl）"
    )
    args = parser.parse_args(argv)

    summary = summarize(args.log)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
