"""使用日志统计（`web/usage.py`）的离线测试（第四阶段 H1）。

要证明的三件事：
1. 数字**算得对**（含"精确/只到文件/编造/无引用/硬事实"分开算）；
2. 坏行**跳过并计数**，不让一行脏数据毁掉全部统计；
3. 日志不存在时**不炸**、返回全 0 —— CI 上没有 `data/`，这条必须过。
"""

from __future__ import annotations

import json

from web.usage import format_summary, main, summarize


def _write_log(tmp_path, records: list[dict]) -> str:
    path = tmp_path / "usage_log.jsonl"
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    return str(path)


def test_summary_adds_up_each_metric_separately(tmp_path):
    """各项独立累加：引用三档、无引用条目、硬事实、降级、花费。"""
    path = _write_log(
        tmp_path,
        [
            {
                "citation_exact": 15,
                "citation_partial": 1,
                "citation_invalid": ["x.pdf:第 9 页"],
                "uncited_points": 2,
                "unsupported_facts": ["68.5（a.pdf:第 2 页）"],
                "degraded": False,
                "cost": {"usd": 0.01},
                "at": "2026-10-19T10:00:00",
            },
            {
                "citation_exact": 5,
                "citation_partial": 0,
                "citation_invalid": [],
                "uncited_points": 0,
                "unsupported_facts": [],
                "degraded": True,
                "cost": {"usd": 0.02},
                "at": "2026-10-20T10:00:00",
            },
        ],
    )

    summary = summarize(path)

    assert summary["questions"] == 2
    assert summary["exact"] == 20
    assert summary["partial"] == 1
    assert summary["invalid"] == 1, "invalid 是列表，要算条数"
    assert summary["uncited"] == 2
    assert summary["unsupported_facts"] == 1
    assert summary["degraded"] == 1
    assert round(summary["usd"], 4) == 0.03
    assert summary["first_day"] == "2026-10-19" and summary["last_day"] == "2026-10-20"


def test_summary_skips_bad_lines_and_counts_them(tmp_path):
    """一行坏数据不该毁掉全部统计——但要**报出来**。"""
    path = tmp_path / "usage_log.jsonl"
    path.write_text(
        '{"citation_exact": 3, "cost": {"usd": 0.01}}\n'
        "这不是 JSON\n"
        '{"citation_exact": 2}\n',
        encoding="utf-8",
    )

    summary = summarize(path)

    assert summary["questions"] == 2
    assert summary["exact"] == 5
    assert summary["bad_lines"] == 1
    assert "跳过坏行" in format_summary(summary), "坏行要在摘要里说一声"
    assert summary["no_cost"] == 1, "没有成本记录的要单独数，不能当 0 混进去"


def test_missing_log_is_reported_not_crashed(tmp_path):
    """日志不存在（CI 就是这样）→ 不炸、全 0、摘要说得清。"""
    summary = summarize(tmp_path / "nope.jsonl")

    assert summary["exists"] is False
    assert summary["questions"] == 0
    assert "不存在" in format_summary(summary)

    assert main(["--summary", "--log", str(tmp_path / "nope.jsonl")]) == 0
