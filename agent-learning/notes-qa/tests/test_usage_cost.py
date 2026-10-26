"""日志里必须**落下花费**（第四阶段 H1 的最后一块）。

为什么单独一条：`summarize()` 会算 `cost.usd`，但 2026-10-26 之前**写入端根本没写这个字段**——
于是"这次问答花了多少钱"只能靠人记（README 里那句"0.71 元"就是这么来的，核不出来）。
这条测试走**真实的 `ask()` 路径**，钉住"写出去的记录里有 cost"。
"""

from __future__ import annotations

import json

from _web_fakes import HITS, FakeRetriever, make_calls
from web.service import ask
from web.usage import summarize


def test_ask_writes_cost_into_the_usage_log(tmp_path):
    """走真实 ask 路径写一条日志：`cost` 必须在，且 `summarize` 能加总（no_cost 为 0）。"""
    log = tmp_path / "usage_log.jsonl"
    plan_call, answer_call = make_calls("有方法，见 刀具磨损监测/综述A.pdf:第 2 页")

    ask(
        "刀具磨损监测有哪些方法？",
        retriever=FakeRetriever(fallback=HITS),
        plan_call=plan_call,
        answer_call=answer_call,
        log_path=log,
    )

    records = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records, "应当写了一条日志"
    record = records[-1]
    assert "cost" in record, "日志必须带 cost —— 否则花费只能靠人记"
    assert "usd" in record["cost"], f"cost 里要有 usd：{record['cost']}"
    assert float(record["cost"]["usd"]) > 0

    summary = summarize(log)
    assert summary["no_cost"] == 0, '有 cost 的记录不该被记成"无成本"'
    assert summary["usd"] > 0
