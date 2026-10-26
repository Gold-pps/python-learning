"""能力说明（`rag/coverage.py`）与"元问题"拦截的离线测试（H4-1）。

要证明的三件事：
1. 能力说明是**一份**结构化的东西（子主题 / 不覆盖 / 示例），界面与健康检查都从它来；
2. 元问题**不检索、不花钱**——用"检索器一次都没被调用"来断言，而不是看返回值像不像说明；
3. 元问题的识别**宁可漏判不要误判**：正常问题绝不能被当成元问题（否则会直接回一段说明，
   把用户的真问题吞掉）。
"""

from __future__ import annotations

import json

from _web_fakes import HITS, FakeRetriever
from fastapi.testclient import TestClient
from rag.coverage import (
    EXAMPLES,
    SUBTOPICS,
    is_capability_question,
    summary,
)
from web.server import create_app


def _parse_sse(text: str) -> list[dict]:
    """把 SSE 文本拆成事件列表。**本地实现**：不借 test_web 的（跨测试文件互相 import 会打架）。"""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def test_summary_is_structured_and_complete():
    """能力说明三件套齐全，且示例问题不少于三条（界面上做成 chip）。"""
    info = summary()

    assert set(info) == {"one_liner", "subtopics", "not_covered", "examples"}
    assert len(info["subtopics"]) == len(SUBTOPICS) >= 5
    assert len(info["examples"]) == len(EXAMPLES) >= 3
    assert info["not_covered"], "必须写明不覆盖什么，否则等于承诺万能"
    assert "资料中没有" in info["one_liner"], "要有“答不了会明说”的承诺"


def test_capability_question_detection_is_conservative():
    """元问题认得出来；**正常问题一个都不许误判**。"""
    for meta in ("你能回答什么问题？", "这个系统能干什么", "能问什么", "你能做什么"):
        assert is_capability_question(meta), meta

    for real in (
        "刀具磨损监测有哪些主流方法？",
        "机床误差补偿怎么建模？",
        "五轴后置处理怎么把刀位文件转成 NC 代码？",
        "AI 辅助工艺编制怎么做？",
    ):
        assert not is_capability_question(real), f"正常问题被误判成元问题：{real}"


def test_health_exposes_capability():
    """界面不该去翻文档才知道能问什么：`/api/health` 直接给。"""
    app = create_app(retriever=FakeRetriever(fallback=HITS), token=None, log_path=None)
    with TestClient(app) as client:
        body = client.get("/api/health").json()

    assert body["capability"]["examples"], "健康检查要带示例问题"
    assert "资料中没有" in body["capability"]["one_liner"]


def test_meta_question_does_not_touch_the_retriever():
    """元问题**不检索、不花钱**：检索器一次都不该被调用（这就是"省一分钱"的硬证据）。"""
    retriever = FakeRetriever(fallback=HITS)
    app = create_app(retriever=retriever, token=None, log_path=None)
    with TestClient(app) as client:
        events = _parse_sse(
            client.post("/api/ask", json={"question": "你能回答什么问题？"}).text
        )

    assert retriever.queries == [], "元问题不该去检索"
    kinds = [event["type"] for event in events]
    assert "capability" in kinds, f"要回一条能力说明事件，实际事件：{kinds}"
    payload = next(event for event in events if event["type"] == "capability")
    assert payload["examples"] == list(EXAMPLES)


def test_real_questions_are_never_treated_as_meta():
    """对照组（**不联网**）：一批真实问题逐一过一遍判据，一个都不能被判成元问题。

    比"发一次 /api/ask 看有没有检索"更省：那条路会真的调模型（花钱）。
    判据是纯函数，这里直接打满它的边界。
    """
    real_questions = [
        "刀具磨损监测有哪些主流方法？",
        "五轴机床后置处理怎么把刀位文件转成 NC 代码？",
        "怎么用切削比能判断刀具磨损？",
        "数字孪生在机床上怎么用？",
        "误差补偿怎么建模，有哪些方法？",
    ]

    for question in real_questions:
        assert not is_capability_question(question), f"被误判成元问题：{question}"


# ---- 问题 → 子主题（加权检索的前置判定）----


def test_subtopic_of_question_recognises_a_clear_direction():
    """问题里明确写了某方向的行话 → 认得出（大小写、全半角不敏感）。"""
    from rag.coverage import subtopic_of_question

    assert subtopic_of_question("刀具磨损监测有哪些主流方法？") == "刀具磨损监测"
    assert (
        subtopic_of_question("五轴机床后置处理怎么把刀位文件转成 NC 代码？")
        == "NC 代码生成与校验"
    )
    assert subtopic_of_question("G 代码自动生成怎么保证正确性？") == "NC 代码生成与校验"


def test_subtopic_of_question_refuses_to_guess_when_ambiguous():
    """**同时命中两个方向时不猜**——加权判错的代价比漏判大得多。"""
    from rag.coverage import subtopic_of_question

    assert subtopic_of_question("工艺路线怎么用数字孪生优化？") is None
    assert subtopic_of_question("误差补偿和刀具磨损有关系吗？") is None


def test_subtopic_of_question_returns_none_for_unrelated_questions():
    """跟语料无关的问题 → None（调用方退回普通检索）。"""
    from rag.coverage import subtopic_of_question

    assert subtopic_of_question("今天天气怎么样？") is None
    assert subtopic_of_question("") is None
