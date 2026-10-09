"""综述型问答（`rag/review_qa.py`）的离线测试 —— 一次真实调用都没有。

`plan_call` / `answer_call` 全部注入假函数，所以主 .venv 里跑、不花 token。

**失败路径与成功路径一样重要**：这个模块一半的代码在处理"模型没按预期回"，
而第 18 周两次真跑踩的坑（被 `max_tokens` 截断、代码与提示词自相矛盾）
全都出在降级分支上——只测 happy path 等于没测。
"""

from __future__ import annotations

import json

from rag.chunk import Chunk
from rag.retriever import SearchHit
from rag.review_qa import (
    ANSWER_PROMPT,
    REVIEW_PLAN_MAX_TOKENS,
    REVIEW_SECTIONS,
    Plan,
    ReviewResult,
    _render,
    answer_review,
    build_candidates_block,
    citation_of,
    parse_answer,
    parse_plan,
    should_use_thinking,
    strip_deep_prefix,
    validate_citations,
)

USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "cached_tokens": 0,
    "finish_reason": "stop",
}


def make_hit(name: str, text: str, locator: str = "第 1 页") -> SearchHit:
    chunk = Chunk.create(
        source_path=name, doc_type="pdf", locator=locator, start=1, end=1, text=text
    )
    return SearchHit(chunk=chunk, score=0.03, dense_rank=1, sparse_rank=1)


class FakeRetriever:
    """按查询词返回预设命中，并记录收到的每个查询（用来断言补检只跑一轮）。"""

    def __init__(self, table: dict[str, list[SearchHit]] | None = None, fallback=None):
        self.table = table or {}
        self.fallback = fallback or []
        self.queries: list[str] = []

    def search(self, query, top_k=5, **kwargs):
        self.queries.append(query)
        hits = self.table.get(query)
        return list(hits if hits is not None else self.fallback)


def plan_reply(enough: bool = True, gaps=(), follows=()) -> str:
    return json.dumps(
        {"enough": enough, "gaps": list(gaps), "follow_up_queries": list(follows)},
        ensure_ascii=False,
    )


def answer_reply(**sections) -> str:
    payload = {key: [] for key in REVIEW_SECTIONS}
    payload.update(sections)
    payload["insufficient"] = False
    payload["note"] = ""
    return json.dumps(payload, ensure_ascii=False)


def _boom(*_args, **_kwargs):
    raise AssertionError("这条路径不该调用模型")


# ---- 思考模式判据（规划 2.4 要求"可执行化"）----


def test_should_use_thinking_matches_review_questions():
    assert should_use_thinking("刀具磨损监测有哪些主流方法？")
    assert should_use_thinking("数字孪生的研究现状如何")
    assert should_use_thinking("/deep 机床误差怎么补")
    # 事实型提问不该开思考模式（省钱）
    assert not should_use_thinking("RERANK_GAP 的初值是多少")


def test_strip_deep_prefix_only_removes_prefix():
    assert strip_deep_prefix("/deep 有哪些方法") == "有哪些方法"
    assert strip_deep_prefix("有哪些方法") == "有哪些方法"


# ---- 解析：宁缺毋滥 ----


def test_parse_plan_accepts_code_fence_and_caps_follow_ups():
    text = "```json\n" + plan_reply(False, follows=["a", "b", "c", "d"]) + "\n```"

    plan = parse_plan(text)

    assert plan == Plan(enough=False, follow_up_queries=("a", "b", "c"))


def test_parse_plan_rejects_non_json_and_bad_shape():
    assert parse_plan("我觉得够了") is None
    assert parse_plan('{"enough": "yes"}') is None  # enough 必须是布尔
    assert parse_plan('{"enough": true, "gaps": "x", "follow_up_queries": []}') is None


def test_parse_answer_requires_four_sections_and_citation_lists():
    text = answer_reply(
        研究现状=[{"point": "有若干方法", "citations": ["刀具/a.pdf:第 1 页"]}]
    )

    payload = parse_answer(text)

    assert payload is not None
    assert payload["研究现状"][0]["point"] == "有若干方法"
    assert payload["方法对比"] == []  # 空数组是合法的"这一段没内容"

    # 缺三段 → 整体失败（不猜）
    assert parse_answer(json.dumps({"研究现状": []})) is None
    # citations 必须是字符串数组
    assert parse_answer(answer_reply(结论=[{"point": "x", "citations": "y"}])) is None


# ---- 引用校验：三档分开记 ----


def test_validate_citations_splits_exact_partial_and_invented():
    hits = [make_hit("刀具/a.pdf", "内容", locator="第 4 页")]
    payload = {key: [] for key in REVIEW_SECTIONS}
    payload["研究现状"] = [
        {
            "point": "P1",
            "citations": ["刀具/a.pdf:第 4 页", "刀具/a.pdf", "编造/b.pdf:第 9 页"],
        },
        {"point": "P2", "citations": []},
    ]

    cleaned, exact, partial, invalid, uncited = validate_citations(payload, hits)

    assert (exact, partial) == (1, 1)
    assert invalid == ["编造/b.pdf:第 9 页"]
    assert uncited == 1
    # 编造的引用被摘出来，条目本身保留（"模型没给引用"是要报告的事实）
    assert cleaned["研究现状"][0]["citations"] == ["刀具/a.pdf:第 4 页", "刀具/a.pdf"]


def test_candidate_block_uses_traceable_labels():
    hit = make_hit("刀具/a.pdf", "正文", locator="第 4 页")

    assert citation_of(hit) == "刀具/a.pdf:第 4 页"
    block = build_candidates_block([hit])

    assert "文件：刀具/a.pdf" in block
    assert "位置：第 4 页" in block


# ---- 主流程 ----


def test_no_candidates_skips_model_calls():
    retriever = FakeRetriever()

    result = answer_review(
        "刀具磨损监测有哪些主流方法", retriever, plan_call=_boom, answer_call=_boom
    )

    assert result.calls == []
    assert result.degraded is False
    assert result.payload["insufficient"] is True
    assert "资料中" in result.payload["note"]


def test_multihop_uses_follow_up_queries_and_merges_candidates():
    primary = [make_hit("刀具/综述.pdf", "第一轮内容")]
    extra = [make_hit("刀具/别的.pdf", "第二轮内容", locator="第 2 页")]
    retriever = FakeRetriever(
        {
            "刀具磨损有哪些方法": primary,
            "刀具磨损 综述": extra,
            "声发射 监测": extra,
        }
    )
    seen: list[str] = []

    def plan_call(prompt, model=None):
        return (
            plan_reply(
                False, gaps=["缺声发射"], follows=["刀具磨损 综述", "声发射 监测"]
            ),
            dict(USAGE),
        )

    def answer_call(prompt, model=None):
        seen.append(prompt)
        return answer_reply(), dict(USAGE)

    result = answer_review(
        "刀具磨损有哪些方法", retriever, plan_call=plan_call, answer_call=answer_call
    )

    # 一次主检索 + 两次补检；两份补检命中同一片段 → 合并后去重
    assert retriever.queries == ["刀具磨损有哪些方法", "刀具磨损 综述", "声发射 监测"]
    assert [hit.chunk.source_path for hit in result.candidates] == [
        "刀具/综述.pdf",
        "刀具/别的.pdf",
    ]
    assert "刀具/别的.pdf" in seen[0]  # 补检回来的片段确实进了最终提示词
    assert len(result.calls) == 2
    assert result.degraded is False


def test_plan_failure_does_not_block_the_answer():
    retriever = FakeRetriever(fallback=[make_hit("刀具/综述.pdf", "内容")])

    def plan_call(prompt, model=None):
        raise RuntimeError("网络抖动")

    def answer_call(prompt, model=None):
        return (
            answer_reply(
                研究现状=[{"point": "有方法", "citations": ["刀具/综述.pdf:第 1 页"]}]
            ),
            dict(USAGE),
        )

    result = answer_review(
        "有哪些方法", retriever, plan_call=plan_call, answer_call=answer_call
    )

    assert result.degraded is False
    assert "盘点调用失败" in result.plan_reason
    assert result.citation_exact == 1
    assert len(retriever.queries) == 1  # 盘点失败 → 不做补检


def test_plan_garbage_is_reported_and_downgrades_to_single_round():
    retriever = FakeRetriever(fallback=[make_hit("刀具/综述.pdf", "内容")])

    def plan_call(prompt, model=None):
        return "我觉得够了", dict(USAGE)

    result = answer_review(
        "有哪些方法",
        retriever,
        plan_call=plan_call,
        answer_call=lambda prompt, model=None: (answer_reply(), dict(USAGE)),
    )

    assert result.plan is None
    assert "不是合法 json" in result.plan_reason
    assert result.degraded is False


def test_answer_truncation_is_reported_as_truncation():
    retriever = FakeRetriever(fallback=[make_hit("刀具/综述.pdf", "内容")])

    def answer_call(prompt, model=None):
        return "", dict(USAGE, finish_reason="length", completion_tokens=4096)

    result = answer_review(
        "有哪些方法", retriever, answer_call=answer_call, multihop=False
    )

    assert result.degraded is True
    assert result.payload is None
    assert "截断" in result.reason
    # 报告不能只给诊断，还要给"下一步动哪个旋钮"（首次真跑 3/3 截断后加的）
    assert "调大 --max-tokens" in result.reason


def test_validate_citations_dedupes_repeated_citation():
    """同一条引用写两遍只算一次（真跑 R21-1 的"电流传感器"那条就这么写的）。"""
    hits = [make_hit("刀具/a.pdf", "内容", locator="第 5 页")]
    payload = {key: [] for key in REVIEW_SECTIONS}
    payload["方法对比"] = [
        {"point": "P", "citations": ["刀具/a.pdf:第 5 页", "刀具/a.pdf:第 5 页"]}
    ]

    cleaned, exact, partial, invalid, uncited = validate_citations(payload, hits)

    assert cleaned["方法对比"][0]["citations"] == ["刀具/a.pdf:第 5 页"]
    assert exact == 1
    assert partial == 0  # 写的是完整的"文件:页"，不该落到"缺定位"那一档
    assert uncited == 0
    assert invalid == []


def test_plan_budget_has_room_above_observed_use():
    """盘点预算不能贴着实测值：真跑里有一题的盘点用了 2043 token（当时上限恰好 2048）。"""
    assert REVIEW_PLAN_MAX_TOKENS >= 4096


def test_answer_prompt_bounds_output_length():
    """预算再大也要有结构上的上限：提示词里的篇幅约束是"能写完"的第一道保险。

    第 21 周首次真跑 3/3 题被 4096 截断（四段式 JSON + 内部推理），
    除了调大预算，还要明确要求模型"宁可少写几条"。
    """
    assert "每段最多 4 条" in ANSWER_PROMPT
    assert "最多挂 2 个引用" in ANSWER_PROMPT


def test_answer_invalid_json_degrades_with_head_of_reply():
    retriever = FakeRetriever(fallback=[make_hit("刀具/综述.pdf", "内容")])

    result = answer_review(
        "有哪些方法",
        retriever,
        answer_call=lambda prompt, model=None: ("这不是 json", dict(USAGE)),
        multihop=False,
    )

    assert result.degraded is True
    assert "不是合法 json" in result.reason
    assert "这不是 json" in result.reason  # 回显回复开头，事后能查出发生了什么


def test_multihop_disabled_skips_plan_call():
    retriever = FakeRetriever(fallback=[make_hit("刀具/综述.pdf", "内容")])

    result = answer_review(
        "有哪些方法",
        retriever,
        plan_call=_boom,
        answer_call=lambda prompt, model=None: (answer_reply(), dict(USAGE)),
        multihop=False,
    )

    assert [call["stage"] for call in result.calls] == ["answer"]
    assert result.plan is None
    assert result.plan_reason == ""


def test_thinking_model_is_only_used_when_predicate_hits():
    retriever = FakeRetriever(fallback=[make_hit("刀具/综述.pdf", "内容")])
    seen: list[str | None] = []

    def plan_call(prompt, model=None):
        seen.append(model)
        return plan_reply(enough=True), dict(USAGE)

    def answer_call(prompt, model=None):
        seen.append(model)
        return answer_reply(), dict(USAGE)

    answer_review(
        "刀具磨损有哪些主流方法？",
        retriever,
        plan_call=plan_call,
        answer_call=answer_call,
        thinking_model="deepseek-reasoner",
    )
    assert seen == ["deepseek-reasoner", "deepseek-reasoner"]

    seen.clear()
    answer_review(
        "RERANK_GAP 的初值是多少",
        retriever,
        plan_call=plan_call,
        answer_call=answer_call,
        thinking_model="deepseek-reasoner",
    )
    assert seen == [None, None]  # 没命中判据就不切模型


# ---- 报告渲染（给人看的产物，也值得直接测）----


def test_render_includes_sections_citations_and_invented_list():
    hits = [make_hit("刀具/a.pdf", "内容", locator="第 4 页")]
    result = ReviewResult(
        question="有哪些方法",
        payload={
            **{key: [] for key in REVIEW_SECTIONS},
            "研究现状": [{"point": "P1", "citations": ["刀具/a.pdf:第 4 页"]}],
            "insufficient": True,
            "note": "资料中没有声发射相关的对比",
        },
        candidates=hits,
        calls=[dict(USAGE, stage="answer")],
        reason="ok",
        citation_exact=1,
        citation_invalid=["编造/b.pdf:第 9 页"],
    )

    report = _render(result, "data/index", 5)

    for key in REVIEW_SECTIONS:
        assert f"### {key}" in report
    assert "[刀具/a.pdf:第 4 页]" in report  # 引用带方括号，方便回查
    assert "资料中没有声发射相关的对比" in report  # 缺口声明必须出现
    assert "编造（候选里没有）：1" in report
    assert "刀具/a.pdf" in report  # 候选清单进报告，人工核对用
    assert "第 1 次（answer）" in report  # 每次调用的 usage 要能看到
