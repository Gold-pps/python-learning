"""LLM 重排的单元测试。

**不联网、不花 token**：模型调用通过 `call` 参数注入假函数。
离线能验的是"流程与判断"，不是"模型排得好不好"——后者要真跑（由使用者在自己时段跑）。

重点覆盖三件容易写错的事：
1. 触发条件（分差视角）到底按哪个窗口算；
2. 模型回复不合规时**必须降级**，而不是抛异常或静默返回错序；
3. usage 记账真的带回来了。
"""

from __future__ import annotations

import pytest
from rag.chunk import Chunk
from rag.rerank import (
    _usage_info,
    adjacent_gaps,
    build_prompt,
    parse_ranking,
    rerank,
    should_rerank,
)
from rag.retriever import SearchHit


def make_hit(name: str, score: float, text: str = "片段正文") -> SearchHit:
    chunk = Chunk.create(
        source_path=name, doc_type="md", locator="1.1", start=1, end=1, text=text
    )
    return SearchHit(chunk=chunk, score=score)


def close_hits(
    n: int = 4, base: float = 0.033, step: float = 0.0005
) -> list[SearchHit]:
    """分数挨得很近的一组命中（相邻差 0.0005 < gap 0.01 → 会触发重排）。"""
    return [make_hit(f"d{i}.md", base - i * step) for i in range(n)]


# ---- 触发条件 ----


def test_adjacent_gaps_reports_each_pair():
    hits = [make_hit("a.md", 0.9), make_hit("b.md", 0.5), make_hit("c.md", 0.1)]
    assert adjacent_gaps(hits) == pytest.approx([0.4, 0.4])


def test_should_rerank_triggers_when_close():
    hits = [make_hit("a.md", 0.0325), make_hit("b.md", 0.0321)]
    need, smallest = should_rerank(hits, 0.01)
    assert need is True
    assert smallest == pytest.approx(0.0004)


def test_should_rerank_skips_when_well_separated():
    hits = [make_hit("a.md", 0.9), make_hit("b.md", 0.5), make_hit("c.md", 0.1)]
    need, smallest = should_rerank(hits, 0.01)
    assert need is False
    assert smallest == pytest.approx(0.4)


def test_should_rerank_false_for_single_hit():
    assert should_rerank([make_hit("a.md", 0.5)], 0.01) == (False, None)


def test_should_rerank_only_looks_at_candidate_window():
    """窗口外再挤也不该触发：那些候选本来也进不了最终 top_n。"""
    hits = [
        make_hit("a.md", 0.9),
        make_hit("b.md", 0.5),
        make_hit("c.md", 0.4),
        make_hit("d.md", 0.399),
    ]
    assert should_rerank(hits, 0.01, candidates=2)[0] is False
    assert should_rerank(hits, 0.01, candidates=4)[0] is True


# ---- 解析模型回复 ----


def test_parse_ranking_maps_to_zero_based_indices():
    assert parse_ranking('{"ranking": [3, 1]}', 5) == [2, 0]


def test_parse_ranking_accepts_code_fence():
    assert parse_ranking('```json\n{"ranking": [2]}\n```', 5) == [1]


def test_parse_ranking_dedupes_repeated_ids():
    assert parse_ranking('{"ranking": [2, 2, 1]}', 5) == [1, 0]


def test_parse_ranking_allows_empty_ranking():
    assert parse_ranking('{"ranking": []}', 5) == []


@pytest.mark.parametrize(
    "reply",
    [
        "我觉得第 2 条最相关",  # 不是 JSON
        '{"top": [1]}',  # 字段名不对
        '{"ranking": "1,2"}',  # 值不是列表
        '{"ranking": [0]}',  # 编号从 1 起，0 非法
        '{"ranking": [6]}',  # 越界
        '{"ranking": [true]}',  # bool 不是编号
        '{"ranking": [1.5]}',  # 不是整数
        "[1, 2]",  # 顶层是列表而不是对象
        "",  # 空回复
    ],
)
def test_parse_ranking_rejects_anything_nonconforming(reply):
    assert parse_ranking(reply, 5) is None


# ---- 主流程 ----


def test_rerank_skips_llm_when_scores_are_separated():
    hits = [make_hit("a.md", 0.9), make_hit("b.md", 0.5), make_hit("c.md", 0.1)]
    calls: list[str] = []

    result = rerank("问题", hits, top_n=2, gap=0.01, call=lambda p: calls.append(p))

    assert calls == []  # 关键：没触发就不该花这次钱
    assert result.triggered is False
    assert result.degraded is False
    assert [h.chunk.source_path for h in result.hits] == ["a.md", "b.md"]


def test_rerank_reorders_and_reports_usage():
    hits = close_hits(4)
    usage = {"prompt_tokens": 1200, "completion_tokens": 20, "cached_tokens": 0}

    result = rerank("问题", hits, call=lambda p: ('{"ranking": [3, 1]}', usage))

    assert result.triggered is True
    assert result.degraded is False
    assert result.usage == usage
    # 只返回被点名的（第 3 条 → 第 1 条）
    assert [h.chunk.source_path for h in result.hits] == ["d2.md", "d0.md"]
    assert "2 条" in result.reason


def test_rerank_truncates_to_top_n():
    result = rerank(
        "问题", close_hits(4), top_n=2, call=lambda p: ('{"ranking": [3, 1]}', {})
    )

    assert [h.chunk.source_path for h in result.hits] == ["d2.md", "d0.md"]


def test_rerank_drops_unmentioned_candidates():
    """未点名 = 模型判为不相关，**不再补回来**。

    第一版会把它们按原顺序补在后面凑满 top_n，理由是"怕模型少答导致召回变少"。
    真跑一次才发现这跟提示词自相矛盾：规则 2 说了"无关的不要放进来"，
    代码又把模型明确排除掉的片段（那次是一个"留给你填"的空白小节）塞回了最终结果。
    调用方本来就拿得到完整候选池，不需要这里替它兜底。
    """
    result = rerank(
        "问题", close_hits(4), call=lambda p: ('{"ranking": [2, 2, 1]}', {})
    )

    # 重复编号去重后只剩 2 个 → 就返回 2 条，不再补 d2/d3
    assert [h.chunk.source_path for h in result.hits] == ["d1.md", "d0.md"]
    assert "未点名的 2 条" in result.reason  # 丢弃数量写进原因里，便于观测


def test_rerank_empty_ranking_means_nothing_relevant():
    """`{"ranking": []}` 是**合法结论**（提示词规则 5），不是解析失败。

    第一版用 `if not order` 判断，把"都不相关"也归进降级分支，
    于是无答案题上拿不到这个信号——它是第 19 周"无答案拒答率"的关键输入。
    """
    result = rerank("问题", close_hits(4), call=lambda p: ('{"ranking": []}', {}))

    assert result.triggered is True
    assert result.degraded is False
    assert result.hits == []
    assert "0 条" in result.reason


def test_rerank_degrades_on_invalid_reply():
    usage = {"prompt_tokens": 900, "completion_tokens": 8, "cached_tokens": 0}

    result = rerank(
        "问题", close_hits(4), call=lambda p: ("我认为第 3 条最相关", usage)
    )

    assert result.triggered is True
    assert result.degraded is True
    assert "json" in result.reason
    assert "我认为第 3 条最相关" in result.reason  # 回复开头回显，FAIL 后能复查
    assert result.usage == usage  # 调用是成功的，钱花了就要记账
    assert [h.chunk.source_path for h in result.hits] == [
        "d0.md",
        "d1.md",
        "d2.md",
        "d3.md",
    ]


def test_rerank_reports_truncation_and_shows_reply_head():
    """真跑第一遍的现场：`completion_tokens` 正好等于 `max_tokens`，JSON 在中途被切断。

    降级本身是对的（规划就是要求失败降级），但**原因必须能看出"是被截断"**——
    否则只能靠重跑一次才知道哪里错了，而每次重跑都要花钱。
    """
    usage = {
        "prompt_tokens": 3946,
        "completion_tokens": 256,
        "cached_tokens": 3712,
        "finish_reason": "length",
    }
    truncated_reply = '{"ranking": [1, 7, 3, 12, 5, 2, 9, 4, 8, 11, 6, 10, 13, 14'

    result = rerank("问题", close_hits(4), call=lambda p: (truncated_reply, usage))

    assert result.degraded is True
    assert "截断" in result.reason
    assert "max_tokens" in result.reason
    assert "completion_tokens=256" in result.reason
    assert '{"ranking"' in result.reason
    assert result.usage == usage


def test_rerank_degrades_when_llm_raises():
    def boom(_prompt: str):
        raise RuntimeError("网络断了")

    result = rerank("问题", close_hits(4), call=boom)

    assert result.triggered is True
    assert result.degraded is True
    assert "RuntimeError" in result.reason
    assert result.usage is None
    assert [h.chunk.source_path for h in result.hits] == [
        "d0.md",
        "d1.md",
        "d2.md",
        "d3.md",
    ]


def test_rerank_with_no_candidates_does_not_call_llm():
    calls: list[str] = []
    result = rerank("问题", [], call=lambda p: calls.append(p))

    assert result.hits == []
    assert result.triggered is False
    assert calls == []


# ---- 提示词 ----


def test_build_prompt_numbers_candidates_and_truncates_long_text():
    hits = [make_hit("长.md", 0.03, "长" * 700), make_hit("短.md", 0.02, "短文本")]

    prompt = build_prompt("为什么小批量用 ONNX？", hits, top_n=3)

    assert "[1]" in prompt and "[2]" in prompt
    assert "为什么小批量用 ONNX？" in prompt
    assert "长" * 700 not in prompt
    assert "…" in prompt
    # json_object 模式要求提示词里出现 json 字样，否则接口会直接报错
    assert "json" in prompt


# ---- 记账 ----


class _Details:
    cached_tokens = 7


class _Usage:
    prompt_tokens = 100
    completion_tokens = 5
    prompt_tokens_details = _Details()


class _UsageWithoutDetails:
    prompt_tokens = 1
    completion_tokens = 2
    prompt_tokens_details = None
    prompt_cache_hit_tokens = 3


def test_usage_info_reads_cached_tokens_from_details():
    assert _usage_info(_Usage()) == {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "cached_tokens": 7,
        "finish_reason": None,
    }


def test_usage_info_falls_back_to_deepseek_field():
    assert _usage_info(_UsageWithoutDetails())["cached_tokens"] == 3


def test_usage_info_records_finish_reason():
    """`finish_reason == "length"` 是"被截断"的唯一证据，必须带回来。"""
    assert _usage_info(_Usage(), "length")["finish_reason"] == "length"
