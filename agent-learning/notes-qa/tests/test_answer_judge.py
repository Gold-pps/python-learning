"""回答级判据的单元测试（离线）。

这套判据决定 G2 的"引用准确率 / 无答案拒答率"两个数字，写错了就是"测量工具坏了
却看起来正常"（第 14 周 T10 的教训）——所以它自己必须有回归测试。
"""

from __future__ import annotations

import pytest
from eval.answer_judge import (
    extract_cited_files,
    has_bracket_citation,
    is_refusal,
    judge_answer,
)
from eval.rag_answer_eval import is_invalid_answer


def test_extract_cited_files_accepts_both_styles():
    text = "见 [第17周笔记.md:40-52]，另见 `工程化改造记录.md:158-166`。"
    assert extract_cited_files(text) == {"第17周笔记.md", "工程化改造记录.md"}


def test_extract_cited_files_normalizes_spaces():
    assert extract_cited_files("见 [第 17 周笔记.md:40-52]") == {"第17周笔记.md"}


def test_format_compliance_is_separate_from_content():
    """反引号写法：**内容认得**（有出处）但**格式不合规**。

    第 14 周 C1/D2 两道题就是被这两件事混在一起判错的（T10）。
    """
    text = "见 `第17周笔记.md:40-52`。"

    assert extract_cited_files(text) == {"第17周笔记.md"}
    assert has_bracket_citation(text) is False


def test_bracket_citation_detected():
    assert has_bracket_citation("见 [第17周笔记.md:40-52]。") is True


@pytest.mark.parametrize(
    "text",
    [
        "资料里没有相关内容。",
        "笔记中没有提及这一点。",
        "未找到相关资料。",
        "无相关记录。",
    ],
)
def test_is_refusal_recognizes_common_phrasings(text):
    """拒答措辞无法穷举，所以用正则（第 14 周定的口径）。"""
    assert is_refusal(text) is True


def test_is_refusal_false_for_normal_answer():
    assert is_refusal("增量重建耗时 0.08 秒 [第17周笔记.md:68-70]。") is False


def test_judge_answer_flags_fabricated_citation_on_negative_case():
    """无答案题只要给出引用就算编造——这是"拒答率"之外要单独统计的一条。"""
    fabricated = judge_answer(
        "据资料，需要先配置集群 [第17周笔记.md:1-2]", negative=True
    )
    assert fabricated.fabricated is True

    clean = judge_answer("资料里没有相关内容。", negative=True)
    assert clean.fabricated is False
    assert clean.refusal is True
    assert clean.no_citation is True


def test_is_invalid_answer_separates_empty_and_truncated_from_real_answers():
    """空回复 / 撞上限的调用**不能算进指标**。

    第 19 周第一版把 10 条空回复（模型的内部思考吃光了 `max_tokens`）算成"零引用"，
    于是"引用准确率 4/20、零引用率 12/20"这两个数字全是假的（见台账 T30）。
    """
    assert is_invalid_answer("", {"finish_reason": "stop"}) is True
    assert is_invalid_answer("   \n", None) is True
    assert is_invalid_answer("有内容的回答", {"finish_reason": "length"}) is True
    assert is_invalid_answer("有内容的回答", {"finish_reason": "stop"}) is False
    assert is_invalid_answer("有内容的回答", None) is False


def test_judge_answer_requires_every_citation_to_be_expected():
    text = "见 [a.md:1-2] 与 [b.md:3-4]"

    assert judge_answer(text, expect_files=["a.md", "b.md"]).cited_expected is True
    # 引了一个期望之外的文件 → 不算准确（哪怕另一个是对的）
    assert judge_answer(text, expect_files=["a.md"]).cited_expected is False
    # 完全没有引用 → 也不算准确
    assert (
        judge_answer("没有出处的一段话", expect_files=["a.md"]).cited_expected is False
    )
