"""模型评估的"测量工具"自检（不联网、不花 token）。

第 12 周的结论是"评估脚本本身也要评估"：断言写错时，评测结果会给出**看起来合理**的分数。
所以这里测的不是模型，而是 `check_model_output` 的判据本身。
"""
import pytest

import config
from eval_cases import MODEL_CASES, check_model_output, extract_cited_files

FOUR_CATEGORIES = {"有答案", "无答案", "跨文档", "需要多跳"}


def test_extract_citations():
    """引用必须写成 `[文件名:行号]` 才算数；文件名里的空格要归一化。"""
    text = "依据 [第7周笔记.md:12-40] 与 [第 9 周笔记.md:3]。另有 (括号) 不算引用。"
    assert extract_cited_files(text) == {"第7周笔记.md", "第9周笔记.md"}


def test_required_citation_must_appear():
    case = {"must_cite_files": ["工程化改造记录.md"]}
    assert check_model_output("答案见 [第13周笔记.md:1-5]。", case) != []
    assert check_model_output("答案见 [工程化改造记录.md:9-12]。", case) == []


def test_min_distinct_files():
    case = {"min_distinct_files": 3}
    assert check_model_output("见 [a.md:1] 与 [b.md:2]。", case) != []
    assert check_model_output("见 [a.md:1]、[b.md:2]、[c.md:3]。", case) == []


def test_must_match_any_accepts_various_refusals():
    """"无答案"类的措辞无法穷举，所以用正则而不是关键词表。

    真实教训：B2 跑出来的回答写的是"笔记里**没有讲**怎么用 PyQt"，
    而当时的关键词表只有"没有找到/笔记中没有/…"，于是判成 FAIL——拒答其实是对的。
    """
    case = {"must_match_any": [r"没有(找到|讲|提及|涉及|相关)|未(找到|提及)"]}
    assert check_model_output("笔记里没有讲怎么用 PyQt 做桌面界面。", case) == []
    assert check_model_output("笔记中没有找到相关内容。", case) == []
    assert check_model_output("笔记里介绍了 PyQt 的基本用法。", case) != []


def test_keyword_matching_normalizes_markdown_and_whitespace():
    """`**输入护栏**`、`输入 护栏` 都应算命中：匹配前要去掉 Markdown 符号与空白。"""
    case = {"must_contain": ["输入护栏"]}
    assert check_model_output("**输入护栏** 会在处理前运行。", case) == []
    assert check_model_output("输入 护栏 会在处理前运行。", case) == []
    assert check_model_output("只提到输出护栏。", case) != []


def test_must_contain_any_needs_one_alternative():
    """拒答措辞不唯一，所以"至少出现其一"才是正确口径。"""
    case = {"must_contain_any": ["没有找到", "未找到"]}
    assert check_model_output("笔记中**没有找到**相关内容。", case) == []
    assert check_model_output("笔记里讲了不少相关内容。", case) != []


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda c: c["name"])
def test_every_case_has_an_assertion(case):
    """每条用例至少要有一条判据，否则它会永远 PASS——这种"假绿灯"比 FAIL 更危险。"""
    assertion_keys = {
        "must_contain", "must_contain_any", "must_cite_files",
        "must_cite_any_of", "min_distinct_files", "must_match_any",
    }
    assert assertion_keys & set(case)


def test_case_set_shape():
    """10 条、四类齐全、每类至少 2 条（第 19 周要在这个基础上扩到 ≥20 条、每类 ≥5 条）。"""
    assert len(MODEL_CASES) == 10

    counts: dict[str, int] = {}
    for case in MODEL_CASES:
        assert case["category"] in FOUR_CATEGORIES
        counts[case["category"]] = counts.get(case["category"], 0) + 1

    assert set(counts) == FOUR_CATEGORIES
    assert all(n >= 2 for n in counts.values()), counts


def test_runner_survives_a_case_that_blows_up(monkeypatch):
    """单条用例抛异常，harness 必须记 FAIL 后继续，而不是崩掉整份评估。

    修复前的现场：C1 撞上 SDK 默认 10 轮上限抛 `MaxTurnsExceeded`，
    整个进程直接退出，后面 5 条用例（C2/D1/D2）连跑的机会都没有。
    """
    import asyncio

    from agents.exceptions import MaxTurnsExceeded

    import eval_cases

    class ExplodingRunner:
        @staticmethod
        async def run(*args, **kwargs):
            raise MaxTurnsExceeded("模拟超轮数")

    monkeypatch.setattr(eval_cases, "Runner", ExplodingRunner)
    passed, total = asyncio.run(eval_cases.run_model_cases(only="A1"))
    assert (passed, total) == (0, 1)


def test_negative_sample_keywords_do_not_leak_into_notes():
    """"无答案"题的关键词一旦出现在资料库里，这道题就失效了——等于把答案卡发下去。

    第 14 周实测踩过这个坑：关键词和"已核实 0 命中"的结论被写进 `工程化改造记录.md`
    （该文件位于笔记根目录内、属于可检索范围），模型于是直接引用那句话来"证明"
    资料里没有相关内容——那一轮的结论不可信。

    注意：这个测试**故意读真实笔记目录**（不请求 `notes_root` fixture），因为要检查的
    正是"真实资料库里有没有泄漏"。
    """
    notes_root = config.NOTES_ROOT
    corpus = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in notes_root.rglob("*.md")
    )
    leaked = [
        keyword
        for case in MODEL_CASES
        for keyword in case.get("negative_keywords", [])
        if keyword in corpus
    ]
    assert not leaked, f"负样本关键词已出现在资料库中，题面失效：{leaked}"
