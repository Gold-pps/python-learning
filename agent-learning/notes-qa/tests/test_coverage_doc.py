"""`使用说明.md` 与 `rag/coverage.py` 的一致性测试（H4-1 的最后一条）。

为什么值得有：能力边界这种东西**最怕两份**——代码里 7 个子主题、文档里写 5 个，
或者文档说"不覆盖 NC 代码"而代码早就加了。这种不一致没有任何运行时症状，
只有"读者按文档理解、然后被现实打脸"。T29 那次（文档承诺与实现不一致）就是这么来的。

这里只钉**可机械对齐**的部分：子主题清单 + "不覆盖"这件事必须写着。
文风、措辞不归测试管。
"""

from __future__ import annotations

from pathlib import Path

from rag.coverage import NOT_COVERED, SUBTOPICS

DOC = Path(__file__).resolve().parents[1] / "web" / "使用说明.md"


def _normalized(text: str) -> str:
    """去掉所有空白再比：文档里写「AI 辅助工艺编制」、代码里写「AI辅助工艺编制」都算一致。"""
    return "".join(text.split())


def test_usage_doc_lists_every_subtopic():
    """代码里能答的子主题，文档里必须一个不少。"""
    doc = _normalized(DOC.read_text(encoding="utf-8"))

    missing = [name for name in SUBTOPICS if _normalized(name) not in doc]
    assert not missing, f"使用说明.md 少了这些子主题：{missing}"


def test_usage_doc_still_states_what_is_not_covered():
    """文档必须保留"不覆盖/答不了"的口径——只写能答什么，等于承诺万能。"""
    text = DOC.read_text(encoding="utf-8")

    assert "不覆盖" in text or "不能答" in text, "边界声明不见了"
    assert NOT_COVERED, "代码里的不覆盖清单不能为空"


def test_usage_doc_does_not_hardcode_a_stale_corpus_size():
    """文档里出现"2501 片段"这种**当前语料**的旧数字就红——历史记录该标注年代而不是裸写。

    现在实际是 3073 片段；如果哪天又补资料，这条会再次提醒去更新。
    """
    text = DOC.read_text(encoding="utf-8")

    assert "2501 片段" not in text, "使用说明里还挂着旧语料规模"
