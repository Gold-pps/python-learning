"""回答级判据：从回答里抽引用、判是否拒答、算引用是否落在期望来源上。

**本模块不 import 任何第三方库、也不 import 项目内其他模块**，因为它有两个使用方：

- `eval_cases.py`（第 14 周起，Agent 级评估，跑在主 `.venv`）；
- `eval/rag_answer_eval.py`（第 19 周起，检索 + 一次回答，跑在 `.venv-rag`）。

后者那个环境里**没有 `openai-agents`**（台账 T18 那条边界），所以判据不能长在
`eval_cases.py` 里——那会让它顺着 `config.py` 把 `agents` 拖进来。
抽到无依赖的位置、两边都引用，与 `constants.py` 是同一种做法。

两条来自第 14 周 T10 的纪律：

1. **内容与格式分开判**：识不认得 `[文件名:行号]`（内容）是一回事，
   有没有守约定格式（`has_bracket_citation`）是另一回事。混在一起会把
   "格式漂移"误报成"没给出处"——那次 C1/D2 两道题就是这么被误判的；
2. **措辞不唯一，所以拒答用正则判**，不用精确匹配。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# 内容判据用宽松的：方括号与反引号两种写法都认（实测模型会漂移成反引号）
_BRACKET_CITATION_RE = re.compile(r"\[([^\[\]:]+?\.md)\s*:[^\[\]]*\]")
_BACKTICK_CITATION_RE = re.compile(r"`([^`\[\]:]+?\.md)\s*:[^`\[\]]*`")

# 拒答措辞无法穷举（第 14 周定的口径：must_match_any 用正则）
_REFUSAL_RE = re.compile(
    r"没有(找到|讲|提及|涉及|相关|收录|介绍|记录|办法|办法)"
    r"|未(找到|提及|涉及|收录|涵盖)"
    r"|无相关"
    r"|资料(里|中)(没有|未|不)"
    r"|笔记(里|中)(没有|未|不)"
    r"|无法(回答|确定|从资料)"
)


def norm_name(name: str) -> str:
    """文件名归一化：回答里可能写成「第 7 周笔记.md」，去掉空白再比。"""
    return re.sub(r"\s+", "", name)


def extract_cited_files(text: str) -> set[str]:
    """抽出被引用的文件名（归一化后）。方括号与反引号两种写法都认。

    比"文件名作为子串出现在回答里"严格得多：必须真的写出 `文件名:行号`；
    又比只认方括号宽松：不会把"格式漂移"误判成"没有引用"。
    """
    bracket = {norm_name(m.group(1)) for m in _BRACKET_CITATION_RE.finditer(text)}
    backtick = {norm_name(m.group(1)) for m in _BACKTICK_CITATION_RE.finditer(text)}
    return bracket | backtick


def has_bracket_citation(text: str) -> bool:
    """是否出现符合约定的 `[文件名:行号]`（用于单独统计格式合规率）。"""
    return _BRACKET_CITATION_RE.search(text) is not None


def is_refusal(text: str) -> bool:
    """回答是否构成"资料里没有"式的拒答。"""
    return _REFUSAL_RE.search(text) is not None


@dataclass(frozen=True)
class AnswerVerdict:
    """一条回答的判定结果。各字段独立，便于分开统计（内容 / 格式 / 拒答）。"""

    citations: frozenset[str]
    has_bracket: bool  # 格式合规：用了约定的 [文件名:行号]
    refusal: bool  # 是否拒答
    cited_expected: bool  # 有引用的题：引用**全部**落在期望来源里
    fabricated: bool  # 无答案题：竟然给出了引用（编造出处）

    @property
    def no_citation(self) -> bool:
        return not self.citations


def judge_answer(
    text: str, *, expect_files: Sequence[str] = (), negative: bool = False
) -> AnswerVerdict:
    """判一条回答。

    `negative=True` 表示这是"无答案"题：此时**给出引用就算编造**
    （第 14 周的判据从"禁止引用"改成这里的一条独立字段，不再与拒答混在一起）。
    """
    citations = frozenset(extract_cited_files(text))
    expected = {norm_name(f) for f in expect_files}
    cited_expected = (
        bool(citations) and citations.issubset(expected) if not negative else False
    )
    return AnswerVerdict(
        citations=citations,
        has_bracket=has_bracket_citation(text),
        refusal=is_refusal(text),
        cited_expected=cited_expected,
        fabricated=negative and bool(citations),
    )
