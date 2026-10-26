"""文档里的"快照数字"守卫（第四阶段 H1）。

台账 **T56** 咬了三次：`657/2501`、`523/3073`、`574 条`——都是"语料或使用变了、文档没变"，
而且**没有任何测试会发现**。这两条测试把教训变成机制：

1. **不许写"累计 N 条引用"**：那种累加值没法一眼重算，过期了也没人察觉（该改成机制性断言）；
2. **语料规模必须与索引一致**：README 写"N 篇…M 片段"，就跑一次真实索引的 meta 来对。
   第二条**只在本地有索引时跑**（`skipif`）——CI 上没有 `data/`，不能让它变红（T51/T54 的教训）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
README = REPO / "README.md"
META = Path(__file__).resolve().parents[1] / "data" / "index" / "meta.json"

_BARE_COUNTER = re.compile(r"累计\s*\*{0,2}\s*\d+\s*条")


def test_readme_does_not_carry_a_bare_accumulated_counter():
    """ "累计 N 条引用"这类累加值不许写死（T56）。

    要写就写成机制性断言（"逐条对账、0 编造"），或者给命令让人现场算。
    """
    text = README.read_text(encoding="utf-8")

    found = _BARE_COUNTER.findall(text)
    assert not found, (
        f"README 里出现了裸的累加值 {found}——改成机制性断言或给命令（T56）"
    )


def test_readme_corpus_size_matches_the_real_index():
    """README 的"N 篇 / M 片段"必须与本地索引的 meta 一致（改语料时最容易忘的一处）。"""
    if not META.exists():
        pytest.skip("本地没有索引（CI 就是这样），跳过——这条只在有数据时有意义")

    meta = json.loads(META.read_text(encoding="utf-8"))
    files = meta.get("n_files")
    chunks = meta.get("n_chunks")
    assert files and chunks, f"meta.json 里没有 n_files/n_chunks：{sorted(meta)}"

    text = README.read_text(encoding="utf-8")
    assert f"{files} 篇" in text, f"README 没写「{files} 篇」——语料变了没同步？"
    assert f"{chunks} 片段" in text, f"README 没写「{chunks} 片段」——语料变了没同步？"
