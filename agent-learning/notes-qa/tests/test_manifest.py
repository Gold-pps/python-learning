"""语料清单生成（`rag/manifest.py`）的离线测试。

要证明三件事：
1. **每一篇都进表**，且带 sha256 前 16 位（否则"按指纹认丢失文件"这件事不成立）；
2. 表头写的是**当前索引**的篇数/片段数（不是写死的旧值 —— 台账 T56）；
3. 空语料**报错**而不是生成一份空清单（空清单比没有更危险：看着像有备份）。
"""

from __future__ import annotations

import re

import pytest
from rag import manifest


def _corpus(tmp_path, count: int = 3):
    corpus = tmp_path / "corpus"
    (corpus / "刀具磨损监测").mkdir(parents=True)
    for index in range(count):
        (corpus / "刀具磨损监测" / f"篇{index}.pdf").write_bytes(
            b"%PDF-1.4 " + str(index).encode()
        )
    return corpus


def _index(tmp_path, files: int = 3, chunks: int = 120):
    index = tmp_path / "index"
    index.mkdir()
    (index / "meta.json").write_text(
        f'{{"n_files": {files}, "n_chunks": {chunks}}}', encoding="utf-8"
    )
    return index


def test_every_file_is_listed_with_a_sha_prefix(tmp_path):
    """每篇都要有一行，且 sha256 前 16 位是 16 个十六进制字符。"""
    text = manifest.build(_corpus(tmp_path, 3), index=_index(tmp_path))

    rows = [
        line
        for line in text.splitlines()
        if line.startswith(("| 1 |", "| 2 |"))
    ]
    assert len(rows) == 2, text[:400]
    assert re.search(r"`[0-9a-f]{16}`", rows[0]), rows[0]
    assert "共 **3 篇" in text


def test_header_uses_the_real_index_numbers(tmp_path):
    """表头写当前索引（120 片段），不是模板里的旧值。"""
    text = manifest.build(
        _corpus(tmp_path, 3), index=_index(tmp_path, files=3, chunks=120)
    )

    assert "索引侧：3 篇 / 120 片段" in text


def test_empty_corpus_raises_instead_of_writing_an_empty_list(tmp_path):
    """空语料必须报错——一份空清单会让人以为"已经备份过了"。"""
    empty = tmp_path / "corpus"
    empty.mkdir()

    with pytest.raises(FileNotFoundError):
        manifest.build(empty, index=None)
