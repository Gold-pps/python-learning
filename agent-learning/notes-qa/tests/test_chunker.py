"""Chunker 的单元测试。

判据有意写得"结构性"而不是"精确值"——切分粒度是会随第 19 周评测调的参数，
钉死具体字符数会让调参时到处误报（这是第 14 周 T13 的教训：钉文档承诺值，
不钉明确待调的初值）。
"""
from __future__ import annotations

from rag.chunker import MAX_CHARS, chunk_blocks
from rag.parsers import parse_md


def test_chunk_blocks_long_text_under_hard_bound(tmp_path):
    p = tmp_path / "t.md"
    p.write_text("# T\n" + "这是一句话。" * 100 + "\n\n短段落\n", encoding="utf-8")
    blocks = parse_md(p)
    chunks = chunk_blocks(blocks, source_path="t.md", doc_type="md")

    assert len(chunks) >= 1
    assert all(c.text.strip() for c in chunks)
    assert all(len(c.text) <= int(MAX_CHARS * 1.5) for c in chunks)

    combined = "".join(c.text for c in chunks)
    assert "这是一句话。" in combined
    assert "短段落" in combined


def test_chunk_blocks_short_doc_is_single_chunk(tmp_path):
    p = tmp_path / "t.md"
    p.write_text("# T\n一句话。\n", encoding="utf-8")
    blocks = parse_md(p)
    chunks = chunk_blocks(blocks, source_path="t.md", doc_type="md")

    assert len(chunks) == 1
    assert chunks[0].text == "一句话。"
    assert chunks[0].doc_type == "md"
    assert chunks[0].locator == "T"
    assert chunks[0].source_path == "t.md"


def test_chunk_blocks_multiple_chunks_have_unique_ids(tmp_path):
    p = tmp_path / "t.md"
    p.write_text("# T\n" + "这是一句话。" * 300, encoding="utf-8")
    blocks = parse_md(p)
    chunks = chunk_blocks(blocks, source_path="t.md", doc_type="md")

    assert len(chunks) >= 2
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_blocks_empty_blocks_return_empty_list():
    assert chunk_blocks([], source_path="t.md", doc_type="md") == []