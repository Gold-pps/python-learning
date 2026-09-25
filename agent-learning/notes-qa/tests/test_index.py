"""Index 构建与增量跳过的单元测试。

用 FakeEmbedder 代替真实模型：它只实现 encode() 与 dim，
不依赖 torch，所以在主 .venv 里就能跑（延续第 17 周"主环境不污染"的设计）。
"""

from __future__ import annotations

import numpy as np
import pytest
from rag.chunk import Chunk
from rag.index import build


class FakeEmbedder:
    """确定性假编码器：相同文本 -> 相同向量；不同文本 -> 不同向量。"""

    backend = "fake"
    dim = 8

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = []
        for t in texts:
            h = abs(hash(t)) % 1000
            rng = np.random.default_rng(h)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= np.linalg.norm(v)
            rows.append(v)
        return np.stack(rows)


@pytest.fixture
def corpus_and_index(tmp_path):
    corpus = tmp_path / "corpus"
    index = tmp_path / "index"
    corpus.mkdir()
    return corpus, index


def _write(corpus, name: str, text: str) -> None:
    (corpus / name).write_text(text, encoding="utf-8", newline="\n")


def test_build_empty_corpus_produces_empty_index(corpus_and_index):
    corpus, index = corpus_and_index
    r = build(corpus, index, FakeEmbedder(), verbose=False)
    assert r["files"] == 0
    assert r["chunks_total"] == 0
    assert (index / "chunks.jsonl").exists()
    assert (index / "vectors.npz").exists()


def test_build_writes_chunks_and_vectors_in_same_order(corpus_and_index):
    corpus, index = corpus_and_index
    _write(corpus, "a.md", "# T\n" + "这是一句话。" * 60 + "\n")
    _write(corpus, "b.md", "# U\n" + "那是另一句。" * 60 + "\n")
    build(corpus, index, FakeEmbedder(), verbose=False)

    chunks = [
        Chunk.from_json(line)
        for line in (index / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors = np.load(index / "vectors.npz")["vectors"]
    assert len(chunks) == vectors.shape[0]
    assert vectors.shape[1] == FakeEmbedder.dim


def test_rebuild_skips_all_when_corpus_unchanged(corpus_and_index):
    corpus, index = corpus_and_index
    _write(corpus, "a.md", "# T\n" + "这是一句话。" * 60 + "\n")

    r1 = build(corpus, index, FakeEmbedder(), verbose=False)
    r2 = build(corpus, index, FakeEmbedder(), verbose=False)

    assert r1["chunks_new"] > 0
    assert r2["chunks_new"] == 0
    assert r1["chunks_total"] == r2["chunks_total"]


def test_rebuild_only_embeds_changed_files(corpus_and_index):
    corpus, index = corpus_and_index
    _write(corpus, "a.md", "# T\n" + "这是一句话。" * 60 + "\n")
    build(corpus, index, FakeEmbedder(), verbose=False)

    # 新增一个文件
    _write(corpus, "b.md", "# U\n" + "那是另一句。" * 60 + "\n")
    r2 = build(corpus, index, FakeEmbedder(), verbose=False)

    assert r2["chunks_new"] > 0
    assert r2["chunks_new"] < r2["chunks_total"]


def test_build_records_failures_without_aborting(corpus_and_index):
    corpus, index = corpus_and_index
    _write(corpus, "good.md", "# T\n有效内容。\n")
    (corpus / "bad.pdf").write_bytes(b"not a real pdf")

    r = build(corpus, index, FakeEmbedder(), verbose=False)
    assert r["files"] == 2
    # bad.pdf 解析失败，但 good.md 仍进索引
    assert r["chunks_total"] >= 1


def test_build_skips_hidden_and_unsupported(corpus_and_index):
    corpus, index = corpus_and_index
    _write(corpus, "ok.md", "# T\n内容。\n")
    _write(corpus, ".hidden.md", "# T\n不该被读到。\n")
    _write(corpus, "notes.csv", "a,b\n")

    r = build(corpus, index, FakeEmbedder(), verbose=False)
    assert r["files"] == 1
