"""Index 构建与增量跳过的单元测试。

用 FakeEmbedder 代替真实模型：它只实现 encode() 与 dim，
不依赖 torch，所以在主 .venv 里就能跑（延续第 17 周"主环境不污染"的设计）。
"""

from __future__ import annotations

import json
from pathlib import Path

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


class OtherBackendEmbedder(FakeEmbedder):
    backend = "fake-other"


def test_rebuild_re_embeds_all_when_backend_changes(corpus_and_index):
    """后端换了必须整体重算。

    增量跳过只按内容 hash 判断，如果不看后端，旧向量会被原样搬过来，
    同时 meta.json 的 model 被改写成新后端——数据没变、标签变了。
    """
    corpus, index = corpus_and_index
    _write(corpus, "a.md", "# T\n" + "这是一句话。" * 60 + "\n")
    build(corpus, index, FakeEmbedder(), verbose=False)

    r2 = build(corpus, index, OtherBackendEmbedder(), verbose=False)

    assert r2["chunks_new"] == r2["chunks_total"] > 0
    meta = json.loads((index / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "fake-other"


def test_rebuild_re_embeds_all_when_meta_missing(corpus_and_index):
    """meta.json 丢了就不再信任旧索引（否则会拿一份说不清来源的向量继续用）。"""
    corpus, index = corpus_and_index
    _write(corpus, "a.md", "# T\n" + "这是一句话。" * 60 + "\n")
    build(corpus, index, FakeEmbedder(), verbose=False)

    (index / "meta.json").unlink()
    r2 = build(corpus, index, FakeEmbedder(), verbose=False)

    assert r2["chunks_new"] == r2["chunks_total"] > 0


def test_rebuild_keeps_vectors_aligned_when_chunks_have_duplicate_text(
    corpus_and_index,
):
    """增量重建后，**每条片段必须仍与自己的向量对齐**。

    触发条件：语料里有两条内容完全相同的片段（hash 相同）。旧实现在重建时用
    `enumerate(旧chunk字典.values())` 生成 "hash → 行号" 表——字典按 hash 去重后，
    它的下标**不再等于 vectors.npz 里的真实行号**，于是重复片段之后的每一行都拿到
    了邻居的向量。后果是**静默的**：向量整体只错位一两行，检索看起来仍然正常，
    只是"永远差一点"（第 19 周实测：稠密通道能找对文件、却几乎找不到对的那一段）。
    """
    corpus, index = corpus_and_index
    same = "# T\n" + "这是一句话。" * 60 + "\n"
    _write(corpus, "a.md", same)
    _write(corpus, "b.md", same)  # 与 a.md 内容相同 → 片段文本相同 → hash 相同
    _write(corpus, "c.md", "# U\n" + "那是另一句。" * 60 + "\n")  # 放在重复项之后

    build(corpus, index, FakeEmbedder(), verbose=False)
    r2 = build(corpus, index, FakeEmbedder(), verbose=False)
    assert r2["chunks_new"] == 0  # 第二次是纯增量，正是出问题的那条路径

    chunks = [
        Chunk.from_json(line)
        for line in (index / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors = np.load(index / "vectors.npz")["vectors"]
    embedder = FakeEmbedder()
    assert len(chunks) >= 3
    for i, chunk in enumerate(chunks):
        expected = embedder.encode([chunk.text])[0]
        assert np.allclose(vectors[i], expected, atol=1e-6), (
            f"第 {i} 行的向量与自己的片段不对齐（{chunk.source_path}）"
        )


def test_fastembed_cache_dir_is_outside_working_directory():
    """fastembed 的模型缓存默认走 `tempfile.gettempdir()`，而 Windows 上它可能
    **静默回退到当前工作目录**——实测 90.8 MB 的 ONNX 模型落进了
    `notes-qa/fastembed_cache/`（与 jieba 词典缓存同一个根因，见台账 T20）。
    """
    from rag.embedder import CACHE_DIR

    assert CACHE_DIR.parent.name == "data"
    assert CACHE_DIR != Path.cwd()
    assert not Path("fastembed_cache").exists()
