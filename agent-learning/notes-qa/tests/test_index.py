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


def test_build_force_full_re_encodes_everything(corpus_and_index):
    """`--all` = 一键重建：即使索引已经是最新的，也要整体重算。

    第 20 周的主命令靠它（语料大改之后要一份"从零构建"的干净结果），
    有了它就不必再手工删 `data/index/*`（第 19 周 T29 就是那么重建的）。
    """
    corpus, index = corpus_and_index
    _write(corpus, "a.md", "# T\n" + "这是一句话。" * 60 + "\n")
    build(corpus, index, FakeEmbedder(), verbose=False)

    r2 = build(corpus, index, FakeEmbedder(), verbose=False, force_full=True)

    assert r2["chunks_new"] == r2["chunks_total"] > 0


def test_build_returns_by_suffix_and_failures(corpus_and_index):
    """建库报告要用到这两项：按格式的文件数、以及**可读的失败清单**。"""
    corpus, index = corpus_and_index
    _write(corpus, "a.md", "# T\n有效内容。\n")
    _write(corpus, "b.txt", "纯文本内容。\n")
    (corpus / "bad.pdf").write_bytes(b"not a real pdf")

    r = build(corpus, index, FakeEmbedder(), verbose=False)

    assert r["by_suffix"] == {"md": 1, "txt": 1}
    assert len(r["failures"]) == 1
    assert "bad.pdf" in r["failures"][0]  # 失败清单里要能看出是哪个文件


def test_iter_skips_files_inside_hidden_directories(corpus_and_index):
    """隐藏**目录**里的文件也要跳过——只判断文件名会漏（`corpus/.trash/x.pdf` 被收进来）。"""
    corpus, index = corpus_and_index
    _write(corpus, "ok.md", "# T\n内容。\n")
    (corpus / ".trash").mkdir()  # _write 不建父目录，这里显式建
    _write(corpus, ".trash/old.pdf", "not a real pdf")

    r = build(corpus, index, FakeEmbedder(), verbose=False)

    assert r["files"] == 1  # 隐藏目录里的那个不算文件
    assert r["failures"] == []  # 也不该因为它而报解析失败


def test_scan_reports_stats_and_failures(corpus_and_index):
    from rag.index import scan

    corpus, _ = corpus_and_index
    _write(corpus, "a.md", "# T\n有效内容。\n")
    _write(corpus, "b.txt", "纯文本内容。\n")
    (corpus / "bad.pdf").write_bytes(b"not a real pdf")

    r = scan(corpus)

    assert r["files"] == 3
    assert r["by_suffix"] == {"md": 1, "txt": 1}
    assert len(r["failures"]) == 1 and "bad.pdf" in r["failures"][0]
    assert r["chunks_total"] >= 2


def test_scan_flags_documents_that_yield_no_text(corpus_and_index):
    """解析成功、却一个片段都没有 —— 扫描版 PDF 的典型症状，预检要单独列出来。

    `pypdf` 对图片型 PDF **不报错**，只是返回空字符串；不专门检查的话，
    它会安静地变成"语料里少了这一篇"，而失败清单上什么都看不到。
    """
    from rag.index import scan

    corpus, _ = corpus_and_index
    _write(corpus, "ok.md", "# T\n有效内容。\n")
    _write(corpus, "blank.md", "   \n\n   \n")

    r = scan(corpus)

    assert r["failures"] == []
    assert r["empty"] == ["blank.md"]


def test_scan_flags_garbled_text_layer(corpus_and_index):
    """文字层是字形编号（缺 ToUnicode 映射）的文件要被单独标出来。

    这类文件**解析不报错、片段也照常生成**，只有读正文才看得出是 `/G21/G22` 而不是文字
    （第 20 周 44 篇真实资料里命中 1 篇，2010 年的期刊）。
    """
    from rag.index import scan

    corpus, _ = corpus_and_index
    _write(corpus, "ok.md", "# T\n正常的中文内容。\n")
    line = " ".join(f"/G{n}" for n in range(21, 80))
    _write(corpus, "garbled.txt", "\n\n".join([line] * 8) + "\n")

    r = scan(corpus)

    assert r["failures"] == []
    assert len(r["suspect"]) == 1
    assert "garbled.txt" in r["suspect"][0]


def test_verify_passes_on_intact_index(corpus_and_index):
    from rag.index import verify

    corpus, index = corpus_and_index
    _write(corpus, "a.md", "# T\n" + "这是一句话。" * 60 + "\n")
    build(corpus, index, FakeEmbedder(), verbose=False)

    result = verify(index, FakeEmbedder(), sample=4)

    assert result["ok"] is True
    assert result["checked"] >= 1
    assert result["worst"] > 0.999


def test_verify_detects_vector_misalignment(corpus_and_index):
    """把 vectors 的行序颠倒（等价于 T29 那种错位），校验必须报**不通过**。

    这条测试的价值在于：错位时检索结果看着仍然合理，只有"把向量与它自己的文本对一遍"
    才看得出来——所以校验逻辑本身也要有回归测试，否则它坏了我们也不知道。
    """
    from rag.index import verify

    corpus, index = corpus_and_index
    _write(corpus, "a.md", "# T\n" + "这是一句话。" * 60 + "\n")
    _write(corpus, "b.md", "# U\n" + "那是另一句。" * 60 + "\n")
    build(corpus, index, FakeEmbedder(), verbose=False)

    vectors = np.load(index / "vectors.npz")["vectors"]
    np.savez(index / "vectors.npz", vectors=vectors[::-1])

    assert verify(index, FakeEmbedder(), sample=2)["ok"] is False


def test_build_report_contains_stats_failures_and_verify(tmp_path, corpus_and_index):
    from rag.index import _write_build_report, verify

    corpus, index = corpus_and_index
    _write(corpus, "a.md", "# T\n内容。\n")
    (corpus / "bad.pdf").write_bytes(b"not a real pdf")
    result = build(corpus, index, FakeEmbedder(), verbose=False)

    report = tmp_path / "报告" / "建库报告.md"
    _write_build_report(report, result, verify(index, FakeEmbedder(), sample=1))

    text = report.read_text(encoding="utf-8")
    assert "建库报告" in text
    assert "解析失败清单" in text and "bad.pdf" in text
    assert "向量对齐校验" in text


def test_fastembed_cache_dir_is_outside_working_directory():
    """fastembed 的模型缓存默认走 `tempfile.gettempdir()`，而 Windows 上它可能
    **静默回退到当前工作目录**——实测 90.8 MB 的 ONNX 模型落进了
    `notes-qa/fastembed_cache/`（与 jieba 词典缓存同一个根因，见台账 T20）。
    """
    from rag.embedder import CACHE_DIR

    assert CACHE_DIR.parent.name == "data"
    assert CACHE_DIR != Path.cwd()
    assert not Path("fastembed_cache").exists()
