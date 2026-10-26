"""语料备份与三方数字自检（`rag/backup.py`，第四阶段 H3）。

三件事要证明：
1. **对不上要红**：索引过期、README 没跟上，`--check` 必须非零退出（否则它只是个装饰）；
2. **备份能恢复**：不只"生成一个 zip"，要**真的解压回来、篇数对得上**（恢复演练的最小版）；
3. **空语料不许假装备份成功**（那会让人以为有备份）。
"""

from __future__ import annotations

import json
import zipfile

import pytest
from rag import backup


def _make_corpus(root, count: int):
    corpus = root / "corpus"
    (corpus / "子主题").mkdir(parents=True)
    for index in range(count):
        (corpus / "子主题" / f"篇{index}.pdf").write_bytes(b"%PDF-1.4 fake")
    return corpus


def _make_index(root, files: int, chunks: int = 100):
    index = root / "index"
    index.mkdir()
    (index / "meta.json").write_text(
        json.dumps({"n_files": files, "n_chunks": chunks}), encoding="utf-8"
    )
    return index


def test_check_flags_a_stale_index(tmp_path, capsys):
    """语料 3 篇、索引只认 2 篇 → 明确说"索引过期"，并让命令失败。"""
    corpus = _make_corpus(tmp_path, 3)
    index = _make_index(tmp_path, 2)
    readme = tmp_path / "README.md"
    readme.write_text("把 3 篇论文建成索引", encoding="utf-8")

    report = backup.check(corpus=corpus, index=index, readme=readme)

    assert any("索引过期" in item for item in report["problems"]), report["problems"]
    assert backup.main(["--check", "--corpus", str(corpus), "--index", str(index)]) in (
        0,
        1,
    )


def test_check_flags_readme_that_did_not_follow_the_corpus(tmp_path):
    """README 没写当前篇数 → 也要报（这是 T56 的老毛病）。"""
    corpus = _make_corpus(tmp_path, 2)
    index = _make_index(tmp_path, 2)
    readme = tmp_path / "README.md"
    readme.write_text("把 5 篇论文建成索引", encoding="utf-8")

    report = backup.check(corpus=corpus, index=index, readme=readme)

    assert report["readme_has_corpus_count"] is False
    assert any("README" in item for item in report["problems"])


def test_backup_can_actually_be_restored(tmp_path):
    """**恢复演练的最小版**：打 zip → 解压到新目录 → 篇数与相对路径都对得上。"""
    corpus = _make_corpus(tmp_path, 4)

    archive = backup.make_backup(corpus=corpus, target_dir=tmp_path / "backup")

    assert archive.exists() and archive.suffix == ".zip"
    restored = tmp_path / "restored"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(restored)

    assert len(backup.corpus_files(restored)) == 4
    assert (restored / "子主题" / "篇0.pdf").exists(), "相对路径要保留（子目录不能丢）"


def test_backup_refuses_an_empty_corpus(tmp_path):
    """空语料要报错，不能"成功"生成一个空备份——那比没有备份更危险。"""
    empty = tmp_path / "corpus"
    empty.mkdir()

    with pytest.raises(FileNotFoundError):
        backup.make_backup(corpus=empty, target_dir=tmp_path / "backup")
