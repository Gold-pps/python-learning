"""Index：把 corpus 建成可检索的索引（chunks.jsonl + vectors.npz）。

两个核心设计：

1. **增量跳过**：用 chunk 的 hash 判断"是否已经算过向量"。
   已存在的从旧索引直接搬向量，只对新增的 chunk 调 embedder——第二次构建显著变快。
2. **延迟导入 embedder**：本模块能被主环境 import（用于测试 build 流程），
   只有传入真实的 Embedder 并调用 build() 时才需要 torch。

产物（都在 index_dir 下）：
- chunks.jsonl：每行一个 Chunk（顺序与 vectors 行号一致）
- vectors.npz：np.savez 格式，含 'vectors'（float32, shape=(N, D)）
- meta.json：模型名、维度、chunk 数、构建时间
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .chunk import Chunk
from .chunker import chunk_blocks
from .parsers import SUPPORTED_SUFFIXES, parse_file


def _iter_corpus_files(corpus_dir: Path):
    """遍历 corpus 下所有支持的格式，跳过隐藏文件。"""
    for p in sorted(corpus_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        yield p


def _load_old_vectors(
    index_dir: Path, backend: str | None
) -> tuple[list[Chunk], np.ndarray | None, str]:
    """读旧索引，返回 `(旧 chunks 列表, 旧 vectors, 弃用原因)`；可用时原因为空串。

    **返回列表而不是 `{hash: Chunk}` 字典**——这一点至关重要：调用方要按
    "列表下标 = `vectors.npz` 的行号"来搬运旧向量，而字典一旦按 hash 去重，
    它的下标就不再等于真实行号了（详见 `build()` 里的说明与台账 T29）。

    三个"不可信就整体重算"的条件，第三条是第 18 周补的：

    1. 三件套不全（比如上次构建崩了）；
    2. chunks 与 vectors 行数不一致；
    3. **backend 与 `meta.json` 里记的不一致**——增量跳过只按内容 hash 判断，
       如果上一份索引是 bge 算的、这次用 fastembed，旧向量会被原样搬过来，
       同时 `meta.json` 的 model 被改写成新后端：**数据没变、标签变了**。
       今天两个后端跑同一个模型、向量等价（交叉相似度 1.0），看不出问题；
       等哪天真的换了模型，这就是一份静默的错误索引。
       宁可多算一次，也不让"跳过"跨过后端边界。
    """
    chunks_path = index_dir / "chunks.jsonl"
    vectors_path = index_dir / "vectors.npz"
    meta_path = index_dir / "meta.json"
    if not chunks_path.exists() or not vectors_path.exists():
        return [], None, "三件套不全"
    try:
        old_backend = json.loads(meta_path.read_text(encoding="utf-8")).get("model")
    except (OSError, ValueError) as exc:
        return [], None, f"meta.json 读不出来（{type(exc).__name__}）"
    if backend is not None and old_backend != backend:
        return [], None, f"后端变了（旧 {old_backend!r} → 新 {backend!r}）"
    old_chunks: list[Chunk] = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            old_chunks.append(Chunk.from_json(line))
    old_vecs = np.load(vectors_path)["vectors"]
    if len(old_chunks) != old_vecs.shape[0]:
        return (
            [],
            None,
            f"行数不一致（chunks {len(old_chunks)} / vectors {old_vecs.shape[0]}）",
        )
    return old_chunks, old_vecs, ""


def build(
    corpus_dir: Path,
    index_dir: Path,
    embedder,
    *,
    verbose: bool = True,
) -> dict:
    """构建或增量更新索引。

    返回统计：{files, chunks_total, chunks_new, elapsed_s}。
    """
    t0 = time.time()
    index_dir.mkdir(parents=True, exist_ok=True)

    # 1. 解析 + 切分（纯 Python，不需要 embedder）
    all_chunks: list[Chunk] = []
    files = 0
    failures: list[str] = []
    for path in _iter_corpus_files(corpus_dir):
        files += 1
        rel = path.relative_to(corpus_dir).as_posix()
        try:
            blocks = parse_file(path)
        except Exception as exc:  # noqa: BLE001 —— 单文件失败不中断整体
            failures.append(f"{rel}: {type(exc).__name__}: {exc}")
            continue
        chunks = chunk_blocks(
            blocks,
            source_path=rel,
            doc_type=path.suffix.lower().lstrip("."),
        )
        all_chunks.extend(chunks)

    # 2. 增量判断（后端换了就不算"增量"，见 _load_old_vectors 的说明）
    old_chunks, old_vecs, old_note = _load_old_vectors(
        index_dir, getattr(embedder, "backend", None)
    )
    old_hashes = {c.hash for c in old_chunks}
    new_chunks = [c for c in all_chunks if c.hash not in old_hashes]

    if verbose:
        print(
            f"[index] 文件 {files}，片段总数 {len(all_chunks)}，其中新增 {len(new_chunks)}"
        )
        if old_note:
            print(f"[index] 旧索引整体重算：{old_note}")
        if failures:
            print(f"[index] 解析失败 {len(failures)} 个：")
            for f in failures:
                print(f"  - {f}")

    # 3. 计算新向量
    if new_chunks:
        new_vecs = embedder.encode([c.text for c in new_chunks])
    else:
        new_vecs = np.zeros((0, embedder.dim), dtype=np.float32)

    # 4. 组装全部向量（按 all_chunks 顺序）
    #
    # ⚠️ 建"hash → 行号"表必须按**旧 chunks 列表**的下标，不能拿"按 hash 去重后的字典"
    # 去 enumerate：语料里只要有两段内容完全相同的片段（hash 相同），字典下标就与
    # vectors.npz 的真实行号错开，**重复片段之后的每一行都会拿到邻居的向量**。
    # 后果是静默的——向量整体只错位一两行，检索看着仍然正常、只是"永远差一点"，
    # 直到第 19 周用严格判据（要求命中"含关键词的那一段"）才暴露出来。
    # 详见台账 T29 与 tests/test_index.py 的同名回归测试。
    old_hash_to_row: dict[str, int] = {}
    for i, c in enumerate(old_chunks):
        old_hash_to_row.setdefault(
            c.hash, i
        )  # 同文本重复出现时取首次行号（向量必然相同）
    new_hash_to_row: dict[str, int] = {}
    for i, c in enumerate(new_chunks):
        new_hash_to_row.setdefault(c.hash, i)
    rows = []
    for c in all_chunks:
        if c.hash in old_hash_to_row:
            rows.append(old_vecs[old_hash_to_row[c.hash]])
        else:
            rows.append(new_vecs[new_hash_to_row[c.hash]])
    if rows:
        vectors = np.stack(rows).astype(np.float32)
    else:
        vectors = np.zeros((0, embedder.dim), dtype=np.float32)

    # 5. 落盘
    chunks_path = index_dir / "chunks.jsonl"
    vectors_path = index_dir / "vectors.npz"
    meta_path = index_dir / "meta.json"

    with chunks_path.open("w", encoding="utf-8", newline="\n") as f:
        for c in all_chunks:
            f.write(c.to_json() + "\n")
    np.savez(vectors_path, vectors=vectors)
    meta_path.write_text(
        json.dumps(
            {
                "model": getattr(embedder, "backend", "unknown"),
                "dim": int(vectors.shape[1]) if vectors.size else embedder.dim,
                "n_chunks": len(all_chunks),
                "n_files": files,
                "n_failures": len(failures),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed = time.time() - t0
    if verbose:
        print(f"[index] 构建完成，耗时 {elapsed:.2f}s，已写入 {index_dir}")
    return {
        "files": files,
        "chunks_total": len(all_chunks),
        "chunks_new": len(new_chunks),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI 入口：python -m rag.index build [--device cuda] [--backend bge]
#
# 需要在 .venv-rag 里跑（依赖 torch）：
#   ..\..\.venv-rag\Scripts\python.exe -m rag.index build
# ---------------------------------------------------------------------------


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="RAG 索引构建")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="构建或增量更新索引")
    p_build.add_argument("--corpus", default="data/corpus")
    p_build.add_argument("--index", default="data/index")
    p_build.add_argument(
        "--backend", default="bge", choices=["bge", "fastembed", "none"]
    )
    p_build.add_argument("--device", default="cuda")
    p_build.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    if args.cmd == "build":
        from .embedder import Embedder

        embedder = Embedder(
            backend=args.backend,
            device=args.device,
            batch_size=args.batch_size,
        )
        r = build(Path(args.corpus), Path(args.index), embedder)
        print(f"结果：{r}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
