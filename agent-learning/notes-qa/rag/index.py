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
    """遍历 corpus 下所有支持的格式。

    跳过**隐藏文件与隐藏目录**：原来只判断 `p.name`，于是 `corpus/.trash/x.pdf`
    会被收进索引（第 20 周核对语料时发现）。

    要排除的文件请放在 `data/corpus` **之外**（例如 `data/corpus-excluded/`），
    不要靠 `corpus/子目录/` 去"藏"——`rglob` 一定会走进去。
    """
    for p in sorted(corpus_dir.rglob("*")):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(corpus_dir).parts
        if any(part.startswith(".") for part in rel_parts):
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
    force_full: bool = False,
) -> dict:
    """构建或增量更新索引。

    `force_full=True`（CLI 的 `--all`）跳过增量判断、整体重算——语料大改之后想要一份
    "从零构建"的干净结果时用它，不必手工删 `data/index/*`（第 19 周 T29 就是靠删文件重建的，
    有了这个开关就不必再走那条路）。

    返回：{files, by_suffix, chunks_total, chunks_new, failures, elapsed_s, backend}。
    `failures` **一并返回而不只是打印**：第 20 周的交付物里就有"失败清单"这一项。
    """
    t0 = time.time()
    index_dir.mkdir(parents=True, exist_ok=True)

    # 1. 解析 + 切分（纯 Python，不需要 embedder）
    all_chunks: list[Chunk] = []
    files = 0
    by_suffix: dict[str, int] = {}
    failures: list[str] = []
    for path in _iter_corpus_files(corpus_dir):
        files += 1
        rel = path.relative_to(corpus_dir).as_posix()
        try:
            blocks = parse_file(path)
        except Exception as exc:  # noqa: BLE001 —— 单文件失败不中断整体
            failures.append(f"{rel}: {type(exc).__name__}: {exc}")
            continue
        suffix = path.suffix.lower().lstrip(".")
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
        chunks = chunk_blocks(
            blocks,
            source_path=rel,
            doc_type=path.suffix.lower().lstrip("."),
        )
        all_chunks.extend(chunks)

    # 2. 增量判断（后端换了就不算"增量"，见 _load_old_vectors 的说明）
    if force_full:
        old_chunks, old_vecs, old_note = [], None, "强制全量（--all）"
    else:
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
        "by_suffix": by_suffix,
        "chunks_total": len(all_chunks),
        "chunks_new": len(new_chunks),
        "failures": failures,
        "elapsed_s": elapsed,
        "backend": getattr(embedder, "backend", "unknown"),
    }


def scan(corpus_dir: Path) -> dict:
    """**只解析 + 切分**，不算向量：正式建库前的预检。

    比 `build` 快一个数量级（不加载模型、不编码），而且**主 .venv 就能跑**（不需要 torch）。
    拿到 30 篇资料的当口先跑它比先跑 build 划算：能立刻看到"哪个文件读不出来"，
    以及**哪些文件解析成功却一个字都没提取到**——后者是扫描版 PDF 的典型症状，
    `pypdf` 不报错、只是返回空（规划风险表里那条）。
    """
    files = 0
    by_suffix: dict[str, int] = {}
    failures: list[str] = []
    empty: list[str] = []
    suspect: list[str] = []
    chunks_total = 0
    for path in _iter_corpus_files(corpus_dir):
        files += 1
        rel = path.relative_to(corpus_dir).as_posix()
        try:
            blocks = parse_file(path)
        except Exception as exc:  # noqa: BLE001 —— 预检只报错、不中断
            failures.append(f"{rel}: {type(exc).__name__}: {exc}")
            continue
        suffix = path.suffix.lower().lstrip(".")
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
        chunks = chunk_blocks(blocks, source_path=rel, doc_type=suffix)
        if not chunks:
            empty.append(rel)
        chunks_total += len(chunks)
        # 老 PDF 可能缺 ToUnicode 映射：抽出来的是字形编号（"/G21/G22"）而不是文字。
        # 这类文件**解析不报错、片段也照常生成**，只有读正文才看得出来——所以单独列出来。
        garbled = sum(1 for c in chunks if c.text.count("/G") >= 5)
        if garbled >= 3:
            suspect.append(f"{rel}（{garbled} 个片段含 /G 字形编号，建议换一个版本）")
    return {
        "files": files,
        "by_suffix": by_suffix,
        "chunks_total": chunks_total,
        "failures": failures,
        "empty": empty,
        "suspect": suspect,
    }


def verify(index_dir: Path, embedder, *, sample: int = 8) -> dict:
    """抽 N 条片段重新编码，与 `vectors.npz` 里的存量向量比余弦。

    这是 T29 那条"静默错位"的常态化检查——错位时检索**看着仍然正常**
    （向量只差一两行，相邻片段常出自同一节），只有把向量和它自己的文本对一遍才看得出来。
    抽样取"等距 + 末尾"而不用随机：同一份索引跑两次要给出同一个结论。
    """
    chunks = [
        Chunk.from_json(line)
        for line in (index_dir / "chunks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not chunks:
        return {"n_chunks": 0, "checked": 0, "worst": None, "ok": True, "rows": []}
    vectors = np.load(index_dir / "vectors.npz")["vectors"]
    if len(chunks) != vectors.shape[0]:
        return {
            "n_chunks": len(chunks),
            "checked": 0,
            "worst": None,
            "ok": False,
            "rows": [],
            "note": f"行数不一致（chunks {len(chunks)} / vectors {vectors.shape[0]}）",
        }
    n = min(sample, len(chunks))
    rows = sorted({round(i * (len(chunks) - 1) / max(n - 1, 1)) for i in range(n)})
    fresh = embedder.encode([chunks[i].text for i in rows])
    cosines = []
    for k, row in enumerate(rows):
        vec = fresh[k] / (np.linalg.norm(fresh[k]) or 1.0)
        cosines.append(float(np.dot(vec, vectors[row])))
    worst = min(cosines)
    return {
        "n_chunks": len(chunks),
        "checked": len(rows),
        "worst": worst,
        "ok": worst > 0.999,
        "rows": [
            {"row": int(r), "cosine": round(c, 4), "source_path": chunks[r].source_path}
            for r, c in zip(rows, cosines)
        ],
    }


def _write_build_report(path: Path, result: dict, verify_result: dict | None) -> None:
    """把建库统计 + 失败清单（+ 可选的对齐校验）写成 UTF-8 markdown。"""
    lines = [
        "# 建库报告（脚本生成）",
        "",
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}｜后端：{result.get('backend')}",
        "> 由 `python -m rag.index build --all --report <本文件>` 生成；语料变了请重跑，不要手改数字。",
        "",
        "## 统计",
        "",
        "| 项           | 数值 |",
        "| ------------ | ---- |",
        f"| 语料文件     | {result['files']} |",
        f"| 片段总数     | {result['chunks_total']} |",
        f"| 本次新增     | {result['chunks_new']} |",
        f"| 耗时         | {result['elapsed_s']:.2f}s |",
        f"| 解析失败     | {len(result['failures'])} |",
        "",
        "| 格式     | 文件数 |",
        "| -------- | ------ |",
    ]
    for suffix, count in sorted(result.get("by_suffix", {}).items()):
        lines.append(f"| .{suffix} | {count} |")
    lines += ["", "## 解析失败清单", ""]
    if result["failures"]:
        lines += [f"- {item}" for item in result["failures"]]
    else:
        lines.append("（无）")
    if verify_result is not None:
        lines += ["", "## 向量对齐校验（防 T29 那类静默错位）", ""]
        if verify_result["checked"]:
            verdict = "通过" if verify_result["ok"] else "**不通过**"
            lines.append(
                f"- 抽查 {verify_result['checked']} / {verify_result['n_chunks']} 条，"
                f"最低余弦 **{verify_result['worst']:.4f}** → {verdict}"
            )
            for row in verify_result["rows"]:
                lines.append(
                    f"  - 行 {row['row']}：{row['cosine']:.4f}　{row['source_path']}"
                )
        else:
            lines.append(f"- 未抽查（{verify_result.get('note', '索引为空')}）")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI 入口（需要在 .venv-rag 里跑，依赖 torch / onnxruntime）：
#
#   # 增量构建
#   ..\..\.venv-rag\Scripts\python.exe -m rag.index build --backend fastembed
#   # 一键全量重建 + 建库报告 + 对齐校验（第 20 周的主命令）
#   ..\..\.venv-rag\Scripts\python.exe -m rag.index build --backend fastembed --all --verify --report eval\建库报告.md
#   # 只查向量对齐（不重建）
#   ..\..\.venv-rag\Scripts\python.exe -m rag.index verify --sample 8
# ---------------------------------------------------------------------------


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="RAG 索引构建 / 校验")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="构建或增量更新索引")
    p_build.add_argument("--corpus", default="data/corpus")
    p_build.add_argument("--index", default="data/index")
    p_build.add_argument(
        "--backend", default="bge", choices=["bge", "fastembed", "none"]
    )
    p_build.add_argument("--device", default="cuda")
    p_build.add_argument("--batch-size", type=int, default=32)
    p_build.add_argument(
        "--all", action="store_true", help="忽略旧索引、整体重算（一键重建）"
    )
    p_build.add_argument("--report", default="", help="把建库报告写成 UTF-8 文件")
    p_build.add_argument("--verify", action="store_true", help="构建后抽查向量对齐")

    p_scan = sub.add_parser(
        "scan", help="只解析 + 切分（不算向量）：建库前先看失败清单与空文档"
    )
    p_scan.add_argument("--corpus", default="data/corpus")

    p_verify = sub.add_parser("verify", help="只抽查向量对齐（不重建）")
    p_verify.add_argument("--index", default="data/index")
    p_verify.add_argument(
        "--backend", default="fastembed", choices=["bge", "fastembed"]
    )
    p_verify.add_argument("--device", default="cuda")
    p_verify.add_argument("--sample", type=int, default=8)

    args = parser.parse_args()

    if args.cmd == "scan":  # 不需要 embedder：主 .venv 里也能跑
        result = scan(Path(args.corpus))
        for suffix, count in sorted(result["by_suffix"].items()):
            print(f"  .{suffix:<6} {count} 篇")
        print(
            f"文件 {result['files']}｜片段 {result['chunks_total']}｜"
            f"解析失败 {len(result['failures'])}｜空文档 {len(result['empty'])}｜"
            f"可疑 {len(result['suspect'])}"
        )
        if result["failures"]:
            print("[解析失败]")
            for item in result["failures"]:
                print(f"  - {item}")
        if result["empty"]:
            print("[解析成功但一个字都没提取到]（常见于扫描版 PDF）")
            for item in result["empty"]:
                print(f"  - {item}")
        if result["suspect"]:
            print("[文字层可疑：抽出的是字形编号而不是文字]")
            for item in result["suspect"]:
                print(f"  - {item}")
        return 0

    from .embedder import Embedder

    if args.cmd == "build":
        embedder = Embedder(
            backend=args.backend,
            device=args.device,
            batch_size=args.batch_size,
        )
        result = build(
            Path(args.corpus), Path(args.index), embedder, force_full=args.all
        )
        print(f"结果：{result}")
        verify_result = verify(Path(args.index), embedder) if args.verify else None
        if verify_result is not None:
            verdict = "通过" if verify_result["ok"] else "不通过"
            print(
                f"对齐校验：抽查 {verify_result['checked']}/{verify_result['n_chunks']} 条"
                f"，最低余弦 {verify_result['worst']} -> {verdict}"
            )
        if args.report:
            _write_build_report(Path(args.report), result, verify_result)
            print(f"[已写入 {args.report}]")
        return 0

    if args.cmd == "verify":
        embedder = Embedder(backend=args.backend, device=args.device)
        verify_result = verify(Path(args.index), embedder, sample=args.sample)
        for row in verify_result["rows"]:
            print(
                f"  行 {row['row']:>5}：余弦 {row['cosine']:.4f}　{row['source_path']}"
            )
        verdict = "通过" if verify_result["ok"] else "不通过"
        print(
            f"抽查 {verify_result['checked']}/{verify_result['n_chunks']} 条，"
            f"最低余弦 {verify_result['worst']} -> {verdict}"
        )
        return 0 if verify_result["ok"] else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
