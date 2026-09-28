"""Retriever：混合检索（BM25 稀疏 + 向量稠密）与 RRF 融合。

分工（与前面几层的关系）：
- `parsers` / `chunker`：把文档变成片段；
- `index`：把片段变成向量并落盘；
- 本模块：给定问题，找出最相关的片段。

三个设计约束：

1. **依赖注入，不 import 重型库**：`embedder` 由调用方传入（真实的 bge / fastembed，
   或测试里的 FakeEmbedder），`jieba` / `rank_bm25` 只在稀疏通道首次使用时才 import。
   于是主 .venv（没有 torch）也能 import 本模块并跑测试——延续第 17 周的隔离设计。
2. **分数可解释**：每个命中同时保留融合分与各通道的排名／原始分。出问题时能直接回答
   "是稠密侧没召回，还是融合把它压下去了"，而不是只能盯着一个总分猜。
3. **RRF 按排名融合，不按分数融合**：BM25 分数没有上界、还可能为负（词典里过于常见的词
   idf 为负），余弦落在 [-1, 1]，两者量纲不可比——归一化只能解决后者的范围问题，
   解决不了"可比性"问题。按排名融合天然免疫量纲差异。

RRF 公式：`score(d) = Σ_channels 1 / (k + rank(d))`，rank 从 1 起算，k 默认 60。
两路都召回的片段必然优于只被一路召回的片段，这正是"混合"的价值所在。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunk import Chunk

# ---- 默认参数 ----
RRF_K = 60  # RRF 平滑常数，原论文取值
DEFAULT_TOP_K = 5  # 最终返回条数
CANDIDATE_POOL = 20  # 每个通道取多少条候选（也是送进 LLM 重排的上限）

# jieba 的词典缓存默认落在**当前工作目录**（一个 9 MB 的 `jieba.cache`）。跑一次测试就在仓库根
# 多一个二进制文件，与台账 T2 的 `_probe_linkdir.md` 是同一类"绿灯下的脏数据"。
#
# 注意**不能**用 `tempfile.gettempdir()` 来"挪走"它：Windows 上若 TMP/TEMP 都没设置，
# `gettempdir()` 会**静默回退到当前工作目录**——看着是挪了，实际原地不动（第 18 周实测踩到，
# 写测试时才发现 cache 仍在仓库根）。所以这里显式指向项目自己的 `data/`（已 gitignore），
# 位置确定，且不依赖任何环境变量。
JIEBA_CACHE_FILE = Path(__file__).resolve().parents[1] / "data" / "jieba.cache"

_TOKEN_KEEP = re.compile(r"[0-9a-zA-Z\u4e00-\u9fff]")
_jieba_ready = False


def _prepare_jieba() -> None:
    """首次使用前做两件一次性的事：静音日志、把词典缓存挪出当前工作目录。"""
    global _jieba_ready
    if _jieba_ready:
        return
    import logging

    import jieba

    # ① 关掉 "Building prefix dict..." / "Loading model cost ..."，别污染 CLI 输出；
    jieba.setLogLevel(logging.ERROR)
    # ② 词典缓存换到 data/ 下（见 JIEBA_CACHE_FILE 的说明）。
    #    jieba 内部用 tempfile.mkstemp(dir=父目录) 落盘，父目录必须先存在。
    JIEBA_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    jieba.dt.tmp_dir = str(JIEBA_CACHE_FILE.parent)
    jieba.dt.cache_file = str(JIEBA_CACHE_FILE)
    _jieba_ready = True


def tokenize(text: str) -> list[str]:
    """默认分词器：jieba 切词 → 丢弃纯标点／空白 → 统一小写。

    中英混排交给 jieba：英文词它本来就不切碎，中文按词典切。
    纯标点 token（如 `，` `-` `）`）对 BM25 只有噪声作用，直接丢掉。
    """
    _prepare_jieba()
    import jieba

    return [tok.casefold() for tok in jieba.lcut(text) if _TOKEN_KEEP.search(tok)]


@dataclass(frozen=True)
class SearchHit:
    """一条检索命中。rank/score 按通道分开存，便于解释分数来源。"""

    chunk: Chunk
    score: float  # 融合分（RRF）或基线分
    dense_rank: int | None = None  # 1 起算；None = 稠密通道没召回它
    dense_score: float | None = None  # 余弦相似度（向量已 L2 归一化，即点积）
    sparse_rank: int | None = None
    sparse_score: float | None = None  # BM25 原始分，可为负

    @property
    def channels(self) -> str:
        """哪些通道召回了它，如 "dense+sparse"。"""
        names = []
        if self.dense_rank is not None:
            names.append("dense")
        if self.sparse_rank is not None:
            names.append("sparse")
        return "+".join(names) or "-"

    def to_dict(self) -> dict:
        return {
            "id": self.chunk.id,
            "source_path": self.chunk.source_path,
            "locator": self.chunk.locator,
            "start": self.chunk.start,
            "end": self.chunk.end,
            "score": round(self.score, 6),
            "channels": self.channels,
            "dense_rank": self.dense_rank,
            "dense_score": None
            if self.dense_score is None
            else round(self.dense_score, 6),
            "sparse_rank": self.sparse_rank,
            "sparse_score": None
            if self.sparse_score is None
            else round(self.sparse_score, 6),
            "text": self.chunk.text,
        }


def rrf_fuse(rankings: dict[str, list[int]], *, k: int = RRF_K) -> dict[int, float]:
    """按排名融合多路结果。

    `rankings` 形如 `{"dense": [行号, ...], "sparse": [行号, ...]}`，列表顺序即排名。
    返回 `{行号: 融合分}`。纯函数，方便单独测。
    """
    fused: dict[int, float] = {}
    for rows in rankings.values():
        for rank, row in enumerate(rows, start=1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (k + rank)
    return fused


def _top_rows(scores: np.ndarray, limit: int) -> list[int]:
    """按分数降序取前 limit 个行号。

    `kind="stable"` 是必须的：默认快排不稳定，同分片段的顺序会在两次运行间跳变，
    使"对比表"无法复现。
    """
    rows: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        rows.append(int(i))
        if len(rows) >= limit:
            break
    return rows


class Retriever:
    """混合检索器。`embedder` 为 None 时自动退化为纯 BM25（显式分支，不是兜底）。"""

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedder=None,
        *,
        tokenizer=tokenize,
        rrf_k: int = RRF_K,
        candidate_pool: int = CANDIDATE_POOL,
    ):
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks（{len(chunks)}）与 vectors（{len(vectors)}）行数不一致"
            )
        self.chunks = list(chunks)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self._bm25 = None
        self._inverted: dict[str, set[int]] = {}

    @classmethod
    def from_index(cls, index_dir: str | Path, embedder=None, **kwargs) -> Retriever:
        """从 `index build` 的三件套里读回 chunks 与 vectors。"""
        index_dir = Path(index_dir)
        chunks = [
            Chunk.from_json(line)
            for line in (index_dir / "chunks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        vectors = np.load(index_dir / "vectors.npz")["vectors"]
        return cls(chunks, vectors, embedder, **kwargs)

    # ---- 两个通道 ----

    def _bm25_index(self):
        """按需构建 BM25 索引与倒排表（jieba 分词 + rank_bm25），只建一次。"""
        if self._bm25 is None:
            if not self.chunks:
                return None
            from rank_bm25 import BM25Okapi

            tokenized = [self.tokenizer(c.text) for c in self.chunks]
            # 空语料会让 BM25Okapi 直接 ZeroDivisionError（实测），上面已挡掉。
            self._bm25 = BM25Okapi(tokenized)
            inverted: dict[str, set[int]] = {}
            for row, tokens in enumerate(tokenized):
                for token in set(tokens):
                    inverted.setdefault(token, set()).add(row)
            self._inverted = inverted
        return self._bm25

    def dense_ranking(self, query: str, limit: int) -> list[tuple[int, float]]:
        """稠密通道：查询向量与所有片段向量做点积（等价余弦，向量已归一化）。"""
        if self.embedder is None or self.vectors.shape[0] == 0:
            return []
        query_vec = np.asarray(self.embedder.encode([query])[0], dtype=np.float32)
        scores = self.vectors @ query_vec
        return [(row, float(scores[row])) for row in _top_rows(scores, limit)]

    def sparse_ranking(self, query: str, limit: int) -> list[tuple[int, float]]:
        """稀疏通道：BM25。查询分词为空时返回空（不是返回随机结果）。

        「命中」的判定用**倒排表**，绝不能用「BM25 分数是否为 0」：
        rank_bm25 会把负 idf 成批替换成 `eps = epsilon × 平均 idf`，
        而当平均 idf 恰好为 0 时（例如 2 篇文档、某词正好出现在 1 篇里），
        eps 也是 0 —— 于是"词确实命中、但分数是 0"完全可能发生。
        按分数过滤会把这些真命中静默丢掉（第 18 周写测试时实测踩到）。
        """
        bm25 = self._bm25_index()
        if bm25 is None:
            return []
        tokens = self.tokenizer(query)
        if not tokens:
            return []
        matched: set[int] = set()
        for token in tokens:
            matched |= self._inverted.get(token, set())
        if not matched:
            return []
        scores = np.asarray(bm25.get_scores(tokens), dtype=np.float64)
        rows = sorted(
            matched, key=lambda r: (-scores[r], r)
        )  # 同分按行号兜底，保证可复现
        return [(row, float(scores[row])) for row in rows[:limit]]

    # ---- 主入口 ----

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        candidates: int | None = None,
        use_dense: bool = True,
        use_sparse: bool = True,
    ) -> list[SearchHit]:
        """混合检索。返回按融合分降序的 `SearchHit` 列表。"""
        if not isinstance(query, str) or not query.strip() or top_k <= 0:
            return []
        pool = self.candidate_pool if candidates is None else candidates

        dense = self.dense_ranking(query, pool) if use_dense else []
        sparse = self.sparse_ranking(query, pool) if use_sparse else []
        if not dense and not sparse:
            return []

        rankings: dict[str, list[int]] = {}
        if dense:
            rankings["dense"] = [row for row, _ in dense]
        if sparse:
            rankings["sparse"] = [row for row, _ in sparse]
        fused = rrf_fuse(rankings, k=self.rrf_k)

        dense_ranks = {row: i for i, (row, _) in enumerate(dense, start=1)}
        sparse_ranks = {row: i for i, (row, _) in enumerate(sparse, start=1)}
        dense_scores = dict(dense)
        sparse_scores = dict(sparse)

        # 同分时先看"最好名次"，再按行号兜底 —— 保证结果可复现。
        rows = sorted(
            fused,
            key=lambda r: (
                -fused[r],
                min(dense_ranks.get(r, 1 << 30), sparse_ranks.get(r, 1 << 30)),
                r,
            ),
        )
        return [
            SearchHit(
                chunk=self.chunks[row],
                score=fused[row],
                dense_rank=dense_ranks.get(row),
                dense_score=dense_scores.get(row),
                sparse_rank=sparse_ranks.get(row),
                sparse_score=sparse_scores.get(row),
            )
            for row in rows[:top_k]
        ]


def keyword_baseline_search(
    query: str, chunks: list[Chunk], top_k: int = DEFAULT_TOP_K
) -> list[SearchHit]:
    """第一阶段 `search_notes` 的老做法：整串子串匹配，不分词、不算相关度。

    保留它不是因为它好用，而是因为第 19 周要拿它当**基线**：
    新方法如果打不过它，就是白做。但要看清它的语义——
    它只认"问题原样出现在片段里"，所以自然语言提问几乎必然 0 命中，
    而短关键词提问（如 `RRF`）命中很准。这两个数字必须分开解读，
    否则很容易得出"混合检索不如关键词"这种错结论。
    """
    if not isinstance(query, str):
        return []
    needle = query.strip().casefold()
    if not needle or top_k <= 0:
        return []
    hits: list[SearchHit] = []
    for chunk in chunks:
        count = chunk.text.casefold().count(needle)
        if count:
            hits.append(SearchHit(chunk=chunk, score=float(count)))
            if len(hits) >= top_k:
                break
    return hits


def make_stdout_forgiving() -> None:
    """让 CLI 不被"打印不出来"打断：遇到控制台编码装不下的字符就替换，而不是抛异常。

    Windows 控制台默认是 GBK，而笔记正文里会出现 ✅ / ✓ / ⚠ 这类字符；
    `print` 一个装不下的字符会直接 `UnicodeEncodeError`，把整个命令打断——
    检索本身没问题，却白跑一次（第 18 周冒烟测试真踩到）。
    """
    import sys

    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):  # 旧解释器 / 被重定向到不支持的对象
        pass


def format_hits(hits: list[SearchHit], *, snippet_chars: int = 60) -> str:
    """把命中列表打印成紧凑的多行文本（CLI 与人工抽查用）。"""
    lines = []
    for i, hit in enumerate(hits, start=1):
        text = hit.chunk.text.replace("\n", " ").strip()
        if len(text) > snippet_chars:
            text = text[:snippet_chars] + "…"
        lines.append(
            f"{i:>2}. [{hit.score:.5f}] {hit.chunk.source_path} "
            f"({hit.chunk.locator}) {hit.channels:<12} {text}"
        )
    return "\n".join(lines) or "（无命中）"


# ---------------------------------------------------------------------------
# CLI：python -m rag.retriever --query "..." [--top-k 5] [--no-dense]
#
# 有稠密通道时需要在 .venv-rag 里跑（要 torch / onnxruntime）：
#   & "..\..\.venv-rag\Scripts\python.exe" -m rag.retriever --query "小批量为什么不用 GPU"
# 只要 BM25 时主 .venv 也能跑：--no-dense
# ---------------------------------------------------------------------------


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="混合检索（BM25 + 向量 + RRF）")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidates", type=int, default=CANDIDATE_POOL)
    parser.add_argument(
        "--no-dense", action="store_true", help="只用 BM25（主环境可直接跑）"
    )
    parser.add_argument("--no-sparse", action="store_true", help="只用向量")
    parser.add_argument(
        "--baseline", action="store_true", help="追加第一阶段的关键词基线结果"
    )
    parser.add_argument("--backend", default="fastembed", choices=["bge", "fastembed"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    make_stdout_forgiving()

    embedder = None
    if not args.no_dense:
        from .embedder import Embedder  # 延迟导入：主环境没 torch 也能走 --no-dense

        embedder = Embedder(backend=args.backend, device=args.device)

    retriever = Retriever.from_index(args.index, embedder)
    hits = retriever.search(
        args.query,
        args.top_k,
        candidates=args.candidates,
        use_dense=not args.no_dense,
        use_sparse=not args.no_sparse,
    )
    print(f"[混合检索] {args.query}")
    print(format_hits(hits))
    if args.baseline:
        base = keyword_baseline_search(args.query, retriever.chunks, args.top_k)
        print("\n[关键词基线（第一阶段做法）]")
        print(format_hits(base))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
