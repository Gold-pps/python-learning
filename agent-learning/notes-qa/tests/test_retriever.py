"""混合检索（BM25 + 向量 + RRF）的单元测试。

全部离线：embedder 用假对象（注入固定查询向量），jieba / rank_bm25 是纯 Python，
所以主 .venv（没有 torch）里就能跑——与 test_index.py 的 FakeEmbedder 思路一致。

RRF 的测试刻意用"一路第 1 名 vs 两路各第 2 名"这种**不对称**例子：
对称例子（A 在稠密第 1、稀疏第 2，B 反过来）两边分数会完全相等，
测不出融合到底有没有生效。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rag import retriever as R
from rag.chunk import Chunk
from rag.index import build
from rag.retriever import Retriever, SearchHit, keyword_baseline_search, rrf_fuse


def make_chunk(name: str, text: str, locator: str = "整篇") -> Chunk:
    return Chunk.create(
        source_path=name, doc_type="md", locator=locator, start=1, end=1, text=text
    )


class FixedEmbedder:
    """按预设表返回查询向量；未登记的文本返回全零向量。"""

    backend = "fixed"
    dim = 4

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = [
            np.asarray(self.mapping.get(t, [0.0] * self.dim), dtype=np.float32)
            for t in texts
        ]
        return np.stack(rows)


def _split_tokenizer(text: str) -> list[str]:
    """测试用分词器：按空格切，让 BM25 的命中范围完全可控。"""
    return text.split()


_USE_DEFAULT_EMBEDDER = object()


def _make_retriever(
    corpus, vectors, query_map=None, embedder=_USE_DEFAULT_EMBEDDER, **kwargs
) -> Retriever:
    """`embedder=None` 与"没传 embedder"要分得清：前者是"只要 BM25"，
    后者是"用按 query_map 说话的假编码器"。用哨兵对象区分，
    否则 `embedder=None` 会被悄悄换成假编码器，稠密通道照样出声（写测试时真踩到）。"""
    chunks = [make_chunk(name, text) for name, text in corpus]
    if embedder is _USE_DEFAULT_EMBEDDER:
        embedder = FixedEmbedder(query_map or {})
    return Retriever(
        chunks,
        np.asarray(vectors, dtype=np.float32),
        embedder,
        tokenizer=_split_tokenizer,
        **kwargs,
    )


# ---- RRF 本体 ----


def test_rrf_fuse_sums_reciprocal_ranks():
    fused = rrf_fuse({"dense": [0, 1], "sparse": [1]}, k=60)
    assert fused[0] == pytest.approx(1 / 61)
    assert fused[1] == pytest.approx(1 / 62 + 1 / 61)
    # 两路都召回（哪怕名次更差）应当压过只被一路召回
    assert fused[1] > fused[0]


def test_rrf_fuse_empty_rankings_is_empty():
    assert rrf_fuse({}) == {}


# ---- 融合排序 ----


def test_both_channels_beat_single_channel_and_order_is_stable():
    """两路都召回 > 只被一路召回（哪怕那一路里它是第 1 名）。

    构造：dense 排名 b(1.0) > a(0.9) > c(0.0)；sparse 只命中 a 与 c。
    - a：dense#2 + sparse#1 = 1/62 + 1/61；
    - c：dense#3 + sparse#2 = 1/63 + 1/62；
    - b：只有 dense#1 = 1/61。
    于是 a、c 两个"两路都召回"的片段排在 b 前面——这正是"混合"要买到的东西。
    """
    corpus = [("b.md", "beta"), ("a.md", "alpha"), ("c.md", "alpha")]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    retriever = _make_retriever(corpus, vectors, {"alpha": [1.0, 0.0, 0.0, 0.0]})

    hits = retriever.search("alpha", top_k=3)

    assert [h.chunk.source_path for h in hits] == ["a.md", "c.md", "b.md"]
    a, c, b = hits
    assert a.channels == "dense+sparse"
    assert (a.dense_rank, a.sparse_rank) == (2, 1)
    assert a.score == pytest.approx(1 / 62 + 1 / 61)
    assert c.channels == "dense+sparse"
    assert (c.dense_rank, c.sparse_rank) == (3, 2)
    assert c.score == pytest.approx(1 / 63 + 1 / 62)
    assert b.channels == "dense"
    assert b.sparse_rank is None
    assert b.dense_score == pytest.approx(1.0)
    assert b.score == pytest.approx(1 / 61)
    # 两条"两路"的结果都压过"单路第 1 名"
    assert a.score > b.score and c.score > b.score


def test_search_respects_top_k():
    corpus = [("a.md", f"alpha {i}") for i in range(6)]
    vectors = np.eye(6, 4, dtype=np.float32)
    retriever = _make_retriever(corpus, vectors, {"alpha": [1.0, 0.0, 0.0, 0.0]})
    assert len(retriever.search("alpha", top_k=2)) == 2


def test_dense_can_be_disabled_explicitly():
    corpus = [("a.md", "alpha"), ("b.md", "beta")]
    vectors = np.eye(2, 4, dtype=np.float32)
    retriever = _make_retriever(corpus, vectors, {"alpha": [1.0, 0.0, 0.0, 0.0]})

    hits = retriever.search("alpha", use_dense=False)

    assert hits and all(h.dense_rank is None for h in hits)


def test_sparse_can_be_disabled_explicitly():
    corpus = [("a.md", "alpha"), ("b.md", "beta")]
    vectors = np.eye(2, 4, dtype=np.float32)
    retriever = _make_retriever(corpus, vectors, {"alpha": [1.0, 0.0, 0.0, 0.0]})

    hits = retriever.search("alpha", use_sparse=False)

    assert hits and all(h.sparse_rank is None for h in hits)


# ---- 退化与边界 ----


def test_without_embedder_falls_back_to_bm25_only():
    corpus = [("a.md", "alpha"), ("b.md", "beta")]
    vectors = np.zeros((2, 4), dtype=np.float32)
    retriever = _make_retriever(corpus, vectors, embedder=None)

    hits = retriever.search("alpha")

    assert [h.chunk.source_path for h in hits] == ["a.md"]
    assert hits[0].dense_rank is None
    assert hits[0].channels == "sparse"


def test_sparse_returns_nothing_when_no_term_matches():
    corpus = [("a.md", "alpha"), ("b.md", "beta")]
    vectors = np.eye(2, 4, dtype=np.float32)
    retriever = _make_retriever(corpus, vectors, {"delta": [1.0, 0.0, 0.0, 0.0]})

    # 命中判定走倒排表：没有任何片段含 "delta" → 稀疏通道为空
    assert retriever.search("delta", use_dense=False) == []


def test_sparse_keeps_hits_whose_bm25_score_is_zero():
    """回归：不能拿"BM25 分数为 0"当"没命中"。

    2 篇文档、查询词正好只出现在 1 篇里时，rank_bm25 算出的 idf 恰为 0
    （负 idf 被替换成 eps = epsilon × 平均 idf，而平均 idf 也是 0），
    于是**命中的那篇分数也是 0**。按分数过滤会把真命中静默丢掉。
    """
    corpus = [("a.md", "alpha"), ("b.md", "beta")]
    vectors = np.zeros((2, 4), dtype=np.float32)
    retriever = _make_retriever(corpus, vectors, embedder=None)

    hits = retriever.search("alpha", use_dense=False)

    assert [h.chunk.source_path for h in hits] == ["a.md"]
    assert hits[0].sparse_score == pytest.approx(0.0)


def test_blank_query_and_non_positive_top_k_return_empty():
    corpus = [("a.md", "alpha")]
    vectors = np.eye(1, 4, dtype=np.float32)
    retriever = _make_retriever(corpus, vectors, {"alpha": [1.0, 0.0, 0.0, 0.0]})

    assert retriever.search("   ") == []
    assert retriever.search("alpha", top_k=0) == []


def test_empty_index_returns_empty_without_crashing():
    retriever = Retriever([], np.zeros((0, 4), dtype=np.float32), None)
    assert retriever.search("alpha") == []


def test_mismatched_chunks_and_vectors_raise():
    with pytest.raises(ValueError, match="行数不一致"):
        Retriever([make_chunk("a.md", "x")], np.zeros((2, 4), dtype=np.float32))


def test_bm25_index_is_built_lazily():
    corpus = [("a.md", "alpha")]
    vectors = np.eye(1, 4, dtype=np.float32)
    retriever = _make_retriever(corpus, vectors, {"alpha": [1.0, 0.0, 0.0, 0.0]})

    assert retriever._bm25 is None  # 私有属性：就是要测"没提前建"
    retriever.search("alpha")
    assert retriever._bm25 is not None


# ---- 分词器 ----


def test_tokenize_lowercases_and_drops_punctuation():
    tokens = R.tokenize("RAG 混合检索！（RRF）")

    for expected in ("rag", "混合", "检索", "rrf"):
        assert expected in tokens
    for dropped in ("！", "（", "）", " "):
        assert dropped not in tokens


def test_jieba_cache_is_written_outside_working_directory():
    """jieba 默认把 9 MB 的词典缓存写在**当前工作目录**，跑一次测试就在仓库根多一个
    `jieba.cache`——与台账 T2 的 `_probe_linkdir.md` 同类的"绿灯下的脏数据"。

    这里顺带钉住一个反直觉的坑：`tempfile.gettempdir()` 在 Windows 上若 TMP/TEMP 未设置，
    会**静默回退到当前工作目录**，所以"挪到临时目录"这种做法等于原地不动。
    """
    R.tokenize("测试分词")

    assert not Path("jieba.cache").exists()
    assert R.JIEBA_CACHE_FILE.parent != Path.cwd()
    assert R.JIEBA_CACHE_FILE.parent.name == "data"


# ---- 关键词基线（第一阶段的对照面） ----


def test_keyword_baseline_needs_literal_match():
    chunks = [
        make_chunk("a.md", "混合检索用 RRF 融合两路结果。"),
        make_chunk("b.md", "另一种做法是纯关键词匹配。"),
    ]

    assert [h.chunk.source_path for h in keyword_baseline_search("RRF", chunks)] == [
        "a.md"
    ]
    assert [h.chunk.source_path for h in keyword_baseline_search("rrf", chunks)] == [
        "a.md"
    ]
    # 自然语言提问：整串子串匹配必然 0 命中——这正是要保留基线的理由
    assert keyword_baseline_search("怎么把两路结果合起来", chunks) == []


def test_keyword_baseline_counts_occurrences_as_score():
    chunks = [make_chunk("a.md", "RRF 很重要，RRF 真的重要。")]
    hits = keyword_baseline_search("RRF", chunks)
    assert hits[0].score == 2.0


# ---- 与 index 的联调 ----


class _FakeEmbedder:
    backend = "fake"
    dim = 8

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % 1000)
            v = rng.standard_normal(self.dim).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)


def test_from_index_round_trip(tmp_path):
    corpus = tmp_path / "corpus"
    index = tmp_path / "index"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "# 标题\n" + "混合检索使用 RRF 融合稀疏与稠密两路结果。" * 12,
        encoding="utf-8",
        newline="\n",
    )
    build(corpus, index, _FakeEmbedder(), verbose=False)

    retriever = Retriever.from_index(index, _FakeEmbedder())

    assert len(retriever.chunks) >= 1
    assert retriever.vectors.shape[0] == len(retriever.chunks)
    assert retriever.search("RRF 融合", top_k=3)


def test_search_hit_to_dict_keeps_both_channels():
    chunk = make_chunk("a.md", "alpha", locator="2.4 双后端")
    hit = SearchHit(
        chunk=chunk,
        score=0.032522,
        dense_rank=2,
        dense_score=0.91,
        sparse_rank=1,
        sparse_score=-0.13,
    )

    data = hit.to_dict()

    assert data["channels"] == "dense+sparse"
    assert data["locator"] == "2.4 双后端"
    assert data["sparse_score"] == pytest.approx(-0.13)


# ---- 综述型问答的两条约束（第 21 周加）----


def test_max_per_source_caps_one_document_but_keeps_diversity():
    """不设上限时同一篇会占满 top-5（实测：5 条里 3 条来自同一篇综述）。

    构造：a.md 三段都命中 alpha 且稠密分最高，b.md / c.md 各一段。
    """
    corpus = [
        ("a.md", "alpha one"),
        ("a.md", "alpha two"),
        ("a.md", "alpha three"),
        ("b.md", "alpha beta"),
        ("c.md", "alpha gamma"),
    ]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.99, 0.0, 0.0, 0.0],
        [0.98, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0],
    ]
    retriever = _make_retriever(corpus, vectors, {"alpha": [1.0, 0.0, 0.0, 0.0]})

    unlimited = retriever.search("alpha", top_k=3)
    capped = retriever.search("alpha", top_k=3, max_per_source=1)

    assert [h.chunk.source_path for h in unlimited] == ["a.md", "a.md", "a.md"]
    assert [h.chunk.source_path for h in capped] == ["a.md", "b.md", "c.md"]


def test_max_per_source_returns_fewer_hits_when_documents_run_out():
    """上限生效时可能凑不满 top_k，**这是正确的**：宁可少给几条，
    也不要为了凑数把同一篇的第二段塞回来——综述层要按"实际找到几篇"来引用。"""
    corpus = [("a.md", "alpha one"), ("a.md", "alpha two"), ("b.md", "alpha three")]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]
    retriever = _make_retriever(corpus, vectors, {"alpha": [1.0, 0.0, 0.0, 0.0]})

    hits = retriever.search("alpha", top_k=5, max_per_source=1)

    assert [h.chunk.source_path for h in hits] == ["a.md", "b.md"]


def test_subtopic_filters_rows_before_ranking():
    """串台实测："研究现状"这类词会把相邻子主题的综述也拉进来，所以要能按子主题收窄。"""
    corpus = [
        ("刀具磨损监测/综述.pdf", "alpha 方法对比"),
        ("数字孪生与智能车间/综述.pdf", "alpha 研究现状"),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]]
    retriever = _make_retriever(corpus, vectors, {"alpha": [1.0, 0.0, 0.0, 0.0]})

    both = retriever.search("alpha", top_k=5)
    only = retriever.search("alpha", top_k=5, subtopic="数字孪生与智能车间")

    assert [h.chunk.source_path for h in both] == [
        "刀具磨损监测/综述.pdf",
        "数字孪生与智能车间/综述.pdf",
    ]
    assert [h.chunk.source_path for h in only] == ["数字孪生与智能车间/综述.pdf"]


def test_subtopic_unknown_returns_empty_not_whole_corpus():
    """子主题写错时返回空，**不静默退回全库**——否则"只在这批资料里找"就成了一句空话。"""
    corpus = [("刀具磨损监测/a.pdf", "alpha")]
    retriever = _make_retriever(corpus, np.eye(1, 4, dtype=np.float32), {})

    assert retriever.search("alpha", subtopic="不存在的子主题") == []


def test_unreadable_sources_are_excluded_by_default():
    """那 2 篇「仅存在性可检索」的论文（文字层是 `/G` 字形编号）默认不进结果。

    "只写在文档里"等于答题时没人执行，所以约束落在检索层，并留 `exclude_unreadable=False`
    这条复现路径。
    """
    bad = R.KNOWN_UNREADABLE_SOURCES[0]
    corpus = [(bad, "alpha 数字孪生"), ("刀具磨损监测/好论文.pdf", "alpha 刀磨损")]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]
    retriever = _make_retriever(corpus, vectors, {"alpha": [1.0, 0.0, 0.0, 0.0]})

    default_hits = retriever.search("alpha", top_k=5)
    with_bad = retriever.search("alpha", top_k=5, exclude_unreadable=False)

    assert [h.chunk.source_path for h in default_hits] == ["刀具磨损监测/好论文.pdf"]
    assert [h.chunk.source_path for h in with_bad] == [
        bad,
        "刀具磨损监测/好论文.pdf",
    ]
    assert len(retriever.unreadable_rows()) == 1


def test_reference_chunks_are_excluded_by_default():
    """参考文献列表段默认不进候选池（第 22 周实测被当候选两次）。

    它们没有正文内容，却因为"书目信息语义上跟很多查询都像"而拿到高分——
    所以约束落在检索层，并留 `exclude_references=False` 的复现路径。
    """
    bibliography = (
        "[1] CAI W, LIU F. Energy allowance[J]. Energy, 2016, 114: 623-633. "
        "[2] CHEN X Z. Cutting parameter optimization[J]. FME, 2021, 16(2): 221-248. "
        "[3] DIAZ N. Environmental impact[J]. Procedia CIRP, 2012, 1: 518-523. "
        "[4] HE Y. Energy consumption analysis[J]. Proc IMechE, 2012."
    )
    corpus = [("刀具/某篇.pdf", bibliography), ("刀具/综述.pdf", "alpha 正文内容")]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]
    retriever = _make_retriever(corpus, vectors, {"alpha": [1.0, 0.0, 0.0, 0.0]})

    default_hits = retriever.search("alpha", top_k=5)
    with_refs = retriever.search("alpha", top_k=5, exclude_references=False)

    assert [h.chunk.source_path for h in default_hits] == ["刀具/综述.pdf"]
    assert {h.chunk.source_path for h in with_refs} == {
        "刀具/某篇.pdf",
        "刀具/综述.pdf",
    }
    assert len(retriever.reference_rows()) == 1


def test_reference_page_majority_marks_the_whole_page():
    """文献表常被切碎（实测某页 10 个片段，每个只剩 1~2 条书目特征）→ 页级多数表决补齐。

    同一 `(source_path, locator)` 下**过半**片段被标记时，整页按参考文献处理；
    只有一两条时不补（避免把"正文末尾接文献表"的页整页误伤）。
    """
    bibliography = (
        "[1] CAI W, LIU F. Energy allowance[J]. Energy, 2016, 114: 623-633. "
        "[2] CHEN X Z. Cutting parameter optimization[J]. FME, 2021, 16(2): 221-248. "
        "[3] DIAZ N. Environmental impact[J]. Procedia CIRP, 2012, 1: 518-523. "
        "[4] HE Y. Energy consumption analysis[J]. Proc IMechE, 2012."
    )
    loose = "317 -331 . ［105］ZHU Z X ， XI X L ， XU X ， et al . Digital twin"
    corpus = [
        ("刀具/某篇.pdf", bibliography),
        ("刀具/某篇.pdf", bibliography),
        ("刀具/某篇.pdf", loose),
    ]
    retriever = Retriever(
        [make_chunk(name, text) for name, text in corpus],
        np.zeros((3, 4), dtype=np.float32),
        None,
    )

    # 两个被标记 + 一个没标记，同一页 → 过半 → 三个都算参考文献
    assert len(retriever.reference_rows()) == 3

    single = Retriever(
        [
            make_chunk("刀具/某篇.pdf", bibliography),
            make_chunk("刀具/某篇.pdf", "ok 正文"),
            make_chunk("刀具/某篇.pdf", loose),
        ],
        np.zeros((3, 4), dtype=np.float32),
        None,
    )
    assert len(single.reference_rows()) == 1  # 1/3 不过半，不补


# ---- 第 25 周：查询侧过滤（遗留 #4 / T27）----


def test_tokenize_query_drops_stopwords_and_single_chars():
    """查询侧比索引侧多丢两类词：虚词/疑问词，以及"多字词还够多"时的单字词。

    为什么值得加这道过滤：问题里满是"有哪些 / 的 / 是"，它们在 BM25 里会命中一大片文档，
    把真正有信息的那几个词挤下去。第 25 周实测（20 题、`data/index-notes`、k=8、离线）：
    混合检索的严格判据 **9/20 → 12/20**，跨文档 1/5 → 2/5、多跳 2/5 → 3/5。
    """
    tokens = R.tokenize_query("刀具磨损监测有哪些主流方法？")

    assert "哪些" not in tokens, "疑问词对区分度没贡献"
    assert all(len(tok) > 1 for tok in tokens)
    assert {"刀具", "磨损", "监测", "主流", "方法"} <= set(tokens)


def test_tokenize_query_keeps_single_chars_when_they_carry_the_meaning():
    """只剩一个多字词时就别丢单字——"轴"在这一句里是有信息的。

    注意 jieba 把"热误差"切成了 `热 / 误差`（这条断言是按**实测**的切法写的，
    不是按我猜的：第一版我断言 `"热误差" in tokens`，它挂了——猜切法就会这样）。
    """
    tokens = R.tokenize_query("轴的热误差")

    assert "轴" in tokens and "误差" in tokens
    assert "的" not in tokens


def test_tokenize_query_falls_back_when_everything_is_a_question_word():
    """ "是什么"这种句子过滤完就空了——退回不过滤，别把查询变成空串。"""
    assert R.tokenize_query("是什么"), "过滤成空等于这一轮检索白跑"


def test_custom_tokenizer_disables_query_side_filtering():
    """注入了自定义分词器时**不做**查询侧过滤。

    索引侧与查询侧必须用同一套切法：索引按一种切法建词袋、查询按另一种切，
    BM25 的词表就对不上，稀疏通道会**静默**变成"什么都没命中"——这种失败最费劲。
    """
    chunk = make_chunk("a.md", "刀具磨损 监测")
    retriever = Retriever(
        [chunk], np.zeros((1, 4), dtype=np.float32), None, tokenizer=str.split
    )

    assert retriever.query_tokenizer is str.split


def test_retrieval_depth_matches_the_measured_value():
    """默认检索深度是**实测调的**：k=5 → k=8 让混合检索的严格判据 7/20 → 9/20，
    再加上查询侧过滤到 12/20（三份原始输出见 `eval/第25周top_k对照*.md`）。

    钉住它是为了：以后有人凭直觉把它改回 5 时，先能看见这条注释与那三份数据。
    """
    from rag.review_qa import REVIEW_TOP_K

    assert R.DEFAULT_TOP_K == 8
    # 数值巧合相同，但两者**解耦**：一个是检索质量参数、一个是 prompt 成本参数
    # （T31 的教训：一个常量被两件事复用，改一处另一处跟着变却不知道）
    assert REVIEW_TOP_K == 8


def test_rerank_trigger_is_always_on_under_rrf_ties():
    """遗留 #3 的定论钉在这里：RRF **大量并列**，相邻分差常为 0，于是判据恒触发。

    这不是"阈值没调对"——RRF 分数是**排名组合**的函数，并列是常态，所以拿它做
    "分得开就不重排"的成本控制**不成立**（第 25 周定论，见 `rag/评测说明.md` 第七节）。
    行为**有意保留**（第 19 周实测重排值这个钱：严格判据 7/20 → 12/20）；
    这条测试的作用是：哪天有人改了 RRF 或阈值，它会提醒"这个判据的行为变了"，
    而不是让人以为成本控制忽然生效了。
    """
    from rag.rerank import should_rerank

    def hit(score: float):
        return SearchHit(chunk=make_chunk("a.md", "text"), score=score)

    need, smallest = should_rerank([hit(0.03), hit(0.03)])

    assert smallest == 0.0
    assert need is True, "并列（分差 0）时 0 < RERANK_GAP，判据必然触发"


def test_subtopics_counts_chunks_and_marks_root_files():
    corpus = [
        ("刀具磨损监测/a.pdf", "alpha"),
        ("刀具磨损监测/b.pdf", "beta"),
        ("note.md", "gamma"),
    ]
    retriever = Retriever(
        [make_chunk(name, text) for name, text in corpus],
        np.zeros((3, 4), dtype=np.float32),
        None,
    )

    assert retriever.subtopics() == {"刀具磨损监测": 2, "（根目录）": 1}


# ---- 子主题先验的回落策略（2026-10-25）----


def _two_subtopic_retriever():
    """两个子主题各一篇：用来验"先紧、不够再放宽"。"""
    parts = []
    for index in range(4):
        parts.append(f"刀具磨损监测/监测{index}.md")
        parts.append(f"切削参数优化/参数{index}.md")
    from rag.chunk import Chunk

    chunks = [
        Chunk.create(
            source_path=path,
            doc_type="md",
            locator="第 1 页",
            start=1,
            end=1,
            text="刀具 磨损 监测 切削 参数",
        )
        for path in parts
    ]
    vectors = np.zeros((len(chunks), 4), dtype=np.float32)
    return Retriever(chunks, vectors, None, tokenizer=str.split)


def test_search_scoped_keeps_the_tight_scope_when_enough_hits():
    """子主题内够数 → 用紧口径，并标 `scoped: true`。"""
    retriever = _two_subtopic_retriever()

    hits, meta = R.search_scoped(retriever, "刀具 磨损", 4, subtopic="刀具磨损监测")

    assert meta["scoped"] is True
    assert meta["fallback_reason"] is None
    assert hits and all(
        hit.chunk.source_path.startswith("刀具磨损监测/") for hit in hits
    )


def test_search_scoped_falls_back_and_says_why():
    """子主题内不够数 → 回落全局，**并且必须说清为什么**（否则调用方分不清口径）。"""
    retriever = _two_subtopic_retriever()

    hits, meta = R.search_scoped(
        retriever, "刀具 磨损", 4, subtopic="刀具磨损监测", min_keep=99
    )

    assert meta["scoped"] is False
    assert "少于 99 条" in (meta["fallback_reason"] or "")
    assert len(hits) == 4, "回落之后应当拿到全局的 top_k"


def test_search_scoped_without_subtopic_is_plain_search():
    """没给子主题就等价于普通检索（不标注、不回落）。"""
    retriever = _two_subtopic_retriever()

    hits, meta = R.search_scoped(retriever, "刀具", 3)

    assert meta["scoped"] is False and meta["fallback_reason"] is None
    assert len(hits) == 3


# ---- 加权口径（2026-10-26）----


def test_search_boosted_lifts_the_subtopic_without_dropping_others():
    """加权：同子主题的分数被乘上去，但**别的方向不会被拍掉**（与过滤的区别就在这里）。"""

    def hit(path: str, score: float):
        return R.SearchHit(
            chunk=make_chunk(path, "刀具"), score=score, dense_rank=1, sparse_rank=1
        )

    class StubRetriever:
        """桩检索器：直接回预设命中、**分数可控**。

        为什么不用迷你索引：那里向量全零、分数都是 0，乘任何系数还是 0——
        测不出"加权有没有生效"。第一版就是这么写的，它挂了，原因不是代码错，是**测试选错了数据**。
        """

        def search(self, query, top_k=5, **kwargs):
            # 切削那条原始分更高；加权 2.0 后刀具那条应当反超
            return [hit("切削参数优化/a.md", 0.04), hit("刀具磨损监测/b.md", 0.03)]

    hits, meta = R.search_boosted(
        StubRetriever(), "刀具 磨损", 4, subtopic="刀具磨损监测", factor=2.0
    )

    assert meta["boosted"] is True and meta["factor"] == 2.0
    assert hits[0].chunk.source_path.startswith("刀具磨损监测/"), (
        "加权后同子主题应当反超"
    )
    assert any(hit.chunk.source_path.startswith("切削参数优化/") for hit in hits), (
        "加权不排除别的方向"
    )


def test_search_boosted_without_subtopic_is_plain_search():
    """没给子主题时等价普通检索，并标 `boosted: false`。"""
    retriever = _two_subtopic_retriever()

    hits, meta = R.search_boosted(retriever, "刀具", 3)

    assert meta["boosted"] is False and meta["factor"] is None
    assert len(hits) == 3
