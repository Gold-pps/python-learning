"""MCP server（`mcp_server/`）的离线测试——不联网、不加载 fastembed、不花 token。

覆盖三件事，按"踩过的坑"排：

1. **只读边界写进了协议**：`search_corpus` / `read_chunk` 的 `ToolAnnotations.read_only_hint=True`、
   `open_world_hint=False`；而 `review_question` 因为要调 DeepSeek，`open_world_hint=True`。
   这类字段必须**断言**——写在代码里没人看，写进协议才对调用方有意义；
2. **成本闸门是真闸门**：默认关闭时**一次模型调用都不发**（用会抛异常的假调用断言，
   而不是"看返回值像不像拒绝"）；开启后有次数上限，超了返回说明而不是继续花钱；
3. **工具之间能配合**：检索只给截断片段（省调用方的上下文），`read_chunk` 给全文——
   所以要用一段**长文本**来验证"检索截断、读原文完整"。

跑法：`uv run pytest agent-learning/notes-qa/tests/test_mcp_server.py -q -p no:cacheprovider`
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _web_fakes import FakeRetriever, answer_reply, make_calls, make_hit
from mcp_server import server as mcp

QUESTION = "刀具磨损监测有哪些方法？"
HIT_A = "刀具磨损监测/综述A.pdf"
HIT_B = "刀具磨损监测/研究B.pdf"
LONG_TEXT = (
    "原文" * 260
)  # 520 字：比 SNIPPET_CHARS（240）长，用来验证"检索截断、读原文完整"


def payload(result) -> dict:
    """从 `CallToolResult` 里取出工具返回的那个 dict。

    两种形态都要认：返回 dict 时 `structured_content` 就是它；返回 str 时会被包成 `{"result": ...}`。
    """
    data = result.structured_content
    if isinstance(data, dict) and set(data) == {"result"}:
        return data["result"]
    if data is None:  # pragma: no cover —— 兜底：从文本内容里解析
        return json.loads(result.content[0].text)
    return data


def call(server, name: str, arguments: dict) -> dict:
    """同步调一个工具（`call_tool` 是协程，测试里不必到处 await）。"""
    return payload(asyncio.run(server.call_tool(name, arguments)))


def explode(*_args, **_kwargs):
    """假模型调用：**一旦被调用就抛异常**——用来证明闸门真的拦住了。"""
    raise AssertionError("闸门没拦住：不该发生的模型调用")


@pytest.fixture(autouse=True)
def clean_state():
    """每个用例都要有干净的检索器与闸门计数（模块级状态，不隔离会互相污染）。"""
    mcp.set_retriever(
        FakeRetriever(
            fallback=[
                make_hit(HIT_A, LONG_TEXT, locator="第 2 页", start=2),
                make_hit(HIT_B, "短片段", locator="第 5 页", start=5),
            ]
        )
    )
    mcp.reset_spend_state()
    yield
    mcp.set_retriever(None)
    mcp.reset_spend_state()


# ---- 1. 协议层：三个工具与它们的只读/外部注解 ----


def test_tools_are_registered_with_the_right_annotations():
    server = mcp.build_server()
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}

    assert set(tools) == {"search_corpus", "read_chunk", "review_question"}

    for name in ("search_corpus", "read_chunk"):
        annotations = tools[name].annotations
        assert annotations is not None, f"{name} 应带注解"
        assert annotations.read_only_hint is True, f"{name} 只读，要写进协议"
        assert annotations.open_world_hint is False, f"{name} 不出网"

    # 花钱的那个：真调外部 API，所以 open_world_hint 为真——这是给调用方看的信号
    assert tools["review_question"].annotations.open_world_hint is True


# ---- 2. 只读工具：能用、不花钱 ----


def test_search_corpus_returns_citations_and_never_calls_the_model():
    server = mcp.build_server(plan_call=explode, answer_call=explode)

    result = call(server, "search_corpus", {"query": "刀具磨损", "top_k": 2})

    assert result["count"] == 2
    first = result["hits"][0]
    assert first["citation"] == f"{HIT_A}:第 2 页"
    assert first["doc_type"] == "pdf"
    assert first["channels"] and first["score"] > 0
    # 检索**只给片段**：整段 520 字塞回调用方的上下文会挤掉它自己的预算
    assert len(first["snippet"]) == mcp.SNIPPET_CHARS
    assert "read_chunk" in result["hint"]


def test_search_corpus_rejects_blank_query_without_crashing():
    server = mcp.build_server()

    assert call(server, "search_corpus", {"query": "   "})["error"]


def test_read_chunk_returns_the_full_text_not_the_snippet():
    server = mcp.build_server()

    result = call(server, "read_chunk", {"source_path": HIT_A, "locator": "第 2 页"})

    assert result["citation"] == f"{HIT_A}:第 2 页"
    assert result["text"] == LONG_TEXT, "读原文要给全文（检索那边只有截断的 snippet）"
    assert len(result["text"]) > mcp.SNIPPET_CHARS


def test_read_chunk_without_locator_lists_what_is_available():
    server = mcp.build_server()

    listed = call(server, "read_chunk", {"source_path": HIT_A})
    assert listed["available"] == ["第 2 页"]

    wrong = call(server, "read_chunk", {"source_path": HIT_A, "locator": "第 99 页"})
    assert "没有位置" in wrong["error"]
    assert wrong["available"] == ["第 2 页"], "位置对不上时要给出可用的位置"

    missing = call(server, "read_chunk", {"source_path": "不存在的.pdf"})
    assert "没有这个文件" in missing["error"]


# ---- 2b. 位置写法与"一页多段"（T49：Inspector 里真撞到的两个问题）----


def test_read_chunk_tolerates_spacing_in_locator():
    """位置写法宽松：索引里是 `第 2 页`，调用方填 `第2页` 也要命中，并**纠回规范写法**。

    真实现场：在 Inspector 里填 `第2页`（没打空格）→ 报"里没有位置"。调用方是照着引用
    **手抄**位置的，多一个少一个空格不该是"找得到 / 找不到"的分水岭（台账 T49）。
    """
    server = mcp.build_server()

    result = call(server, "read_chunk", {"source_path": HIT_A, "locator": "第2页"})

    assert result.get("error") is None, result
    assert result["locator"] == "第 2 页", "返回的必须是索引里的规范写法（下次照抄它）"
    assert result["citation"] == f"{HIT_A}:第 2 页"
    assert result["text"] == LONG_TEXT


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("八", "8"),
        ("十", "10"),
        ("十八", "18"),
        ("二十三", "23"),
        ("一百零五", "105"),
        ("一千二百三十四", "1234"),
    ],
)
def test_chinese_numerals_convert_to_arabic(raw, expected):
    """中文数字要能换成阿拉伯数字（`第八页` 是使用者真写过的，台账 T50）。

    只处理页码/行号这种小整数——够用就行，不必做一个通用中文数字库。
    """
    assert mcp._cn_number_to_arabic(raw) == expected


def test_loose_handles_whitespace_chinese_and_fullwidth():
    """三种写法都要归一到同一个宽松键：空格、中文数字、全角数字。"""
    canonical = mcp._loose("第 8 页")

    assert mcp._loose("第8页") == canonical
    assert mcp._loose("第八页") == canonical
    assert mcp._loose("第８页") == canonical
    assert mcp._loose("第 八 页") == canonical
    # 不是数字的串别被改坏（md 候选的位置长这样）
    assert mcp._loose("行 12-30") == "行12-30"


def test_read_chunk_accepts_chinese_and_fullwidth_page_numbers():
    """索引里是 `第 8 页`，调用方写 `第八页` / `第８页` 也要命中（T50）。"""
    mcp.set_retriever(
        FakeRetriever(
            fallback=[make_hit(HIT_A, "第八页的内容。", locator="第 8 页", start=8)]
        )
    )
    server = mcp.build_server()

    for written in ("第八页", "第８页", "第 8 页"):
        result = call(server, "read_chunk", {"source_path": HIT_A, "locator": written})
        assert result.get("error") is None, f"{written} 应当命中：{result}"
        assert result["locator"] == "第 8 页", "返回的必须是索引里的规范写法"
        assert result["text"] == "第八页的内容。"


def test_read_chunk_returns_every_part_of_a_locator():
    """**同一个位置有多段时要一次全给**（PDF 一页常被切成 4~5 段）。

    早先用 `(文件, 位置) → 片段` 的单值字典，只留最后一段、其余静默丢掉——
    调用方以为读到了整页。这是 T49 里比"空格"更严重的那个（丢内容）。
    """
    mcp.set_retriever(
        FakeRetriever(
            fallback=[
                make_hit(HIT_A, "第一段内容。", locator="第 2 页", start=2),
                make_hit(HIT_A, "第二段内容。", locator="第 2 页", start=2),
                make_hit(HIT_A, "第三段内容。", locator="第 3 页", start=3),
            ]
        )
    )
    server = mcp.build_server()

    result = call(server, "read_chunk", {"source_path": HIT_A, "locator": "第 2 页"})

    assert result["parts_count"] == 2, "同一位置的两段都要返回"
    assert "第一段内容。" in result["text"] and "第二段内容。" in result["text"]
    assert "第三段内容。" not in result["text"], "别的页不要混进来"
    assert "2 段" in result["note"], "要说明这是拼起来的，别让人以为是原文顺序"


def test_read_chunk_lists_unique_locators():
    """位置清单要去重：一个位置只列一次（早先同一页重复列好几遍，全是噪声）。"""
    mcp.set_retriever(
        FakeRetriever(
            fallback=[
                make_hit(HIT_A, "一段。", locator="第 2 页", start=2),
                make_hit(HIT_A, "又一段。", locator="第 2 页", start=2),
                make_hit(HIT_A, "另一页的。", locator="第 3 页", start=3),
            ]
        )
    )
    server = mcp.build_server()

    listed = call(server, "read_chunk", {"source_path": HIT_A})

    assert listed["available"] == ["第 2 页", "第 3 页"]


# ---- 3. 花钱的工具：默认关、开了有上限 ----


def test_review_question_is_off_by_default_and_spends_nothing(monkeypatch):
    monkeypatch.delenv(mcp.SPEND_ENV, raising=False)
    server = mcp.build_server(plan_call=explode, answer_call=explode)

    result = call(server, "review_question", {"question": QUESTION})

    assert mcp.SPEND_ENV in result["how_to_enable"], "要告诉调用方怎么开"
    assert result["spend"]["used"] == 0
    assert mcp.spend_state()["used"] == 0


def test_review_question_runs_when_the_gate_is_open(monkeypatch):
    monkeypatch.setenv(mcp.SPEND_ENV, "1")
    plan_call, answer_call = make_calls(
        answer_reply("间接法更适合在线监测", f"{HIT_A}:第 2 页")
    )
    server = mcp.build_server(plan_call=plan_call, answer_call=answer_call)

    result = call(server, "review_question", {"question": QUESTION})

    assert result["sections"]["研究现状"], "四段式要有内容"
    assert result["citations"]["exact"] == 1
    assert result["citations"]["invalid"] == [], "编造必须为 0"
    assert result["cost"]["usd"] > 0, "花了钱就要报出来"
    assert result["spend"] == {
        "allowed": True,
        "env": mcp.SPEND_ENV,
        "used": 1,
        "max": mcp.MAX_SPEND_CALLS,
    }
    # 候选只给"怎么去读原文"，不带正文——正文由 read_chunk 负责
    assert result["candidates"][0]["citation"] == f"{HIT_A}:第 2 页"
    assert "text" not in result["candidates"][0]


# ---- 4. 真加载路径（这一组是本轮补的：上面那些都注入假检索器，那条路没人走）----


class _FakeEmbedder:
    """确定性假编码器：与 `test_retriever.py` 里那个同款（主 .venv 里就能跑）。"""

    backend = "fake"
    dim = 8

    def __init__(self, backend: str | None = None):
        self.backend = backend or "fake"

    def encode(self, texts: list[str]) -> np.ndarray:
        import numpy as np

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.array(
            [[float(len(text) % 7)] * self.dim for text in texts], dtype=np.float32
        )


def test_get_retriever_really_loads_an_index(tmp_path, monkeypatch):
    """**不注入检索器**，走一遍真加载：读 `NOTES_QA_INDEX` → 读 meta → `Retriever.from_index`。

    这条就是为 T48 补的：此前 9 条测试全都 `set_retriever(假的)`，于是
    `get_retriever()` 里那行 `from rag.index import load_index`（**一个不存在的函数**）
    一路绿到了 Inspector。测试里"能注入"是好事，但**至少得有一条走真路径**。
    """
    from rag.index import build

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "# 标题\n" + "混合检索使用 RRF 融合稀疏与稠密两路结果。" * 12,
        encoding="utf-8",
        newline="\n",
    )
    index_dir = tmp_path / "index"
    build(corpus, index_dir, _FakeEmbedder(), verbose=False)

    monkeypatch.setenv("NOTES_QA_INDEX", str(index_dir))
    # 把向量模型换成假的：这里要验的是"加载接线对不对"，不是"模型好不好"
    monkeypatch.setattr("rag.embedder.Embedder", _FakeEmbedder)
    mcp.set_retriever(None)

    retriever = mcp.get_retriever()

    assert retriever.chunks, "真加载应当读到片段"
    assert retriever.vectors.shape[0] == len(retriever.chunks)
    assert mcp.get_retriever() is retriever, "第二次调用应当命中缓存（模型只加载一次）"


def test_default_index_path_does_not_depend_on_cwd(monkeypatch):
    """默认索引位置**按模块算，不看 cwd**——MCP server 是被客户端拉起的，cwd 不由我们决定。

    早先写的是相对路径 `data/index`：从仓库根跑就去找 `python-learning/data/index`（不存在），
    报"索引不可用"，而真索引就在旁边（台账 T48）。
    """
    monkeypatch.delenv("NOTES_QA_INDEX", raising=False)

    path = mcp.index_dir_from_env()

    assert path.is_absolute()
    assert path.parts[-3:] == ("notes-qa", "data", "index")
    assert path.exists(), "默认位置应当真是本项目的索引目录"


def test_null_optional_arguments_are_rejected_by_the_protocol_layer():
    """把没填的字段发成 `null`，**协议层在调用工具之前就按 schema 拒了**，回一条点名参数的 ToolError。

    这条钉的是"**MCP 层替我们挡了 null**"这个事实——H2 排查时一度以为 Inspector 那个 Tool Error
    是 `int(None)` 引起的，其实那是探针**直接调函数**才绕过了校验（台账 T48）。
    所以下面那个收敛函数是为"直接调用"准备的第二道，不是主路径。
    """
    from mcp.server.mcpserver.exceptions import ToolError

    server = mcp.build_server()

    with pytest.raises(ToolError, match="top_k"):
        asyncio.run(
            server.call_tool("search_corpus", {"query": "刀具磨损", "top_k": None})
        )


def test_a_crashing_tool_hides_its_reason_from_the_client():
    """工具内部抛异常时，客户端只拿到一句**通用**的 `Error executing tool ...`。

    这条是 H2 花时间最多的那个坑的"官方说明"：`ImportError: cannot import name 'load_index'`
    被 MCP 层换成了 "Error executing tool search_corpus"，**原因留在服务端**——
    所以排查时应当去看服务端日志，或者**把工具函数直接拿来调**（台账 T48）。
    """
    from mcp.server.mcpserver.exceptions import UnexpectedToolError

    class Broken(FakeRetriever):
        @property
        def chunks(self):
            raise RuntimeError("我是内部原因，不该出现在客户端面前")

    mcp.set_retriever(Broken())
    server = mcp.build_server()

    with pytest.raises(UnexpectedToolError) as caught:
        asyncio.run(server.call_tool("read_chunk", {"source_path": "a.pdf"}))

    assert "内部原因" not in str(caught.value), "异常原文不该传给客户端"
    assert "read_chunk" in str(caught.value)


def test_argument_coercion_falls_back_instead_of_raising():
    """第二道（直接调用时才走得到）：把参数收敛成正整数，而不是抛出去。

    `None` / 非数字 / 0 一律回落到默认值——工具的参数是**外部输入**，
    宁可给一个合理的默认，也不该让调用方收到一句看不出所以然的异常。
    """
    assert mcp._positive_int(None, 8) == 8
    assert mcp._positive_int("5", 8) == 5
    assert mcp._positive_int(0, 8) == 8
    assert mcp._positive_int("abc", 8) == 8


def test_stdio_handshake_lists_the_three_tools():
    """真起一个子进程走 stdio：`initialize` → `tools/list`。

    前面几条测的是"工具函数对不对"，这条测的是"**MCP 客户端能不能跟它说上话**"——
    入口、`__main__`、协议握手这些东西只有真起进程才验得到（`--check` 也不行，
    它在 `server.run()` 之前就 return 了）。
    """
    notes_qa = Path(__file__).resolve().parents[1]
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "mcp_server"],
        cwd=notes_qa,
        input="".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )

    assert proc.returncode == 0, f"stdio 起不来：{proc.stderr[-500:]}"
    replies = {}
    for line in proc.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in message:
            replies[message["id"]] = message

    assert "result" in replies.get(1, {}), "initialize 没回成功"
    tools = {tool["name"]: tool for tool in replies[2]["result"]["tools"]}
    assert set(tools) == {"search_corpus", "read_chunk", "review_question"}
    # 注解在**协议线上是驼峰**（Python 侧是蛇形）——客户端看到的是这个
    assert tools["search_corpus"]["annotations"]["readOnlyHint"] is True
    assert tools["review_question"]["annotations"]["openWorldHint"] is True


def test_review_question_stops_at_the_call_limit(monkeypatch):
    """开了闸门也要有上限：调用方是 Agent，它会在一次任务里反复问。"""
    monkeypatch.setenv(mcp.SPEND_ENV, "1")
    monkeypatch.setattr(mcp, "MAX_SPEND_CALLS", 1)
    plan_call, answer_call = make_calls(answer_reply("有方法", f"{HIT_A}:第 2 页"))
    server = mcp.build_server(plan_call=plan_call, answer_call=answer_call)

    first = call(server, "review_question", {"question": QUESTION})
    assert first["spend"]["used"] == 1

    # 第二次：换成"一调就炸"的假调用，证明它**没有**再花钱
    server = mcp.build_server(plan_call=explode, answer_call=explode)
    second = call(server, "review_question", {"question": QUESTION})

    assert "用完" in second["error"]
    assert second["spend"]["used"] == 1, "被拦下的调用不该计数"
