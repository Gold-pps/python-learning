"""MCP server：把 notes-qa 的能力做成**别的 Agent 能调用**的标准接口（第三阶段 H2）。

三个工具，按"花不花钱"分层——这是本模块最重要的设计：

| 工具 | 花钱 | 说明 |
| ---- | ---- | ---- |
| `search_corpus` | 否 | 本地混合检索（BM25 + 向量 + RRF）。别的 Agent **可以随便调** |
| `read_chunk` | 否 | 按 `文件:位置` 取回原文片段。与上一个配合：先搜、再读原文 |
| `review_question` | **是** | 完整四段式问答（2 次 deepseek 调用，约 2~7 分钱）。**默认关** |

为什么花钱的那个必须有闸门：前两个是我自己在用，而 MCP 是**别的 Agent 自动调用**——
它不知道"这次调用要花钱"，也不会替你省。所以闸门放在代码里（环境变量 + 次数上限 +
每次返回花费与累计），而不是写在工具描述里"请少调用"。

三条硬约束，都是本项目踩过的：

1. **不顶层 import fastembed**：CI 跑的是 `uv sync --frozen`（不带 `--extra rag`），
   顶层 import 会让 CI 全红。检索器懒加载——与 `web/server.py` 同一套做法（第 23 周结论）；
2. **传输只走 stdio**：本机子进程、不占端口、不出网络，与"不做云部署与公网访问"一致；
3. **用的是 mcp 2.x 的 API**：`FastMCP` 已改名 `MCPServer`
   （`from mcp.server.mcpserver import MCPServer`）。网上多数教程仍是 1.x 的
   `mcp.server.fastmcp` 写法，照抄会直接 `ModuleNotFoundError`——第 25 周探针实测过，
   迁移提示见 <https://py.sdk.modelcontextprotocol.io/v2/migration/>。

接线方法（Codex 的 `~/.codex/config.toml`）见同目录 `README.md`。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 把 `notes-qa` 自己接进 sys.path——**必须在下面的导入之前**。
#
# 为什么需要：`mcp dev mcp_server/server.py`（官方 Inspector）是用
# `importlib.util.spec_from_file_location` **按文件路径**加载这个文件的，
# 那种加载方式**不会把所在包目录加进 sys.path**，于是顶层的
# `from rag.retriever import ...` 直接 `ModuleNotFoundError: No module named 'rag'`
# （第三阶段 H2 第一次跑 Inspector 就撞上）。
#
# `python -m mcp_server` 与 pytest 都不需要这几行（一个是 cwd 在路径上、一个是 pyproject 的
# pythonpath），但多这几行**对它们无害**——已经在 sys.path 上就不重复插。
_NOTES_QA = Path(__file__).resolve().parents[1]
if str(_NOTES_QA) not in sys.path:
    sys.path.insert(0, str(_NOTES_QA))

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from rag.retriever import DEFAULT_TOP_K, KNOWN_UNREADABLE_SOURCES

# ---- 成本闸门 ----

SPEND_ENV = "NOTES_QA_MCP_ALLOW_SPEND"
"""开启花钱工具的环境变量。值写 `1` / `true` / `yes` 都算开启（大小写不敏感）。"""

MAX_SPEND_CALLS = 5
"""**单进程**内 `review_question` 的调用上限。

为什么要有上限：MCP 的调用方是 Agent，它可能在一次任务里反复问（"再多查几次"）。
闸门只管"开没开"，上限管"开了之后能花多少"——两个都要，缺一个都是敞口。
上限之外的调用**不报错**，而是返回一句说明（附已用次数），让 Agent 知道怎么办。
"""

SNIPPET_CHARS = 240
"""`search_corpus` 每条命中返回的片段长度。

**故意只给片段**：MCP 的返回会进调用方的上下文，把 600 字的候选整段塞回去会挤掉它自己的预算。
要看原文就调 `read_chunk`——这是两个工具配合的设计，不是省事。
"""

_spend_calls = 0  # 已发生的花钱调用次数（模块级：单进程一份）


def spend_allowed() -> bool:
    """闸门开了没有。"""
    return os.environ.get(SPEND_ENV, "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def spend_state() -> dict:
    """给工具返回值与自检用的闸门状态。"""
    return {
        "allowed": spend_allowed(),
        "env": SPEND_ENV,
        "used": _spend_calls,
        "max": MAX_SPEND_CALLS,
    }


def reset_spend_state() -> None:
    """清空计数（测试用；也便于使用者在同一进程里重新开始计数）。"""
    global _spend_calls
    _spend_calls = 0


# ---- 检索器：懒加载 + 可注入 ----

DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index"
"""默认索引位置：**按模块位置算，不看当前工作目录**。

MCP server 是被客户端**当子进程拉起**的，cwd 完全不由我们决定——
Inspector 会继承 `mcp dev` 的 cwd、Codex 那边更说不准。
早先这里写的是相对路径 `data/index`，从仓库根跑一次就去找
`python-learning/data/index`（不存在），报出来的是"索引不可用"，而**真索引就在旁边**——
这种错最容易把人引到错的方向去（台账 T48）。
"""

_retriever = None
_chunks_by_exact: dict[tuple[str, str], list] | None = None
_locator_canonical: dict[tuple[str, str], str] | None = None
_locators_by_file: dict[str, list[str]] | None = None


def set_retriever(retriever) -> None:
    """注入检索器（测试与"演示模式"用）。传 None 表示清空缓存。"""
    global _retriever, _chunks_by_exact, _locator_canonical, _locators_by_file
    _retriever = retriever
    _chunks_by_exact = None
    _locator_canonical = None
    _locators_by_file = None


def index_dir_from_env() -> Path:
    """索引目录：`NOTES_QA_INDEX` 优先（显式指定，按 cwd 解析），否则用模块旁边的默认值。"""
    env = os.environ.get("NOTES_QA_INDEX", "").strip()
    return Path(env).expanduser().resolve() if env else DEFAULT_INDEX_DIR


def _positive_int(value, default: int) -> int:
    """把客户端传来的参数收敛成正整数。

    MCP 客户端经常把**没填的可选参数发成 `null`**（Inspector 的工具表单就是这样）。
    早先这里直接 `int(top_k)`，于是 `int(None)` 抛 TypeError，客户端只看到
    `Error executing tool search_corpus`——它根本看不出"是自己漏填了一个字段"（台账 T48）。
    工具的参数是**外部**输入，宽松接住比严格报错合适。
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def get_retriever():
    """取检索器：没注入过就按 `NOTES_QA_INDEX`（默认 `notes-qa/data/index`）现场加载。

    **懒加载**：向量模型只在第一次真检索时才读进内存（约 1~3 秒）。

    **这段照抄 `web/server.py` 那份跑通的写法，不重写第二份**——第三阶段 H2 的第一版
    在这里写了个根本不存在的 `rag.index.load_index()`，9 条测试全绿、一接真的就炸：
    测试全都注入了假检索器，**这条路径没人走**（台账 T48）。
    加一条"走真加载路径"的测试就是为了堵这个口子。

    与 web 层唯一的差别：**后端从 `meta.json` 读**，不写死 `fastembed`——
    索引是用哪个后端建的，加载时就得用哪个（写死会让"用 bge 建的索引"静默对不上维度）。
    """
    global _retriever
    if _retriever is not None:
        return _retriever

    index_dir = index_dir_from_env()

    from web.server import read_index_meta  # 复用同一份"索引能不能用"的判断，不另写

    info = read_index_meta(index_dir)
    if not info["usable"]:
        hint = info.get("hint") or ""
        raise RuntimeError(
            f"索引不可用（{index_dir}）：{'、'.join(info['problems'])}。{hint}"
        )

    backend = "fastembed"
    meta_path = index_dir / "meta.json"
    if meta_path.exists():
        backend = (
            json.loads(meta_path.read_text(encoding="utf-8")).get("model") or backend
        )

    try:
        from rag.embedder import Embedder
    except ImportError as exc:  # pragma: no cover —— 环境问题
        raise RuntimeError(
            f"缺少向量化后端：{exc}（跑 `uv sync --extra rag`）"
        ) from exc

    from rag.retriever import Retriever

    _retriever = Retriever.from_index(index_dir, Embedder(backend=backend))
    return _retriever


_CN_DIGITS = {
    "〇": 0,
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def _cn_number_to_arabic(text: str) -> str:
    """把一段中文数字换成阿拉伯数字（只处理页码/行号这种小整数）。

    `八` → `8`、`二十三` → `23`、`一百零五` → `105`；解析不了就原样返回。
    """
    total = section = number = 0
    for char in text:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            section += (number or 1) * _CN_UNITS[char]
            number = 0
        else:  # 不是数字字符
            return text
    return str(total + section + number)


def _cn_numerals_to_arabic(text: str) -> str:
    """把串里每一段连续的中文数字都换成阿拉伯数字，其余字符原样保留。"""
    out: list[str] = []
    run: list[str] = []
    for char in text:
        if char in _CN_DIGITS or char in _CN_UNITS:
            run.append(char)
            continue
        if run:
            out.append(_cn_number_to_arabic("".join(run)))
            run = []
        out.append(char)
    if run:
        out.append(_cn_number_to_arabic("".join(run)))
    return "".join(out)


def _loose(text: str) -> str:
    """比对位置时的**宽松**写法：去空白 / 首尾括号、统一小写、**中文与全角数字转阿拉伯数字**。

    只用于比对；返回给调用方的永远是索引里的规范写法。三条宽松规则都是真踩出来的：

    - 空白（T49）：索引里是 `第 2 页`（半角空格），调用方会写 `第2页`；
    - **中文数字（T50）**：使用者会写 `第八页`，而索引里是 `第 8 页`——
      "手抄也能对上"这件事躲不开中文数字；
    - 全角数字（T50 同批）：中文输入法很容易打出 `第８页`（全角 ８）。

    与 `rag.review_qa._canon`（给"两条引用算不算同一条"用）是同一个思路，
    但那边只去半角空格——位置的写法比引用更随人，所以这里的规则更多，**但都只用于比对**。
    """
    stripped = "".join((text or "").split()).strip("[]（）()").casefold()
    return _cn_numerals_to_arabic(stripped.translate(_FULLWIDTH_DIGITS))


def _build_chunk_tables():
    """建三张查表（只建一次）。

    为什么是**三张**而不是一张 `(文件, 位置) → 片段`——两个都是真踩出来的（台账 T49）：

    1. **同一个位置会有多个片段**：真实索引里那个 PDF 每页 4~5 段，
       用单值字典会把其余几段**静默丢掉**（调用方以为读到了整页，其实只有一段）；
    2. **位置写法只认一种拼法**：索引里存的是 `第 2 页`（半角空格），
       而 `第2页` / `第 2页` / `第2 页` / `第八页` / `第８页` 全匹配不上——
       所以另存一张"宽松"的表（见 `_loose`），既能把调用方的写法**纠回规范写法**，
       又能在真拼错时给出可用的位置清单。
    """
    global _chunks_by_exact, _locator_canonical, _locators_by_file
    if _chunks_by_exact is None:
        exact: dict[tuple[str, str], list] = {}
        canonical: dict[tuple[str, str], str] = {}
        locators: dict[str, list[str]] = {}
        for chunk in get_retriever().chunks:
            exact.setdefault((chunk.source_path, chunk.locator), []).append(chunk)
            canonical.setdefault(
                (chunk.source_path, _loose(chunk.locator)), chunk.locator
            )
            seen = locators.setdefault(chunk.source_path, [])
            if chunk.locator not in seen:
                seen.append(chunk.locator)  # 去重：一个位置只列一次
        _chunks_by_exact = exact
        _locator_canonical = canonical
        _locators_by_file = locators
    return _chunks_by_exact, _locator_canonical or {}, _locators_by_file or {}


# ---- 建 server ----


def build_server(
    *, name: str = "notes-qa", plan_call=None, answer_call=None
) -> MCPServer:
    """建一个 MCP server。`plan_call` / `answer_call` 可注入（测试用假模型，不花钱）。"""
    server = MCPServer(
        name=name,
        title="notes-qa 资料问答",
        version="0.1.0",
        # **这一行是给 Inspector 用的**：`mcp dev` 会把这个列表拼进它拉起的子进程命令行
        # （`uv run --with mcp==X --with <这里> mcp run ...`）。少了 `mcp[cli]`，
        # 子进程里的 `mcp run` 直接报 `Error: typer is required` —— 界面上卡片显示 "Failed"，
        # 而那条错误跟 server 本身毫无关系（台账 T47）。
        # 生产路径（`python -m mcp_server`）不需要 typer，列在这里不影响它。
        dependencies=["mcp[cli]"],
        instructions=(
            "本地科研资料（数控 / 智能制造方向的论文）检索与问答。"
            "先用 search_corpus 找候选，再用 read_chunk 读原文；"
            "需要跨多篇组织答案时用 review_question——**它会花钱**，默认关闭。"
        ),
    )

    @server.tool(
        title="检索本地资料",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def search_corpus(
        query: str, top_k: int = DEFAULT_TOP_K, subtopic: str = ""
    ) -> dict:
        """在本地资料库里检索片段（混合检索：BM25 + 向量 + RRF）。

        只读、本地、**不花钱**，可以随便调用。

        Args:
            query: 自然语言问题或关键词，用中文或英文都行。
            top_k: 返回多少条，默认 8（第 25 周实测：8 比 5 严格判据高 2 题）。
            subtopic: 只在这个子主题里找（如"刀具磨损监测"）；留空表示全库。

        Returns:
            dict：`hits` 是命中列表（含 `citation`＝`文件:位置`，要原文请把它交给 read_chunk），
            外加本次检索的 `index` 与 `count`。
        """
        if not isinstance(query, str) or not query.strip():
            return {"error": "query 不能为空", "hits": [], "count": 0}
        retriever = get_retriever()
        hits = retriever.search(
            query.strip(),
            _positive_int(top_k, DEFAULT_TOP_K),
            subtopic=((subtopic or "").strip() or None),
        )
        return {
            "query": query.strip(),
            "count": len(hits),
            "hits": [
                {
                    "citation": f"{hit.chunk.source_path}:{hit.chunk.locator}",
                    "source_path": hit.chunk.source_path,
                    "locator": hit.chunk.locator,
                    "doc_type": hit.chunk.doc_type,
                    "score": round(float(hit.score), 6),
                    "channels": hit.channels,
                    "dense_rank": hit.dense_rank,
                    "sparse_rank": hit.sparse_rank,
                    "snippet": hit.chunk.text[:SNIPPET_CHARS],
                }
                for hit in hits
            ],
            "hint": "要看某条命中的原文，把它的 citation 交给 read_chunk。",
        }

    @server.tool(
        title="读原文片段",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def read_chunk(source_path: str, locator: str = "") -> dict:
        """按 `文件:位置` 取回**原文全文**（检索只给 240 字片段，这里给这一处的完整文本）。

        只读、本地、**不花钱**。三条已经踩过的规矩（台账 T49）：

        - **位置写法宽松**：`第2页` / `第 2 页` / `第 2页` / `第八页` / `第８页` 都能命中 `第 8 页`——
          索引里存的是阿拉伯数字加半角空格，但位置是**手抄**出来的，不该因为空格、
          中文数字或全角数字就失败；返回里的 `locator` 一律是索引里的**规范写法**，下次照抄它就不会错；
        - **一个位置可能有多段**：PDF 同一页往往切成 4~5 个片段，这里**一次全给**（按顺序拼接），
          并用 `parts_count` 说明拼了几段——早先只返回最后一段，调用方以为读到了整页；
        - **位置对不上时给可用的清单**，别只说"没有"。

        Args:
            source_path: 文件相对路径，如 `刀具磨损监测/某篇.pdf`。
            locator: 位置，如 `第 4 页`；**留空会列出该文件有哪些位置**（全是去重后的）。

        Returns:
            dict：命中时给 `text`（原文）与 `citation`；对不上时给 `available` 与 `hint`。
        """
        exact, canonical, locators = _build_chunk_tables()
        source_path = (source_path or "").strip()
        locator = (locator or "").strip()

        if not source_path:
            return {"error": "source_path 不能为空"}
        if source_path in KNOWN_UNREADABLE_SOURCES:
            return {
                "citation": source_path,
                "error": (
                    "这篇 PDF 的文字层缺 ToUnicode 映射，抽出来是字形编号（如 /G21），"
                    "引用不出可读原文——它只算“仅存在性可检索”，检索时也被默认排除。"
                ),
            }
        if source_path not in locators:
            return {
                "error": f"资料库里没有这个文件：{source_path}",
                "hint": "先用 search_corpus 检索，从 hits 里取 source_path。",
            }
        if not locator:
            return {
                "source_path": source_path,
                "available": locators[source_path][:20],
                "hint": "把这个文件里想读的位置填进 locator；一页可能有多段，读一次会给全。",
            }

        parts = exact.get((source_path, locator))
        if parts is None:
            # 拼法不同（多/少空格、全角空格、大小写）→ 宽松再试一次，并纠回规范写法
            canonical_locator = canonical.get((source_path, _loose(locator)))
            if canonical_locator is not None:
                parts = exact.get((source_path, canonical_locator))
                locator = canonical_locator
        if not parts:
            return {
                "error": f"{source_path} 里没有位置 `{locator}`",
                "available": locators[source_path][:20],
                "hint": "位置要取自 search_corpus 的 hit.locator（或本返回的 available），原样照抄。",
            }

        text = "\n\n".join(chunk.text for chunk in parts)
        result = {
            "citation": f"{source_path}:{locator}",
            "source_path": source_path,
            "locator": locator,
            "doc_type": parts[0].doc_type,
            "parts_count": len(parts),
            "text": text,
        }
        if len(parts) > 1:
            result["note"] = (
                f"这个位置有 {len(parts)} 段（PDF 一页常被切成多段），已按顺序拼成上面的 text。"
            )
        return result

    @server.tool(
        title="综述问答（会花钱）",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
    )
    def review_question(question: str, multihop: bool = True) -> dict:
        """跨多篇组织一份四段式答案（研究现状 / 方法对比 / 结论 / 可复用点），逐条挂引用。

        **这个工具会花钱**（一次约 2~7 分钱，最多 2 次模型调用），所以默认关闭：
        需要先设环境变量 `NOTES_QA_MCP_ALLOW_SPEND=1` 启动本 server。
        单进程最多调用 `MAX_SPEND_CALLS` 次（默认 5），超出会返回说明而不是继续花钱。

        Args:
            question: 自然语言问题（"这个方向有哪些主流方法、各自优缺点"这类）。
            multihop: 是否允许补检一轮（多一次模型调用，默认允许）。

        Returns:
            dict：`sections` 是四段内容（每条结论带引用）、`citations` 是引用校验计数
            （**编造必须为 0**）、`cost` 是本次花费、`spend` 是本进程的累计情况。
        """
        state = spend_state()
        if not isinstance(question, str) or not question.strip():
            return {"error": "question 不能为空"}
        if not state["allowed"]:
            return {
                "error": "综述问答默认关闭（它会调用 DeepSeek，产生费用）。",
                "how_to_enable": (
                    f"在启动本 server 的环境里设置 {SPEND_ENV}=1 再重试；"
                    "只读的 search_corpus / read_chunk 不受影响。"
                ),
                "spend": state,
            }
        if state["used"] >= MAX_SPEND_CALLS:
            return {
                "error": (
                    f"本进程的付费调用已用完（{state['used']}/{MAX_SPEND_CALLS}）。"
                    "重启 server 或调大上限后继续。"
                ),
                "spend": state,
            }

        # 复用 Web 层那条链路，**不写第二份实现**（T43 的教训：两份实现必然漂移）
        from web.service import ask

        global _spend_calls
        _spend_calls += 1
        view = ask(
            question.strip(),
            get_retriever(),
            multihop=bool(multihop),
            plan_call=plan_call,
            answer_call=answer_call,
        )
        cost = view.get("cost") or {}
        return {
            "question": view.get("question"),
            "retrieval_query": view.get("retrieval_query"),
            "sections": view.get("sections"),
            "insufficient": view.get("insufficient"),
            "note": view.get("note"),
            "citations": view.get("citations"),
            "candidates": [
                {
                    "citation": cand.get("citation"),
                    "source_path": cand.get("source_path"),
                    "locator": cand.get("locator"),
                }
                for cand in view.get("candidates") or []
            ],
            "cost": {
                "usd": cost.get("usd"),
                "cny": cost.get("cny"),
                "peak": cost.get("peak"),
            },
            "elapsed": view.get("elapsed"),
            "degraded": view.get("degraded"),
            "reason": view.get("reason"),
            "spend": spend_state(),
            "hint": "要看某条引用的原文，把 citation 交给 read_chunk。",
        }

    return server


server = build_server()
"""模块级 server：`python -m mcp_server` 用的就是它。测试请用 `build_server()` 另建。"""


def main(argv: list[str] | None = None) -> int:
    """stdio 启动入口。`--check` 只自检（不启动服务、不加载向量模型）。"""
    import argparse

    parser = argparse.ArgumentParser(description="notes-qa 的 MCP server（stdio）")
    parser.add_argument("--check", action="store_true", help="只自检并退出")
    parser.add_argument(
        "--allow-spend",
        action="store_true",
        help=(
            f"开启会花钱的 review_question（等价于环境变量 {SPEND_ENV}=1）。"
            "MCP 客户端的配置里未必支持 env 键，所以留了这个开关——把闸门写进 args 更直白"
        ),
    )
    args = parser.parse_args(argv)

    if args.allow_spend:
        # 设成环境变量而不是另走一条分支：`spend_allowed()` 只有这一个判据，
        # 两处判断必然漂移（T41 的教训）
        os.environ[SPEND_ENV] = "1"

    if args.check:
        index_dir = index_dir_from_env()
        parts = ["chunks.jsonl", "vectors.npz", "meta.json"]
        missing = [p for p in parts if not (index_dir / p).exists()]
        report = {
            "index_dir": str(index_dir),
            "index_ready": not missing,
            "missing": missing,
            "spend": spend_state(),
            "tools": ["search_corpus", "read_chunk", "review_question"],
            "transport": "stdio",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if not missing else 1

    server.run()  # transport 默认 stdio（不占端口、不出网络）
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
