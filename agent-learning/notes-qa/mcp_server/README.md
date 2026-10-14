# notes-qa 的 MCP server（第三阶段 H2）

把本项目的检索与问答能力做成**别的 Agent 能调用**的标准接口（MCP），
让 Codex 这类客户端直接查这批论文资料，而不必经过网页界面。

## 三个工具（按"花不花钱"分层）

| 工具 | 花钱 | 做什么 | 协议注解 |
| ---- | ---- | ------ | -------- |
| `search_corpus(query, top_k=8, subtopic="")` | 否 | 混合检索（BM25 + 向量 + RRF），返回命中与 **`文件:位置` 引用**、只给 240 字片段 | `read_only_hint=True`、`open_world_hint=False` |
| `read_chunk(source_path, locator="")` | 否 | 按 `文件:位置` 取回**原文全文**；`locator` 留空则列出该文件有哪些位置（去重）。位置写法宽松（`第2页` = `第 2 页`），返回的是**规范写法**；一个位置有多个片段时**一次全给**并报 `parts_count` | 同上 |
| `review_question(question, multihop=True)` | **是** | 四段式综述答案（研究现状 / 方法对比 / 结论 / 可复用点）+ 引用校验计数 + 花费 | `open_world_hint=True`（会调外部 API） |

**为什么分层**：前两个是我自己在用的本地检索，别的 Agent **可以随便调**；
第三个一次要花 2~7 分钱（最多 2 次模型调用），而**自动调用它的不是人**——
它不会替你省钱，所以闸门必须在代码里。

## 怎么跑

在本目录（`agent-learning/notes-qa`）下：

```powershell
uv run python -m mcp_server --check    # 自检：索引三件套 / 闸门状态 / 工具清单（不启动、不加载模型）
uv run python -m mcp_server            # stdio 启动（不占端口；由客户端当子进程拉起）
```

**成本闸门**（默认关）：

```powershell
$env:NOTES_QA_MCP_ALLOW_SPEND = "1"    # 开启 review_question
uv run python -m mcp_server
```

- 闸门关着时 `review_question` **一次模型请求都不发**，返回里说明怎么开启；
- 开着也有上限：单进程最多 `MAX_SPEND_CALLS`（默认 5，见 `server.py`）；
- 每次调用返回 `cost`（本次花费）与 `spend`（本进程累计），调用方能看见钱。

索引位置默认是**本目录旁边的** `data/index`（按模块位置算，**不看当前工作目录**——
MCP server 是被客户端拉起的子进程，cwd 不由我们决定）；用 `NOTES_QA_INDEX` 可指向别的索引
（如 `data/index-notes`，那时按你给的路径解析）。

> **报错看不到原因？** MCP 层只把一句通用的 `Error executing tool <名字>` 发给客户端，
> **异常原文留在服务端**（台账 T48）。所以排查时：先跑 `--check`，再把出问题的工具函数
> 直接拿来调一遍（`uv run pytest tests/test_mcp_server.py -q` 里有现成的调用方式），
> 别对着那句通用错误猜。

## 接线到 Codex

在 `~/.codex/config.toml` 里加一段（沿用第 11 周接 filesystem MCP 的写法）：

```toml
[mcp_servers.notes_qa]
command = "uv"
args = [
  "--directory", "c:/Users/czy/Desktop/github-repositories/python-learning/agent-learning/notes-qa",
  "run", "python", "-m", "mcp_server",
]
# 想让 Codex 也能用会花钱的 review_question，再加这一行：
# env = { NOTES_QA_MCP_ALLOW_SPEND = "1" }
```

- 路径用**正斜杠**（TOML 里反斜杠要转义，没必要自找麻烦）；
- 改完要**重开会话**才加载（第 11 周踩过：MCP 加载失败时 Codex 会说"本会话没有加载任何 MCP server"）；
- 先用 `--check` 确认索引齐了再接线，省得把"索引没建"当成"MCP 配错了"。

## 手动调（不想接线时）

官方 Inspector（临时装 CLI，不进项目依赖）：

```powershell
uv run --with "mcp[cli]" mcp version
uv run --with "mcp[cli]" mcp dev mcp_server/server.py --with "mcp[cli]"
```

**结尾那个 `--with "mcp[cli]"` 不是多余的**：`mcp dev` 会**另起一个子进程**跑你的 server，
而它拼命令行时只带上 `mcp==<版本>` 加上 server 对象自己声明的依赖。少了 `mcp[cli]`，
子进程里的 `mcp run` 会直接报 `Error: typer is required`，界面上卡片显示 **Failed**——
那条错误跟你的 server 毫无关系（台账 T47）。
`server.py` 里 `dependencies=["mcp[cli]"]` 就是为此声明的（有了它，即使忘了这个 flag 也能起来）。

Inspector 会开一个网页，可以逐个工具点着调。**注意**：Inspector 认的是文件里创建的
server 对象（本文件里叫 `server`）；如果它认不出来，就直接跑上面的 `--check`
再用 `python -m mcp_server` 手动接一个 stdio 客户端。

> **踩过的坑（台账 T46）**：`mcp dev` 是按**文件路径**用 `importlib` 加载这个文件的，
> 那种加载**不会把所在目录加进 `sys.path`**——所以 `server.py` 顶部有一段 sys.path 自举
> （长得像"多余的样板"，其实不是）。第一次跑 Inspector 就是被这个炸的：
> `ModuleNotFoundError: No module named 'rag'`。别把那几行删掉。

## 五条约束（都踩过，别再踩）

1. **不顶层 import fastembed**：CI 跑 `uv sync --frozen`（**不带** `--extra rag`），
   顶层 import 会让 CI 全红。检索器在 `get_retriever()` 里懒加载——与 `web/server.py` 同款；
2. **只用 stdio**：本机子进程、不占端口、不出网络，与"不做云部署与公网访问"一致；
3. **API 是 mcp 2.x**：`FastMCP` 已改名 `MCPServer`
   （`from mcp.server.mcpserver import MCPServer`）。网上多数教程还是 1.x 的
   `mcp.server.fastmcp` 写法，照抄会直接 `ModuleNotFoundError`（第 25 周探针实测，
   迁移提示：<https://py.sdk.modelcontextprotocol.io/v2/migration/>）；
4. **返回值要省着花**：这是喂给**别的 Agent 的上下文**。所以检索只给片段、候选只给引用，
   要原文请走 `read_chunk`——两个工具是配合设计，不是图省事；
5. **`sys.path` 要自举**：`mcp dev`（Inspector）按**文件路径**用 `importlib` 加载，
   不会把 `notes-qa` 加进 `sys.path`——所以 `server.py` 顶部那几行看着像样板，其实是必需的，
   **别删**（第一次跑 Inspector 就是被这个炸的，台账 T46）。

## 测试

```powershell
uv run pytest agent-learning/notes-qa/tests/test_mcp_server.py -q -p no:cacheprovider
```

8 条，全部离线（假检索器 + 假模型调用，不联网、不花钱）：协议注解、
只读工具能用、闸门默认拦得住（**用"一调就炸"的假调用断言**，不是看返回值像不像拒绝）、
开了能跑且报花费、超上限停在说明上。
