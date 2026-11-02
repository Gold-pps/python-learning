# notes-qa 的 MCP server（第三阶段 H2）

把本项目的检索与问答能力做成**别的 Agent 能调用**的标准接口（MCP），
让 Codex 这类客户端直接查这批论文资料，而不必经过网页界面。

> **用之前先知道一件事**：`search_corpus` / `read_chunk` 给的是**原料**（命中 + 原文），
> 最终答案由调用方的 Agent 自己组织——**那些引用没有经过我们的对账**
> （`validate_citations` 只作用在 `review_question` / Web 界面产出的四段式答案上）。
> 要"引用 0 编造"这个保证，就得用**会花钱的** `review_question`。
> 2026-10-19 第一次真接线时就出现了这个差别：Codex 用前两个工具自己写了一篇带 6 条引用的答案，
> 读起来很顺，但那 6 条是它自己抄的、没人对账。

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

在 `~/.codex/config.toml` 里加一段（沿用第 11 周接 filesystem MCP 的写法：`command` + `args`）：

```toml
[mcp_servers.notes_qa]
command = "uv"
args = [
  "--directory", "C:/Users/czy/Desktop/github-repositories/python-learning/agent-learning/notes-qa",
  "run", "python", "-m", "mcp_server",
]
```

- 路径用**正斜杠**（TOML 里反斜杠要转义，没必要自找麻烦）；
- **这条命令已实测**：从一个无关的 cwd（`%TEMP%`）发起 `uv --directory <notes-qa> run python -m mcp_server`，
  发 `initialize` + `tools/list` 能拿回三个工具——**MCP 客户端拉子进程时不给 shell、cwd 也不由我们决定**，
  所以特意用 `--directory` 把目录写死（这也是 T48 那条"默认路径别看 cwd"的同一个道理）；
- 改完要**重开会话**才加载（第 11 周踩过：MCP 加载失败时 Codex 会说"本会话没有加载任何 MCP server"）；
- 先用 `--check` 确认索引齐了再接线，省得把"索引没建"当成"MCP 配错了"。

**若 Codex 里报 uv 缓存权限错**（实测见过一次：`failed to open file ...uv\cache\...\.git: 拒绝访问`，
它的沙箱会挡掉 uv 缓存里的写入），换这条**不依赖 uv** 的等价命令——同样实测过、同样在无关 cwd 下握手成功：

```toml
[mcp_servers.notes_qa]
command = "C:/Users/czy/Desktop/github-repositories/python-learning/.venv/Scripts/python.exe"
args = [
  "-c",
  "import sys; sys.path.insert(0, r'C:/Users/czy/Desktop/github-repositories/python-learning/agent-learning/notes-qa'); from mcp_server.server import main; raise SystemExit(main())",
]
```

两条命令的区别只在"谁来准备环境"：前者让 `uv` 按 `--directory` 找项目，后者直接用项目的 venv、
并由 `-c` 那段把 `notes-qa` 塞进 `sys.path`（**cwd 无关**，这正是 T48 的教训）。

**想让 Codex 也能用会花钱的 `review_question`**：在 `args` 末尾加 `"--allow-spend"`（推荐），
或者在配置里加 `env = { NOTES_QA_MCP_ALLOW_SPEND = "1" }`——两者等价，
但前者不依赖客户端支持 `env` 键：

```toml
args = [
  "--directory", "C:/Users/czy/Desktop/github-repositories/python-learning/agent-learning/notes-qa",
  "run", "python", "-m", "mcp_server",
  "--allow-spend",        # 会花钱的那个工具只在加了它之后才可用（单进程上限 5 次）
]
```

## 接线到 DSH（DeepSeek Harness，2026-11-02 实测）

DSH 官方发行版**自带** `@deepseek-ai/dsh-mcp-client`，不用另装插件，只加一条配置。

**配置文件**（改之前先备份，只往末尾追加）：

```
$DSH_HOME/profiles/<profile>/cordis.patch.yml
```

`DSH_HOME` 默认是 `~/.dsh`；桌面端的 profile 叫 `desktop`，`npx dsh web` 那个叫 `web`，
改哪个取决于你平时开哪个。文件是**顶层 YAML 数组**，原有的条目一条都不要删。

```yaml
# notes-qa 的 MCP server（本地 stdio：不占端口、不出网络）
- insert:
    - id: mcp-notes-qa
      name: "@deepseek-ai/dsh-mcp-client"
      config:
        serverName: notes_qa
        transport: stdio
        command: C:/Users/czy/Desktop/github-repositories/python-learning/.venv/Scripts/python.exe
        args:
          - -c
          - 'import sys; sys.path.insert(0, r"C:/Users/czy/Desktop/github-repositories/python-learning/agent-learning/notes-qa"); from mcp_server.server import main; raise SystemExit(main())'
```

三点说明：

1. **`- insert:` 不能省**（第一次就是栽在这）：顶层直接写 `- id: xxx` 是**按 id 覆盖已有条目**，
   新 id 找不到对应项就被**静默忽略**——不报错、不生效，表现只是"工具一直不出现"。
   新增条目必须包在 `insert` 列表里。
2. **`command` 用 venv 里的 python.exe，不用 `uv run`**：DSH 的 shell 沙箱里 uv 缓存
   （`%LOCALAPPDATA%\uv\cache`）可能写不进去，而 venv 里的解释器不需要缓存；
   `-c` 那段自己把 `notes-qa` 塞进 `sys.path`，因此**与 cwd 无关**（同 Codex 那条的道理，见 T46）。
3. **`serverName` 决定工具名前缀**，接上是 `mcp__notes_qa__search_corpus` 这种形式；改它等于改工具名。

**改完要完全退出 DSH 再启动**（配置只在启动时读），然后**没有"工具栏"可点**——MCP 工具不进界面，
只在对话里可见。验证方法是在会话里直接点名：

> 用 `mcp__notes_qa__search_corpus` 搜“NC 代码生成与校验的具体方法”，top_k 8。不要读代码、不要写脚本，只调工具。

消息里出现 `mcp__notes_qa__search_corpus` 的工具卡片即为接通。**第一次调用要加载向量模型**，
约 10 秒；之后在同一进程里很快。

**两个容易误判的现象**（都是 DSH 沙箱策略，与本 server 无关）：

| 现象 | 原因 |
| ---- | ---- |
| DSH 里 `uv run` 起不来（uv 缓存被拒） | 缓存目录在 `%LOCALAPPDATA%`，不在它允许写入的范围；换 venv python 即可 |
| 在 `agent-learning/notes-qa` 里写/删文件被拒 | 沙箱 shell 是低完整性，该目录是 Medium；删文件在普通 PowerShell 里做 |

**想让 DSH 也能用 `review_question`**：在 `config` 下加（不建议——客户端会自动反复调用，
单进程 5 次很快用完，而且那层"结论对不对"的保证它仍然没有）：

```yaml
        env:
          NOTES_QA_MCP_ALLOW_SPEND: '1'
```

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
