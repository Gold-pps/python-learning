# AGENTS.md

本文件写给进入本仓库的 AI 助手（Codex 等）。开始工作前请先读这里。

> 最后更新：2026-10-12（第 23 周完成：Web 界面 + 跨机可用，**G4 五条已具备证据**；下一步第 24 周，G4 验收 + 全阶段复盘）

## 仓库是什么

个人 Python 学习仓库，主攻 **Agent 开发**。

- 模型后端：**DeepSeek API**（所有请求发往 `https://api.deepseek.com`，不使用 OpenAI 服务）；
- 开发框架：**OpenAI Agents SDK**（仅作为本地开发框架）；
- 模型名统一用 `deepseek-flash`，**不要写 `gpt-*`**；
- 默认关闭 tracing（`set_tracing_disabled(disabled=True)`），不开 OpenAI 云端追踪；
- Python 要求 **≥3.12**（`main.py` 的 `asyncio.timeout` 需 3.11+，
  `eval_cases.py` 的 `Path.is_junction()` 需 3.12+），本机实际使用 **3.13.14**；
- 前端冒烟需要 **node**（可选）：没装时 `tests/test_web_frontend.py` 会自动 skip，不算失败。

## 当前进度

- 第一阶段 12 周已完成（2026-09-13 收尾）；
- 第二阶段（第 13~24 周）：**已完成**，四组目标 G1~G4 全部达成；
- 第三阶段（第 25~36 周）：**已规划**（见下），主题是"从自己用的工具到别人与别的 Agent 都能用的能力"：
  H1 结清地基 → H2 **MCP server** → H3 让结论可被独立核对 → H4 交付与复盘；
- 恢复上下文请先读 `agent-learning/学习进度.md`；
- 计划与验收标准见 `agent-learning/第二阶段学习规划.md`（已完成）与
  `agent-learning/第三阶段学习规划.md`（第 25~36 周；H2 要用 `mcp` 包，
  **注意 mcp 2.x 把 `FastMCP` 改名成了 `MCPServer`**，网上多数教程还是 1.x 写法）；
- 技术债台账见 `agent-learning/工程化改造记录.md`（T1~T43）。

## 目录结构

```text
python-learning/
├── AGENTS.md                  本文件（写给 AI 助手）
├── README.md                  仓库门面（给人看）
├── pyproject.toml             项目元数据 + pytest 配置（依赖真相来源）
├── uv.lock                    依赖锁文件
├── ruff.toml                  ruff 配置（有意独立于 pyproject，理由见文件头）
├── requirements.txt           说明文件，指向 pyproject.toml
├── .github/workflows/ci.yml   push / PR 自动跑 ruff + pytest
├── .codex/skills/
│   └── notes-qa-eval/SKILL.md 自定义 Skill：跑 notes-qa 评估
├── agent-learning/            Agent 学习主目录（★ 同时是 notes-qa 的笔记根目录）
│   ├── 00~07_*.py             第 1~7 周练习脚本（9 个；03 有"多轮记忆 / 无历史"两份）
│   ├── _common.py             周脚本共用的 DeepSeek 接入样板
│   ├── 学习Agent规划.md        第一阶段 12 周计划
│   ├── 第二阶段学习规划.md     第 13~24 周计划与 G1~G4 验收标准
│   ├── 学习进度.md             当前进度（**恢复上下文先读它**）
│   ├── 笔记索引.md             每周主题 / 笔记 / 产出对照
│   ├── 学习评审报告.md          2026-09-15 评审 + 处理状态
│   ├── 工程化改造记录.md        技术债台账（T1~T43）
│   ├── 第1~23周笔记.md         每周笔记（21 篇；第 15、19 周并入台账 T15/T16/T26~T30）
│   ├── 第12周演示稿.md         10 分钟组会汇报底稿
│   ├── .env                   API Key（**不要提交**）
│   └── notes-qa/              第 8 周起项目：个人资料问答助手
│       ├── 项目设计.md
│       ├── config.py          DeepSeek + SDK 配置
│       ├── tools.py           list_notes / search_notes / read_note
│       ├── agent.py           notes_qa Agent 定义
│       ├── main.py            交互式命令行入口
│       ├── eval_cases.py      本地评估（12 单元 + 10 模型）
│       ├── exp_history_tokens.py  T3 会话历史 token 曲线实验
│       ├── smoke_test.py      环境自检
│       ├── constants.py       无依赖共享常量（断开 rag 与 agents 的依赖链）
│       ├── rag/              第 16 周起：RAG（解析 / 切分 / 向量化 / 索引）
│       │   ├── chunk.py       Chunk 数据结构
│       │   ├── parsers.py     .md / .txt / .pdf / .docx 解析
│       │   ├── chunker.py     中文友好切分
│       │   ├── embedder.py    双后端（bge / fastembed）
│       │   ├── index.py       构建 + 增量跳过 + CLI
│       │   ├── retriever.py   混合检索（BM25 稀疏 + 向量稠密 + RRF + 每篇上限/子主题过滤）
│       │   ├── rerank.py      LLM 重排（严格 JSON + 失败降级 + 记账）
│       │   └── review_qa.py   综述问答（四段式 + JSON Output + 引用校验，第 21 周）
│       ├── eval/              第 18 周起：抽查集 / 离线对比 / 建库报告 / 综述题与覆盖表
│       ├── web/               第 23 周起：Web 界面（FastAPI + SSE + 单页前端）
│       │   ├── server.py      应用 + 启动入口 + 自检（/api/ask 走 SSE 阶段级进度流）
│       │   ├── service.py     编排：复用 answer_review()，补每步耗时 / 成本 / 引用↔候选
│       │   ├── pricing.py     成本估算（价格快照 + 峰谷判定 + 缓存命中单价）
│       │   ├── static/index.html  单页前端（原生 HTML/JS，无框架、无构建、无 CDN）
│       │   └── 使用说明.md     使用说明 + 能力边界声明 + 数据外发说明
│       ├── mcp_server/        MCP server（第三阶段 H2）：search_corpus / read_chunk / review_question
│       │   ├── server.py      三个工具 + 成本闸门（review_question 默认关）+ stdio 入口 + --check
│       │   └── README.md      怎么跑、怎么接进 Codex、四条踩过的约束
│       ├── data/             语料与索引（不进 Git）
│       └── tests/             pytest 用例 + conftest.py + 前端冒烟（frontend_smoke.mjs）
└── .venv/                     Python 虚拟环境（不提交）
```

> 仓库根原有的 4 个 `.pptx`、PPT 工具与产物、`python-base/` 基础练习，
> 已于 2026-09-22 移出仓库到 `~/python-learning-extras-20260922/`，并做了历史重写
> （`.git` 从 113 MB 降到 588 KB）。仓库内仍可能出现旧路径引用，以本节为准。

## 常用命令

```bash
# 环境（首次或换机后）
uv sync                # 按 uv.lock 安装到 .venv（含 dev 组 pytest / ruff）
uv sync --no-dev       # 只要运行依赖

# 测试与检查（仓库根）
uv run pytest -q       # 220 passed, 1 skipped（skip 是 Windows 非管理员平台限制）
uv run ruff check .    # All checks passed

# 前端 DOM 冒烟（第 23 周起；不联网、不花钱；没装 node 时 pytest 里会自动 skip）
# 它把 index.html 的内联脚本放进迷你 DOM 里跑：进度条渲染、引用点开看原文、SSE 分帧到渲染
node agent-learning/notes-qa/tests/frontend_smoke.mjs

# Web 界面（第 23 周起；在 agent-learning/notes-qa 下；界面本身不花钱，提问才花 token）
uv run python -m web.server --check                  # 自检：索引三件套 / fastembed / .env
uv run python -m web.server                          # 本机 http://127.0.0.1:8000
uv run python -m web.server --host 0.0.0.0 --token "notes-qa-2026"   # 局域网；非回环必须给口令
# 口令**只能可见 ASCII**：HTTP 头按 ISO-8859-1 传，中文口令会让浏览器直接拒绝发请求（台账 T40）

# MCP server（第三阶段 H2；在 agent-learning/notes-qa 下；stdio，不占端口、不出网络）
uv run python -m mcp_server --check      # 自检：索引三件套 / 成本闸门 / 工具清单（不加载模型）
uv run python -m mcp_server              # stdio 启动（由 MCP 客户端当子进程拉起）
# 会花钱的 review_question 默认关：$env:NOTES_QA_MCP_ALLOW_SPEND = "1" 才开（单进程上限 5 次）
# 注意 mcp 2.x 已把 FastMCP 改名成 MCPServer（from mcp.server.mcpserver import MCPServer），
# 网上多数教程还是 1.x 写法，照抄会 ModuleNotFoundError；接线方法见 mcp_server/README.md
# RAG 环境（独立 .venv-rag，需 torch；在 agent-learning/notes-qa 下）
# Windows：
& "..\..\.venv-rag\Scripts\python.exe" -m rag.index build
# Linux / macOS：
../../.venv-rag/bin/python -m rag.index build

# 第 20 周：建库前先预检（只解析不编码，主 .venv 即可）
uv run python -m rag.index scan
# 一键全量重建 + 对齐校验 + 建库报告
& "..\..\.venv-rag\Scripts\python.exe" -m rag.index build --backend fastembed --all --verify --report eval\建库报告.md
# 索引健康检查（抽查向量对齐，防 T29 那类静默错位）
& "..\..\.venv-rag\Scripts\python.exe" -m rag.index verify --sample 8

# 混合检索 / 第 19 周四组离线对比（不花钱，同样用 .venv-rag）
& "..\..\.venv-rag\Scripts\python.exe" -m rag.retriever --query "小批量为什么不用 GPU" --baseline
& "..\..\.venv-rag\Scripts\python.exe" -m eval.rag_eval --index data/index-notes --out eval/第19周对比表.md
# 回答级评测：引用准确率 / 无答案拒答率（**会花 token**，用户自己跑）
& "..\..\.venv-rag\Scripts\python.exe" -m eval.rag_answer_eval --index data/index-notes --out eval/第19周回答级评测.md
# LLM 重排（**会花 token**，由用户自己在低谷时段跑）
& "..\..\.venv-rag\Scripts\python.exe" -m rag.rerank --query "..."

# 综述型问答（第 21 周；主 .venv 即可，需 `uv sync --extra rag`，不装 torch）
uv run python -m eval.review_coverage --out eval/第21周检索冒烟.md   # 检索侧覆盖（离线、不花钱）
uv run python -m rag.review_qa --query "刀具磨损监测有哪些主流方法？" --out eval/第21周综述问答.md --log data/usage_log.jsonl  # **会花 token**（日志进 data/，不进 Git）

# 项目：交互式问答
cd agent-learning/notes-qa
uv run python main.py                # /new 清空历史，/exit 退出

# 评估
uv run python eval_cases.py          # 完整（12 单元 + 10 模型，需联网、消耗 token）
uv run python eval_cases.py --unit-only   # 只跑单元测试（无需网络）
```

## 不要做的事

- 不要把 API Key 写进代码或提交到 Git（Key 只放在 `agent-learning/.env`）；
- 不要使用 OpenAI 服务、不要写 `gpt-*` 模型名、不要开启 tracing；
- 不要放宽 `notes-qa` 工具的笔记根目录边界（`agent-learning/` 之外的路径一律拒绝）；
- 不要把 `notes-qa/` 当作笔记内容读取（已在 `EXCLUDE_DIRS` 中排除）；
- 不要在用户未明确要求时执行 `git commit` / `git push` / `git push --force`；
- 不要用 `git checkout --` / `git restore` 清临时改动——它是文件级的，会连带冲掉该文件
  所有未提交改动，而且不报错（台账 T9 的现场）。清临时改动用 `apply_patch`；
- 不要删除 `.codex/`（存的是项目级 Skill）；
- **模型评估由用户在自己时段跑**（会花 token）。AI 侧只跑 `--unit-only` 与离线测试；
- **无答案题的关键词不能写进 `agent-learning/*.md`**——那等于把答案卡交给被检索的 Agent
  （台账 T8"发现一"）。题库与答案卡只写在 `notes-qa/` 内（该目录已在 `EXCLUDE_DIRS` 中）。

- **Web 层的改动必须同时跑前端冒烟**：`node agent-learning/notes-qa/tests/frontend_smoke.mjs`。
  这层没有构建步骤，冒烟是唯一能挡住"后端全绿、界面没画出来"的网
  （第 23 周同一块功能连坏两次，T37/T39）。改 `index.html` 里用到的字段时，
  **夹具要与真实数据字段一致**——T43 就是夹具里手写了 `doc_type`、真数据没有它，测试全绿而界面错；
- **界面上的话不许说"做了但实际没做"的事**：横幅/计数这类展示字段必须来自计算结果，
  不能由"开关开着"推断（T41：界面说"带上了上一问"，日志里根本没拼）；
- **给使用者的示例必须能直接用**：口令、URL、命令都会被原样照抄，
  中文占位符害过一次（T40：HTTP 头按 ISO-8859-1 传，中文口令让请求发不出去）；
- **Windows 上开发**：命令用 PowerShell，超过三行的脚本写成临时 `.py` 文件再跑
  （`python -c "长字符串"` 的引号会被 PowerShell 吃掉）；
- **跑 pytest 需要管理员权限**（或开 Windows 开发者模式）：普通用户建不了文件符号链接，
  会有 1 条 skip。这是平台限制，不是代码 bug。

## 关于沙箱网络（2026-09-22 修正）

`AGENTS.md` 此前写"不要在 Codex 沙箱里跑模型评估——沙箱没有网络"。
第 13 周实测：沙箱内 `https://api.deepseek.com` 可达（`codex doctor` 的到达性探针 +
8 次真实调用全部成功）。**真正的约束不是"没网络"，而是"会花 token，所以要带超时、
要限量、要记账"**。离线任务仍优先用 `uv run pytest -q` / `--unit-only`。