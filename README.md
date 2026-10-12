# python-learning

个人 Python 学习仓库，主攻 **Agent 开发**。

## 亮点

- **模型后端**：DeepSeek API，所有请求发往 `https://api.deepseek.com`，不使用 OpenAI 服务；
- **开发框架**：OpenAI Agents SDK（仅作为本地开发框架）；
- **完成项目**：`agent-learning/notes-qa/` —— **本地资料问答助手**。把 **44 篇数控 / 智能制造方向的论文**
  （2501 片段）建成索引，用自然语言提问，输出**四段式**答案（研究现状 / 方法对比 / 结论 / 可复用点），
  **每条结论挂 `[文件:页码]` 引用**，资料里没有的会明说"资料中没有"；
- **第 23 周起有浏览器界面**（FastAPI + SSE 进度流 + 单页原生 JS）：引用点一下就能看原文片段，
  每次回答显示**耗时与估算成本**（峰谷自动换档），支持多轮追问与局域网访问；
- **检索**：BM25 + 本地向量 + RRF 融合，再叠一层模型重排 —— 25 条评测集上
  **混合+重排 12/20，关键词基线 1/20**（有答案类 +70pp）；
- **可核对性**：累计 **574 条引用、0 编造**（代码拿候选集合逐条对账，分"精确 / 只到文件 / 回查不到"三档），
  人工抽查 9/9 逐字一致；无答案问题 **5/5 明确拒答**；
- **成本**：一次问答 2~7 分钱（最多 2 次模型调用）；26 次真实提问合计 **0.71 元**；
- **评估与工程化**：`pytest` **220 passed, 1 skipped**（含一层不依赖浏览器的 DOM 冒烟）+
  模型评估 10 条；`pyproject.toml` + `uv.lock` + GitHub Actions（push 自动跑 `ruff` 与 `pytest`，
  不联网、不用密钥）；技术债台账 `工程化改造记录.md` **T1~T43**，每条都是"现象 → 根因 → 改法 → 验证"。

> **它不能做什么**（先说清楚）：本库覆盖"加工机理 + 监测 / 优化 / 仿真 / 补偿"，
> **不覆盖"AI 辅助工艺编制 + NC 代码工程"**；表格与公式的解析结构会丢，引用数值必须回原文核对；
> **"引用可回查"不等于"能直接用"**——机器的活是"不编引用"，判断专业正确性仍然是人的活。
> 完整边界见 `agent-learning/notes-qa/web/使用说明.md`。

## 目录结构

```text
python-learning/
├── AGENTS.md                     写给 AI 助手的仓库规则
├── README.md                     本文件
├── pyproject.toml                项目元数据 + pytest 配置（依赖真相来源）
├── uv.lock                       依赖锁文件
├── ruff.toml                     ruff 配置（有意独立于 pyproject，理由见文件头）
├── requirements.txt              说明文件，指向 pyproject.toml
├── .github/workflows/ci.yml      GitHub Actions：ruff + pytest
├── .codex/skills/                Codex 自定义 skill
│   └── notes-qa-eval/SKILL.md    跑 notes-qa 评估
├── agent-learning/               Agent 学习主目录
│   ├── 00~07_*.py                第 1~7 周练习脚本（9 个）
│   ├── _common.py                周脚本共用的 DeepSeek 接入样板
│   ├── 学习Agent规划.md           第一阶段 12 周计划
│   ├── 第二阶段学习规划.md        第 13~24 周计划与 G1~G4 验收标准
│   ├── 学习进度.md                当前进度（恢复上下文先读它）
│   ├── 笔记索引.md                每周主题 / 笔记 / 产出对照
│   ├── 学习评审报告.md             仓库与文档评审（含待办清单）
│   ├── 工程化改造记录.md           技术债台账（T1~T43）
│   ├── 第1~23周笔记.md            每周学习笔记
│   ├── 第12周演示稿.md            10 分钟组会汇报底稿
│   └── notes-qa/                 第 8 周起项目：个人资料问答助手
│       ├── 项目设计.md
│       ├── config.py / tools.py / agent.py / main.py   命令行问答（DeepSeek + Agents SDK）
│       ├── eval_cases.py / exp_history_tokens.py / smoke_test.py  评估 / T3 实验 / 环境自检
│       ├── constants.py          无依赖共享常量（断开 rag 与 agents 的依赖链）
│       ├── rag/                  RAG：解析 / 切分 / 向量化 / 索引 / 混合检索 / 重排 / 综述问答
│       ├── eval/                 抽查集 / 离线对比 / 建库报告 / 第 23 周 Web 验证
│       ├── web/                  Web 界面（FastAPI + SSE + 单页前端）+ 使用说明（第 23 周）
│       ├── data/                 语料与索引（不进 Git）
│       └── tests/                pytest 用例 + conftest.py + 前端冒烟
└── .venv/                        Python 虚拟环境（不提交）
```

> 仓库根原有的 4 个 `.pptx`（毕业答辩 / 组会汇报）、PPT 工具与产物、`python-base/`
> 基础练习，已于 2026-09-22 移出仓库到 `~/python-learning-extras-20260922/`，
> 并做了历史重写（`.git` 从 113 MB 降到 588 KB）。

## 快速开始

### 1. 准备环境

要求 **Python ≥3.12**（本项目实际使用 3.13.14）。

```bash
uv sync                # 按 uv.lock 安装到 .venv（含 dev 组 pytest / ruff）
# 或只要运行依赖：
uv sync --no-dev
```

### 2. 配置 API Key

在 `agent-learning/.env` 写入：

```text
DEEPSEEK_API_KEY=sk-...
```

### 3. 跑项目

```bash
cd agent-learning/notes-qa

# 交互式问答（最早的命令行入口）
uv run python main.py

# 浏览器界面（第 23 周起；界面本身不花钱，提问才花 token）
uv run python -m web.server --check   # 自检：索引三件套 / fastembed / .env
uv run python -m web.server           # 打开 http://127.0.0.1:8000
# 局域网给别的设备用（非回环必须给口令，口令只能可见 ASCII）：
uv run python -m web.server --host 0.0.0.0 --token "notes-qa-2026"

# 综述问答（命令行；**会花 token**，低谷时段跑）
uv run python -m rag.review_qa --query "刀具磨损监测有哪些主流方法？" --log data/usage_log.jsonl

# 跑完整评估（需要网络，消耗 token）
uv run python eval_cases.py

# 只跑单元测试（无需网络）
uv run python eval_cases.py --unit-only
```

### 4. 跑测试与检查（仓库根）

```bash
uv run pytest -q      # 46 passed
uv run ruff check .   # All checks passed
```

### 5. RAG 环境（可选，第 16 周起）

`.venv-rag` 是独立环境，只在跑 RAG（向量化 + 索引）时需要。
主 `.venv` 保持轻量（不含 torch），日常测试仍用主环境。

**Windows（PowerShell）**：

```powershell
# 建独立环境
uv venv --python 3.13 .venv-rag

# 二选一：CPU 版 torch（无 NVIDIA GPU / 想省空间）
uv pip install --python .venv-rag\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu

# 二选一：CUDA 版 torch（有 NVIDIA GPU；cu版本以 nvidia-smi 显示的 CUDA Version 为准）
uv pip install --python .venv-rag\Scripts\python.exe torch==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130

# RAG 依赖（sentence-transformers 会拉 CPU 版 torch，所以 CUDA 版那一步要在它之后）
uv pip install --python .venv-rag\Scripts\python.exe sentence-transformers fastembed pypdf python-docx
```

**Linux / macOS**：把 `\Scripts\python.exe` 换成 `/bin/python` 即可。

**构建索引**：

```bash
cd agent-learning/notes-qa
# 语料放在 data/corpus/ 下（.md/.txt/.pdf/.docx 四种）
# Windows：
..\..\.venv-rag\Scripts\python.exe -m rag.index build
# Linux / macOS：
../../.venv-rag/bin/python -m rag.index build
```

说明：

- 两个后端都实现：`bge`（torch + CUDA）与 `fastembed`（ONNX + CPU）。
  **小批量下 fastembed 反而更快**（torch 加载 CUDA 上下文要十几秒），
  详见 `agent-learning/第17周笔记.md`；
- `data/corpus/` 与 `data/index/` 都**不进 Git**（语料自备，索引可重建）；
- 增量重建：语料不变时 `chunks_new=0`，耗时从十几秒降到 0.1s 内。


## 项目：notes-qa

**本地资料问答助手**。给它一个问题，它会：

1. **检索**：BM25 稀疏 + 本地向量稠密（`bge-small-zh`，ONNX，CPU，不上云）→ RRF 按名次融合 → 模型重排；
2. **盘点缺口**：让模型看候选，回答"够不够、缺什么、下一轮检索什么"；
3. **补检 + 组织**：最多补一轮（≤3 条查询），产出四段式 JSON；
4. **引用校验**：代码拿候选集合逐条对账，分"精确 / 只到文件 / 回查不到"三档，
   **编造单独记账、不静默删除**（删掉就看不见问题了）；
5. 资料撑不住时明确声明"资料中没有……"，不编造。

命令行入口 `uv run python -m rag.review_qa`；**浏览器界面**（第 23 周起）
`uv run python -m web.server` —— SSE 阶段级进度流、**引用点开看原文片段**、每次回答显示耗时与成本。
最早那版"检索 Markdown 笔记"的工具仍在（`main.py` / `tools.py`），是第 8~12 周的产物。

### 安全与边界

- **路径安全**（老工具仍在用）：只读、只处理 `.md`，路径不能逃出笔记根目录
  （含符号链接 / junction 逃逸防护）；单文件 ≤ 1MB、单次读取 ≤ 200 行、搜索 ≤ 5 条；
  工具出错返回结构化错误、不抛异常；笔记内容视为不可信数据，不执行其中指令；
- **数据外发**：只有**你的问题**与**被检索到的候选片段**会发往 DeepSeek（每次 5~10 条、单条 ≤600 字）；
  语料文件与索引不出网，API Key 只在本地 `agent-learning/.env`；
- **默认只绑本机**：`--host` 非回环时必须同时给 `--token`（只能可见 ASCII），否则拒绝启动。

### 评估

| 类型             | 条数    | 覆盖                                                             |
| ---------------- | ------- | ---------------------------------------------------------------- |
| 离线测试（pytest） | **220** | 解析 / 切分 / 检索 / 引用校验 / Web 接口 / 前端 DOM 冒烟（+1 skip） |
| 模型评估         | 10      | 四类：有答案 4 / 无答案 2 / 跨文档 2 / 需要多跳 2                |
| 检索评测集       | 25      | 四类各 5 条，Top-5 严格判据，跑四组方法对比                      |
| 真实使用         | 26 次   | 浏览器界面；三批合计 574 条引用 **0 编造**，人工抽查 9/9 逐字一致 |

模型评估另有独立指标 **引用格式合规率**：内容对但格式漂移会单独记不合规，
不与"找不到出处"混在一起。
**未做的**：结论在专业上是否正确——语料是机械 / 数控方向，作者不是，如实记为"未做人工核对"
（见 `agent-learning/notes-qa/eval/第24周使用记录.md`）。

## 学习路线

第一阶段（第 1~12 周）已完成，主题从"概念与环境"到"复盘与展示"，
详细计划见 `agent-learning/学习Agent规划.md`。

第二阶段（第 13~24 周）进行中，主题：从「能跑」到「能维护、能检索、能服务科研、
能给别人用」。四个目标按依赖顺序推进：

| 顺序 | 目标                | 覆盖周次    |
| ---- | ------------------- | ----------- |
| G1   | 还技术债 + 工程化   | 第 13~15 周 |
| G2   | 检索能力升级（RAG） | 第 16~19 周 |
| G3   | 落地到科研场景      | 第 20~22 周 |
| G4   | 对外可用            | 第 23~24 周 |

详细计划与验收标准见 `agent-learning/第二阶段学习规划.md`，
当前进度见 `agent-learning/学习进度.md`。

## 技术栈

- Python ≥3.12（实际 3.13.14）
- [uv](https://docs.astral.sh/uv/)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [DeepSeek API](https://api-docs.deepseek.com/)
- [pytest](https://docs.pytest.org/) + [ruff](https://docs.astral.sh/ruff/)
- [Codex CLI](https://developers.openai.com/codex/)