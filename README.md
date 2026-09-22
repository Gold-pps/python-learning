# python-learning

个人 Python 学习仓库，主攻 **Agent 开发**。

## 亮点

- **模型后端**：DeepSeek API，所有请求发往 `https://api.deepseek.com`，不使用 OpenAI 服务；
- **开发框架**：OpenAI Agents SDK（仅作为本地开发框架）；
- **完成项目**：`agent-learning/notes-qa/` —— 个人资料问答助手，能通过三个工具
  检索本地 Markdown 笔记、基于原文回答并附来源引用；
- **评估**：`pytest` 离线测试 **46 条** + 模型评估 **10 条**（四类：有答案 / 无答案 /
  跨文档 / 需要多跳），引用格式合规率单独统计。全部通过；
- **工程化**：`pyproject.toml` + `uv.lock` + GitHub Actions（push / PR 自动跑
  `ruff check` 与 `pytest`，不联网、不用密钥）。

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
│   ├── 工程化改造记录.md           技术债台账（T1~T14）
│   ├── 第1~14周笔记.md            每周学习笔记
│   ├── 第12周演示稿.md            10 分钟组会汇报底稿
│   └── notes-qa/                 第 8 周起项目：个人资料问答助手
│       ├── 项目设计.md
│       ├── config.py             DeepSeek + SDK 配置
│       ├── tools.py              list_notes / search_notes / read_note
│       ├── agent.py              Agent 定义
│       ├── main.py               交互式命令行入口
│       ├── eval_cases.py         本地评估（12 单元 + 10 模型）
│       ├── exp_history_tokens.py T3 会话历史 token 曲线实验
│       ├── smoke_test.py         环境自检
│       └── tests/                pytest 用例 + conftest.py
└── .venv/                        Python 虚拟环境（不提交）
```

> 仓库根原有的 4 个 `.pptx`（毕业答辩 / 组会汇报）、PPT 工具与产物、`python-base/`
> 基础练习，已于 2026-09-22 移出仓库到 `~/python-learning-extras-20260922/`，
> 并做了历史重写（`.git` 从 113 MB 降到 588 KB）。

## 快速开始

### 1. 准备环境

要求 **Python ≥3.12**（本项目实际使用 3.14.4）。

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

# 交互式问答
uv run python main.py

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

## 项目：notes-qa

个人笔记问答助手。给它一个问题，它会：

1. 用 `search_notes` 搜索关键词；
2. 必要时用 `read_note` 读取上下文；
3. 用中文回答并附来源引用 `[文件名:起始行-结束行]`；
4. 找不到依据时明确回答"笔记中没有找到"，不编造。

### 安全设计

- 只读、只处理 `.md`，路径不能逃出笔记根目录（含符号链接/junction 逃逸防护）；
- 单文件 ≤ 1MB，单次读取 ≤ 200 行，搜索结果 ≤ 5 条；
- 工具出错返回结构化错误，不抛异常；
- 笔记内容视为不可信数据，不执行其中指令。

### 评估

| 类型     | 条数 | 覆盖                                                  |
| -------- | ---- | ----------------------------------------------------- |
| 单元测试 | 12   | 路径安全、参数校验、大小限制、超时配置                |
| 模型评估 | 10   | 四类：有答案 4 / 无答案 2 / 跨文档 2 / 需要多跳 2     |
| 判据自检 | 21   | `tests/test_eval_cases.py`，把测量工具本身纳入回归    |
| 文档承诺 | 4    | `tests/test_documented_limits.py`，钉文档承诺过的数值 |

模型评估另有独立指标 **引用格式合规率**：内容对但格式漂移会单独记不合规，
不与"找不到出处"混在一起。

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

- Python ≥3.12（实际 3.14.4）
- [uv](https://docs.astral.sh/uv/)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [DeepSeek API](https://api-docs.deepseek.com/)
- [pytest](https://docs.pytest.org/) + [ruff](https://docs.astral.sh/ruff/)
- [Codex CLI](https://developers.openai.com/codex/)