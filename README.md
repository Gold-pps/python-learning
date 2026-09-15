# python-learning

个人 Python 学习仓库，主攻 **Agent 开发**。从零起步，12 周完成从 API 调用到可用的 Agent 小项目（2026-09-13 收尾）。

## 亮点

- **模型后端**：DeepSeek API，所有请求发往 `https://api.deepseek.com`，不使用 OpenAI 服务；
- **开发框架**：OpenAI Agents SDK（仅作为本地开发框架）；
- **完成项目**：`agent-learning/notes-qa/` —— 个人资料问答助手，能通过三个工具
  检索本地 Markdown 笔记、基于原文回答并附来源引用；
- **评估**：本地评估脚本，**单元测试 12 条 + 模型评估 3 条 = 15 条，全部通过**
  （2026-09-15 复跑单元测试仍为 12/12）。

## 目录结构

```text
python-learning/
├── AGENTS.md                     写给 AI 助手的仓库规则
├── README.md                     本文件
├── requirements.txt              依赖清单（备选，版本以 .venv 实际为准）
├── .codex/skills/                Codex 自定义 skill
│   └── notes-qa-eval/SKILL.md    跑 notes-qa 评估
├── agent-learning/               Agent 学习主目录
│   ├── 00~07_*.py                第 1~7 周练习脚本（9 个）
│   ├── 学习Agent规划.md           12 周学习计划
│   ├── 学习进度.md                当前进度（恢复上下文先读它）
│   ├── 笔记索引.md                每周主题 / 笔记 / 产出对照
│   ├── 学习评审报告.md             仓库与文档评审（含待办清单）
│   ├── 第1~12周笔记.md            每周学习笔记（12 篇）
│   ├── 第12周演示稿.md            10 分钟组会汇报底稿
│   └── notes-qa/                 第 8~11 周项目
│       ├── 项目设计.md
│       ├── config.py             DeepSeek + SDK 配置
│       ├── tools.py              list_notes / search_notes / read_note
│       ├── agent.py              Agent 定义
│       ├── main.py               交互式命令行入口
│       ├── eval_cases.py         本地评估（12 + 3 = 15 条）
│       └── smoke_test.py         环境自检（可删）
├── python-base/                  基础 Python 练习（CS50 风格）
└── .venv/                        Python 虚拟环境（不提交）
```

> 仓库根另有 4 个 `.pptx`（毕业答辩 / 组会汇报）与 1 张截图，与学习内容无关。

## 快速开始

### 1. 准备环境

要求 **Python ≥3.12**（本项目实际使用 3.14.4）。

```bash
uv venv .venv
uv pip install --python .venv/bin/python openai-agents python-dotenv
```

`requirements.txt` 是备选清单，其钉住的版本与当前 `.venv` 存在小幅差异，以实际环境为准。

### 2. 配置 API Key

在 `agent-learning/.env` 写入：

```text
DEEPSEEK_API_KEY=sk-...
```

### 3. 跑项目

```bash
cd agent-learning/notes-qa

# 交互式问答
python main.py

# 跑完整评估（需要网络）
python eval_cases.py

# 只跑单元测试（无需网络）
python eval_cases.py --unit-only
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

| 类型     | 条数 | 覆盖                                   |
| -------- | ---- | -------------------------------------- |
| 单元测试 | 12   | 路径安全、参数校验、大小限制、超时配置 |
| 模型评估 | 3    | 跨笔记检索、护栏概念、无答案不编造     |

## 12 周学习路线

| 周次 | 主题                                |
| ---- | ----------------------------------- |
| 1    | 概念与环境                          |
| 2    | 跑通第一个 Agent                    |
| 3    | 让 Agent 学会使用工具               |
| 4    | 理解运行循环与状态                  |
| 5    | 多 Agent 编排与交接                 |
| 6    | 安全与护栏                          |
| 7    | 追踪与评估                          |
| 8    | 项目启动：确定范围与设计            |
| 9    | 项目 MVP：三个工具 + 评估           |
| 10   | 项目收尾：交互升级 + 安全修复       |
| 11   | 产品线进阶：AGENTS.md / Skill / MCP / Subagents（载荷传递未成功） |
| 12   | 复盘与展示                          |

详细计划见 `agent-learning/学习Agent规划.md`，每周笔记与产出对照见 `agent-learning/笔记索引.md`，
当前进度见 `agent-learning/学习进度.md`。

## 技术栈

- Python ≥3.12（实际 3.14.4）
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [DeepSeek API](https://api-docs.deepseek.com/)
- [Codex CLI](https://developers.openai.com/codex/)
