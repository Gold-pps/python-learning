# AGENTS.md

本文件写给进入本仓库的 AI 助手（Codex 等）。开始工作前请先读这里。

> 最后更新：2026-09-15（已按仓库实际情况校正）

## 仓库是什么

个人 Python 学习仓库，主攻 **Agent 开发**。12 周学习计划已全部完成（2026-09-13 收尾）。

- 模型后端：**DeepSeek API**（所有请求发往 `https://api.deepseek.com`，不使用 OpenAI 服务）；
- 开发框架：**OpenAI Agents SDK**（仅作为本地开发框架）；
- 模型名统一用 `deepseek-flash`，**不要写 `gpt-*`**；
- 默认关闭 tracing（`set_tracing_disabled(disabled=True)`），不开 OpenAI 云端追踪；
- Python 要求 **≥3.12**（`main.py` 的 `asyncio.timeout` 需 3.11+，`eval_cases.py` 的 `Path.is_junction()` 需 3.12+），本机实际使用 3.14.4。

## 目录结构

```text
python-learning/
├── AGENTS.md                  本文件（写给 AI 助手）
├── README.md                  仓库门面（给人看）
├── requirements.txt           依赖清单（备选，版本以 .venv 实际为准）
├── .codex/skills/
│   └── notes-qa-eval/SKILL.md 自定义 Skill：跑 notes-qa 评估
├── agent-learning/            Agent 学习主目录（★ 同时是 notes-qa 的笔记根目录）
│   ├── 00~07_*.py             第 1~7 周练习脚本（9 个；03 有"多轮记忆 / 无历史"两份）
│   ├── 学习Agent规划.md        12 周计划（勾选已按实际回填）
│   ├── 学习进度.md             当前进度（**恢复上下文先读它**）
│   ├── 笔记索引.md             每周主题 / 笔记 / 产出对照
│   ├── 学习评审报告.md          2026-09-15 仓库与文档评审 + 处理状态
│   ├── 第1~12周笔记.md         每周笔记（12 篇，含第 1~3、8 周补写）
│   ├── 第12周演示稿.md         10 分钟组会汇报底稿
│   ├── .env                   API Key（**不要提交**）
│   └── notes-qa/              第 8~11 周项目：个人资料问答助手
│       ├── 项目设计.md
│       ├── config.py          DeepSeek + SDK 配置
│       ├── tools.py           list_notes / search_notes / read_note
│       ├── agent.py           notes_qa Agent 定义
│       ├── main.py            交互式命令行入口
│       ├── eval_cases.py      本地评估（12 单元 + 3 模型 = 15 条）
│       └── smoke_test.py      环境自检（可删）
├── python-base/               基础练习（CS50 风格，暂未整理）
└── .venv/                     Python 虚拟环境（不提交）
```

> 仓库根还有 4 个 `.pptx`（毕业答辩 / 组会汇报）和 1 张截图，与学习内容无关，是 `.git` 体积达到约 93MB 的主因（见 `agent-learning/学习评审报告.md`）。

## 常用命令

```bash
# 环境（首次或换机后）
uv venv .venv
uv pip install --python .venv/bin/python openai-agents python-dotenv

# 项目：交互式问答
cd agent-learning/notes-qa
python main.py                      # /new 清空历史，/exit 退出

# 评估
python eval_cases.py                # 完整（12 单元 + 3 模型，需联网、消耗 token）
python eval_cases.py --unit-only    # 只跑单元测试（无需网络）

# 周练习脚本
cd agent-learning
python 00_first_api_call.py         # 同理可跑到 07_evaluate_agents.py
```

## 不要做的事

- 不要把 API Key 写进代码或提交到 Git（Key 只放在 `agent-learning/.env`）；
- 不要使用 OpenAI 服务、不要写 `gpt-*` 模型名、不要开启 tracing；
- 不要放宽 `notes-qa` 工具的笔记根目录边界（`agent-learning/` 之外的路径一律拒绝）；
- 不要把 `notes-qa/` 当作笔记内容读取（已在 `EXCLUDE_DIRS` 中排除）；
- 不要在 Codex 沙箱里跑模型评估——沙箱没有网络，会一直卡住（用 `--unit-only`）；
- 不要在没有用户明确要求时执行 `git commit` / `git push`；
- 不要删除 `.codex/`（存的是项目级 Skill）。
