# AGENTS.md

本文件写给进入本仓库的 AI 助手（Codex 等）。开始工作前请先读这里。

## 仓库是什么

个人 Python 学习仓库，主攻 **Agent 开发**。

- 模型后端：**DeepSeek API**（所有请求发往 `https://api.deepseek.com`，不使用 OpenAI 服务）；
- 开发框架：**OpenAI Agents SDK**（仅作为本地开发框架）；
- 模型名统一用 `deepseek-flash`，**不要写 `gpt-*`**；
- 默认关闭 tracing（`set_tracing_disabled(disabled=True)`），不开 OpenAI 云端追踪。

## 目录结构

```text
python-learning/
├── agent-learning/           Agent 学习目录
│   ├── 00~07_*.py            第 1~7 周练习脚本
│   ├── 学习进度.md            当前进度（恢复上下文先读它）
│   ├── 学习Agent规划.md       12 周计划
│   ├── 第4~10周笔记.md        每周笔记
│   ├── .env                  API Key（**不要提交**）
│   └── notes-qa/             第 8~10 周项目：个人资料问答助手
│       ├── 项目设计.md
│       ├── config.py         DeepSeek + SDK 配置
│       ├── tools.py          list_notes / search_notes / read_note
│       ├── agent.py          notes_qa Agent 定义
│       ├── main.py           交互式命令行入口
│       └── eval_cases.py     本地评估（8 条）
├── python-base/              基础练习
└── .venv/                    Python 虚拟环境