# AGENTS.md

本文件写给进入本仓库的 AI 助手（Codex 等）。开始工作前请先读这里。

> 最后更新：2026-09-22（第 15 周 G1 工程化收口：历史重写、依赖与 CI 落地）

## 仓库是什么

个人 Python 学习仓库，主攻 **Agent 开发**。

- 模型后端：**DeepSeek API**（所有请求发往 `https://api.deepseek.com`，不使用 OpenAI 服务）；
- 开发框架：**OpenAI Agents SDK**（仅作为本地开发框架）；
- 模型名统一用 `deepseek-flash`，**不要写 `gpt-*`**；
- 默认关闭 tracing（`set_tracing_disabled(disabled=True)`），不开 OpenAI 云端追踪；
- Python 要求 **≥3.12**（`main.py` 的 `asyncio.timeout` 需 3.11+，
  `eval_cases.py` 的 `Path.is_junction()` 需 3.12+），本机实际使用 3.14.4。

## 当前进度

- 第一阶段 12 周已完成（2026-09-13 收尾）；
- 第二阶段进行中：第 13、14 周已完成并通过验收，当前在第 15 周（G1 验收周）；
- 恢复上下文请先读 `agent-learning/学习进度.md`；
- 计划与验收标准见 `agent-learning/第二阶段学习规划.md`；
- 技术债台账见 `agent-learning/工程化改造记录.md`（T1~T14）。

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
│   ├── 工程化改造记录.md        技术债台账（T1~T14）
│   ├── 第1~14周笔记.md         每周笔记（14 篇）
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
│       └── tests/             pytest 用例 + conftest.py
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
uv run pytest -q       # 46 passed
uv run ruff check .    # All checks passed

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

## 关于沙箱网络（2026-09-22 修正）

`AGENTS.md` 此前写"不要在 Codex 沙箱里跑模型评估——沙箱没有网络"。
第 13 周实测：沙箱内 `https://api.deepseek.com` 可达（`codex doctor` 的到达性探针 +
8 次真实调用全部成功）。**真正的约束不是"没网络"，而是"会花 token，所以要带超时、
要限量、要记账"**。离线任务仍优先用 `uv run pytest -q` / `--unit-only`。