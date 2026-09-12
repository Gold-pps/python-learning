---
name: notes-qa-eval
description: 运行 notes-qa 项目的本地评估（单元测试 + 模型评估），报告通过率和 token 用量。当用户说"跑一下评估""检查项目""项目自检"时使用。
---

## MCP

- 已配置 `filesystem` MCP（写在 `~/.codex/config.toml`，不在仓库里）；
- 允许访问目录：仓库根 `python-learning/`；
- 用途：需要跨目录遍历、批量读文件、边界检查时优先用 MCP 工具，
  而不是内置 shell 命令。

# notes-qa 评估

运行 `agent-learning/notes-qa/eval_cases.py`，汇报结果。

## 环境前提

Codex 默认沙箱**没有网络权限**，直接跑完整评估会在模型评估阶段报
`APIConnectionError`。所以：

1. **先跑单元测试**（`--unit-only`，不需要网络）；
2. 单元测试通过后，**不要**尝试跑模型评估——直接告诉用户手动跑命令。

## 步骤

1. 进入目录：`cd agent-learning/notes-qa`
2. 运行：`python eval_cases.py --unit-only`
3. 汇报单元测试结果；
4. 输出下面这段提示给用户（原样）：

   > 单元测试已完成。模型评估需要访问 DeepSeek API，Codex 沙箱没有网络权限，
   > 请你在自己的终端手动运行：
   >
   > ```
   > cd agent-learning/notes-qa
   > python eval_cases.py
   > ```

## 汇报格式

- **单元测试**：X/Y 通过；
- **失败项**（如有）：用例名 + 错误信息；
- **下一步**：上面那段"手动跑模型评估"的提示。

## 异常判断

- 单元测试失败 → 代码可能被改坏了，重点看 `tools.py` 和 `main.py`；
- 模型评估失败（用户手动跑后反馈） → 先看是 Agent 真答错，还是期望词不匹配
  （参考第 9、10 周笔记里的"假阴性"教训）。

## 不要做的事

- 不要改评估用例的期望词来"骗"通过率；
- 不要跳过失败的用例；
- 不要在沙箱里强行尝试联网跑模型评估——会一直卡住；
- 不要提交任何东西（提交由用户决定）。