# Agent 学习规划

> 制定日期：2026-09-05
> 学习目标：从零学会“使用 Agent”和“开发 Agent”，最终独立完成一个能调工具、多步工作的 Agent 小项目。
> 模型后端：**DeepSeek API**。OpenAI 系 SDK 只是本地开发框架，所有请求都发往 `https://api.deepseek.com`，全程不需要 OpenAI 充值。

## 一、先建立正确认知

Agent 不是普通的“问答模型”，而是能围绕目标主动工作的程序：

- **模型**：负责理解、推理和生成内容；
- **指令**：告诉它角色、目标和工作方式；
- **工具**：让它能查资料、算数据、操作文件或访问外部系统；
- **循环**：模型提出计划 → 调用工具 → 看到结果 → 继续行动，直到完成；
- **状态**：让它记住上下文、跨轮次持续推进。

你现在正在使用的 Codex 本身就是 Agent，所以学习素材就在手边。

## 二、两条学习线（可以并行）

### A 线：产品使用与调教（不依赖编程）

1. 日常主动让 Codex / ChatGPT Work 完成多步任务；
2. 观察它如何拆解任务、调用工具、请求授权、修改文件；
3. 学习用 `AGENTS.md` 设定项目规则；
4. 学习用 Skills 封装可复用流程；
5. 学习用 MCP 接入外部系统；
6. 学习用 Subagents 做任务分工。

### B 线：动手开发（需要 Python 或 TypeScript）

推荐路线以 Python 为主：

1. 用 DeepSeek API 跑通首次调用，再接上最小 Agent；
2. 给它添加第一个函数工具；
3. 理解运行循环、会话与状态；
4. 做多 Agent 分工与交接（handoffs）；
5. 加入护栏、人工审批；
6. 用本地日志、成本账单和评估用例改进效果。

## 三、12 周执行计划

### 第 1 周：概念与环境准备

- [ ] 阅读 DeepSeek 官方 [首次调用 API](https://api-docs.deepseek.com/quick_start/your_first_api_call/) 和 [模型与价格](https://api-docs.deepseek.com/quick_start/pricing/)
- [ ] 了解 OpenAI Agents SDK 只是开发框架：改 `base_url` 后请求会发往 DeepSeek，不调用 OpenAI 服务（官方说明见 [Models and providers](https://openai.github.io/openai-agents-python/models/#ways-to-integrate-non-openai-providers)）
- [ ] 用 Codex 完成 3 个多步任务，记录它的工作过程
- [ ] 安装 Python 3.10+、pip、Git
- [ ] 到 [DeepSeek 开放平台](https://platform.deepseek.com) 注册账号 → 实名认证 → 用支付宝/微信小额充值 → 创建 API Key
- [ ] 把 Key 放进本地环境变量 `DEEPSEEK_API_KEY`（不要写进代码、不要提交到 Git）
- [ ] 建立自己的学习目录：`python学习/agent/`

产出：理解“Agent = 模型 + 指令 + 工具 + 循环”；本机已配置好 DeepSeek Key。
费用建议：学习阶段默认用 `deepseek-v4-flash`，复杂任务再换 `deepseek-v4-pro`；先小额充值，用完看账单再充。

### 第 2 周：跑通第一个 Agent

- [x] 先用 DeepSeek 官方文档的示例完成一次最普通的聊天调用（OpenAI SDK + `base_url="https://api.deepseek.com"`）
- [x] 安装依赖：`pip install openai-agents`
- [x] 再把 OpenAI Agents SDK 接上 DeepSeek，写出最小示例：
  ```python
  import asyncio
  import os

  from openai import AsyncOpenAI
  from agents import (
      Agent,
      Runner,
      set_default_openai_api,
      set_default_openai_client,
      set_tracing_disabled,
  )

  # 1. 让 OpenAI SDK 的客户端指向 DeepSeek
  client = AsyncOpenAI(
      api_key=os.environ.get("DEEPSEEK_API_KEY"),
      base_url="https://api.deepseek.com",
  )

  # 2. 关键三步：默认客户端指向 DeepSeek、走 Chat Completions、关闭发往 OpenAI 的追踪
  set_default_openai_client(client=client, use_for_tracing=False)
  set_default_openai_api("chat_completions")
  set_tracing_disabled(disabled=True)

  agent = Agent(
      name="History tutor",
      instructions="You answer history questions clearly and concisely.",
      model="deepseek-v4-flash",
  )

  async def main():
      result = await Runner.run(agent, "When did the Roman Empire fall?")
      print(result.final_output)

  asyncio.run(main())
  ```
- [ ] 弄懂 `AsyncOpenAI(base_url=...)`、`set_default_openai_api("chat_completions")`、`Agent`、`Runner.run`、`final_output`

产出：请求真正发往 DeepSeek 且能跑通一个 Agent；代码保存为 `01_hello_agent.py`。

### 第 3 周：让 Agent 学会使用工具

- [ ] 阅读 DeepSeek 官方 [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/) 文档
- [ ] 对照阅读 OpenAI Agents SDK 的 [Using tools](https://developers.openai.com/api/docs/guides/tools)（看框架写法，模型名换成 DeepSeek）
- [ ] 用 `function_tool` 写第一个自定义工具（计算器、天气查询、文件读取均可）
- [ ] 观察“模型决定调用工具 → 代码执行 → 结果返回模型”的完整循环
- [ ] 练习一个“不用工具答不了、用了工具才能答对”的题目
- [ ] 注意：本章仍走 Chat Completions 兼容路径，避开 OpenAI Responses 专属工具能力

产出：Agent 能通过工具解决单步任务；代码保存为 `02_agent_with_tool.py`。

### 第 4 周：理解运行循环与状态

- [ ] 阅读官方 [Running agents](https://developers.openai.com/api/docs/guides/agents/running-agents)
- [ ] 掌握多轮对话如何保留上下文（history / session / continuation）
- [ ] 不使用 OpenAI 专属的服务端 continuation，改为手动传 `history` 或用 SDK 本地 session
- [ ] 做一个小练习：连续提问 3 次，Agent 能记住之前的信息

产出：理解会话状态，做出多轮对话 Demo。

### 第 5 周：多 Agent 编排与交接

- [ ] 阅读官方 [Orchestration and handoffs](https://developers.openai.com/api/docs/guides/agents/orchestration)
- [ ] 创建一个“总控 Agent + 两个专长 Agent”（例如历史老师 + 数学老师）
- [ ] 让总控 Agent 根据问题类型自动交接给合适的专长 Agent
- [ ] 了解 Agent as tool 与 handoffs 的适用区别

产出：能自动分流的作业问答 Demo。

### 第 6 周：安全与护栏

- [ ] 阅读官方 [Guardrails and human review](https://developers.openai.com/api/docs/guides/agents/guardrails-approvals)
- [ ] 给 Agent 加输入/输出护栏
- [ ] 练习设置“敏感操作需人工确认”的流程

产出：一份安全设计笔记 + 带护栏的小 Demo。

### 第 7 周：追踪与评估

- [ ] 由于没有 OpenAI Key，保持代码里的 `set_tracing_disabled(disabled=True)`，不在 OpenAI 平台开追踪
- [ ] 需要调试时在本地打印关键日志：模型回复、工具调用参数、工具返回结果、交接目标
- [ ] 熟悉 DeepSeek 平台的用量/账单页面，学会每次实验后查看花费
- [ ] 阅读官方 [Evaluate agent workflows](https://developers.openai.com/api/docs/guides/agent-evals) 理解评估思路，但评估脚本在本地自己写
- [ ] 为前几周的 Agent 写 2～3 条评估用例

产出：能用本地日志定位问题、通过账单控制成本，并用评估用例衡量改进。

### 第 8～10 周：完成一个小项目（三选一）

1. **个人资料问答助手**：读取本地 Markdown 笔记 / 文档，回答你的问题；
2. **研究方向检索分析 Agent**：结合数控、智能制造方向做资料检索与要点总结；
3. **自动化报告 Agent**：从数据 → 分析 → 图表 → 导出文档。

里程碑安排：

- [ ] 第 8 周：确定项目范围，画出功能结构
- [ ] 第 9 周：跑通最小可用版本（只做最核心的一件事）
- [ ] 第 10 周：加工具、状态、错误处理并做演示

### 第 11 周：产品线进阶（A 线深化）

- [ ] 阅读官方 [Customization 文档](https://learn.chatgpt.com/docs/customization/overview)
- [ ] 为当前项目写一份 `AGENTS.md`
- [ ] 做一个自己的 Skill
- [ ] 尝试接入一个 MCP 工具
- [ ] 练习用 Subagents 分派任务

### 第 12 周：复盘与展示

- [ ] 整理学习笔记与项目代码
- [ ] 准备一次 10 分钟的组会汇报或演示
- [ ] 写下“下一步想做的方向”

## 四、每周执行建议

- 每天投入 30～60 分钟，优先保证“动手跑通”，不要只看文档；
- 每周至少留下一个可运行的产出；
- 卡住时先查官方文档，再问 AI，最后再搜博客；
- 所有代码统一放在 `python学习/agent/`，笔记用 Markdown 记录；
- 默认模型只用 `deepseek-v4-flash`；某一步质量确实不够时再临时换成 `deepseek-v4-pro`；
- 每周看一次 DeepSeek 平台账单，把花费记在笔记里；
- 每完成一周，回来看这份文件打勾，并写下 3 句话复盘。

## 五、官方参考

### DeepSeek 官方（实际使用的模型服务）

- [首次调用 API](https://api-docs.deepseek.com/quick_start/your_first_api_call/)
- [模型与价格](https://api-docs.deepseek.com/quick_start/pricing/)
- [Tool Calls（工具调用）](https://api-docs.deepseek.com/guides/tool_calls/)
- [开放平台（注册/充值/API Key）](https://platform.deepseek.com)

### OpenAI Agents SDK（仅作为开发框架，请求已指向 DeepSeek）

- [Models and providers：非 OpenAI 模型接入方式](https://openai.github.io/openai-agents-python/models/#ways-to-integrate-non-openai-providers)
- [Agents SDK 指南](https://developers.openai.com/api/docs/guides/agents)
- [Agents SDK Quickstart](https://developers.openai.com/api/docs/guides/agents/quickstart)
- [Agent definitions](https://developers.openai.com/api/docs/guides/agents/define-agents)
- [Running agents](https://developers.openai.com/api/docs/guides/agents/running-agents)
- [Using tools](https://developers.openai.com/api/docs/guides/tools)
- [Orchestration and handoffs](https://developers.openai.com/api/docs/guides/agents/orchestration)
- [Guardrails and human review](https://developers.openai.com/api/docs/guides/agents/guardrails-approvals)
- [Evaluate agent workflows](https://developers.openai.com/api/docs/guides/agent-evals)

### 说明

上面 OpenAI 概念文档里的示例都写着 `gpt-*` 模型名，阅读时把它们换成 DeepSeek 的 `deepseek-v4-flash` 或 `deepseek-v4-pro`，并且必须先做第 2 周的客户端配置，请求才会发往 DeepSeek。

### 其他

- [Customization（Codex 使用与调教）](https://learn.chatgpt.com/docs/customization/overview)

## 六、我的背景与期望（待填写）

- 现有编程基础：
- 每周可投入时间：
- 最想做的 Agent 应用场景：
- 三个月后的目标：
