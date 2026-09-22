"""第 2 周 · 第 2 步：把 OpenAI Agents SDK 接到 DeepSeek，跑通第一个 Agent。

运行前准备：
1. 已在 platform.deepseek.com 创建 API Key；
2. 已设置环境变量 DEEPSEEK_API_KEY，或同目录存在 .env 文件；
3. 已安装 openai-agents；
4. 在对应 Python 环境里运行：
   python 01_hello_agent.py
"""

import asyncio

# 1~2. 让 OpenAI SDK 指向 DeepSeek：默认客户端指向 DeepSeek、走 Chat Completions、
#      关闭发往 OpenAI 的追踪。这三步配置与密钥读取都收在 `_common.py` 里，
#      import 时就完成了（第 14 周 W14-4 抽出，周脚本共用一份）。
from _common import MODEL
from agents import (
    Agent,
    Runner,
)

# 3. 定义一个最小 Agent：模型用 DeepSeek，而不是 gpt-*
agent = Agent(
    name="History tutor",
    instructions="You answer history questions clearly and concisely.",
    model=MODEL,
)


async def main():
    result = await Runner.run(agent, "罗马帝国是什么时候灭亡的？")
    usage = result.context_wrapper.usage
    print(result.final_output)
    print(
        f"请求数={usage.requests} 输入tokens={usage.input_tokens} "
        f"输出tokens={usage.output_tokens} 合计={usage.total_tokens}"
    )


if __name__ == "__main__":
    asyncio.run(main())
