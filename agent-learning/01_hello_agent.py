"""第 2 周 · 第 2 步：把 OpenAI Agents SDK 接到 DeepSeek，跑通第一个 Agent。

运行前准备：
1. 已在 platform.deepseek.com 创建 API Key；
2. 已设置环境变量 DEEPSEEK_API_KEY，或同目录存在 .env 文件；
3. 已安装 openai-agents；
4. 在对应 Python 环境里运行：
   python 01_hello_agent.py
"""

import asyncio
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit(
        "没有找到 DEEPSEEK_API_KEY。\n"
        "请先在 platform.deepseek.com 创建 API Key，"
        "再设置环境变量或创建 .env 文件后重试。"
    )

# 1. 让 OpenAI SDK 的客户端指向 DeepSeek，而不是 OpenAI
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

# 2. 关键三步：默认客户端指向 DeepSeek、走 Chat Completions、关闭发往 OpenAI 的追踪
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# 3. 定义一个最小 Agent：模型用 DeepSeek，而不是 gpt-*
agent = Agent(
    name="History tutor",
    instructions="You answer history questions clearly and concisely.",
    model="deepseek-v4-flash",
)


async def main():
    result = await Runner.run(agent, "罗马帝国是什么时候灭亡的？")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
