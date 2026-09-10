"""第 4 周对照实验：不传 history，验证 Agent 没有记忆。

学习目标：
- 和 03_multiturn_memory.py 用相同的问题，但不拼接历史；
- 每一轮只把当前这一句单独发给模型；
- 观察“刚才告诉过它的信息”它是否还记得。

运行方式：
    python 03_no_history_test.py
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

# 和前面脚本相同的 DeepSeek 接入配置
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

agent = Agent(
    name="Memory assistant",
    instructions=(
        "你是一个中文助手。你需要从对话历史中寻找用户提到过的信息来回答问题；"
        "如果历史里没有答案，就老实说不知道，不要编造。以你能达到的最少字数回答。没有提问时就回答一个ok"
    ),
    model="deepseek-flash",
)


async def main():
    questions = [
        "请记住：我叫张伟，专业是机械设计制造及其自动化，最近在读《深度学习入门》。",
        "我叫什么名字？",
        "我的专业是什么？",
        "我最近在读哪本书？",
    ]

    for question in questions:
        result = await Runner.run(agent, question)
        print(f"问：{question}")
        print(f"答：{result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())
