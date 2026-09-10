"""第 4 周：多轮对话 Demo —— 用“手动传递 history”让 Agent 记住之前的信息。

学习目标：
- 明白 Agent 本身没有记忆，每次 Runner.run 都是一次独立的“短对话”；
- 如果要让它记得上一轮说过什么，必须把历史消息一起传进去；
- 这里使用 result.to_input_list() 取出上一轮完整消息，再拼接新问题。

运行方式：
    python 03_multiturn_memory.py
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

    history: list = []

    for question in questions:
        # 把新问题追加到历史末尾，然后整段发给模型
        history.append({"role": "user", "content": question})
        result = await Runner.run(agent, history)

        # 取出“用户问题 + 模型回答”的完整历史，用于下一轮
        history = result.to_input_list()

        print(f"问：{question}")
        print(f"答：{result.final_output}\n")

    print(f"本轮一共传给模型的完整消息条数：{len(history)}")


if __name__ == "__main__":
    asyncio.run(main())
