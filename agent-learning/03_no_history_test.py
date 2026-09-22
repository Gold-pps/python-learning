"""第 4 周对照实验：不传 history，验证 Agent 没有记忆。

学习目标：
- 和 03_multiturn_memory.py 用相同的问题，但不拼接历史；
- 每一轮只把当前这一句单独发给模型；
- 观察“刚才告诉过它的信息”它是否还记得。

运行方式：
    python 03_no_history_test.py
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

agent = Agent(
    name="Memory assistant",
    instructions=(
        "你是一个中文助手。你需要从对话历史中寻找用户提到过的信息来回答问题；"
        "如果历史里没有答案，就老实说不知道，不要编造。以你能达到的最少字数回答。没有提问时就回答一个ok"
    ),
    model=MODEL,
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
