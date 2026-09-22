"""第 4 周：多轮对话 Demo —— 用“手动传递 history”让 Agent 记住之前的信息。

学习目标：
- 明白 Agent 本身没有记忆，每次 Runner.run 都是一次独立的“短对话”；
- 如果要让它记得上一轮说过什么，必须把历史消息一起传进去；
- 这里使用 result.to_input_list() 取出上一轮完整消息，再拼接新问题。

运行方式：
    python 03_multiturn_memory.py
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
