"""第 5 周：多 Agent 编排与交接（handoffs）Demo。

结构：
    triage_agent（总控/分流）
      ├── history_agent（历史老师）
      └── math_agent（数学老师）

总控不回答问题，只负责判断问题属于哪个学科，
然后把对话“交接”给对应的专长 Agent。

运行方式：
    python 04_multi_agent_handoff.py
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

# 两个专长 Agent（必须先定义，总控的 handoffs 才能引用它们）
history_agent = Agent(
    name="history_teacher",
    handoff_description="适合回答历史、朝代、事件、人物等问题",
    instructions=(
        "你是一位耐心的历史老师。用简洁准确的中文回答问题；"
        "如果问题与历史无关，就告诉用户这更适合其他学科。"
    ),
    model=MODEL,
)

math_agent = Agent(
    name="math_teacher",
    handoff_description="适合回答数学、计算、几何、方程等问题",
    instructions=(
        "你是一位耐心的数学老师。先解释思路，再给出答案；"
        "如果问题与数学无关，就告诉用户这更适合其他学科。"
    ),
    model=MODEL,
)

# 总控 Agent：本身不答题，只做学科分流
triage_agent = Agent(
    name="triage_agent",
    handoff_description="作业问答的总入口",
    instructions=(
        "你是作业问答系统的总控。判断用户问题属于哪个学科"
        "你不被允许回答问题。"
        "只能把对话交接给对应专长的老师：历史人文类问题交给擅长历史的老师，"
        "数学计算类问题交给擅长数学的老师。简单寒暄可以自己直接回应；"
        "拿不准时请用户补充说明，不要自己硬答专业问题。"
    ),
    model=MODEL,
    handoffs=[history_agent, math_agent],
)


async def main():
    questions = [
        "How are you怎么翻译？",
        "明朝是哪一年建立的？",
        "小明有 7 个苹果，又买了 5 个，再吃掉 3 个，现在还剩几个？",
        "你好，谢谢！",
    ]

    for question in questions:
        result = await Runner.run(triage_agent, question)
        print("问题：", question)
        print("实际回答的 Agent：", result.last_agent.name)
        print("回答：", result.final_output)
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())
