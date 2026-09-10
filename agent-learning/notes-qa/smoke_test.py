"""Day 1 冒烟测试：确认 config.py 能让 Agent 连上 DeepSeek。

通过后这个文件可以删掉，或留着当环境自检。
"""
import asyncio

import config  # noqa: F401  导入即完成 DeepSeek 配置
from agents import Agent, Runner


async def main():
    agent = Agent(
        name="smoke",
        instructions="用一句中文回答，不要展开。",
        model=config.MODEL,
    )
    result = await Runner.run(agent, "1+1 等于几？")
    print("回答：", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())