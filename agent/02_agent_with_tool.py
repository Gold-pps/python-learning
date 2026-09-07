"""第 3 周：让 Agent 学会使用工具。

运行前准备：
1. 已设置 DEEPSEEK_API_KEY（或同目录存在 .env）；
2. 在 miniconda base（或其他安装过 openai-agents 的环境）里运行：
   python 02_agent_with_tool.py

学习目标：
- 理解“模型决定调用工具 → 代码执行 → 结果返回模型”的循环；
- 观察 Agent 在回答前主动调用 get_beijing_time 工具；
- 这个工具返回的是程序实时计算的北京时间，模型不可能凭空知道。
"""

import asyncio
import datetime
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
    function_tool,
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

# 和 01 完全相同的 DeepSeek 接入配置
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)


@function_tool
def get_beijing_time() -> str:
    """返回当前的北京时间，格式为“YYYY-MM-DD HH:MM:SS”。"""
    beijing_tz = datetime.timezone(datetime.timedelta(hours=8))
    now = datetime.datetime.now(beijing_tz)
    text = now.strftime("%Y-%m-%d %H:%M:%S")
    # 这行会直接打印在终端里，方便你确认工具真的被调用了
    print(f"[工具被调用]")
    return text


agent = Agent(
    name="Time assistant",
    instructions=(
        "你是一个中文助手。当用户询问当前时间时，"
        "你必须调用 get_beijing_time 工具获取真实时间，再简洁回答。"
    ),
    model="deepseek-v4-flash",
    tools=[get_beijing_time],
)


async def main():
    result = await Runner.run(agent, "1123152454*23524762542=?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
