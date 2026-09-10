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
def count_students_rows() -> str:
    """统计 python-base 目录下 students.csv 的数据行数（不含表头）。"""
    csv_path = Path(__file__).resolve().parent.parent / "python-base" / "students.csv"
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    print("[调用count_students_rows]")
    return str(max(0, len(lines) - 1))

@function_tool
def get_beijing_time() -> str:
    """返回当前的北京时间，格式为“YYYY-MM-DD HH:MM:SS”。"""
    beijing_tz = datetime.timezone(datetime.timedelta(hours=8))
    now = datetime.datetime.now(beijing_tz)
    text = now.strftime("%Y-%m-%d %H:%M:%S")
    # 这行会直接打印在终端里，方便你确认工具真的被调用了
    print(f"[工具被调用]")
    return text

@function_tool
def find_student_home(name: str) -> str:
    """在 python-base 目录的 students.csv 中查找指定学生的家乡，找不到就返回提示。"""
    print(f"[工具被调用] find_student_home(name={name})")
    csv_path = Path(__file__).resolve().parent.parent / "python-base" / "students.csv"
    for line in csv_path.read_text(encoding="utf-8").strip().splitlines()[1:]:
        student, home = line.split(",")
        if student == name:
            return home
    return "未找到该学生"


agent = Agent(
    name="Time assistant",
    instructions=(
        "你是一个中文助手。当用户询问当前时间时，"
        "你必须调用 get_beijing_time 工具获取真实时间，再简洁回答。"
        "当需要查看本地文件、统计行数或读取文件内容时，"
        "你必须调用对应工具获取真实数据，再简洁回答。"
        "若问题中没有对应的工具，则尝试所有工具"
    ),
    model="deepseek-flash",
    tools=[get_beijing_time,count_students_rows,find_student_home],
)


async def main():
    result = await Runner.run(agent, "cain的家在哪？")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
