"""第 6 周（下）：敏感操作的人工审批（human-in-the-loop）。

学习目标：
- 用 @function_tool(needs_approval=True) 把敏感工具标记为“需要审批”；
- 观察 Runner.run 暂停后返回的 result.interruptions；
- 用 result.to_state() 拿到可恢复状态，再 approve 或 reject；
- 理解“审批不是新问题，而是同一轮运行的暂停与恢复”。

运行方式：
    python 06_human_approval.py              # 运行时询问你 y/N
    python 06_human_approval.py --decision approve
    python 06_human_approval.py --decision reject
"""

import argparse
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

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)


@function_tool(needs_approval=True)
def delete_file(path: str) -> str:
    """删除指定文件（演示用，不会真的删除磁盘文件）。"""
    print(f"[工具已执行] delete_file(path={path})")
    return f"已删除文件：{path}（演示结果）"


agent = Agent(
    name="file_assistant",
    instructions=(
        "你是文件管理助手。用户明确要求删除文件时，"
        "你必须调用 delete_file 工具，不要自己假装完成。"
    ),
    model="deepseek-flash",
    tools=[delete_file],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人工审批 Demo")
    parser.add_argument(
        "--decision",
        choices=("ask", "approve", "reject"),
        default="ask",
        help="审批决定；默认 ask 会在运行时询问你",
    )
    return parser.parse_args()


def get_decision(mode: str) -> bool:
    if mode == "approve":
        return True
    if mode == "reject":
        return False
    answer = input("是否批准执行 delete_file？(y/N): ").strip().lower()
    return answer in {"y", "yes"}


async def main() -> None:
    args = parse_args()
    result = await Runner.run(agent, "请删除 agent学习/临时文件.txt")

    if not result.interruptions:
        print("模型没有请求调用工具，因此没有进入审批流程。")
        print("最终回答：", result.final_output)
        return

    print("运行已暂停，等待人工审批：")
    for item in result.interruptions:
        print(f"- 工具：{item.name}")
        print(f"  参数：{item.arguments}")

    approved = get_decision(args.decision)
    state = result.to_state()

    for item in result.interruptions:
        if approved:
            state.approve(item)
        else:
            state.reject(item, rejection_message="用户拒绝执行删除操作")

    result = await Runner.run(agent, state)
    print("\n审批结果：", "批准" if approved else "拒绝")
    print("最终回答：", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
