"""命令行入口：一次回答一个问题。"""
import asyncio
import sys

import config  # noqa: F401
from agents import Runner

from agent import notes_qa_agent


async def answer(question: str) -> str:
    result = await Runner.run(notes_qa_agent, question)
    usage = result.context_wrapper.usage
    print(
        f"[用量] 请求数={usage.requests} "
        f"输入={usage.input_tokens} 输出={usage.output_tokens}",
        file=sys.stderr,
    )
    return result.final_output


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：python main.py '你的问题'")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    print(asyncio.run(answer(question)))


if __name__ == "__main__":
    main()