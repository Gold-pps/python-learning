"""命令行入口：交互式循环问答。

命令：
  /exit 或 /quit  退出
  /new            清空对话历史，开新会话

超时：单次请求和整轮提问都有上限（见 config.py），超时会打印错误并回到输入提示符。
"""
import asyncio
import sys

import config  # noqa: F401
from agents import Runner

from agent import notes_qa_agent

MAX_QUESTION_LEN = 500


def check_input(question: str) -> str | None:
    """返回错误提示，或 None 表示合法。"""
    if len(question) > MAX_QUESTION_LEN:
        return f"问题过长（{len(question)} 字符），上限 {MAX_QUESTION_LEN} 字符。"
    return None

async def ask(question: str, history: list) -> tuple[str, list]:
    """问一个问题，返回 (回答, 新 history)。"""
    # 把历史 + 本轮新问题拼成完整输入
    input_items = history + [{"role": "user", "content": question}]
    # 总超时只包住 Agent 运行这一段：历史里的输入是上一轮已经完成的，
    # 超时后重试时不会把它们重新跑一遍。
    async with asyncio.timeout(config.OVERALL_TIMEOUT):
        result = await Runner.run(notes_qa_agent, input_items)
    usage = result.context_wrapper.usage
    print(
        f"[用量] 请求={usage.requests} "
        f"输入={usage.input_tokens} 输出={usage.output_tokens}",
        file=sys.stderr,
    )
    # to_input_list() 把本轮所有消息（工具调用 + 回答）序列化，供下一轮续接
    return result.final_output, result.to_input_list()


async def main_loop() -> None:
    history: list = []
    print("笔记问答助手已启动。输入问题，或 /exit 退出，/new 开新会话。")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not question:
            continue
        if question in ("/exit", "/quit"):
            print("再见。")
            break
        if question == "/new":
            history = []
            print("已清空历史，开新会话。")
            continue

        err = check_input(question)
        if err:
            print(f"[跳过] {err}")
            continue

        try:
            answer, history = await ask(question, history)
        except Exception as e:
            print(f"[错误] {type(e).__name__}: {e}")
            continue

        print(answer)


if __name__ == "__main__":
    asyncio.run(main_loop())
