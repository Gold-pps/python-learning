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

# ---- 会话历史的两个上限 ----
# history 只增不减的话，输入 token 会随会话长度线性增长：一轮提问的完整记录
# 里包含工具调用的返回原文（read_note 一次最多 200 行），几轮之后就能膨胀到上万字符。
# 这两条上限把"每一轮要发给模型的东西"钉在常数规模：
#   MAX_HISTORY_TURNS      保留最近多少轮对话，更早的整轮丢弃；
#   MAX_HISTORY_ITEM_CHARS 单个条目保留多少字符，超长部分截断。
# 放在 main.py 而不是 config.py：config.py 管的是"怎么连模型"（地址、超时、重试），
# 这两条管的是"这个命令行会话怎么记事儿"，与 MAX_QUESTION_LEN 同属一类。
MAX_HISTORY_TURNS = 8
MAX_HISTORY_ITEM_CHARS = 2000


def check_input(question: str) -> str | None:
    """返回错误提示，或 None 表示合法。"""
    if len(question) > MAX_QUESTION_LEN:
        return f"问题过长（{len(question)} 字符），上限 {MAX_QUESTION_LEN} 字符。"
    return None


def _truncate_text(text: str, limit: int = MAX_HISTORY_ITEM_CHARS) -> str:
    """超长文本截断，并在结尾留下可观测的痕迹。"""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…[历史截断，原文 {len(text)} 字符]"


def _trim_item(item):
    """把一个历史条目里过长的文本截断，返回**新对象**（不改调用方给的原对象）。"""
    if not isinstance(item, dict):
        # chat_completions 路径下条目是 dict；万一以后换成 Responses 路径拿到的是
        # 对象，就原样放行，宁可少截一次也不要写坏结构。
        return item
    trimmed = dict(item)

    content = trimmed.get("content")
    if isinstance(content, str):
        # 用户消息（≤500 字符）和模型回答都走这里
        trimmed["content"] = _truncate_text(content)
    elif isinstance(content, list):
        # 富文本内容：只截其中的 text 片段，其他字段原样保留
        trimmed["content"] = [
            {**part, "text": _truncate_text(part["text"])}
            if isinstance(part, dict) and isinstance(part.get("text"), str)
            else part
            for part in content
        ]

    # 工具返回：Chat Completions 路径下是一个 JSON 字符串，装着整个工具返回 dict
    output = trimmed.get("output")
    if isinstance(output, str):
        trimmed["output"] = _truncate_text(output)
    return trimmed


def _keep_recent_turns(history: list) -> list:
    """只保留最近 MAX_HISTORY_TURNS 轮。

    **必须从 user 消息处切开**：一条 user 消息 + 它引发的 function_call /
    function_call_output / 最终回答才是完整的一轮。如果从中间切，history 里就会出现
    没有对应 function_call 的 function_call_output（孤儿工具结果），发出去会被
    API 直接拒掉——这是"截断"最容易踩的坑，而且只在被截断的那一次才暴露。
    """
    starts = [
        i for i, item in enumerate(history)
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    if len(starts) <= MAX_HISTORY_TURNS:
        return history          # 不超限就原样返回，避免无谓复制
    return history[starts[-MAX_HISTORY_TURNS]:]


def prepare_history(history: list) -> list:
    """把 history 收敛到常数规模：先按轮裁剪，再截断超长条目。"""
    kept = _keep_recent_turns(history)
    if len(kept) < len(history):
        print(
            f"[历史] 超过 {MAX_HISTORY_TURNS} 轮，丢弃较早的 {len(history) - len(kept)} 条",
            file=sys.stderr,
        )
    return [_trim_item(item) for item in kept]


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
    # 收敛动作只放在这一个入口：history 是"只写这里、只读下一轮"，所以
    # 在这里截一次就够了，不需要在每个用到 history 的地方各截一遍。
    return result.final_output, prepare_history(result.to_input_list())


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
