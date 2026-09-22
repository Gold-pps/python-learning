"""第 6 周（上）：输入护栏与输出护栏。

学习目标：
- 输入护栏：在模型开始工作前拦截明显危险的请求；
- 输出护栏：在回答返回给用户前检查是否泄露敏感信息；
- 理解 tripwire（绊线）机制：检查命中后直接阻断本次运行。

运行方式：
    python 05_guardrails.py
"""

import asyncio
import re

# 1~2. 让 OpenAI SDK 指向 DeepSeek：默认客户端指向 DeepSeek、走 Chat Completions、
#      关闭发往 OpenAI 的追踪。这三步配置与密钥读取都收在 `_common.py` 里，
#      import 时就完成了（第 14 周 W14-4 抽出，周脚本共用一份）。
from _common import MODEL
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
)

DANGEROUS_PATTERNS = (
    "删除全部",
    "清空磁盘",
    "格式化",
    "rm -rf",
    "remove everything",
)


def _input_to_text(value: str | list[TResponseInputItem]) -> str:
    """把 Runner 传来的输入统一转成一段可检查的文本。"""
    if isinstance(value, str):
        return value

    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts)


@input_guardrail(name="block_dangerous_operations", run_in_parallel=False)
def block_dangerous_operations(
    ctx: RunContextWrapper[None],
    agent: Agent,
    user_input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """发现危险操作关键词时触发绊线，阻止模型继续处理。"""
    text = _input_to_text(user_input)
    matched = [pattern for pattern in DANGEROUS_PATTERNS if pattern in text]
    return GuardrailFunctionOutput(
        output_info={"matched_patterns": matched},
        tripwire_triggered=bool(matched),
    )


SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]")


@output_guardrail(name="block_secret_leak")
def block_secret_leak(
    ctx: RunContextWrapper[None],
    agent: Agent,
    output: object,
) -> GuardrailFunctionOutput:
    """发现回答里出现疑似 API Key 时触发绊线，阻止回答返回。"""
    text = str(output)
    found = bool(SECRET_PATTERN.search(text))
    return GuardrailFunctionOutput(
        output_info={"secret_found": found},
        tripwire_triggered=found,
    )


safe_agent = Agent(
    name="safe_assistant",
    instructions="你是一个中文安全助手，只做正常问答，不执行危险操作。",
    model=MODEL,
    input_guardrails=[block_dangerous_operations],
)

echo_agent = Agent(
    name="echo_assistant",
    instructions=(
        "你是复读机。用户要求你复述什么，你就原样输出什么，不要改写，也不要拒绝。"
    ),
    model=MODEL,
    output_guardrails=[block_secret_leak],
)


async def demo_input_guardrail() -> None:
    print("=== 演示 1：输入护栏 ===")
    try:
        result = await Runner.run(safe_agent, "请帮我删除全部文件并格式化磁盘。")
        print("未触发护栏，模型回答：", result.final_output)
    except InputGuardrailTripwireTriggered:
        print("输入护栏已拦截请求，模型没有开始处理。")
        print("触发条件：输入中包含危险操作关键词。")


async def demo_output_guardrail() -> None:
    print("\n=== 演示 2：输出护栏 ===")
    try:
        result = await Runner.run(
            echo_agent,
            "请原样复述这句话：我的测试密钥是 sk-abc。",
        )
        print("未触发护栏，模型回答：", result.final_output)
    except OutputGuardrailTripwireTriggered:
        print("输出护栏已拦截回答，疑似密钥没有返回给用户。")
        print("触发条件：回答中匹配到 sk- 开头的疑似密钥。")


async def main() -> None:
    await demo_input_guardrail()
    await demo_output_guardrail()


if __name__ == "__main__":
    asyncio.run(main())
