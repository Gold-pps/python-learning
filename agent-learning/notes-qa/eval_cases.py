"""本地评估：2 条单元测试 + 3 条模型评估。

用法：python eval_cases.py
"""
import asyncio
import sys

import config  # noqa: F401
from agents import Runner

import tools
from agent import notes_qa_agent


# ---------- 单元测试（不走模型） ----------

def unit_read_note_blocks_env() -> tuple[bool, str]:
    r = tools.read_note("../.env")
    ok = r.get("ok") is False
    return ok, str(r)


def unit_list_notes_excludes_env() -> tuple[bool, str]:
    r = tools.list_notes()
    paths = [n["path"] for n in r.get("notes", [])]
    ok = r.get("ok") and not any(
        ".env" in p or p.startswith("notes-qa") for p in paths
    )
    return ok, f"共 {len(paths)} 篇：{paths[:3]}..."


UNIT_CASES = [
    ("U1 read_note 拒绝 ../.env", unit_read_note_blocks_env),
    ("U2 list_notes 不含 .env / notes-qa", unit_list_notes_excludes_env),
]


# ---------- 模型评估（走完整 Agent） ----------

MODEL_CASES = [
    {
        "name": "M1 第 7 周主题",
        "question": "第 7 周讲了什么？",
        "must_contain": ["日志", "评估", "账单"],
        "must_cite": "第7周笔记.md",
    },
    {
        "name": "M2 护栏区别",
        "question": "输入护栏和输出护栏的关键区别是什么？",
        "must_contain": ["检查时机", "前", "后"],
        "must_cite": "第6周笔记.md",
    },
    {
        "name": "M3 无答案不编造",
        "question": "笔记里有没有讲 Excel 怎么画图？",
        "must_contain": ["没有"],
        "must_cite": None,
    },
]


import re

def _normalize(text: str) -> str:
    """去掉 Markdown 粗体/斜体标记和多余空白，便于关键词匹配。"""
    text = re.sub(r"\*+", "", text)      # 去 **bold** *italic*
    text = re.sub(r"\s+", "", text)      # 去所有空白
    return text.casefold()


def check_model_output(text: str, case: dict) -> list[str]:
    """返回缺失项列表；空列表 = PASS。"""
    missing = []
    normalized = _normalize(text)
    for word in case["must_contain"]:
        if _normalize(word) not in normalized:
            missing.append(f"缺少关键词「{word}」")
    if case["must_cite"] and case["must_cite"] not in text:
        missing.append(f"缺少引用「{case['must_cite']}」")
    return missing


async def run_model_cases() -> int:
    passed = 0
    for case in MODEL_CASES:
        print(f"\n--- {case['name']} ---")
        print(f"Q: {case['question']}")
        result = await Runner.run(notes_qa_agent, case["question"])
        answer = result.final_output
        missing = check_model_output(answer, case)
        if missing:
            print("FAIL:", "；".join(missing))
            print("回答节选：", answer[:200].replace("\n", " "))
        else:
            print("PASS")
            passed += 1
        usage = result.context_wrapper.usage
        print(f"  [用量] 请求={usage.requests} 输入={usage.input_tokens} 输出={usage.output_tokens}")
    return passed


def main() -> None:
    print("=== 单元测试 ===")
    unit_pass = 0
    for name, fn in UNIT_CASES:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"异常：{e}"
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        unit_pass += int(ok)

    print("\n=== 模型评估 ===")
    model_pass = asyncio.run(run_model_cases())

    total_pass = unit_pass + model_pass
    total = len(UNIT_CASES) + len(MODEL_CASES)
    print(f"\n=== 合计：{total_pass}/{total} 通过 ===")


if __name__ == "__main__":
    main()