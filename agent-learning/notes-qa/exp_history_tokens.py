"""T3 实测：会话历史收敛前后的真实 token 曲线（第 13 周的一次性实验脚本）。

产物（数字）记入 `../第13周笔记.md` 的实测记录，跑完可以删掉本文件。

用法（在 notes-qa 目录下）：

    python exp_history_tokens.py --turns 8                  # 修复后：history 收敛
    python exp_history_tokens.py --turns 8 --no-trim        # 修复前：history 只增不减

    python exp_history_tokens.py --turns 8 --json after.json

两臂的差别**只有一处**：`--no-trim` 时把 `result.to_input_list()` 原样存下来，
等价于 T3 修复前的那一行；其余代码路径完全相同，因此曲线可比。

省钱提示：高峰时段是工作日 09:00-12:00、14:00-18:00（北京时间），其余时间和周末是低谷价
（半价）。8 轮 × 两臂的总量在几万 token 量级，实际花费不到一毛钱，但请放在低谷跑。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import config  # noqa: F401  —— import 之后所有请求自动发往 DeepSeek
from agents import ModelSettings, RunConfig, Runner

import main
from agent import notes_qa_agent

# 固定问题表：必须能触发工具调用，才会把 read_note 的返回正文灌进历史。
# 8 条问题刚好一轮一个，超过 8 轮就循环使用（历史里会重复出现，正好放大差异）。
QUESTIONS = [
    "第 7 周笔记里关于成本的那部分讲了什么？",
    "第 10 周做了哪些输入边界检查？",
    "护栏和人工审批的区别是什么？",
    "第 11 周发现了哪些真 bug？",
    "notes-qa 的工具是怎么防路径穿越的？",
    "第 12 周演示稿分成哪几个部分？",
    "第 9 周的项目 MVP 包含哪几个文件？",
    "第 6 周的两个护栏分别拦什么？",
]

# deepseek-flash 单价（每 100 万 token，美元，2026-09 快照，见第 7 周笔记）
PRICE = {
    "peak": {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2},
    "off_peak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.6},
}


def _chars(items: list) -> int:
    """history 的体积（字符数），作为 token 之外的第二个观测量。"""
    return sum(len(json.dumps(item, ensure_ascii=False)) for item in items)


def _cached_tokens(usage) -> int:
    """缓存命中的输入 token。命中价只有未命中的约 1/50，不记这一项会高估成本。"""
    return getattr(usage.input_tokens_details, "cached_tokens", 0) or 0


async def run(turns: int, trim: bool, temperature: float) -> list[dict]:
    history: list = []
    rows: list[dict] = []
    # Runner.run() 本身不收 model_settings：采样参数要走 run_config。
    # （注意与 agent 自身的 model_settings 的优先级：agent 上设的会盖过 run_config，
    #  本项目 notes_qa_agent 没设，所以这里生效。）
    run_config = RunConfig(model_settings=ModelSettings(temperature=temperature))

    for i in range(turns):
        question = QUESTIONS[i % len(QUESTIONS)]
        input_items = history + [{"role": "user", "content": question}]

        started = time.perf_counter()
        async with asyncio.timeout(config.OVERALL_TIMEOUT):
            result = await Runner.run(notes_qa_agent, input_items, run_config=run_config)
        elapsed = time.perf_counter() - started

        usage = result.context_wrapper.usage
        raw = result.to_input_list()
        # 唯一的分叉点：修复后收一次，修复前原样留着
        stored = main.prepare_history(raw) if trim else raw

        row = {
            "turn": i + 1,
            "input_tokens": usage.input_tokens,
            "cached_input_tokens": _cached_tokens(usage),
            "output_tokens": usage.output_tokens,
            "requests": usage.requests,
            "history_items": len(stored),
            "history_chars": _chars(stored),
            "seconds": round(elapsed, 2),
        }
        rows.append(row)
        print(
            f"第 {row['turn']:>2} 轮 | 输入 {row['input_tokens']:>6}"
            f"（缓存命中 {row['cached_input_tokens']:>6}）"
            f" | 输出 {row['output_tokens']:>5}"
            f" | history {row['history_items']:>3} 条 / {row['history_chars']:>6} 字符"
            f" | {row['seconds']:>5.1f}s",
            flush=True,
        )
        history = stored
    return rows


def summarize(label: str, rows: list[dict]) -> None:
    if not rows:
        return
    first, last = rows[0], rows[-1]
    miss_in = sum(r["input_tokens"] - r["cached_input_tokens"] for r in rows)
    hit_in = sum(r["cached_input_tokens"] for r in rows)
    out = sum(r["output_tokens"] for r in rows)

    print(f"\n=== {label} ===")
    print(f"输入 token：第 1 轮 {first['input_tokens']} → 第 {last['turn']} 轮 {last['input_tokens']}"
          f"（倍数 {last['input_tokens'] / max(first['input_tokens'], 1):.2f}×）")
    print(f"history  ：第 1 轮 {first['history_chars']} 字符 → 第 {last['turn']} 轮 {last['history_chars']} 字符")
    print(f"合计：输入 {hit_in + miss_in}（缓存命中 {hit_in} / 未命中 {miss_in}）、输出 {out}")
    for name, price in PRICE.items():
        cost = (miss_in * price["cache_miss"] + hit_in * price["cache_hit"] + out * price["output"]) / 1e6
        print(f"  按{'高峰' if name == 'peak' else '低谷'}价估算：${cost:.4f}（约 {cost * 7.1:.3f} 元）")

    # 直接贴进笔记的表格行
    print("\n贴进笔记用：")
    for r in rows:
        print(f"| {r['turn']} | {r['input_tokens']} | {r['cached_input_tokens']} "
              f"| {r['history_chars']} | {r['seconds']} |")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="会话历史 token 曲线实测")
    parser.add_argument("--turns", type=int, default=8, help="问答轮数（默认 8）")
    parser.add_argument("--no-trim", action="store_true",
                        help="不收敛 history（等价于 T3 修复前的行为）")
    parser.add_argument("--temp", type=float, default=0.0,
                        help="采样温度，默认 0 让两臂更可比")
    parser.add_argument("--json", dest="json_path", help="把原始数据存成 JSON 文件")
    args = parser.parse_args()

    label = "修复前（不收敛）" if args.no_trim else "修复后（收敛）"
    print(f"开始：{label}，{args.turns} 轮，temperature={args.temp}\n")
    rows = asyncio.run(run(args.turns, trim=not args.no_trim, temperature=args.temp))
    summarize(label, rows)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump({"label": label, "turns": args.turns, "rows": rows},
                      fh, ensure_ascii=False, indent=2)
        print(f"\n原始数据已写入 {args.json_path}")


if __name__ == "__main__":
    main_cli()
