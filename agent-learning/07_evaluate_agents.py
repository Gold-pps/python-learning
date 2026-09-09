"""第 7 周：本地 Agent 评估脚本。

思路：把“问题 → 期望结果”写成几条可重复运行的评估用例，
运行前几周的 Demo 脚本并检查输出是否包含期望内容。
每条用例都会真实调用 DeepSeek（消耗少量 token），适合多次运行观察稳定性。

运行方式（在激活的虚拟环境中）：
    python 07_evaluate_agents.py
"""

import subprocess
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parent


def run_script(filename: str) -> str:
    """运行 agent-learning 下的脚本，返回它的完整输出。"""
    proc = subprocess.run(
        [sys.executable, str(AGENT_DIR / filename)],
        capture_output=True,
        text=True,
        cwd=str(AGENT_DIR),
        timeout=180,
    )
    if proc.returncode != 0:
        return proc.stdout + proc.stderr
    return proc.stdout


def check_output(output: str, expected: list[str]) -> list[str]:
    """返回缺失的期望片段；全部命中时返回空列表。"""
    lowered = output.casefold()
    return [word for word in expected if word.casefold() not in lowered]


CASES = [
    {
        "name": "第3周：文件工具读取",
        "script": "02_agent_with_tool.py",
        "expected": ["工具被调用", "find_student_home", "cain", "bible"],
        "note": "Agent 必须调用工具读文件，才能答出 cain 的家乡。",
    },
    {
        "name": "第4周：多轮记忆",
        "script": "03_multiturn_memory.py",
        "expected": ["张伟", "机械设计制造及其自动化", "深度学习入门"],
        "note": "Agent 必须利用 history 记住连续对话中的个人信息。",
    },
    {
        "name": "第5周：多 Agent 交接",
        "script": "04_multi_agent_handoff.py",
        "expected": ["history_teacher", "math_teacher", "9"],
        "note": "历史题交给 history_teacher，数学题交给 math_teacher。",
    },
]


def main() -> None:
    print(f"评估目录：{AGENT_DIR}")
    passed = 0
    for case in CASES:
        print("\n" + "=" * 50)
        print(f"用例：{case['name']}")
        print(f"说明：{case['note']}")
        output = run_script(case["script"])
        missing = check_output(output, case["expected"])
        if missing:
            print("结果：FAIL")
            print("缺少期望内容：", ", ".join(missing))
        else:
            print("结果：PASS")
            passed += 1
        print("-" * 50)
        # 只展示最后几行输出，避免刷屏
        tail_lines = output.strip().splitlines()[-6:]
        for line in tail_lines:
            print(line)

    print("\n" + "=" * 50)
    print(f"评估结果：{passed}/{len(CASES)} 通过")
    if passed != len(CASES):
        print("提示：模型输出有随机性，可多运行几次观察通过率。")
    print("注意：每次运行都会消耗 DeepSeek token，记得去平台看账单。")


if __name__ == "__main__":
    main()
