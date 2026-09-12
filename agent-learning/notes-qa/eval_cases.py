"""本地评估：12 条单元测试 + 3 条模型评估。

用法：python eval_cases.py
"""
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

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
        Path(p).name.startswith(".") or p == "notes-qa" or p.startswith("notes-qa/")
        for p in paths
    )
    return ok, f"共 {len(paths)} 篇：{paths[:3]}..."

def unit_read_note_blocks_excluded_dir() -> tuple[bool, str]:
    r = tools.read_note("notes-qa/项目设计.md")
    ok = r.get("ok") is False and "排除目录" in r.get("error", "")
    return ok, str(r)

def unit_read_note_blocks_outside_path() -> tuple[bool, str]:
    '''真正走到"路径越界"分支的越界路径（.md 结尾，dot/dotdot 在中间）。'''
    r = tools.read_note("sub/../../../secret.md")
    ok = r.get("ok") is False and "越界" in r.get("error", "")
    return ok, str(r)

def unit_read_note_blocks_symlink_escape() -> tuple[bool, str]:
    """根目录内指向根目录外的链接必须被拒绝；无权限建链接时按跳过处理。"""
    link = tools.config.NOTES_ROOT / "_probe_link.md"
    # 用 lexists：断掉的链接 exists() 是 False，不能让它混过去
    if os.path.lexists(link):
        return False, f"临时路径已存在：{link}"
    target = tools.config.NOTES_ROOT.parent / "_probe_secret.md"
    try:
        target.write_text("secret\n", encoding="utf-8")
        try:
            link.symlink_to(target)
        except OSError as e:
            return True, f"跳过：当前账户无法创建符号链接（{type(e).__name__}, WinError {e.winerror}）"
        r = tools.read_note("_probe_link.md")
        ok = r.get("ok") is False and "越界" in r.get("error", "")
        return ok, str(r)
    finally:
        try:
            if link.is_symlink():
                link.unlink()
        except OSError:
            pass
        target.unlink(missing_ok=True)

def unit_list_notes_skips_link_escape() -> tuple[bool, str]:
    """链接指向根目录外时，list_notes 不能枚举、search_notes 不能搜到、read_note 不能读取。

    链接名故意用 .md 结尾：这样它会被 rglob("*.md") 扫到，能真正考验
    search_notes / list_notes 里的 resolve() 复核。Windows 上普通用户建不了
    符号链接但能建 junction，所以两种链接都覆盖。
    """
    root = tools.config.NOTES_ROOT
    link = root / "_probe_linkdir.md"
    target = root.parent / "_probe_linkdir_target"
    if os.path.lexists(link) or os.path.lexists(target):
        return False, f"临时路径已存在：{link}"
    cleanup_cmd = None
    try:
        target.mkdir(parents=True, exist_ok=True)
        (target / "_probe_outside.md").write_text("OUTSIDE-SECRET\n", encoding="utf-8")
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as e:
            if os.name != "nt":
                return True, f"跳过：当前账户无法创建目录链接（{type(e).__name__}, WinError {e.winerror}）"
            created = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"New-Item -ItemType Junction -Path '{link}' -Target '{target}' | Out-Null",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if created.returncode != 0:
                return False, f"junction 创建失败：{created.stderr.strip()[:200] or created.stdout.strip()[:200]}"
            cleanup_cmd = ["powershell", "-NoProfile", "-Command", f"Remove-Item -LiteralPath '{link}' -Force"]
        listed = tools.list_notes()
        leaked = [n["path"] for n in listed.get("notes", []) if "_probe_linkdir" in n["path"] or "_probe_outside" in n["path"]]
        searched = tools.search_notes("OUTSIDE-SECRET")
        search_leak = searched.get("matches", [])
        read_via_link = tools.read_note("_probe_linkdir.md/_probe_outside.md")
        ok = (
            listed.get("ok") is True
            and not leaked
            and searched.get("ok") is True
            and not search_leak
            and read_via_link.get("ok") is False
            and "越界" in read_via_link.get("error", "")
        )
        if leaked:
            return False, f"列表泄露 {leaked}"
        if search_leak:
            return False, f"搜索泄露 {search_leak}"
        if read_via_link.get("ok") is not False:
            return False, f"经 .md 链接竟被读出：{read_via_link}"
        return ok, (
            f"列表未枚举（共 {len(listed.get('notes', []))} 篇），搜索 0 命中，"
            f"读取被拒：{read_via_link.get('error')}"
        )
    finally:
        # 链接必须先删。注意 is_junction()：断掉的 junction 不算 symlink，
        # 单靠 is_symlink() 会漏删，留下一个谁也读不到的链接目录。
        try:
            if link.is_symlink() or link.is_junction():
                os.rmdir(link)
            elif cleanup_cmd:
                subprocess.run(cleanup_cmd, check=False, capture_output=True)
        except OSError:
            pass
        shutil.rmtree(target, ignore_errors=True)

def unit_search_notes_clamps_and_validates() -> tuple[bool, str]:
    r = tools.search_notes("笔记", 99)
    ok = r.get("ok") is True and len(r.get("matches", [])) <= tools.MAX_SEARCH_RESULTS
    detail = f"按上限收敛到 {len(r.get('matches', []))} 条"
    r = tools.search_notes("x" * (tools.MAX_KEYWORD_LEN + 1))
    ok = ok and r.get("ok") is False and "过长" in r.get("error", "")
    return ok, detail + "；超长关键词被拒绝"

def unit_tools_do_not_raise_on_bad_types() -> tuple[bool, str]:
    """畸形参数（模型完全可能传出来）必须返回结构化错误，而不是抛异常。"""
    proxies = [
        ("read_note(None)", lambda: tools.read_note(None)),
        ("search_notes('x', None)", lambda: tools.search_notes("x", None)),
        ("search_notes('x', 'many')", lambda: tools.search_notes("x", "many")),
        ("read_note(.., start_line='abc')", lambda: tools.read_note("学习进度.md", start_line="abc")),
    ]
    details = []
    ok = True
    for label, fn in proxies:
        try:
            r = fn()
        except Exception as e:
            ok = False
            details.append(f"{label} 抛出 {type(e).__name__}")
            continue
        if r.get("ok") is not False:
            ok = False
            details.append(f"{label} 未返回错误：{r}")
        else:
            details.append(f"{label} -> {r.get('error')}")
    return ok, "；".join(details)

def unit_read_note_rejects_oversize() -> tuple[bool, str]:
    p = tools.config.NOTES_ROOT / "_probe_big.md"
    try:
        p.write_text("x" * (tools.MAX_FILE_BYTES + 1), encoding="utf-8", newline="\n")
        r = tools.read_note("_probe_big.md")
        ok = r.get("ok") is False and "过大" in r.get("error", "")
        return ok, str(r)
    finally:
        p.unlink(missing_ok=True)

def unit_read_note_clamps_and_truncation_flag() -> tuple[bool, str]:
    p = tools.config.NOTES_ROOT / "_probe_lines.md"
    p_short = tools.config.NOTES_ROOT / "_probe_lines_short.md"
    try:
        p.write_text("".join(f"line-{i}\n" for i in range(1, 251)), encoding="utf-8", newline="\n")
        p_short.write_text("".join(f"line-{i}\n" for i in range(1, 101)), encoding="utf-8", newline="\n")
        clamped = tools.read_note("_probe_lines.md", 1, 9999)
        caught = tools.read_note("_probe_lines_short.md")
        ok = (
            clamped.get("ok") is True
            and clamped.get("end_line") == tools.MAX_READ_LINES
            and clamped.get("truncated") is True
            and clamped.get("total_lines") == 250
            and caught.get("ok") is True
            and caught.get("end_line") == 100
            and caught.get("truncated") is False
        )
        return ok, (
            f"请求 9999 行 -> end_line={clamped.get('end_line')} truncated={clamped.get('truncated')}；"
            f"默认读到结尾 -> end_line={caught.get('end_line')} truncated={caught.get('truncated')}"
        )
    finally:
        p.unlink(missing_ok=True)
        p_short.unlink(missing_ok=True)

def unit_check_input_boundaries() -> tuple[bool, str]:
    from main import MAX_QUESTION_LEN, check_input
    at_limit = check_input("a" * MAX_QUESTION_LEN)
    over_limit = check_input("a" * (MAX_QUESTION_LEN + 1))
    ok = at_limit is None and over_limit is not None and "过长" in over_limit
    return ok, f"边界 {MAX_QUESTION_LEN}：{at_limit}；超一字符：{over_limit}"

def unit_agent_timeout_config() -> tuple[bool, str]:
    """第 10 周修复的卡死问题：请求层和整轮都要有上限，且别退化成无限等待。"""
    ok = (
        isinstance(config.REQUEST_TIMEOUT, (int, float))
        and config.REQUEST_TIMEOUT > 0
        and isinstance(config.OVERALL_TIMEOUT, (int, float))
        and config.OVERALL_TIMEOUT > 0
        and isinstance(config.MAX_RETRIES, int)
        and config.MAX_RETRIES >= 0
    )
    return ok, (
        f"REQUEST_TIMEOUT={config.REQUEST_TIMEOUT} MAX_RETRIES={config.MAX_RETRIES} "
        f"OVERALL_TIMEOUT={config.OVERALL_TIMEOUT}"
    )

UNIT_CASES = [
    ("U1 read_note 拒绝 ../.env", unit_read_note_blocks_env),
    ("U2 list_notes 不含隐藏文件 / notes-qa", unit_list_notes_excludes_env),
    ("U3 read_note 拒绝 notes-qa/", unit_read_note_blocks_excluded_dir),
    ("U4 read_note 拒绝根目录外路径（真正走到越界分支）", unit_read_note_blocks_outside_path),
    ("U5 read_note 拒绝指向根外的符号链接", unit_read_note_blocks_symlink_escape),
    ("U6 list_notes / search_notes 不跟随链接列出或搜到根外文件", unit_list_notes_skips_link_escape),
    ("U7 search_notes 收敛条数并拒绝超长关键词", unit_search_notes_clamps_and_validates),
    ("U8 工具对畸形参数返回结构化错误", unit_tools_do_not_raise_on_bad_types),
    ("U9 read_note 拒绝超过 1MB 的文件", unit_read_note_rejects_oversize),
    ("U10 read_note 单次最多 200 行，truncated 表示未到文件结尾", unit_read_note_clamps_and_truncation_flag),
    ("U11 check_input 500 字符边界", unit_check_input_boundaries),
    ("U12 超时与重试配置有效", unit_agent_timeout_config),
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
        "must_contain": ["检查时机", "输入护栏", "输出护栏"],
        "must_cite": "第6周笔记.md",
    },
    {
        "name": "M3 无答案不编造",
        "question": "笔记里有没有讲 Excel 怎么画图？",
        "must_contain": ["没有找到"],
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
    if case["must_cite"] and _normalize(case["must_cite"]) not in normalized:
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="只跑单元测试，跳过模型评估（无需网络）",
    )
    args = parser.parse_args()

    print("=== 单元测试 ===")
    unit_pass = 0
    for name, fn in UNIT_CASES:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"异常：{e}"
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        unit_pass += int(ok)

    if args.unit_only:
        print(f"\n=== 合计：{unit_pass}/{len(UNIT_CASES)} 通过（已跳过模型评估）===")
        return

    print("\n=== 模型评估 ===")
    model_pass = asyncio.run(run_model_cases())

    total_pass = unit_pass + model_pass
    total = len(UNIT_CASES) + len(MODEL_CASES)
    print(f"\n=== 合计：{total_pass}/{total} 通过 ===")


if __name__ == "__main__":
    main()
