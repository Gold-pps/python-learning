"""本地评估：12 条单元测试 + 3 条模型评估。

用法：python eval_cases.py
"""

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import config
import tools
from agent import notes_qa_agent
from agents import Runner
from agents.exceptions import MaxTurnsExceeded

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
    """真正走到"路径越界"分支的越界路径（.md 结尾，dot/dotdot 在中间）。"""
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
            return (
                True,
                f"跳过：当前账户无法创建符号链接（{type(e).__name__}, WinError {e.winerror}）",
            )
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
                return (
                    True,
                    f"跳过：当前账户无法创建目录链接（{type(e).__name__}, WinError {e.winerror}）",
                )
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
                return (
                    False,
                    f"junction 创建失败：{created.stderr.strip()[:200] or created.stdout.strip()[:200]}",
                )
            cleanup_cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Remove-Item -LiteralPath '{link}' -Force",
            ]
        listed = tools.list_notes()
        leaked = [
            n["path"]
            for n in listed.get("notes", [])
            if "_probe_linkdir" in n["path"] or "_probe_outside" in n["path"]
        ]
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
        # 链接必须先删，且两种链接要用两种删法：
        # - 符号链接：os.unlink（对链接调用 os.rmdir 会抛 NotADirectoryError，
        #   链接留在原地，下次运行 U6 会因为"临时路径已存在"直接失败）；
        # - junction（Windows）：os.rmdir。is_junction() 在 Linux 上恒为 False，
        #   两个分支互不干扰；断掉的 junction 不算 symlink，少一个分支就会漏删。
        try:
            if link.is_symlink():
                os.unlink(link)
            elif link.is_junction():
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
        (
            "read_note(.., start_line='abc')",
            lambda: tools.read_note("学习进度.md", start_line="abc"),
        ),
    ]
    details = []
    ok = True
    for label, fn in proxies:
        try:
            r = fn()
        # 这组用例专门验证"畸形参数不抛异常"：任何异常都必须被记为失败，
        # 所以这里刻意 catch-all，而不是只抓特定类型
        except Exception as e:  # noqa: BLE001
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
        p.write_text(
            "".join(f"line-{i}\n" for i in range(1, 251)),
            encoding="utf-8",
            newline="\n",
        )
        p_short.write_text(
            "".join(f"line-{i}\n" for i in range(1, 101)),
            encoding="utf-8",
            newline="\n",
        )
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
    (
        "U4 read_note 拒绝根目录外路径（真正走到越界分支）",
        unit_read_note_blocks_outside_path,
    ),
    ("U5 read_note 拒绝指向根外的符号链接", unit_read_note_blocks_symlink_escape),
    (
        "U6 list_notes / search_notes 不跟随链接列出或搜到根外文件",
        unit_list_notes_skips_link_escape,
    ),
    (
        "U7 search_notes 收敛条数并拒绝超长关键词",
        unit_search_notes_clamps_and_validates,
    ),
    ("U8 工具对畸形参数返回结构化错误", unit_tools_do_not_raise_on_bad_types),
    ("U9 read_note 拒绝超过 1MB 的文件", unit_read_note_rejects_oversize),
    (
        "U10 read_note 单次最多 200 行，truncated 表示未到文件结尾",
        unit_read_note_clamps_and_truncation_flag,
    ),
    ("U11 check_input 500 字符边界", unit_check_input_boundaries),
    ("U12 超时与重试配置有效", unit_agent_timeout_config),
]

# ---------- 模型评估（走完整 Agent） ----------

# 四类标注（有答案 / 无答案 / 跨文档 / 需要多跳）与第 19 周的检索评测集共用一套分类，
# 不是另起一套。每条用例的"答案确实在资料里"都事先 grep 核实过——
# 一道本身无解的题会被误读成"模型能力差"，这是评测集最贵的错误。
#
# 断言口径（比原来的"单个关键词 + 单个文件子串"严格）：
#   must_contain        全部关键词都要出现
#   must_contain_any    至少出现其一（用于"拒答"这类措辞不唯一的场景）
#   must_match_any      至少匹配其中一条正则（措辞完全无法穷举时用，如"没有找到/没有讲/未提及"）
#   must_cite_files     这些文件的引用必须都出现
#   must_cite_any_of    至少引用其中某一个（同一事实可能散在几篇笔记里）
#   min_distinct_files  引用到的不同文件数下限（跨文档/多跳类的主要判据）
#   require_bracket_citation  引用必须写成约定的 [文件名:行号]（格式合规，独立于内容对错）
MODEL_CASES = [
    {
        "name": "A1 第 7 周做了哪几件事",
        "category": "有答案",
        "question": "第 7 周这一周主要做了哪几件事？",
        "must_contain": ["日志", "评估", "账单"],
        # 第 7 周的事在 `第7周笔记.md`、`学习进度.md`、`学习Agent规划.md` 里都有记载，
        # 指定唯一出处会制造假失败（实测：同一问题两次运行引用了不同的文件）。
        "must_cite_any_of": ["第7周笔记.md", "学习进度.md", "学习Agent规划.md"],
        "require_bracket_citation": True,
    },
    {
        "name": "A2 两个护栏的运行时机",
        "category": "有答案",
        "question": "输入护栏和输出护栏分别在什么时机运行？",
        "must_contain": ["输入护栏", "输出护栏"],
        "must_cite_files": ["第6周笔记.md"],
        "require_bracket_citation": True,
    },
    {
        "name": "A3 路径安全的做法",
        "category": "有答案",
        "question": "第 9 周是怎么防止读取笔记目录之外文件的？",
        "must_contain": ["越界"],
        "must_cite_files": ["第9周笔记.md"],
        "require_bracket_citation": True,
    },
    {
        "name": "A4 账单金额（数字型答案）",
        "category": "有答案",
        "question": "2026-09-01 到 09-10 的 DeepSeek 账单是多少？",
        "must_contain": ["20.7878"],
        "must_cite_any_of": [
            "第7周笔记.md",
            "学习进度.md",
            "第12周演示稿.md",
            "学习Agent规划.md",
        ],
        "require_bracket_citation": True,
    },
    {
        "name": "B1 无答案：容器编排部署（资料里 0 命中）",
        "category": "无答案",
        "question": "笔记里有讲用 Rust 做系统级并发编程吗？",
        "must_match_any": [
            r"没有(找到|讲|提及|涉及|相关|收录|介绍)|未(找到|提及|涉及)|无相关"
        ],
        # 只给 tests/test_eval_cases.py 的"泄漏检查"用：这个词一旦出现在资料库里，
        # 这道题就失效了（模型可以直接抄资料库里的"已核实 0 命中"，而不必真的拒答）
        "negative_keywords": ["Rust"],
    },
    {
        "name": "B2 无答案：桌面界面框架（资料里 0 命中）",
        "category": "无答案",
        "question": "笔记里有没有讲怎么用 Kotlin 写移动端界面？",
        "must_match_any": [
            r"没有(找到|讲|提及|涉及|相关|收录|介绍)|未(找到|提及|涉及)|无相关"
        ],
        "negative_keywords": ["Kotlin"],
    },
    {
        "name": "C1 跨文档：哪些周做了安全工作",
        "category": "跨文档",
        "question": "第一阶段 12 周里，哪些周做了和安全相关的工作？",
        "must_contain_any": ["安全", "护栏", "越界", "拒绝"],
        "min_distinct_files": 2,
        "require_bracket_citation": True,
    },
    {
        "name": "C2 跨文档：notes-qa 从第 9 周到第 12 周的变化",
        "category": "跨文档",
        "question": "notes-qa 这个项目从第 9 周到第 12 周分别有什么变化？",
        "must_contain_any": ["第9周", "第10周"],
        "min_distinct_files": 3,
        "require_bracket_citation": True,
    },
    {
        "name": "D1 多跳：演示稿里的漏洞追到出处",
        "category": "需要多跳",
        "question": "第 12 周演示稿里提到的那个安全漏洞，最初是在哪一周、通过什么方式发现的？",
        "must_contain_any": ["junction", "逃逸"],
        "must_cite_files": ["第11周笔记.md"],
        "require_bracket_citation": True,
    },
    {
        "name": "D2 多跳：规划里的要求追到落地记录",
        "category": "需要多跳",
        "question": "第二阶段规划要求把 Codex 的明文 Key 改成环境变量注入，这件事在第 13 周是怎么落地的？",
        "must_contain": ["env_key"],
        "must_cite_files": ["工程化改造记录.md"],
        "min_distinct_files": 2,
        "require_bracket_citation": True,
    },
]


import re

# 引用格式：`agent.py` 的指令约定写成 [文件名:起始行-结束行]（也接受 [文件名:12] 这种单行号）。
#
# 但实测模型在长回答里会漂移成反引号：`工程化改造记录.md:158-166`。
# 所以解析分成两个：
#   - 内容判据用宽松的（两种都认），回答的是"有没有给出可回查的出处"；
#   - 格式合规单独判定（has_bracket_citation），回答的是"有没有守住输出契约"。
# 混在一起会把"格式不合规"误报成"没找到出处"——第 14 周 T10 就是这么踩到的。
_BRACKET_CITATION_RE = re.compile(r"\[([^\[\]:]+?\.md)\s*:[^\[\]]*\]")
_BACKTICK_CITATION_RE = re.compile(r"`([^`\[\]:]+?\.md)\s*:[^`\[\]]*`")


def _normalize(text: str) -> str:
    """去掉 Markdown 粗体/斜体标记和多余空白，便于关键词匹配。"""
    text = re.sub(r"\*+", "", text)  # 去 **bold** *italic*
    text = re.sub(r"\s+", "", text)  # 去所有空白
    return text.casefold()


def _norm_name(name: str) -> str:
    """文件名归一化：回答里可能写成「第 7 周笔记.md」，去掉空白再比。"""
    return re.sub(r"\s+", "", name)


def extract_cited_files(text: str) -> set[str]:
    """从回答里抽出被引用的文件名（归一化后），方括号与反引号两种写法都认。

    这比最初"文件名作为子串出现在回答里"严格得多：必须真的写出 `文件名:行号` 才算引用。
    比只认方括号的版本宽松：那种写法会把"格式漂移"误判成"没有引用"。
    """
    bracket = {_norm_name(m.group(1)) for m in _BRACKET_CITATION_RE.finditer(text)}
    backtick = {_norm_name(m.group(1)) for m in _BACKTICK_CITATION_RE.finditer(text)}
    return bracket | backtick


def has_bracket_citation(text: str) -> bool:
    """是否出现符合约定的 `[文件名:行号]` 引用（用于单独统计格式合规率）。"""
    return _BRACKET_CITATION_RE.search(text) is not None


def check_model_output(text: str, case: dict) -> list[str]:
    """返回缺失项列表；空列表 = PASS。"""
    missing = []
    normalized = _normalize(text)
    cited = extract_cited_files(text)

    for word in case.get("must_contain", []):
        if _normalize(word) not in normalized:
            missing.append(f"缺少关键词「{word}」")

    any_words = case.get("must_contain_any")
    if any_words and not any(_normalize(w) in normalized for w in any_words):
        missing.append(f"未出现其中任一必需表达：{any_words}")

    patterns = case.get("must_match_any")
    if patterns and not any(re.search(p, normalized) for p in patterns):
        missing.append(f"未匹配任一必需表达（正则）：{patterns}")

    required = {_norm_name(f) for f in case.get("must_cite_files", [])}
    if not required <= cited:
        missing.append(f"缺少引用：{sorted(required - cited)}")

    any_cites = case.get("must_cite_any_of")
    if any_cites and not cited & {_norm_name(f) for f in any_cites}:
        missing.append(f"未引用其中任一文件：{any_cites}")

    min_files = case.get("min_distinct_files")
    if min_files and len(cited) < min_files:
        missing.append(f"引用文件数 {len(cited)} < {min_files}：{sorted(cited)}")

    # 格式合规是独立判据：内容对了但格式漂移，也要如实记差评——
    # 第 23 周的 Web 界面要按这个格式解析来源，G2 的"引用准确率"也依赖它。
    if case.get("require_bracket_citation") and not has_bracket_citation(text):
        missing.append(
            "引用格式不合规：应写成 [文件名:起始行-结束行]，回答里没有出现该格式"
        )

    return missing


MAX_TURNS_PER_CASE = 16
"""单条用例里 Agent 最多"思考 + 调用工具"多少轮。

SDK 默认 10 轮，实测跨文档类问题会撞上限并抛 `MaxTurnsExceeded`。
"撞上限"本身是真实信号（检索循环没收敛），但有两个原则：
① 给足轮数再判定，否则测的是 harness 的限制而不是模型的能力；
② 单条用例异常**绝不能**让整份评估崩掉——修复前 C1 抛异常直接把后面的用例全丢了。
"""


async def run_model_cases(
    only: str | None = None, save_path: str | None = None
) -> tuple[int, int]:
    """跑模型评估，返回 (通过数, 总数)；save_path 非空时把完整结果写成 JSON。"""
    passed = 0
    cases = [c for c in MODEL_CASES if not only or only in c["name"]]
    by_category: dict[str, list[int]] = {}
    format_ok = format_checked = 0
    records: list[dict] = []

    def record(category: str, ok: bool) -> None:
        stat = by_category.setdefault(category, [0, 0])
        stat[0] += int(ok)
        stat[1] += 1

    for case in cases:
        print(f"\n--- [{case['category']}] {case['name']} ---")
        print(f"Q: {case['question']}")
        try:
            result = await Runner.run(
                notes_qa_agent, case["question"], max_turns=MAX_TURNS_PER_CASE
            )
        except MaxTurnsExceeded as exc:
            print(f"FAIL: 超过 {MAX_TURNS_PER_CASE} 轮仍未收敛（检索循环在打转）")
            record(case["category"], ok=False)
            if save_path:
                records.append(
                    {
                        "name": case["name"],
                        "category": case["category"],
                        "question": case["question"],
                        "error": f"MaxTurnsExceeded: {exc}",
                    }
                )
            continue
        except Exception as exc:  # noqa: BLE001 —— 单条用例失败不能拖垮整份评估
            print(f"FAIL: 调用异常 {type(exc).__name__}: {exc}")
            record(case["category"], ok=False)
            if save_path:
                records.append(
                    {
                        "name": case["name"],
                        "category": case["category"],
                        "question": case["question"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue

        answer = result.final_output
        cited = sorted(extract_cited_files(answer))
        missing = check_model_output(answer, case)
        print(f"引用：{cited or '（无）'}")
        if case.get("require_bracket_citation"):
            bracket = has_bracket_citation(answer)
            format_checked += 1
            format_ok += int(bracket)
            print(f"格式：{'合规' if bracket else '不合规（没写成 [文件名:行号]）'}")
        if missing:
            print("FAIL:", "；".join(missing))
            # 失败时打印**完整回答**：只留节选会让 FAIL 无法复查
            # （第 14 周 T10 就是靠人肉读节选才发现"其实是格式漂移"）
            print("回答全文：\n" + answer)
        else:
            print("PASS")
            passed += 1
        record(case["category"], ok=not missing)
        usage = result.context_wrapper.usage
        print(
            f"  [用量] 请求={usage.requests} 输入={usage.input_tokens} 输出={usage.output_tokens}"
        )
        if save_path:
            records.append(
                {
                    "name": case["name"],
                    "category": case["category"],
                    "question": case["question"],
                    "passed": not missing,
                    "missing": missing,
                    "cited_files": cited,
                    "bracket_citation_ok": has_bracket_citation(answer)
                    if case.get("require_bracket_citation")
                    else None,
                    "requests": usage.requests,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "answer": answer,
                }
            )

    print("\n--- 分类汇总 ---")
    for category, (ok, total) in by_category.items():
        print(f"{category}：{ok}/{total}")
    if format_checked:
        print(
            f"引用格式合规：{format_ok}/{format_checked}"
            f"（与上面的通过率是两件事：内容对、格式漂移也会记不合规）"
        )

    if save_path:
        # 写文件是阻塞调用：丢到线程里，别卡住事件循环（ASYNC230）
        await asyncio.to_thread(
            Path(save_path).write_text,
            json.dumps({"records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n完整结果（含回答全文）已写入 {save_path}")
    return passed, len(cases)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="只跑单元测试，跳过模型评估（无需网络）",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="只跑名字里含该子串的模型用例（例如 --only A1），便于单条调试",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="把完整结果（含回答全文、引用、用量）写成 JSON，便于事后复查",
    )
    args = parser.parse_args()

    print("=== 单元测试 ===")
    unit_pass = 0
    for name, fn in UNIT_CASES:
        try:
            ok, detail = fn()
        # 单条用例崩了不能拖垮整份评估，刻意 catch-all 并记为 FAIL
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"异常：{e}"
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        unit_pass += int(ok)

    if args.unit_only:
        print(f"\n=== 合计：{unit_pass}/{len(UNIT_CASES)} 通过（已跳过模型评估）===")
        return

    print("\n=== 模型评估 ===")
    model_pass, model_total = asyncio.run(run_model_cases(args.only, args.save))

    total_pass = unit_pass + model_pass
    total = len(UNIT_CASES) + model_total
    print(f"\n=== 合计：{total_pass}/{total} 通过 ===")


if __name__ == "__main__":
    main()
