"""核对清单生成器（`web/audit.py`）的离线测试——不联网、不花钱。

它要证明的是**清单真的把判断所需的东西都摆出来了**：每条结论旁边有它引用的原文、
没挂引用的条目标出来了、编造的引用列在正文之外、以及"谁核的 / 哪一档证据"留了必填位。
最后一条走命令行（`--view` 那条路径），证明不跑模型也能生成。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from _web_fakes import FIXTURE_PATH, build_sample_view
from web.audit import build_checklist

NOTES_QA = Path(__file__).resolve().parents[1]


def fixture_view() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_checklist_puts_the_original_text_next_to_each_conclusion():
    """每条结论后面都要有它引用的**原文**，否则核对着还得自己去翻索引。"""
    text = build_checklist(fixture_view(), source="tests/fixtures/sample_view.json")

    assert "有两种主流做法" in text
    assert "方法A" in text and "优点：便宜" in text and "局限：精度差" in text
    # 精确引用的候选原文必须出现（模型当时看到的就是这段）
    assert "这是候选片段的原文全文，点开引用要能看到它。" in text
    assert text.count("判断：") == 2, "两条待核对结论各一个判断框"
    assert text.count("☐ 支持") == 2


def test_uncited_conclusion_is_flagged_as_the_riskiest():
    """一条引用都没挂的条目要单独标出来——那是风险最高的一类。"""
    text = build_checklist(fixture_view())

    assert "一条引用都没挂" in text
    assert "无引用条目 1" in text, "头部的对账数字要能对上"


def test_partial_citation_is_marked_and_shows_the_file_level_hits():
    """只到文件的引用：要说明"没给到页"，并把该文件本次命中的片段摆出来。"""
    text = build_checklist(fixture_view())

    assert "只到文件" in text
    assert "只到文件、没给到页" in text
    assert "B 的原文。" in text, "只到文件的引用也要给出可核的原文"


def test_fabricated_citations_are_listed_outside_the_body():
    """编造的引用**不在正文里**（校验那一步就摘掉了），但要单独列出来让人看见。"""
    text = build_checklist(fixture_view())

    assert "编造的引用（不在正文里，单独列出）" in text
    assert "幽灵.pdf:第 9 页" in text
    body = text.split("## 编造的引用", 1)[1].split("---", 1)[1]
    assert "幽灵.pdf" not in body, "编造引用不该出现在结论旁边（那等于承认它是条出处）"


def test_checklist_demands_auditor_and_evidence_tier():
    """清单本身不许声称"已核对"：谁核的、什么背景、哪一档证据，都是必填位。"""
    text = build_checklist(fixture_view())

    assert "核对者" in text and "背景" in text
    assert "非本领域读者" in text, "要给出适合本项目的那一档（作者不是这个领域的）"
    assert "判不了“专业上对不对”" in text, "要把这一档证据的边界写清"


def test_degraded_and_insufficient_are_surfaced_at_the_top():
    """降级回答、模型自认资料不足——这两种情况核对着必须先知道。"""
    view = build_sample_view()
    view["degraded"] = True
    view["reason"] = "JSON 解析失败（演示用）"
    view["insufficient"] = True
    view["note"] = "资料中没有直接答案"

    text = build_checklist(view)

    assert "这一次是降级回答" in text and "JSON 解析失败" in text
    assert "模型自己声明资料不足以回答" in text and "资料中没有直接答案" in text


def test_cli_generates_from_a_view_file_without_spending(tmp_path):
    """`--view` 这条路径不跑模型：一条命令把视图 JSON 变成清单文件。"""
    out = tmp_path / "核对清单.md"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "web.audit",
            "--view",
            str(FIXTURE_PATH),
            "--out",
            str(out),
        ],
        cwd=NOTES_QA,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    written = out.read_text(encoding="utf-8")
    assert "待核对结论数：**2**" in written
    assert "核对者" in written


def test_cli_refuses_when_neither_input_is_given():
    """两个入口必须二选一——不然它会去找一个不存在的默认值。"""
    proc = subprocess.run(
        [sys.executable, "-m", "web.audit"],
        cwd=NOTES_QA,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )

    assert proc.returncode == 2, "argparse 的用法错误应当是 2"
