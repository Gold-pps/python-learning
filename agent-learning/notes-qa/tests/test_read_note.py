"""read_note 的契约测试。

覆盖原 `eval_cases.py` 的 U1、U3、U4、U5、U8(前半)、U9、U10。
与原版最大的区别：所有探针都写在 `tmp_path` 里，测试不再碰真实仓库。
"""

import pytest
import tools


@pytest.mark.parametrize(
    "path",
    [
        ".env",  # 隐藏文件（也是密钥所在）
        "../.env",  # 上跳
        "/etc/passwd",  # 绝对路径
        "笔记.txt",  # 非 .md
    ],
)
def test_rejects_non_markdown_paths(write, path):
    """U1：只接受笔记根内的 .md 文件，其它一律结构化报错（不抛异常）。"""
    result = tools.read_note(path)
    assert result["ok"] is False
    assert result["error"]


def test_rejects_excluded_dir(write):
    """U3：排除目录（notes-qa）不可读。"""
    write("notes-qa/项目设计.md", "# 项目设计")
    result = tools.read_note("notes-qa/项目设计.md")
    assert result["ok"] is False
    assert "排除目录" in result["error"]


def test_rejects_path_outside_root(notes_root):
    """U4：真正走到"路径越界"分支（.md 结尾、dotdot 在中间）。"""
    result = tools.read_note("sub/../../../secret.md")
    assert result["ok"] is False
    assert "越界" in result["error"]


def test_rejects_symlink_escape(make_file_link):
    """U5：根目录内指向根外的符号链接必须拒绝。"""
    make_file_link("逃逸链接.md")
    result = tools.read_note("逃逸链接.md")
    assert result["ok"] is False
    assert "越界" in result["error"]


def test_rejects_oversize_file(write):
    """U9：超过 1MB 的文件拒绝读取（大文件不再进内存）。"""
    write("大文件.md", "x" * (tools.MAX_FILE_BYTES + 1))
    result = tools.read_note("大文件.md")
    assert result["ok"] is False
    assert "过大" in result["error"]


def test_clamps_lines_and_reports_truncation(write):
    """U10：单次最多 200 行；`truncated` 表示"还有内容没读到"。"""
    write("长文.md", "".join(f"line-{i}\n" for i in range(1, 251)))
    write("短文.md", "".join(f"line-{i}\n" for i in range(1, 101)))

    clamped = tools.read_note("长文.md", 1, 9999)
    assert clamped["end_line"] == tools.MAX_READ_LINES
    assert clamped["total_lines"] == 250
    assert clamped["truncated"] is True

    complete = tools.read_note("短文.md")
    assert complete["end_line"] == 100
    assert complete["truncated"] is False


@pytest.mark.parametrize(
    "call",
    [
        lambda: tools.read_note(None),
        lambda: tools.read_note("任意.md", start_line="abc"),
        lambda: tools.read_note("任意.md", end_line=None),
    ],
)
def test_bad_types_return_structured_errors(call):
    """U8：畸形参数（模型完全可能传出来）必须返回结构化错误，而不是抛异常。"""
    result = call()
    assert result["ok"] is False
