"""list_notes / search_notes 的契约测试。

覆盖原 `eval_cases.py` 的 U2、U6、U7、U8(后半)，外加第 13 周 T1 修复后补上的
"搜索跳过超大文件"用例（当时特意留到 pytest 迁移这一步再加，避免"12 条"这个口径来回改）。
"""

import pytest
import tools


def test_list_excludes_hidden_and_excluded_dirs(write):
    """U2：隐藏文件、隐藏目录、排除目录都不出现在列表里。"""
    write("可见.md", "# 可见")
    write(".隐藏.md", "# 隐藏")
    write(".git/配置.md", "# git 内部")
    write("__pycache__/缓存.md", "# 编译缓存")
    write("notes-qa/项目设计.md", "# 项目设计")

    result = tools.list_notes()
    assert result["ok"] is True
    assert [n["path"] for n in result["notes"]] == ["可见.md"]


def test_link_escape_not_listed_searched_or_read(write, make_dir_link):
    """U6：链接指向根外时，列表不枚举、搜索不命中、读取被拒。"""
    write("正常.md", "# 正常")
    _, target = make_dir_link("链接目录.md")
    (target / "根外秘密.md").write_text("OUTSIDE-SECRET\n", encoding="utf-8")

    assert [n["path"] for n in tools.list_notes()["notes"]] == ["正常.md"]
    assert tools.search_notes("OUTSIDE-SECRET")["matches"] == []

    via_link = tools.read_note("链接目录.md/根外秘密.md")
    assert via_link["ok"] is False
    assert "越界" in via_link["error"]


def test_search_clamps_results_and_rejects_long_keyword(write):
    """U7：结果条数收敛到上限；超长关键词被拒绝。"""
    for i in range(10):
        write(f"笔记{i}.md", f"# 笔记 {i}\n这里有关键词\n")

    result = tools.search_notes("关键词", 99)
    assert result["ok"] is True
    assert len(result["matches"]) <= tools.MAX_SEARCH_RESULTS

    too_long = tools.search_notes("x" * (tools.MAX_KEYWORD_LEN + 1))
    assert too_long["ok"] is False
    assert "过长" in too_long["error"]


def test_search_skips_oversize_file(write):
    """T1（第 13 周）：超大文件不读入内存，跳过并回传计数，避免"没搜到"被当成"没有"。"""
    write("大文件.md", "UNIQUE-BIG-KEYWORD\n" + "x" * (tools.MAX_FILE_BYTES + 1))
    write("小文件.md", "# 正常\n")

    result = tools.search_notes("UNIQUE-BIG-KEYWORD")
    assert result["ok"] is True
    assert result["matches"] == []
    assert result["skipped_large_files"] == 1


@pytest.mark.parametrize(
    "call",
    [
        lambda: tools.search_notes("x", None),
        lambda: tools.search_notes("x", "many"),
        lambda: tools.search_notes(None),
    ],
)
def test_bad_types_return_structured_errors(call):
    """U8：畸形参数返回结构化错误。"""
    result = call()
    assert result["ok"] is False
