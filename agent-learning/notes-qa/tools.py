"""笔记工具：list_notes / read_note。

约定：
- 所有路径为相对笔记根目录的相对路径；
- 出错返回 {"ok": false, "error": "..."}，不抛异常；
- 只允许读取笔记根目录内的 .md 文件。
"""
from pathlib import Path

import config

# ---- 常量 ----
EXCLUDE_DIRS = {"__pycache__", ".venv", ".git", ".vscode", "notes-qa"}
MAX_FILE_BYTES = 1 * 1024 * 1024   # 1MB
MAX_READ_LINES = 200


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _safe_resolve(rel_path: str) -> Path:
    """把相对路径解析成绝对路径，并确认仍在 NOTES_ROOT 内。

    不通过则抛 ValueError，由调用方转成结构化错误。
    """
    if not rel_path or rel_path.startswith("/"):
        raise ValueError("只接受相对路径")
    if not rel_path.endswith(".md"):
        raise ValueError("只允许读取 .md 文件")

    root = config.NOTES_ROOT.resolve()
    target = (root / rel_path).resolve()

    # 必须仍在根目录内
    if not target.is_relative_to(root):
        raise ValueError("路径越界")
    if target.suffix != ".md":
        raise ValueError("只允许读取 .md 文件")
    return target


def list_notes() -> dict:
    """列出笔记根目录下所有 .md 文件（相对路径）。"""
    root = config.NOTES_ROOT
    notes = []
    for p in sorted(root.rglob("*.md")):
        rel_parts = p.relative_to(root).parts
        # 跳过被排除目录、隐藏文件
        if any(part in EXCLUDE_DIRS or _is_hidden(part) for part in rel_parts[:-1]):
            continue
        if _is_hidden(p.name):
            continue
        notes.append({
            "path": str(p.relative_to(root)),
            "size": p.stat().st_size,
            "modified": p.stat().st_mtime,
        })
    return {"ok": True, "notes": notes}


def read_note(file: str, start_line: int = 1, end_line: int = 120) -> dict:
    """读取一个 .md 文件的指定行区间。"""
    try:
        target = _safe_resolve(file)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    if not target.exists():
        return {"ok": False, "error": f"文件不存在：{file}"}

    size = target.stat().st_size
    if size > MAX_FILE_BYTES:
        return {"ok": False, "error": f"文件过大（{size} 字节），拒绝读取"}

    start_line = max(1, int(start_line))
    end_line = max(start_line, int(end_line))
    if end_line - start_line + 1 > MAX_READ_LINES:
        end_line = start_line + MAX_READ_LINES - 1

    lines = target.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    actual_end = min(end_line, total)
    content = "\n".join(lines[start_line - 1:actual_end])

    return {
        "ok": True,
        "file": file,
        "start_line": start_line,
        "end_line": actual_end,
        "total_lines": total,
        "truncated": actual_end < end_line,
        "content": content,
    }

MAX_SEARCH_RESULTS = 5
MAX_KEYWORD_LEN = 50
SNIPPET_LEN = 100


def search_notes(keyword: str, max_results: int = MAX_SEARCH_RESULTS) -> dict:
    """在笔记中搜索关键词，大小写不敏感，返回文件、行号、片段。"""
    if not isinstance(keyword, str):
        return {"ok": False, "error": "keyword 必须是字符串"}
    keyword = keyword.strip()
    if not keyword:
        return {"ok": False, "error": "keyword 不能为空"}
    if len(keyword) > MAX_KEYWORD_LEN:
        return {"ok": False, "error": f"keyword 过长（≤{MAX_KEYWORD_LEN} 字符）"}

    max_results = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    needle = keyword.casefold()
    root = config.NOTES_ROOT
    matches = []

    for p in sorted(root.rglob("*.md")):
        rel_parts = p.relative_to(root).parts
        if any(part in EXCLUDE_DIRS or _is_hidden(part) for part in rel_parts[:-1]):
            continue
        if _is_hidden(p.name):
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            continue
        for i, line in enumerate(lines, start=1):
            if needle in line.casefold():
                snippet = line.strip()
                if len(snippet) > SNIPPET_LEN:
                    snippet = snippet[:SNIPPET_LEN] + "..."
                matches.append({
                    "file": str(p.relative_to(root)),
                    "line": i,
                    "snippet": snippet,
                })
                if len(matches) >= max_results:
                    return {"ok": True, "matches": matches}

    return {"ok": True, "matches": matches}