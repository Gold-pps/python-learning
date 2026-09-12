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
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError("只接受相对路径字符串")
    if rel_path.startswith(("/", "\\")):
        raise ValueError("只接受相对路径")
    if not rel_path.endswith(".md"):
        raise ValueError("只允许读取 .md 文件")

    root = config.NOTES_ROOT.resolve()
    target = (root / rel_path).resolve()

    # 解析后必须仍在根目录内：其中的 resolve() 已经展开符号链接 / junction，
    # 指向根目录外的链接会在这里被拦下。
    if not target.is_relative_to(root):
        raise ValueError("路径越界：解析后的目标不在笔记根目录内")
    if not target.exists():
        raise ValueError(f"文件不存在：{rel_path}")
    if target.is_symlink():
        raise ValueError("不接受符号链接或 junction")
    if not target.is_file():
        raise ValueError("目标不是普通文件")
    if target.suffix != ".md":
        raise ValueError("只允许读取 .md 文件")
    # 排除目录检查（与 list_notes 保持一致）。放在最后：若链接把隐藏文件映射进根目录，
    # 也会因为解析后的名字以 . 开头而被拦下。
    rel_parts = target.relative_to(root).parts
    if any(part in EXCLUDE_DIRS or _is_hidden(part) for part in rel_parts):
        raise ValueError("该路径在排除目录内，不可读取")
    return target


def list_notes() -> dict:
    """列出笔记根目录下所有 .md 文件（相对路径）。"""
    root = config.NOTES_ROOT.resolve()
    notes = []
    for p in sorted(root.rglob("*.md")):
        # 链接（符号链接 / junction）指向根目录外时不列出，避免枚举泄露外部文件
        try:
            if not p.resolve().is_relative_to(root):
                continue
        except OSError:
            continue
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

    size = target.stat().st_size
    if size > MAX_FILE_BYTES:
        return {"ok": False, "error": f"文件过大（{size} 字节），拒绝读取"}

    try:
        start_line = max(1, int(start_line))
        end_line = max(start_line, int(end_line))
    except (TypeError, ValueError):
        return {"ok": False, "error": "start_line / end_line 必须是整数"}
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
        # 只要这次读到的内容没到文件结尾，就说明还有内容可继续读
        "truncated": actual_end < total,
        "content": content,
    }

MAX_SEARCH_RESULTS = 5
MAX_PER_FILE = 2
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

    try:
        max_results = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    except (TypeError, ValueError):
        return {"ok": False, "error": "max_results 必须是整数"}
    needle = keyword.casefold()
    root = config.NOTES_ROOT.resolve()
    matches = []

    for p in sorted(root.rglob("*.md")):
        # 与 list_notes 一致：链接指向根目录外时跳过，避免搜索泄露外部文件
        try:
            if not p.resolve().is_relative_to(root):
                continue
        except OSError:
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in EXCLUDE_DIRS or _is_hidden(part) for part in rel_parts[:-1]):
            continue
        if _is_hidden(p.name):
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        file_hits = 0
        for i, line in enumerate(lines, start=1):
            if needle not in line.casefold():
                continue
            snippet = line.strip()
            if len(snippet) > SNIPPET_LEN:
                snippet = snippet[:SNIPPET_LEN] + "..."
            matches.append({
                "file": str(p.relative_to(root)),
                "line": i,
                "snippet": snippet,
            })
            file_hits += 1
            if file_hits >= MAX_PER_FILE:
                break
            if len(matches) >= max_results:
                break

        if len(matches) >= max_results:
            break

    return {"ok": True, "matches": matches}
