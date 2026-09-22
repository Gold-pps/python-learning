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
MAX_FILE_BYTES = 1 * 1024 * 1024  # 1MB
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


def _iter_notes():
    """遍历笔记根下所有合格笔记，产出 (相对路径, 绝对路径)。

    三类过滤规则集中在这里，且**只在这里**：
      1. 链接逃逸：符号链接 / junction 指向根目录外的一律跳过（避免枚举或搜到外部文件）；
      2. 排除目录：EXCLUDE_DIRS 里的目录不进入；
      3. 隐藏文件：以 `.` 开头的文件与目录不进入。

    重构前 `list_notes` 与 `search_notes` 各自抄了一份这三条规则。安全过滤被抄成两份，
    意味着**改一处、漏一处**——第 10 周修 `read_note` 的排除目录漏洞时，
    `search_notes` 就得单独再修一次。以后新增检索入口（第 16 周的 RAG 入库）也直接用它。
    """
    root = config.NOTES_ROOT.resolve()
    for p in sorted(root.rglob("*.md")):
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
        yield str(p.relative_to(root)), p


def list_notes() -> dict:
    """列出笔记根目录下所有 .md 文件（相对路径）。"""
    notes = []
    for rel, path in _iter_notes():
        try:
            stat = path.stat()
        except OSError:
            continue
        notes.append({"path": rel, "size": stat.st_size, "modified": stat.st_mtime})
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
    content = "\n".join(lines[start_line - 1 : actual_end])

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
    matches = []
    skipped_large = 0

    for rel, p in _iter_notes():
        # 大小检查必须在 read_text 之前：否则"检查"本身就已经把大文件读进内存了。
        # read_note 遇到大文件报错（用户要的是那一个文件），搜索则跳过并计数
        # （一个坏文件不该让整次搜索失败），计数会回传给模型，避免它把
        # "没搜到"误当成"笔记里没有"。
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                skipped_large += 1
                continue
        except OSError:
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue

        file_hits = 0
        for i, line in enumerate(lines, start=1):
            if needle not in line.casefold():
                continue
            snippet = line.strip()
            if len(snippet) > SNIPPET_LEN:
                snippet = snippet[:SNIPPET_LEN] + "..."
            matches.append(
                {
                    "file": rel,
                    "line": i,
                    "snippet": snippet,
                }
            )
            file_hits += 1
            if file_hits >= MAX_PER_FILE:
                break
            if len(matches) >= max_results:
                break

        if len(matches) >= max_results:
            break

    return {"ok": True, "matches": matches, "skipped_large_files": skipped_large}
