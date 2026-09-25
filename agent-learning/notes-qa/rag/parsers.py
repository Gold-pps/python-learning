"""四种格式的解析层：把文件变成「带位置的段落列表」。

本模块只负责“把文件读成结构化的段落”，不做切分——切分在 chunker.py。

每种格式返回统一的中间结构 `ParsedBlock`：
- text：段落正文
- locator：人类可读位置（PDF 页码、docx 标题路径、md 标题路径）
- start / end：机器可读位置（语义见 chunk.py）

安全边界与 notes-qa 的既有工具保持一致：
- 只允许 .md / .txt / .pdf / .docx 四种后缀；
- 单文件大小上限沿用 config.MAX_FILE_BYTES；
- 解析失败返回空列表（由调用方记入失败清单），不抛异常。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constants import MAX_FILE_BYTES
from docx import Document
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}


@dataclass(frozen=True)
class ParsedBlock:
    text: str
    locator: str
    start: int
    end: int


def _check_size(path: Path) -> None:
    """超过 MAX_FILE_BYTES 直接拒绝——与 read_note 的策略一致（报错，不静默跳过）。"""
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"文件超过 {MAX_FILE_BYTES} 字节：{path}（{size} 字节）")


def parse_md(path: Path) -> list[ParsedBlock]:
    """按行读 .md，保留标题层级作为 locator。

    切分交给 chunker；这里以「标题 + 段落」为最小单位：
    - 遇到 `# ` 开头的行，更新当前标题路径；
    - 非空非标题行，累积成段落；空行表示段落结束。
    """
    _check_size(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    blocks: list[ParsedBlock] = []
    title_stack: list[str] = []
    buf: list[str] = []
    buf_start = 0

    def flush(end_line: int) -> None:
        nonlocal buf, buf_start
        if not buf:
            return
        locator = " > ".join(title_stack) if title_stack else "(无标题)"
        blocks.append(
            ParsedBlock(
                text="\n".join(buf),
                locator=locator,
                start=buf_start,
                end=end_line,
            )
        )
        buf = []
        buf_start = 0

    for i, line in enumerate(lines, start=1):
        if line.startswith("#"):
            flush(i - 1)
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            title_stack = title_stack[: level - 1]
            title_stack.append(title)
            continue
        if not line.strip():
            flush(i - 1)
            continue
        if not buf:
            buf_start = i
        buf.append(line)

    flush(len(lines))
    return blocks


def parse_txt(path: Path) -> list[ParsedBlock]:
    """按空行分段读 .txt；locator 用行号范围。"""
    _check_size(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    blocks: list[ParsedBlock] = []
    buf: list[str] = []
    buf_start = 0

    def flush(end_line: int) -> None:
        nonlocal buf, buf_start
        if not buf:
            return
        blocks.append(
            ParsedBlock(
                text="\n".join(buf),
                locator=f"行 {buf_start}-{end_line}",
                start=buf_start,
                end=end_line,
            )
        )
        buf = []
        buf_start = 0

    for i, line in enumerate(lines, start=1):
        if not line.strip():
            flush(i - 1)
            continue
        if not buf:
            buf_start = i
        buf.append(line)

    flush(len(lines))
    return blocks


def parse_pdf(path: Path) -> list[ParsedBlock]:
    """按页提取 PDF 文本。

    - 一页一个 ParsedBlock（不跨页合并，保证 locator 是干净的页码）；
    - start == end == 页码（1 起）；
    - 空白页跳过（扫描件会整页空白，由调用方记入失败清单）。
    """
    _check_size(path)
    reader = PdfReader(str(path))
    blocks: list[ParsedBlock] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        blocks.append(
            ParsedBlock(
                text=text,
                locator=f"第 {i} 页",
                start=i,
                end=i,
            )
        )
    return blocks


def parse_docx(path: Path) -> list[ParsedBlock]:
    """按段落提取 .docx，保留标题层级作为 locator。

    - 一个非空段落一个 ParsedBlock；
    - start == end == 段落号（1 起，含标题段落）；
    - 标题（Heading N）会更新标题栈，正文段落 locator 为标题路径。
    """
    _check_size(path)
    doc = Document(str(path))

    blocks: list[ParsedBlock] = []
    title_stack: list[str] = []

    for i, para in enumerate(doc.paragraphs, start=1):
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name if para.style else "") or ""
        if style_name.startswith("Heading"):
            level_str = style_name.replace("Heading", "").strip()
            try:
                level = int(level_str) if level_str else 1
            except ValueError:
                level = 1
            title_stack = title_stack[: level - 1]
            title_stack.append(text)
            continue
        locator = " > ".join(title_stack) if title_stack else "(无标题)"
        blocks.append(
            ParsedBlock(
                text=text,
                locator=locator,
                start=i,
                end=i,
            )
        )
    return blocks


# 后缀 → 解析函数的派发表。PDF / docx 在下一步补上。
_PARSERS = {
    ".md": parse_md,
    ".txt": parse_txt,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
}


def parse_file(path: Path) -> list[ParsedBlock]:
    """按后缀派发；不支持的后缀抛 ValueError。"""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的格式：{suffix}")
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise NotImplementedError(f"尚未实现：{suffix}")
    return parser(path)
