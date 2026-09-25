"""Chunker：把 ParsedBlock 切成 300~500 字、重叠 50 字的 Chunk。

策略（第二阶段规划第 16 周）：
1. **短段落先合并**：累积到 >= MIN_CHARS 才吐出一个 Chunk，避免大量碎片；
2. **单段过长则按句切**：目标 MAX_CHARS，句间保留 OVERLAP 字重叠，
   防止切在句子中间丢信息；
3. **不跨 locator 合并**（v1 简化）：一个 Chunk 用起始块的 locator；
   start/end 覆盖合并进来的所有块，回查时按 (source_path, start, end) 定位。

不变量：
- 每个 Chunk 的 text 非空；
- **典型长度 300~500，但短尾合并可能把它顶到 `MAX_CHARS * 1.5`（≈750）**；
  这是有意的取舍：宁可与前一段合并，也不留下 3~150 字的碎片；
- Chunk 顺序与原文件一致；
- 合并进来的块之间以空行分隔，保留原文段落结构。
"""

from __future__ import annotations

from dataclasses import replace

from .chunk import Chunk
from .parsers import ParsedBlock

MIN_CHARS = 300
MAX_CHARS = 500
OVERLAP = 50

# 中英文句末标点
_SENTENCE_ENDS = "。！？!?."


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """按句切长文本，句间保留 overlap 字重叠。"""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_ENDS and len(buf) >= max_chars:
            pieces.append(buf)
            buf = buf[-overlap:] if overlap > 0 else ""
    if buf.strip():
        pieces.append(buf)
    # 短尾合并：最后一段远小于 max_chars 就并到前一段，避免碎片
    if len(pieces) >= 2 and len(pieces[-1]) < max_chars // 2:
        pieces[-2] = pieces[-2] + pieces[-1]
        pieces.pop()
    return pieces


def chunk_blocks(
    blocks: list[ParsedBlock],
    *,
    source_path: str,
    doc_type: str,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
) -> list[Chunk]:
    """把同源 ParsedBlock 列表切成 Chunk 列表。"""
    chunks: list[Chunk] = []
    buf_text = ""
    buf_start = 0
    buf_end = 0
    buf_locator = ""

    def flush() -> None:
        nonlocal buf_text
        if not buf_text.strip():
            buf_text = ""
            return
        for piece in _split_long(buf_text, max_chars, overlap):
            if not piece.strip():
                continue
            chunks.append(
                Chunk.create(
                    source_path=source_path,
                    doc_type=doc_type,
                    locator=buf_locator,
                    start=buf_start,
                    end=buf_end,
                    text=piece,
                )
            )
        buf_text = ""

    for b in blocks:
        if not b.text.strip():
            continue
        # 加上这一块会超上限 → 先把当前缓冲吐出
        if buf_text and len(buf_text) + len(b.text) + 2 > max_chars:
            flush()
        if not buf_text:
            buf_start = b.start
            buf_locator = b.locator
        buf_text = (buf_text + "\n\n" + b.text) if buf_text else b.text
        buf_end = b.end
        # 达到下限 → 立即吐出，避免攒成超大块
        if len(buf_text) >= min_chars:
            flush()

    flush()

    # 残余小段合并：最后一块远小于 min_chars，尝试并到前一块
    if len(chunks) >= 2 and len(chunks[-1].text) < min_chars:
        last = chunks.pop()
        prev = chunks.pop()
        merged_text = prev.text + "\n\n" + last.text
        if len(merged_text) <= int(max_chars * 1.5):
            chunks.append(
                Chunk.create(
                    source_path=source_path,
                    doc_type=doc_type,
                    locator=prev.locator,
                    start=prev.start,
                    end=last.end,
                    text=merged_text,
                )
            )
        else:
            chunks.append(prev)
            chunks.append(last)

    # id 去重：长文本按句切时，同一 start 位置的多个 chunk 可能因文本相同而 hash 相同，
    # 加后缀保证 id 唯一（id 是索引主键，重复会互相覆盖）
    seen: dict[str, int] = {}
    deduped: list[Chunk] = []
    for c in chunks:
        n = seen.get(c.id, 0)
        seen[c.id] = n + 1
        deduped.append(c if n == 0 else replace(c, id=f"{c.id}-{n}"))
    return deduped
