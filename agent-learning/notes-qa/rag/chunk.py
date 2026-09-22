"""RAG 片段数据结构。

本模块只定义数据结构与序列化，不含解析逻辑——解析在 parsers.py，切分在 chunker.py。

三个字段设计约束（来自第二阶段规划第 16 周）：
1. **可回查**：locator + start + end 三者合起来能定位到原文的具体位置；
2. **可增量**：hash 用于索引时跳过未变更的片段；
3. **可序列化**：能直接落成 JSONL（chunks.jsonl），不依赖 pickle。

start / end 的语义随 doc_type 不同：
- md / txt：从 1 开始的行号；
- pdf：页码（1 起）；start 为 0 时表示"整页"；
- docx：从 1 开始的段落号。
具体由各自的 parser 保证一致；Chunk 本身不解释这两个字段。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str            # 稳定标识：source_path:start:hash 前 16 位
    source_path: str   # 相对笔记根的路径，如 "第7周笔记.md"
    doc_type: str      # "md" / "txt" / "pdf" / "docx"
    locator: str       # 人类可读的位置：PDF 页码 / 标题路径
    start: int         # 片段起始位置（闭区间，含）
    end: int           # 片段结束位置（闭区间，含）
    text: str          # 片段正文
    hash: str          # text 的 sha256 前 16 位

    @classmethod
    def create(
        cls,
        *,
        source_path: str,
        doc_type: str,
        locator: str,
        start: int,
        end: int,
        text: str,
    ) -> Chunk:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        cid = f"{source_path}:{start}:{h}"
        return cls(
            id=cid,
            source_path=source_path,
            doc_type=doc_type,
            locator=locator,
            start=start,
            end=end,
            text=text,
            hash=h,
        )

    def to_json(self) -> str:
        """序列化为一行 JSON（用于 chunks.jsonl）。"""
        return json.dumps(
            {
                "id": self.id,
                "source_path": self.source_path,
                "doc_type": self.doc_type,
                "locator": self.locator,
                "start": self.start,
                "end": self.end,
                "text": self.text,
                "hash": self.hash,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> Chunk:
        """从一行 JSON 反序列化。"""
        d = json.loads(line)
        return cls(**d)