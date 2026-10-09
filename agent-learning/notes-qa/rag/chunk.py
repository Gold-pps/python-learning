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

第 22 周增加 `is_reference`：论文里的**参考文献列表页**是稠密书目信息，对很多查询都
"语义上很像"，实测两次被当成候选（一次模型主动拒绝、一次进了四段）。所以在建索引时
就把它标出来，检索时默认不送进候选（见 `retriever.search(exclude_references=...)`）。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

# ---- 「参考文献段」判定 ----
#
# **第一版判据写错过，教训值得留在这儿**：当时用「`[数字]` 紧跟字母/汉字」（如 `[18]LIU`）
# 当命中特征，再加一个"每百字 0.5 条"的密度阈值。结果中文综述正文里满地都是
# "XING Q Q 等[18]提出…"，密度轻易超标——2501 个片段里有 **664 条（26.6%）** 被标成
# 参考文献，单篇最多 76 条，且散布在正文各页。**特征必须挑正文里几乎不出现的**：
# 文献类型标识 `[J]/[C]/[M]`、DOI、`et al.`/`等.`、英文"编号. 姓, 首字母."、`[CrossRef]`。

_REF_HEADING = re.compile(r"参考文献|[Rr]eferences")
_TYPE_TAG = re.compile(r"\[[JCMDNRSP]\]")  # 中文文献类型标识
_DOI = re.compile(r"(?:https?://)?doi\.org/10\.|doi:\s*10\.", re.IGNORECASE)
# 注意写成 `\bet al\b` 而不是 `\bet al\.`：PDF 抽取出来常带空格（"et al ."），
# 严格要求点号紧跟会漏掉整批英文文献（第 22 周实测：某页 10 个片段一个都没标上）。
_ETAL = re.compile(r"\bet al\b|等\s*[.．]")  # "et al." / "李敏，…，等."
_EN_ENTRY = re.compile(
    r"\d{1,3}\.\s+[A-Z][a-z\u00c0-\u024f]+,?\s+[A-Z]\."
)  # 35. Liu, C.
_CROSSREF = re.compile(r"\[CrossRef\]")  # MDPI 样式的条目标记
_NUMBERED = re.compile(r"\[\d{1,3}\]\s*[A-Za-z\u4e00-\u9fff]")

REFERENCE_MIN_MARKERS = 4  # 至少这么多条"书目专属"特征


def _bibliography_markers(text: str) -> int:
    """数一段文本里的"书目专属"特征条数（正文里几乎不出现的那些）。"""
    return (
        len(_TYPE_TAG.findall(text))
        + len(_DOI.findall(text))
        + len(_ETAL.findall(text))
        + len(_EN_ENTRY.findall(text))
        + len(_CROSSREF.findall(text))
    )


def looks_like_reference_list(text: str) -> bool:
    """判断一段文本是不是"参考文献列表"。

    主判据：**书目专属特征 ≥ 4 条**（见 `_bibliography_markers`）。
    兜底判据：标题里明确写着"参考文献 / References"**且**有 ≥4 条编号式条目
    （挡那些既没有类型标识、也没有 DOI 的老式中文文献表）。
    """
    markers = _bibliography_markers(text)
    if markers >= REFERENCE_MIN_MARKERS:
        return True
    return bool(_REF_HEADING.search(text)) and len(_NUMBERED.findall(text)) >= 4


@dataclass(frozen=True)
class Chunk:
    id: str  # 稳定标识：source_path:start:hash 前 16 位
    source_path: str  # 相对笔记根的路径，如 "第7周笔记.md"
    doc_type: str  # "md" / "txt" / "pdf" / "docx"
    locator: str  # 人类可读的位置：PDF 页码 / 标题路径
    start: int  # 片段起始位置（闭区间，含）
    end: int  # 片段结束位置（闭区间，含）
    text: str  # 片段正文
    hash: str  # text 的 sha256 前 16 位
    is_reference: bool = False  # 是否是参考文献列表段（建索引时判定，检索层默认排除）

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
            is_reference=looks_like_reference_list(text),
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
                "is_reference": self.is_reference,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> Chunk:
        """从一行 JSON 反序列化。

        `is_reference` 有默认值：老索引（第 22 周之前建的）里没有这个字段也能读进来，
        只是全部按"不是参考文献"处理——所以**加字段之后要重建一次索引**才生效。
        """
        d = json.loads(line)
        return cls(**d)
