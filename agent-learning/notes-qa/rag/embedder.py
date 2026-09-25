"""Embedder：把文本变成向量。支持切换后端。

三个设计约束：
1. **延迟加载模型**：import 时不加载，第一次 encode() 才加载；
2. **延迟导入 torch**：本模块能被主环境（无 torch）import，
   只要不调用 encode() 就不需要 torch；跑索引时用 .venv-rag 的 python；
3. **后端可切换**：--embedder bge|fastembed|none。第 17 周只实现 bge，
   fastembed 留到第 18 周做混合检索时再补——这是**显式开关**，不是口头兜底。


实测（2026-09-25，Windows + RTX 3050 Laptop + 15.7 GB，60 条短文本）：

| 后端 | 冷启动（含加载） | 热启动（纯推理） |
| --- | --- | --- |
| bge (torch+cuda) | 13.08s | 0.021s |
| fastembed (onnx+cpu) | **0.33s** | 0.035s |

**反直觉结论**：小批量下 fastembed 更快，因为 torch 加载 CUDA 上下文就要十几秒，
ONNX 只要零点几秒。GPU 的优势要在批量足够大时才体现。
对本项目场景（几十到几百 chunk），**冷启动时间主导**，所以：
- 日常小规模构建：`fastembed` 更合适；
- 大规模批量（几千 chunk 以上）：`bge` + cuda 才划算。
两者产出向量交叉相似度 1.0（同模型不同实现），可互相校验。

"""

from __future__ import annotations

from typing import Literal

import numpy as np

Backend = Literal["bge", "fastembed", "none"]

# 两个后端使用同一个模型名（fastembed 内置了 BAAI/bge-small-zh-v1.5 的 ONNX 版），
# 保证切换后端时语义空间一致（向量值因数值精度略有差异，但可互检）。
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DIM = 512


class Embedder:
    """句向量编码器。

    典型用法：
        e = Embedder(backend="bge", device="cuda", batch_size=32)
        vecs = e.encode(["文本一", "文本二"])   # -> np.ndarray, shape (2, 512)
    """

    def __init__(
        self,
        backend: Backend = "bge",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.backend == "bge":
            # 延迟导入：主环境没有 torch 也能 import 本模块
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME, device=self.device)
        elif self.backend == "fastembed":
            # 降级分支：ONNX + CPU，不依赖 torch。
            # 本机（RTX 3050 + 15.7 GB）用不到，但为换机/分享保留。
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=MODEL_NAME)
        else:
            raise ValueError(f"未知后端：{self.backend!r}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """把一批文本编码成 L2 归一化的向量。

        归一化后，余弦相似度 = 点积，第 18 周做检索时不用再除模长。
        """
        if self.backend == "none":
            raise RuntimeError("backend='none' 时不应调用 encode()")
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        self._load()

        if self.backend == "bge":
            return self._model.encode(  # type: ignore[union-attr]
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > self.batch_size,
                convert_to_numpy=True,
            )

        # fastembed 的 embed() 返回生成器；它内部对 BGE 已做 L2 归一化。
        vecs = list(self._model.embed(texts, batch_size=self.batch_size))  # type: ignore[union-attr]
        return np.stack(vecs).astype(np.float32)

    @property
    def dim(self) -> int:
        return DIM
