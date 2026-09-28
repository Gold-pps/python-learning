"""（已弃用）第 18 周的离线对比入口。

第 19 周起请用 **`eval/rag_eval.py`**：它是本脚本的超集——
同一套四组对比，但判据升级为"按来源文件 + 关键词逐条配对"（跨文档 / 多跳题必须这样才判得准），
题量从 10 题扩到 25 题并分四类，另带 ground truth 可达性自检与负样本污染自检。

保留本文件只是为了让人能顺着第 18 周的记录找到"当时是哪个脚本产出的"：
`eval/第18周对比记录.md` 是它的产物（跑在全量构建出来的正确索引上，见台账 T29）。

    & "..\\..\\.venv-rag\\Scripts\\python.exe" -m eval.rag_eval --out eval/第19周对比表.md
"""

from __future__ import annotations

NOTICE = """本脚本已在第 19 周被 `eval/rag_eval.py` 取代（逻辑是它的超集）。

请改用：
    python -m eval.rag_eval --out eval/第19周对比表.md
（加 --with-rerank 才会调用模型、花 token）

第 18 周那份记录在 eval/第18周对比记录.md，无需重跑本脚本。
"""


def main() -> int:
    print(NOTICE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
