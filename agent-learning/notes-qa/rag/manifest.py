"""语料清单生成（第四阶段遗留）：把"哪 70 篇、多少 MB、sha256 前 16 位"写成一份 markdown。

**为什么它必须存在**：`data/` 不进 Git，PDF 丢了没法从仓库恢复；而 `语料清单.md` 是**唯一**
能让人按"文件名 + 大小 + sha256"认出"丢了哪几篇、从哪再下回来"的东西。
它**会随语料变化**——所以不能手写，得能一条命令重算（台账 T56：写死的统计迟早过期）。

用法（在 `agent-learning/notes-qa` 下）：

    uv run python -m rag.manifest                 # 重写 语料清单.md
    uv run python -m rag.manifest --out /tmp/x.md # 写到别处（CI/校对用）

**故意不做的事**：不碰 `data/` 以外的任何文件，也不改索引——它只读语料、只写那一份清单。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from rag.backup import corpus_files, index_numbers

QA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = QA_DIR / "语料清单.md"


def _sha_prefix(path: Path, length: int = 16) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:length]


def build(corpus: Path, *, index: Path | None = None, generated: str = "") -> str:
    """生成清单正文（纯函数：给定语料目录 → 一段 markdown，不写文件）。"""
    files = sorted(corpus_files(corpus), key=lambda path: (path.parent.name, path.name))
    if not files:
        raise FileNotFoundError(f"语料目录是空的：{corpus}")

    total_mb = sum(path.stat().st_size for path in files) / 1024 / 1024
    numbers = (
        index_numbers(index)
        if index
        else {"n_files": 0, "n_chunks": 0, "usable": False}
    )

    lines = [
        f"# 语料清单（{len(files)} 篇）",
        "",
        f"> 生成于 {generated or '（未注明日期）'}｜共 **{len(files)} 篇 / {total_mb:.1f} MB**"
        + (
            f"｜索引侧：{numbers['n_files']} 篇 / {numbers['n_chunks']} 片段。"
            if numbers["usable"]
            else "｜索引侧：未建索引。"
        ),
        "",
        "**这份文件的用途**：`data/` 不进 Git，PDF 丢了你没法从仓库恢复。",
        "但只要有这张表，就能按「文件名 + 大小 + sha256 前 16 位」**认出丢失的是哪几篇、从哪儿再下回来**——",
        "把「不可替代」变回「可重建」。所以它**应该进 Git**（只有元数据，不含 PDF 本体）。",
        "",
        "**重算方式**（语料一变就重跑，别手改）：",
        "",
        "```powershell",
        "uv run python -m rag.manifest",
        "```",
        "",
        "## 恢复步骤",
        "",
        "1. 按下面的子主题与文件名，从原出处（期刊官网 / 知网 / arXiv）重新下载；",
        "2. 放回 `data/corpus/<子主题>/`（子目录名就是子主题，别改名）；",
        "3. 核对 `sha256` 前 16 位是否一致（一致即为同一版本）；",
        "4. `uv run python -m rag.backup --check` 看三方是否一致，再 `uv run python -m rag.refresh` 重建索引。",
        "",
        "## 明细",
        "",
        "| # | 子主题 | 文件名 | MB | sha256(前16) |",
        "| - | ------ | ------ | -- | ------------ |",
    ]
    for index_number, path in enumerate(files, start=1):
        megabytes = path.stat().st_size / 1024 / 1024
        lines.append(
            f"| {index_number} | {path.parent.name} | {path.name} | {megabytes:.1f} | `{_sha_prefix(path)}` |"
        )

    lines += [
        "",
        "## 如果只能保住一样东西",
        "",
        "保**这份清单 + 各篇的出处笔记**（`eval/语料来源-*.md`），而不是那份 160MB 的 zip：",
        "清单几十 KB，能放进 Git、邮件、网盘；zip 只救得了一次，清单让你**任何时候都能重建**。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成/重算语料清单（离线、只读语料）")
    parser.add_argument("--corpus", default=str(QA_DIR / "data" / "corpus"))
    parser.add_argument("--index", default=str(QA_DIR / "data" / "index"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--date", default="", help="清单里写的生成日期（默认不写，避免假的精确）"
    )
    args = parser.parse_args(argv)

    text = build(Path(args.corpus), index=Path(args.index), generated=args.date)
    out = Path(args.out)
    out.write_text(text, encoding="utf-8")
    print(f"written: {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
