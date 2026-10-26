"""语料备份与"三方数字"自检（第四阶段 H3）。

**为什么备份的是语料而不是索引**（2026-10-26 改了主意）：

- `data/index` 是**派生物**：66 篇重建一次 3724 片段、约 3 分钟（`rag.refresh` 就是干这个的）；
- `data/corpus/` 那批 PDF **既不在 Git（`data/` 在 `.gitignore`）、也没有任何备份**——
  丢了就得重新下载，有些还未必找得回来。**它才是唯一不可替代的资产。**

**为什么要有"三方数字自检"**：语料、索引、文档三处各有一个篇数，任何一处变了另两处不会自动跟。
今天已经因此踩过两次（T56、以及 H2 里"58 处噪声"那次），所以做成一条命令：

    uv run python -m rag.backup --check      # 对不上就非零退出
    uv run python -m rag.backup --backup     # 把 data/corpus 打成一个 zip（默认 data/backup/）

恢复演练（真做过一次才算 H3 通过）：

    Expand-Archive -Path data/backup/corpus-<时间>.zip -DestinationPath data/corpus -Force
    uv run python -m rag.backup --check
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

QA_DIR = Path(__file__).resolve().parents[1]
REPO = QA_DIR.parents[1]
CORPUS = QA_DIR / "data" / "corpus"
INDEX = QA_DIR / "data" / "index"
BACKUP_DIR = QA_DIR / "data" / "backup"
README = REPO / "README.md"

CORPUS_SUFFIXES = {".pdf", ".md", ".txt", ".docx"}


def corpus_files(corpus: Path = CORPUS) -> list[Path]:
    return [
        path
        for path in corpus.rglob("*")
        if path.is_file() and path.suffix in CORPUS_SUFFIXES
    ]


def index_numbers(index: Path = INDEX) -> dict:
    """索引自报的家底（`meta.json`）：篇数与片段数。"""
    meta = index / "meta.json"
    if not meta.exists():
        return {"n_files": 0, "n_chunks": 0, "usable": False}
    data = json.loads(meta.read_text(encoding="utf-8"))
    return {
        "n_files": int(data.get("n_files") or 0),
        "n_chunks": int(data.get("n_chunks") or 0),
        "usable": bool(data.get("n_files")),
    }


def check(corpus: Path = CORPUS, index: Path = INDEX, readme: Path = README) -> dict:
    """三方对账：语料篇数 / 索引篇数 / README 里写的篇数。

    **只报不改**：文档里那处数字该写什么，是人的决定（历史记录里本来就有旧数字）。
    """
    files = len(corpus_files(corpus))
    numbers = index_numbers(index)
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""

    report = {
        "corpus_files": files,
        "index_files": numbers["n_files"],
        "index_chunks": numbers["n_chunks"],
        "index_usable": numbers["usable"],
        "readme_has_corpus_count": bool(files) and f"{files} 篇" in text,
        "problems": [],
    }
    if not files:
        report["problems"].append(f"语料目录是空的或不存在：{corpus}")
    if not numbers["usable"]:
        report["problems"].append(f"索引不可用（没有 meta.json 或 n_files=0）：{index}")
    elif numbers["n_files"] != files:
        report["problems"].append(
            f"语料 {files} 篇 vs 索引 {numbers['n_files']} 篇——索引过期了，跑 `python -m rag.refresh`"
        )
    if files and not report["readme_has_corpus_count"]:
        report["problems"].append(f"README 里没有「{files} 篇」——语料变了没同步")
    return report


def make_backup(corpus: Path = CORPUS, target_dir: Path = BACKUP_DIR) -> Path:
    """把语料打成一个 zip。文件名带时间戳，**已存在的不覆盖**（避免手滑覆盖掉好备份）。"""
    files = corpus_files(corpus)
    if not files:
        raise FileNotFoundError(f"语料目录是空的，没什么可备份：{corpus}")

    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).astimezone().strftime("%Y%m%d-%H%M%S")
    archive = target_dir / f"corpus-{stamp}.zip"
    if archive.exists():  # pragma: no cover —— 同一秒备份两次才会遇到
        raise FileExistsError(f"备份已存在，不覆盖：{archive}")

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=str(path.relative_to(corpus)))
    return archive


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台是 GBK：✓/✗ 直接 print 会 UnicodeEncodeError（台账 T23），用仓库里的兜底
    from rag.retriever import make_stdout_forgiving

    make_stdout_forgiving()

    parser = argparse.ArgumentParser(
        description="语料备份 / 三方数字自检（第四阶段 H3）"
    )
    parser.add_argument(
        "--check", action="store_true", help="对账：语料 / 索引 / README"
    )
    parser.add_argument("--backup", action="store_true", help="把 data/corpus 打成 zip")
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--index", default=str(INDEX))
    args = parser.parse_args(argv)

    if not args.check and not args.backup:
        parser.print_help()
        return 2

    if args.backup:
        try:
            archive = make_backup(Path(args.corpus))
        except (FileNotFoundError, FileExistsError) as exc:
            print(f"✗ {exc}")
            return 1
        print(f"备份完成：{archive}（{len(corpus_files(Path(args.corpus)))} 篇）")

    if args.check:
        report = check(corpus=Path(args.corpus), index=Path(args.index))
        print(
            f"语料 {report['corpus_files']} 篇｜索引 {report['index_files']} 篇 / "
            f"{report['index_chunks']} 片段｜索引可用 {report['index_usable']}"
        )
        if report["problems"]:
            print("✗ 对不上：")
            for item in report["problems"]:
                print(f"  - {item}")
            return 1
        print("✓ 三方一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
