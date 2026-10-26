"""补资料**一条命令**（第四阶段 H2）：预检 → 重建 → 冒烟 → 报"哪几处文档数字过期了"。

为什么需要它：补资料这件事，手工流程是四步，其中**第 4 步最容易漏**——
每次语料一变，README / 使用说明 / 技术版 / 学习进度里的"N 篇 / M 片段"就要手改，
漏一处就变成"文档撒谎"（台账 **T56** 咬了三次）。

用法（在 `agent-learning/notes-qa` 下）：

    uv run python -m rag.refresh              # 全流程（建库约 3 分钟，0 元）
    uv run python -m rag.refresh --dry-run    # 只做预检 + 报待改文档（几秒，不写索引）

**设计上的两条**：
1. 预检直接调 `rag.index.scan()`（拿结构化结果，能判断"有失败就停"）；
2. 重建与冒烟走**子进程**、命令与文档里写的一模一样——不猜内部签名，也不会有第二套房。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

QA_DIR = Path(__file__).resolve().parents[1]
REPO = QA_DIR.parents[1]

# 需要跟着语料规模走的文档（都在"描述现状"的位置上）
DOCS = [
    REPO / "README.md",
    QA_DIR / "web" / "使用说明.md",
    REPO / "agent-learning" / "汇报材料-技术版.md",
    REPO / "agent-learning" / "学习进度.md",
]

_NUMBER_WORDS = re.compile(r"\d+\s*篇|\d+\s*片段")

# 只认"描述现状"的写法；历史叙述整行跳过。第一版没有这两条时，58 行全是噪声
# （"44 篇入库""2501 片段"这种历史记录被一并列出，等于没筛——T52 同族：命中率高就是噪声）。
_STATE_HINTS = re.compile(
    r"(篇数控|篇论文|篇 PDF|共 \d+ 篇|（\d+ 片段|\(\d+ 片段|索引：)"
)
_HISTORY_HINTS = re.compile(
    r"(历史|第 \d+ 周|旧的|当时|曾经|那次|补前|原先是|改为|已清)"
)


def stale_docs(meta: dict, docs: list[Path] | None = None) -> list[dict]:
    """找出**写着旧语料规模**的行：该出现"当前篇数 / 当前片段数"的地方出现了别的数字。

    只报**带单位**的数字（`N 篇` / `N 片段`），且只在"等于某个别的篇数"时才提示——
    历史记录里本来就会有旧数字（比如"44 篇入库"是当时的事实），所以这里**只提示、不改文件**，
    由人判断哪些是"描述现状"、哪些是"历史记录"。
    """
    files = docs or DOCS
    files_count = int(meta.get("n_files") or 0)
    chunks_count = int(meta.get("n_chunks") or 0)
    if not files_count or not chunks_count:
        return []

    stale: list[dict] = []
    for path in files:
        if not path.exists():
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _HISTORY_HINTS.search(line) or not _STATE_HINTS.search(line):
                continue  # 历史叙述、或不是描述现状的写法 → 不报（宁可漏报，别刷屏）
            for word in _NUMBER_WORDS.findall(line):
                amount = int(re.sub(r"\D", "", word))
                if ("篇" in word and amount != files_count) or (
                    "片段" in word and amount != chunks_count
                ):
                    stale.append(
                        {
                            "file": str(path.relative_to(REPO))
                            if REPO in path.parents
                            else path.name,
                            "line": number,
                            "text": word,
                            "expected": f"{files_count} 篇 / {chunks_count} 片段",
                        }
                    )
    return stale


def _run(command: list[str]) -> int:
    return subprocess.run(command, cwd=QA_DIR, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="补资料一条命令：预检 → 重建 → 冒烟 → 报待改文档"
    )
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--index", default="data/index")
    parser.add_argument(
        "--dry-run", action="store_true", help="只预检 + 报待改文档，不建库"
    )
    parser.add_argument(
        "--skip-smoke", action="store_true", help="跳过检索冒烟（省 20 秒）"
    )
    args = parser.parse_args(argv)

    from rag import index as index_mod

    # ---- 1. 预检：失败就停，绝不把坏文件放进索引 ----
    report = index_mod.scan(Path(args.corpus))
    failures = report.get("failures") or []
    print(
        f"[1/4] 预检：文件 {report.get('files', 0)}、片段 {report.get('chunks_total', 0)}、"
        f"失败 {len(failures)}、空文档 {len(report.get('empty') or [])}"
    )
    if failures:
        print("  ✗ 有文件解析失败，**停在这一步**（先把它们修好或移出 corpus）：")
        for item in failures[:10]:
            print(f"    - {item}")
        return 1

    if args.dry_run:
        print("[dry-run] 跳过建库与冒烟")
    else:
        # ---- 2. 全量重建（命令与文档一致，不另写一套） ----
        code = _run(
            [
                sys.executable,
                "-m",
                "rag.index",
                "build",
                "--backend",
                "fastembed",
                "--all",
                "--verify",
                "--corpus",
                args.corpus,
                "--index",
                args.index,
                "--report",
                "eval/建库报告-自动.md",
            ]
        )
        print(f"[2/4] 建库：退出码 {code}")
        if code != 0:
            print("  ✗ 建库失败，**不要**继续跑冒烟（数字会来自半成品索引）")
            return 1

        # ---- 3. 检索冒烟：低于阈值只报警，不回退（索引已经建好了） ----
        if not args.skip_smoke:
            smoke = _run(
                [
                    sys.executable,
                    "-m",
                    "eval.review_coverage",
                    "--index",
                    args.index,
                    "--top-k",
                    "5",
                    "--max-per-source",
                    "2",
                    "--out",
                    "eval/检索冒烟.md",
                ]
            )
            print(
                f"[3/4] 检索冒烟：退出码 {smoke}（低于阈值时请看上面的表，**不回退索引**）"
            )

    # ---- 4. 报"哪几处文档数字过期了" ----
    meta_path = Path(args.index) / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stale = stale_docs(meta)
        print(
            f"[4/4] 待改文档：{len(stale)} 处（历史记录里的旧数字会一并列出，自行判断）"
        )
        for item in stale[:20]:
            print(
                f"  - {item['file']}:{item['line']} 写着 {item['text']}（当前 {item['expected']}）"
            )
        if len(stale) > 20:
            print(f"  ……共 {len(stale)} 处")
    else:
        print("[4/4] 没有 meta.json，跳过文档核对")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
