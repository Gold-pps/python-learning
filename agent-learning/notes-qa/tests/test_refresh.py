"""补资料一条命令（`rag/refresh.py`）的离线测试（第四阶段 H2）。

三条要证明的事：
1. **预检失败就停**，绝不带着坏文件去建库（这是 H2 验收里点名要考的那条）；
2. **冒烟掉分不回退索引**，但要报出来（索引已经建好了，回退反而更糟）；
3. **能报出"文档里写着旧规模"的行**，且不误杀历史记录式表述的能力边界（只提示、不改文件）。
"""

from __future__ import annotations

from pathlib import Path

from rag import refresh


def test_stale_docs_flags_numbers_that_no_longer_match_the_index(tmp_path):
    """文档里写 `56 篇`、索引是 66 篇 → 要报出来；**历史叙述那行不报**（这是 T56 的形态）。

    注意粒度是**按行**的：同一行里既有现状又有历史时整行跳过（宁可漏报，别刷屏——
    第一版没有这道闸时，真实语料上一口气报了 58 行，等于没筛）。
    """
    doc = tmp_path / "doc.md"
    doc.write_text(
        "# 说明\n"
        "把 **56 篇论文**（3073 片段）建成索引。\n"
        "第 20 周是 44 篇入库（历史记录）。\n",
        encoding="utf-8",
    )

    stale = refresh.stale_docs({"n_files": 66, "n_chunks": 3724}, docs=[doc])

    words = {(item["line"], item["text"]) for item in stale}
    assert (2, "56 篇") in words, "旧篇数要报"
    assert (2, "3073 片段") in words, "旧片段数要报"
    assert not any(line == 3 for line, _ in words), "历史记录那行整行跳过，别刷屏"


def test_refresh_stops_before_build_when_scan_fails(tmp_path, monkeypatch, capsys):
    """预检有失败 → 返回 1，且**不调用建库**（H2 验收点名的那条）。"""
    calls: list[list[str]] = []

    def fake_scan(corpus: Path) -> dict:
        return {"files": 3, "chunks_total": 10, "failures": ["坏文件.pdf"], "empty": []}

    monkeypatch.setattr("rag.index.scan", fake_scan)
    monkeypatch.setattr(refresh, "_run", lambda command: calls.append(command) or 0)

    code = refresh.main(["--dry-run"])

    assert code == 1
    assert calls == [], "预检失败时不许调用任何子进程步骤"
    assert "停在这一步" in capsys.readouterr().out


def test_refresh_keeps_the_index_when_smoke_regresses(tmp_path, monkeypatch, capsys):
    """冒烟掉分 → 只报警、不回退：建库已经完成，回退反而更糟。"""
    monkeypatch.setattr(
        "rag.index.scan",
        lambda corpus: {"files": 1, "chunks_total": 1, "failures": [], "empty": []},
    )

    codes = iter([0, 2])  # 建库成功、冒烟退出码 2
    ran: list[list[str]] = []

    def fake_run(command: list[str]) -> int:
        ran.append(command)
        return next(codes)

    monkeypatch.setattr(refresh, "_run", fake_run)

    code = refresh.main(["--index", str(tmp_path)])

    assert code == 0, "冒烟掉分不该让整条命令失败"
    assert any("review_coverage" in " ".join(part for part in item) for item in ran), (
        "冒烟要真跑"
    )
    out = capsys.readouterr().out
    assert "不回退索引" in out
