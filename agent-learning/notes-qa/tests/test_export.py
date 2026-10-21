"""导出（`web/export.py`）的离线测试（H4-2）。

钉三件事：
1. Markdown 导出的内容就是**界面上看到的**（四段 + 引用 + 耗时与成本），不是另写的一份；
2. JSON 导出能原样往返（以后重生成、二次分析都靠它）；
3. **中文文件名不进 HTTP 头**——走 RFC 5987，并留 ASCII 兜底（T40 的同类坑）。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from _web_fakes import build_sample_view
from web.export import content_disposition, filename, render, safe_slug


def test_markdown_export_is_the_same_rendering_as_the_ui():
    """导出与界面同源：四段标题、引用、耗时与成本都要在。"""
    view = build_sample_view()

    text, media = render(view, "md")

    assert media.startswith("text/markdown")
    for section in ("研究现状", "方法对比", "结论", "可复用点"):
        assert section in text
    assert "刀具/A.pdf:第 2 页" in text, "引用要能带出去"
    assert "估算" in text or "成本" in text, "耗时与成本是界面视角的关键，别丢"


def test_json_export_round_trips():
    """JSON 导出必须能原样读回来（存档 / 二次分析用）。"""
    view = build_sample_view()

    text, media = render(view, "json")

    assert media.startswith("application/json")
    assert json.loads(text) == view


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError):
        render(build_sample_view(), "pdf")


def test_chinese_filename_uses_rfc5987_with_ascii_fallback():
    """中文文件名不能直接塞进 `filename=`（HTTP 头按 ISO-8859-1 传，会发不出去）。"""
    view = {"question": "刀具磨损监测有哪些主流方法？"}
    name = filename(view, "md", now=datetime(2026, 10, 21, 15, 30, 0, tzinfo=UTC))

    assert name.endswith(".md")
    assert "20261021-153000" in name

    header = content_disposition(name)
    assert "filename*=UTF-8''" in header, "中文名要走 RFC 5987"
    assert header.count("filename=") == 1
    ascii_part = header.split('filename="', 1)[1].split('"', 1)[0]
    assert ascii_part.isascii(), "filename= 里必须只剩 ASCII（兜底用）"
    assert ascii_part.endswith(".md")


def test_safe_slug_strips_path_and_illegal_chars():
    """问题里带路径分隔符 / 冒号问号时，不能生成非法文件名。"""
    slug = safe_slug('刀具/磨损: 有哪些?"*方法"')

    assert "/" not in slug and ":" not in slug and "?" not in slug and "*" not in slug
    assert slug


# ---- HTTP 层：接口本身（只测函数不算测过"真路径"，T48 的教训）----


def test_export_endpoint_returns_a_downloadable_file():
    """POST /api/export → 200 + Markdown + 带 RFC 5987 文件名的附件头。"""
    from _web_fakes import HITS, FakeRetriever
    from fastapi.testclient import TestClient
    from web.server import create_app

    app = create_app(retriever=FakeRetriever(fallback=HITS), token=None, log_path=None)
    with TestClient(app) as client:
        resp = client.post(
            "/api/export", json={"view": build_sample_view(), "fmt": "md"}
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/markdown")
    disposition = resp.headers["content-disposition"]
    assert "attachment" in disposition
    assert "filename*=UTF-8''" in disposition, "中文名要走 RFC 5987"
    assert "研究现状" in resp.text, "导出的应当就是界面上那份内容"


def test_export_endpoint_rejects_payload_without_a_view():
    """没有 view（或 view 里没有 sections）就是 400，并说清缺什么。"""
    from _web_fakes import HITS, FakeRetriever
    from fastapi.testclient import TestClient
    from web.server import create_app

    app = create_app(retriever=FakeRetriever(fallback=HITS), token=None, log_path=None)
    with TestClient(app) as client:
        missing = client.post("/api/export", json={"fmt": "md"})
        wrong_fmt = client.post(
            "/api/export", json={"view": build_sample_view(), "fmt": "pdf"}
        )

    assert missing.status_code == 400
    assert "view" in missing.json()["detail"]
    assert wrong_fmt.status_code == 400
    assert "pdf" in wrong_fmt.json()["detail"]
