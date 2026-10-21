"""把一次问答的视图导出成可存档的文件（H4-2）。

两条刻意的设计：

1. **不写第二份渲染**：Markdown 直接复用 `web.service.render_view_markdown`（界面视角，
   含耗时与成本）。这里只负责"文件名、内容类型、JSON 该带什么"——两份渲染器必然漂移。
2. **文件名里的中文不许直接进 HTTP 头**：`Content-Disposition` 的 `filename=` 按
   ISO-8859-1 传，中文会让浏览器直接拒绝（第 23 周 T40 就是这么踩的：中文口令让请求发不出去）。
   所以走 RFC 5987 的 `filename*=UTF-8''…`，另给一个纯 ASCII 的 `filename=` 兜底。
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from urllib.parse import quote

from web.service import render_view_markdown

FORMATS: tuple[str, ...] = ("md", "json")
MEDIA_TYPES = {
    "md": "text/markdown; charset=utf-8",
    "json": "application/json; charset=utf-8",
}
_UNSAFE = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def safe_slug(question: str, limit: int = 24) -> str:
    """从问题里取一段能当文件名的片段（去掉路径分隔符与 Windows 非法字符）。"""
    cleaned = _UNSAFE.sub("", question or "").strip()
    return cleaned[:limit] or "问答"


def filename(view: dict, fmt: str, *, now: datetime | None = None) -> str:
    """导出文件名：`notes-qa-<时间戳>-<问题片段>.<格式>`。"""
    stamp = (now or datetime.now(tz=UTC).astimezone()).strftime("%Y%m%d-%H%M%S")
    return f"notes-qa-{stamp}-{safe_slug(view.get('question') or '')}.{fmt}"


def content_disposition(name: str) -> str:
    """中文文件名走 RFC 5987，另给 ASCII 兜底（理由见模块 docstring 第 2 条）。"""
    ascii_fallback = name.encode("ascii", "ignore").decode("ascii") or "notes-qa-export"
    if not ascii_fallback.endswith((".md", ".json")):
        ascii_fallback += name[name.rfind(".") :] if "." in name else ""
    return f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{quote(name)}"


def render(view: dict, fmt: str) -> tuple[str, str]:
    """返回 `(内容, media_type)`。未知格式抛 `ValueError`（由 HTTP 层转成 400）。"""
    if fmt == "md":
        return render_view_markdown(view), MEDIA_TYPES["md"]
    if fmt == "json":
        return json.dumps(view, ensure_ascii=False, indent=2) + "\n", MEDIA_TYPES[
            "json"
        ]
    raise ValueError(f"不支持的导出格式：{fmt}（只支持 {' / '.join(FORMATS)}）")
