"""FastAPI 应用：一页前端 + 一条 SSE 进度流 + 一个自检端点。

三条设计取舍（都不是顺手写的）：

1. **进度是"阶段级"的，不是 token 级的**。规划原文写的是"`/ask` 支持流式返回"，
   但 `answer_review()` 的产物是**四段式 JSON**：token 级流式要么放弃 JSON 约束、
   要么边流边猜结构，第 21 周刚立起来的"引用校验拿候选集合对账"会直接失效。
   所以这里流的是**进度**：检索 → 盘点 → 补检 → 组织 → 校验，每一步都推一个事件。
   一次问答要等 30~60 秒，**"知道它在干什么"比"看到字一个个蹦出来"更有用**，
   而且最后那条 `result` 事件的载荷是完整的、校验过的 JSON。
2. **模型调用跑在线程里，事件从线程推回事件循环**。检索与两次模型调用都是同步阻塞的，
   直接在协程里跑会把整个服务卡住（进度事件也发不出去）。用 `asyncio.to_thread` +
   `call_soon_threadsafe` 是最省事的正确做法。顺带一个已知边界：**客户端中途关页面，
   已经在跑的模型调用不会停**（那笔钱已经花了）——只在说明里讲清楚，不假装能取消。
3. **默认只绑回环地址，往外开必须带口令**。规划写的是"本机 / 局域网足矣、不做鉴权"，
   这一条照做；但"默认就监听 0.0.0.0"是另一回事——那等于服务一启动，
   同网段任何人都能拿你的 API Key 提问。所以：`--host 0.0.0.0` 必须**同时**给 `--token`，
   否则直接拒绝启动（不是警告，是拒绝）。口令只是"别让路人白刷"，不是账号体系。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from constants import DEEPSEEK_MODEL
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from web.pricing import PRICE_SNAPSHOT_DATE, PRICE_SOURCE_URL, describe_pricing, is_peak
from web.service import (
    MAX_QUESTION_CHARS,
    SessionStore,
    ask,
    new_session_id,
    will_carry,
)

BASE_DIR = Path(__file__).resolve().parent.parent  # .../notes-qa
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_INDEX_DIR = BASE_DIR / "data" / "index"
DEFAULT_LOG_PATH = BASE_DIR / "data" / "usage_log.jsonl"
ENV_FILE = BASE_DIR.parent / ".env"  # agent-learning/.env

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def local_addresses() -> list[str]:
    """本机在"默认路由那张网卡"上的 IPv4 地址，用来告诉使用者"另一台机器该填哪个地址"。

    做法只有一句：对一个外部地址做 UDP `connect()`。它**不发任何包**（UDP 无连接），
    只是让内核按路由表选出口网卡，于是 `getsockname()` 就是本机在该网段的地址。
    立即返回，不需要 DNS。

    **为什么不解析主机名**（`socket.getaddrinfo(socket.gethostname())`）：
    试过，它会做一次 DNS 反查，而校园网/单位网的 DNS 偶尔很慢——
    实测那一次让整个测试套件从 3.5s 变成 **25.5s**。启动横幅卡 20 秒是不能接受的，
    而且这种等待**时快时慢**（DNS 缓存命中就快），比"慢一点"更难排查。
    宁可查不到（返回空列表、让使用者去看 `ipconfig`），也不在网络调用上无限等。

    机器多网卡时它只给默认路由那张卡——对"让同一网段的另一台机器连上"这个用途足够了。
    """
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        addr = probe.getsockname()[0]
    except OSError:
        return []
    finally:
        probe.close()
    return [addr] if not addr.startswith("127.") else []


class TokenRejected(ValueError):
    """口令里出现了 HTTP 头装不下的字符（见 `token_problem`）。"""


def token_problem(token: str | None) -> str | None:
    """口令能不能用；不能用就返回原因（能用返回 None）。

    **规则：只允许可见 ASCII（字母 / 数字 / 英文标点，不含空格）。** 这不是洁癖，
    是 HTTP 头本身的限制（RFC 9110：字段值按 ISO-8859-1 传）：

    - 浏览器侧：`fetch` 见到非 Latin-1 的头部值会**直接抛错、请求根本不发**——
      跨机试用第一天就用一个中文口令踩到了：`TypeError: ... String contains
      non ISO-8859-1 code point`，界面上表现为"自检失败 + 连接中断"（台账 T40）；
    - 服务端侧：ASGI 把头部字节按 **latin-1** 解码，所以就算客户端硬发原始 UTF-8 字节，
      这里拿到的也是乱码、永远比不中。

    两头都对不上，所以**别让这种口令进系统**：启动时拒绝，比让人对着一个
    "看起来输了却什么都发不出去"的口令排查半天要好。
    """
    if token is None:
        return None
    bad: list[str] = []  # 按**出现顺序**记，不按码点排序
    for ch in token:
        if not 0x21 <= ord(ch) <= 0x7E and ch not in bad:
            bad.append(ch)
    if bad:
        # 按出现顺序打印，使用者一眼能认出"这就是我刚输的那串"；
        # 排序后（一个令口定己自）反而看不出是哪把口令。
        return (
            f"口令里有 HTTP 头装不下的字符：{''.join(bad)[:10]}。"
            "口令只能用可见 ASCII（字母、数字、英文标点，不含空格）——"
            "中文/emoji 会让浏览器直接拒绝发送请求。"
        )
    return None


class AskBody(BaseModel):
    """提问请求体。长度上限**手动校验**（而不是靠 pydantic 的 max_length）：
    报错信息要说人话，而且要与 `main.py` 的提示保持一致。"""

    question: str = ""
    session_id: str | None = None
    carry_context: bool = True  # 追问时是否带上上一问（界面上的开关）
    multihop: bool = True  # 是否跑"盘点 + 补检"那一轮（省一次模型调用）


def read_index_meta(index_dir: Path) -> dict:
    """只读 `meta.json`，**不加载模型、不读 chunks** —— 自检要的是"能不能用"，不是"内容是什么"。"""
    meta_path = index_dir / "meta.json"
    info: dict = {"dir": str(index_dir), "exists": index_dir.exists()}
    problems: list[str] = []
    for name in ("chunks.jsonl", "vectors.npz", "meta.json"):
        if not (index_dir / name).exists():
            problems.append(f"缺 {name}")
    if meta_path.exists():
        try:
            info.update(json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            problems.append(f"meta.json 读不出来（{type(exc).__name__}）")
    info["problems"] = problems
    info["usable"] = not problems
    if problems:
        info["hint"] = (
            "索引不完整。在 agent-learning/notes-qa 下重建："
            '& "..\\..\\.venv-rag\\Scripts\\python.exe" -m rag.index build '
            "--backend fastembed --all --verify"
        )
    return info


def check_runtime(index_dir: Path) -> dict:
    """启动自检：索引 + 依赖 + 密钥，三样都给出**可执行的**下一步。"""
    index = read_index_meta(index_dir)
    try:
        import fastembed  # noqa: F401 —— 只探测能不能 import

        fastembed_ok, fastembed_note = True, "已安装"
    except ImportError as exc:  # pragma: no cover —— 环境问题，不走单测
        fastembed_ok, fastembed_note = (
            False,
            f"{type(exc).__name__}（跑 uv sync --extra rag）",
        )
    return {
        "index": index,
        "fastembed": {"ok": fastembed_ok, "note": fastembed_note},
        # **只说"文件在不在"，不说"Key 能不能用"**：真正发请求时由
        # `rag/rerank._get_client()` 自己 `load_dotenv`，与本进程的环境变量无关。
        # 早先这里报的是 `api_key_in_env`，自检会显示"未发现 Key"——而提问其实完全正常，
        # 属于自己吓自己（自检说错话比不报还糟）。
        "api_key": {
            "in_env": bool(os.environ.get("DEEPSEEK_API_KEY")),
            "env_file": str(ENV_FILE),
            "env_file_exists": ENV_FILE.exists(),
        },
        "ready": bool(index["usable"] and fastembed_ok),
    }


def create_app(
    *,
    index_dir: str | Path | None = None,
    retriever=None,
    log_path: str | Path | None = None,
    token: str | None = None,
    thinking_model: str | None = None,
    sessions: SessionStore | None = None,
    static_dir: str | Path | None = None,
) -> FastAPI:
    """构造应用。**依赖注入是为了测试**：注入假检索器后，整条链路可以离线跑完
    （见 `tests/test_web.py`），不需要 fastembed、不联网、不花 token。"""
    problem = token_problem(token)
    if problem:  # 程序化调用也不许把这种口令放进来，见 token_problem 的说明
        raise TokenRejected(problem)
    app = FastAPI(title="notes-qa 资料问答", docs_url="/api/docs", redoc_url=None)
    app.state.index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
    app.state.retriever = retriever  # None 表示"首次提问时懒加载"
    app.state.log_path = Path(log_path) if log_path else None
    app.state.token = token
    app.state.thinking_model = thinking_model
    app.state.sessions = sessions or SessionStore()
    app.state.static_dir = Path(static_dir or STATIC_DIR)

    def get_retriever():
        """懒加载检索器：第一次提问时才把模型读进内存（约 1~3 秒）。

        错误信息必须能直接照做——"索引不存在"和"没装 fastembed"是两回事，
        报同一句"初始化失败"等于让人去猜。
        """
        if app.state.retriever is not None:
            return app.state.retriever
        index_dir = app.state.index_dir
        info = read_index_meta(index_dir)
        if not info["usable"]:
            raise RuntimeError(
                f"索引不可用（{index_dir}）：{'、'.join(info['problems'])}"
            )
        try:
            from rag.embedder import Embedder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                f"缺少向量化后端：{exc}（跑 `uv sync --extra rag`）"
            ) from exc
        from rag.retriever import Retriever

        app.state.retriever = Retriever.from_index(
            index_dir, Embedder(backend="fastembed")
        )
        return app.state.retriever

    app.state.get_retriever = get_retriever

    def check_token(request: Request, provided: str | None) -> None:
        if not app.state.token:
            return
        if (
            provided == app.state.token
            or request.query_params.get("token") == app.state.token
        ):
            return
        raise HTTPException(status_code=401, detail="缺少或错误的访问口令")

    @app.get("/", response_class=HTMLResponse)
    async def index_page() -> HTMLResponse:
        page = app.state.static_dir / "index.html"
        if not page.exists():  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"前端文件不存在：{page}")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health(
        request: Request, x_token: str | None = Header(default=None)
    ) -> dict:
        from rag.coverage import summary as capability_summary

        if app.state.token:
            provided = x_token or request.query_params.get("token")
            if provided != app.state.token:
                # 没通关令时只回答"这是哪儿、要不要口令"，**不泄露索引规模与建库时间**。
                # 前端靠这个分支决定"先弹口令框"还是"直接进界面"。
                return {"service": "notes-qa web", "auth": True, "authorized": False}
        report = check_runtime(app.state.index_dir)
        peak = is_peak()
        # 检索器已经注入或加载过（测试、以及"先跑起来再建库"的场景）就不看索引文件了：
        # 那一句 `ready` 回答的是"现在能不能提问"，不是"目录里有什么"。
        ready = bool(report["ready"] or app.state.retriever is not None)
        return {
            "service": "notes-qa web",
            "auth": bool(app.state.token),
            "authorized": True,
            "ready": ready,
            "index": report["index"],
            "fastembed": report["fastembed"],
            "api_key": report["api_key"],
            "model": DEEPSEEK_MODEL,
            "thinking_model": app.state.thinking_model,
            "retriever_loaded": app.state.retriever is not None,
            "pricing": {
                "peak": peak,
                "snapshot": PRICE_SNAPSHOT_DATE,
                "source": PRICE_SOURCE_URL,
                "text": describe_pricing(peak),
            },
            # 能力说明的**唯一来源**在 `rag/coverage.py`（H4-1）：界面靠它放"能问什么"与示例问题。
            # 局部导入是为了不让 web 顶层随功能一路加依赖。
            "capability": capability_summary(),
            "limits": {"max_question_chars": MAX_QUESTION_CHARS},
            "log": str(app.state.log_path) if app.state.log_path else None,
        }

    @app.post("/api/export")
    async def api_export(request: Request, x_token: str | None = Header(default=None)):
        """把界面上那一次问答导成文件（H4-2）。

        内容由客户端把**它手里的视图**发回来（服务端不存视图，避免又多一份状态），
        渲染复用 `render_view_markdown` —— 导出的就是你看到的，不另写一份。
        """
        check_token(request, x_token)
        from web.export import content_disposition, filename, render

        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="请求体不是合法 JSON。"
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="请求体应当是对象。")
        view = payload.get("view")
        fmt = str(payload.get("fmt") or "md")
        if not isinstance(view, dict) or not view.get("sections"):
            raise HTTPException(
                status_code=400, detail="缺少 view（应当是一次问答的结果对象）。"
            )
        try:
            text, media = render(view, fmt)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return Response(
            content=text,
            media_type=media,
            headers={"Content-Disposition": content_disposition(filename(view, fmt))},
        )

    @app.post("/api/ask")
    async def api_ask(
        body: AskBody, request: Request, x_token: str | None = Header(default=None)
    ):
        check_token(request, x_token)
        question = (body.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空。")
        if len(question) > MAX_QUESTION_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"问题过长（{len(question)} 字符），上限 {MAX_QUESTION_CHARS} 字符。",
            )

        from rag.coverage import is_capability_question
        from rag.coverage import summary as capability_summary

        if is_capability_question(question):
            # 元问题（"你能回答什么"）**不检索、不花钱**，直接回能力说明——第 24 周真实使用里
            # 这句话被当成普通问题检索、回了一段资料摘录（H4-1 就是补这个洞）。
            # 走同一条 SSE 通道，前端只需认一个新事件类型；也不写使用日志（没有检索、没有模型调用）。
            payload = {
                "type": "capability",
                "question": question,
                **capability_summary(),
            }

            async def capability_stream():
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                capability_stream(), media_type="text/event-stream"
            )

        session_id = body.session_id or new_session_id()
        sessions_store: SessionStore = app.state.sessions
        previous = (
            sessions_store.last_question(session_id) if body.carry_context else None
        )
        # **界面说的必须和实际做的一致**：早先这里直接把 `previous` 当成"带上了上一问"发出去，
        # 于是判据说"不拼"的时候界面照样打横幅（台账 T41）。这里问一次同一个判据。
        carried = will_carry(question, previous, carry=body.carry_context)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(event: dict) -> None:
            """从 worker 线程把事件扔回事件循环（见模块 docstring 第 2 条）。

            循环已经关掉（服务停机 / 测试收尾）时 `call_soon_threadsafe` 会抛 RuntimeError：
            这是"没人听进度了"，不是错误，吞掉即可。
            """
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                pass

        def work() -> dict:
            retriever = get_retriever()  # 模型加载放在线程里，别卡住事件循环
            return ask(
                question,
                retriever,
                previous_question=previous,
                carry_context=body.carry_context,
                multihop=body.multihop,
                thinking_model=app.state.thinking_model,
                on_event=emit,
                log_path=app.state.log_path,
                index_label=str(app.state.index_dir),
            )

        cold_start = app.state.retriever is None

        async def run_and_finish() -> None:
            try:
                if cold_start:
                    emit(
                        {
                            "type": "stage",
                            "stage": "load",
                            "status": "start",
                            "label": "首次提问：正在把检索模型读进内存（约 1~3 秒）",
                        }
                    )
                view = await asyncio.to_thread(work)
                turn = sessions_store.add(
                    session_id,
                    question=question,
                    retrieval_query=view["retrieval_query"],
                )
                emit(
                    {
                        "type": "result",
                        "session_id": session_id,
                        "turn": turn,
                        "data": view,
                    }
                )
            except Exception as exc:  # noqa: BLE001 —— 任何异常都要变成一条可读的事件
                emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                emit({"type": "__end__"})

        async def stream():
            # 先报"收到了"，用户不必等服务端把模型加载完才看到动静
            yield _sse(
                {
                    "type": "start",
                    "session_id": session_id,
                    "question": question,
                    "carried_previous": carried,
                }
            )
            task = asyncio.create_task(run_and_finish())
            try:
                while True:
                    event = await queue.get()
                    if event.get("type") == "__end__":
                        break
                    yield _sse(event)
            finally:
                # 客户端断开时**不 cancel**：`to_thread` 里的模型调用停不下来，
                # cancel 只会让"钱已经花了"这件事多一条噪音日志。让任务自己跑完、事件丢弃。
                # `del` 只是明确"这个引用到此为止"（持有引用是为了不让事件循环回收任务）。
                del task

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=SSE_HEADERS
        )

    @app.post("/api/reset")
    async def reset(
        body: AskBody, request: Request, x_token: str | None = Header(default=None)
    ) -> dict:
        """清空某个会话的上下文（界面上的"新对话"）。"""
        check_token(request, x_token)
        if body.session_id:
            app.state.sessions.clear(body.session_id)
        return {"ok": True, "session_id": body.session_id}

    return app


def _sse(event: dict) -> str:
    """SSE 帧。只发 `data:` 一行：事件类型在 JSON 的 `type` 字段里，
    前端一种解析路径就够（多一种写法就多一处会错的地方）。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ---- 启动入口 ----


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m web.server",
        description="notes-qa 的本地 Web 界面（DeepSeek 后端，检索在本地）",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="默认只绑本机；局域网要显式开"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_DIR),
        help="索引目录（默认真实资料 data/index）",
    )
    parser.add_argument(
        "--log", default=str(DEFAULT_LOG_PATH), help="使用日志（JSONL，追加）"
    )
    parser.add_argument("--no-log", action="store_true", help="不写使用日志")
    parser.add_argument(
        "--token", default=None, help="访问口令；--host 非本机时**必须**给"
    )
    parser.add_argument(
        "--thinking-model",
        default=None,
        help="命中思考判据时改用这个模型名（**以控制台为准**，不给就一直用 deepseek-flash）",
    )
    parser.add_argument(
        "--check", action="store_true", help="只做自检并退出，不启动服务"
    )
    return parser.parse_args(argv)


def _is_local(host: str) -> bool:
    return host in LOCAL_HOSTS


def main(argv: list[str] | None = None) -> int:
    from rag.retriever import make_stdout_forgiving

    # 复用第 18 周那条：Windows 控制台默认 GBK，印一个装不下的字符会**直接抛异常**
    # 把命令打断（检索没问题却白跑一次，台账 T23）。自检输出里满是中文与 `｜`。
    make_stdout_forgiving()

    args = _parse_args(argv)
    index_dir = Path(args.index).resolve()
    report = check_runtime(index_dir)
    if args.check:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[自检] 就绪：{report['ready']}")
        return 0 if report["ready"] else 1

    # 两条**参数级**的检查放在环境检查之前：命令行打错了就该先听到命令行的问题
    # （顺序也决定了报错是否确定——先查参数，测试在没有索引的机器上也跑得出一致结论）。
    problem = token_problem(args.token)
    if problem:
        print(f"[拒绝启动] {problem}")
        print('  例：python -m web.server --host 0.0.0.0 --token "notes-qa-2026"')
        return 2
    if not _is_local(args.host) and not args.token:
        print(
            "[拒绝启动] --host 不是本机回环地址，必须同时给 --token。\n"
            "  理由：这个服务会花你的 token 调模型，而且没有账号体系；\n"
            "  绑到 0.0.0.0 而不设口令，等于同网段任何人都能用你的额度提问。\n"
            '  例：python -m web.server --host 0.0.0.0 --token "notes-qa-2026"'
        )
        return 2
    if not report["ready"]:
        print("[拒绝启动] 索引或依赖不齐：")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    import uvicorn

    app = create_app(
        index_dir=index_dir,
        log_path=None if args.no_log else Path(args.log).resolve(),
        token=args.token,
        thinking_model=args.thinking_model,
    )
    print(f"[notes-qa] 本机打开 http://127.0.0.1:{args.port}")
    if not _is_local(args.host):
        # 绑 0.0.0.0 时**不能把 0.0.0.0 当网址印出来**（它不是能打开的地址）；
        # 这一步最需要的恰恰是"另一台机器填哪个"。
        for addr in local_addresses():
            print(f"[notes-qa] 局域网打开 http://{addr}:{args.port}")
    print(f"[notes-qa] 索引：{index_dir}")
    if args.no_log:
        print("[notes-qa] 使用日志：已关闭（--no-log）")
    else:
        print(
            f"[notes-qa] 使用日志：{Path(args.log).resolve()}（提问会记进去，本地文件、不进 Git）"
        )
    if not _is_local(args.host):
        print("[notes-qa] 另一台机器若连不上，按这个顺序查：")
        print("           ① 两台机器在同一网段吗（前面几段数字要一样）")
        print(
            "           ② 从本机试 curl 上面的局域网地址——通不过就是 Windows 防火墙挡了入站"
        )
        print(
            "           ③ 都通了还连不上，看学校网络是否隔离了终端（那就改用热点或网线直连）"
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
