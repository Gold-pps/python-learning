"""Web 层（`web/`）的离线测试 —— 不联网、不加载 fastembed、不花 token。

整条链路（检索 → 盘点 → 组织 → 引用校验 → 界面视图 → SSE 事件）跑的都是**真实代码**，
只替换两处：`plan_call` / `answer_call`（模型调用）与检索器。

这本身就是一次验证：**换掉这两个注入点，Web 层与 `review_qa` 的逻辑一个都没少**——
如果哪天真需要给 Web 单独写一套问答流程，说明这里的接缝选错了。

覆盖重点与理由：

1. **成本口径**：缓存命中价只有未命中的约 1/50，算错就是几十倍的偏差；
   而"高估成本"在这里是有害的（会让人不敢用），所以峰谷与缓存两件事都要钉住；
2. **降级与边界**：无候选不调模型、模型乱答走降级、超长问题被拒——这些是"对外可用"的底线；
3. **多轮拼接的可见性**：拼了才拼、说了才拼、界面上必须能看见拼了什么。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from rag.chunk import Chunk
from rag.retriever import SearchHit
from web.pricing import (
    BEIJING,
    PRICE_OFFPEAK,
    describe_pricing,
    estimate_cost,
    is_peak,
    unit_prices,
)
from web.server import create_app
from web.service import (
    MAX_QUESTION_CHARS,
    RunTrace,
    SessionStore,
    TimingRetriever,
    _link_citations,
    ask,
    compose_query,
    looks_like_follow_up,
    new_session_id,
    render_view_markdown,
)

USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 200,
    "cached_tokens": 400,
    "finish_reason": "stop",
}


# ---- 夹具：假检索器与假模型调用 ----


def make_hit(name: str, text: str, locator: str = "第 1 页") -> SearchHit:
    chunk = Chunk.create(
        source_path=name, doc_type="pdf", locator=locator, start=1, end=1, text=text
    )
    return SearchHit(chunk=chunk, score=0.03, dense_rank=1, sparse_rank=1)


class FakeRetriever:
    """按查询词返回预设命中，并记录收到的每个查询。"""

    def __init__(self, table=None, fallback=None):
        self.table = table or {}
        self.fallback = fallback or []
        self.queries: list[str] = []
        self.marker = "我是检索器上挂着的属性"  # 给 __getattr__ 转发测试用

    def search(self, query, top_k=5, **kwargs):
        self.queries.append(query)
        hits = self.table.get(query)
        return list(hits if hits is not None else self.fallback)


def plan_reply(enough=True, gaps=(), follows=()) -> str:
    return json.dumps(
        {"enough": enough, "gaps": list(gaps), "follow_up_queries": list(follows)},
        ensure_ascii=False,
    )


def answer_reply(text: str, citation: str, key: str = "研究现状") -> str:
    payload = {k: [] for k in ("研究现状", "方法对比", "结论", "可复用点")}
    payload[key] = [{"point": text, "citations": [citation]}]
    payload["insufficient"] = False
    payload["note"] = ""
    return json.dumps(payload, ensure_ascii=False)


def make_calls(answer_text: str, *, plan: str | None = None, sleep: float = 0.0):
    """构造 (plan_call, answer_call) 两个假调用，签名与真实的一致。"""

    def plan_call(prompt, **kwargs):
        if sleep:
            import time

            time.sleep(sleep)
        return (plan or plan_reply(True)), dict(USAGE)

    def answer_call(prompt, **kwargs):
        if sleep:
            import time

            time.sleep(sleep)
        return answer_text, dict(USAGE)

    return plan_call, answer_call


HITS = [
    make_hit(
        "刀具磨损监测/综述A.pdf", "振动信号与电流信号是工业可接受的两种监测手段。"
    ),
    make_hit(
        "刀具磨损监测/研究B.pdf",
        "力传感器精度高但成本高、不易安装。",
        locator="第 5 页",
    ),
]


@pytest.fixture
def client():
    """默认客户端：注入了假检索器，所以不需要索引文件、不加载 fastembed。"""
    app = create_app(retriever=FakeRetriever(fallback=HITS), token=None, log_path=None)
    with TestClient(app) as c:
        yield c


# ---- 价格与峰谷（纯函数）----


def test_offpeak_prices_match_snapshot():
    assert unit_prices(peak=False) == PRICE_OFFPEAK
    peak = unit_prices(peak=True)
    assert peak["out"] == PRICE_OFFPEAK["out"] * 2  # 低谷价是高峰价的一半


@pytest.mark.parametrize(
    ("hour", "weekday_offset", "expected"),
    [
        (9, 0, True),  # 周一 09:00 —— 高峰起点
        (11, 0, True),
        (12, 0, False),  # 12:00 整已经进低谷（左闭右开）
        (14, 0, True),  # 下午高峰起点
        (17, 0, True),
        (18, 0, False),  # 18:00 起低谷，一直到次日 09:00
        (23, 0, False),
        (2, 0, False),
        (10, 5, False),  # 周六整日低谷
        (10, 6, False),  # 周日
    ],
)
def test_is_peak_boundaries(hour, weekday_offset, expected):
    # 2026-10-05 是周一，往后偏移得到周六（10-10）/周日（10-11）
    monday = datetime(2026, 10, 5, hour, 0, tzinfo=BEIJING)
    moment = monday + timedelta(days=weekday_offset)
    assert is_peak(moment) is expected


def test_estimate_cost_splits_cached_tokens():
    """缓存命中必须单独计价：拿 prompt 全额乘未命中价会把成本高估几十倍。

    现实里指令前缀稳定时命中率很高（第 18 周单次实测 94%），所以用 90% 这一档
    才看得出"算不算缓存"的差别有多大。
    """
    cost = estimate_cost(1_000_000, 900_000, 0, peak=False)

    assert cost["usd"] == pytest.approx(100_000 * 0.15 / 1e6 + 900_000 * 0.003 / 1e6)
    naive = 1_000_000 * 0.15 / 1e6  # 不拆缓存的口径
    assert cost["usd"] < naive / 5  # 高估到 8 倍以上，绝不是"小数点后的差异"


def test_estimate_cost_clamps_inconsistent_usage():
    """`cached > prompt` 这种自相矛盾的数不能算出负数或重复计价。"""
    cost = estimate_cost(100, 999, 100, peak=False)
    # prompt 全额按"缓存命中"计价（未命中部分为 0），不会变成负数、也不会算两遍
    assert cost["usd"] == pytest.approx((100 * 0.003 + 100 * 0.6) / 1e6)


def test_describe_pricing_mentions_snapshot():
    text = describe_pricing(peak=False)
    assert "低谷" in text and "快照" in text


# ---- 多轮拼接（纯函数）----


def test_compose_query_carries_previous_only_for_follow_ups():
    previous = "刀具磨损监测有哪些主流方法？"

    query, carried = compose_query("那它的局限呢？", previous, carry=True)
    assert carried == previous
    assert query == f"{previous} 那它的局限呢？"

    # 完整问句（离开上文也检索得到）不拼——避免把上一问的噪声带进检索
    query, carried = compose_query(
        "机床热误差补偿有哪些主流方法？", previous, carry=True
    )
    assert carried is None and query == "机床热误差补偿有哪些主流方法？"


def test_compose_query_respects_switch_and_missing_previous():
    previous = "刀具磨损监测有哪些主流方法？"
    assert compose_query("那它的局限呢？", previous, carry=False) == (
        "那它的局限呢？",
        None,
    )
    assert compose_query("那它的局限呢？", None, carry=True) == ("那它的局限呢？", None)


def test_looks_like_follow_up():
    assert looks_like_follow_up("还有呢？")
    assert looks_like_follow_up("它适用于什么场景")
    assert looks_like_follow_up("那他们的局限呢")
    assert looks_like_follow_up("继续")
    assert not looks_like_follow_up("数字孪生在机床误差补偿中有哪些应用？")


def test_follow_up_heuristic_does_not_swallow_short_standalone_questions():
    """首跑记录（2026-10-10，台账 T37）：**短**不等于**追问**。

    真实现场：问完“AI 辅助工艺编制怎么做？”（正好 12 字），接着问“异星工厂是什么”，
    后者被长度判据当成追问，把上一问粘成了检索词——候选整批跑到数字孪生那边，
    明明一个无关的小问题，却检索出满屏"数字孪生"。
    """
    previous = "AI 辅助工艺编制怎么做？"

    query, carried = compose_query("异星工厂是什么", previous, carry=True)
    assert carried is None, "7 字的独立问题不该被当成追问"
    assert query == "异星工厂是什么"

    # 12 字的独立问题同样不该被判成追问（第一版的阈值恰好是 ≤12）
    assert looks_like_follow_up(previous) is False
    # 而真正的追问仍要认出来
    assert compose_query("那他们的局限呢", previous, carry=True)[1] == previous


def test_question_length_matches_cli_rule():
    """同一条产品规则在两个入口各写一份，**用测试钉住不许漂**。"""
    from main import MAX_QUESTION_LEN

    assert MAX_QUESTION_CHARS == MAX_QUESTION_LEN


# ---- ask()：视图结构 ----


def test_ask_view_links_citations_to_full_text():
    citation = "刀具磨损监测/研究B.pdf:第 5 页"
    plan_call, answer_call = make_calls(
        answer_reply("力传感器成本高", citation), sleep=0.01
    )

    view = ask(
        "刀具磨损监测有哪些方法？",
        FakeRetriever(fallback=HITS),
        plan_call=plan_call,
        answer_call=answer_call,
    )

    row = view["sections"]["研究现状"][0]
    linked = row["citations"][0]
    assert linked["kind"] == "exact"
    # 点开看原文靠的就是这个下标：它必须指回候选，且能拿到**全文**（不是提示词里的截断版）
    assert (
        view["candidates"][linked["hit_index"]]["text"]
        == "力传感器精度高但成本高、不易安装。"
    )
    assert view["citations"]["exact"] == 1
    assert view["citations"]["invalid"] == []


def test_ask_view_records_elapsed_and_cost_per_call():
    plan_call, answer_call = make_calls(
        answer_reply("有方法", "刀具磨损监测/综述A.pdf:第 1 页"), sleep=0.01
    )
    view = ask(
        "刀具磨损监测有哪些方法？",
        FakeRetriever(fallback=HITS),
        plan_call=plan_call,
        answer_call=answer_call,
    )

    assert [c["stage"] for c in view["calls"]] == ["plan", "answer"]
    assert all(c["elapsed_s"] and c["elapsed_s"] > 0 for c in view["calls"])
    assert (
        view["elapsed"]["model_s"] >= sum(c["elapsed_s"] for c in view["calls"]) * 0.9
    )
    assert view["elapsed"]["total_s"] >= view["elapsed"]["model_s"]
    assert view["cost"]["usd"] > 0 and view["cost"]["cny"] > 0
    assert view["tokens"]["prompt"] == 2000 and view["tokens"]["completion"] == 400


def test_ask_logs_second_round_when_plan_says_not_enough():
    follow_up = "力传感器 安装"
    retriever = FakeRetriever(
        table={
            follow_up: [make_hit("刀具磨损监测/补充C.pdf", "安装方式影响部署成本。")]
        },
        fallback=HITS,
    )
    plan_call, answer_call = make_calls(
        answer_reply("有方法", "刀具磨损监测/综述A.pdf:第 1 页"),
        plan=plan_reply(False, gaps=["缺安装"], follows=[follow_up]),
    )

    view = ask(
        "刀具磨损监测有哪些方法？",
        retriever,
        plan_call=plan_call,
        answer_call=answer_call,
    )

    assert retriever.queries == ["刀具磨损监测有哪些方法？", follow_up]
    assert [r["round"] for r in view["retrieval"]] == [1, 2]
    assert len(view["candidates"]) == 3  # 两轮合并、去重


def test_ask_skips_model_when_no_candidates():
    """没有候选就不该花钱问模型"你知道吗"——拒答的价值在于代价低且诚实。"""
    called = []

    def boom(*_args, **_kwargs):
        called.append(1)
        raise AssertionError("无候选时不该调用模型")

    view = ask(
        "笔记里没写过的问题",
        FakeRetriever(fallback=[]),
        plan_call=boom,
        answer_call=boom,
    )

    assert called == []
    assert view["insufficient"] is True
    assert view["cost"]["usd"] == 0
    assert view["calls"] == []


def test_ask_degrades_when_answer_is_not_json():
    plan_call, _ = make_calls("", plan=plan_reply(True))
    view = ask(
        "刀具磨损监测有哪些方法？",
        FakeRetriever(fallback=HITS),
        plan_call=plan_call,
        answer_call=lambda prompt, **kw: (
            "这不是 json",
            dict(USAGE, finish_reason="length"),
        ),
    )

    assert view["degraded"] is True
    assert "截断" in view["reason"]
    assert all(not rows for rows in view["sections"].values())


def test_ask_writes_web_log_record(tmp_path):
    log = tmp_path / "usage_log.jsonl"
    plan_call, answer_call = make_calls(
        answer_reply("有方法", "刀具磨损监测/综述A.pdf:第 1 页")
    )
    ask(
        "刀具磨损监测有哪些方法？",
        FakeRetriever(fallback=HITS),
        plan_call=plan_call,
        answer_call=answer_call,
        log_path=log,
        index_label="data/index",
    )

    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["source"] == "web"  # 与命令行那条记录区分开
    assert record["index"] == "data/index"
    assert record["prompt_tokens"] == 2000
    assert record["adopted"] is None and record["note"] == ""  # 这两格只有人能填
    assert record["citation_exact"] == 1


def test_ask_survives_broken_progress_callback():
    """进度条坏了不该让一次已经付过钱的回答丢掉。"""

    def broken(_event):
        raise RuntimeError("回调炸了")

    plan_call, answer_call = make_calls(
        answer_reply("有方法", "刀具磨损监测/综述A.pdf:第 1 页")
    )
    view = ask(
        "刀具磨损监测有哪些方法？",
        FakeRetriever(fallback=HITS),
        plan_call=plan_call,
        answer_call=answer_call,
        on_event=broken,
    )

    assert view["sections"]["研究现状"][0]["point"] == "有方法"


def test_timing_retriever_forwards_unknown_attributes():
    trace = RunTrace()
    proxy = TimingRetriever(FakeRetriever(fallback=HITS), trace)

    assert proxy.marker == "我是检索器上挂着的属性"
    proxy.search("任意查询")
    assert trace.retrieval_rounds[0]["hits"] == 2


def test_link_citations_marks_unresolved_branch():
    """“回查不到”这条分支正常跑不到（校验已经把它挑出来了）。

    留着是因为**它比静默删掉诚实**：万一以后校验口径变了，界面会显示
    "这条引用回查不了"，而不是把它画成一个能点开的样子（T28 的教训）。
    """
    hit = make_hit("刀具磨损监测/综述A.pdf", "正文")
    linked = _link_citations(
        {"研究现状": [{"point": "x", "citations": ["不存在的文件.pdf:第 9 页"]}]},
        [hit],
    )
    assert linked["研究现状"][0]["citations"][0]["kind"] == "unresolved"


def test_render_view_markdown_marks_partial_and_keeps_numbers():
    citation = "刀具磨损监测/综述A.pdf"  # 只给文件名 → 缺定位
    plan_call, answer_call = make_calls(answer_reply("有方法", citation))
    view = ask(
        "刀具磨损监测有哪些方法？",
        FakeRetriever(fallback=HITS),
        plan_call=plan_call,
        answer_call=answer_call,
    )

    text = render_view_markdown(view)
    assert "缺定位" in text
    assert "候选：2 条" in text
    assert "成本：" in text and "耗时：" in text


# ---- 会话表 ----


def test_session_store_keeps_last_question_and_clears():
    store = SessionStore(max_turns=2)
    sid = new_session_id()
    assert store.last_question(sid) is None

    store.add(sid, question="第一问", retrieval_query="第一问")
    store.add(sid, question="第二问", retrieval_query="第一问 第二问")
    assert store.last_question(sid) == "第二问"

    store.add(sid, question="第三问", retrieval_query="第三问")
    assert [t["question"] for t in store.history(sid)] == [
        "第二问",
        "第三问",
    ]  # 上限生效
    store.clear(sid)
    assert store.history(sid) == []
    assert (
        store.add(None, question="无会话", retrieval_query="") == 0
    )  # 无 session 不记


# ---- HTTP：SSE 与边界 ----


def parse_sse(text: str) -> list[dict]:
    events = []
    for frame in text.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


def post_ask(client, monkeypatch, question="刀具磨损监测有哪些方法？", **body):
    """打一次 /api/ask 并把模型调用换成假的（monkeypatch 打在 service 的默认值上）。

    注意打的是 `web.service` 里那两个名字：server 通过 `ask(plan_call=...)` 的默认参数取值，
    所以替换模块属性即可生效——这也是"默认参数取模块级函数"的一个好处。
    """
    from web import service

    # 给假调用一点真实耗时：不然"耗时"四舍五入进两位小数就成了 0.0，
    # 断言会变成在测"计时器有没有装"，而不是"计时器准不准"。
    plan_call, answer_call = make_calls(
        answer_reply("振动与电流是工业可接受的两类", "刀具磨损监测/综述A.pdf:第 1 页"),
        sleep=0.01,
    )
    monkeypatch.setattr(service, "default_plan_call", plan_call)
    monkeypatch.setattr(service, "default_answer_call", answer_call)
    payload = {"question": question, **body}
    return client.post("/api/ask", json=payload)


def test_api_ask_streams_stages_then_result(client, monkeypatch):
    resp = post_ask(client, monkeypatch)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "result"
    stages = [(e["stage"], e["status"]) for e in events if e["type"] == "stage"]
    assert ("retrieve", "start") in stages
    assert ("retrieve", "done") in stages
    assert ("plan", "start") in stages
    assert ("answer", "done") in stages

    view = events[-1]["data"]
    assert view["sections"]["研究现状"][0]["citations"][0]["kind"] == "exact"
    assert view["cost"]["usd"] > 0
    assert view["elapsed"]["total_s"] > 0


def test_api_ask_carries_previous_question_only_for_follow_ups(client, monkeypatch):
    retriever = client.app.state.retriever
    first = post_ask(client, monkeypatch, question="刀具磨损监测有哪些主流方法？")
    session_id = parse_sse(first.text)[0]["session_id"]

    resp = post_ask(
        client, monkeypatch, question="那它的局限呢？", session_id=session_id
    )
    view = parse_sse(resp.text)[-1]["data"]

    assert view["carried_from"] == "刀具磨损监测有哪些主流方法？"
    assert view["retrieval_query"] == "刀具磨损监测有哪些主流方法？ 那它的局限呢？"
    # 检索真的用了拼好的词（界面显示的与实际执行的一致）
    assert retriever.queries[-1] == view["retrieval_query"]
    # 界面上那句提示靠的就是 carried_previous 事件
    assert parse_sse(resp.text)[0]["carried_previous"] == "刀具磨损监测有哪些主流方法？"

    # 关掉开关就不拼（用户能看见、也能控制）
    resp = post_ask(
        client,
        monkeypatch,
        question="那它的局限呢？",
        session_id=session_id,
        carry_context=False,
    )
    view = parse_sse(resp.text)[-1]["data"]
    assert view["carried_from"] is None
    assert view["retrieval_query"] == "那它的局限呢？"


def test_api_ask_rejects_empty_and_overlong_questions(client):
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400
    too_long = client.post(
        "/api/ask", json={"question": "a" * (MAX_QUESTION_CHARS + 1)}
    )
    assert too_long.status_code == 400
    assert str(MAX_QUESTION_CHARS) in too_long.json()["detail"]


def test_api_ask_requires_token_when_configured(monkeypatch):
    app = create_app(
        retriever=FakeRetriever(fallback=HITS), token="s3cret", log_path=None
    )
    with TestClient(app) as c:
        assert c.get("/api/health").json() == {
            "service": "notes-qa web",
            "auth": True,
            "authorized": False,
        }
        assert c.post("/api/ask", json={"question": "随便问问"}).status_code == 401

        from web import service

        plan_call, answer_call = make_calls(
            answer_reply("有方法", "刀具磨损监测/综述A.pdf:第 1 页")
        )
        monkeypatch.setattr(service, "default_plan_call", plan_call)
        monkeypatch.setattr(service, "default_answer_call", answer_call)
        ok = c.post(
            "/api/ask",
            json={"question": "刀具磨损监测有哪些方法？"},
            headers={"X-Token": "s3cret"},
        )
        assert ok.status_code == 200


def test_api_reset_forgets_session(client, monkeypatch):
    resp = post_ask(client, monkeypatch)
    session_id = parse_sse(resp.text)[0]["session_id"]

    assert (
        client.post("/api/reset", json={"session_id": session_id}).json()["ok"] is True
    )
    assert client.app.state.sessions.history(session_id) == []


def test_index_page_is_served(client):
    resp = client.get("/")

    assert resp.status_code == 200
    assert "notes-qa" in resp.text
    assert "text/html" in resp.headers["content-type"]


def test_health_reports_missing_index(tmp_path):
    app = create_app(index_dir=tmp_path / "没有这个索引", token=None)
    with TestClient(app) as c:
        body = c.get("/api/health").json()

    assert body["ready"] is False
    assert body["index"]["usable"] is False
    assert "chunks.jsonl" in " ".join(body["index"]["problems"])
    assert "rag.index build" in body["index"]["hint"]
    assert body["pricing"]["text"]  # 价格说明与索引状态无关，仍然给出
    # 自检**不许替模型调用下结论**：真正发请求时由 rag.rerank 自己 load_dotenv，
    # 与本进程的环境变量无关。这里只报"文件在不在"。
    assert set(body["api_key"]) == {"in_env", "env_file", "env_file_exists"}
    assert body["api_key"]["env_file"].endswith(".env")


def test_health_ready_when_retriever_injected(tmp_path):
    """注入了检索器就不依赖索引文件：这是测试与"演示模式"能离线跑的前提。"""
    app = create_app(index_dir=tmp_path / "空", retriever=FakeRetriever(fallback=HITS))
    with TestClient(app) as c:
        body = c.get("/api/health").json()

    assert body["ready"] is True and body["retriever_loaded"] is True


def test_api_ask_without_index_reports_actionable_error(tmp_path):
    """索引不可用时，错误必须变成一条可读事件，并说清**缺什么**（而不是一句"初始化失败"）。

    这条路径在给别人试用时最可能出现：克隆下来、还没建库就想打开看看。
    """
    app = create_app(index_dir=tmp_path / "还没建库", retriever=None, log_path=None)
    with TestClient(app) as c:
        events = parse_sse(c.post("/api/ask", json={"question": "随便问问"}).text)

    error = [e for e in events if e["type"] == "error"]
    assert error, events
    assert "索引不可用" in error[0]["message"]
    assert "chunks.jsonl" in error[0]["message"]
