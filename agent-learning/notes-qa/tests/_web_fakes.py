"""Web 层测试的**夹具底座**：假检索器、假模型调用，以及从真实代码派生的视图夹具。

为什么单独抽一个模块（而不是留在 `test_web.py` 里）：假件现在有**两个**使用者——
`test_web.py`（离线测试）与 `test_web_frontend.py`（把一份视图喂给前端冒烟）。
抄两份的代价已经付过一次，就是台账 T43：

> 前端冒烟的视图是**手写**的，里面写了 `doc_type: "pdf"`；
> 而真实视图当时**没有**这个字段（`SearchHit.to_dict()` 里没有），
> 于是测试全绿、界面上 PDF 被标成"行/段 4~4"。

所以这里的 `build_sample_view()` 不手写字段，而是**真跑一遍 `ask()`**
（只换掉模型调用与检索器这两个既有注入点），把真实产物存成夹具。
夹具一旦与真实输出不一致，`test_web_frontend.py` 里的"新鲜度"测试会直接红。

**归一化**：`elapsed` / `cost`（含每次调用的）会被换成固定值——它们依赖"当前是不是高峰时段"，
不归一化的话夹具会在上午和晚上长得不一样，diff 全是噪声；
这几项的行为由 `test_web.py` 单独断言（那是纯函数，测得住）。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

# 直接跑本文件时（`uv run python agent-learning/notes-qa/tests/_web_fakes.py`），
# `sys.path[0]` 是 `tests/`，而 `rag` / `web` 在上一级——pytest 靠 `pyproject.toml` 的
# `pythonpath` 解决了这件事，命令行跑就得自己接一下。所以这几行**必须在导入之前**。
_NOTES_QA = Path(__file__).resolve().parents[1]
if str(_NOTES_QA) not in sys.path:
    sys.path.insert(0, str(_NOTES_QA))

from rag.chunk import Chunk
from rag.retriever import SearchHit
from web.service import ask

USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 200,
    "cached_tokens": 400,
    "finish_reason": "stop",
}


def make_hit(
    name: str,
    text: str,
    locator: str = "第 1 页",
    start: int = 1,
    end: int | None = None,
) -> SearchHit:
    chunk = Chunk.create(
        source_path=name,
        doc_type="pdf",
        locator=locator,
        start=start,
        end=end if end is not None else start,
        text=text,
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


# 两个候选：一个能被精确引用（文件 + 页码），一个只能回到文件层面。
# `幽灵.pdf` **故意不在候选里**——它用来产生"回查不到"那一档引用。
A_TEXT = "这是候选片段的原文全文，点开引用要能看到它。"
B_TEXT = "B 的原文。"

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

SAMPLE_HITS = [
    make_hit("刀具/A.pdf", A_TEXT, locator="第 2 页", start=2),
    make_hit("刀具/B.pdf", B_TEXT, locator="第 5 页", start=5),
]

# 这份答案刻意凑齐界面要展示的三种引用档 + 一个空段 + 缺口声明：
#   研究现状 → 精确引用 1、只到文件 1、回查不到 1
#   方法对比 → 有内容但**没有**引用（界面要能画出空引用行）
#   结论 / 可复用点 → 空（界面要写"候选不足以支撑这一段"，不能静默省略）
SAMPLE_ANSWER = json.dumps(
    {
        "研究现状": [
            {
                "point": "有两种主流做法",
                "citations": ["刀具/A.pdf:第 2 页", "刀具/B.pdf", "幽灵.pdf:第 9 页"],
            }
        ],
        "方法对比": [
            {"point": "方法A", "pros": "便宜", "cons": "精度差", "citations": []}
        ],
        "结论": [],
        "可复用点": [],
        "insufficient": True,
        "note": "资料中没有直接答案，以下是相邻证据。",
    },
    ensure_ascii=False,
)

SAMPLE_QUESTION = "刀具磨损监测有哪些主流方法？"

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_view.json"


def _normalize(view: dict) -> dict:
    """把随时间/时段变化的字段换成固定值（见模块 docstring 的"归一化"）。

    **每一处都得列全**：漏一个，夹具就会"每次生成都不一样"，新鲜度测试立刻红——
    这不是理论担心，加这个函数时 `retrieval[].elapsed_s`（每一轮检索的耗时）就漏了，
    测试当场把它抓出来了。
    """
    view = copy.deepcopy(view)
    view["generated_at"] = "2026-01-01T00:00:00+08:00"  # 真实值 = 生成时刻，每次都不同
    view["elapsed"] = {"total_s": 20.0, "retrieval_s": 0.05, "model_s": 20.0}
    view["cost"] = {"peak": False, "usd": 0.0027, "cny": 0.019}
    for call in view["calls"]:
        call["elapsed_s"] = 10.0
        if "cost" in call:
            call["cost"] = {"usd": 0.0014, "cny": 0.0099}
    for round_ in view["retrieval"]:
        round_["elapsed_s"] = 0.05
    return view


def build_sample_view() -> dict:
    """跑一遍真实的 `ask()`（只换模型与检索器），返回可喂给前端的视图。"""
    plan_call, answer_call = make_calls(SAMPLE_ANSWER)
    view = ask(
        SAMPLE_QUESTION,
        FakeRetriever(fallback=SAMPLE_HITS),
        plan_call=plan_call,
        answer_call=answer_call,
    )
    return _normalize(view)


def write_fixture(path: Path | None = None) -> Path:
    target = Path(path) if path else FIXTURE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_sample_view(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def main() -> int:
    target = write_fixture()
    print(f"已重新生成夹具：{target}")
    print(
        "（它是从真实 ask() 产物导出的，不要手改；改了要被 test_web_frontend.py 抓出来）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
