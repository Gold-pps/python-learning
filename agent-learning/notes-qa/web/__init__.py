"""Web 界面（第 23 周，G4）：把 notes-qa 的检索与综述问答放到浏览器里用。

分工：

- `pricing.py`：token → 金额的估算（价格快照 + 峰谷判定，纯函数）；
- `service.py`：编排层，复用 `rag/review_qa.answer_review()`，补上"每步耗时 / 成本 /
  引用↔候选对应关系"这些**只给界面看**的东西；
- `server.py`：FastAPI 应用 + SSE 进度流 + 启动入口；
- `static/index.html`：单页前端（原生 HTML/JS，不引框架）；
- `使用说明.md`：交付给试用者的说明与能力边界声明。

设计约束（两条来自仓库现状，不是偏好）：

1. **本层不 import `fastembed`**：CI 跑的是 `uv sync --frozen`，**不带 `--extra rag`**，
   顶层 import 会让 CI 直接红。真实后端的构造放在"首次提问时"的懒加载里
   （与 `rag/retriever.py` 把 embedder 交给调用方是同一个理由）；
2. **不 import `config.py`**：那是 Agents SDK 的接入配置（会拖入 `openai-agents`），
   本层只需要 `rag/` 与 `constants.py`。
"""
