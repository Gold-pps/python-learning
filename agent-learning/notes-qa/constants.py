"""跨模块共享的常量。

本模块**不 import 任何项目内模块、也不 import 第三方库**——
因为 rag/ 在 .venv-rag 里跑（只有 torch 相关依赖），
而 tools.py 在主 .venv 里跑（有 openai-agents）。
两边都需要 MAX_FILE_BYTES，抽到这里就避免了 rag 被迫拖入 agents。

第 18 周追加 DeepSeek 的接入点（base_url 与模型名），理由与上面**完全相同**：
`rag/rerank.py` 要在 .venv-rag 里发请求，而 `config.py` 依赖 `openai-agents`
（就是 T18 那条依赖链），所以重排只能从这里取接入点。
同时 `config.py` 改为引用本模块——同一个模型名一旦抄成两份，迟早会漂移。
"""

MAX_FILE_BYTES = (
    1048576  # 1 MiB —— **笔记工具**的单文件上限（read_note / search_notes）
)

# ---- 语料解析的单文件上限（第 20 周补）----
# 与上面那个**故意不是同一个值**，因为它们防的根本不是一回事：
#   MAX_FILE_BYTES：防"把超大文本塞进模型的上下文"——1 MiB 对本项目的 .md 笔记已经很宽；
#   MAX_PARSE_BYTES：防"误把 GB 级文件读进内存"——对 PDF 语料要按"论文"的量级给，
#                    1 MiB 会把一篇 2 MB 的期刊论文直接挡在门外
#                    （第 20 周实测：44 篇真实资料里被挡掉 29 篇）。
# 同 T1 那条"同一个常量、两种策略"：**别为了统一把两种合理语义合并成一条**。
MAX_PARSE_BYTES = 64 * 1024 * 1024  # 64 MiB

# ---- DeepSeek 接入点（唯一真相来源：config.py 与 rag/rerank.py 都引用这里）----
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
