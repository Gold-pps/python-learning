"""跨模块共享的常量。

本模块**不 import 任何项目内模块、也不 import 第三方库**——
因为 rag/ 在 .venv-rag 里跑（只有 torch 相关依赖），
而 tools.py 在主 .venv 里跑（有 openai-agents）。
两边都需要 MAX_FILE_BYTES，抽到这里就避免了 rag 被迫拖入 agents。
"""

MAX_FILE_BYTES = 1048576  # 1 MiB
