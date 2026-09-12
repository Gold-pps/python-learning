"""DeepSeek + OpenAI Agents SDK 配置。

其他模块 import 这个文件后，请求会自动发往 DeepSeek。
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

# ---- 路径 ----
NOTES_QA_DIR = Path(__file__).resolve().parent          # .../agent-learning/notes-qa
AGENT_LEARNING_DIR = NOTES_QA_DIR.parent                # .../agent-learning
NOTES_ROOT = AGENT_LEARNING_DIR                         # 笔记根目录

# ---- 统一模型名 ----
MODEL = "deepseek-flash"

# ---- 超时与重试 ----
# 单次 HTTP 请求超时（秒）。SDK 默认 600 秒、重试 2 次，
# 一旦网络卡住，命令行会长时间没有任何输出。
REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 1

# 一轮提问的总上限（秒）。覆盖“模型思考 + 工具调用 + 再次请求”的完整循环，
# 防止个别请求反复重试把一次提问拖到不可接受的长度。
OVERALL_TIMEOUT = 120.0

# ---- 读取 .env ----
load_dotenv(AGENT_LEARNING_DIR / ".env")

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError(
        "没有找到 DEEPSEEK_API_KEY，请检查 agent-learning/.env 是否存在且格式为 "
        "DEEPSEEK_API_KEY=sk-..."
    )

# ---- 指向 DeepSeek 的三步配置 ----
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    timeout=REQUEST_TIMEOUT,
    max_retries=MAX_RETRIES,
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)
