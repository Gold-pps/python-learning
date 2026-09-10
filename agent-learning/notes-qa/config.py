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

# ---- 读取 .env ----
load_dotenv(AGENT_LEARNING_DIR / ".env")

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError(
        "没有找到 DEEPSEEK_API_KEY，请检查 agent-learning/.env 是否存在且格式为 "
        "DEEPSEEK_API_KEY=sk-..."
    )

# ---- 指向 DeepSeek 的三步配置 ----
client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# ---- 统一模型名 ----
MODEL = "deepseek-flash"