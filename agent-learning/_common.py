"""周脚本共用的 DeepSeek 接入配置（第 14 周 W14-4 抽出）。

第 2~6 周的 8 个脚本里，这段配置被抄了 8 遍：读 `.env` → 取密钥 → 建客户端 →
三步设置（默认客户端指向 DeepSeek / 走 Chat Completions / 关闭追踪），
连"没有找到 DEEPSEEK_API_KEY"的提示语都抄了 8 遍。

抄 8 遍的代价不在写的时候，而在改的时候：换 `base_url`、加超时、换模型名，
就要改 8 个文件，而且**一定会漏掉一两个**。这里定成唯一的一份。

用法（脚本与本文件同目录，直接 import）：

    from _common import MODEL                # Agents SDK 路径（01~06）
    from _common import MODEL, sync_client   # 同步路径（00_first_api_call.py）

import 本模块就完成了接入配置，脚本里不需要再写 setUp 那几行。

为什么不让 `notes-qa/config.py` 也用这里：那个文件额外带超时与重试配置，
而且属于项目代码（`notes-qa/`）而不是周练习脚本；两者相同点只是"都指向 DeepSeek"，
跨目录共享会引入一个不必要的依赖方向（项目依赖练习脚本）。
"""

import os
from pathlib import Path

from agents import (
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"  # 全项目统一用它，不要写 gpt-*

AGENT_LEARNING_DIR = Path(__file__).resolve().parent
load_dotenv(AGENT_LEARNING_DIR / ".env")

_api_key = os.environ.get("DEEPSEEK_API_KEY")
if not _api_key:
    raise SystemExit(
        "没有找到 DEEPSEEK_API_KEY。\n"
        "请先在 platform.deepseek.com 创建 API Key，"
        "再设置环境变量或在 agent-learning/.env 里配置后重试。"
    )

# 同步客户端：给不走 Agents SDK 的脚本（00_first_api_call.py）用
sync_client = OpenAI(api_key=_api_key, base_url=BASE_URL)

# 异步客户端 + 三步设置：import 本模块即生效
client = AsyncOpenAI(api_key=_api_key, base_url=BASE_URL)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)
