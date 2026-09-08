"""第 2 周 · 第 1 步：先不碰 Agent 框架，直接调用一次 DeepSeek。

运行前准备：
1. 在 platform.deepseek.com 创建 API Key；
2. 设置环境变量 DEEPSEEK_API_KEY，或在同目录建 .env 文件；
3. 在安装过 openai-agents 的 Python 环境里运行：
   python 00_first_api_call.py
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit(
        "没有找到 DEEPSEEK_API_KEY。\n"
        "请先在 platform.deepseek.com 创建 API Key，"
        "再设置环境变量或创建 .env 文件后重试。"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "你是一个耐心的老师。"},
        {"role": "user", "content": "用一句话解释什么是 Agent。"},
    ],
)

print(response.choices[0].message.content)
