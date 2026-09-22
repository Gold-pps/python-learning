"""第 2 周 · 第 1 步：先不碰 Agent 框架，直接调用一次 DeepSeek。

运行前准备：
1. 在 platform.deepseek.com 创建 API Key；
2. 设置环境变量 DEEPSEEK_API_KEY，或在同目录建 .env 文件；
3. 在安装过 openai-agents 的 Python 环境里运行：
   python 00_first_api_call.py
"""

# 读 .env、取密钥、建客户端都收在 `_common.py` 里（第 14 周 W14-4 抽出）。
# 本脚本不走 Agents SDK，所以用同步客户端 sync_client。
from _common import MODEL, sync_client

response = sync_client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "你是一个耐心的老师。"},
        {"role": "user", "content": "用一句话解释什么是 Agent。"},
    ],
)

print(response.choices[0].message.content)
