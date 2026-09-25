from pathlib import Path

p = Path("README.md")
s = p.read_text(encoding="utf-8")

anchor = "## 项目：notes-qa"
assert anchor in s, "未找到插入锚点"

new_section = """### 5. RAG 环境（可选，第 16 周起）

`.venv-rag` 是独立环境，只在跑 RAG（向量化 + 索引）时需要。
主 `.venv` 保持轻量（不含 torch），日常测试仍用主环境。

**Windows（PowerShell）**：

```powershell
# 建独立环境
uv venv --python 3.13 .venv-rag

# 二选一：CPU 版 torch（无 NVIDIA GPU / 想省空间）
uv pip install --python .venv-rag\\Scripts\\python.exe torch --index-url https://download.pytorch.org/whl/cpu

# 二选一：CUDA 版 torch（有 NVIDIA GPU；cu版本以 nvidia-smi 显示的 CUDA Version 为准）
uv pip install --python .venv-rag\\Scripts\\python.exe torch==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130

# RAG 依赖（sentence-transformers 会拉 CPU 版 torch，所以 CUDA 版那一步要在它之后）
uv pip install --python .venv-rag\\Scripts\\python.exe sentence-transformers fastembed pypdf python-docx