"""前端冒烟（node）：把 `index.html` 的内联脚本放进一个迷你 DOM 里跑一遍。

为什么值得单独一个入口——第 23 周 **头两次真跑，坏的全是"画出来"这一步**：

- **T37**：进度条压根不渲染（`<li>` 只塞进 Map，没 `appendChild` 到 DOM）；
- **T39**：改 `stepLine` 函数签名漏改调用处，界面上直接是
  "连接中断：TypeError: Cannot read properties of undefined (reading 'stage')"。

两次后端都是对的：SSE 事件序列、字段、引用校验全绿——**"协议通过"不等于"界面画出来了"**。
前端没有构建步骤，也不值得为它引一整套 jsdom/puppeteer；`tests/frontend_smoke.mjs`
用四十行 `makeEl()` 就能把这一类 bug 挡在提交前，所以这里只做一层转接。

**没有 node 就 skip**：它是附加保险，不是运行前提。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "frontend_smoke.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 才能跑前端冒烟")
def test_frontend_smoke_renders_progress_and_opens_citations():
    proc = subprocess.run(
        ["node", str(SCRIPT)],
        capture_output=True,
        text=True,
        # Windows 上子进程输出的编码不一定等于本机 locale，解码失败会让断言拿不到内容
        # （conftest 建 junction 那处踩过同一个坑：本该 skip 的用例报成 FAILED）。
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert proc.returncode == 0, f"前端冒烟失败：\n{proc.stdout}\n{proc.stderr}"
    assert "FRONTEND SMOKE OK" in proc.stdout
