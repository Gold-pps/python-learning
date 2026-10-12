"""前端冒烟（node）：把 `index.html` 的内联脚本放进一个迷你 DOM 里跑一遍。

为什么值得单独一个入口——第 23 周 **头两次真跑，坏的全是"画出来"这一步**：

- **T37**：进度条压根不渲染（`<li>` 只塞进 Map，没 `appendChild` 到 DOM）；
- **T39**：改 `stepLine` 函数签名漏改调用处，界面上直接是
  "连接中断：TypeError: Cannot read properties of undefined (reading 'stage')"。

两次后端都是对的：SSE 事件序列、字段、引用校验全绿——**"协议通过"不等于"界面画出来了"**。
前端没有构建步骤，也不值得为它引一整套 jsdom/puppeteer；`tests/frontend_smoke.mjs`
用四十行 `makeEl()` 就能把这一类 bug 挡在提交前，所以这里只做一层转接。

**视图不再手写**（第 25 周改）：那份 view 由 `_web_fakes.build_sample_view()` 从**真实
`ask()` 产物**导出（T43/T44 的教训：手写的夹具写对了字段，真数据却没有它，
于是一个走不到的分支看起来是绿的）。因此这里有两个测试：

1. `test_frontend_fixture_matches_real_view`：夹具与真实输出不一致就红（**不需要 node**）；
2. `test_frontend_smoke_uses_fresh_view`：喂**刚生成**的那份给 node 跑（即使有人忘了重新生成
   夹具，冒烟测的也是真数据的形状）。

**没有 node 就 skip**：它是附加保险，不是运行前提。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from _web_fakes import FIXTURE_PATH, build_sample_view, write_fixture

SCRIPT = Path(__file__).resolve().parent / "frontend_smoke.mjs"


def test_frontend_fixture_matches_real_view():
    """夹具 = 真实 `ask()` 的产物（除时间/成本被归一化）。

    这条测试是**漂移检测**：`web/service.py` 改了视图字段、或 `review_qa` 改了引用口径，
    而夹具没跟着重新生成时，它会红——而不是让冒烟继续拿旧形状测绿。
    """
    fresh = build_sample_view()
    committed = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert fresh == committed, (
        "前端夹具与真实代码的输出不一致。"
        "跑 `uv run python agent-learning/notes-qa/tests/_web_fakes.py` 重新生成；"
        "**不要手改夹具**——它存在的意义就是“长得跟真数据一样”。"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 才能跑前端冒烟")
def test_frontend_smoke_uses_fresh_view(tmp_path):
    """用**刚生成**的视图跑冒烟：测的是真数据形状，不是某次手抄下来的快照。"""
    fixture = write_fixture(tmp_path / "sample_view.json")

    proc = subprocess.run(
        ["node", str(SCRIPT), str(fixture)],
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
