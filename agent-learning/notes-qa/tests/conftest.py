"""pytest 公共装置。

两个关键设计：

1. **测试不碰真实仓库**：`notes_root` 把 `config.NOTES_ROOT` 换成 `tmp_path` 下的临时目录，
   探针文件（大文件、符号链接、隐藏文件）全部写在里面，由 pytest 兜底回收。

   这正是第 13 周 T2 的教训：旧评估脚本把探针写在仓库根，靠手写 `finally` 清理，
   结果 `os.rmdir` 删不掉符号链接、异常又被 `except OSError: pass` 吞掉，
   表现为"测试全绿 + 仓库多一个断链"。现在**清理逻辑不再由我们负责**，
   这类 bug 在结构上就没有了。

2. **测试不需要真实密钥**：先把 `DEEPSEEK_API_KEY` 塞一个占位值，再 import `config`。
   否则 `config.py` 会因为找不到 Key 直接抛 RuntimeError，第 15 周的 CI（不联网、不带密钥）
   就永远跑不起来。`load_dotenv()` 默认不覆盖已存在的环境变量，所以 `setdefault` 足够。
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-placeholder")

NOTES_QA_DIR = Path(__file__).resolve().parent.parent
if str(NOTES_QA_DIR) not in sys.path:
    # pytest.ini 里已经配了 pythonpath，这里再兜一次：无论从哪个目录调用 pytest 都能 import 到
    sys.path.insert(0, str(NOTES_QA_DIR))

import pytest  # noqa: E402

import config  # noqa: E402


@pytest.fixture
def notes_root(tmp_path, monkeypatch):
    """把工具看到的笔记根目录换成临时目录，返回该目录。

    工具内部是 `config.NOTES_ROOT.resolve()`（调用时才读），所以替换这个属性即可生效，
    无需改动被测代码——这也是"配置从一处读取"带来的可测性。
    """
    root = tmp_path / "笔记根"
    root.mkdir()
    monkeypatch.setattr(config, "NOTES_ROOT", root)
    return root


@pytest.fixture
def write(notes_root):
    """在临时笔记根里写一个文件（自动建父目录），返回它的路径。"""
    def _write(rel: str, text: str) -> Path:
        path = notes_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return path
    return _write


_SANDBOX_GUARD = "探针只能建在临时目录里：一旦落到真实仓库，就会变成下一个 `_probe_*` 残留"


def _make_link(link: Path, target: Path, *, is_dir: bool) -> None:
    """建目录/文件链接；平台或权限不允许时直接 skip，而不是让测试假绿。"""
    try:
        link.symlink_to(target, target_is_directory=is_dir)
        return
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"当前环境无法创建链接（{type(exc).__name__}）：{exc}")
    # Windows 普通用户可以建 junction（不需要开发者模式）
    created = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"New-Item -ItemType Junction -Path '{link}' -Target '{target}' | Out-Null"],
        check=False, capture_output=True, text=True,
    )
    if created.returncode != 0:
        pytest.skip(f"junction 创建失败：{created.stderr.strip()[:200]}")


@pytest.fixture
def make_dir_link(notes_root, tmp_path):
    """返回一个"在笔记根内建目录链接"的函数，链接目标默认放在笔记根之外。

    **必须显式依赖 `notes_root`**：`monkeypatch` 的生效靠 fixture 依赖链触发，
    少写一个参数就会出现"链接建到了真实仓库"这种偶发污染
    （第 14 周迁移测试时真踩过：只有 `make_file_link` 的那条用例把链接建进了
    `agent-learning/`，而同时用了 `write` 的用例因为 `write` 依赖 `notes_root` 而"碰巧"正确）。
    """
    def _make(rel_link: str) -> tuple[Path, Path]:
        target = tmp_path / "根外目录"
        target.mkdir(exist_ok=True)
        link = notes_root / rel_link
        assert link.is_relative_to(tmp_path), f"{_SANDBOX_GUARD}（{link}）"
        _make_link(link, target, is_dir=True)
        return link, target
    return _make


@pytest.fixture
def make_file_link(notes_root, tmp_path):
    """返回一个"在笔记根内建文件链接"的函数，链接目标默认放在笔记根之外。"""
    def _make(rel_link: str, content: str = "OUTSIDE-SECRET\n") -> tuple[Path, Path]:
        target = tmp_path / "根外文件.md"
        target.write_text(content, encoding="utf-8", newline="\n")
        link = notes_root / rel_link
        assert link.is_relative_to(tmp_path), f"{_SANDBOX_GUARD}（{link}）"
        link.parent.mkdir(parents=True, exist_ok=True)
        _make_link(link, target, is_dir=False)
        return link, target
    return _make
