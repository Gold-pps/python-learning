"""命令行入口与配置的契约测试（原 U11、U12）。

这两个用例不需要笔记目录，测的是"入口的边界"与"超时配置没退化成无限等待"。
"""

import config
from main import MAX_QUESTION_LEN, check_input


def test_check_input_boundaries():
    """U11：刚好 500 字符放行，501 字符被拦下（不走模型）。"""
    assert check_input("a" * MAX_QUESTION_LEN) is None

    error = check_input("a" * (MAX_QUESTION_LEN + 1))
    assert error is not None
    assert "过长" in error


def test_timeout_and_retry_config():
    """U12：请求层与整轮都有上限，重试次数不是负数。"""
    assert isinstance(config.REQUEST_TIMEOUT, (int, float))
    assert config.REQUEST_TIMEOUT > 0
    assert isinstance(config.OVERALL_TIMEOUT, (int, float))
    assert config.OVERALL_TIMEOUT > 0
    assert isinstance(config.MAX_RETRIES, int)
    assert config.MAX_RETRIES >= 0
