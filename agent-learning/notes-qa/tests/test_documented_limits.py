"""钉住"文档里对外承诺过的数字"。

行为测试（`test_read_note` / `test_list_search_notes`）都用 `tools.MAX_READ_LINES`
这类常量做比较，好处是"调参"不会误报；代价是**常量被误改也没人发现**。
第 14 周 W14-6 用变异测试验证过这一点：把 `MAX_READ_LINES` 从 200 改成 199，
两条测试入口（pytest 与 `eval_cases.py --unit-only`）**都全绿**。

所以这里单独守住"已经写进文档、对外承诺过"的那几个数字：改它们就得同时改文档，
别让文档与代码悄悄分叉。

**不钉**那些明确标注"初值、待调"的参数（例如 T3 的 `MAX_HISTORY_TURNS`）——
钉住待调参数只会让本来正常的调整莫名其妙变红。
"""

import pytest
import tools
from main import MAX_QUESTION_LEN

DOCUMENTED_LIMITS = [
    (
        "单文件读取上限 1MB",
        tools.MAX_FILE_BYTES,
        1 * 1024 * 1024,
        "项目设计.md 第 122/138 行",
    ),
    ("单次读取行数上限", tools.MAX_READ_LINES, 200, "项目设计.md 第 122/138 行"),
    ("搜索结果条数上限", tools.MAX_SEARCH_RESULTS, 5, "项目设计.md 第 115/138 行"),
    ("问题长度上限", MAX_QUESTION_LEN, 500, "项目设计.md 第 198 行"),
]


@pytest.mark.parametrize(
    ("label", "actual", "expected", "source"),
    DOCUMENTED_LIMITS,
    ids=[item[0] for item in DOCUMENTED_LIMITS],
)
def test_documented_limits_stay_in_sync(label, actual, expected, source):
    assert actual == expected, f"{label} 与文档不一致（文档见 {source}）"
