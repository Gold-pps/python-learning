"""第 21 周：综述型问题的题目与判据。

**为什么不能沿用第 19 周的 25 条评测集**：那批题是笔记语料上的**事实型**问题
（答案能在某一小段里找到，判据是"来源文件 + 片段含关键词"）；
第 21 周换了 44 篇论文语料，问的也是**综述型**问题（"有哪些方法、各自优缺点"），
答案天然**跨多篇**，判据必须跟着换：

- 检索侧：先看"有没有把该方向的几篇都捞上来"——**期望子主题命中率**与**不同论文数**，
  离线可测、不花 token（跑法见 `eval/review_coverage.py`）；
- 生成侧：再看"四段式结构是否成立、每条结论的引用能不能回查到原文"
  （跑法见 `rag/review_qa.py`，**会花 token**，由使用者自己跑）。

规划要求"3 个综述型问题实测"，这里备了 5 题（5 个子主题各一题），挑 3 题真跑即可。
`method_keywords` 是人工核对答案时的抓手（"这个方法族到底有没有被提到"），
**不是**自动判分器：综述型答案的措辞差异太大，机器判分会把"换个说法"判成错
（第 19 周判据第一版就是这么错的，见台账 T10）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewCase:
    id: str
    subtopic: str  # 期望命中的子主题（= 语料的一级目录名）
    question: str
    method_keywords: tuple[str, ...]  # 人工核对时看这些方法族有没有被提到


CASES: tuple[ReviewCase, ...] = (
    ReviewCase(
        id="R21-1",
        subtopic="刀具磨损监测",
        question="刀具磨损监测有哪些主流方法？各自的优缺点是什么？",
        method_keywords=(
            "直接测量",
            "间接监测",
            "振动",
            "声发射",
            "电流",
            "图像",
            "深度学习",
            "剩余寿命",
        ),
    ),
    ReviewCase(
        id="R21-2",
        subtopic="切削参数优化",
        question="切削参数优化与加工能耗预测都有哪些建模方法？",
        method_keywords=(
            "经验模型",
            "有限元",
            "响应面",
            "代理模型",
            "机器学习",
            "遗传算法",
            "表面粗糙度",
        ),
    ),
    ReviewCase(
        id="R21-3",
        subtopic="数字孪生与智能车间",
        question="数字孪生在机械加工与车间管理中的研究现状如何？关键技术有哪些？",
        method_keywords=(
            "五维模型",
            "虚实映射",
            "数据采集",
            "仿真",
            "车间调度",
            "智能管控",
        ),
    ),
    ReviewCase(
        id="R21-4",
        subtopic="数控编程与轨迹规划",
        question="数控加工中的轨迹规划与插补方法有哪些主流做法？",
        method_keywords=(
            "NURBS",
            "样条",
            "前瞻",
            "加减速",
            "进给速度",
            "五轴",
        ),
    ),
    ReviewCase(
        id="R21-5",
        subtopic="机床误差补偿",
        question="机床热误差与几何误差的补偿方法有哪些？",
        method_keywords=(
            "热误差",
            "几何误差",
            "误差建模",
            "迁移学习",
            "辨识",
            "补偿策略",
        ),
    ),
)
