"""第 19 周评测集：25 条，四类（有答案 / 无答案 / 跨文档 / 需要多跳），英文 3 条。

**定位**：这是 G2 验收用的**检索级**评测集，判的是"能不能把正确的片段捞进前 5"。

来源与沿革：
- `W18-*` 是第 18 周的 10 题抽查集（原文保留、id 不变，台账与笔记里的引用都对得上）；
- `W19-*` 是第 19 周按四类补足的题；
- 第 14 周的 10 条**模型评估**用例不在这里——那是 Agent 级（判据含引用格式），
  本文件是检索级；G2 的"引用准确率 / 无答案拒答率"要用 Agent 级跑分（另见 `rag/评测说明.md`）。

四条纪律（每条都是踩过才有的）：
1. **题面与答案卡只写在这里**：`notes-qa/eval/` 不在 `tools.EXCLUDE_DIRS` 之外（它在里面），
   检索不到。写进 `agent-learning/*.md` 就等于把答案卡发给被评测的 Agent（台账 T8"发现一"）；
2. **无答案题的关键词必须在语料里 0 命中**——`rag_eval.py` 每次跑都会自动复查（T8 的自动检查）；
3. **有答案题的 ground truth 必须逐条核实**：`rag_eval.py` 会验证每个"期望来源 + 关键词"
   真的能在索引里找到，找不到就报 `!!`（第 14 周 T8：一道无解的题会被读成"模型能力差"）；
4. **跨文档 / 多跳题要求"每个期望来源都被覆盖"**，并额外报覆盖率（2 个里中 1 个也算信息）。

判据（`rag_eval.py` 实现）：
- 严格：`top_k` 里存在片段，其 `source_path` 命中某期待来源**且**正文含该来源的关键词；
- 宽松：只看来源文件；两者都报（第 18 周实测差 3 条，"文件对了、片段不对"要单独看，见笔记 3.1）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 四类（与第 14 周定的分类一致，第 19 周沿用）
CATEGORY_ANSWER = "有答案"
CATEGORY_NO_ANSWER = "无答案"
CATEGORY_CROSS = "跨文档"
CATEGORY_MULTIHOP = "需要多跳"
CATEGORIES = (CATEGORY_ANSWER, CATEGORY_NO_ANSWER, CATEGORY_CROSS, CATEGORY_MULTIHOP)


@dataclass(frozen=True)
class Expect:
    """一个"应该被检索到"的来源：某个文件里、含指定关键词之一的片段。"""

    file: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    question: str
    expects: tuple[Expect, ...] = ()
    negative_keywords: tuple[str, ...] = ()
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def expect_files(self) -> tuple[str, ...]:
        return tuple(e.file for e in self.expects)

    @property
    def expect_keywords(self) -> tuple[str, ...]:
        return tuple(k for e in self.expects for k in e.keywords)


def _one(file: str, *keywords: str) -> tuple[Expect, ...]:
    return (Expect(file=file, keywords=keywords),)


CASES: list[Case] = [
    # ---------------- 有答案（10） ----------------
    Case(
        id="W18-1",
        category=CATEGORY_ANSWER,
        question="第 13 周做会话历史 token 实验时，为什么“收敛历史”最后没能省钱？",
        expects=_one("第13周笔记.md", "缓存命中", "工具往返"),
        note="T3 的两条反转结论：请求次数主导输入量 + 截断可能压低缓存命中率。",
        tags=("自然语言",),
    ),
    Case(
        id="W18-2",
        category=CATEGORY_ANSWER,
        question="为什么小批量做向量化时，ONNX 后端反而比 CUDA 后端更快？",
        expects=_one("第17周笔记.md", "CUDA 上下文", "加载"),
        note="T19：总耗时 = 加载 + 推理，短任务里加载成本主导。",
        tags=("自然语言",),
    ),
    Case(
        id="W18-3",
        category=CATEGORY_ANSWER,
        question="read_note 和 search_notes 碰到超大文件时的处理方式有什么不同？为什么这么设计？",
        expects=_one("工程化改造记录.md", "跳过并计数", "报错"),
        note="T1：同一个常量两种合理语义，别为了“统一”强行合并。",
        tags=("自然语言",),
    ),
    Case(
        id="W18-4",
        category=CATEGORY_ANSWER,
        question="prepare_history",
        expects=_one("工程化改造记录.md", "prepare_history"),
        note="对照组：短关键词提问，第一阶段的关键词基线本来就该命中。",
        tags=("关键词型",),
    ),
    Case(
        id="W18-5",
        category=CATEGORY_ANSWER,
        question="为什么不能拿 git 的还原命令去清掉自己刚写的临时改动？",
        expects=_one("工程化改造记录.md", "文件级"),
        note="T9 的现场：不报错，但把该文件所有未提交改动一起冲掉。",
        tags=("自然语言",),
    ),
    Case(
        id="W18-6",
        category=CATEGORY_ANSWER,
        question="换一台电脑以后，怎么用一条命令把运行环境装回来？",
        expects=_one("学习进度.md", "uv sync"),
        note="“换机重建”那一节。",
        tags=("自然语言",),
    ),
    Case(
        id="W18-7",
        category=CATEGORY_ANSWER,
        question="把长文档切成片段时，为什么相邻片段之间要故意留一段重叠？",
        expects=_one("第16周笔记.md", "重叠"),
        note="切分参数：目标 300~500 字 / 重叠 50 字。",
        tags=("自然语言",),
    ),
    Case(
        id="W18-8",
        category=CATEGORY_ANSWER,
        question="What does RRF stand for, and why fuse by rank rather than by score?",
        expects=_one("第二阶段学习规划.md", "Reciprocal Rank Fusion"),
        note="英文提问（G2 要求英文 ≥3 条，这是第 1 条）。",
        tags=("自然语言", "英文"),
    ),
    Case(
        id="W19-1",
        category=CATEGORY_ANSWER,
        question="What is the one-command way to restore the development environment on a new machine?",
        expects=_one("学习进度.md", "uv sync"),
        note="英文提问 2：与 W18-6 同一答案，用来对照中英提问的召回差异。",
        tags=("自然语言", "英文"),
    ),
    Case(
        id="W19-2",
        category=CATEGORY_ANSWER,
        question="长文本按句切分时，片段 id 为什么会撞在一起？最后是怎么修的？",
        expects=_one("工程化改造记录.md", "Chunk.id"),
        note="T17：id = source:start:hash，同一行切出多个片段时 start 相同。",
        tags=("自然语言",),
    ),
    # ---------------- 无答案（5） ----------------
    Case(
        id="W18-9",
        category=CATEGORY_NO_ANSWER,
        question="这个仓库用 Jenkins 做持续集成是怎么配置的？",
        negative_keywords=("Jenkins",),
        note="资料里只有 GitHub Actions。",
        tags=("无答案",),
    ),
    Case(
        id="W18-10",
        category=CATEGORY_NO_ANSWER,
        question="Milvus 向量数据库的集群该怎么部署？",
        negative_keywords=("Milvus",),
        note="规划明确不引入独立数据库服务。",
        tags=("无答案",),
    ),
    Case(
        id="W19-3",
        category=CATEGORY_NO_ANSWER,
        question="项目的 Prometheus 监控指标是怎么采集和告警的？",
        negative_keywords=("Prometheus",),
        note="笔记里没有任何运维监控栈。",
        tags=("无答案",),
    ),
    Case(
        id="W19-4",
        category=CATEGORY_NO_ANSWER,
        question="Elasticsearch 的倒排索引在这个项目里怎么配置？",
        negative_keywords=("Elasticsearch",),
        note="检索走 BM25 + 本地向量，没有引入 ES。",
        tags=("无答案",),
    ),
    Case(
        id="W19-5",
        category=CATEGORY_NO_ANSWER,
        question="Grafana 面板上要展示哪些指标？",
        negative_keywords=("Grafana",),
        note="没有可视化监控栈。",
        tags=("无答案",),
    ),
    # ---------------- 跨文档（5） ----------------
    Case(
        id="W19-6",
        category=CATEGORY_CROSS,
        question="评测用例的“四类”是哪一周定的？这套分类在哪份文档里被沿用到第 19 周？",
        expects=(
            Expect("第14周笔记.md", ("四类", "无答案")),
            Expect("第二阶段学习规划.md", ("四类各", "四类（")),
        ),
        note="答案分散在“当周笔记”和“阶段规划”两处。",
        tags=("自然语言",),
    ),
    Case(
        id="W19-7",
        category=CATEGORY_CROSS,
        question="第 17 周测出的双后端冷启动耗时，一份文档给读者看、一份文档给维护者看，分别在哪里？",
        expects=(
            Expect("第17周笔记.md", ("0.33s", "冷启动")),
            Expect("工程化改造记录.md", ("13.08s", "冷启动")),
        ),
        note="同一组数字在两处各记一次：周笔记讲结论，台账讲决策。",
        tags=("自然语言",),
    ),
    Case(
        id="W19-8",
        category=CATEGORY_CROSS,
        question="索引的增量跳过凭什么判断“这个片段已经算过了”？为什么不看文件修改时间？",
        expects=(
            Expect("第17周笔记.md", ("hash 对比", "mtime")),
            Expect("工程化改造记录.md", ("增量跳过", "三件套")),
        ),
        note="机制与理由在周笔记，任务清单与产物名在台账第 17 周小节。",
        tags=("自然语言",),
    ),
    Case(
        id="W19-9",
        category=CATEGORY_CROSS,
        question="仓库瘦身前后的 .git 体积差多少？完整的操作步骤和风险记在哪份文档里？",
        expects=(
            Expect("学习进度.md", ("588 KB", "113 MB")),
            Expect("工程化改造记录.md", ("113 MB", "588 KB")),
        ),
        note="进度文档给结论，台账 T15 给步骤与风险。",
        tags=("自然语言",),
    ),
    Case(
        id="W19-10",
        category=CATEGORY_CROSS,
        question="“junction 逃逸”和“truncated 语义”这两个 bug 是第几周靠 AI 审查发现的？"
        "相关的加固与验证过程记在哪份文档里？",
        expects=(
            Expect("第11周笔记.md", ("junction 逃逸", "truncated 语义")),
            Expect("工程化改造记录.md", ("junction", "符号链接")),
        ),
        note="发现记在周笔记，加固过程在台账 T2 / T6。",
        tags=("自然语言",),
    ),
    # ---------------- 需要多跳（5） ----------------
    Case(
        id="W19-11",
        category=CATEGORY_MULTIHOP,
        question="“有 GPU 就用 GPU”这条常识被哪一周的实验推翻？结论后来被写到哪个文件的什么位置，"
        "以免下次改代码的人再犯同样的假设？",
        expects=(
            Expect("第17周笔记.md", ("被推翻", "fastembed")),
            Expect("工程化改造记录.md", ("embedder.py", "docstring")),
        ),
        note="第一跳找结论，第二跳找“结论落到哪里”。",
        tags=("自然语言",),
    ),
    Case(
        id="W19-12",
        category=CATEGORY_MULTIHOP,
        question="“主环境不装 torch、RAG 环境不装 agents”这条边界，是被哪一次具体的报错逼出来的？",
        expects=(
            Expect("学习进度.md", ("不互相污染",)),
            Expect("工程化改造记录.md", ("ModuleNotFoundError", "re-export")),
        ),
        note="边界写在进度文档，起因是台账 T18 的 ImportError。",
        tags=("自然语言",),
    ),
    Case(
        id="W19-13",
        category=CATEGORY_MULTIHOP,
        question="为什么“合成负载下验证通过”不等于“真实负载下有收益”？涉及哪两个文件里的实验结论？",
        expects=(
            Expect("工程化改造记录.md", ("合成负载", "真实负载")),
            Expect("第13周笔记.md", ("真实规模", "线性增长")),
        ),
        note="台账给教训的抽象版本，周笔记给真实数字。",
        tags=("自然语言",),
    ),
    Case(
        id="W19-14",
        category=CATEGORY_MULTIHOP,
        question="“把测量工具本身也纳入回归测试”是哪一周做的？触发它的那次判据错误是什么？",
        expects=(
            Expect("第14周笔记.md", ("判据自检",)),
            Expect("工程化改造记录.md", ("判据自检", "格式")),
        ),
        note="周笔记记动作，台账 T10 记“格式 vs 内容混装”这个错因。",
        tags=("自然语言",),
    ),
    Case(
        id="W19-15",
        category=CATEGORY_MULTIHOP,
        question="Which experiment showed that a GPU is not always the faster choice, and where was "
        "that conclusion recorded so a future maintainer would see it?",
        expects=(
            Expect("第17周笔记.md", ("fastembed", "CUDA 上下文")),
            Expect("工程化改造记录.md", ("T19", "embedder.py")),
        ),
        note="英文提问 3：多跳 + 跨语言，用来暴露最弱的一类。",
        tags=("自然语言", "英文"),
    ),
]
