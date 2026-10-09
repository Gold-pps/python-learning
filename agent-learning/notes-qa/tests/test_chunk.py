"""`Chunk` 的序列化与「参考文献段」判定（`rag/chunk.py`）。

第 22 周加 `is_reference` 的动机：论文的参考文献列表页是稠密书目信息，
对很多查询都"语义上很像"，实测两次被当成候选（一次模型主动拒绝、一次进了四段）。
判定放在**建索引时**（跟着 chunk 落盘），检索层才有依据默认排除它。
"""

from __future__ import annotations

import json

from rag.chunk import Chunk, looks_like_reference_list

# 一段真实的参考文献页长这样（取自第 22 周实测里被误当候选的那一页）
BIBLIOGRAPHY = (
    "[1] CAI W，LIU F，ZHOU X N，et al. Fine energy consumption allowance of workpieces"
    " in the mechanical manufacturing industry[J]. Energy，2016，114：623-633. "
    "[2] CHEN X Z，LI C B，TANG Y，et al. Energy efficient cutting parameter optimization[J]."
    " Frontiers of Mechanical Engineering，2021，16(2)：221-248. "
    "[3] DIAZ N，NINOMIYA K，NOBLE J，et al. Environmental impact characterization of milling[J]."
    " Procedia CIRP，2012，1：518-523. "
    "[4] HE Y，LIU F，WU T，et al. Analysis and estimation of energy consumption for numerical"
    " control machining[J]. Proceedings of the Institution of Mechanical Engineers，2012."
)


def test_looks_like_reference_list_flags_dense_bibliography():
    assert looks_like_reference_list(BIBLIOGRAPHY)
    # 带明确标题的（正文里出现"参考文献"且书目特征够多）
    assert looks_like_reference_list("参考文献\n" + BIBLIOGRAPHY)


def test_looks_like_reference_list_ignores_body_text_with_many_inline_citations():
    """**这是回归测试，来自一次真实的误判事故。**

    第一版判据把「`[数字]` 紧跟字母/汉字」当命中特征 + 密度阈值，于是中文综述的正文页
    （"XING Q Q 等[18]提出…"这种句子满篇都是）被整片标成参考文献：2501 片段命中 664 条，
    单篇最多 76 条，散布在正文各页。下面这段就是当时被误标的真实正文。
    """
    body = (
        "作为车削过程中的刀具磨损特征，采用重力搜索算法和最小二乘支持向量机结合的方法"
        "来判断刀具磨损状态。XING Q Q 等[18]提出了一种铣刀磨损监测方法，将切向、径向"
        "和轴向切削力系数融合以指示刀具磨损，根据多通道切削力系数估计渐进式刀具磨损值。"
        "LIU T S 等[19]提出了一种基于切屑厚度重建和切削力信号的不对称微铣刀具磨损估计的"
        "齿形监测方法。KAYA B 等[48]将三轴切削力、扭矩、切削条件与实测后刀面磨损数据结合，"
        "建立人工神经网络模型，再根据实测数据验证 ANN 模型。"
    )

    assert not looks_like_reference_list(body)


def test_looks_like_reference_list_flags_english_bibliography():
    """英文（MDPI 样式）的文献表没有 `[J]`，DOI 写作 `https://doi.org/`——也要能认出来。

    这是第二版判据补的：只看中文特征时，英文论文的文献页一条都标不出来。
    """
    english = (
        "35. Liu, C.; Huang, Z.; Huang, S.; He, Y.; Yang, Z.; Tuo, J. Surface Roughness"
        " Prediction in CNC Milling. Materials 2022, 20, 2990–2998."
        " https://doi.org/10.1016/j.jmrt.2022.08.075. [CrossRef]"
        " 36. Çelik, Y.H.; Fidan, Ş. Analysis of Cutting Parameters on Tool Wear."
        " Metals 2021, 11, 1234. [CrossRef]"
        " 37. Moldovan, D.; Slowik, A. Energy consumption prediction. Expert Syst. Appl."
        " 2022, 204, 117555. https://doi.org/10.1016/j.eswa.2022.117555. [CrossRef]"
    )

    assert looks_like_reference_list(english)


def test_looks_like_reference_list_tolerates_spacing_in_et_al():
    """PDF 抽取常把 "et al." 变成 "et al ."（点号前多一个空格）。

    严格要求点号紧跟会漏掉整批英文文献：第 22 周实测有一页 10 个片段一个都没标上。
    """
    text = (
        "［105］ZHU Z X, XI X L, XU X, et al . Digital twin-driven manufacturing"
        " systems[J]. 2020, 26(1): 1-17. "
        "［106］UEBE H, et al . Approach for the development of digital twins[J]. 2019, 79: 1-6. "
        "［107］TAO F, ZHANG M, et al . Digital twin shop-floor[J]. 2017, 5: 1-10. "
        "［108］LIU Q, ZHANG H, et al . Digital twin-driven product design[J]. 2019, 8: 1-12."
    )

    assert looks_like_reference_list(text)


def test_create_marks_reference_and_json_round_trip_keeps_it():
    chunk = Chunk.create(
        source_path="刀具磨损监测/某篇.pdf",
        doc_type="pdf",
        locator="第 10 页",
        start=10,
        end=10,
        text=BIBLIOGRAPHY,
    )

    assert chunk.is_reference is True

    restored = Chunk.from_json(chunk.to_json())
    assert restored == chunk  # 冻结 dataclass：整条相等即可
    assert restored.is_reference is True


def test_from_json_tolerates_old_index_without_the_field():
    """老索引（第 22 周之前建的）没有 `is_reference` 字段也要读得进来——默认 False。

    代价是"老索引里这个标记全为假"，所以加字段后**必须重建一次索引**才生效，
    这条测试是把这个前提钉住，免得以后有人以为读旧索引也能有标记。
    """
    payload = {
        "id": "a.md:1:abc",
        "source_path": "a.md",
        "doc_type": "md",
        "locator": "1-3 行",
        "start": 1,
        "end": 3,
        "text": "正文",
        "hash": "abc",
    }

    chunk = Chunk.from_json(json.dumps(payload, ensure_ascii=False))

    assert chunk.is_reference is False
