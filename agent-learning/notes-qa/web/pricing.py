"""成本估算：把 usage 里的 token 换成"能报给别人听"的金额。

三条边界先说清楚，免得把估算当成账单：

1. **价格是快照，不是实时值**。常量抄自 `第7周笔记.md` 第 5 节（抓取于 2026-09-10 的官方
   价格页），官方随时会调。所以界面上写的是"**估算**"，要精确对账请去看 DeepSeek 平台的
   账单页——这也是第 18 周定的规矩：**定价不进代码逻辑，只进展示**。
2. **峰谷要按"请求发生的时点"判断**，不能按"我什么时候想起它"。规则（北京时间）：
   周一至周五 09:00–12:00、14:00–18:00 为高峰，其余时间与整个周末为低谷，
   低谷价是高峰价的一半。判断必须用**北京时间**而不是本机本地时间：
   这台机器的时区恰好是 +08:00，换台机器就不成立了。
3. **缓存命中价只有未命中的约 1/50**（$0.003 vs $0.15）。所以不能拿 `prompt_tokens` 直接乘
   输入单价——必须把 `cached_tokens` 拆出来单独计价。这里"高估"是有害的：它会让人
   误以为"这个系统很贵"而不敢用，而这恰恰是本阶段最不想看到的结果。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

BEIJING = timezone(timedelta(hours=8))  # 价格按时区划分峰谷，固定用北京时间

# ---- 价格快照（美元 / 每 100 万 token）----
PRICE_SNAPSHOT_DATE = "2026-09-10"
PRICE_SOURCE_URL = "https://api-docs.deepseek.com/quick_start/pricing"
PRICE_OFFPEAK = {
    "in_hit": 0.003,  # 输入：缓存命中
    "in_miss": 0.15,  # 输入：缓存未命中
    "out": 0.6,  # 输出
}
PEAK_FACTOR = 2.0  # 官方口径：低谷价是高峰价的一半

# 仅供直觉参考的粗折算。**不是汇率真相来源**，要精确请看平台账单（账单本身就是人民币）。
USD_TO_CNY = 7.1

# 高峰时段（北京时间，周一至周五）：左闭右开，所以 12:00 整已经算低谷
PEAK_WINDOWS = ((9, 12), (14, 18))


def is_peak(moment: datetime | None = None) -> bool:
    """该时刻是否处于高峰计价时段。传入 naive 时间会被当作北京时间处理。"""
    now = moment or datetime.now(tz=BEIJING)
    if now.tzinfo is None:
        now = now.replace(tzinfo=BEIJING)
    now = now.astimezone(BEIJING)
    if now.weekday() >= 5:  # 周六 / 周日整日低谷
        return False
    return any(start <= now.hour < end for start, end in PEAK_WINDOWS)


def unit_prices(*, peak: bool) -> dict[str, float]:
    """某个时段的单价（美元 / 每 100 万 token）。"""
    factor = PEAK_FACTOR if peak else 1.0
    return {key: value * factor for key, value in PRICE_OFFPEAK.items()}


def estimate_cost(
    prompt_tokens: int | None,
    cached_tokens: int | None,
    completion_tokens: int | None,
    *,
    peak: bool,
) -> dict:
    """按 token 估算一次调用（或一组调用）的花费，返回 `{"usd", "cny"}`。

    `cached_tokens` 会被夹在 `[0, prompt_tokens]` 之间：上游偶发给出不一致的数时，
    宁可少算也不要算出负数或把缓存部分算两遍。
    """
    prompt = max(int(prompt_tokens or 0), 0)
    completion = max(int(completion_tokens or 0), 0)
    cached = min(max(int(cached_tokens or 0), 0), prompt)
    uncached = prompt - cached
    prices = unit_prices(peak=peak)
    usd = (
        uncached * prices["in_miss"]
        + cached * prices["in_hit"]
        + completion * prices["out"]
    ) / 1_000_000
    return {"usd": usd, "cny": usd * USD_TO_CNY}


def format_cost(cost: dict) -> str:
    """金额的展示写法：金额常常小到 1e-4 美元，所以小数位要按量级给。"""
    usd = cost.get("usd") or 0.0
    cny = cost.get("cny") or 0.0
    if usd and usd < 0.0001:
        return "不足 $0.0001（约 0.001 元以内）"
    return f"${usd:.4f}（约 {cny:.3f} 元）"


def describe_pricing(peak: bool | None = None) -> str:
    """一句话说明当前计价时段与价格来源，给界面底部用。"""
    if peak is None:
        peak = is_peak()
    when = "高峰" if peak else "低谷"
    prices = unit_prices(peak=peak)
    return (
        f"当前为{when}计价（输入未命中 ${prices['in_miss']:.2f} / "
        f"缓存命中 ${prices['in_hit']:.3f} / 输出 ${prices['out']:.2f}，每 100 万 token）；"
        f"价格快照 {PRICE_SNAPSHOT_DATE}，以官方页为准"
    )
