"""兜底话术（架构文档 12.1）。

客服系统不能因为模型或接口抖动就静默失败。三个可用的兜底环节分处不同模块，
这里只做**模板兜底**，避免同一件事有两套实现：

    L1 草稿保留 —— `Orchestrator._finalize`：有草稿就照用，并标记 degraded
    L2 模板兜底 —— 本模块：按意图给固定话术
    L4 转人工   —— `Orchestrator._handoff` + Q5：带上下文转坐席

此前还声明过 L3「规则应答」和一个 L1/L3/L4 分支，但它们**永远走不到**：
L2 的意图模板已覆盖全部意图（未知意图回落 CHITCHAT），且调用方固定传 L2。
已删除这些死分支 —— 保留「写了但从不执行」的能力声明，比没有更糟。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# —— L2：意图模板兜底 ——
INTENT_FALLBACKS: dict[str, str] = {
    "ORDER_STATUS": "您的订单信息暂时查询较慢，稍后为您回复，或点击「订单中心」自助查看。",
    "LOGISTICS": "物流信息正在同步中，稍后为您更新最新进度。",
    "RETURN_APPLY": "退换货服务暂时繁忙，稍后为您处理，请先准备好订单号。",
    "COUPON_USE": "优惠券服务暂时繁忙，稍后为您发放可用优惠券。",
    "PRODUCT_QA": "商品详情正在加载中，稍后为您补充完整参数。",
    "SIZE_ADVICE": "库存信息正在同步中，稍后为您确认可售规格。",
    "STOCK_CHECK": "库存信息正在同步中，稍后为您确认。",
    "PAYMENT_INVOICE": "支付与开票服务暂时繁忙，稍后为您处理。",
    "MEMBER_BENEFIT": "会员权益正在核算中，稍后为您确认。",
    "COMPLAINT": "非常抱歉，已为您转接专属客服跟进处理。",
    "CHITCHAT": "您好～请问有什么可以帮您？",
}


@dataclass
class DegradeResult:
    level: str  # 目前只有 L2
    text: str
    need_handoff: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


def degrade(*, intent: str, facts: dict[str, Any] | None = None, reason: str = "") -> DegradeResult:
    """按意图生成模板兜底话术。未知意图回落到 CHITCHAT。"""
    template = INTENT_FALLBACKS.get(intent, INTENT_FALLBACKS["CHITCHAT"])
    return DegradeResult(level="L2", text=template, payload={"reason": reason, "facts": facts or {}})
