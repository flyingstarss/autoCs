"""封闭意图集与槽位定义（架构文档 4 / P1、P2）。

意图集必须封闭且版本化：任何新意图都要在此登记，否则下游路由表无法覆盖。
路由表、槽位规格、工具映射都从这里读取，避免散落在 Prompt 里。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

INTENT_SET_VERSION = "1.0"


@dataclass(frozen=True)
class SlotSpec:
    name: str
    description: str
    required: bool = False
    aliases: tuple[str, ...] = ()
    # 该槽位是否必须由服务端二次核验（如订单号）
    verify: bool = False


@dataclass(frozen=True)
class IntentSpec:
    name: str
    description: str
    domain: str  # perception / decision / execution / growth
    slot_specs: tuple[SlotSpec, ...] = ()
    # 该意图的主执行 Agent（执行域编号），None 表示纯对话
    executor: str | None = None
    # 路由类型：direct_tool（单工具短路）/ agent_chain（串行链）/ plan_execute
    route: str = "agent_chain"
    # 是否触发写操作（需幂等键 + 可能进入人工闸门）
    write: bool = False
    # 需要检索知识库证据
    needs_knowledge: bool = False
    keywords: tuple[str, ...] = field(default=())


def _s(
    name: str,
    desc: str,
    required: bool = False,
    verify: bool = False,
    aliases: tuple[str, ...] = (),
) -> SlotSpec:
    """槽位声明的小工具。aliases 用关键字传参，避免与位置参数混淆。"""
    return SlotSpec(name=name, description=desc, required=required, verify=verify, aliases=tuple(aliases))


ORDER_SLOTS = (
    _s("order_no", "订单号", required=True, verify=True, aliases=("订单号", "单号")),
    _s("sku_id", "商品SKU", aliases=("sku", "商品编码")),
    _s("size", "尺码", aliases=("码数",)),
    _s("color", "颜色"),
    _s("amount", "金额", aliases=("钱", "多少")),
    _s("reason", "退换原因"),
    _s("expect_date", "期望时间"),
    _s("quantity", "数量", aliases=("几件",)),
)

# 只读查询类：订单号可由身份锚定回溯最近订单，不强制要求用户口述
_ORDER_OPTIONAL = _s("order_no", "订单号", required=False, verify=True, aliases=("订单号", "单号"))

INTENTS: dict[str, IntentSpec] = {
    "ORDER_STATUS": IntentSpec(
        name="ORDER_STATUS",
        description="查询订单状态、详情、构成",
        domain="execution",
        slot_specs=(_ORDER_OPTIONAL,),
        executor="E1",
        route="direct_tool",
        keywords=("订单状态", "订单信息", "订单详情", "订单进度", "我的单", "下单了", "查订单", "看订单"),
    ),
    "LOGISTICS": IntentSpec(
        name="LOGISTICS",
        description="物流轨迹、预计送达、异常件",
        domain="execution",
        slot_specs=(_ORDER_OPTIONAL,),
        executor="E2",
        route="direct_tool",
        keywords=("物流", "快递", "到哪了", "什么时候到", "签收", "派送", "轨迹"),
    ),
    "RETURN_APPLY": IntentSpec(
        name="RETURN_APPLY",
        description="退换货、维修、补发申请",
        domain="execution",
        slot_specs=(
            _s("order_no", "订单号", required=True, verify=True),
            _s("reason", "退换原因", required=True, aliases=("原因",)),
            _s("type", "售后类型", aliases=("退还是换",)),
        ),
        executor="E3",
        route="agent_chain",
        write=True,
        needs_knowledge=True,
        keywords=("退货", "退款", "换货", "我要退", "想退", "退掉", "不想要了", "不要了", "维修", "补发"),
    ),
    "SIZE_ADVICE": IntentSpec(
        name="SIZE_ADVICE",
        description="尺码建议、颜色规格可用性",
        domain="execution",
        slot_specs=(_s("size", "尺码"), _s("color", "颜色"), _s("sku_id", "商品SKU")),
        executor="E4",
        route="direct_tool",
        keywords=("尺码", "尺寸", "多大码", "穿多大", "码数", "有黑色", "有白色", "尺码建议", "选多大"),
    ),
    "STOCK_CHECK": IntentSpec(
        name="STOCK_CHECK",
        description="实时库存查询",
        domain="execution",
        slot_specs=(_s("sku_id", "商品SKU"), _s("size", "尺码"), _s("color", "颜色")),
        executor="E4",
        route="direct_tool",
        keywords=("库存", "有货吗", "还有吗", "断货", "现货", "补货"),
    ),
    "PRODUCT_QA": IntentSpec(
        name="PRODUCT_QA",
        description="商品详情、参数、对比、导购",
        domain="execution",
        slot_specs=(_s("sku_id", "商品SKU"),),
        executor="E5",
        route="agent_chain",
        needs_knowledge=True,
        # 注意：不要收录「怎么样」「多大码」这类泛疑问词，否则会抢走澄清场景的置信度
        keywords=("参数", "材质", "尺码表", "区别", "对比", "介绍", "详情",
                  "有哪些商品", "有什么商品", "卖什么", "在售"),
    ),
    "COUPON_USE": IntentSpec(
        name="COUPON_USE",
        description="优惠券使用、发放、活动规则",
        domain="execution",
        slot_specs=(_s("sku_id", "商品SKU"), _s("amount", "优惠金额")),
        executor="E7",
        route="agent_chain",
        write=True,
        needs_knowledge=True,
        keywords=("优惠券", "券", "满减", "折扣", "红包", "怎么用", "领券"),
    ),
    "PAYMENT_INVOICE": IntentSpec(
        name="PAYMENT_INVOICE",
        description="支付状态、发票申请与抬头核验",
        domain="execution",
        slot_specs=(_s("order_no", "订单号", required=True, verify=True),),
        executor="E6",
        route="agent_chain",
        write=True,
        keywords=("支付", "付款", "发票", "开票", "抬头", "报销", "扣款"),
    ),
    "MEMBER_BENEFIT": IntentSpec(
        name="MEMBER_BENEFIT",
        description="会员等级、积分、权益计算",
        domain="execution",
        slot_specs=(_s("sku_id", "商品SKU"),),
        executor="E9",
        route="agent_chain",
        keywords=("会员", "积分", "等级", "权益", "钻石", "金卡", "权益"),
    ),
    "COMPLAINT": IntentSpec(
        name="COMPLAINT",
        description="投诉、差评、监管与舆情风险",
        domain="perception",
        route="handoff",
        write=False,
        keywords=("投诉", "差评", "举报", "曝光", "12315", "消协", "起诉", "维权", "骗人"),
    ),
    "CHITCHAT": IntentSpec(
        name="CHITCHAT",
        description="闲聊与通用问候",
        domain="perception",
        route="agent_chain",
        keywords=("你好", "在吗", "谢谢", "辛苦", "hello"),
    ),
}

# —— 路由表（配置驱动，与 Prompt 解耦，支持热更新）——
ROUTE_TABLE: dict[str, dict[str, Any]] = {
    "ORDER_STATUS": {"route": "direct_tool", "targets": ["E1"], "parallel": False},
    # 物流查询：订单快照与轨迹互不依赖 → 并行扇出（编排器规则「并行优先」）
    "LOGISTICS": {
        "route": "direct_tool",
        "targets": ["E1", "E2"],
        "parallel": True,
        "reason": "订单快照与物流轨迹无依赖，并行查询",
    },
    "SIZE_ADVICE": {"route": "direct_tool", "targets": ["E4"], "parallel": False},
    "STOCK_CHECK": {"route": "direct_tool", "targets": ["E4"], "parallel": False},
    "RETURN_APPLY": {
        "route": "agent_chain",
        "targets": ["Q1", "E3"],
        "parallel": False,
        "reason": "先取政策证据再判定售后资格",
    },
    "COUPON_USE": {
        "route": "agent_chain",
        "targets": ["Q1", "E7"],
        "parallel": False,
        "reason": "活动规则需证据支撑，发放需幂等键",
    },
    "PRODUCT_QA": {"route": "agent_chain", "targets": ["Q1", "E5"], "parallel": False},
    "PAYMENT_INVOICE": {"route": "agent_chain", "targets": ["E6"], "parallel": False},
    "MEMBER_BENEFIT": {"route": "agent_chain", "targets": ["E8", "E9"], "parallel": False},
    "COMPLAINT": {"route": "handoff", "targets": ["Q5"], "parallel": False},
    "CHITCHAT": {"route": "agent_chain", "targets": [], "parallel": False},
}

# 需要与其它意图组合执行（复合请求 → 规划-执行）
COMPOSITE_HINTS: dict[str, tuple[str, ...]] = {
    "RETURN_APPLY": ("COUPON_USE", "LOGISTICS"),
    "COUPON_USE": ("MEMBER_BENEFIT", "PRODUCT_QA"),
    "ORDER_STATUS": ("LOGISTICS",),
}


def get_intent(name: str) -> IntentSpec | None:
    return INTENTS.get(name)


def required_slots(intent: str) -> list[str]:
    spec = INTENTS.get(intent)
    if not spec:
        return []
    return [s.name for s in spec.slot_specs if s.required]


def slot_specs(intent: str) -> list[SlotSpec]:
    spec = INTENTS.get(intent)
    return list(spec.slot_specs) if spec else []


def all_slot_names() -> list[str]:
    names: list[str] = []
    for spec in INTENTS.values():
        names.extend(s.name for s in spec.slot_specs)
    return sorted(set(names))


# 「修改已有的退款诉求」的信号。
# 只判断意图，不解析金额 —— 用户报的数字不可信，金额一律由服务端重算。
_REVISION_RE = re.compile(
    r"(改|调整|修改|更改|更正|重算)[^。！？]{0,8}(金额|退款|退|价)"
    r"|(改成|改为|调整成|调成)\s*\d+"
)


def is_revision_request(text: str) -> bool:
    """用户是否在要求修改已经提出的退款诉求。

    P2 用它豁免「原因槽位防幻觉清理」：用户说「我要改退款金额」时
    通常不会重复讲原因，那是指代上一轮，不是缺槽位。
    """
    return bool(_REVISION_RE.search(str(text or "")))
