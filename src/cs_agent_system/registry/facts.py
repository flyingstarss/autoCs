"""事实投影层：把工具的原始返回投影成「话术需要的事实」。

工具层与表现层互不知情：
- 工具层（`tools.py`）只管返回数据，不知道话术模板长什么样；
- 表现层（D3 响应生成）只管插值，不知道工具返回哪些字段；
- **本模块是两者之间唯一的翻译点**，声明「哪些字段进话术、缺字段算不算完整命中」。

投影是纯函数：输入工具返回值与上下文，输出事实字典与命中状态。
同一份规则可被单元与能力共用，避免事实契约在多处重复维护。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.confidence import DETERMINISTIC, EMPTY, ERROR, EXACT, PARTIAL, REJECTED
from ..core.envelope import AgentEnvelope, SessionContext


@dataclass(frozen=True)
class ProjectionContext:
    """投影所需的上下文。

    `ok` / `error` 来自工具调用结果，用于区分「工具报错」「查无此项」
    「命中但字段不全」三种截然不同的情形——它们的处置方式完全不同。

    `tool_data` 是工具返回的完整数据（不只是 `value`）。
    投影层有时需要知道「列表是不是被筛过的」——例如用户明确要「已取消的订单」，
    此时不能沿用「跳过已取消单」的默认规则，否则会筛出空列表。
    """

    ok: bool
    error: str | None
    session: SessionContext
    env: AgentEnvelope
    tool_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactProjection:
    """一次投影的结果。

    - `facts`：进入话术的事实字典，键与 D3 模板占位符一一对应；
    - `outcome`：命中状态（见 `core.confidence`），决定置信度与是否转人工；
    - `extra`：不进话术、但需要回传给调用方的原始数据；
    - `need_handoff`：是否需要人工介入。
    """

    facts: dict[str, Any]
    outcome: str
    need_handoff: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _tool_failed(ctx: ProjectionContext) -> FactProjection:
    """工具报错：统一处置为转人工。"""
    return FactProjection(facts={}, outcome=ERROR, need_handoff=True, extra={"error": ctx.error})


def _primary_from_list(orders: list[Any]) -> dict[str, Any] | None:
    """从订单列表里挑事实来源单：跳过「已取消」。"""
    for order in orders:
        if isinstance(order, dict) and order.get("status") != "已取消":
            return order
    return orders[0] if orders and isinstance(orders[0], dict) else None


def project_order(value: Any, ctx: ProjectionContext) -> FactProjection:
    """订单查询投影。

    工具可能返回三种形态：报错、单笔订单、订单列表（用户没给单号）。
    「查不到」一律不返回任何订单信息——订单不存在与订单不属于本人
    在响应上不可区分，否则会泄露他人订单。
    """
    if not ctx.ok:
        error = ctx.error or ""
        return FactProjection(
            facts={},
            outcome=ERROR,
            need_handoff=True,
            extra={"error": ctx.error, "need_more_info": "order_no" in error},
        )
    if value is None:
        # 用户本轮明确指定了订单号，但这一单查不到（不存在或不属于本人）。
        # 此时必须明确告知「查不到这一单」——
        # 绝不能回退到「最近一单」，那等于回答了另一个问题，
        # 还会把别人的单号泄露给用户。
        requested = str(ctx.session.explicit_order_no or "")
        if requested:
            return FactProjection(
                facts={"not_found_order_no": requested},
                outcome=REJECTED,
                need_handoff=False,
                extra={"found": False, "not_found": True, "requested_order_no": requested},
            )
        return FactProjection(
            facts={},
            outcome=REJECTED,
            need_handoff=True,
            extra={"found": False, "message": "未查询到属于你的订单，请确认订单号"},
        )
    if isinstance(value, list):
        # 未指定订单号：返回列表。
        #
        # 两种诉求，两种渲染：
        # 1. 用户要的是「某类状态的订单」（没发货 / 已取消 / 在途…）
        #    → 工具层已按 status_filter 筛过，这里把全部命中单都渲染出来；
        # 2. 用户只是笼统问「我的订单」
        #    → 以最近一单作为事实来源（跳过已取消：那不是用户要找的），
        #      列表其余内容作为补充信息。
        orders = [o for o in value if isinstance(o, dict)]
        if not orders:
            # 筛选后为空：如实说没有匹配的订单（不是「查不到」）
            return FactProjection(
                facts={"empty_order_list": True},
                outcome=EMPTY,
                need_handoff=False,
                extra={"found": False, "list": True, "orders": [], "empty_list": True},
            )
        primary = _primary_from_list(orders)
        latest = primary or {}
        # 列表是否已被用户诉求筛过（如「已取消的订单」）。
        # 筛过的列表不能再跳过已取消单 —— 那会把用户要的订单筛没。
        filtered = bool(ctx.tool_data.get("status_filter"))
        # 用户是不是明确在问「一批订单」（「我有哪些订单」）。
        # 是的话，即使没有状态筛选，也要把列表整体渲染出来，
        # 不能只挑最近一单 —— 用户问的是清单，回答一单就是答非所问。
        collection = bool(ctx.tool_data.get("collection"))
        if filtered and not latest:
            # 筛选结果全是已取消单（如用户要「已取消的」）：取第一单作为事实来源
            latest = orders[0]
        facts = {
            "order_no": latest.get("order_no"),
            "status": latest.get("status"),
            "amount": latest.get("amount"),
            "title": latest.get("title"),
        }
        # 列表摘要：给 D3 一个可直接渲染的多单文本
        summary = "；".join(
            f"{o.get('order_no')}（{o.get('status')}，{o.get('amount')} 元）" for o in orders
        )
        # 只有「用户要清单」或「按状态筛选」时才产出列表事实。
        # 笼统问「我的订单」不是清单诉求，保持单单渲染（沿用事实来源单），
        # 否则每次查询都会变成一长串列表，反而更难读。
        if filtered or collection:
            facts["order_count"] = len(orders)
            facts["orders_summary"] = summary
            # 区分「全部订单」与「按条件筛出的订单」，话术措辞不同
            facts["list_all_orders"] = collection and not filtered
        return FactProjection(
            facts=facts,
            outcome=PARTIAL,
            extra={
                "found": True,
                "list": True,
                "orders": orders,
                "order_count": len(orders),
                "orders_summary": summary,
                # 全部命中单的状态清单，供话术逐单说明
                "order_lines": [
                    f"{o.get('order_no')}｜{o.get('title')}｜{o.get('status')}｜{o.get('amount')} 元"
                    for o in orders
                ],
            },
        )
    facts = {
        "order_no": value.get("order_no"),
        "status": value.get("status"),
        "amount": value.get("amount"),
        "title": value.get("title"),
    }
    complete = all(facts[k] not in (None, "") for k in ("order_no", "status", "amount", "title"))
    return FactProjection(
        facts=facts,
        outcome=EXACT if complete else PARTIAL,
        extra={"found": True, "order": value},
    )


def project_logistics(value: Any, ctx: ProjectionContext) -> FactProjection:
    """物流轨迹投影。"""
    if not ctx.ok:
        return _tool_failed(ctx)
    if not value:
        return FactProjection(facts={}, outcome=EMPTY, need_handoff=True, extra={"found": False})
    facts = {
        "carrier": value.get("carrier"),
        "last_node": value.get("last_node"),
        "last_time": value.get("last_time"),
        "last_desc": value.get("last_desc"),
    }
    complete = all(v not in (None, "") for v in facts.values())
    return FactProjection(
        facts=facts,
        outcome=EXACT if complete else PARTIAL,
        extra={"found": True, "logistics": value},
    )


def project_inventory(value: Any, ctx: ProjectionContext) -> FactProjection:
    """库存与规格投影。

    `available` 为 None（接口没返回）与 0（确定缺货）必须区分：
    前者不能对用户说「缺货」，只能说以页面为准。
    """
    if not ctx.ok:
        return _tool_failed(ctx)
    if not value:
        return FactProjection(facts={}, outcome=EMPTY, extra={"found": False})
    available = value.get("available")
    if available is None:
        # 用户没指定尺码（「库存是多少」）：按尺码逐项报库存。
        # 只回「以页面显示为准」等于没回答 —— 工具明明已经拿到了每个尺码的库存。
        stock_map = {k: v for k, v in (value.get("stock") or {}).items()}
        hot = [f"{size} 有货（剩 {count} 件）" for size, count in stock_map.items() if count > 0]
        cold = [f"{size} 暂时缺货" for size, count in stock_map.items() if count <= 0]
        pieces = ["、".join(hot)] if hot else []
        if cold:
            pieces.append("、".join(cold))
        if pieces:
            stock_desc = "；".join(pieces)
            outcome = EXACT
        else:
            stock_desc = "各尺码库存以页面显示为准"
            outcome = PARTIAL
    elif available > 0:
        stock_desc = f"有货（剩余 {available} 件）"
        outcome = EXACT
    else:
        stock_desc = "暂时缺货"
        outcome = EXACT
    facts = {
        "title": value.get("title"),
        "size": value.get("size") or "标准",
        "price": value.get("price"),
        "stock_desc": stock_desc,
    }
    return FactProjection(
        facts=facts,
        outcome=outcome,
        extra={"found": True, "inventory": value},
    )


# 请求句式：用于从用户原话里剥离出「想买的那个东西」。
# 只服务于「查无此商品」这一句明确回答，不做实体识别，也不参与检索。
_REQUEST_PREFIXES = ("我想买", "我要买", "想买", "帮我看看", "我要", "有没有", "你们有", "有卖")
_REQUEST_CHARS = ("推荐", "一下", "你", "我", "有", "什么", "吗", "呢", "啊", "呀", "的", "款", "个", "件", "条")
# 泛指词：剥完只剩这些说明用户问的是「有哪些商品」，不是某个具体商品
_GENERIC_ITEMS = ("商品", "产品", "东西", "哪些", "什么", "所有", "全部")


def _requested_item(text: str) -> str:
    """从用户原话里取出想买的商品名（如「我想买帽子」→「帽子」）。

    取不到或只剩泛指词时返回空串 —— 此时不追加「暂时没有 X」，
    否则会把「你们有哪些商品」答成「暂时没有『哪些商品』」。
    """
    s = (text or "").strip()
    if not s:
        return ""
    # 只取第一个分句：后面通常是对系统的要求（「你有什么推荐吗」）
    for sep in ("，", ",", "。", "！", "!", "？", "?", "\n"):
        s = s.split(sep)[0]
    for prefix in _REQUEST_PREFIXES:
        s = s.replace(prefix, "")
    for word in _REQUEST_CHARS:
        s = s.replace(word, "")
    s = s.strip()
    if not s or len(s) > 10:
        return ""
    if any(g in s for g in _GENERIC_ITEMS):
        return ""
    return s


def project_product(value: Any, ctx: ProjectionContext) -> FactProjection:
    """商品咨询投影。

    定位不到具体单品时回落商品目录，回答「你们有哪些商品」这类导购问题。
    目录是有效答案而非失败，故为部分命中。
    """
    if not ctx.ok:
        return _tool_failed(ctx)
    if not value:
        from .data import list_products

        # 只列前 8 款：整库拼接会远超渠道 500 字上限，最终被硬截断成
        # 「…（SKU-70203，329 元，25…」这种半句话，观感极差。
        # 先给少量可读的候选，用户问具体款再走单品查询。
        products = list_products()
        head = products[:8]
        catalog = "；".join(
            f"{p.title}（{p.sku_id}，{p.price:g} 元，尺码 {'/'.join(p.sizes)}）"
            for p in head
        )
        if len(products) > len(head):
            catalog += f" 等共 {len(products)} 款"
        facts: dict[str, Any] = {"catalog": catalog or "暂无在售商品"}
        # 用户问的是某个具体商品但库里没有（「我想买帽子」）：
        # 必须先明确说「暂时没有帽子」，再列在售商品。
        # 只甩一份目录会让用户误以为系统没听懂问题。
        requested = _requested_item(str((ctx.env.payload or {}).get("text") or ""))
        if requested:
            facts["not_found_item"] = requested
        return FactProjection(
            facts=facts,
            outcome=PARTIAL,
            extra={"found": False},
        )
    params_map = value.get("params") or {}
    params_desc = "，".join(f"{k} {v}" for k, v in params_map.items()) if params_map else "暂无更多参数"
    facts = {
        "title": value.get("title"),
        "params_desc": params_desc,
        "price": value.get("price"),
    }
    complete = facts["title"] not in (None, "") and facts["price"] not in (None, "")
    return FactProjection(
        facts=facts,
        outcome=EXACT if complete else PARTIAL,
        extra={"found": True, "product": value},
    )


def project_payment(value: Any, ctx: ProjectionContext) -> FactProjection:
    """支付与开票投影。布尔支付状态在此翻成中文，不交给表现层判断。"""
    if not ctx.ok:
        return _tool_failed(ctx)
    if not value:
        return FactProjection(facts={}, outcome=EMPTY, need_handoff=True, extra={"found": False})
    invoice_types = "、".join(value.get("invoice_types") or [])
    facts = {
        "order_no": value.get("order_no"),
        "paid_status": "已支付" if value.get("paid") else "未支付",
        "invoice_types": invoice_types,
    }
    complete = facts["order_no"] not in (None, "") and value.get("paid") is not None and bool(invoice_types)
    return FactProjection(
        facts=facts,
        outcome=EXACT if complete else PARTIAL,
        extra={"found": True, "payment": value},
    )


def project_member(value: Any, ctx: ProjectionContext) -> FactProjection:
    """会员身份投影。"""
    if not ctx.ok:
        # 会员信息查不到不影响主流程，无需惊动人工。
        return FactProjection(facts={}, outcome=ERROR, extra={"error": ctx.error})
    data = value or {}
    facts = {"level": data.get("level"), "points": data.get("points")}
    complete = facts["level"] not in (None, "") and facts["points"] is not None
    return FactProjection(
        facts=facts,
        outcome=EXACT if complete else PARTIAL,
        extra={"member": data},
    )


# 能力 → 投影函数的登记表，供能力层按编号查找。
PROJECTIONS = {
    "E1": project_order,
    "E2": project_logistics,
    "E4": project_inventory,
    "E5": project_product,
    "E6": project_payment,
    "E8": project_member,
}

# 能力 → 任务档位（置信度上限）。查单号精确匹配最可靠，其余为确定性事实查询。
PROJECTION_TIERS = {
    "E1": "exact_match",
    "E2": DETERMINISTIC,
    "E4": DETERMINISTIC,
    "E5": DETERMINISTIC,
    "E6": DETERMINISTIC,
    "E8": DETERMINISTIC,
}


__all__ = [
    "FactProjection",
    "ProjectionContext",
    "PROJECTIONS",
    "PROJECTION_TIERS",
    "project_order",
    "project_logistics",
    "project_inventory",
    "project_product",
    "project_payment",
    "project_member",
]
