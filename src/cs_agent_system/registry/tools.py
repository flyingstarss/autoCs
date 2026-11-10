"""内置工具集（Mock 实现 + 声明式注册）。

只读工具走正常退避重试；四个写工具禁止重试，必须带幂等键。
写操作的「目标实体」由 target_field 指定，用于计算业务锁键与幂等键；
标记 `business_scoped` 的工具还要保证「同一目标同一版本只写一次」。
"""

from __future__ import annotations

from typing import Any

from ..config import settings
from .data import (
    PolicyService,
    calc_final_price,
    find_order,
    find_order_with_tracking,
    find_product_by_text,
    get_member,
    get_product,
    get_tracking,
    list_user_coupons,
    list_user_orders,
)
from .oms import build_oms_client
from .tool_registry import ToolCallContext, ToolRegistry, ToolTimeout


def build_registry() -> ToolRegistry:
    """构造并注册全部工具。"""
    reg = ToolRegistry()
    # 写操作后端：local 写本地 SQLite / http 推给真实 OMS。
    # 在这里实例化一次，所有写工具共用同一个客户端。
    oms = build_oms_client()

    # ——— 只读工具 ———
    @reg.tool(
        "order.query",
        description="按用户身份查询订单详情（订单号须归属该用户）",
        required_slots=(),
        timeout_ms=2000,
        max_retries=2,
    )
    def order_query(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        user_id = ctx.user_id
        order_no = params.get("order_no")
        # 用户本轮明确口述的订单号（可能核验失败）。
        # 它和槽位里的订单号是两个来源：口述的更权威（用户刚说的），
        # 但查不到时必须明确报「查不到」，不能回退到「最近一单」。
        explicit = str(params.get("explicit_order_no") or "")
        target = explicit or order_no
        if target:
            order = find_order(user_id, target)
            if order is None:
                # 订单不存在或不属于该用户 —— 不泄露任何信息
                return {
                    "value": None,
                    "found": False,
                    "verified": False,
                    # 明确告知上层「用户指定了单号，但这一单查不到」，
                    # 上层据此明确回复，而不是回退到列表模式。
                    "requested_order_no": target,
                    "not_found": True,
                }
            return {"value": order.to_dict(), "found": True, "verified": True}
        # 未指定单号：按状态筛选返回订单列表
        statuses = params.get("status_filter") or []
        orders = list_user_orders(user_id, limit=20)
        if statuses:
            wanted = {str(s) for s in statuses}
            orders = [o for o in orders if o.status in wanted]
        return {
            "value": [o.to_dict() for o in orders],
            "found": bool(orders),
            "verified": True,
            "list": True,
            # 筛选条件回传，供话术层说明「按什么维度筛的」
            "status_filter": list(statuses),
            # 用户是不是在问「一批订单」（「我有哪些订单」）。
            # 投影层据此把列表整体渲染出来，而不是只挑最近一单。
            "collection": bool(params.get("collection")),
        }

    @reg.tool(
        "logistics.track",
        description="查询物流轨迹与预计送达",
        timeout_ms=2000,
        max_retries=2,
    )
    def logistics_track(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        user_id = ctx.user_id
        order = find_order(user_id, params.get("order_no"))
        if order is None:
            # 未指定订单号：取最近一单**有物流轨迹**的订单。
            # 盲取最近一单会踩坑——最新单可能是「待付款/已取消」，没有运单号，
            # 用户问物流却得到「查不到」。
            order = find_order_with_tracking(user_id)
            if order is None:
                return {"value": None, "found": False}
        nodes = get_tracking(order.tracking_no)
        if not nodes:
            return {"value": None, "found": False}
        last = nodes[-1]
        return {
            "value": {
                "carrier": order.carrier,
                "tracking_no": order.tracking_no,
                "nodes": nodes,
                "last_node": last["node"],
                "last_time": last["time"],
                "last_desc": last["desc"],
            },
            "found": True,
        }

    @reg.tool(
        "inventory.check",
        description="查询 SKU 实时库存与颜色尺码可用性",
        timeout_ms=2000,
        max_retries=2,
    )
    def inventory_check(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        sku_id = params.get("sku_id") or ""
        product = get_product(sku_id)
        if product is None:
            product = find_product_by_text(str(params.get("text") or ""))
        if product is None:
            return {"value": None, "found": False}
        size = params.get("size")
        color = params.get("color")
        stock = dict(product.stock)
        available = None
        if size:
            available = product.stock.get(size)
        return {
            "value": {
                "sku_id": product.sku_id,
                "title": product.title,
                "price": product.price,
                "sizes": product.sizes,
                "colors": product.colors,
                "stock": stock,
                "size": size,
                "color": color,
                "available": available,
                "in_stock": (available is None or available > 0),
            },
            "found": True,
        }

    @reg.tool(
        "product.detail",
        description="查询商品详情与参数",
        timeout_ms=2000,
        max_retries=2,
    )
    def product_detail(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        product = get_product(params.get("sku_id")) or find_product_by_text(str(params.get("text") or ""))
        if product is None:
            return {"value": None, "found": False}
        return {
            "value": {
                "sku_id": product.sku_id,
                "title": product.title,
                "category": product.category,
                "price": product.price,
                "sizes": product.sizes,
                "colors": product.colors,
                "params": product.params,
            },
            "found": True,
        }

    @reg.tool(
        "payment.query",
        description="查询支付状态与可开票信息",
        timeout_ms=2000,
        max_retries=2,
    )
    def payment_query(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        order = find_order(ctx.user_id, params.get("order_no"))
        if order is None:
            return {"value": None, "found": False}
        # 支付状态读订单上的 paid 事实，不从 status 推断：
        # 已取消的订单可能已付款（退款中），待付款的订单也可能被取消。
        invoice_types = order.invoice_types or list(PolicyService().get("invoice_types", ()) or ())
        return {
            "value": {
                "order_no": order.order_no,
                "paid": order.paid,
                "status": order.status,
                "amount": order.amount,
                "invoice_types": invoice_types,
            },
            "found": True,
        }

    @reg.tool(
        "member.profile",
        description="查询会员等级、积分与身份标签",
        timeout_ms=2000,
        max_retries=2,
    )
    def member_profile(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        return {"value": get_member(ctx.user_id), "found": True}

    @reg.tool(
        "promotion.calc",
        description="服务端计算到手价（唯一权威口径，模型不得自行运算）",
        timeout_ms=2000,
        max_retries=2,
    )
    def promotion_calc(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        result = calc_final_price(
            sku_id=str(params.get("sku_id") or ""),
            amount=params.get("amount"),
            coupon_id=params.get("coupon_id"),
            user_id=ctx.user_id,
        )
        return {"value": result, "found": True}

    @reg.tool(
        "aftersale.status",
        description="按幂等键回查售后申请状态（写超时后使用）",
        timeout_ms=2000,
    )
    def aftersale_status(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        key = params.get("idempotency_key")
        record = reg.lookup_idempotent(key) if key else None
        return {"value": record["data"] if record else None, "executed": record is not None}

    @reg.tool(
        "coupon.status",
        description="按幂等键回查发券状态（写超时后使用）",
        timeout_ms=2000,
    )
    def coupon_status(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        key = params.get("idempotency_key")
        record = reg.lookup_idempotent(key) if key else None
        return {"value": record["data"] if record else None, "executed": record is not None}

    # ——— 写工具（白名单内，必须带幂等键，禁止自动重试）———
    @reg.tool(
        "aftersale.apply",
        kind="write",
        description="发起退换货/维修/补发申请",
        required_slots=("order_no", "reason"),
        timeout_ms=5000,
        target_field="order_no",
        business_scoped=True,
        approval_amount_threshold=settings.write_approval_amount,
    )
    def aftersale_apply(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        order = find_order(ctx.user_id, params.get("order_no"))
        if order is None:
            raise ValueError("订单不存在或不属于当前用户，禁止代他人操作")
        # 演示故障注入：让超时可被测试覆盖
        if params.get("__simulate") == "timeout":
            raise ToolTimeout("simulated timeout")
        policy = PolicyService()
        sla_days = policy.require("refund_sla_days")
        max_comp = policy.require("max_compensation")
        # 赔付不超过策略上限，也不超过订单实付
        amount = min(float(order.amount), float(max_comp))
        # 审批闸门必须在**产生副作用之前**：先提交申请再判金额，
        # 「需审批」就只是事后通知，钱已经退出去了，闸门形同虚设。
        #
        # 闸门按**订单金额**判定，而不是封顶后的赔付额：
        # 赔付上限（max_compensation=200）通常低于审批阈值（500），
        # 若拿 amount 去比，闸门永远不触发，配置就成了摆设。
        # 业务上「高价值订单的售后要人工过一眼」本来就该按订单价值衡量。
        #
        # `__approval_granted` 是管理员批准后重放时由审批接口注入的标记，
        # 用于跳过闸门——否则重放会再次命中 needs_approval，形成死循环。
        gate_amount = max(float(order.amount), amount)
        if gate_amount > float(settings.write_approval_amount) and not params.get(
            "__approval_granted"
        ):
            return {
                "value": {
                    "needs_approval": True,
                    "amount": amount,
                    "gate_amount": gate_amount,
                    "order_no": order.order_no,
                    # 原样带回幂等键与重放参数：批准后必须用**同一把键**重放，
                    # 否则就是第二笔退款。
                    "idempotency_key": params.get("__idempotency_key") or "",
                    "replay_params": {
                        "order_no": order.order_no,
                        "reason": str(params.get("reason") or ""),
                        "type": params.get("type") or "退货退款",
                    },
                },
                "found": True,
            }
        result = oms.apply_aftersale(
            user_id=ctx.user_id,
            order_no=order.order_no,
            kind=params.get("type") or "退货退款",
            reason=str(params.get("reason") or ""),
            refund_amount=amount,
            refund_sla_days=int(sla_days),
            idempotency_key=params.get("__idempotency_key") or "",
        )
        return {"value": result.to_dict()}

    @reg.tool(
        "aftersale.cancel",
        kind="write",
        description="撤销一笔售后/退款申请（金额变更时先撤旧单，再提交新单）",
        required_slots=("order_no", "aftersale_no"),
        timeout_ms=5000,
        # 撤销的业务实体是「哪一张售后单」，不是订单：
        # 同一订单可能有多张历史售后单，按订单加锁会把它们一起锁死。
        target_field="aftersale_no",
        business_scoped=True,
    )
    def aftersale_cancel(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        order = find_order(ctx.user_id, params.get("order_no"))
        if order is None:
            raise ValueError("订单不存在或不属于当前用户，禁止代他人撤销")
        if params.get("__simulate") == "timeout":
            raise ToolTimeout("simulated timeout")
        result = oms.cancel_aftersale(
            user_id=ctx.user_id,
            aftersale_no=str(params.get("aftersale_no") or ""),
            reason=str(params.get("reason") or "用户撤销"),
            idempotency_key=params.get("__idempotency_key") or "",
        )
        return {"value": result.to_dict()}

    @reg.tool(
        "coupon.issue",
        kind="write",
        description="发放优惠券（必须带幂等键）",
        required_slots=("sku_id",),
        timeout_ms=5000,
        target_field="sku_id",
    )
    def coupon_issue(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        if params.get("__simulate") == "timeout":
            raise ToolTimeout("simulated timeout")
        policy = PolicyService()
        # 持有上限用 max_coupon_hold（账号可持有几张券），
        # 不能用 max_stack（那是「单笔订单能叠加几张」，语义完全不同）。
        max_hold = policy.require("max_coupon_hold")
        existing = list_user_coupons(ctx.user_id)
        if len(existing) >= int(max_hold):
            return {"value": {"issued": False, "reason": "已达可领券上限", "max_stack": max_hold}}
        result = oms.issue_coupon(
            user_id=ctx.user_id,
            amount=float(params.get("amount") or 20.0),
            threshold=float(params.get("threshold") or 99.0),
            category=params.get("category") or "服饰",
            idempotency_key=params.get("__idempotency_key") or "",
        )
        return {"value": {"issued": result.ok, **result.payload}}

    @reg.tool(
        "order.action",
        kind="write",
        description="订单写操作（取消/修改地址等）",
        required_slots=("order_no",),
        timeout_ms=5000,
        target_field="order_no",
    )
    def order_action(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        order = find_order(ctx.user_id, params.get("order_no"))
        if order is None:
            raise ValueError("订单不存在或不属于当前用户")
        if params.get("__simulate") == "timeout":
            raise ToolTimeout("simulated timeout")
        action = params.get("action") or "cancel"
        if order.status == "已签收" and action == "cancel":
            raise ValueError("已签收订单不可取消")
        result = oms.update_order(
            user_id=ctx.user_id,
            order_no=order.order_no,
            action=action,
            idempotency_key=params.get("__idempotency_key") or "",
        )
        return {"value": result.to_dict()}

    # —— 可选：把 MCP Server 的工具一并注册进注册中心 ——
    # 进程内工具优先级更高（更快、无外部依赖）；MCP 工具作为补充，
    # 用于接入本仓库之外部署的业务能力。
    if settings.mcp_enabled and settings.tools_via_mcp:
        try:
            reg.register_mcp_tools()
        except Exception:  # noqa: BLE001 —— MCP 不可用不影响进程内工具正常工作
            pass

    return reg
