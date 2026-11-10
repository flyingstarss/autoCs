"""MCP Server：把客服系统的业务能力暴露为标准 MCP 工具。

为什么要有这一层：
1. **能力与编排解耦** —— 业务查询能力一旦被 MCP 化，任何 MCP 客户端
   （Claude Desktop / 其他 Agent / 本项目的 MCP 客户端）都能复用，不必走 HTTP API；
2. **进程隔离可选** —— 业务数据访问可以独立部署、独立扩容；
3. **协议标准化** —— 工具描述、参数 schema、错误语义由 MCP 规范统一，
   不再是本项目自定义的私有 JSON 约定。

数据来源：D 盘 SQLite（`data/cs_agent.db`），由 `_seed_db.py` 播种。
本模块只做「协议适配」，不含业务逻辑——业务口径仍由 `registry.data` 唯一持有。

启动方式：
    # stdio（供 MCP 客户端拉起，最常用）
    python -m cs_agent_system.mcp.server

    # streamable-http（供本项目 MCP 客户端连接）
    python -m cs_agent_system.mcp.server --http
"""

from __future__ import annotations

import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..config import settings
from ..registry import data

mcp = FastMCP(
    "cs-agent-business-data",
    instructions=(
        "电商客服系统的业务数据查询工具。所有数据来自本地 SQLite 真实数据库。\n"
        "重要约定：\n"
        "1. 订单查询以 user_id 为主键，传他人订单号会返回 not_found（防越权）；\n"
        "2. 退款时效、赔付上限、优惠叠加等承诺值必须取自 policy.* 工具，不得自行推算；\n"
        "3. 到手价必须用 pricing.calc 计算，不得自行加减乘除。"
    ),
)


def _dump(value: Any) -> str:
    """统一 JSON 序列化（ensure_ascii=False 让中文可读）。"""
    return json.dumps(value, ensure_ascii=False, default=str)


@mcp.tool()
def order_query(user_id: str, order_no: str | None = None) -> str:
    """查询订单。传 order_no 返回单笔；不传则返回该用户最近订单列表。

    Args:
        user_id: 用户标识（必填，订单归属以此为准）。
        order_no: 订单号；为空时按创建时间倒序返回该用户订单列表。
    """
    if not user_id:
        return _dump({"found": False, "error": "缺少 user_id"})
    if order_no:
        order = data.find_order(user_id, order_no)
        if order is None:
            # 订单不存在与不属于本人不可区分，避免探测他人订单
            return _dump({"found": False, "error": "未查询到属于你的订单"})
        return _dump({"found": True, "order": order.to_dict()})
    orders = data.list_user_orders(user_id)
    # 标出事实来源单：跳过已取消，否则调用方会拿一单已取消的订单当答案
    primary = data.pick_primary_order(orders)
    return _dump(
        {
            "found": bool(orders),
            "count": len(orders),
            "primary_order_no": primary.order_no if primary else None,
            "orders": [o.to_dict() for o in orders],
        }
    )


@mcp.tool()
def logistics_track(user_id: str, order_no: str | None = None) -> str:
    """查询物流轨迹。

    Args:
        user_id: 用户标识。
        order_no: 订单号；为空时返回该用户最近一单的物流轨迹。
    """
    if not user_id:
        return _dump({"found": False, "error": "缺少 user_id"})
    order = None
    if order_no:
        order = data.find_order(user_id, order_no)
    else:
        # 取最近一单有物流轨迹的订单（最新单可能是待付款/已取消，没有运单号）
        order = data.find_order_with_tracking(user_id)
    if order is None:
        return _dump({"found": False, "error": "未查询到属于你的订单"})
    trackings = data.get_tracking(order.tracking_no)
    return _dump(
        {
            "found": bool(trackings),
            "carrier": order.carrier,
            "tracking_no": order.tracking_no,
            "trackings": trackings,
        }
    )


@mcp.tool()
def product_detail(sku_id: str | None = None, text: str | None = None) -> str:
    """查询商品详情与参数。传 sku_id 精确查；传 text 按标题/别名模糊定位。

    Args:
        sku_id: 商品编码。
        text: 自由描述（如「卫衣」「跑鞋」），用于模糊定位。
    """
    product = None
    if sku_id:
        product = data.get_product(sku_id)
    if product is None and text:
        product = data.find_product_by_text(text)
    if product is None:
        catalog = [
            {"sku_id": p.sku_id, "title": p.title, "price": p.price, "sizes": p.sizes}
            for p in data.list_products()
        ]
        return _dump({"found": False, "catalog": catalog})
    return _dump(
        {
            "found": True,
            "sku_id": product.sku_id,
            "title": product.title,
            "category": product.category,
            "price": product.price,
            "sizes": product.sizes,
            "colors": product.colors,
            "stock": product.stock,
            "params": product.params,
        }
    )


@mcp.tool()
def inventory_check(sku_id: str | None = None, size: str | None = None, color: str | None = None) -> str:
    """查询库存与规格。size 为空时返回全部尺码库存。

    Args:
        sku_id: 商品编码。
        size: 尺码。
        color: 颜色。
    """
    product = data.get_product(sku_id)
    if product is None:
        return _dump({"found": False, "error": "商品不存在"})
    stock = product.stock
    if size:
        available = stock.get(size)
        if available is None:
            return _dump(
                {
                    "found": True,
                    "sku_id": product.sku_id,
                    "title": product.title,
                    "size": size,
                    "available": None,
                    "stock_desc": "各尺码库存以页面显示为准",
                }
            )
        desc = f"有货（剩余 {available} 件）" if available > 0 else "暂时缺货"
        return _dump(
            {
                "found": True,
                "sku_id": product.sku_id,
                "title": product.title,
                "size": size,
                "color": color,
                "available": available,
                "price": product.price,
                "stock_desc": desc,
            }
        )
    return _dump(
        {
            "found": True,
            "sku_id": product.sku_id,
            "title": product.title,
            "price": product.price,
            "stock": stock,
            "sizes": product.sizes,
            "colors": product.colors,
        }
    )


@mcp.tool()
def member_profile(user_id: str) -> str:
    """查询会员身份（等级、积分、成长值、标签）。

    Args:
        user_id: 用户标识。
    """
    if not user_id:
        return _dump({"error": "缺少 user_id"})
    return _dump({"member": data.get_member(user_id)})


@mcp.tool()
def payment_query(user_id: str, order_no: str) -> str:
    """查询订单支付状态与可开票类型。

    Args:
        user_id: 用户标识。
        order_no: 订单号。
    """
    order = data.find_order(user_id, order_no)
    if order is None:
        return _dump({"found": False, "error": "未查询到属于你的订单"})
    policy = data.PolicyService()
    invoice_types = order.invoice_types or list(policy.get("invoice_types", ()) or ())
    return _dump(
        {
            "found": True,
            "order_no": order.order_no,
            # 支付状态读订单上的 paid 事实，不从 status 推断
            "paid": order.paid,
            "status": order.status,
            "amount": order.amount,
            "invoice_types": invoice_types,
        }
    )


@mcp.tool()
def coupon_list(user_id: str) -> str:
    """列出用户当前可用（未使用）优惠券。

    Args:
        user_id: 用户标识。
    """
    if not user_id:
        return _dump({"error": "缺少 user_id"})
    return _dump({"coupons": data.list_user_coupons(user_id)})


@mcp.tool()
def pricing_calc(user_id: str, sku_id: str, coupon_id: str | None = None, amount: float | None = None) -> str:
    """计算到手价（唯一权威口径，模型不得自行运算）。

    Args:
        user_id: 用户标识（决定会员折扣与券归属）。
        sku_id: 商品编码。
        coupon_id: 优惠券编号；为空则不叠加券。
        amount: 指定商品金额；为空取商品原价。
    """
    result = data.calc_final_price(
        sku_id=sku_id,
        amount=amount,
        coupon_id=coupon_id,
        user_id=user_id,
    )
    return _dump(result)


@mcp.tool()
def policy_get(key: str | None = None) -> str:
    """读取策略配置（对外承诺值的唯一来源）。

    Args:
        key: 策略键；为空则返回全部策略。
    """
    service = data.PolicyService()
    if key:
        value = service.get(key)
        if value is None:
            return _dump({"found": False, "error": f"策略项 {key} 不存在"})
        return _dump({"found": True, "key": key, "value": value})
    return _dump({"found": True, "policies": data.POLICY})


@mcp.tool()
def data_source_status() -> str:
    """数据源诊断：当前走 DB 还是内存兜底、各表行数、数据库文件路径。"""
    return _dump(data.data_source_info())


def main() -> int:
    """启动 MCP Server。默认 stdio，`--http` 走 streamable-http。"""
    if not settings.mcp_server_enabled:
        # MCP_SERVER_ENABLED=0 时拒绝启动，而不是照常起来再让配置失效
        print("MCP_SERVER_ENABLED=0，MCP Server 未启动。", file=sys.stderr)
        return 1
    use_http = "--http" in sys.argv
    if use_http:
        import uvicorn

        # FastMCP 的 streamable-http 应用挂载点
        app = mcp.streamable_http_app()
        uvicorn.run(app, host=settings.mcp_server_host, port=settings.mcp_server_port)
    else:
        mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
