"""业务数据与策略配置服务。

数据源分层（可切换，自动降级）：
1. **SQLite（默认）**：真实数据库，文件落在 D 盘 `data/cs_agent.db`，
   由 `_seed_db.py` 播种；
2. **内存字典（兜底）**：DB 不可用（未播种 / 文件损坏 / 权限问题）时自动回落，
   保证「数据库挂了客服还能答话」，而不是整条链路崩掉。

切换开关：`DATA_SOURCE=db|mock`（见 `config.Settings.data_source`）。

三条设计红线在这里落地：
1. 订单等强个人数据以「用户身份」为主键查询 —— `find_order(user_id, order_no)`，
   用户口述的订单号若不属于该用户，直接返回 None，交给上层做二次核验；
2. 退款时效、赔付上限、优惠叠加等承诺值一律从策略服务读取 ——
   `PolicyService.get(key)`，任何处理单元都不得自行推算；
3. **计价只有一个权威口径** —— `calc_final_price`，模型不得参与加减乘除。
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..config import settings
from . import db
from .db import dumps, loads

# —— 策略配置：所有对外承诺值的唯一来源 ——
POLICY: dict[str, Any] = {
    "refund_sla_days": 3,
    "return_window_days": 15,
    "max_compensation": 200.0,
    "free_shipping_threshold": 99.0,
    "max_stack": 2,
    # 账号可持有的可用券总数上限（与 max_stack 是两件事，不可混用）
    "max_coupon_hold": 5,
    "stackable_categories": ("服饰", "家居"),
    "member_discount": {"普通会员": 1.0, "银卡": 0.98, "金卡": 0.95, "钻石": 0.9},
    "point_rule": {"per_yuan": 1, "point_value": 0.01},
    "delivery_promise_days": 3,
    "invoice_types": ("电子普票", "电子专票"),
}


class PolicyService:
    """策略配置服务。所有承诺值必须经此读取。

    数据源优先级：SQLite `policies` 表 → 内存 POLICY 兜底。
    内存副本是「数据库未播种时的最小可用集」，保证系统冷启动不报错。
    """

    def get(self, key: str, default: Any = None) -> Any:
        if _use_db():
            try:
                row = db.query_one("SELECT value FROM policies WHERE key = ?", (key,))
                if row is not None:
                    return loads(row["value"], default)
            except sqlite3.Error:
                pass
        return POLICY.get(key, default)

    def require(self, key: str) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(f"策略项 {key} 不存在，禁止使用默认值兜底")
        return value

    def snapshot(self, keys: tuple[str, ...] | list[str]) -> dict[str, Any]:
        return {k: self.get(k) for k in keys}


_MISSING = object()


def _use_db() -> bool:
    """是否走 SQLite 数据源。DB 不可用时自动回落内存字典。"""
    return db.is_available()


def _db_ready() -> bool:
    """DB 可用且已播种（products 表非空）。未播种则回落内存，避免查空库返回空话术。"""
    if not _use_db():
        return False
    try:
        row = db.query_one("SELECT COUNT(*) AS c FROM products")
        return bool(row) and int(row["c"]) > 0
    except sqlite3.Error:
        return False


# —— 业务数据 ——
@dataclass
class Order:
    order_no: str
    user_id: str
    sku_id: str
    title: str
    amount: float
    status: str = "已发货"
    created_at: str = "2026-10-28 10:22"
    ship_time: str = "2026-10-29 09:10"
    carrier: str = "顺丰速运"
    tracking_no: str = "SF1234567890"
    # 支付状态是独立于订单状态的事实：已取消的订单可能已付款（退款中），
    # 未付款的订单也可能已取消。不能用 status 推断 paid。
    paid: bool = True
    invoice_types: list[str] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_no": self.order_no,
            "sku_id": self.sku_id,
            "title": self.title,
            "amount": self.amount,
            "status": self.status,
            "created_at": self.created_at,
            "ship_time": self.ship_time,
            "carrier": self.carrier,
            "tracking_no": self.tracking_no,
            "paid": self.paid,
            "invoice_types": list(self.invoice_types),
            "items": self.items,
        }


@dataclass
class Product:
    sku_id: str
    title: str
    category: str
    price: float
    sizes: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    stock: dict[str, int] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    # 口语化别名，用于从用户自由描述中定位商品
    aliases: tuple[str, ...] = ()


ORDERS: dict[str, Order] = {
    "SO202610280001": Order(
        order_no="SO202610280001",
        user_id="u_1001",
        sku_id="SKU-10086",
        title="纯棉圆领卫衣·雾霾蓝",
        amount=189.0,
        status="已发货",
        items=[{"sku_id": "SKU-10086", "size": "M", "color": "雾霾蓝", "quantity": 1}],
    ),
    "SO202610180233": Order(
        order_no="SO202610180233",
        user_id="u_1001",
        sku_id="SKU-20431",
        title="轻氧跑鞋·碳晶黑",
        amount=459.0,
        status="已签收",
        items=[{"sku_id": "SKU-20431", "size": "42", "color": "碳晶黑", "quantity": 1}],
    ),
    "SO202611010077": Order(
        order_no="SO202611010077",
        user_id="u_2002",
        sku_id="SKU-10086",
        title="纯棉圆领卫衣·雾霾蓝",
        amount=189.0,
        status="待付款",
        paid=False,
        items=[{"sku_id": "SKU-10086", "size": "L", "color": "米白", "quantity": 2}],
    ),
}

PRODUCTS: dict[str, Product] = {
    "SKU-10086": Product(
        sku_id="SKU-10086",
        title="纯棉圆领卫衣",
        category="服饰",
        price=189.0,
        sizes=["S", "M", "L", "XL"],
        colors=["雾霾蓝", "米白", "黑色"],
        stock={"S": 12, "M": 8, "L": 0, "XL": 3},
        params={"材质": "100% 精梳棉", "克重": "320g", "版型": "宽松", "适用季节": "春秋"},
        aliases=("卫衣", "上衣", "圆领", "外套"),
    ),
    "SKU-20431": Product(
        sku_id="SKU-20431",
        title="轻氧跑鞋",
        category="运动",
        price=459.0,
        sizes=["40", "41", "42", "43"],
        colors=["碳晶黑", "月白"],
        stock={"40": 5, "41": 2, "42": 6, "43": 0},
        params={"鞋面": "飞织网布", "中底": "超临界发泡", "重量": "单只 235g"},
        aliases=("跑鞋", "鞋", "运动鞋"),
    ),
    "SKU-30021": Product(
        sku_id="SKU-30021",
        title="陶瓷马克杯",
        category="家居",
        price=69.0,
        sizes=["标准"],
        colors=["奶白", "雾灰"],
        stock={"标准": 40},
        params={"容量": "420ml", "材质": "高白瓷", "是否可微波": "是"},
        aliases=("马克杯", "杯子", "水杯"),
    ),
}

COUPONS: dict[str, dict[str, Any]] = {
    "CPN-NEW-20": {
        "coupon_id": "CPN-NEW-20",
        "user_id": "u_1001",
        "amount": 20.0,
        "threshold": 99.0,
        "category": "服饰",
        "valid_until": "2026-12-31",
        "status": "unused",
    }
}

MEMBERS: dict[str, dict[str, Any]] = {
    "u_1001": {"user_id": "u_1001", "level": "金卡", "points": 1280, "growth": 3200, "tags": ["高复购", "尺码稳定"]},
    "u_2002": {"user_id": "u_2002", "level": "普通会员", "points": 60, "growth": 60, "tags": ["新客"]},
}

# 售后单内存副本（DB 模式下写 aftersales 表，此处仅兜底）
AFTERSALES: dict[str, dict[str, Any]] = {}

TRACKINGS: dict[str, list[dict[str, str]]] = {
    "SF1234567890": [
        {"time": "2026-10-29 09:10", "node": "已发出", "desc": "包裹已从上海仓发出"},
        {"time": "2026-10-29 14:22", "node": "运输中", "desc": "快件已到达杭州转运中心"},
        {"time": "2026-10-30 08:05", "node": "派送中", "desc": "快件正在派送，请保持电话畅通"},
    ],
    "SF1234500111": [
        {"time": "2026-10-18 11:00", "node": "已发出", "desc": "包裹已从上海仓发出"},
        {"time": "2026-10-19 16:40", "node": "已签收", "desc": "本人已签收"},
    ],
}


def _order_from_row(row: Any) -> Order:
    """DB 行 → Order 对象（含明细行）。"""
    order_no = row["order_no"]
    items = [
        {
            "sku_id": r["sku_id"],
            "size": r["size"],
            "color": r["color"],
            "quantity": r["quantity"],
        }
        for r in db.query_all(
            "SELECT sku_id, size, color, quantity FROM order_items WHERE order_no = ? ORDER BY id",
            (order_no,),
        )
    ]
    return Order(
        order_no=order_no,
        user_id=row["user_id"],
        sku_id=row["sku_id"],
        title=row["title"],
        amount=float(row["amount"]),
        status=row["status"],
        created_at=row["created_at"],
        ship_time=row["ship_time"],
        carrier=row["carrier"],
        tracking_no=row["tracking_no"],
        paid=bool(row["paid"]),
        invoice_types=[t for t in (row["invoice_types"] or "").split(",") if t],
        items=items,
    )


def _product_from_row(row: Any) -> Product:
    """DB 行 → Product 对象。"""
    return Product(
        sku_id=row["sku_id"],
        title=row["title"],
        category=row["category"],
        price=float(row["price"]),
        sizes=list(loads(row["sizes"], []) or []),
        colors=list(loads(row["colors"], []) or []),
        stock=dict(loads(row["stock"], {}) or {}),
        params=dict(loads(row["params"], {}) or {}),
        aliases=tuple(loads(row["aliases"], []) or []),
    )


# —— 查询接口 ——
def find_order(user_id: str, order_no: str | None) -> Order | None:
    """以用户身份为主键查询订单。订单号对不上用户身份则返回 None。"""
    if not order_no:
        return None
    if _db_ready():
        try:
            row = db.query_one(
                "SELECT * FROM orders WHERE order_no = ? AND user_id = ?",
                (order_no, user_id),
            )
            if row is None:
                # 订单不存在或不属于本人 —— 两者在响应上不可区分，避免泄露他人订单
                return None
            return _order_from_row(row)
        except sqlite3.Error:
            pass
    order = ORDERS.get(order_no)
    if order is None or order.user_id != user_id:
        return None
    return order


def list_user_orders(user_id: str, limit: int = 5) -> list[Order]:
    if _db_ready():
        try:
            rows = db.query_all(
                "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [_order_from_row(r) for r in rows]
        except sqlite3.Error:
            pass
    orders = [o for o in ORDERS.values() if o.user_id == user_id]
    orders.sort(key=lambda o: o.created_at, reverse=True)
    return orders[:limit]


def get_product(sku_id: str | None) -> Product | None:
    if not sku_id:
        return None
    if _db_ready():
        try:
            row = db.query_one("SELECT * FROM products WHERE sku_id = ?", (sku_id,))
            return _product_from_row(row) if row is not None else None
        except sqlite3.Error:
            pass
    return PRODUCTS.get(sku_id)


def list_products() -> list[Product]:
    """在售商品目录（「你们有哪些商品」这类导购问题的唯一事实源）。"""
    if _db_ready():
        try:
            rows = db.query_all("SELECT * FROM products ORDER BY sku_id")
            return [_product_from_row(r) for r in rows]
        except sqlite3.Error:
            pass
    return list(PRODUCTS.values())


def find_product_by_text(text: str) -> Product | None:
    """从自由描述中定位商品：先精确标题，再别名（长别名优先）。"""
    if not text:
        return None
    catalog = list_products()
    for product in catalog:
        if product.title and product.title in text:
            return product
    best: tuple[int, Product] | None = None
    for product in catalog:
        for alias in product.aliases:
            if alias in text and (best is None or len(alias) > best[0]):
                best = (len(alias), product)
    return best[1] if best else None


def list_user_coupons(user_id: str) -> list[dict[str, Any]]:
    if _db_ready():
        try:
            rows = db.query_all(
                "SELECT * FROM coupons WHERE user_id = ? AND status = 'unused'",
                (user_id,),
            )
            return [dict(r) for r in rows]
        except sqlite3.Error:
            pass
    return [c for c in COUPONS.values() if c.get("user_id") == user_id and c.get("status") == "unused"]


def issue_coupon(
    user_id: str,
    amount: float,
    threshold: float,
    category: str = "服饰",
    valid_until: str = "2026-12-31",
) -> dict[str, Any]:
    """发放优惠券并持久化。

    必须落库：发券是写操作，只写内存等于没发 —— 用户下次查询看不到券，
    体验上就是「客服说发了券但没收到」。
    """
    coupon_id = f"CPN-NEW-{random.randint(10, 99)}"
    record = {
        "coupon_id": coupon_id,
        "user_id": user_id,
        "amount": float(amount),
        "threshold": float(threshold),
        "category": category,
        "valid_until": valid_until,
        "status": "unused",
    }
    if _db_ready():
        try:
            db.execute(
                """INSERT INTO coupons
                   (coupon_id, user_id, amount, threshold, category, valid_until, status)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    coupon_id,
                    user_id,
                    float(amount),
                    float(threshold),
                    category,
                    valid_until,
                    "unused",
                ),
            )
            return record
        except sqlite3.Error:
            pass
    COUPONS[coupon_id] = record
    return record


def update_order_status(order_no: str, status: str) -> bool:
    """更新订单状态并持久化。

    写操作必须真的改数据：取消订单后状态不变，用户再问一次还是「可取消」，
    客服就成了复读机。
    """
    if _db_ready():
        try:
            rows = db.execute(
                "UPDATE orders SET status = ? WHERE order_no = ?",
                (status, order_no),
            )
            return rows > 0
        except sqlite3.Error:
            pass
    order = ORDERS.get(order_no)
    if order is None:
        return False
    order.status = status
    return True


def create_aftersale(
    user_id: str,
    order_no: str,
    kind: str,
    reason: str,
    refund_amount: float,
    refund_sla_days: int,
    idempotency_key: str = "",
) -> dict[str, Any]:
    """创建售后单并持久化。

    售后单号规则与内存实现保持一致（`AS` + 订单号后 6 位），
    这样 demo 与测试断言在 DB / 内存两种模式下都成立。

    `idempotency_key` 落库并建唯一索引：同一幂等键重复提交只会有一张售后单，
    这是「下游不出现两笔退款」的最后一道闸门（前面还有幂等库与业务锁）。
    """
    record = {
        "aftersale_no": _next_aftersale_no(order_no),
        "order_no": order_no,
        "user_id": user_id,
        "type": kind,
        "reason": reason,
        "refund_amount": round(float(refund_amount), 2),
        "refund_sla_days": int(refund_sla_days),
        "status": "已受理",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "order_status_before": _order_status(order_no),
        "idempotency_key": idempotency_key,
    }
    if _db_ready():
        try:
            db.execute(
                """INSERT INTO aftersales
                   (aftersale_no, order_no, user_id, type, reason,
                    refund_amount, refund_sla_days, status, created_at,
                    order_status_before, idempotency_key)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["aftersale_no"],
                    order_no,
                    user_id,
                    kind,
                    reason,
                    record["refund_amount"],
                    record["refund_sla_days"],
                    record["status"],
                    record["created_at"],
                    record["order_status_before"],
                    idempotency_key,
                ),
            )
            return record
        except sqlite3.Error:
            pass
    AFTERSALES[record["aftersale_no"]] = record
    return record


def _next_aftersale_no(order_no: str) -> str:
    """同一订单可以有多张售后单（改额会产生新版），单号必须唯一。

    首单保持 `AS` + 后 6 位的旧格式，避免改变既有演示输出；
    后续单追加序号。
    """
    base = f"AS{order_no[-6:]}"
    if _db_ready():
        try:
            row = db.query_one(
                "SELECT COUNT(*) AS c FROM aftersales WHERE order_no = ?", (order_no,)
            )
            count = int(row["c"]) if row is not None else 0
            return base if count == 0 else f"{base}-{count + 1}"
        except sqlite3.Error:
            pass
    count = sum(1 for r in AFTERSALES.values() if r.get("order_no") == order_no)
    return base if count == 0 else f"{base}-{count + 1}"


def _order_status(order_no: str) -> str:
    """读订单当前状态（售后撤销时用来恢复）。"""
    if _db_ready():
        try:
            row = db.query_one("SELECT status FROM orders WHERE order_no = ?", (order_no,))
            if row is not None:
                return str(row["status"])
        except sqlite3.Error:
            pass
    order = ORDERS.get(order_no)
    return order.status if order is not None else ""


def find_aftersale_by_key(idempotency_key: str) -> dict[str, Any] | None:
    """按幂等键查售后单：下游去重（同键只认第一张单）。"""
    if not idempotency_key:
        return None
    if _db_ready():
        try:
            row = db.query_one(
                "SELECT * FROM aftersales WHERE idempotency_key = ?", (idempotency_key,)
            )
            return dict(row) if row is not None else None
        except sqlite3.Error:
            pass
    for record in AFTERSALES.values():
        if record.get("idempotency_key") == idempotency_key:
            return record
    return None


def cancel_aftersale(aftersale_no: str) -> dict[str, Any] | None:
    """撤销售后单：状态置「已撤销」，并把订单状态恢复到提交前的值。

    幂等：已撤销的单再次撤销直接返回成功，不报错（撤销请求重发是常态）。
    """
    if _db_ready():
        try:
            row = db.query_one("SELECT * FROM aftersales WHERE aftersale_no = ?", (aftersale_no,))
            if row is None:
                return None
            record = dict(row)
            if record.get("status") != "已撤销":
                db.execute(
                    "UPDATE aftersales SET status = '已撤销' WHERE aftersale_no = ?",
                    (aftersale_no,),
                )
                record["status"] = "已撤销"
                before = str(record.get("order_status_before") or "")
                if before:
                    update_order_status(str(record["order_no"]), before)
            return record
        except sqlite3.Error:
            pass
    record = AFTERSALES.get(aftersale_no)
    if record is None:
        return None
    if record.get("status") != "已撤销":
        record["status"] = "已撤销"
        before = str(record.get("order_status_before") or "")
        if before:
            update_order_status(str(record["order_no"]), before)
    return record


def get_member(user_id: str) -> dict[str, Any]:
    if _db_ready():
        try:
            row = db.query_one("SELECT * FROM members WHERE user_id = ?", (user_id,))
            if row is not None:
                return {
                    "user_id": row["user_id"],
                    "level": row["level"],
                    "points": int(row["points"]),
                    "growth": int(row["growth"]),
                    "tags": loads(row["tags"], []) or [],
                }
        except sqlite3.Error:
            pass
    return MEMBERS.get(user_id, {"user_id": user_id, "level": "普通会员", "points": 0, "growth": 0, "tags": []})


def get_tracking(tracking_no: str) -> list[dict[str, str]]:
    if not tracking_no:
        return []
    if _db_ready():
        try:
            rows = db.query_all(
                "SELECT time, node, desc FROM trackings WHERE tracking_no = ? ORDER BY seq",
                (tracking_no,),
            )
            return [{"time": r["time"], "node": r["node"], "desc": r["desc"]} for r in rows]
        except sqlite3.Error:
            pass
    return TRACKINGS.get(tracking_no, [])


# —— 计价（唯一权威口径，模型不得自行加减乘除）——
def calc_final_price(
    sku_id: str,
    amount: float | None = None,
    coupon_id: str | None = None,
    user_id: str = "",
) -> dict[str, Any]:
    """服务端计算到手价。返回明细供 D3 渲染，禁止模型参与运算。"""
    product = get_product(sku_id)
    base = float(amount if amount is not None else (product.price if product else 0.0))
    lines: list[dict[str, Any]] = [{"label": "商品原价", "value": base}]

    discount = 0.0
    if coupon_id:
        coupon = _get_coupon(coupon_id)
        if coupon and coupon.get("user_id") == user_id:
            if base >= float(coupon.get("threshold", 0)):
                discount += float(coupon["amount"])
                lines.append({"label": f"优惠券 {coupon_id}", "value": -float(coupon["amount"])})
            else:
                lines.append({"label": f"优惠券 {coupon_id}", "value": 0.0, "note": f"未达门槛 {coupon['threshold']} 元"})
        else:
            lines.append({"label": f"优惠券 {coupon_id}", "value": 0.0, "note": "优惠券不属于当前用户"})

    member = get_member(user_id)
    # 会员折扣率必须读策略服务，不在计价逻辑里写死
    discount_table = PolicyService().require("member_discount")
    rate = discount_table.get(member.get("level", "普通会员"), 1.0)
    member_cut = round(base * (1 - rate), 2)
    if member_cut > 0:
        discount += member_cut
        lines.append({"label": f"会员折扣 {member.get('level')}", "value": -member_cut})

    final = round(base - discount, 2)
    lines.append({"label": "到手价", "value": final})
    return {"base": base, "discount": round(discount, 2), "final": final, "lines": lines}


def pick_primary_order(orders: list[Order]) -> Order | None:
    """从订单列表里挑「事实来源单」。

    排除已取消订单：用户问「我的订单」时，客服报一单已取消的订单属于答非所问。
    已取消的订单仍然出现在列表里（用户要看到全量），只是不作为话术的事实来源。
    """
    for order in orders:
        if order.status != "已取消":
            return order
    return orders[0] if orders else None


def find_order_with_tracking(user_id: str) -> Order | None:
    """找出该用户最近一单**有物流轨迹**的订单。

    不能盲目取最近一单：最新订单可能是「待付款 / 已取消」，没有运单号，
    查物流会返回空轨迹，用户看到「查不到物流」而不是正确答案。
    """
    for order in list_user_orders(user_id, limit=10):
        if order.tracking_no and get_tracking(order.tracking_no):
            return order
    return None


def _get_coupon(coupon_id: str) -> dict[str, Any] | None:
    """按券号查券（DB 优先，内存兜底）。"""
    if _db_ready():
        try:
            row = db.query_one("SELECT * FROM coupons WHERE coupon_id = ?", (coupon_id,))
            return dict(row) if row is not None else None
        except sqlite3.Error:
            pass
    return COUPONS.get(coupon_id)


def data_source_info() -> dict[str, Any]:
    """数据源诊断信息（/healthz 与控制台展示用）。"""
    info: dict[str, Any] = {
        "mode": "db" if _db_ready() else "mock",
        "configured": settings.data_source,
        "db_path": str(settings.db_path),
        "tables": {},
    }
    if _use_db():
        info["tables"] = db.table_counts()
        info["db_exists"] = settings.db_path.exists()
    return info
