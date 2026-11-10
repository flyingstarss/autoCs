"""数据播种：把拟真业务数据写入 D 盘 SQLite。

用法：
    python -m cs_agent_system.seed              # 建库 + 灌数据（幂等，可重复执行）
    python -m cs_agent_system.seed --reset      # 先清空再灌
    python -m cs_agent_system.seed --check      # 只看各表行数

数据与 `registry/data.py` 的内存字典一一对应 —— 订单号、SKU、会员、运单号
全部保持不变，这样 demo 与冒烟测试的断言（`SO202610280001`、`SKU-10086`、
`u_1001`、`CPN-NEW-20`、`SF1234567890`）在切换到真实数据库后依然成立。
"""

from __future__ import annotations

import sys

from .config import settings
from .registry import db
from .seed_data import (
    EXTRA_AFTERSALES,
    EXTRA_COUPONS,
    EXTRA_MEMBERS,
    EXTRA_ORDER_ITEMS,
    EXTRA_ORDERS,
    EXTRA_PRODUCTS,
    EXTRA_TRACKINGS,
)

NOW = "2026-11-05 10:00"

ORDERS = [
    # (order_no, user_id, sku_id, title, amount, status, created_at, ship_time,
    #  carrier, tracking_no, paid, invoice_types)
    ("SO202610280001", "u_1001", "SKU-10086", "纯棉圆领卫衣·雾霾蓝", 189.0, "已发货",
     "2026-10-28 10:22", "2026-10-29 09:10", "顺丰速运", "SF1234567890", 1, "电子普票,电子专票"),
    ("SO202610180233", "u_1001", "SKU-20431", "轻氧跑鞋·碳晶黑", 459.0, "已签收",
     "2026-10-18 09:05", "2026-10-18 11:00", "顺丰速运", "SF1234500111", 1, "电子普票"),
    ("SO202611010077", "u_2002", "SKU-10086", "纯棉圆领卫衣·雾霾蓝", 189.0, "待付款",
     "2026-11-01 20:41", "2026-11-02 08:00", "顺丰速运", "SF1234999888", 0, ""),
    # 覆盖「已取消 / 退款中 / 待发货」等状态，让口径更真实
    ("SO202610250911", "u_1001", "SKU-30021", "陶瓷马克杯·奶白", 69.0, "退款中",
     "2026-10-25 15:32", "2026-10-26 09:40", "中通快递", "ZT20261026001", 1, "电子普票"),
    ("SO202611030512", "u_2002", "SKU-20431", "轻氧跑鞋·月白", 459.0, "待发货",
     "2026-11-03 11:18", "2026-11-04 09:00", "顺丰速运", "SF1234777888", 1, "电子普票,电子专票"),
    ("SO202610290733", "u_1001", "SKU-10086", "纯棉圆领卫衣·米白", 189.0, "已取消",
     "2026-10-29 19:55", "", "顺丰速运", "", 0, ""),
]

ORDER_ITEMS = [
    ("SO202610280001", "SKU-10086", "M", "雾霾蓝", 1),
    ("SO202610180233", "SKU-20431", "42", "碳晶黑", 1),
    ("SO202611010077", "SKU-10086", "L", "米白", 2),
    ("SO202610250911", "SKU-30021", "标准", "奶白", 1),
    ("SO202611030512", "SKU-20431", "42", "月白", 1),
    ("SO202610290733", "SKU-10086", "M", "米白", 1),
]

PRODUCTS = [
    # (sku_id, title, category, price, sizes, colors, stock, params, aliases)
    ("SKU-10086", "纯棉圆领卫衣", "服饰", 189.0,
     ["S", "M", "L", "XL"], ["雾霾蓝", "米白", "黑色"],
     {"S": 12, "M": 8, "L": 0, "XL": 3},
     {"材质": "100% 精梳棉", "克重": "320g", "版型": "宽松", "适用季节": "春秋"},
     ["卫衣", "上衣", "圆领", "外套"]),
    ("SKU-20431", "轻氧跑鞋", "运动", 459.0,
     ["40", "41", "42", "43"], ["碳晶黑", "月白"],
     {"40": 5, "41": 2, "42": 6, "43": 0},
     {"鞋面": "飞织网布", "中底": "超临界发泡", "重量": "单只 235g"},
     ["跑鞋", "鞋", "运动鞋"]),
    ("SKU-30021", "陶瓷马克杯", "家居", 69.0,
     ["标准"], ["奶白", "雾灰"],
     {"标准": 40},
     {"容量": "420ml", "材质": "高白瓷", "是否可微波": "是"},
     ["马克杯", "杯子", "水杯"]),
    ("SKU-40077", "记忆棉护颈枕", "家居", 129.0,
     ["标准"], ["深空灰", "米杏"],
     {"标准": 26},
     {"材质": "慢回弹记忆棉", "高度": "10/12cm 双高度", "可否水洗": "枕套可洗"},
     ["护颈枕", "枕头", "颈枕"]),
    ("SKU-50118", "不锈钢保温杯 500ml", "家居", 89.0,
     ["500ml"], ["哑光黑", "奶白绿", "樱粉"],
     {"500ml": 58},
     {"材质": "316 不锈钢", "保温时长": "12 小时", "可否微波": "否"},
     ["保温杯", "杯子", "水杯", "焖烧杯"]),
    ("SKU-60302", "纯棉商务polo衫", "服饰", 259.0,
     ["M", "L", "XL", "XXL"], ["藏青", "炭灰", "白"],
     {"M": 21, "L": 18, "XL": 9, "XXL": 4},
     {"材质": "精梳棉混纺", "克重": "240g", "领型": "珠地翻领", "袖长": "短袖"},
     ["polo", "polo衫", "短袖", "衬衫"]),
]

MEMBERS = [
    ("u_1001", "金卡", 1280, 3200, ["高复购", "尺码稳定"]),
    ("u_2002", "普通会员", 60, 60, ["新客"]),
    ("u_3003", "银卡", 460, 980, ["浏览多", "价格敏感"]),
    ("u_4004", "钻石", 5680, 12400, ["高价值", "复购王", "偏好新品"]),
]

COUPONS = [
    ("CPN-NEW-20", "u_1001", 20.0, 99.0, "服饰", "2026-12-31", "unused"),
    ("CPN-BIG-50", "u_1001", 50.0, 299.0, "服饰", "2026-12-31", "unused"),
    ("CPN-HOME-15", "u_1001", 15.0, 79.0, "家居", "2026-12-15", "unused"),
    ("CPN-NEW10", "u_2002", 10.0, 49.0, "服饰", "2026-11-30", "unused"),
    ("CPN-VIP-80", "u_4004", 80.0, 399.0, "服饰", "2027-01-31", "unused"),
]

TRACKINGS = [
    ("SF1234567890", 1, "2026-10-29 09:10", "已发出", "包裹已从上海仓发出"),
    ("SF1234567890", 2, "2026-10-29 14:22", "运输中", "快件已到达杭州转运中心"),
    ("SF1234567890", 3, "2026-10-30 08:05", "派送中", "快件正在派送，请保持电话畅通"),
    ("SF1234500111", 1, "2026-10-18 11:00", "已发出", "包裹已从上海仓发出"),
    ("SF1234500111", 2, "2026-10-19 16:40", "已签收", "本人已签收"),
    ("SF1234999888", 1, "2026-11-02 08:00", "已发出", "包裹已从上海仓发出"),
    ("SF1234777888", 1, "2026-11-04 09:00", "已发出", "包裹已从上海仓发出"),
    ("ZT20261026001", 1, "2026-10-26 09:40", "已发出", "包裹已从上海仓发出"),
    ("ZT20261026001", 2, "2026-10-27 10:15", "运输中", "快件已到达杭州转运中心"),
]

# 策略配置：与 registry/data.py::POLICY 保持一致，是所有对外承诺值的唯一来源
POLICIES = {
    "refund_sla_days": 3,
    "return_window_days": 15,
    "max_compensation": 200.0,
    "free_shipping_threshold": 99.0,
    # 单笔订单可叠加的券张数上限
    "max_stack": 2,
    # 账号可持有的可用券总数上限（与 max_stack 是两件事）
    "max_coupon_hold": 5,
    "stackable_categories": ["服饰", "家居"],
    "member_discount": {"普通会员": 1.0, "银卡": 0.98, "金卡": 0.95, "钻石": 0.9},
    "point_rule": {"per_yuan": 1, "point_value": 0.01},
    "delivery_promise_days": 3,
    "invoice_types": ["电子普票", "电子专票"],
}


def reset() -> None:
    """清空所有业务表。"""
    for table in ("order_items", "trackings", "orders", "products", "members", "coupons", "policies", "aftersales"):
        db.execute(f"DELETE FROM {table}")
    # 长任务栈是会话态而非业务数据，但 reset 的语义是「回到干净初始态」，
    # 留着旧任务会让下一轮测试被历史任务干扰。
    db.execute("DELETE FROM tasks")
    print("已清空全部业务表与长任务栈")


def seed() -> None:
    """灌入拟真数据（幂等：先删后插）。

    基础数据 + `seed_data.py` 的扩充数据一起写入：扩充部分覆盖
    各状态订单、各等级会员、各品类券、异常物流轨迹等边界场景。
    """
    conn = db.connect()
    with db._write_lock:
        for table in ("order_items", "trackings", "orders", "products", "members", "coupons", "policies", "aftersales"):
            conn.execute(f"DELETE FROM {table}")

        conn.executemany(
            """INSERT INTO orders
               (order_no, user_id, sku_id, title, amount, status, created_at, ship_time,
                carrier, tracking_no, paid, invoice_types)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            ORDERS + EXTRA_ORDERS,
        )
        conn.executemany(
            """INSERT INTO order_items (order_no, sku_id, size, color, quantity)
               VALUES (?,?,?,?,?)""",
            ORDER_ITEMS + EXTRA_ORDER_ITEMS,
        )
        conn.executemany(
            """INSERT INTO products
               (sku_id, title, category, price, sizes, colors, stock, params, aliases)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                (sku, title, cat, price, db.dumps(sizes), db.dumps(colors),
                 db.dumps(stock), db.dumps(params), db.dumps(aliases))
                for sku, title, cat, price, sizes, colors, stock, params, aliases
                in (PRODUCTS + EXTRA_PRODUCTS)
            ],
        )
        conn.executemany(
            "INSERT INTO members (user_id, level, points, growth, tags) VALUES (?,?,?,?,?)",
            [(uid, lvl, pts, gth, db.dumps(tags)) for uid, lvl, pts, gth, tags in (MEMBERS + EXTRA_MEMBERS)],
        )
        conn.executemany(
            """INSERT INTO coupons
               (coupon_id, user_id, amount, threshold, category, valid_until, status)
               VALUES (?,?,?,?,?,?,?)""",
            COUPONS + EXTRA_COUPONS,
        )
        conn.executemany(
            """INSERT INTO trackings (tracking_no, seq, time, node, desc)
               VALUES (?,?,?,?,?)""",
            TRACKINGS + EXTRA_TRACKINGS,
        )
        conn.executemany(
            """INSERT INTO aftersales
               (aftersale_no, order_no, user_id, type, reason, refund_amount,
                refund_sla_days, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            EXTRA_AFTERSALES,
        )
        conn.executemany(
            "INSERT INTO policies (key, value, updated_at) VALUES (?,?,?)",
            [(k, db.dumps(v), NOW) for k, v in POLICIES.items()],
        )
        conn.commit()


def check() -> None:
    """打印各表行数与抽样数据。"""
    counts = db.table_counts()
    print(f"数据库文件：{db.db_file()}")
    print("-" * 56)
    for table, count in counts.items():
        print(f"  {table:<14} {count:>5} 行")
    print("-" * 56)
    for r in db.query_all(
        "SELECT order_no, user_id, title, amount, status FROM orders ORDER BY created_at DESC LIMIT 3"
    ):
        print(f"  订单 {r['order_no']}  {r['user_id']}  {r['title']}  ¥{r['amount']}  {r['status']}")
    row = db.query_one("SELECT title, price, stock FROM products LIMIT 1")
    if row:
        print(f"  商品 {row['title']}  ¥{row['price']}  库存 {db.loads(row['stock'])}")
    pol = db.query_one("SELECT value FROM policies WHERE key='refund_sla_days'")
    if pol:
        print(f"  策略 refund_sla_days = {db.loads(pol['value'])}")


def main() -> int:
    if not db.is_available():
        print(f"数据源未启用（DATA_SOURCE={settings.data_source}, DB_ENABLED={settings.db_enabled}）")
        return 1
    db.init_schema()
    args = set(sys.argv[1:])
    if "--reset" in args:
        reset()
    if "--check" in args:
        check()
        return 0
    seed()
    print(f"已写入数据库：{db.db_file()}")
    check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
