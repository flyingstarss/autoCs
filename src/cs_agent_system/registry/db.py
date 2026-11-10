"""SQLite 数据访问层。

设计要点：
1. **零外部服务**：只用 Python 标准库 `sqlite3`，不需要装 MySQL / Redis / ORM；
2. **文件落在 D 盘**：`settings.db_path` 默认指向工程根目录的 `data/cs_agent.db`；
3. **连接按线程复用**：`sqlite3` 连接不能跨线程共享，这里用 `threading.local`
   持有每线程一个连接，既满足 `check_same_thread=False` 的并发读，又避免重复建连；
4. **WAL 模式**：读写不互相阻塞，允许编排器并行扇出查询；
5. **失败即降级**：任何 DB 异常都不向上抛，由 `registry.data` 兜底回内存字典，
   保证「数据库挂了客服还能答话」而不是整条链路崩掉。

建表用 `CREATE TABLE IF NOT EXISTS`，重复执行安全；索引按查询主键建立。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from ..config import settings

_local = threading.local()
_write_lock = threading.Lock()

SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_no    TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        sku_id      TEXT NOT NULL,
        title       TEXT NOT NULL,
        amount      REAL NOT NULL,
        status      TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        ship_time   TEXT NOT NULL,
        carrier     TEXT NOT NULL,
        tracking_no TEXT NOT NULL,
        paid        INTEGER NOT NULL DEFAULT 0,
        invoice_types TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS order_items (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        order_no TEXT NOT NULL,
        sku_id   TEXT NOT NULL,
        size     TEXT NOT NULL DEFAULT '',
        color    TEXT NOT NULL DEFAULT '',
        quantity INTEGER NOT NULL DEFAULT 1
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_no)",
    """
    CREATE TABLE IF NOT EXISTS products (
        sku_id   TEXT PRIMARY KEY,
        title    TEXT NOT NULL,
        category TEXT NOT NULL,
        price    REAL NOT NULL,
        sizes    TEXT NOT NULL DEFAULT '[]',
        colors   TEXT NOT NULL DEFAULT '[]',
        stock    TEXT NOT NULL DEFAULT '{}',
        params   TEXT NOT NULL DEFAULT '{}',
        aliases  TEXT NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS members (
        user_id TEXT PRIMARY KEY,
        level   TEXT NOT NULL,
        points  INTEGER NOT NULL DEFAULT 0,
        growth  INTEGER NOT NULL DEFAULT 0,
        tags    TEXT NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS coupons (
        coupon_id  TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL,
        amount     REAL NOT NULL,
        threshold  REAL NOT NULL,
        category   TEXT NOT NULL DEFAULT '',
        valid_until TEXT NOT NULL DEFAULT '',
        status     TEXT NOT NULL DEFAULT 'unused'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trackings (
        tracking_no TEXT NOT NULL,
        seq         INTEGER NOT NULL,
        time        TEXT NOT NULL,
        node        TEXT NOT NULL,
        desc        TEXT NOT NULL,
        PRIMARY KEY (tracking_no, seq)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS policies (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS aftersales (
        aftersale_no   TEXT PRIMARY KEY,
        order_no       TEXT NOT NULL,
        user_id        TEXT NOT NULL,
        type           TEXT NOT NULL,
        reason         TEXT NOT NULL DEFAULT '',
        refund_amount  REAL NOT NULL,
        refund_sla_days INTEGER NOT NULL DEFAULT 0,
        status         TEXT NOT NULL DEFAULT '已受理',
        created_at     TEXT NOT NULL DEFAULT '',
        -- 提交时的订单原状态：撤销退款要把它恢复回去，
        -- 否则订单会永远停在「退款中」
        order_status_before TEXT NOT NULL DEFAULT '',
        -- 下游按这个键去重：同一个幂等键重复提交不得产生第二张售后单
        idempotency_key TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_aftersales_order ON aftersales(order_no)",
    """
    CREATE TABLE IF NOT EXISTS idempotency (
        key          TEXT PRIMARY KEY,
        tool         TEXT NOT NULL,
        trace_id     TEXT NOT NULL DEFAULT '',
        executed_at  TEXT NOT NULL,
        result       TEXT NOT NULL DEFAULT '{}',
        -- 业务锁：同一 (工具, 目标, 版本) 只允许一条生效记录
        business_key TEXT,
        order_no     TEXT NOT NULL DEFAULT '',
        version      INTEGER NOT NULL DEFAULT 1,
        -- pending / executed / uncertain / rejected / failed / voided
        status       TEXT NOT NULL DEFAULT 'executed'
    )
    """,
    # 长任务栈：用户中途切去别的任务后，原任务挂在这里等待续接。
    # 落库而非只放内存 —— 进程重启后用户说「继续」仍能找回原任务。
    """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id     TEXT PRIMARY KEY,
        session_id  TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        intent      TEXT NOT NULL,
        slots       TEXT NOT NULL DEFAULT '{}',
        missing     TEXT NOT NULL DEFAULT '[]',
        status      TEXT NOT NULL DEFAULT 'pending',
        created_at  REAL NOT NULL DEFAULT 0,
        updated_at  REAL NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id, status, updated_at DESC)",
    # 人工工单：接管与审批的待办载体。
    # 为什么必须落库：工单是**跨进程、跨会话**的待办 ——
    # 管理员在另一个进程（甚至重启后）打开审核台，处理的正是之前那一轮
    # 产生的待办。只放在内存里的话，进程一重启，待办就跟着消失了。
    """
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id       TEXT PRIMARY KEY,
        -- handoff=转人工接管；approval=写操作待审批
        kind            TEXT NOT NULL DEFAULT 'handoff',
        -- pending / approved / rejected / uncertain
        status          TEXT NOT NULL DEFAULT 'pending',
        reason          TEXT NOT NULL DEFAULT '',
        summary         TEXT NOT NULL DEFAULT '',
        intent          TEXT NOT NULL DEFAULT '',
        user_id         TEXT NOT NULL DEFAULT '',
        session_id      TEXT NOT NULL DEFAULT '',
        trace_id        TEXT NOT NULL DEFAULT '',
        order_no        TEXT NOT NULL DEFAULT '',
        amount          REAL NOT NULL DEFAULT 0,
        -- 审批通过后要重放的写操作：工具名 + 参数 + 幂等键来源
        tool            TEXT NOT NULL DEFAULT '',
        params          TEXT NOT NULL DEFAULT '{}',
        idempotency_key TEXT NOT NULL DEFAULT '',
        decided_by      TEXT NOT NULL DEFAULT '',
        decided_at      REAL NOT NULL DEFAULT 0,
        decision_note   TEXT NOT NULL DEFAULT '',
        result          TEXT NOT NULL DEFAULT '{}',
        created_at      REAL NOT NULL DEFAULT 0,
        updated_at      REAL NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status, created_at DESC)",
    # 可观测性：每次对话落一条可回溯的轨迹。
    # 为什么落库而不是只放内存：线上出问题时进程可能早已重启，
    # 只靠内存态 + 复现脚本无法回答「当时那一轮到底走了哪条路径」。
    """
    CREATE TABLE IF NOT EXISTS traces (
        trace_id        TEXT PRIMARY KEY,
        session_id      TEXT NOT NULL,
        user_id         TEXT NOT NULL,
        turn            INTEGER NOT NULL DEFAULT 0,
        ts              REAL NOT NULL,
        latency_ms      REAL NOT NULL DEFAULT 0,
        text            TEXT NOT NULL DEFAULT '',
        intent          TEXT NOT NULL DEFAULT '',
        confidence      REAL NOT NULL DEFAULT 0,
        route           TEXT NOT NULL DEFAULT '',
        targets         TEXT NOT NULL DEFAULT '[]',
        need_handoff    INTEGER NOT NULL DEFAULT 0,
        degraded        INTEGER NOT NULL DEFAULT 0,
        degraded_reasons TEXT NOT NULL DEFAULT '[]',
        quality_score   REAL NOT NULL DEFAULT 0,
        resolved        INTEGER NOT NULL DEFAULT 0,
        tokens          INTEGER NOT NULL DEFAULT 0,
        response        TEXT NOT NULL DEFAULT '',
        events          TEXT NOT NULL DEFAULT '[]',
        -- 本轮产生的工单号（转人工或待审批时才有）
        ticket_id       TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id, ts DESC)",
)

# —— 老库迁移 ——
# `CREATE TABLE IF NOT EXISTS` 不会给已存在的表补列，所以新增字段必须单独 ALTER。
# 用 PRAGMA table_info 判断后再加，保证重复执行安全。
_MIGRATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "idempotency": (
        ("business_key", "TEXT"),
        ("order_no", "TEXT NOT NULL DEFAULT ''"),
        ("version", "INTEGER NOT NULL DEFAULT 1"),
        ("status", "TEXT NOT NULL DEFAULT 'executed'"),
    ),
    "aftersales": (
        ("order_status_before", "TEXT NOT NULL DEFAULT ''"),
        ("idempotency_key", "TEXT NOT NULL DEFAULT ''"),
    ),
    # trace 与工单关联：查一轮对话时能直接定位到它产生的工单，
    # 查一张工单时也能回溯到它来自哪一轮对话。
    "traces": (
        ("ticket_id", "TEXT NOT NULL DEFAULT ''"),
    ),
}

# 依赖新增列，必须等迁移之后才能建。
_POST_MIGRATION_SCHEMA: tuple[str, ...] = (
    # 业务锁：同一业务键只允许一条「活着」的记录。
    # 状态过滤掉 rejected / failed / voided —— 失败或已撤销的不再占用锁。
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_idem_business_active
    ON idempotency(business_key)
    WHERE business_key IS NOT NULL
      AND status IN ('pending', 'executed', 'uncertain')
    """,
    "CREATE INDEX IF NOT EXISTS idx_idem_order ON idempotency(order_no, status)",
    "CREATE INDEX IF NOT EXISTS idx_idem_tool_version ON idempotency(tool, order_no, version)",
    # 下游去重：同一幂等键只能有一张售后单
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_aftersales_idem
    ON aftersales(idempotency_key)
    WHERE idempotency_key <> ''
    """,
)


def _migrate(conn: sqlite3.Connection) -> None:
    """给老库补齐新列（幂等）。"""
    for table, columns in _MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, ddl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")



def db_file() -> Path:
    """数据库文件路径（确保父目录存在）。"""
    path = settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def is_available() -> bool:
    """DB 是否可用：开关打开且文件所在盘可写。"""
    if not settings.db_enabled or settings.data_source != "db":
        return False
    try:
        db_file().parent.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def connect() -> sqlite3.Connection:
    """取当前线程的连接（不存在则新建并初始化 schema）。"""
    path = str(db_file())
    conn = getattr(_local, "conn", None)
    # 路径变了必须重连：缓存只看「有没有连接」而不看「连的是哪个库」，
    # 切换 db_path（热重载、测试指向临时库）就会继续往旧库写，
    # 表现为「写成功但读不到」，且跨线程时尤其难查。
    if conn is not None and getattr(_local, "path", None) == path:
        return conn
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
        _local.conn = None
    conn = sqlite3.connect(
        path,
        check_same_thread=False,
        timeout=5.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if settings.db_echo:
        conn.set_trace_callback(print)
    _local.conn = conn
    _local.path = path
    init_schema(conn)
    return conn


def close() -> None:
    """关闭当前线程连接（测试与热重载用）。"""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
        _local.path = None


def init_schema(conn: sqlite3.Connection | None = None) -> None:
    """建表 + 补列 + 建索引（幂等）。"""
    conn = conn or connect()
    with _write_lock:
        for stmt in SCHEMA:
            conn.execute(stmt)
        _migrate(conn)
        for stmt in _POST_MIGRATION_SCHEMA:
            conn.execute(stmt)
        conn.commit()


def query_all(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    """查询多行。异常由调用方兜底。"""
    return connect().execute(sql, tuple(params)).fetchall()


def query_one(sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    """查询单行。"""
    return connect().execute(sql, tuple(params)).fetchone()


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    """写入并提交。返回受影响行数。"""
    conn = connect()
    with _write_lock:
        cur = conn.execute(sql, tuple(params))
        conn.commit()
        return cur.rowcount


def executemany(sql: str, seq: Iterable[Iterable[Any]]) -> int:
    """批量写入并提交。"""
    conn = connect()
    with _write_lock:
        cur = conn.executemany(sql, [tuple(p) for p in seq])
        conn.commit()
        return cur.rowcount


# —— JSON 字段编解码：SQLite 无原生 JSON 列，统一存文本 ——
def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def loads(raw: str | None, default: Any = None) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    """sqlite3.Row → dict。"""
    return dict(row) if row is not None else {}


def table_counts() -> dict[str, int]:
    """各表行数（诊断与 /healthz 用）。"""
    counts: dict[str, int] = {}
    for table in (
        "orders",
        "order_items",
        "products",
        "members",
        "coupons",
        "trackings",
        "policies",
        "aftersales",
        "tasks",
    ):
        try:
            row = query_one(f"SELECT COUNT(*) AS c FROM {table}")
            counts[table] = int(row["c"]) if row else 0
        except sqlite3.Error:
            counts[table] = -1
    return counts
