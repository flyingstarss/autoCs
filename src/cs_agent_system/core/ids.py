"""ID、幂等键与业务锁键。

这里有两把不同的锁，职责不能混：

- **业务锁键**（`new_business_key`）：描述「一笔业务事实」。
  同一 `(工具, 目标实体, 版本)` 在 `idempotency` 表里只允许存在一条生效记录，
  由数据库的部分唯一索引兜底。它回答的是「这个订单的这次退款到底做没做过」。
- **幂等键**（`new_idempotency_key`）：描述「一次请求尝试」。
  包含 `trace_id` 与金额，用于超时后回查、以及同一轮内的重复调用回放。
  它回答的是「这次调用是不是刚才那次的重发」。

为什么必须分开：只用幂等键防不住「用户中途改金额」——`trace_id` 每轮重建，
第二笔必然是新键；只在幂等键里塞业务唯一性又会把「重发」和「新决策」混为一谈。
业务锁负责不重复，版本负责让「修改」成为一次显式的、可撤销的新事实。
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime, timezone

# 写工具全集 —— 硬约束：系统内只允许这几个工具产生副作用。
# `aftersale.cancel` 是唯一允许的逆向操作：撤销必须走独立工具，
# 不能靠删幂等记录假装没发生过（钱可能已经退出去了）。
WRITE_TOOLS: frozenset[str] = frozenset(
    {"aftersale.apply", "aftersale.cancel", "coupon.issue", "order.action"}
)


def new_trace_id() -> str:
    """全链路追踪 ID，一次会话内跨 Agent 不变。"""
    return f"tr_{uuid.uuid4().hex[:16]}"


def new_session_id(user_id: str, channel_type: str) -> str:
    """会话 ID：私聊为用户+渠道，群聊为群+渠道。"""
    scope = user_id if channel_type != "group" else f"group:{user_id}"
    return f"ses_{scope}_{channel_type}"


def new_business_key(action: str, target_id: str, version: int = 1) -> str:
    """业务锁键 = sha256(工具|目标实体|版本)。

    不含 `trace_id`、不含金额：同一订单的同一版本退款，无论重试多少次、
    换几个会话、隔几轮再说一次，都落到同一个键上，只能有一条生效记录。
    金额变更必须通过 `version + 1` 变成新事实，并显式撤销旧版本。
    """
    raw = f"{action}|{target_id}|v{version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def new_idempotency_key(
    trace_id: str,
    action: str,
    user_id: str,
    target_id: str,
    *,
    version: int = 1,
    amount: float | None = None,
) -> str:
    """幂等键 = sha256(业务键 + 尝试上下文)。

    相比旧实现多带两样东西：
    - `version`：金额变更后是新请求，不能命中旧版本的缓存；
    - `amount`：即使版本号没变，金额不同也必须视为不同请求（防御性）。
    """
    business = f"{action}|{target_id}|v{version}"
    amount_part = "" if amount is None else f"{float(amount):.2f}"
    attempt = f"{trace_id}|{user_id}|{amount_part}"
    return hashlib.sha256(f"{business}#{attempt}".encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def today() -> date:
    return date.today()
