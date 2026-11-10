"""人工工单：接管与审批的待办载体。

解决什么问题
------------
系统里有两类「必须有人来看一眼」的待办：

1. **接管工单**（kind=handoff）：Q4 质量分低、合规阻断、风险过高等导致的转人工。
   管理员需要看到：用户是谁、在问什么、为什么被转过来、最近几单是什么。
2. **审批工单**（kind=approval）：写操作金额超阈值，在**产生副作用之前**被拦住。
   管理员需要看到：哪一单、多少钱、要执行什么写操作。

为什么必须落库
--------------
工单是**跨进程、跨会话**的待办。管理员可能在另一个进程、或者服务重启之后
才打开审核台，处理的是之前那一轮产生的待办。只放内存里的���，进程一重启就跟着消失，
「已提交人工审核」这句话对用户就是空头支票。

而且审批工单要能**重放**：批准后必须用**当初那把幂等键**重新执行写操作，
否则就是第二笔退款 —— 幂等键里带着 trace_id 与金额，重放必须能重建同一把键，
所以工单要连同 trace_id 一起存下来。

降级
----
DB 不可用时退化为进程内字典，保证「转人工了但没有工单」这种情况不会发生；
此时工单随进程消失，但接口仍可用（返回 503 提示 DB 不可用会让前端无从下手）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from . import db

# 工单状态机：
#   pending   待处理（管理员还没动）
#   approved  已批准（approval 类：写操作已重放执行）
#   rejected  已驳回
#   uncertain 状态未知（重放时超时 —— 需要人工去 OMS 侧确认，不能自动重试）
PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
UNCERTAIN = "uncertain"

_COLUMNS = (
    "ticket_id", "kind", "status", "reason", "summary", "intent",
    "user_id", "session_id", "trace_id", "order_no", "amount",
    "tool", "params", "idempotency_key",
    "decided_by", "decided_at", "decision_note", "result",
    "created_at", "updated_at",
)
_JSON_COLUMNS = ("params", "result")

_MEM: dict[str, dict[str, Any]] = {}


@dataclass
class Ticket:
    """一张工单。"""

    ticket_id: str
    # handoff=转人工接管；approval=写操作待审批
    kind: str = "handoff"
    status: str = PENDING
    reason: str = ""
    summary: str = ""
    intent: str = ""
    user_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    order_no: str = ""
    amount: float = 0.0
    # 审批通过后要重放的写操作
    tool: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""
    decided_by: str = ""
    decided_at: float = 0.0
    decision_note: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "kind": self.kind,
            "status": self.status,
            "reason": self.reason,
            "summary": self.summary,
            "intent": self.intent,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "order_no": self.order_no,
            "amount": self.amount,
            "tool": self.tool,
            "params": dict(self.params),
            "idempotency_key": self.idempotency_key,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "decision_note": self.decision_note,
            "result": dict(self.result),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def make_ticket_id(trace_id: str) -> str:
    """工单号：与 trace_id 绑定，便于从工单反查那一轮对话。"""
    return f"TK-{trace_id[-8:]}"


def _row_to_ticket(row: Any) -> Ticket:
    record: dict[str, Any] = {}
    keys = row.keys()
    for column in _COLUMNS:
        if column not in keys:
            continue
        value = row[column]
        if column in _JSON_COLUMNS:
            try:
                value = json.loads(value or "{}")
            except (TypeError, ValueError):
                value = {}
        record[column] = value
    record.setdefault("params", {})
    record.setdefault("result", {})
    return Ticket(**record)


def _available() -> bool:
    return db.is_available()


def create(ticket: Ticket) -> Ticket:
    """登记一张工单。同号覆盖（重放同一轮时不会产生两张）。"""
    now = time.time()
    ticket.created_at = ticket.created_at or now
    ticket.updated_at = now
    if not _available():
        _MEM[ticket.ticket_id] = ticket.to_dict()
        return ticket
    try:
        placeholders = ",".join("?" for _ in _COLUMNS)
        values = [
            json.dumps(getattr(ticket, column), ensure_ascii=False)
            if column in _JSON_COLUMNS
            else getattr(ticket, column)
            for column in _COLUMNS
        ]
        db.execute(
            f"INSERT OR REPLACE INTO tickets ({','.join(_COLUMNS)}) VALUES ({placeholders})",
            values,
        )
    except Exception:  # noqa: BLE001 —— 工单落库失败不能拖垮客服链路
        _MEM[ticket.ticket_id] = ticket.to_dict()
    return ticket


def get(ticket_id: str) -> Ticket | None:
    """按工单号取工单。"""
    if not ticket_id:
        return None
    if _available():
        try:
            row = db.query_one("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
            if row is not None:
                return _row_to_ticket(row)
        except Exception:  # noqa: BLE001
            pass
    record = _MEM.get(ticket_id)
    return Ticket(**record) if record else None


def list_tickets(
    status: str | None = None,
    kind: str | None = None,
    limit: int = 50,
) -> list[Ticket]:
    """列出工单，默认按创建时间倒序（最近的在前）。"""
    capped = max(1, min(int(limit), 200))
    if _available():
        try:
            clauses: list[str] = []
            args: list[Any] = []
            if status:
                clauses.append("status = ?")
                args.append(status)
            if kind:
                clauses.append("kind = ?")
                args.append(kind)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            args.append(capped)
            rows = db.query_all(
                f"SELECT * FROM tickets {where} ORDER BY created_at DESC LIMIT ?",
                tuple(args),
            )
            return [_row_to_ticket(row) for row in rows]
        except Exception:  # noqa: BLE001
            pass
    items = [
        Ticket(**record)
        for record in _MEM.values()
        if (not status or record.get("status") == status)
        and (not kind or record.get("kind") == kind)
    ]
    items.sort(key=lambda t: t.created_at, reverse=True)
    return items[:capped]


def decide(
    ticket_id: str,
    status: str,
    *,
    operator: str = "",
    note: str = "",
    result: dict[str, Any] | None = None,
) -> Ticket | None:
    """更新工单状态（批准/驳回/状态未知），返回更新后的工单。

    工单不存在返回 None —— 调用方据此回 404。
    """
    ticket = get(ticket_id)
    if ticket is None:
        return None
    ticket.status = status
    ticket.decided_by = operator
    ticket.decided_at = time.time()
    ticket.decision_note = note
    if result is not None:
        ticket.result = result
    ticket.updated_at = ticket.decided_at
    if not _available():
        _MEM[ticket.ticket_id] = ticket.to_dict()
        return ticket
    try:
        db.execute(
            """UPDATE tickets SET status=?, decided_by=?, decided_at=?, decision_note=?,
               result=?, updated_at=? WHERE ticket_id=?""",
            (
                ticket.status,
                ticket.decided_by,
                ticket.decided_at,
                ticket.decision_note,
                json.dumps(ticket.result, ensure_ascii=False),
                ticket.updated_at,
                ticket.ticket_id,
            ),
        )
    except Exception:  # noqa: BLE001
        _MEM[ticket.ticket_id] = ticket.to_dict()
    return ticket
