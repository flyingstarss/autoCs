"""幂等存储：写操作去重与业务锁的持久化实现。

这里存两类信息，职责必须分清：

1. **尝试记录**（主键 `key`）：一次具体请求。超时后按它回查「到底执行了没」。
2. **业务锁**（`business_key` 上的部分唯一索引）：一笔业务事实。
   `(工具, 目标实体, 版本)` 同时只允许一条 `pending / executed / uncertain` 记录。

状态机：

    pending ──成功──> executed ──撤销──> voided
       │                  │
       │                  └──超过 TTL 无结果──> uncertain（禁止重发，只能回查）
       ├──超时/异常──────> uncertain
       ├──需人工审批──────> rejected（不占用业务锁，审批通过后可重新提交）
       └──业务失败────────> failed（不占用业务锁）

为什么必须落库：幂等记录只在内存里时，服务重启后就查不到了 ——
恰恰在最需要它的时候（重启后收到重复请求、超时后回查）失效。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from . import db

# pending 超过这个时长仍无结果，视为「状态未知」而不是「还在跑」。
# 十五分钟足够覆盖一次正常的 OMS 调用；再久不回，说明进程已经没了。
PENDING_TTL = timedelta(minutes=15)

ACTIVE_STATUSES: tuple[str, ...] = ("pending", "executed", "uncertain")


@dataclass(frozen=True)
class IdempotentRecord:
    """一次写操作的执行记录。"""

    key: str
    tool: str
    trace_id: str
    executed_at: str
    result: dict[str, Any] = field(default_factory=dict)
    business_key: str | None = None
    order_no: str = ""
    version: int = 1
    status: str = "executed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "tool": self.tool,
            "trace_id": self.trace_id,
            "executed_at": self.executed_at,
            "result": self.result,
            "business_key": self.business_key,
            "order_no": self.order_no,
            "version": self.version,
            "status": self.status,
        }

    @property
    def is_stale_pending(self) -> bool:
        """pending 太久 = 执行进程已经没了，状态未知。"""
        if self.status != "pending":
            return False
        try:
            started = datetime.strptime(self.executed_at, "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return True
        return datetime.now() - started > PENDING_TTL


@dataclass(frozen=True)
class Claim:
    """抢占业务锁的结果。"""

    record: IdempotentRecord
    created: bool
    """True 表示本次新建（可以继续执行）；False 表示已有记录（走回放/冲突处理）。"""


class IdempotencyStore(Protocol):
    """幂等存储协议。"""

    def get(self, key: str) -> IdempotentRecord | None: ...

    def put(self, key: str, tool: str, trace_id: str, result: dict[str, Any]) -> None: ...

    def delete(self, key: str) -> None: ...

    def claim(
        self,
        *,
        key: str,
        tool: str,
        trace_id: str,
        business_key: str | None,
        order_no: str,
        version: int,
    ) -> Claim: ...

    def set_status(self, key: str, status: str, result: dict[str, Any] | None = None) -> None: ...

    def find_active(self, business_key: str) -> IdempotentRecord | None: ...

    def list_by_order(self, order_no: str, tool: str | None = None) -> list[IdempotentRecord]: ...

    def max_version(self, tool: str, order_no: str) -> int: ...


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MemoryIdempotencyStore:
    """内存实现：DB 不可用时的兜底。重启后失效。"""

    def __init__(self) -> None:
        self._records: dict[str, IdempotentRecord] = {}

    def get(self, key: str) -> IdempotentRecord | None:
        return self._records.get(key)

    def put(self, key: str, tool: str, trace_id: str, result: dict[str, Any]) -> None:
        old = self._records.get(key)
        self._records[key] = IdempotentRecord(
            key=key,
            tool=tool,
            trace_id=trace_id,
            executed_at=_now(),
            result=result,
            business_key=old.business_key if old else None,
            order_no=old.order_no if old else "",
            version=old.version if old else 1,
            status="executed",
        )

    def delete(self, key: str) -> None:
        self._records.pop(key, None)

    def claim(
        self,
        *,
        key: str,
        tool: str,
        trace_id: str,
        business_key: str | None,
        order_no: str,
        version: int,
    ) -> Claim:
        existing = self._records.get(key)
        if existing is not None:
            if existing.status in ("rejected", "failed", "voided"):
                self.set_status(key, "pending", {})
                record = self.get(key)
                assert record is not None
                return Claim(record=record, created=True)
            return Claim(record=existing, created=False)
        if business_key:
            active = self.find_active(business_key)
            if active is not None:
                return Claim(record=active, created=False)
        record = IdempotentRecord(
            key=key,
            tool=tool,
            trace_id=trace_id,
            executed_at=_now(),
            result={},
            business_key=business_key,
            order_no=order_no,
            version=version,
            status="pending",
        )
        self._records[key] = record
        return Claim(record=record, created=True)

    def set_status(self, key: str, status: str, result: dict[str, Any] | None = None) -> None:
        old = self._records.get(key)
        if old is None:
            return
        self._records[key] = IdempotentRecord(
            key=old.key,
            tool=old.tool,
            trace_id=old.trace_id,
            executed_at=old.executed_at,
            result=old.result if result is None else result,
            business_key=old.business_key,
            order_no=old.order_no,
            version=old.version,
            status=status,
        )

    def find_active(self, business_key: str) -> IdempotentRecord | None:
        for record in self._records.values():
            if record.business_key != business_key or record.status not in ACTIVE_STATUSES:
                continue
            if record.is_stale_pending:
                # 过期 pending 不再占用业务锁，转为「状态未知」
                self.set_status(record.key, "uncertain")
                return self.get(record.key)
            return record
        return None

    def list_by_order(self, order_no: str, tool: str | None = None) -> list[IdempotentRecord]:
        rows = [
            r
            for r in self._records.values()
            if r.order_no == order_no and (tool is None or r.tool == tool)
        ]
        rows.sort(key=lambda r: (r.version, r.executed_at))
        return rows

    def max_version(self, tool: str, order_no: str) -> int:
        versions = [r.version for r in self._records.values() if r.tool == tool and r.order_no == order_no]
        return max(versions, default=0)


class DbIdempotencyStore:
    """SQLite 实现：跨进程、跨重启可回查。

    表结构见 `registry/db.py::SCHEMA` 的 `idempotency` 表；业务锁由
    `idx_idem_business_active` 这个部分唯一索引保证，插入冲突即是「已存在」。
    """

    _COLUMNS = "key, tool, trace_id, executed_at, result, business_key, order_no, version, status"

    def _row_to_record(self, row: sqlite3.Row) -> IdempotentRecord:
        return IdempotentRecord(
            key=row["key"],
            tool=row["tool"],
            trace_id=row["trace_id"],
            executed_at=row["executed_at"],
            result=db.loads(row["result"], {}) or {},
            business_key=row["business_key"],
            order_no=row["order_no"] or "",
            version=int(row["version"] or 1),
            status=row["status"] or "executed",
        )

    def get(self, key: str) -> IdempotentRecord | None:
        try:
            row = db.query_one(f"SELECT {self._COLUMNS} FROM idempotency WHERE key = ?", (key,))
        except sqlite3.Error:
            return None
        return self._row_to_record(row) if row is not None else None

    def put(self, key: str, tool: str, trace_id: str, result: dict[str, Any]) -> None:
        db.execute(
            """INSERT INTO idempotency (key, tool, trace_id, executed_at, result, status)
               VALUES (?,?,?,?,?,'executed')
               ON CONFLICT(key) DO UPDATE SET
                   executed_at = excluded.executed_at,
                   result = excluded.result,
                   status = 'executed'""",
            (key, tool, trace_id, _now(), json.dumps(result, ensure_ascii=False, default=str)),
        )

    def delete(self, key: str) -> None:
        db.execute("DELETE FROM idempotency WHERE key = ?", (key,))

    def claim(
        self,
        *,
        key: str,
        tool: str,
        trace_id: str,
        business_key: str | None,
        order_no: str,
        version: int,
    ) -> Claim:
        existing = self.get(key)
        if existing is not None:
            if existing.status in ("rejected", "failed", "voided"):
                # 没有产生过副作用：允许用同一个请求键重新提交
                db.execute(
                    "UPDATE idempotency SET status='pending', result='{}', executed_at=? WHERE key=?",
                    (_now(), key),
                )
                record = self.get(key)
                assert record is not None
                return Claim(record=record, created=True)
            return Claim(record=existing, created=False)
        if business_key:
            active = self.find_active(business_key)
            if active is not None:
                return Claim(record=active, created=False)
        try:
            db.execute(
                """INSERT INTO idempotency
                   (key, tool, trace_id, executed_at, result, business_key, order_no, version, status)
                   VALUES (?,?,?,?,'{}',?,?,?,'pending')""",
                (key, tool, trace_id, _now(), business_key, order_no, version),
            )
        except sqlite3.IntegrityError:
            # 并发：另一个线程/进程刚抢到同一业务锁。回读已有记录。
            active = self.find_active(business_key) if business_key else None
            if active is not None:
                return Claim(record=active, created=False)
            again = self.get(key)
            if again is not None:
                return Claim(record=again, created=False)
            raise
        record = self.get(key)
        assert record is not None  # 刚插入成功，必然能读到
        return Claim(record=record, created=True)

    def set_status(self, key: str, status: str, result: dict[str, Any] | None = None) -> None:
        if result is None:
            db.execute("UPDATE idempotency SET status = ? WHERE key = ?", (status, key))
        else:
            db.execute(
                "UPDATE idempotency SET status = ?, result = ? WHERE key = ?",
                (status, json.dumps(result, ensure_ascii=False, default=str), key),
            )

    def find_active(self, business_key: str) -> IdempotentRecord | None:
        try:
            row = db.query_one(
                f"""SELECT {self._COLUMNS} FROM idempotency
                    WHERE business_key = ? AND status IN ('pending','executed','uncertain')
                    ORDER BY version DESC, executed_at DESC LIMIT 1""",
                (business_key,),
            )
        except sqlite3.Error:
            return None
        if row is None:
            return None
        record = self._row_to_record(row)
        if record.is_stale_pending:
            self.set_status(record.key, "uncertain")
            return self.get(record.key)
        return record

    def list_by_order(self, order_no: str, tool: str | None = None) -> list[IdempotentRecord]:
        sql = f"SELECT {self._COLUMNS} FROM idempotency WHERE order_no = ?"
        params: list[Any] = [order_no]
        if tool:
            sql += " AND tool = ?"
            params.append(tool)
        sql += " ORDER BY version ASC, executed_at ASC"
        try:
            rows = db.query_all(sql, params)
        except sqlite3.Error:
            return []
        return [self._row_to_record(r) for r in rows]

    def max_version(self, tool: str, order_no: str) -> int:
        try:
            row = db.query_one(
                "SELECT MAX(version) AS v FROM idempotency WHERE tool = ? AND order_no = ?",
                (tool, order_no),
            )
        except sqlite3.Error:
            return 0
        return int(row["v"] or 0) if row is not None else 0


def build_idempotency_store() -> IdempotencyStore:
    """按数据源可用性选择实现。"""
    if db.is_available():
        try:
            db.init_schema()
            return DbIdempotencyStore()
        except sqlite3.Error:
            pass
    return MemoryIdempotencyStore()


__all__ = [
    "ACTIVE_STATUSES",
    "Claim",
    "DbIdempotencyStore",
    "IdempotencyStore",
    "IdempotentRecord",
    "MemoryIdempotencyStore",
    "PENDING_TTL",
    "build_idempotency_store",
]
