"""长任务栈：用户中断后未完成的任务在这里等待续接。

为什么需要独立的任务栈（而不是只靠会话态的 slots）：
1. **多任务并存**：用户可能同时挂着「退款」和「查物流」两个未完成任务。
   单一 slots/intent 只能记住最后一个，切换话题就把前一个顶掉了。
2. **跨进程**：会话态默认在内存里，进程重启即丢；任务栈落 SQLite，
   重启后用户说「继续」仍能找回原任务。

任务栈语义：
- 每个 `task_id` = `sha256(session_id + intent)`，所以**同一会话同一意图只有一个任务**，
  重复发起是「更新原任务」而非「堆一个新任务」；
- 多个不同意图的任务并存，按 `updated_at` 倒序排列（最近碰过的在最前）；
- 任务办完标记 `done`（不再出现在续接候选里），但保留记录用于回溯。

存储接口与 `registry/idempotency.py` 保持同构：协议 + DB 实现 + 内存兜底。
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import db

# 任务栈里最多保留的未完成任务数。
# 超过这个数说明用户在乱点，按「最近更新的优先」裁掉最老的，
# 避免任务栈无限膨胀把上下文撑爆。
MAX_TASKS_PER_SESSION = 8


@dataclass
class Task:
    """一个未完成的长任务。"""

    task_id: str
    session_id: str
    user_id: str
    intent: str
    slots: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    status: str = "pending"  # pending / done
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def is_pending(self) -> bool:
        return self.status == "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "intent": self.intent,
            "slots": dict(self.slots),
            "missing": list(self.missing),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        return cls(
            task_id=row["task_id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            intent=row["intent"],
            slots=db.loads(row["slots"], {}) or {},
            missing=db.loads(row["missing"], []) or [],
            status=row["status"],
            created_at=float(row["created_at"] or 0),
            updated_at=float(row["updated_at"] or 0),
        )


def make_task_id(session_id: str, intent: str) -> str:
    """任务 id = session + intent 的哈希。

    同一会话同一意图恒定同一个 id —— 这样「再次发起同一件事」是更新原任务，
    而不是在栈里堆重复任务。
    """
    raw = f"{session_id}|{intent}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


class TaskStore(Protocol):
    """任务栈存储协议。"""

    def list_pending(self, session_id: str) -> list[Task]:
        """列出该会话的未完成任务，按最近更新倒序。"""
        ...

    def get(self, task_id: str) -> Task | None: ...

    def upsert(self, task: Task) -> None:
        """创建或更新任务。"""
        ...

    def complete(self, task_id: str) -> None:
        """标记任务完成。"""
        ...

    def drop(self, task_id: str) -> None:
        """彻底删除任务。"""
        ...


class MemoryTaskStore:
    """内存实现：DB 不可用时的兜底。重启后失效。"""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def list_pending(self, session_id: str) -> list[Task]:
        items = [t for t in self._tasks.values() if t.session_id == session_id and t.is_pending]
        items.sort(key=lambda t: t.updated_at, reverse=True)
        return items

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def upsert(self, task: Task) -> None:
        self._tasks[task.task_id] = task
        self._trim(task.session_id)

    def complete(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task is not None:
            task.status = "done"
            task.updated_at = time.time()

    def drop(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)

    def _trim(self, session_id: str) -> None:
        items = self.list_pending(session_id)
        for task in items[MAX_TASKS_PER_SESSION:]:
            self._tasks.pop(task.task_id, None)


class DbTaskStore:
    """SQLite 实现：跨进程、跨重启可续接。

    表结构见 `registry/db.py::SCHEMA` 的 `tasks` 表。
    """

    def list_pending(self, session_id: str) -> list[Task]:
        try:
            rows = db.query_all(
                """SELECT * FROM tasks
                   WHERE session_id = ? AND status = 'pending'
                   ORDER BY updated_at DESC""",
                (session_id,),
            )
        except sqlite3.Error:
            return []
        return [Task.from_row(r) for r in rows]

    def get(self, task_id: str) -> Task | None:
        try:
            row = db.query_one("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        except sqlite3.Error:
            return None
        return Task.from_row(row) if row is not None else None

    def upsert(self, task: Task) -> None:
        db.execute(
            """INSERT INTO tasks
                   (task_id, session_id, user_id, intent, slots, missing,
                    status, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(task_id) DO UPDATE SET
                   slots = excluded.slots,
                   missing = excluded.missing,
                   status = excluded.status,
                   updated_at = excluded.updated_at""",
            (
                task.task_id,
                task.session_id,
                task.user_id,
                task.intent,
                db.dumps(task.slots),
                db.dumps(task.missing),
                task.status,
                task.created_at,
                task.updated_at,
            ),
        )
        self._trim(task.session_id)

    def complete(self, task_id: str) -> None:
        db.execute(
            "UPDATE tasks SET status = 'done', updated_at = ? WHERE task_id = ?",
            (time.time(), task_id),
        )

    def drop(self, task_id: str) -> None:
        db.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))

    def _trim(self, session_id: str) -> None:
        """超出上限时裁掉最老的未完成任务。"""
        try:
            rows = db.query_all(
                """SELECT task_id FROM tasks
                   WHERE session_id = ? AND status = 'pending'
                   ORDER BY updated_at DESC""",
                (session_id,),
            )
        except sqlite3.Error:
            return
        for row in rows[MAX_TASKS_PER_SESSION:]:
            db.execute("DELETE FROM tasks WHERE task_id = ?", (row["task_id"],))


def build_task_store() -> TaskStore:
    """按数据源可用性选择实现。"""
    if db.is_available():
        try:
            db.init_schema()
            return DbTaskStore()
        except sqlite3.Error:
            pass
    return MemoryTaskStore()


__all__ = [
    "Task",
    "TaskStore",
    "MemoryTaskStore",
    "DbTaskStore",
    "build_task_store",
    "make_task_id",
    "MAX_TASKS_PER_SESSION",
]
