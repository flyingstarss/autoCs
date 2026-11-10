"""三重预算控制（架构文档 9.3 规则三 / 12.2）。

预算有三：最大 Agent 调用数、最大模型 Token、最大墙钟时间。
任一超限即降级，不允许「再试一次看看」。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .envelope import BudgetInfo


class BudgetExceeded(RuntimeError):
    """预算耗尽。调用方应走降级链，而不是重试。"""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"预算超限[{kind}]: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass
class BudgetGuard:
    """一次会话内的预算账本。并行扇出时线程安全。"""

    budget: BudgetInfo
    calls: int = 0
    tokens: int = 0
    _started_at: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # —— 查询 ——
    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._started_at) * 1000)

    @property
    def remaining_ms(self) -> int:
        return max(0, self.budget.deadline_ms - self.elapsed_ms)

    @property
    def remaining_calls(self) -> int:
        with self._lock:
            return max(0, self.budget.max_agent_calls - self.calls)

    @property
    def remaining_tokens(self) -> int:
        with self._lock:
            return max(0, self.budget.max_tokens - self.tokens)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "calls": self.calls,
                "tokens": self.tokens,
                "elapsed_ms": self.elapsed_ms,
                "max_agent_calls": self.budget.max_agent_calls,
                "max_tokens": self.budget.max_tokens,
                "deadline_ms": self.budget.deadline_ms,
            }

    # —— 消耗 ——
    def check(self) -> None:
        """预检查，不足则抛 BudgetExceeded。"""
        if self.remaining_calls <= 0:
            raise BudgetExceeded("agent_calls", f"已用 {self.calls} 次调用")
        if self.remaining_tokens <= 0:
            raise BudgetExceeded("tokens", f"已用 {self.tokens} tokens")
        if self.remaining_ms <= 0:
            raise BudgetExceeded("wall_clock", f"已耗时 {self.elapsed_ms}ms")

    def consume_call(self, n: int = 1) -> None:
        with self._lock:
            self.calls += n

    def consume_tokens(self, n: int) -> None:
        with self._lock:
            self.tokens += max(0, n)

    def ensure_tokens(self, need: int) -> None:
        if self.remaining_tokens < need:
            raise BudgetExceeded("tokens", f"需要 {need}，剩余 {self.remaining_tokens}")
