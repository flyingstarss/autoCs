"""核心契约层：通信信封、意图集、预算、会话态与模型网关。

这一层不依赖任何业务域，只定义跨 Agent 的硬约束：
信封字段、封闭意图集、三重预算、会话态写入接口、模型分级路由。
"""

from .budget import BudgetExceeded, BudgetGuard
from .envelope import (
    AgentEnvelope,
    BudgetInfo,
    ChannelInfo,
    EmotionInfo,
    Evidence,
    HistoryInfo,
    IntentInfo,
    SessionContext,
    TraceEvent,
    UserIdentity,
)
from .ids import WRITE_TOOLS, new_idempotency_key, new_session_id, new_trace_id

__all__ = [
    "AgentEnvelope",
    "Evidence",
    "SessionContext",
    "UserIdentity",
    "ChannelInfo",
    "IntentInfo",
    "EmotionInfo",
    "HistoryInfo",
    "BudgetInfo",
    "TraceEvent",
    "BudgetGuard",
    "BudgetExceeded",
    "WRITE_TOOLS",
    "new_trace_id",
    "new_session_id",
    "new_idempotency_key",
]
