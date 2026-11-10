"""统一通信信封与会话上下文（对应架构文档 9.2 / 10.2）。

设计要点：
1. `trace_id` 贯穿全链路，任何一次对话都能还原完整调用树；
2. `confidence` 是一等公民，每个 Agent 都必须给出，由编排器统一裁决。
   取值语义见 `core.confidence`：它是「产出可被信任的程度」，不是概率；
   `None` 表示本次未给出信号，由基类兜底为 0.0（必定触发澄清或转人工）；
3. `need_handoff` 让任何 Agent 都能主动请求人工接管；
4. 会话态遵循「单一写入者原则」——Agent 只返回 `state_delta` 增量，
   由编排器独占写入会话存储。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["low", "medium", "high"]
Sentiment = Literal["positive", "neutral", "negative"]


class Evidence(BaseModel):
    """Q1 产出的证据片段，必须带来源与版本，过期证据自动降权。"""

    source: str
    snippet: str
    layer: str = "policy"  # policy / script / faq / case / product
    version: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    score: float = 0.0

    def is_effective(self, on: date | None = None) -> bool:
        on = on or date.today()
        if self.effective_from and on < self.effective_from:
            return False
        if self.effective_to and on > self.effective_to:
            return False
        return True

    @property
    def expired(self) -> bool:
        return not self.is_effective()


class AgentEnvelope(BaseModel):
    """Agent 间统一消息信封。

    字段语义严格对齐架构文档 9.2：
    - `from_agent`（别名 from）= 发送方
    - `to_agent`（别名 to）= 接收方
    - `produced_by` = 产出本信封 payload 的处理单元（单元产出时等于自身编号）
    """

    model_config = ConfigDict(populate_by_name=True)

    trace_id: str
    session_id: str
    turn: int
    from_agent: str = Field(alias="from")
    to_agent: str = Field(alias="to")
    intent: str | None = None
    slots: dict[str, Any] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float | None = None
    need_handoff: bool = False
    cost_ms: int = 0
    degraded: bool = False
    error: str | None = None
    produced_by: str | None = None

    def child(self, to_agent: str, **kwargs: Any) -> "AgentEnvelope":
        """派生子信封：当前持有者发往 to_agent，继承 trace/session/turn 与上下文。"""
        base = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "turn": self.turn,
            "from": self.to_agent,
            "to": to_agent,
            "intent": self.intent,
            "slots": dict(self.slots),
            "missing": list(self.missing),
            "payload": dict(self.payload),
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            # need_handoff 是跨单元的判定信号（P2 判定拒答/追问无果），
            # 必须在派生子信封时继承；否则后续单元的 child() 会把它重置为 False，
            # 表现就是「上游已决定转人工，下游却继续正常执行」。
            "need_handoff": self.need_handoff,
        }
        base.update(kwargs)
        return AgentEnvelope(**base)

    def merge_slots(self, slots: dict[str, Any] | None) -> None:
        if slots:
            self.slots.update(slots)

    def delta(self, **kwargs: Any) -> dict[str, Any]:
        """构造会话态增量（由编排器独占写入）。"""
        return dict(kwargs)


class UserIdentity(BaseModel):
    """身份锚定。强个人数据的查询主键只能来自这里，不能来自用户口述。"""

    id: str
    channel: str = "web"
    is_guest: bool = False
    member_level: str = "普通会员"
    tags: list[str] = Field(default_factory=list)


class ChannelInfo(BaseModel):
    """渠道能力描述，决定 D3 生成的排版与长度。"""

    type: str = "web"
    max_len: int = 500
    is_group: bool = False


class IntentInfo(BaseModel):
    primary: str = "CHITCHAT"
    confidence: float = 0.0
    multi: list[str] = Field(default_factory=list)


class EmotionInfo(BaseModel):
    sentiment: Sentiment = "neutral"
    anger_score: float = 0.0
    risk_level: RiskLevel = "low"
    complaint_intent: bool = False
    escalation_signals: list[str] = Field(default_factory=list)


class HistoryInfo(BaseModel):
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    resolved_topics: list[str] = Field(default_factory=list)


class BudgetInfo(BaseModel):
    """三重预算：最大调用数 + 最大 Token + 最大墙钟时间。"""

    max_agent_calls: int = 12
    max_tokens: int = 8000
    deadline_ms: int = 15000


class SessionContext(BaseModel):
    """P4 产出的统一上下文视图，所有下游 Agent 的标准输入。"""

    # —— 身份锚定 ——
    user: UserIdentity = Field(default_factory=lambda: UserIdentity(id="anonymous"))
    channel: ChannelInfo = Field(default_factory=ChannelInfo)
    # —— 本轮理解 ——
    intent: IntentInfo = Field(default_factory=IntentInfo)
    slots: dict[str, Any] = Field(default_factory=dict)
    # 槽位来源元数据：{槽位名: {"turn": 抽取轮次, "intent": 抽取时的意图}}
    #
    # 为什么需要：槽位一旦填上就会跨轮留存，但用户可能换话题、换对象。
    # 例如第 1 轮查过 SO202610180233，第 2 轮问「我还有哪些没发货的」——
    # 旧槽位会让系统继续查那一单，而不是按新诉求列清单。
    # 有了来源信息，就能判断「这个槽位是不是上一轮别的意图留下的」。
    slot_meta: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # 每个关键槽位已被追问的次数：{槽位名: 次数}。
    # 追问必须有上限，否则用户拒答时会陷入「同一句问题反复问」的死循环。
    ask_counts: dict[str, int] = Field(default_factory=dict)
    # 本轮用户是否在文本里明确给了订单号：明确给了就以它为准，覆盖旧槽位。
    explicit_order_no: str = ""
    missing: list[str] = Field(default_factory=list)
    emotion: EmotionInfo = Field(default_factory=EmotionInfo)
    # —— 记忆 ——
    history: HistoryInfo = Field(default_factory=HistoryInfo)
    # —— 业务快照 ——
    recent_orders: list[dict[str, Any]] = Field(default_factory=list)
    active_coupons: list[dict[str, Any]] = Field(default_factory=list)
    cart_items: list[dict[str, Any]] = Field(default_factory=list)
    # —— 治理 ——
    budget: BudgetInfo = Field(default_factory=BudgetInfo)
    turn_count: int = 0
    trace_id: str = ""
    session_id: str = ""
    # 上一轮识别出的意图：用于「继续」「接着刚才」这类指代在模型放弃时的兜底。
    # 由 API 层从会话态恢复，编排器在每轮落库时更新。
    prev_intent: str = ""
    # 未完成任务栈：list[{"task_id","intent","slots","missing","updated_at"}]
    # 用户中途切去别的任务后，原任务不会消失，而是挂在这里等待续接。
    # 支持**多个**任务并存（退款 + 查物流可以同时挂着），按最近更新倒序。
    # 持久化在 `registry/tasks.py`（SQLite），进程重启后仍可续接。
    pending_tasks: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def pending_task(self) -> dict[str, Any]:
        """兼容视图：栈顶任务（最近更新的那个）。"""
        return self.pending_tasks[0] if self.pending_tasks else {}

    def has_slot(self, key: str) -> bool:
        value = self.slots.get(key)
        return value is not None and value != "" and value != []

    def require(self, keys: list[str]) -> list[str]:
        """返回缺失的关键槽位（缺什么比抽到什么更重要）。"""
        return [k for k in keys if not self.has_slot(k)]


class TraceEvent(BaseModel):
    """可观测性事件（对应架构文档 13.3）。"""

    agent_id: str
    event: str
    detail: str = ""
    cost_ms: int = 0
    confidence: float | None = None
    skipped: bool = False
    verdict: str | None = None
