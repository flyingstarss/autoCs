"""编排器状态与上下文。

编排器是会话态的唯一写入者（单一写入者原则）：
Agent 只返回增量，由编排器在此统一合并落库。
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..core.envelope import AgentEnvelope, Evidence, SessionContext, TraceEvent


class OrchestratorState(TypedDict, total=False):
    """LangGraph 状态。"""

    # —— 请求上下文 ——
    trace_id: str
    session_id: str
    turn: int
    text: str
    user_id: str
    channel_type: str
    is_group: bool
    # —— 会话上下文 ——
    session_ctx: SessionContext
    envelope: AgentEnvelope
    # —— 感知/决策结果 ——
    intent: str
    intent_confidence: float
    multi_intents: list[str]
    slots: dict[str, Any]
    missing: list[str]
    # 用户是否放弃了当前任务（P1 识别到的取消信号）
    cancelled: bool
    # 本轮是否走了续接兜底（从任务栈恢复到原意图）
    resumed: bool
    emotion: dict[str, Any]
    route: str
    route_reason: str
    route_targets: list[str]
    route_parallel: bool
    clarify_options: list[str]
    # 感知层给的澄清话术（如「您说的是哪一单」）：澄清节点优先采用
    clarify_hint: str
    plan_steps: list[dict[str, Any]]
    # —— 执行结果 ——
    facts: dict[str, Any]
    evidence: list[Evidence]
    tool_results: list[dict[str, Any]]
    agent_outputs: dict[str, dict[str, Any]]
    needs_approval: bool
    approval_reason: str
    # —— 生成与校验 ——
    draft: str
    fact_check_passed: bool
    fact_check_detail: str
    compliance_passed: bool
    compliance_detail: str
    reflect_count: int
    # —— 输出 ——
    response: str
    need_handoff: bool
    # 本轮转人工/待审批时生成的工单号（无待办为空串）
    ticket_id: str
    handoff_payload: dict[str, Any]
    degraded: bool
    degrade_level: str
    degraded_reasons: list[str]
    quality_score: float
    resolved: bool
    # —— 可观测 ——
    events: list[TraceEvent]
    errors: list[str]


def new_state(
    *,
    trace_id: str,
    session_id: str,
    turn: int,
    text: str,
    user_id: str,
    channel_type: str,
    is_group: bool,
    session_ctx: SessionContext,
    envelope: AgentEnvelope,
) -> OrchestratorState:
    return OrchestratorState(
        trace_id=trace_id,
        session_id=session_id,
        turn=turn,
        text=text,
        user_id=user_id,
        channel_type=channel_type,
        is_group=is_group,
        session_ctx=session_ctx,
        envelope=envelope,
        intent=session_ctx.intent.primary,
        intent_confidence=session_ctx.intent.confidence,
        multi_intents=list(session_ctx.intent.multi),
        slots=dict(session_ctx.slots),
        missing=list(session_ctx.missing),
        cancelled=False,
        resumed=False,
        emotion=session_ctx.emotion.model_dump(),
        # —— 路由 ——（全部键都必须有初值：LangGraph 的空通道在节点读取时会报错）
        route="",
        route_reason="",
        route_targets=[],
        route_parallel=False,
        clarify_options=[],
        clarify_hint="",
        plan_steps=[],
        # —— 执行 ——
        facts={},
        evidence=[],
        tool_results=[],
        agent_outputs={},
        needs_approval=False,
        approval_reason="",
        # —— 生成与校验 ——
        draft="",
        fact_check_passed=True,
        fact_check_detail="",
        compliance_passed=True,
        compliance_detail="",
        reflect_count=0,
        # —— 输出 ——
        response="",
        need_handoff=False,
        ticket_id="",
        handoff_payload={},
        degraded=False,
        degrade_level="",
        degraded_reasons=[],
        quality_score=0.0,
        resolved=False,
        # —— 可观测 ——
        events=[],
        errors=[],
    )
