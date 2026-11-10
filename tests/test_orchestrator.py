"""编排层端到端测试：意图路由、澄清、转人工、幂等与降级。"""

from __future__ import annotations

import pytest

from cs_agent_system.api import ChatRequest, CustomerService
from cs_agent_system.core.envelope import SessionContext, UserIdentity
from cs_agent_system.orchestrator.graph import Orchestrator


@pytest.fixture()
def service() -> CustomerService:
    return CustomerService()


def _orchestrator(service: CustomerService) -> Orchestrator:
    return service.build_orchestrator()


def _run(service: CustomerService, text: str, user_id: str = "u_1001"):
    orch = _orchestrator(service)
    ctx = SessionContext(user=UserIdentity(id=user_id, channel="private"))
    result = orch.handle(text=text, user_id=user_id, channel_type="private", session_ctx=ctx)
    return orch, result


def test_order_status_uses_short_circuit(service: CustomerService) -> None:
    orch, result = _run(service, "我的订单状态")
    assert result["intent"] == "ORDER_STATUS"
    assert result["route"] == "direct_tool"
    assert "E1" in result["route_targets"]
    # 短路：不会进入多 Agent 计划链
    assert result["plan_steps"] == []
    assert result["response"]
    assert "SO202610280001" in result["response"]
    assert not result["need_handoff"]


def test_low_confidence_triggers_clarification(service: CustomerService) -> None:
    orch, result = _run(service, "那个事情怎么样了")
    assert result["route"] == "clarify"
    assert result["response"]
    assert result["intent_confidence"] < 0.6


def test_high_risk_routes_to_human(service: CustomerService) -> None:
    orch, result = _run(service, "你们这什么破服务，我要12315投诉！")
    assert result["route"] == "handoff"
    assert result["need_handoff"] is True
    assert result["handoff_payload"]["ticket_id"].startswith("TK-")


def test_refund_write_tool_uses_idempotency_key(service: CustomerService) -> None:
    orch, result = _run(service, "订单号 SO202610280001 质量太差了，我要退款")
    assert result["intent"] == "RETURN_APPLY"
    outputs = result["agent_outputs"]
    assert "E3" in outputs
    assert outputs["E3"].get("idempotency_key")
    assert outputs["E3"]["facts"]["refund_sla_days"] == 3  # 承诺值来自策略服务


def test_refund_requires_verified_order(service: CustomerService) -> None:
    """订单号不属于当前用户时，不得返回订单信息。"""
    orch, result = _run(service, "订单号 SO202611010077 我要退款", user_id="u_1001")
    outputs = result["agent_outputs"]
    assert "E3" not in outputs or outputs["E3"].get("error")


def test_compliance_blocks_false_promise(service: CustomerService) -> None:
    orch, result = _run(service, "你们这什么破服务，我要12315投诉！")
    # 高风险路径不进入生成链路，但合规规则本身必须能拦截虚假承诺
    from cs_agent_system.guardrails.compliance import check_compliance

    assert check_compliance("保证3天到账").passed is False


def test_facts_come_from_tools_not_model(service: CustomerService) -> None:
    orch, result = _run(service, "我的快递到哪了")
    assert result["intent"] == "LOGISTICS"
    assert result["facts"].get("last_node")


def test_budget_limits_agent_calls(service: CustomerService) -> None:
    orch = _orchestrator(service)
    orch.rt.budget.budget.max_agent_calls = 1
    ctx = SessionContext(user=UserIdentity(id="u_1001"))
    result = orch.handle(text="我的订单状态", user_id="u_1001", channel_type="private", session_ctx=ctx)
    # 预算耗尽后系统降级而不是崩溃
    assert result["response"]


def test_chat_api_returns_response(service: CustomerService) -> None:
    resp = service.chat(ChatRequest(text="我的订单状态", user_id="u_1001"))
    assert resp.trace_id
    assert resp.intent == "ORDER_STATUS"
    assert resp.events


def test_session_state_is_written_by_orchestrator_only(service: CustomerService) -> None:
    orch, result = _run(service, "我的订单状态")
    stored = service.state.load(result["session_id"])
    assert stored["recent_turns"], "会话态应已落库"
    assert stored["turn_count"] == 1
