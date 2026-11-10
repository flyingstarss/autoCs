"""核心契约测试：信封、预算、幂等、只读/写工具边界。"""

from __future__ import annotations

import uuid

import pytest

from cs_agent_system.core.budget import BudgetExceeded, BudgetGuard
from cs_agent_system.core.envelope import AgentEnvelope, BudgetInfo, SessionContext
from cs_agent_system.core.ids import WRITE_TOOLS, new_idempotency_key
from cs_agent_system.registry.tool_registry import ToolCallContext, ToolRegistry, ToolSpec


def _ctx(trace_id: str = "tr_1") -> ToolCallContext:
    return ToolCallContext(trace_id=trace_id, session_id="ses_1", user_id="u_1001")


def test_envelope_child_inherits_trace_and_session() -> None:
    env = AgentEnvelope(trace_id="tr_x", session_id="ses_y", turn=1, **{"from": "L0", "to": "P1"})
    child = env.child("P2", confidence=0.8)
    assert child.trace_id == "tr_x"
    assert child.session_id == "ses_y"
    assert child.to_agent == "P2"
    assert child.confidence == 0.8


def test_budget_exhausts_on_agent_calls() -> None:
    guard = BudgetGuard(budget=BudgetInfo(max_agent_calls=2, max_tokens=100, deadline_ms=10_000))
    guard.consume_call()
    guard.consume_call()
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_budget_exhausts_on_tokens() -> None:
    guard = BudgetGuard(budget=BudgetInfo(max_agent_calls=10, max_tokens=10, deadline_ms=10_000))
    guard.consume_tokens(10)
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_idempotency_key_is_stable_and_scoped() -> None:
    a = new_idempotency_key("tr_1", "aftersale.apply", "u_1", "SO1")
    b = new_idempotency_key("tr_1", "aftersale.apply", "u_1", "SO1")
    c = new_idempotency_key("tr_2", "aftersale.apply", "u_1", "SO1")
    assert a == b
    assert a != c


def test_registry_rejects_undeclared_write_tool() -> None:
    reg = ToolRegistry()
    with pytest.raises(ValueError):
        reg.register(
            ToolSpec(name="danger.delete", handler=lambda p, c: {}, kind="write", target_field="id")
        )


def test_write_tool_requires_target_field() -> None:
    reg = ToolRegistry()
    with pytest.raises(ValueError):
        reg.register(ToolSpec(name="aftersale.apply", handler=lambda p, c: {}, kind="write"))


def test_write_tool_is_idempotent() -> None:
    reg = ToolRegistry()
    calls: list[int] = []

    def handler(params: dict, ctx: ToolCallContext) -> dict:
        calls.append(1)
        return {"value": {"ok": True}}

    reg.register(
        ToolSpec(name="aftersale.apply", handler=handler, kind="write", target_field="order_no")
    )
    # trace_id 每次运行都不同：幂等键由 trace_id 派生，写死一个固定值会把
    # 上一次运行留下的记录带进来，导致「第一次执行」直接命中重放。
    ctx = _ctx("tr_idem_" + uuid.uuid4().hex)
    first = reg.execute("aftersale.apply", {"order_no": "SO1"}, ctx)
    second = reg.execute("aftersale.apply", {"order_no": "SO1"}, ctx)
    assert first.ok and second.ok
    assert len(calls) == 1  # 幂等命中，不产生第二次副作用
    assert second.data.get("idempotent_replay") is True


def test_write_tool_timeout_does_not_retry() -> None:
    from cs_agent_system.registry.tool_registry import ToolTimeout

    reg = ToolRegistry()
    attempts = {"n": 0}

    def handler(params: dict, ctx: ToolCallContext) -> dict:
        attempts["n"] += 1
        raise ToolTimeout("slow")

    reg.register(
        ToolSpec(name="aftersale.apply", handler=handler, kind="write", target_field="order_no")
    )
    result = reg.execute(
        "aftersale.apply", {"order_no": "SO1"}, _ctx("tr_timeout_" + uuid.uuid4().hex)
    )
    assert not result.ok
    assert result.status_unknown is True
    assert attempts["n"] == 1  # 写操作超时禁止自动重试


def test_read_tool_retries_with_backoff() -> None:
    from cs_agent_system.registry.tool_registry import ToolTimeout

    reg = ToolRegistry()
    attempts = {"n": 0}

    def handler(params: dict, ctx: ToolCallContext) -> dict:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ToolTimeout("flaky")
        return {"value": {"ok": True}}

    reg.register(ToolSpec(name="order.query", handler=handler, kind="read", max_retries=2))
    result = reg.execute("order.query", {}, _ctx())
    assert result.ok
    assert attempts["n"] == 2


def test_write_tool_whitelist_is_exact() -> None:
    # 撤销（aftersale.cancel）也是写操作：它会改变下游资金状态
    assert WRITE_TOOLS == frozenset(
        {"aftersale.apply", "aftersale.cancel", "coupon.issue", "order.action"}
    )


def test_session_context_require_reports_missing() -> None:
    ctx = SessionContext()
    ctx.slots = {"order_no": "SO1"}
    assert ctx.require(["order_no", "reason"]) == ["reason"]
