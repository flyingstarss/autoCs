"""保障层测试：合规规则与事实校验。"""

from __future__ import annotations

from cs_agent_system.core.envelope import Evidence
from cs_agent_system.guardrails.compliance import check_compliance
from cs_agent_system.guardrails.factcheck import check_facts


def test_false_promise_is_blocked() -> None:
    result = check_compliance("您的退款保证3个工作日到账")
    assert result.passed is False
    assert any(i.code == "false_promise" for i in result.issues)


def test_absolute_words_are_rewritten_not_blocked() -> None:
    result = check_compliance("这是市面上最好的材质")
    assert result.passed is True
    assert result.rewritten_text is not None
    assert "最好" not in result.rewritten_text


def test_inducement_is_blocked() -> None:
    result = check_compliance("加我微信 13800001111 私下给你便宜")
    assert result.passed is False


def test_clean_text_passes() -> None:
    result = check_compliance("您的订单已发出，物流信息稍后更新。")
    assert result.passed is True
    assert result.issues == []


def test_numbers_without_evidence_fail_fact_check() -> None:
    result = check_facts("退款金额 189 元，预计 3 天到账。")
    assert result.passed is False
    assert len(result.unsupported) == 1


def test_numbers_backed_by_facts_pass_fact_check() -> None:
    result = check_facts("退款金额 189 元，预计 3 天到账。", facts={"amount": 189, "sla": 3})
    assert result.passed is True


def test_order_id_is_not_treated_as_fact_claim() -> None:
    evidence = [Evidence(source="订单", snippet="订单号 SO202610280001")]
    result = check_facts("您的订单 SO202610280001 已发出。", evidence=evidence)
    assert result.passed is True


def test_evidence_covers_number() -> None:
    evidence = [Evidence(source="政策", snippet="退款到账时效为 3 个工作日")]
    result = check_facts("预计 3 天到账", evidence=evidence)
    assert result.passed is True
