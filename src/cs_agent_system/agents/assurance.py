"""保障域：Q1 知识检索 / Q2 事实校验 / Q3 合规检查 / Q4 质量评估 / Q5 人工接管。

保障层对产出有否决权——不是旁路检查，而是编排器里的必经节点。
"""

from __future__ import annotations

from typing import Any

from ..core.confidence import (
    EMPTY,
    ERROR,
    EXACT,
    HANDOFF,
    PARTIAL,
    POLICY,
    REJECTED,
    RETRIEVAL,
    VERIFICATION,
    resolve,
)
from ..core.envelope import AgentEnvelope, SessionContext
from .base import Unit


class KnowledgeRetriever(Unit):
    """Q1 知识检索单元：混合检索 + 重排，返回带引用的证据片段。

    单元性质：确定性检索，零模型调用。
    """

    unit_id = "Q1"
    domain = "assurance"
    description = "知识检索"
    kind = "capability"

    # 各意图的检索关键词模板：意图 + 槽位拼成查询
    QUERY_TEMPLATES: dict[str, str] = {
        "RETURN_APPLY": "退货 换货 退款 政策 规则",
        "COUPON_USE": "优惠券 叠加 满减 活动 规则",
        "PRODUCT_QA": "商品 参数 材质 介绍",
        "PAYMENT_INVOICE": "发票 开票 抬头 规则",
        "LOGISTICS": "物流 配送 异常 超时",
    }

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        intent = env.intent or session.intent.primary
        query = str(env.payload.get("query") or "")
        if not query:
            # 用意图模板 + 槽位拼查询，避免纯关键词检索召回不足
            template = self.QUERY_TEMPLATES.get(intent, intent)
            parts = [template]
            for key in ("sku_id", "reason", "type", "category"):
                if session.slots.get(key):
                    parts.append(str(session.slots[key]))
            query = " ".join(parts)
        layers = self._layers_for(intent)
        evidence = self.rt.retriever.search(query, top_k=3, layers=layers)
        out = env.child(self.unit_id, confidence=resolve(EXACT if evidence else EMPTY, RETRIEVAL))
        out.evidence = evidence
        out.payload = {
            "query": query,
            "evidence": [e.model_dump() for e in evidence],
            "layers": list(layers),
            "hit": bool(evidence),
        }
        if not evidence:
            out.degraded = True
        return out

    @staticmethod
    def _layers_for(intent: str) -> tuple[str, ...]:
        if intent in ("RETURN_APPLY", "COUPON_USE", "PAYMENT_INVOICE"):
            return ("policy", "faq", "script")
        if intent == "PRODUCT_QA":
            return ("product", "script", "faq")
        if intent == "LOGISTICS":
            return ("faq", "policy")
        return ("policy", "faq", "script")


class FactChecker(Unit):
    """Q2 事实校验单元：逐句比对生成内容与证据/业务数据，标记幻觉。

    单元性质：**守卫**——有否决权，命中即阻断。
    无证据的数字断言一律判为无依据；这是反思循环的回边触发条件。
    """

    unit_id = "Q2"
    domain = "assurance"
    description = "事实校验"
    kind = "guard"

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        text = str(env.payload.get("text", ""))
        evidence = env.evidence
        facts: dict[str, Any] = env.payload.get("facts") or {}
        result = self.check_facts(text, evidence=evidence, facts=facts)
        out = env.child(self.unit_id, confidence=resolve(EXACT if result.passed else REJECTED, VERIFICATION))
        out.payload = result.to_dict()
        # 首次不通过由编排器走反思循环；连续失败才转人工，因此此处不主动请求接管
        self.rt.log(self.unit_id, "verdict", result.reason, verdict="pass" if result.passed else "fail")
        return out


class ComplianceGuard(Unit):
    """Q3 合规检查单元：敏感词、绝对化用语、虚假承诺、诱导话术。

    单元性质：**守卫**——独立否决权，不可与被检查者合并。
    阻断级问题直接否决；可改写级问题改写后放行。
    """

    unit_id = "Q3"
    domain = "assurance"
    description = "合规检查"
    kind = "guard"

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        text = str(env.payload.get("text", ""))
        result = self.check_compliance(text)
        out = env.child(self.unit_id, confidence=resolve(EXACT if result.passed else REJECTED, POLICY))
        out.payload = result.to_dict()
        if not result.passed:
            out.need_handoff = True
        if result.rewritten_text:
            out.payload["text"] = result.rewritten_text
        self.rt.log(self.unit_id, "verdict", "; ".join(i.message for i in result.issues), verdict="pass" if result.passed else "block")
        return out


class QualityEvaluator(Unit):
    """Q4 质量评估单元：打分、意图命中与解决率判定，驱动回流。

    单元性质：确定性评分，零模型调用。
    """

    unit_id = "Q4"
    domain = "assurance"
    description = "质量评估"
    kind = "capability"

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        text = str(env.payload.get("text", ""))
        facts = env.payload.get("facts") or {}
        has_facts = bool(facts)
        resolved = bool(env.payload.get("resolved", True)) and has_facts
        score = 0.0
        if has_facts:
            score += 0.6
        if text and len(text) >= 8:
            score += 0.2
        if not env.need_handoff:
            score += 0.2
        score = round(min(1.0, score), 3)
        # 质量分与置信度是两件事：score 是「本次回答好不好」的评估结果，
        # confidence 是「本次评估本身可不可信」。此处只说有没有事实可评，
        # 不把 score 塞进 confidence（此前两者共用一个字段，语义被混淆）。
        if not has_facts:
            outcome = EMPTY
        elif resolved:
            outcome = EXACT
        else:
            outcome = PARTIAL
        out = env.child(self.unit_id, confidence=resolve(outcome, VERIFICATION))
        out.payload = {
            "score": score,
            "resolved": resolved,
            "failure_class": None if resolved else "knowledge_missing" if not has_facts else "generation",
            "intent_hit": True,
        }
        self.rt.log(self.unit_id, "score", f"quality={score}", verdict="pass" if resolved else "fail")
        return out


class HandoffBuilder(Unit):
    """Q5 人工接管单元：生成坐席交接摘要、创建工单、控制接管阈值。

    单元性质：**守卫**——接管即终止自动应答链路。
    """

    unit_id = "Q5"
    domain = "assurance"
    description = "人工接管"
    kind = "guard"

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        reason = str(env.payload.get("reason") or "触发人工闸门").strip()
        orders = session.recent_orders[:3]
        summary = (
            f"用户 {session.user.id}（{session.user.member_level}）"
            f"意图 {session.intent.primary}，"
            f"槽位 {session.slots or '无'}，"
            f"风险等级 {session.emotion.risk_level}。"
            f"转接原因：{reason}。"
        )
        ticket_id = f"TK-{env.trace_id[-8:]}"
        out = env.child(self.unit_id, confidence=resolve(EXACT, HANDOFF))
        out.need_handoff = True
        out.payload = {
            "ticket_id": ticket_id,
            "summary": summary,
            "reason": reason,
            "recent_orders": orders,
            "mode": "full_takeover",
            "facts": {"ticket_id": ticket_id},
        }
        self.rt.log(self.unit_id, "handoff", reason, verdict="handoff")
        return out
