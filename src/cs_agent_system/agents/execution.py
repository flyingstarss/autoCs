"""执行域：能力单元与写操作单元。

术语（此前把这里全部称作「智能体」是不准确的）：
- `kind="capability"`：E1/E2/E4/E5/E6/E8 —— 零模型调用的确定性能力，
  只做「取槽位 → 调工具 → 投影事实」，事实投影规则集中在 `registry.facts`；
- `kind="writer"`：E3/E7 —— 写操作，必须带幂等键，超时禁止重试；
- `kind="computed"`：E9 —— 规则型权益计算，跨会员表与策略表。

原则：执行域做确定性工具调用，模型只负责解释与呈现。
事务型任务（写操作）由工具执行，模型不参与决策。
"""

from __future__ import annotations

from typing import Any

from ..core.confidence import COMPUTED, ERROR, EXACT, MISSING, PARTIAL, REJECTED, resolve
from ..core.envelope import AgentEnvelope, SessionContext
from ..registry.data import PolicyService
from ..registry.facts import PROJECTION_TIERS, PROJECTIONS, ProjectionContext
from ..registry.refund_revision import RefundRevisionService
from ..registry.tool_registry import ToolCallContext
from .base import Unit
# 状态筛选词表只在感知层维护一份：执行层复用，避免两处词表漂移
# （用户说「取消的」在感知层被识别出来，执行层却按另一套词表筛，就会答非所问）。
from .perception import detect_status_filter, wants_collection


class _ProjectedCapability(Unit):
    """能力单元基类：取槽位 → 调工具 → 投影事实。

    事实投影规则集中在 `registry.facts`，本基类只负责编排这三步，
    保证工具层、投影层、表现层三者互不知情。
    """

    kind = "capability"
    tool_name: str = ""
    projection_key: str = ""

    def build_params(self, env: AgentEnvelope, session: SessionContext) -> dict[str, Any]:
        """从槽位与上下文组装工具参数（子类按需覆盖）。"""
        return {}

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        params = self.build_params(env, session)
        result = self.call_tool(self.tool_name, params, session, env)
        projection = PROJECTIONS[self.projection_key](
            result.data.get("value"),
            ProjectionContext(
                ok=result.ok,
                error=result.error,
                session=session,
                env=env,
                # 工具返回的完整数据：投影层据此判断「列表是否被筛过」等上下文
                tool_data=dict(result.data or {}),
            ),
        )
        tier = PROJECTION_TIERS[self.projection_key]
        out = env.child(self.unit_id, confidence=resolve(projection.outcome, tier))
        payload: dict[str, Any] = dict(projection.extra)
        payload["facts"] = projection.facts
        # 命中具体商品（E4/E5）→ 把 sku 回写给编排器写进会话槽位。
        # 用户不会每轮都报一遍型号：上一句问「蓝牙耳机」，下一句只问「库存是多少」，
        # 不把这个指代记下来，下一轮工具层只能拿到空参数，答非所问。
        item = projection.extra.get("product") or projection.extra.get("inventory") or {}
        sku_id = str((item or {}).get("sku_id") or "")
        if sku_id:
            payload["slots_patch"] = {"sku_id": sku_id}
        out.payload = payload
        if projection.need_handoff:
            out.need_handoff = True
        if projection.outcome == ERROR:
            # 工具报错：保留错误原文，便于上层区分「缺参数」与「服务异常」
            out.payload["error"] = projection.extra.get("error")
        return out


class OrderCapability(_ProjectedCapability):
    """E1 订单查询能力。

    两种模式由 `params` 区分：
    1. **指定单号**：查那一单。查不到就明确回答「未查询到属于您的订单」，
       绝不能回退到「最近一单」—— 那等于回答了另一个问题。
    2. **未指定单号**：列出该用户的订单（可按状态筛选），
       话术由 `registry/facts.py::project_order` 按用户诉求渲染。
    """

    unit_id = "E1"
    domain = "execution"
    description = "订单查询"
    tool_name = "order.query"
    projection_key = "E1"

    def build_params(self, env: AgentEnvelope, session: SessionContext) -> dict[str, Any]:
        params: dict[str, Any] = {"order_no": session.slots.get("order_no")}
        # 本轮用户明确口述的订单号（可能是核验失败的）：
        # 必须传给工具层，让它能区分「查不到」与「没指定」。
        explicit = str(session.explicit_order_no or "")
        if explicit:
            params["explicit_order_no"] = explicit
        # 用户本轮的筛选诉求（如「没发货的」「已取消的」）：决定列表要返回什么
        params["status_filter"] = detect_status_filter(str(env.payload.get("text", "")))
        # 用户本轮是不是在问「一批订单」（「我有哪些订单」）：
        # 是的话投影层会把整份清单渲染出来，而不是只回最近一单。
        params["collection"] = wants_collection(str(env.payload.get("text", "")))
        return params


class LogisticsCapability(_ProjectedCapability):
    """E2 物流跟踪能力。"""

    unit_id = "E2"
    domain = "execution"
    description = "物流跟踪"
    tool_name = "logistics.track"
    projection_key = "E2"

    def build_params(self, env: AgentEnvelope, session: SessionContext) -> dict[str, Any]:
        return {"order_no": session.slots.get("order_no")}


class AfterSaleWriter(Unit):
    """E3 售后写操作单元：必须带幂等键，超时禁止重试。

    政策承诺值（到账时效、赔付上限）从策略服务读取，不自行推算。
    """

    unit_id = "E3"
    domain = "execution"
    description = "售后处理"
    kind = "writer"

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        params = {
            "order_no": session.slots.get("order_no"),
            "reason": session.slots.get("reason"),
            "type": session.slots.get("type") or "退货退款",
        }
        # 改额走「先撤后建」：这是幂等键覆盖不到的部分，
        # 必须显式撤销旧单，否则下游会出现两笔退款。
        # 这一步要放在「槽位缺失」检查之前 —— 用户说「改金额」时通常不会重复说原因，
        # 沿用原申请的原因即可，不该因为槽位被判定失效就退化成反问。
        order_no = str(params.get("order_no") or "")
        revision = RefundRevisionService(self.rt.tools)
        text = str(env.payload.get("text") or "")
        if order_no and revision.wants_revision(text) and revision.has_active(order_no):
            params = dict(params)
            params["reason"] = params.get("reason") or revision.previous_reason(order_no)
            return self._revise(revision, params, session, env)

        if not params.get("order_no") or not params.get("reason"):
            out = env.child(self.unit_id, confidence=resolve(MISSING, COMPUTED))
            out.payload = {"need_more_info": True, "missing": [k for k in ("order_no", "reason") if not params.get(k)]}
            return out

        result = self.call_tool("aftersale.apply", params, session, env)
        out = env.child(self.unit_id, confidence=resolve(ERROR, COMPUTED))

        if result.status_unknown:
            # 超时≠失败：按幂等键回查状态，而不是重试
            lookup = self.call_tool("aftersale.status", {"idempotency_key": result.idempotency_key}, session, env)
            executed = bool(lookup.data.get("executed")) if lookup.ok else False
            # 已确认执行 = 完整命中；状态仍未知 = 部分命中（宁可转人工也不重复提交）
            out.confidence = resolve(EXACT if executed else PARTIAL, COMPUTED)
            out.payload = {
                "status_unknown": True,
                "idempotency_key": result.idempotency_key,
                "executed": executed,
                "message": "申请已提交，正在确认结果" if executed else "申请状态待确认，稍后为您核实",
            }
            out.need_handoff = not executed
            return out

        if not result.ok:
            out.payload = {"error": result.error}
            out.need_handoff = True
            return out

        data = result.data.get("value") or {}
        facts = {
            "type": data.get("type"),
            "aftersale_no": data.get("aftersale_no"),
            "refund_amount": data.get("refund_amount"),
            "refund_sla_days": data.get("refund_sla_days"),
        }
        # 关键承诺值齐全才算完整命中，否则话术里的时效/金额会缺项
        complete = all(v not in (None, "") for v in facts.values())
        out.confidence = resolve(EXACT if complete else PARTIAL, COMPUTED)
        out.payload = {
            "aftersale": data,
            "facts": facts,
            "needs_approval": result.needs_approval,
            "idempotency_key": result.idempotency_key,
        }
        if result.needs_approval:
            out.need_handoff = True
        return out

    def _revise(
        self,
        revision: RefundRevisionService,
        params: dict[str, Any],
        session: SessionContext,
        env: AgentEnvelope,
    ) -> AgentEnvelope:
        """金额变更：撤销旧退款 → 以新版本重新提交。

        撤销失败或旧单状态未知时一律转人工，绝不「先下新单再说」。
        """
        ctx = ToolCallContext(
            trace_id=env.trace_id,
            session_id=env.session_id,
            user_id=session.user.id,
            turn=env.turn,
            channel=session.channel.type,
        )
        outcome = revision.revise(
            order_no=str(params["order_no"]),
            reason=str(params.get("reason") or ""),
            ctx=ctx,
            extra={"type": params.get("type")},
        )
        out = env.child(self.unit_id, confidence=resolve(EXACT if outcome.ok else ERROR, COMPUTED))
        out.payload = {
            "revised": True,
            "voided": outcome.voided,
            "version": outcome.version,
            "reason": outcome.reason,
        }
        applied = outcome.apply
        if not outcome.ok or applied is None:
            out.need_handoff = True
            out.payload["message"] = "原退款申请未能撤销，已转人工核实；不会重复退款"
            return out

        data = applied.data.get("value") or {}
        facts = {
            "type": data.get("type"),
            "aftersale_no": data.get("aftersale_no"),
            "refund_amount": data.get("refund_amount"),
            "refund_sla_days": data.get("refund_sla_days"),
        }
        complete = all(v not in (None, "") for v in facts.values())
        out.confidence = resolve(EXACT if complete else PARTIAL, COMPUTED)
        out.payload.update(
            {
                "aftersale": data,
                "facts": facts,
                "needs_approval": applied.needs_approval,
                "idempotency_key": applied.idempotency_key,
            }
        )
        if applied.needs_approval:
            out.need_handoff = True
        return out


class InventoryCapability(_ProjectedCapability):
    """E4 库存与规格能力。"""

    unit_id = "E4"
    domain = "execution"
    description = "库存与规格"
    tool_name = "inventory.check"
    projection_key = "E4"

    def build_params(self, env: AgentEnvelope, session: SessionContext) -> dict[str, Any]:
        return {
            "sku_id": session.slots.get("sku_id"),
            "size": session.slots.get("size"),
            "color": session.slots.get("color"),
            "text": env.payload.get("text", ""),
        }


class ProductCapability(_ProjectedCapability):
    """E5 商品咨询能力：详情 + 参数 + 导购，事实来自工具。"""

    unit_id = "E5"
    domain = "execution"
    description = "商品咨询"
    tool_name = "product.detail"
    projection_key = "E5"

    def build_params(self, env: AgentEnvelope, session: SessionContext) -> dict[str, Any]:
        return {"sku_id": session.slots.get("sku_id"), "text": env.payload.get("text", "")}


class PaymentCapability(_ProjectedCapability):
    """E6 支付与开票能力。"""

    unit_id = "E6"
    domain = "execution"
    description = "支付与开票"
    tool_name = "payment.query"
    projection_key = "E6"

    def build_params(self, env: AgentEnvelope, session: SessionContext) -> dict[str, Any]:
        return {"order_no": session.slots.get("order_no")}


class CouponWriter(Unit):
    """E7 优惠与活动写操作单元：发放必须带幂等键。

    叠加规则与上限从策略服务读取；到手价由服务端 promotion.calc 计算。
    """

    unit_id = "E7"
    domain = "execution"
    description = "优惠与活动"
    kind = "writer"

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        policy = PolicyService()
        max_stack = policy.require("max_stack")
        stackable = policy.get("stackable_categories", ())

        params = {
            "sku_id": session.slots.get("sku_id") or "SKU-10086",
            "amount": session.slots.get("amount") or 20.0,
            "threshold": session.slots.get("threshold") or 99.0,
            "category": session.slots.get("category") or "服饰",
        }
        result = self.call_tool("coupon.issue", params, session, env)
        out = env.child(self.unit_id, confidence=resolve(ERROR, COMPUTED))

        if result.status_unknown:
            lookup = self.call_tool("coupon.status", {"idempotency_key": result.idempotency_key}, session, env)
            executed = bool(lookup.data.get("executed")) if lookup.ok else False
            out.confidence = resolve(EXACT if executed else PARTIAL, COMPUTED)
            out.payload = {
                "status_unknown": True,
                "idempotency_key": result.idempotency_key,
                "executed": executed,
            }
            out.need_handoff = not executed
            return out

        if not result.ok:
            out.payload = {"error": result.error}
            out.need_handoff = True
            return out

        data = result.data.get("value") or {}
        if not data.get("issued"):
            # 未发放成功：结论明确（发不了），不是查询失败
            out.confidence = resolve(REJECTED, COMPUTED)
            out.payload = {"issued": False, "reason": data.get("reason"), "facts": {}}
            return out

        # 到手价由服务端计算，模型不参与运算
        calc = self.call_tool(
            "promotion.calc",
            {"sku_id": params["sku_id"], "coupon_id": data.get("coupon_id")},
            session,
            env,
        )
        price_facts = calc.data.get("value") if calc.ok else None
        stack_note = f"本单最多可用 {max_stack} 张"
        if stackable:
            stack_note += f"，支持叠加类目：{'、'.join(stackable)}"
        # 券面额与门槛齐全才算完整命中；到手价算不出来时降级为部分命中
        complete = (
            data.get("coupon_id") not in (None, "")
            and data.get("amount") not in (None, "")
            and price_facts is not None
        )
        out.confidence = resolve(EXACT if complete else PARTIAL, COMPUTED)
        out.payload = {
            "issued": True,
            "coupon": data,
            "pricing": price_facts,
            "policy": {"max_stack": max_stack, "stackable_categories": list(stackable)},
            "facts": {
                "coupon_id": data.get("coupon_id"),
                "amount": data.get("amount"),
                "threshold": params["threshold"],
                "stack_note": stack_note,
            },
        }
        return out


class MemberCapability(_ProjectedCapability):
    """E8 会员身份能力。"""

    unit_id = "E8"
    domain = "execution"
    description = "会员身份"
    tool_name = "member.profile"
    projection_key = "E8"


class BenefitCalculator(Unit):
    """E9 会员权益计算单元：规则型，权益值来自策略服务。"""

    unit_id = "E9"
    domain = "execution"
    description = "会员权益计算"
    kind = "computed"

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        member_result = self.call_tool("member.profile", {}, session, env)
        member = member_result.data.get("value") if member_result.ok else {}
        member = member or {}
        level = member.get("level", "普通会员")

        policy = PolicyService()
        discounts = policy.require("member_discount")
        rate = discounts.get(level, 1.0)
        point_rule = policy.require("point_rule")

        # 规则计算本身是确定性的：会员信息查不到就降级到默认档，结论仍然可靠；
        # 但「查不到」和「查到了」不该是同一个置信度，故会员信息缺失时记为部分命中。
        degraded_member = not member_result.ok or not member
        out = env.child(
            self.unit_id,
            confidence=resolve(PARTIAL if degraded_member else EXACT, COMPUTED),
        )
        out.payload = {
            "benefit": {"level": level, "discount_rate": rate, "point_rule": point_rule},
            "facts": {
                "level": level,
                "points": member.get("points", 0),
                "member_note": f"{int(round((1 - rate) * 100))}% 折扣",
            },
        }
        return out
