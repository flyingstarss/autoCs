"""退款改额：先撤旧单，再下新单。

这个模块解决的是「幂等键解决不了的那一半问题」：
幂等键只能保证「同一个请求不执行两次」，管不住「两个不同请求该不该都执行」。
用户把退款金额改掉，必然产生第二个请求，于是必须显式地把第一个请求作废。

顺序不可交换：

    撤销旧单成功  ──>  才允许提交新单
         │
         └── 撤销失败 / 状态未知 ──> 中止，转人工（绝不连下两笔）

「作废」分两层，缺一不可：
1. 下游作废：调 `aftersale.cancel`，让支付/OMS 侧的那笔真的失效；
2. 本地作废：`ToolRegistry.void(key)` 把幂等记录置 `voided`，释放业务锁。
只做第 2 步等于自欺欺人 —— 钱可能已经退出去了，本地删记录不会让它回来。

金额由服务端重算（`min(订单实付, 策略上限)`），**不接受用户指定金额**：
用户能决定的是「要不要改」，不能决定「改多少」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.intents import is_revision_request
from .tool_registry import ToolCallContext, ToolRegistry, ToolResult


@dataclass
class RevisionOutcome:
    """一次改额的结果。"""

    ok: bool
    reason: str = ""
    voided: list[str] = field(default_factory=list)
    apply: ToolResult | None = None
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "voided": list(self.voided),
            "version": self.version,
        }


class RefundRevisionService:
    """售后改额的「先撤后建」流程。"""

    TOOL = "aftersale.apply"

    def __init__(self, reg: ToolRegistry) -> None:
        self._reg = reg

    @staticmethod
    def wants_revision(text: str) -> bool:
        """用户是否在要求修改已有的退款诉求。"""
        return is_revision_request(text)

    def has_active(self, order_no: str) -> bool:
        """该订单是否已有生效中的退款申请。"""
        return bool(order_no) and bool(self._reg.active_records(order_no, self.TOOL))

    def previous_reason(self, order_no: str) -> str:
        """沿用上一版申请的原因。

        用户说「把金额改一下」时不会重复讲一遍原因，
        这不是缺槽位，而是指代上一轮。
        """
        active = self._reg.active_records(order_no, self.TOOL)
        for record in sorted(active, key=lambda r: r.version, reverse=True):
            value = (record.result or {}).get("value")
            if isinstance(value, dict) and value.get("reason"):
                return str(value["reason"])
            if record.result.get("reason"):
                return str(record.result["reason"])
        return ""

    def revise(
        self,
        *,
        order_no: str,
        reason: str,
        ctx: ToolCallContext,
        extra: dict[str, Any] | None = None,
    ) -> RevisionOutcome:
        """撤销该订单所有生效中的退款，再以新版本重新提交。

        `extra` 是重新提交时要沿用的槽位（如售后类型），不含金额。
        """
        active = self._reg.active_records(order_no, self.TOOL)
        if not active:
            return RevisionOutcome(ok=False, reason="no_active_refund")

        voided: list[str] = []
        for record in active:
            if record.status == "pending" and not record.is_stale_pending:
                # 旧单还在写：此时撤销会与写入竞争，必须等它落定
                return RevisionOutcome(ok=False, reason="old_refund_in_flight", voided=voided)
            aftersale_no = _aftersale_no(record.result)
            if record.status == "uncertain" or record.is_stale_pending:
                # 状态未知：先回查。确认不了就不动，宁可转人工也不连下两笔
                aftersale_no = aftersale_no or self._confirm(record.key)
                if not aftersale_no:
                    return RevisionOutcome(
                        ok=False, reason="old_refund_unconfirmed", voided=voided
                    )
            if not aftersale_no:
                return RevisionOutcome(ok=False, reason="old_refund_no_number", voided=voided)

            cancel = self._reg.execute(
                "aftersale.cancel",
                {
                    "order_no": order_no,
                    "aftersale_no": aftersale_no,
                    "reason": f"金额变更，撤销原申请（{reason or '用户修改退款金额'}）",
                },
                ctx,
            )
            if not cancel.ok:
                # 撤销失败绝不能继续下新单 —— 否则就是两笔
                return RevisionOutcome(
                    ok=False,
                    reason=f"cancel_failed: {cancel.error or cancel.reason}",
                    voided=voided,
                )
            self._reg.void(record.key, {"voided_by": "refund_revision", "aftersale_no": aftersale_no})
            voided.append(aftersale_no)

        version = self._reg.next_version(self.TOOL, order_no)
        params: dict[str, Any] = {
            "order_no": order_no,
            "reason": reason,
            "__version": version,
        }
        for key, value in (extra or {}).items():
            if key not in params and value not in (None, ""):
                params[key] = value
        applied = self._reg.execute("aftersale.apply", params, ctx)
        return RevisionOutcome(
            ok=applied.ok,
            reason="" if applied.ok else (applied.error or applied.reason or "apply_failed"),
            voided=voided,
            apply=applied,
            version=version,
        )

    def _confirm(self, idempotency_key: str) -> str:
        """按幂等键回查旧单是否真的提交成功；查不到返回空串。"""
        record = self._reg.lookup_idempotent(idempotency_key)
        if not record:
            return ""
        return _aftersale_no(record.get("data") or {})


def _aftersale_no(result: dict[str, Any]) -> str:
    """从工具返回结构里取售后单号（兼容 `{value: {...}}` 与扁平两种形态）。"""
    value = result.get("value")
    if isinstance(value, dict):
        return str(value.get("aftersale_no") or "")
    return str(result.get("aftersale_no") or "")


__all__ = ["RefundRevisionService", "RevisionOutcome"]
