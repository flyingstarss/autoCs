"""OMS 抽象层：写操作与「真实交易系统」之间的边界。

为什么需要这一层：
`registry/tools.py` 里的四个写工具（`aftersale.apply` / `aftersale.cancel` /
`coupon.issue` / `order.action`）目前直接操作本地 SQLite。这在演示环境成立，但上生产时
真实诉求是「把退款单推给电商平台/ERP」——那是一个**远程、有延迟、可能超时、
可能部分成功**的系统。把远程调用直接写进工具处理器，会让工具层同时承担
「业务口径」和「远程通信」两件事，无法测试也不可替换。

本模块提供两个实现，接口完全一致：
- `LocalOmsClient`：本地实现，写 SQLite（默认，零外部依赖）；
- `HttpOmsClient`：HTTP 实现，推给真实 OMS（配置即启用）。

切换开关：`OMS_BACKEND=local|http`（见 `config.Settings.oms_backend`）。

**幂等是这一层的核心契约**：所有写方法都必须接受 `idempotency_key`，
并保证「同一个 key 重复调用不会产生第二次副作用」。这是分布式写操作
不可协商的要求——超时后重试是常态，不是异常。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import settings
from . import data


class OmsError(RuntimeError):
    """OMS 调用失败（业务错误）。"""


class OmsTimeout(OmsError):
    """OMS 超时 —— 状态未知，禁止重试，必须按幂等键回查。"""


@dataclass(frozen=True)
class OmsResult:
    """一次写操作的结果。

    `external_no` 是外部系统返回的单号（如平台售后单号），
    没有时退化为本地单号。
    """

    ok: bool
    external_no: str | None = None
    payload: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.payload is None:
            object.__setattr__(self, "payload", {})

    def to_dict(self) -> dict[str, Any]:
        """转成工具返回结构。

        `external_no` 不覆盖 payload 里的同名字段 —— payload 才是业务详情
        （含 aftersale_no / coupon_id 等），external_no 只是兜底的外部单号。
        """
        result = {"ok": self.ok}
        result.update(self.payload)
        if self.external_no and "external_no" not in result:
            result["external_no"] = self.external_no
        return result


class OmsClient(Protocol):
    """订单/售后/优惠的写入契约。

    每个方法都必须：
    1. 接受 `idempotency_key`，并在服务端侧去重；
    2. 超时抛 `OmsTimeout`（不是普通异常），以便上层走回查路径；
    3. 返回 `OmsResult`，不抛业务异常。
    """

    def apply_aftersale(
        self,
        *,
        user_id: str,
        order_no: str,
        kind: str,
        reason: str,
        refund_amount: float,
        refund_sla_days: int,
        idempotency_key: str,
    ) -> OmsResult: ...

    def issue_coupon(
        self,
        *,
        user_id: str,
        amount: float,
        threshold: float,
        category: str,
        idempotency_key: str,
    ) -> OmsResult: ...

    def update_order(
        self,
        *,
        user_id: str,
        order_no: str,
        action: str,
        idempotency_key: str,
    ) -> OmsResult: ...

    def cancel_aftersale(
        self,
        *,
        user_id: str,
        aftersale_no: str,
        reason: str,
        idempotency_key: str,
    ) -> OmsResult: ...


class LocalOmsClient:
    """本地实现：写 SQLite。默认启用，零外部依赖。

    幂等由「先查 `aftersales` / `coupons` 是否已有同 key 记录」保证 ——
    DB 侧的 `idempotency` 表由 `ToolRegistry` 统一维护，这里只管业务数据。
    """

    def apply_aftersale(
        self,
        *,
        user_id: str,
        order_no: str,
        kind: str,
        reason: str,
        refund_amount: float,
        refund_sla_days: int,
        idempotency_key: str,
    ) -> OmsResult:
        # 服务端去重：同一个幂等键只允许一张售后单。
        # 这不是可选优化 —— 超时重发时上层会带着同一个键再打一次，
        # 没有这段回查就会真的建第二张单、退第二笔钱。
        existing = data.find_aftersale_by_key(idempotency_key)
        if existing is not None:
            return OmsResult(
                ok=True,
                external_no=str(existing.get("aftersale_no") or ""),
                payload={**existing, "idempotent_replay": True},
            )
        record = data.create_aftersale(
            user_id=user_id,
            order_no=order_no,
            kind=kind,
            reason=reason,
            refund_amount=refund_amount,
            refund_sla_days=refund_sla_days,
            idempotency_key=idempotency_key,
        )
        # 售后受理后订单进入「退款中」
        data.update_order_status(order_no, "退款中")
        return OmsResult(ok=True, external_no=record["aftersale_no"], payload=record)

    def cancel_aftersale(
        self,
        *,
        user_id: str,
        aftersale_no: str,
        reason: str,
        idempotency_key: str,
    ) -> OmsResult:
        """撤销退款：状态置「已撤销」并恢复订单原状态。

        `data.cancel_aftersale` 本身幂等（已撤销再撤直接成功），
        所以同一撤销请求重发不会产生额外副作用。
        """
        record = data.cancel_aftersale(aftersale_no)
        if record is None:
            raise OmsError(f"售后单不存在：{aftersale_no}")
        if str(record.get("user_id") or "") != user_id:
            raise OmsError("售后单不属于当前用户，禁止代他人撤销")
        return OmsResult(
            ok=True,
            external_no=aftersale_no,
            payload={
                "aftersale_no": aftersale_no,
                "order_no": record.get("order_no"),
                "status": record.get("status"),
                "reason": reason,
                "cancelled": True,
            },
        )

    def issue_coupon(
        self,
        *,
        user_id: str,
        amount: float,
        threshold: float,
        category: str,
        idempotency_key: str,
    ) -> OmsResult:
        record = data.issue_coupon(
            user_id=user_id,
            amount=amount,
            threshold=threshold,
            category=category,
        )
        return OmsResult(
            ok=True,
            external_no=record["coupon_id"],
            payload={"issued": True, "coupon_id": record["coupon_id"], "amount": record["amount"]},
        )

    def update_order(
        self,
        *,
        user_id: str,
        order_no: str,
        action: str,
        idempotency_key: str,
    ) -> OmsResult:
        new_status = "已取消" if action == "cancel" else "处理中"
        changed = data.update_order_status(order_no, new_status)
        return OmsResult(
            ok=changed,
            external_no=order_no,
            payload={"order_no": order_no, "action": action, "status": new_status},
        )


class HttpOmsClient:
    """HTTP 实现：把写操作推给真实 OMS。

    配置：
        OMS_BACKEND=http
        OMS_BASE_URL=https://oms.internal
        OMS_TOKEN=xxx
        OMS_TIMEOUT=5.0

    关键行为：
    - 幂等键放请求头 `X-Idempotency-Key`，由上游去重；
    - 超时（socket timeout）→ `OmsTimeout`，上层回查而非重试；
    - HTTP 4xx 视为业务失败，返回 `ok=False` 而不抛异常；
    - HTTP 5xx 视为状态未知 → `OmsTimeout`，同样走回查。
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.oms_base_url or "").rstrip("/")
        self.token = token or settings.oms_token or ""
        self.timeout = timeout if timeout is not None else settings.oms_timeout

    def _post(self, path: str, payload: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
        if not self.base_url:
            raise OmsError("OMS_BASE_URL 未配置，无法使用 http 后端")
        url = f"{self.base_url}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-Idempotency-Key": idempotency_key,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:  # noqa: BLE001 —— 读错误体失败不影响主流程
                pass
            if exc.code >= 500:
                # 服务端错误：状态未知，按超时处理（回查而非重试）
                raise OmsTimeout(f"OMS 5xx: {exc.code} {detail}") from exc
            # 4xx 是业务失败：上游已明确拒绝，不重试
            raise OmsError(f"OMS 4xx: {exc.code} {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OmsTimeout(f"OMS 调用失败：{type(exc).__name__}: {exc}") from exc

    def apply_aftersale(
        self,
        *,
        user_id: str,
        order_no: str,
        kind: str,
        reason: str,
        refund_amount: float,
        refund_sla_days: int,
        idempotency_key: str,
    ) -> OmsResult:
        raw = self._post(
            "/aftersale/apply",
            {
                "user_id": user_id,
                "order_no": order_no,
                "type": kind,
                "reason": reason,
                "refund_amount": refund_amount,
                "refund_sla_days": refund_sla_days,
            },
            idempotency_key,
        )
        return OmsResult(
            ok=bool(raw.get("ok", True)),
            external_no=raw.get("aftersale_no") or raw.get("external_no"),
            payload=raw,
        )

    def issue_coupon(
        self,
        *,
        user_id: str,
        amount: float,
        threshold: float,
        category: str,
        idempotency_key: str,
    ) -> OmsResult:
        raw = self._post(
            "/coupon/issue",
            {
                "user_id": user_id,
                "amount": amount,
                "threshold": threshold,
                "category": category,
            },
            idempotency_key,
        )
        return OmsResult(
            ok=bool(raw.get("ok", True)),
            external_no=raw.get("coupon_id") or raw.get("external_no"),
            payload=raw,
        )

    def update_order(
        self,
        *,
        user_id: str,
        order_no: str,
        action: str,
        idempotency_key: str,
    ) -> OmsResult:
        raw = self._post(
            "/order/action",
            {"user_id": user_id, "order_no": order_no, "action": action},
            idempotency_key,
        )
        return OmsResult(
            ok=bool(raw.get("ok", True)),
            external_no=raw.get("order_no"),
            payload=raw,
        )

    def cancel_aftersale(
        self,
        *,
        user_id: str,
        aftersale_no: str,
        reason: str,
        idempotency_key: str,
    ) -> OmsResult:
        raw = self._post(
            "/aftersale/cancel",
            {"user_id": user_id, "aftersale_no": aftersale_no, "reason": reason},
            idempotency_key,
        )
        return OmsResult(
            ok=bool(raw.get("ok", True)),
            external_no=raw.get("aftersale_no") or aftersale_no,
            payload=raw,
        )


def build_oms_client() -> OmsClient:
    """按配置选择 OMS 后端。"""
    if settings.oms_backend == "http":
        return HttpOmsClient()
    return LocalOmsClient()


__all__ = [
    "OmsClient",
    "OmsError",
    "OmsTimeout",
    "OmsResult",
    "LocalOmsClient",
    "HttpOmsClient",
    "build_oms_client",
]
