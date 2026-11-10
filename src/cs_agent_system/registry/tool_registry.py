"""Tool Registry（架构文档 L4 能力层）。

四条硬规则落在代码里，而不是文档里：
1. 写操作只有白名单内的几个（`core.ids.WRITE_TOOLS`）；
   其余工具一律只读，注册时即校验，无法绕过。
2. 写操作必须携带幂等键；标记为 `business_scoped` 的写工具还要抢业务锁，
   保证「同一订单的同一版本退款」全库只有一条生效记录。
3. 写操作超时禁止自动重试（超时≠失败），改为按幂等键回查状态；
   只有读操作才允许退避重试。
4. 撤销是独立写操作（`aftersale.cancel`），不能靠删幂等记录假装没发生过。
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.ids import WRITE_TOOLS, new_business_key, new_idempotency_key
from .idempotency import (
    ACTIVE_STATUSES,
    IdempotencyStore,
    IdempotentRecord,
    build_idempotency_store,
)
from .oms import OmsError, OmsTimeout

ToolHandler = Callable[[dict[str, Any], "ToolCallContext"], dict[str, Any]]

# 只读工具重试的退避基数（秒）：第 n 次重试等待 base * 2^(n-1)。
# 此前声明了「退避重试」却没有真正退避——连续失败时会瞬间打满下游。
RETRY_BACKOFF_BASE = 0.05

# 工具执行器：给同步处理器套上超时。
# 处理器是同步函数（本地 SQLite / HTTP），Python 没有原生的同步超时，
# 只能放到线程里用 future.result(timeout=...) 掐时间。
# 超时后线程不会被强杀（Python 无此能力），但它会自行结束；
# 写操作已按「超时≠失败」语义标记 status_unknown，由幂等键回查兜底。
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="tool")



@dataclass
class ToolCallContext:
    """工具执行上下文：由编排器注入，工具本身不持有会话态。"""

    trace_id: str
    session_id: str
    user_id: str
    turn: int = 0
    channel: str = "web"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    """工具的声明式元数据。权限、幂等、超时都在这里声明。"""

    name: str
    handler: ToolHandler
    description: str = ""
    kind: str = "read"  # read / write
    required_slots: tuple[str, ...] = ()
    timeout_ms: int = 2000
    max_retries: int = 0
    # 写操作的目标实体标识字段，用于计算幂等键
    target_field: str | None = None
    # 是否启用「业务锁」：同一 (工具, 目标, 版本) 只允许一条生效记录。
    # 只对「一笔业务事实只能发生一次」的写操作开启（退款），
    # 像发券这类允许重复领取的工具不启用，避免把正常的多轮领取挡掉。
    business_scoped: bool = False
    # 写操作的人工审批阈值（金额），None 表示不需要审批。
    # 只在处理器**产生副作用之前**生效：处理器判定金额超阈值时直接返回
    # `needs_approval`，不执行写入。
    approval_amount_threshold: float | None = None

    @property
    def is_write(self) -> bool:
        return self.kind == "write"


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    idempotency_key: str | None = None
    cost_ms: int = 0
    attempts: int = 1
    # 写操作超时时置为 True：状态未知，需按幂等键回查，禁止重试
    status_unknown: bool = False
    # 是否需要人工审批
    needs_approval: bool = False
    reason: str = ""

    @property
    def value(self) -> Any:
        return self.data.get("value")


class ToolTimeout(RuntimeError):
    pass


class ToolRegistry:
    """工具注册中心。"""

    def __init__(self, idempotency_store: IdempotencyStore | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        # 幂等记录必须跨进程可查：写操作超时后靠它回查状态，
        # 只存内存的话服务一重启就失效了。
        self._store: IdempotencyStore = idempotency_store or build_idempotency_store()

    # —— 注册 ——
    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.is_write:
            if spec.name not in WRITE_TOOLS:
                raise ValueError(
                    f"工具 {spec.name} 被声明为写操作，但不在白名单 {sorted(WRITE_TOOLS)} 内。"
                    "新增写操作必须先修改 core.ids.WRITE_TOOLS 并接受幂等与审批约束。"
                )
            if not spec.target_field:
                raise ValueError(f"写工具 {spec.name} 必须声明 target_field 以生成幂等键")
        self._tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        *,
        kind: str = "read",
        description: str = "",
        required_slots: tuple[str, ...] = (),
        timeout_ms: int = 2000,
        max_retries: int = 0,
        target_field: str | None = None,
        business_scoped: bool = False,
        approval_amount_threshold: float | None = None,
    ) -> Callable[[ToolHandler], ToolHandler]:
        def decorator(fn: ToolHandler) -> ToolHandler:
            self.register(
                ToolSpec(
                    name=name,
                    handler=fn,
                    description=description,
                    kind=kind,
                    required_slots=required_slots,
                    timeout_ms=timeout_ms,
                    max_retries=max_retries,
                    target_field=target_field,
                    business_scoped=business_scoped,
                    approval_amount_threshold=approval_amount_threshold,
                )
            )
            return fn

        return decorator

    # —— 查询 ——
    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"未注册的工具：{name}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "kind": t.kind,
                "description": t.description,
                "required_slots": list(t.required_slots),
                "timeout_ms": t.timeout_ms,
                "max_retries": t.max_retries,
            }
            for t in sorted(self._tools.values(), key=lambda x: x.name)
        ]

    # —— 执行 ——
    def _invoke(
        self,
        spec: ToolSpec,
        params: dict[str, Any],
        ctx: ToolCallContext,
    ) -> dict[str, Any]:
        """在有超时的线程里执行处理器。

        `timeout_ms` 此前只是声明 —— 执行路径从不读它，工具卡死会一直
        占用预算直到墙钟兜底。这里把它变成真正的约束。

        注意：Python 无法强杀已在运行的线程，所以超时只是「放弃等待」，
        线程仍会自行跑完。写操作因此必须靠幂等键回查确认最终状态
        （见 `execute()` 的 `status_unknown` 分支）。
        """
        timeout_s = max(0.001, spec.timeout_ms / 1000.0)
        future = _EXECUTOR.submit(spec.handler, params, ctx)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()  # 尚未开始的任务可取消；已在跑的任务无法强杀
            raise ToolTimeout(f"工具 {spec.name} 超过 {spec.timeout_ms}ms 未返回") from exc

    def execute(
        self,
        name: str,
        params: dict[str, Any],
        ctx: ToolCallContext,
        *,
        allow_retry: bool = True,
    ) -> ToolResult:
        spec = self.get(name)

        missing = [s for s in spec.required_slots if params.get(s) in (None, "")]
        if missing:
            return ToolResult(ok=False, error=f"NEED_MORE_INFO: missing={','.join(missing)}", reason="missing_slots")

        idempotency_key: str | None = None
        if spec.is_write:
            target = str(params.get(spec.target_field or "", "") or "")
            version = _as_version(params.get("__version"))
            amount = _as_amount(params.get("amount", params.get("refund_amount")))
            idempotency_key = new_idempotency_key(
                ctx.trace_id,
                spec.name,
                ctx.user_id,
                target,
                version=version,
                amount=amount,
            )
            business_key = (
                new_business_key(spec.name, target, version)
                if spec.business_scoped and target
                else None
            )
            claim = self._store.claim(
                key=idempotency_key,
                tool=spec.name,
                trace_id=ctx.trace_id,
                business_key=business_key,
                order_no=target,
                version=version,
            )
            if not claim.created:
                return self._replay_or_conflict(claim.record, idempotency_key)

        attempts = 0
        max_attempts = 1 + (spec.max_retries if (allow_retry and not spec.is_write) else 0)
        started = time.perf_counter()
        last_error: str | None = None

        while attempts < max_attempts:
            attempts += 1
            try:
                # 写操作把幂等键透传给处理器：OMS 层需要它做服务端去重，
                # 远端超时后回查也靠它。只读工具不需要。
                call_params = params
                if spec.is_write and idempotency_key:
                    call_params = {**params, "__idempotency_key": idempotency_key}
                data = self._invoke(spec, call_params, ctx)
                cost_ms = int((time.perf_counter() - started) * 1000)
                # 审批由处理器在写入前判定：金额超阈值时它返回 needs_approval
                # 而不产生副作用。注册中心只透传这个信号，不再自己猜金额——
                # 此前按返回值猜金额并比对，判断发生在写入之后，闸门等于没有。
                needs_approval = False
                if spec.approval_amount_threshold is not None and isinstance(data, dict):
                    inner = data.get("value")
                    if isinstance(inner, dict):
                        needs_approval = bool(inner.get("needs_approval"))
                if spec.is_write and idempotency_key:
                    # 需审批的请求没有产生副作用，不能占用业务锁：
                    # 标记 rejected，审批通过后才能以同一版本重新提交。
                    self._store.set_status(
                        idempotency_key,
                        "rejected" if needs_approval else "executed",
                        data,
                    )
                return ToolResult(
                    ok=True,
                    data=data,
                    idempotency_key=idempotency_key,
                    cost_ms=cost_ms,
                    attempts=attempts,
                    needs_approval=needs_approval,
                    reason="approval_required" if needs_approval else "",
                )
            except ToolTimeout as exc:
                last_error = f"timeout: {exc}"
                if spec.is_write:
                    # 写操作超时：状态未知，禁止重试，交由幂等键回查
                    self._mark_uncertain(idempotency_key, last_error)
                    return ToolResult(
                        ok=False,
                        error=last_error,
                        idempotency_key=idempotency_key,
                        cost_ms=int((time.perf_counter() - started) * 1000),
                        attempts=attempts,
                        status_unknown=True,
                        reason="write_timeout_no_retry",
                    )
            except OmsTimeout as exc:
                # OMS 远端超时：同样是「状态未知」，语义与 ToolTimeout 一致
                last_error = f"oms_timeout: {exc}"
                if spec.is_write:
                    self._mark_uncertain(idempotency_key, last_error)
                    return ToolResult(
                        ok=False,
                        error=last_error,
                        idempotency_key=idempotency_key,
                        cost_ms=int((time.perf_counter() - started) * 1000),
                        attempts=attempts,
                        status_unknown=True,
                        reason="oms_timeout_no_retry",
                    )
            except OmsError as exc:
                # OMS 业务失败（4xx）：上游明确拒绝，不重试也不回查。
                # 标记 failed，释放业务锁 —— 用户可以修正后重新提交。
                last_error = f"oms_error: {exc}"
                if spec.is_write:
                    self._store.set_status(idempotency_key, "failed", {"error": last_error})
                    return ToolResult(
                        ok=False,
                        error=last_error,
                        idempotency_key=idempotency_key,
                        cost_ms=int((time.perf_counter() - started) * 1000),
                        attempts=attempts,
                        reason="oms_rejected",
                    )
            except Exception as exc:  # noqa: BLE001 —— 工具边界，允许隔离失败
                last_error = f"{type(exc).__name__}: {exc}"
                if spec.is_write:
                    # 处理器内部报错（参数非法、订单不属于本人等）：
                    # 确定为「没写成功」，释放业务锁。
                    self._store.set_status(idempotency_key, "failed", {"error": last_error})
                if not spec.is_write and attempts >= max_attempts:
                    break

            # 只读工具才退避重试：写操作一律不重试（下面直接返回或跳出）。
            if spec.is_write:
                break
            if attempts < max_attempts:
                # 退避：连续失败时给下游一点恢复时间，避免瞬间打满
                time.sleep(RETRY_BACKOFF_BASE * (2 ** (attempts - 1)))

        return ToolResult(
            ok=False,
            error=last_error or "unknown error",
            idempotency_key=idempotency_key,
            cost_ms=int((time.perf_counter() - started) * 1000),
            attempts=attempts,
            reason="failed",
        )

    # —— 写操作的状态处置 ——
    def _mark_uncertain(self, idempotency_key: str | None, error: str) -> None:
        """超时 = 状态未知。占用业务锁，禁止重发，只能按 key 回查。"""
        if idempotency_key:
            self._store.set_status(idempotency_key, "uncertain", {"error": error})

    def _replay_or_conflict(self, record: IdempotentRecord, request_key: str) -> ToolResult:
        """已有记录时不产生第二次副作用。

        四种情形必须分开处理，混为一谈就会要么重复写、要么把未知当失败：
        - `pending`（未超时）：另一路正在写，不能并发再写一遍；
        - `uncertain` / 过期 pending：状态未知，交给调用方按 key 回查；
        - `executed`：回放首次结果（同键=重发，异键=同一业务事实的重复请求）；
        - `rejected/failed/voided`：不产生过副作用，允许重新提交。
        """
        if record.status == "pending" and not record.is_stale_pending:
            return ToolResult(
                ok=False,
                error=f"WRITE_IN_FLIGHT: {record.key}",
                idempotency_key=request_key,
                reason="business_in_flight",
            )
        if record.status == "uncertain" or record.is_stale_pending:
            return ToolResult(
                ok=False,
                error="WRITE_STATUS_UNKNOWN",
                idempotency_key=record.key,
                status_unknown=True,
                reason="status_unknown",
            )
        if record.status in ("rejected", "failed", "voided"):
            return ToolResult(
                ok=False,
                error=f"WRITE_{record.status.upper()}",
                idempotency_key=record.key,
                reason=record.status,
            )
        data = dict(record.result)
        data["idempotent_replay"] = True
        return ToolResult(
            ok=True,
            data=data,
            idempotency_key=record.key,
            attempts=1,
            reason="idempotent_replay" if record.key == request_key else "business_replay",
        )

    def void(self, idempotency_key: str, result: dict[str, Any] | None = None) -> None:
        """作废一条已执行的记录。

        只改本地状态，**不代表下游已经退钱**。调用方必须先在 OMS 侧
        真正撤销（`aftersale.cancel`），拿到成功回执后再调用本方法。
        """
        self._store.set_status(idempotency_key, "voided", result)

    def active_records(self, order_no: str, tool: str) -> list[IdempotentRecord]:
        """某订单某个工具下所有仍然生效的记录（可被撤销的候选）。"""
        return [
            r for r in self._store.list_by_order(order_no, tool) if r.status in ACTIVE_STATUSES
        ]

    def next_version(self, tool: str, order_no: str) -> int:
        """下一个业务版本号：同一订单同一工具历史最大版本 + 1。"""
        return self._store.max_version(tool, order_no) + 1

    # —— 幂等状态回查 ——
    def lookup_idempotent(self, idempotency_key: str) -> dict[str, Any] | None:
        """按幂等键回查执行状态（写操作超时后的正确处置方式）。

        返回 `{"data": {...}}` 形态，与旧的内存实现保持一致，
        避免调用方（`aftersale.status` 等工具）感知存储后端变化。
        """
        record = self._store.get(idempotency_key)
        if record is None:
            return None
        return {"data": record.result, "tool": record.tool, "executed_at": record.executed_at}

    # —— MCP 工具接入 ——
    def register_mcp_tools(self, transport: str = "stdio") -> list[str]:
        """把 MCP Server 提供的工具注册进本注册中心。

        MCP 工具名与进程内工具名一一对应（如 `order.query` ↔ `order_query`），
        注册时保留本地的超时/重试/权限声明 —— MCP 只负责执行，
        治理规则仍由本注册中心统一把控，避免「接了 MCP 就绕过约束」。

        返回实际注册的工具名列表。
        """
        from ..mcp import client as mcp_client

        specs = mcp_client.list_tools_sync(transport)
        registered: list[str] = []
        for spec in specs:
            mcp_name = spec["name"]
            local_name = mcp_name.replace("_", ".")
            if self.has(local_name):
                # 本地已有同名工具：不覆盖（进程内版本更快且无外部依赖）
                continue
            self.register(
                ToolSpec(
                    name=local_name,
                    handler=_make_mcp_handler(mcp_name, transport),
                    description=spec.get("description", ""),
                    kind="read",
                    timeout_ms=5000,
                    max_retries=1,
                )
            )
            registered.append(local_name)
        return registered


def _as_version(raw: Any) -> int:
    """业务版本号：非法值一律回落 1，绝不因为脏参数把锁放开。"""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1
    return value if value >= 1 else 1


def _as_amount(raw: Any) -> float | None:
    """金额：只接受数字；字符串不猜，避免「300 变成 0」混进幂等键。"""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _make_mcp_handler(mcp_name: str, transport: str) -> ToolHandler:
    """构造调用 MCP 工具的处理器。

    MCP 工具统一按只读处理：写操作必须走本地的写工具（带幂等键与审批），
    不允许通过 MCP 通道绕过 `WRITE_TOOLS` 白名单。
    """

    def handler(params: dict[str, Any], ctx: ToolCallContext) -> dict[str, Any]:
        from ..mcp import client as mcp_client

        try:
            result = mcp_client.call_tool_sync(mcp_name, dict(params), transport)
        except Exception as exc:  # noqa: BLE001 —— MCP 故障按工具失败处理，不静默返回假数据
            raise ToolTimeout(f"MCP 调用失败：{type(exc).__name__}: {exc}") from exc
        if result.get("found") is False:
            # 查无此项：返回空值而非报错，语义与进程内工具保持一致
            return {"value": None, "found": False}
        return {"value": result, "found": True}

    return handler
