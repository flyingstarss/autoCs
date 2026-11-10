"""处理单元基类与运行时。

每个处理单元有单一职责、明确输入输出契约，必须给出置信度与耗时。
基类统一处理：耗时统计、置信度兜底、异常隔离、事件记录。

术语：`Unit` 是「处理单元」，全系统 21 个，其中仅 4 个 `uses_llm=True`。
把 21 个单元统称「智能体」会掩盖真实成本——单元数 ≠ 模型调用次数。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.budget import BudgetExceeded, BudgetGuard
from ..core.confidence import UNKNOWN, VERIFICATION, resolve
from ..core.envelope import AgentEnvelope, SessionContext, TraceEvent
from ..core.llm import LLMGateway
from ..core.state import SessionState
from ..guardrails.compliance import check_compliance
from ..guardrails.factcheck import check_facts
from ..knowledge.retriever import HybridRetriever
from ..registry.tool_registry import ToolCallContext, ToolRegistry, ToolResult


@dataclass
class AgentRuntime:
    """运行时装配：所有共享依赖集中在此，便于替换与测试。"""

    llm: LLMGateway
    tools: ToolRegistry
    retriever: HybridRetriever
    state: SessionState
    budget: BudgetGuard
    # 长任务栈存储：SQLite 持久化，进程重启后未完成任务仍在。
    # 编排器负责维护（创建/合并/完成），感知层只读。
    tasks: Any = None
    events: list[TraceEvent] = field(default_factory=list)
    # 本轮发生的降级原因（如真实模型不可用、退回规则网关）。
    # 外部据此判断「这一轮的答案是不是模型真答的」。
    degraded_reasons: list[str] = field(default_factory=list)
    # 流式事件订阅者（SSE 接口每轮注入）。为 None 时 stream_llm 退化为普通调用，
    # 保证「有没有人订阅」不改变业务结果。
    #
    # 订阅的是**结构化事件**而不是裸字符串：
    #   {"type": "stage", "stage": "execute", "detail": "查询订单…"}
    #   {"type": "delta", "text": "已为您"}
    # 只传字符串时，编排器各节点无法告诉前端「我正在查订单」，
    # 用户只能看到一个静止的「思考中」——这正是此前流式体感缺失的根因。
    stream_sink: Callable[[dict[str, Any]], None] | None = None

    def emit_event(self, event: dict[str, Any]) -> None:
        """向流式订阅者推送一条事件；没有订阅者时静默丢弃。

        订阅者异常（如已断开的 SSE 连接）只丢这一条事件，不影响业务。
        """
        sink = self.stream_sink
        if sink is None:
            return
        try:
            sink(event)
        except Exception:  # noqa: BLE001 —— 订阅者断开不应中断编排
            pass

    def stage(self, stage: str, detail: str = "") -> None:
        """推送一条阶段进度事件，让前端在文字产出前就能看到流水线位置。"""
        self.emit_event({"type": "stage", "stage": stage, "detail": detail})

    def call_llm(
        self,
        *,
        unit_id: str,
        task: str,
        prompt: str,
        tier: Any = None,
        context: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> Any:
        """调用模型并**统一记账**。

        为什么必须走这个入口：token 记账如果散落在各单元的 `handle()` 里，
        必然漏记 —— 历史上 P1/P2/P3 三处模型调用就没记账，
        导致 `max_tokens` 约束形同虚设（只有 D3 记了）。

        这里做三件事：
        1. 调用前预检：预算已耗尽则直接抛 `BudgetExceeded`，不浪费一次调用；
        2. 调用成功后按网关返回的 usage 记实际消耗的 token；
        3. 记一条 trace 事件（含耗时与 token），便于观测「哪次调用花了多少」。
        """
        from ..core.llm import TaskTier

        use_tier = tier if tier is not None else TaskTier.SMALL
        # tier 允许传字符串（"small"/"large"）或枚举本身
        if isinstance(use_tier, str):
            use_tier = TaskTier(use_tier)
        # 预检：只在预算已经耗尽时拦（remaining_tokens == 0）。
        # 不按经验值预留余量——真实消耗要等响应回来才知道，
        # 凭猜测预留会误伤本该放行的调用，反而把可用预算白白浪费掉。
        if self.budget.remaining_tokens <= 0:
            raise BudgetExceeded("tokens", f"token 预算已耗尽（已用 {self.budget.tokens}）")
        started = time.perf_counter()
        try:
            if schema is not None:
                result = self.llm.invoke(
                    task=task,
                    prompt=prompt,
                    tier=use_tier,
                    schema=schema,
                    context=context,
                )
            elif use_tier == TaskTier.LARGE:
                result = self.llm.generate(task=task, prompt=prompt, context=context)
            else:
                result = self.llm.classify(task=task, prompt=prompt, context=context)
        except Exception as exc:
            # 调用失败时网关拿不到 usage，token 无法准确记账。
            # 这里只留一条失败事件（不虚记 token），真实消耗留给网关侧日志。
            self.log(
                unit_id,
                "llm_error",
                f"{task}: {type(exc).__name__}",
                cost_ms=int((time.perf_counter() - started) * 1000),
                skipped=True,
            )
            raise
        used = int(getattr(result, "tokens", 0) or 0)
        self.budget.consume_tokens(used)
        if getattr(result, "degraded", False):
            # 模型不可用 → 已退回规则网关。必须留痕，否则外部无从察觉。
            reason = str(getattr(result, "degraded_reason", "") or "模型降级")
            self.degraded_reasons.append(f"{unit_id}: {reason}")
            self.log(
                unit_id,
                "llm_degraded",
                detail=reason,
                verdict="degraded",
                cost_ms=int((time.perf_counter() - started) * 1000),
            )
        self.log(
            unit_id,
            "llm",
            detail=f"{task} {getattr(result, 'model', '')} tok={used}",
            cost_ms=int((time.perf_counter() - started) * 1000),
        )
        return result

    def stream_llm(
        self,
        *,
        unit_id: str,
        task: str,
        prompt: str,
        tier: Any = None,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """流式调用模型，与 `call_llm` 共用同一套预算/记账/事件收口。

        三条边界：
        1. `stream_sink` 为空（HTTP 非流式、冒烟、内部调用）→ 退化为 `call_llm`，
           保证有没有订阅者都不改变业务结果；
        2. 记账仍然只在这里发生：流式过程拿不到逐块 usage 时，由网关在
           流结束时回填总 tokens，再一次性记账（避免事故一那种漏记）；
        3. 订阅者（如已断开的 SSE 连接）抛异常只丢增量，不影响本轮生成。
        """
        from ..core.llm import TaskTier, build_result

        sink = self.stream_sink
        if sink is None:
            return self.call_llm(unit_id=unit_id, task=task, prompt=prompt, tier=tier, context=context)

        use_tier = tier if tier is not None else TaskTier.SMALL
        if isinstance(use_tier, str):
            use_tier = TaskTier(use_tier)
        if self.budget.remaining_tokens <= 0:
            raise BudgetExceeded("tokens", f"token 预算已耗尽（已用 {self.budget.tokens}）")

        started = time.perf_counter()
        stats: dict[str, Any] = {}
        pieces: list[str] = []
        try:
            for piece in self.llm.stream(
                task=task, prompt=prompt, tier=use_tier, context=context, stats=stats
            ):
                pieces.append(piece)
                try:
                    sink({"type": "delta", "text": piece})
                except Exception:  # noqa: BLE001 —— 订阅者断开不应中断生成
                    pass
        except Exception as exc:  # noqa: BLE001
            self.log(
                unit_id,
                "llm_error",
                f"{task}: {type(exc).__name__}",
                cost_ms=int((time.perf_counter() - started) * 1000),
                skipped=True,
            )
            raise

        used = int(stats.get("tokens") or 0)
        self.budget.consume_tokens(used)
        degraded = bool(stats.get("degraded"))
        reason = str(stats.get("degraded_reason") or "")
        if degraded:
            self.degraded_reasons.append(f"{unit_id}: {reason or '模型降级'}")
            self.log(
                unit_id,
                "llm_degraded",
                detail=reason or "模型降级",
                verdict="degraded",
                cost_ms=int((time.perf_counter() - started) * 1000),
            )
        self.log(
            unit_id,
            "llm_stream",
            detail=f"{task} {stats.get('model', '')} tok={used}",
            cost_ms=int((time.perf_counter() - started) * 1000),
        )
        return build_result(
            text="".join(pieces),
            task=task,
            tokens=used,
            model=str(stats.get("model") or ""),
            tier=use_tier,
            degraded=degraded,
            degraded_reason=reason,
        )

    def log(
        self,
        unit_id: str,
        event: str,
        detail: str = "",
        *,
        cost_ms: int = 0,
        confidence: float | None = None,
        skipped: bool = False,
        verdict: str | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(
                agent_id=unit_id,
                event=event,
                detail=detail,
                cost_ms=cost_ms,
                confidence=confidence,
                skipped=skipped,
                verdict=verdict,
            )
        )


class Unit(ABC):
    """处理单元基类：编排器可调用的最小单元。

    术语说明（此前把 24 个单元统称「智能体」是不准确的）：
    - `kind="agent"`：调用大模型，输出不可预测（P1/P2/P3/D3）；
    - `kind="router"` / `"planner"`：在选路径，不调模型（D1/D2）；
    - `kind="guard"`：有否决权，命中即阻断（Q1~Q5）；
    - `kind="capability"`：确定性能力，零模型调用（E1/E2/E4/E5/E6/E8）；
    - `kind="writer"`：写操作，幂等 + 状态机（E3/E7）。

    `uses_llm` 显式标注是否消耗模型 token：全系统只有 4 个单元为 True。
    """

    unit_id: str = "U0"
    domain: str = "base"
    description: str = ""
    kind: str = "capability"
    uses_llm: bool = False

    def __init__(self, runtime: AgentRuntime) -> None:
        self.rt = runtime

    @property
    def agent_id(self) -> str:
        """兼容别名：对外仍以 `agent_id` 标识单元编号（trace / 路由表 / API 契约）。"""
        return self.unit_id

    @abstractmethod
    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        """执行单元逻辑。返回新的信封（不修改入参）。"""

    # —— 统一包装：计时、置信度兜底、异常隔离 ——
    def run(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        started = time.perf_counter()
        try:
            result = self.handle(env, session)
        except BudgetExceeded as exc:
            cost = int((time.perf_counter() - started) * 1000)
            self.rt.log(self.unit_id, "budget_exceeded", str(exc), cost_ms=cost, skipped=True)
            out = env.child(
                self.unit_id,
                confidence=resolve(UNKNOWN, VERIFICATION),
                need_handoff=True,
                error=str(exc),
            )
            out.payload = {"degraded": True, "reason": "budget_exceeded", "detail": str(exc)}
            return out
        except Exception as exc:  # noqa: BLE001 —— 单单元失败不影响已完成结果
            cost = int((time.perf_counter() - started) * 1000)
            self.rt.log(self.unit_id, "error", f"{type(exc).__name__}: {exc}", cost_ms=cost, skipped=True)
            out = env.child(
                self.unit_id,
                confidence=resolve(UNKNOWN, VERIFICATION),
                degraded=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            out.payload = {"degraded": True, "reason": "agent_error", "detail": str(exc)}
            return out

        cost = int((time.perf_counter() - started) * 1000)
        result.cost_ms = cost
        # 信封语义：from = 产出者，to = 接收方（单元产出一律交回编排器）
        result.produced_by = self.unit_id
        result.from_agent = self.unit_id
        result.to_agent = "ORCH"
        if result.confidence is None:
            # 未给出信号 = 不可信：按最低处理，保证必定触发澄清或转人工，
            # 而不是让一个"看起来还行"的默认值蒙混过关。
            result.confidence = resolve(UNKNOWN, VERIFICATION)
        self.rt.budget.consume_call()
        self.rt.log(
            self.unit_id,
            "invoke",
            detail=self.description,
            cost_ms=cost,
            confidence=result.confidence,
        )
        return result

    # —— 便捷方法 ——
    def call_tool(
        self,
        name: str,
        params: dict[str, Any],
        session: SessionContext,
        env: AgentEnvelope,
    ) -> ToolResult:
        """调用工具并记录事件。"""
        ctx = ToolCallContext(
            trace_id=env.trace_id,
            session_id=env.session_id,
            user_id=session.user.id,
            turn=env.turn,
            channel=session.channel.type,
        )
        result = self.rt.tools.execute(name, params, ctx)
        self.rt.log(
            self.unit_id,
            "tool",
            detail=f"{name} ok={result.ok}",
            cost_ms=result.cost_ms,
            verdict="ok" if result.ok else "fail",
        )
        return result

    def check_facts(self, text: str, evidence: list[Any], facts: dict[str, Any]) -> Any:
        return check_facts(text, evidence=evidence, facts=facts)

    def check_compliance(self, text: str) -> Any:
        return check_compliance(text)
