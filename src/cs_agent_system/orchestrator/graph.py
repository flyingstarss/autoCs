"""LangGraph 编排状态机（架构文档第 5、9、11、12 章的代码落点）。

图拓扑：

    perceive(P1→P2→P3→P4) → route(D1)
                                   ├─ direct_tool / agent_chain ─→ execute ─→ generate(D3) ─→ verify(Q2+Q3)
                                   ├─ plan_execute ─→ plan(D2) ─→ execute ─┘        │
                                   ├─ clarify ─→ clarify_reply ───────────────────┐   ├─ pass → quality(Q4) → finalize
                                   └─ handoff ─→ handoff(Q5) ─────────────────────┤   ├─ reflect → generate（反思循环，最多 1 次）
                                                                            └───────┘   └─ block → handoff

编排器六条规则在此落地：短路优先、并行优先、三重预算、置信度裁决、失败隔离、幂等执行。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..agents.base import AgentRuntime, Unit
from ..core.budget import BudgetExceeded
from ..core.confidence import UNKNOWN, VERIFICATION, resolve
from ..core.envelope import AgentEnvelope, Evidence, SessionContext
from ..core.intents import get_intent
from ..registry import tickets
from ..registry.tasks import Task, make_task_id
from .degrade import degrade
from .state import OrchestratorState, new_state

MAX_REFLECT = 1  # 反思循环最多 1 轮


class Orchestrator:
    """编排器：唯一有权决定「调用谁、调用几次、能否放行」的地方。"""

    def __init__(self, runtime: AgentRuntime, agents: dict[str, Unit]) -> None:
        self.rt = runtime
        self.agents = agents
        self.graph = self._build()

    # —— 对外入口 ——
    def handle(
        self,
        *,
        text: str,
        user_id: str,
        channel_type: str = "web",
        session_id: str | None = None,
        trace_id: str | None = None,
        is_group: bool = False,
        session_ctx: SessionContext | None = None,
    ) -> OrchestratorState:
        from ..core.ids import new_session_id, new_trace_id

        trace_id = trace_id or new_trace_id()
        session_id = session_id or new_session_id(user_id, channel_type)
        ctx = session_ctx or SessionContext(
            trace_id=trace_id,
            session_id=session_id,
        )
        ctx.user.id = user_id
        ctx.channel.type = channel_type
        ctx.channel.is_group = is_group

        env = AgentEnvelope(
            trace_id=trace_id,
            session_id=session_id,
            turn=ctx.turn_count + 1,
            payload={"text": text, "channel": channel_type, "user_id": user_id},
            **{"from": "L0", "to": "P1"},
        )
        state = new_state(
            trace_id=trace_id,
            session_id=session_id,
            turn=env.turn,
            text=text,
            user_id=user_id,
            channel_type=channel_type,
            is_group=is_group,
            session_ctx=ctx,
            envelope=env,
        )
        result = self.graph.invoke(state)
        return result  # type: ignore[return-value]

    # —— 节点实现 ——
    def _perceive(self, state: OrchestratorState) -> dict[str, Any]:
        """感知域串行链：P1 → P2 → P3 → P4。P3 与 P1 无依赖，可并行。"""
        env: AgentEnvelope = state["envelope"]
        ctx: SessionContext = state["session_ctx"]

        # 感知是整条链路里最慢的一段（三次模型调用），细分进度能让用户
        # 在 2 秒级的等待里持续看到变化，而不是盯着一句静止的「识别意图」。
        p1 = self._run("P1", env, ctx)
        env.intent = p1.payload.get("intent")
        # 意图置信度是唯一的路由裁决依据，必须是实数：
        # None（无信号）按 0.0 处理，必定低于阈值走澄清。
        intent_confidence = float(p1.confidence or 0.0)
        env.confidence = intent_confidence
        ctx.intent.primary = env.intent or "CHITCHAT"
        ctx.intent.confidence = intent_confidence
        ctx.intent.multi = list(p1.payload.get("multi") or [])

        self._stage("perceive", "抽取槽位…")
        p2 = self._run("P2", env, ctx)
        # 注意不能用 `p2.slots or ctx.slots`：P2 判定了槽位失效时会返回空字典，
        # 空字典是假值，会被 `or` 回落到上一轮的旧槽位 —— 旧单号又被捡回来。
        env.slots = dict(p2.slots if p2.slots is not None else ctx.slots)
        env.missing = list(p2.missing or [])
        ctx.slots = dict(env.slots)
        ctx.missing = list(env.missing)
        # 槽位来源元数据：记录每个槽位是哪一轮、哪个意图抽取的。
        # 下一轮 P2 据此判断旧槽位是否已失效（B 方案的判定依据）。
        ctx.slot_meta = dict(ctx.slot_meta or {})
        stale = list(p2.payload.get("stale_slots") or [])
        for key in stale:
            ctx.slot_meta.pop(key, None)
        # 本轮被确认的槽位 → 打上来源标记
        for key in list(ctx.slots.keys()):
            if key in (p2.payload.get("extracted") or {}):
                ctx.slot_meta[key] = {"turn": ctx.turn_count, "intent": ctx.intent.primary}
            elif key not in ctx.slot_meta:
                # 继承下来的旧槽位：保留原有 meta，没有就补一个来源标记
                ctx.slot_meta.setdefault(key, {"turn": ctx.turn_count, "intent": ctx.intent.primary})
        # 本轮明确给出的订单号（可能是核验失败的）——工具层据此区分
        # 「用户指定了单号但查不到」与「用户没指定单号」
        ctx.explicit_order_no = str(p2.payload.get("explicit_order_no") or "")
        # 澄清话术（如「您说的是哪一单」）：沿 envelope 传给路由节点，
        # 由澄清节点原样使用，避免把指代问题渲染成「请提供订单号」的腔调。
        env.payload["clarify_hint"] = str(p2.payload.get("clarify_hint") or "")
        # 追问无果/用户拒答（P2 判定）必须一路带到 D1：
        # P2 的产出信封在这里会被丢弃（state 里放的是原始 env），
        # 只把它算进 state["need_handoff"] 不够 —— D1 读的是 env 本身。
        env.payload["force_handoff"] = str(p2.payload.get("force_handoff") or "")
        env.need_handoff = bool(getattr(p2, "need_handoff", False))

        self._stage("perceive", "评估情绪与风险…")
        p3 = self._run("P3", env, ctx)
        ctx.emotion = _emotion_from(p3.payload)
        env.need_handoff = bool(env.need_handoff or getattr(p3, "need_handoff", False))

        p4 = self._run("P4", env, ctx)
        ctx = _context_from(p4.payload, ctx)

        return {
            "intent": ctx.intent.primary,
            "intent_confidence": ctx.intent.confidence,
            "multi_intents": list(ctx.intent.multi),
            "slots": dict(ctx.slots),
            "missing": list(ctx.missing),
            "emotion": ctx.emotion.model_dump(),
            "session_ctx": ctx,
            "envelope": env,
            "need_handoff": bool(env.need_handoff),
            # 用户是否放弃了当前任务（P1 识别到的取消信号）
            "cancelled": bool(p1.payload.get("cancelled")),
            # 本轮是否走了续接兜底
            "resumed": bool(p1.payload.get("resumed")),
        }

    def _route(self, state: OrchestratorState) -> dict[str, Any]:
        env: AgentEnvelope = state["envelope"]
        ctx: SessionContext = state["session_ctx"]
        d1 = self._run("D1", env, ctx)
        payload = d1.payload
        return {
            "route": payload.get("route", "handoff"),
            "route_reason": payload.get("reason", ""),
            "route_targets": list(payload.get("targets") or []),
            "route_parallel": bool(payload.get("parallel")),
            "clarify_options": list(payload.get("clarify_options") or []),
            # 感知层给的澄清话术（指代未给号码时）：澄清节点优先采用
            "clarify_hint": str(env.payload.get("clarify_hint") or ""),
            "need_handoff": bool(d1.need_handoff) or state.get("need_handoff", False),
        }

    def _plan(self, state: OrchestratorState) -> dict[str, Any]:
        env: AgentEnvelope = state["envelope"]
        ctx: SessionContext = state["session_ctx"]
        d2 = self._run("D2", env, ctx)
        steps = d2.payload.get("steps") or []
        if not d2.payload.get("valid", True):
            # 任务图不可执行 → 降级为人工接管
            return {
                "plan_steps": [],
                "route": "handoff",
                "route_reason": f"任务规划不可执行：{d2.payload.get('problem')}",
                "need_handoff": True,
            }
        return {"plan_steps": steps, "route": "plan_execute"}

    def _execute(self, state: OrchestratorState) -> dict[str, Any]:
        """执行节点：按路由目标调用执行域/保障域处理单元。

        无依赖的目标并行扇出；有依赖的按波次串行。
        """
        env: AgentEnvelope = state["envelope"]
        ctx: SessionContext = state["session_ctx"]
        route = state.get("route")
        targets = list(state.get("route_targets") or [])

        if route == "plan_execute":
            steps = list(state.get("plan_steps") or [])
            targets = [s["agent"] for s in steps]
            # 按波次执行：波内并行，波间串行
            return self._execute_steps(state, env, ctx, steps)

        if not targets:
            return {"facts": state.get("facts", {}), "agent_outputs": state.get("agent_outputs", {})}

        parallel = bool(state.get("route_parallel"))
        return self._execute_agents(state, env, ctx, targets, parallel=parallel)

    def _execute_steps(
        self,
        state: OrchestratorState,
        env: AgentEnvelope,
        ctx: SessionContext,
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        facts: dict[str, Any] = dict(state.get("facts") or {})
        outputs: dict[str, dict[str, Any]] = dict(state.get("agent_outputs") or {})
        evidence: list[Evidence] = list(state.get("evidence") or [])
        tool_results: list[dict[str, Any]] = list(state.get("tool_results") or [])
        needs_approval = bool(state.get("needs_approval"))
        approval_reason = state.get("approval_reason", "")
        need_handoff = bool(state.get("need_handoff"))
        slots_patch: dict[str, Any] = {}

        remaining = list(steps)
        done: set[str] = set()
        while remaining:
            layer = [s for s in remaining if all(d in done for d in s.get("depends_on", []))]
            if not layer:
                break
            # 每个步骤用「本步骤意图」的独立信封：
            # 复合请求里主意图是 LOGISTICS 时，Q1/E3 仍要按 RETURN_APPLY 取政策、办售后；
            # 同时把已收集的证据传给后续波次。
            step_envs: list[AgentEnvelope] = []
            for step in layer:
                step_env = env.child(step["agent"], intent=step.get("intent") or env.intent)
                step_env.slots = dict(env.slots or {})
                step_env.evidence = list(evidence)
                step_envs.append(step_env)
            results = self._run_many(
                [(step["agent"], step_env) for step, step_env in zip(layer, step_envs)],
                ctx,
                parallel=len(layer) > 1,
            )
            for step, result in zip(layer, results):
                done.add(step["id"])
                payload = result.payload or {}
                outputs[result.produced_by or result.to_agent] = payload
                facts.update(payload.get("facts") or {})
                evidence.extend(result.evidence or [])
                slots_patch.update(payload.get("slots_patch") or {})
                if result.need_handoff:
                    need_handoff = True
                    approval_reason = approval_reason or f"{result.produced_by} 请求人工审批"
                if payload.get("needs_approval"):
                    needs_approval = True
                    approval_reason = approval_reason or "写操作金额超过阈值，需人工审批"
                for item in payload.get("tool_results") or []:
                    tool_results.append(item)
            remaining = [s for s in remaining if s["id"] not in done]

        self._apply_slots_patch(ctx, state, slots_patch)
        return {
            "facts": facts,
            "agent_outputs": outputs,
            "evidence": evidence,
            "tool_results": tool_results,
            "needs_approval": needs_approval,
            "approval_reason": approval_reason,
            "need_handoff": need_handoff,
            "slots": dict(ctx.slots),
            "route_targets": [s["agent"] for s in steps],
        }

    def _execute_agents(
        self,
        state: OrchestratorState,
        env: AgentEnvelope,
        ctx: SessionContext,
        targets: list[str],
        *,
        parallel: bool,
    ) -> dict[str, Any]:
        facts: dict[str, Any] = dict(state.get("facts") or {})
        outputs: dict[str, dict[str, Any]] = dict(state.get("agent_outputs") or {})
        evidence: list[Evidence] = list(state.get("evidence") or [])
        need_handoff = bool(state.get("need_handoff"))
        needs_approval = bool(state.get("needs_approval"))
        approval_reason = state.get("approval_reason", "")
        slots_patch: dict[str, Any] = {}

        if parallel:
            self._stage("execute", f"并行查询 {'、'.join(targets)}…")
            results = self._run_many([(aid, env) for aid in targets], ctx, parallel=True)
            for result in results:
                payload = result.payload or {}
                outputs[result.produced_by or result.to_agent] = payload
                facts.update(payload.get("facts") or {})
                evidence.extend(result.evidence or [])
                slots_patch.update(payload.get("slots_patch") or {})
                if result.need_handoff:
                    need_handoff = True
                    approval_reason = approval_reason or f"{result.produced_by} 请求人工接管"
                if payload.get("needs_approval"):
                    needs_approval = True
                    approval_reason = approval_reason or "写操作金额超过阈值，需人工审批"
        else:
            # 串行链：证据沿链传递，后一个单元能看到前一个的产出
            results = []
            for aid in targets:
                self._stage("execute", f"查询 {aid}…")
                env.evidence = list(evidence)
                result = self._run(aid, env, ctx)
                payload = result.payload or {}
                outputs[result.produced_by or result.to_agent] = payload
                facts.update(payload.get("facts") or {})
                evidence.extend(result.evidence or [])
                slots_patch.update(payload.get("slots_patch") or {})
                if result.need_handoff:
                    need_handoff = True
                    approval_reason = approval_reason or f"{result.produced_by} 请求人工接管"
                if payload.get("needs_approval"):
                    needs_approval = True
                    approval_reason = approval_reason or "写操作金额超过阈值，需人工审批"
                results.append(result)

        self._apply_slots_patch(ctx, state, slots_patch)
        return {
            "facts": facts,
            "agent_outputs": outputs,
            "evidence": evidence,
            "need_handoff": need_handoff,
            "needs_approval": needs_approval,
            "approval_reason": approval_reason,
            "slots": dict(ctx.slots),
        }

    @staticmethod
    def _apply_slots_patch(
        ctx: SessionContext, state: dict[str, Any], patch: dict[str, Any]
    ) -> None:
        """把执行层解析出的业务对象回写进会话槽位（编排器是唯一写入者）。

        解决「上一句问蓝牙耳机，下一句只问库存」的指代断裂：
        E4/E5 在工具层已经把商品定位到了，若只把它用在当轮话术里、
        不记进会话，下一轮工具层只能拿到空参数，答非所问。
        """
        if not patch:
            return
        intent = str(state.get("intent") or ctx.intent.primary or "")
        for key, value in patch.items():
            if value in (None, "", []):
                continue
            ctx.slots[key] = value
            ctx.slot_meta[key] = {"turn": ctx.turn_count, "intent": intent}

    def _generate(self, state: OrchestratorState) -> dict[str, Any]:
        env: AgentEnvelope = state["envelope"]
        ctx: SessionContext = state["session_ctx"]
        gen_env = env.child("D3")
        gen_env.payload = {"facts": state.get("facts") or {}, "text": state.get("text", "")}
        gen_env.evidence = list(state.get("evidence") or [])
        d3 = self._run("D3", gen_env, ctx)
        return {
            "draft": d3.payload.get("text", ""),
            "envelope": gen_env,
            # D3 的产出同样入 agent_outputs：否则 dropped_segments（被静默丢弃的段落）
            # 无法观测，置信度降级也就无从追溯。
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "D3": d3.payload,
            },
        }

    def _verify(self, state: OrchestratorState) -> dict[str, Any]:
        """保障层校验：Q2 事实校验 + Q3 合规检查（两者无依赖，并行）。"""
        env: AgentEnvelope = state["envelope"]
        ctx: SessionContext = state["session_ctx"]
        draft = state.get("draft", "")
        facts = state.get("facts") or {}
        evidence = list(state.get("evidence") or [])

        q2_env = env.child("Q2")
        q2_env.payload = {"text": draft, "facts": facts}
        q2_env.evidence = evidence
        q3_env = env.child("Q3")
        q3_env.payload = {"text": draft}

        q2, q3 = self._run_many([("Q2", q2_env), ("Q3", q3_env)], ctx, parallel=True)

        fact_ok = bool(q2.payload.get("passed", True))
        compliance_ok = bool(q3.payload.get("passed", True))
        draft_out = q3.payload.get("text") or draft

        return {
            "draft": draft_out,
            "fact_check_passed": fact_ok,
            "fact_check_detail": q2.payload.get("reason", ""),
            "compliance_passed": compliance_ok,
            "compliance_detail": "; ".join(
                i.get("message", "") for i in (q3.payload.get("issues") or [])
            ),
            "need_handoff": bool(q2.need_handoff or q3.need_handoff) or bool(state.get("need_handoff")),
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "Q2": q2.payload,
                "Q3": q3.payload,
            },
        }

    def _reflect(self, state: OrchestratorState) -> dict[str, Any]:
        """反思回边：生成 → 校验 → 不通过则回到生成重做（最多 1 轮）。"""
        return {"reflect_count": int(state.get("reflect_count", 0)) + 1}


    def _quality(self, state: OrchestratorState) -> dict[str, Any]:
        env: AgentEnvelope = state["envelope"]
        ctx: SessionContext = state["session_ctx"]
        q4_env = env.child("Q4")
        q4_env.payload = {
            "text": state.get("draft", ""),
            "facts": state.get("facts") or {},
            "resolved": not state.get("need_handoff", False),
        }
        q4 = self._run("Q4", q4_env, ctx)
        return {
            "quality_score": q4.payload.get("score", 0.0),
            "resolved": q4.payload.get("resolved", False),
            "agent_outputs": {**state.get("agent_outputs", {}), "Q4": q4.payload},
        }

    def _clarify(self, state: OrchestratorState) -> dict[str, Any]:
        """澄清节点：一次只问一个关键槽位，或给二选一。"""
        intent = state.get("intent", "CHITCHAT")
        missing = state.get("missing") or []
        options = state.get("clarify_options") or []
        hint = str(state.get("clarify_hint") or "")
        spec = get_intent(intent)

        if hint:
            # 感知层已经给定了更贴用户原话的澄清方式（「您说的是哪一单」），
            # 直接采用，不再套「请提供订单号」的模板腔调。
            text = hint
        elif missing:
            slot_name = missing[0]
            description = ""
            if spec:
                for s in spec.slot_specs:
                    if s.name == slot_name:
                        description = s.description
            text = f"为了帮您准确处理，麻烦提供一下{description or slot_name}，方便回复吗？"
        elif options:
            text = f"想帮您处理的是哪一项呢？{options[0]}，还是{options[1]}？"
        else:
            text = "麻烦再具体描述一下您的问题，我马上为您处理。"

        return {"response": text, "resolved": False}

    def _handoff(self, state: OrchestratorState) -> dict[str, Any]:
        """人工接管节点：Q5 生成交接摘要与工单。"""
        env: AgentEnvelope = state["envelope"]
        ctx: SessionContext = state["session_ctx"]
        reason = state.get("route_reason") or state.get("approval_reason") or state.get("compliance_detail") or "触发人工闸门"
        h_env = env.child("Q5")
        h_env.payload = {"reason": reason}
        q5 = self._run("Q5", h_env, ctx)
        # 用户侧话术与坐席侧摘要分离：用户看不到内部交接细节
        user_text = (
            "非常抱歉给您带来不好的体验，已为您转接专属客服，客服会尽快跟进处理。"
        )
        return {
            "response": user_text,
            "need_handoff": True,
            "handoff_payload": q5.payload,
            "resolved": False,
            "agent_outputs": {**state.get("agent_outputs", {}), "Q5": q5.payload},
        }

    def _finalize(self, state: OrchestratorState) -> dict[str, Any]:
        """收口：降级链处理 + 会话态写入（编排器是唯一写入者）。"""
        response = state.get("response") or state.get("draft") or ""
        degraded = False
        degrade_level = ""
        need_handoff = bool(state.get("need_handoff"))

        # 模型层降级（如真实模型不可用、退回规则网关）：
        # 回复看起来正常，但答案不是模型真答的 —— 必须显式标记，
        # 否则外部（前端/运维/质检）无法察觉模型已经挂了。
        llm_degraded = list(self.rt.degraded_reasons)
        if llm_degraded:
            degraded = True
            degrade_level = degrade_level or "L1"

        # 兜底：整条链路都没产出文本时，用意图模板顶上（L2）
        if not response:
            result = degrade(
                intent=state.get("intent", "CHITCHAT"),
                facts=state.get("facts") or {},
                reason=state.get("route_reason", ""),
            )
            response = result.text
            degraded = True
            degrade_level = result.level
            need_handoff = need_handoff or result.need_handoff

        # 写操作需审批：走阻塞审批形态
        if state.get("needs_approval") and not need_handoff:
            need_handoff = True
            response = response or "您的申请金额较高，已提交人工审核，稍后会有专属客服与您联系。"

        # 会话态落库（单一写入者）
        ctx: SessionContext = state["session_ctx"]
        self.rt.state.append_turn(
            state["session_id"],
            "user",
            state.get("text", ""),
            turn=state.get("turn", 0),
        )
        self.rt.state.append_turn(
            state["session_id"],
            "agent",
            response,
            intent=state.get("intent"),
            quality=state.get("quality_score"),
        )

        # —— 未完成任务栈维护 ——
        # 本轮因缺槽位而没办完 → 挂成待续任务（同一意图复用同一个 task）；
        # 本轮办完 → 把该意图的任务标记完成。
        # 这样用户切去别的任务再回来时，原任务还在，可以续。
        self._sync_tasks(state, ctx)

        current_intent = str(state.get("intent") or "")
        self.rt.state.apply_delta(
            state["session_id"],
            {
                "slots": state.get("slots") or {},
                # 槽位来源元数据：跨轮保留，下一轮据此判断旧槽位是否失效。
                # 丢了它，「用户换话题后旧订单号仍生效」的问题就会复现。
                "slot_meta": dict(ctx.slot_meta or {}),
                # 追问次数跨轮累计：没有它就无法判断「这个槽位是不是已经问过了」
                "ask_counts": dict(ctx.ask_counts or {}),
                "intent": current_intent,
                # 上一轮意图：供下一轮做「继续」类指代兜底的辅助线索
                "prev_intent": current_intent,
                "turn_count": state.get("turn", 0),
            },
        )

        # —— 工单落库 ——
        # 只在这里落一次：_finalize 是收口节点，无论从哪条分支（接管 / 审批 /
        # 降级兜底）走到这，都保证「说出口的转人工」一定有对应工单。
        # 之前工单只活在 Q5 的 payload 里，进程一重启就消失了，
        # 「已提交人工审核」对用户是空头支票。
        ticket_id = self._persist_ticket(state, need_handoff)

        return {
            "response": response,
            "degraded": degraded,
            "degrade_level": degrade_level,
            "degraded_reasons": llm_degraded,
            "need_handoff": need_handoff,
            "ticket_id": ticket_id,
            "session_ctx": ctx,
        }

    def _persist_ticket(self, state: OrchestratorState, need_handoff: bool) -> str:
        """把本轮的人工待办落成工单，返回工单号（无待办返回空串）。

        两类工单必须分开：
        - **审批单**（approval）：写操作被闸门拦住，钱还没出去。工单要带
          幂等键与重放参数，管理员批准后据此重新提交 —— 这是唯一能真正把钱
          退出去的通路。
        - **接管单**（handoff）：链路自己办不了，转给坐席。工单只承载上下文，
          没有可重放的写操作。
        """
        trace_id = str(state.get("trace_id") or "")
        # 信封与 Q5 的工单号口径一致，保证「对话里说的号」和「库里存的号」是同一个
        ticket_id = tickets.make_ticket_id(trace_id)

        if state.get("needs_approval"):
            payload = self._approval_payload(state)
            inner = payload.get("aftersale") or {}
            replay = dict(inner.get("replay_params") or {})
            tickets.create(
                tickets.Ticket(
                    ticket_id=ticket_id,
                    kind="approval",
                    status=tickets.PENDING,
                    reason=str(state.get("approval_reason") or "写操作金额超过阈值，需人工审批"),
                    summary=(
                        f"订单 {inner.get('order_no') or '未知'} 申请退款 "
                        f"{inner.get('amount') or 0:.2f} 元"
                        f"（订单金额 {inner.get('gate_amount') or 0:.2f} 元超过审批阈值），待审批。"
                    ),
                    intent=str(state.get("intent") or ""),
                    user_id=str(state.get("user_id") or ""),
                    session_id=str(state.get("session_id") or ""),
                    trace_id=trace_id,
                    order_no=str(inner.get("order_no") or ""),
                    amount=float(inner.get("amount") or 0.0),
                    tool="aftersale.apply",
                    params=replay,
                    # 带上原幂等键：批准后重放必须命中同一条记录，
                    # 否则就是第二笔退款。
                    idempotency_key=str(
                        inner.get("idempotency_key") or payload.get("idempotency_key") or ""
                    ),
                )
            )
            return ticket_id

        if not need_handoff:
            return ""

        q5 = (state.get("agent_outputs") or {}).get("Q5") or {}
        reason = (
            str(state.get("route_reason") or "")
            or str(state.get("approval_reason") or "")
            or str(state.get("compliance_detail") or "")
            or str(q5.get("reason") or "")
            or "触发人工闸门"
        )
        tickets.create(
            tickets.Ticket(
                ticket_id=ticket_id,
                kind="handoff",
                status=tickets.PENDING,
                reason=reason,
                summary=str(q5.get("summary") or reason),
                intent=str(state.get("intent") or ""),
                user_id=str(state.get("user_id") or ""),
                session_id=str(state.get("session_id") or ""),
                trace_id=trace_id,
            )
        )
        return ticket_id

    @staticmethod
    def _approval_payload(state: OrchestratorState) -> dict[str, Any]:
        """找出产生 needs_approval 的那个执行单元 payload。"""
        for payload in (state.get("agent_outputs") or {}).values():
            if isinstance(payload, dict) and payload.get("needs_approval"):
                return payload
        return {}

    # —— 任务栈维护 ——
    def _sync_tasks(self, state: OrchestratorState, ctx: SessionContext) -> None:
        """把本轮的执行结果同步到长任务栈。

        规则：
        1. 本轮缺槽位（没办完）→ 挂起 / 更新一个待续任务；
        2. 本轮办完了 → 把该意图的待续任务标记完成；
        3. 同一会话同一意图只有一个任务（task_id 由 session+intent 派生），
           重复发起是「更新原任务」而不是堆新任务。

        TaskStore 不可用（未装配 / DB 挂了）时静默跳过：
        任务栈是增强能力，不能因为它挂掉就中断客服链路。
        """
        store = self.rt.tasks
        if store is None:
            return
        session_id = state["session_id"]

        # 任务取消：用户说「算了不退了」→ 把该会话所有待续任务清空。
        # 清空而不是只删一个：用户说「算了」时通常放弃的是当前这摊事，
        # 留着其他任务只会让后续「继续」指向用户已经不想做的事。
        if state.get("cancelled"):
            try:
                for task in store.list_pending(session_id):
                    store.drop(task.task_id)
                    self.rt.log("ORCH", "task_cancelled", task.intent)
            except Exception as exc:  # noqa: BLE001 —— 任务栈异常不能拖垮链路
                self.rt.log("ORCH", "task_error", f"{type(exc).__name__}: {exc}", skipped=True)
            return

        intent = str(state.get("intent") or "")
        if not intent:
            return
        task_id = make_task_id(session_id, intent)
        missing = list(state.get("missing") or [])

        if missing:
            # 还没办完：挂起（保留之前已填的槽位，本轮补上的合并进去）
            try:
                existing = store.get(task_id)
                slots = dict(existing.slots) if existing else {}
                slots.update(state.get("slots") or {})
                now = time.time()
                store.upsert(
                    Task(
                        task_id=task_id,
                        session_id=session_id,
                        user_id=str(state.get("user_id") or ""),
                        intent=intent,
                        slots=slots,
                        missing=missing,
                        status="pending",
                        created_at=existing.created_at if existing else now,
                        updated_at=now,
                    )
                )
                self.rt.log("ORCH", "task_pending", f"{intent} 缺 {','.join(missing)}")
            except Exception as exc:  # noqa: BLE001 —— 任务栈异常不能拖垮链路
                self.rt.log("ORCH", "task_error", f"{type(exc).__name__}: {exc}", skipped=True)
            return

        # 办完了：把该意图的待续任务结清
        try:
            existing = store.get(task_id)
            if existing is not None and existing.is_pending:
                store.complete(task_id)
                self.rt.log("ORCH", "task_done", intent)
        except Exception as exc:  # noqa: BLE001 —— 任务栈异常不能拖垮链路
            self.rt.log("ORCH", "task_error", f"{type(exc).__name__}: {exc}", skipped=True)

    # —— 路由 ——
    def _route_after_route(self, state: OrchestratorState) -> str:
        route = state.get("route", "handoff")
        if route == "clarify":
            return "clarify"
        if route == "handoff":
            return "handoff"
        if route == "plan_execute":
            return "plan"
        return "execute"

    def _route_after_verify(self, state: OrchestratorState) -> str:
        if not state.get("compliance_passed", True):
            return "handoff"
        if not state.get("fact_check_passed", True):
            if int(state.get("reflect_count", 0)) < MAX_REFLECT:
                return "reflect"
            return "handoff"
        if state.get("need_handoff"):
            return "handoff"
        return "quality"

    # —— 图装配 ——
    def _stage(self, stage: str, detail: str = "") -> None:
        """推送阶段进度（无流式订阅者时静默丢弃）。"""
        self.rt.stage(stage, detail)

    def _node(self, stage: str, detail: str, fn: Any) -> Any:
        """包装节点：进入节点前先推一条进度事件。

        没有这层包装，前端在 D3 开始吐字前只能看到一个静止的「思考中」，
        完全看不到感知/路由/执行走到哪一步 —— 而这三步才是主要耗时。
        """
        def wrapped(state: OrchestratorState) -> dict[str, Any]:
            self._stage(stage, detail)
            return fn(state)

        return wrapped

    def _build(self) -> Any:
        g = StateGraph(OrchestratorState)

        g.add_node("perceive", self._node("perceive", "识别意图与槽位…", self._perceive))
        g.add_node("route", self._node("route", "判断处理路径…", self._route))
        g.add_node("plan", self._node("plan", "规划执行步骤…", self._plan))
        g.add_node("execute", self._node("execute", "查询业务数据…", self._execute))
        g.add_node("generate", self._node("generate", "组织回复…", self._generate))
        g.add_node("verify", self._node("verify", "校验事实与合规…", self._verify))
        g.add_node("reflect", self._node("reflect", "修正表述…", self._reflect))
        g.add_node("quality", self._node("quality", "评估回复质量…", self._quality))
        g.add_node("clarify", self._node("clarify", "整理澄清问题…", self._clarify))
        g.add_node("handoff", self._node("handoff", "转接人工客服…", self._handoff))
        g.add_node("finalize", self._node("finalize", "收口…", self._finalize))

        g.add_edge(START, "perceive")
        g.add_edge("perceive", "route")
        g.add_conditional_edges(
            "route",
            self._route_after_route,
            {"clarify": "clarify", "handoff": "handoff", "plan": "plan", "execute": "execute"},
        )
        g.add_edge("plan", "execute")
        g.add_edge("execute", "generate")
        g.add_edge("generate", "verify")
        g.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"reflect": "reflect", "handoff": "handoff", "quality": "quality"},
        )
        g.add_edge("reflect", "generate")
        g.add_edge("quality", "finalize")
        g.add_edge("clarify", "finalize")
        g.add_edge("handoff", "finalize")
        g.add_edge("finalize", END)

        return g.compile()

    # —— 单元调用辅助 ——
    def _run(self, unit_id: str, env: AgentEnvelope, ctx: SessionContext) -> AgentEnvelope:
        unit = self.agents.get(unit_id)
        if unit is None:
            out = env.child(unit_id, confidence=resolve(UNKNOWN, VERIFICATION))
            out.payload = {"error": f"未注册的处理单元 {unit_id}"}
            return out
        try:
            self.rt.budget.check()
        except BudgetExceeded as exc:
            out = env.child(
                unit_id,
                confidence=resolve(UNKNOWN, VERIFICATION),
                need_handoff=True,
                error=str(exc),
            )
            out.payload = {"degraded": True, "reason": "budget_exceeded", "detail": str(exc)}
            self.rt.log(unit_id, "budget_exceeded", str(exc), skipped=True)
            return out
        return unit.run(env, ctx)

    def _run_many(
        self,
        tasks: list[Any],
        ctx: SessionContext,
        *,
        parallel: bool,
    ) -> list[AgentEnvelope]:
        """批量调用。tasks 元素为 agent_id 字符串或 (agent_id, env) 元组。"""
        items = [(t, None) if isinstance(t, str) else t for t in tasks]
        if not parallel or len(items) == 1:
            return [self._run(aid, env or ctx_envelope(ctx), ctx) for aid, env in items]
        with ThreadPoolExecutor(max_workers=min(8, len(items))) as pool:
            futures = [pool.submit(self._run, aid, env or ctx_envelope(ctx), ctx) for aid, env in items]
            return [f.result() for f in futures]


def ctx_envelope(ctx: SessionContext) -> AgentEnvelope:
    """为缺少信封的调用构造一个最小信封。"""
    # from / to 是 Python 保留字，只能通过别名字典构造
    return AgentEnvelope(
        trace_id=ctx.trace_id or "tr_unknown",
        session_id=ctx.session_id or "ses_unknown",
        turn=ctx.turn_count + 1,
        **{"from": "ORCH", "to": "agent"},
    )


# —— 小工具 ——
def _emotion_from(payload: dict[str, Any]) -> Any:
    from ..core.envelope import EmotionInfo

    data = payload or {}
    return EmotionInfo(
        sentiment=data.get("sentiment", "neutral"),
        anger_score=float(data.get("anger_score", 0.0)),
        complaint_intent=bool(data.get("complaint_intent")),
        risk_level=data.get("risk_level", "low"),
        escalation_signals=list(data.get("escalation_signals") or []),
    )


def _context_from(payload: dict[str, Any], fallback: SessionContext) -> SessionContext:
    data = (payload or {}).get("context")
    if not data:
        return fallback
    return SessionContext.model_validate(data)
