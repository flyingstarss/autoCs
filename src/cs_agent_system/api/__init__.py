"""接入层（L0）：多渠道适配与统一入口。

渠道差异被收敛在适配器里；上游只看到统一的 ChatRequest / ChatResponse。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..agents import build_agents
from ..agents.base import AgentRuntime
from ..config import settings
from ..core.budget import BudgetGuard
from ..core.envelope import BudgetInfo, ChannelInfo, SessionContext, UserIdentity
from ..core.ids import new_session_id
from ..core.llm import LLMGateway, build_llm_gateway
from ..core.state import SessionState, build_session_state
from ..knowledge.retriever import HybridRetriever, build_default_retriever
from ..orchestrator.graph import Orchestrator
from ..registry import db, tickets, traces
from ..registry.tasks import TaskStore, build_task_store
from ..registry.tool_registry import ToolCallContext
from ..registry.tools import build_registry
from .admin_console import ADMIN_HTML
from .console import CONSOLE_HTML


class TicketDecision(BaseModel):
    """审核动作的入参。operator 必填——审批必须留痕到具体的人。"""

    operator: str = Field(..., description="操作人（坐席/管理员）")
    note: str = Field("", description="审批备注")


# —— 统一消息信封 ——
class ChatRequest(BaseModel):
    text: str = Field(..., description="用户消息")
    user_id: str = Field(..., description="用户标识")
    channel: str = Field("private", description="渠道类型：private/group/web（未知值回落 private）")
    session_id: str | None = None
    is_group: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    trace_id: str
    session_id: str
    turn: int
    response: str
    intent: str
    confidence: float
    route: str
    need_handoff: bool
    # 转人工/待审批时生成的工单号，前端可据此展示「你的受理编号」
    ticket_id: str = ""
    degraded: bool = False
    # 降级原因清单：非空说明这一轮的答案并非模型真答（如模型不可用退回规则网关）
    degraded_reasons: list[str] = Field(default_factory=list)
    # 供前端回放链路用：路由目标、质量分、是否自助解决
    targets: list[str] = Field(default_factory=list)
    quality_score: float = 0.0
    resolved: bool = False
    events: list[dict[str, Any]] = Field(default_factory=list)
    # 预算快照：调用数 / token / 耗时，让「这一轮花了多少」可观测
    budget: dict[str, int] = Field(default_factory=dict)


class ChannelAdapter:
    """渠道能力描述：决定 D3 的排版与长度上限。

    入参归一化此前也放在这里，但 `/chat` 直接接收 `ChatRequest`，
    `normalize` 从未被调用过（唯一调用方是已删除的 `handle_raw`），
    已移除；渠道差异现在只剩长度与群聊标记。
    """

    name = "base"

    def capabilities(self) -> ChannelInfo:
        return ChannelInfo(type=self.name)


class PrivateChatAdapter(ChannelAdapter):
    """私域单聊适配器。"""

    name = "private"

    def capabilities(self) -> ChannelInfo:
        return ChannelInfo(type=self.name, max_len=500)


class GroupChatAdapter(ChannelAdapter):
    """私域群聊适配器。"""

    name = "group"

    def capabilities(self) -> ChannelInfo:
        return ChannelInfo(type=self.name, is_group=True, max_len=300)


class WebAdapter(PrivateChatAdapter):
    """网页控制台，与私域单聊同能力，仅渠道标识不同。"""

    name = "web"


ADAPTERS: dict[str, ChannelAdapter] = {
    "private": PrivateChatAdapter(),
    "group": GroupChatAdapter(),
    "web": WebAdapter(),
}


@dataclass
class _SessionRuntime:
    """一个会话的运行时：预算、单元实例与编排器跨轮复用。"""

    runtime: AgentRuntime
    agents: dict[str, Any]
    orchestrator: Orchestrator


class CustomerService:
    """系统装配：把运行时、处理单元、编排器、渠道适配器组装成一个可调用服务。"""

    def __init__(
        self,
        llm: LLMGateway | None = None,
        tools: Any | None = None,
        retriever: HybridRetriever | None = None,
        state: SessionState | None = None,
        tasks: TaskStore | None = None,
    ) -> None:
        self.llm = llm or build_llm_gateway()
        self.tools = tools or build_registry()
        self.retriever = retriever or build_default_retriever()
        self.state = state or build_session_state(settings.redis_url)
        # 任务栈是**进程级共享**的（不是会话级）：SQLite 持久化，
        # 所有会话共用一份存储，进程重启后未完成任务仍在。
        self.tasks = tasks or build_task_store()
        self.adapters = ADAPTERS
        self._sessions: dict[str, _SessionRuntime] = {}

    # —— 运行时装配 ——
    @staticmethod
    def _fresh_budget() -> BudgetGuard:
        """新建一本预算账本。

        三重预算（调用数 / Token / 墙钟）按「轮」计量：一轮内超限即降级，
        下一轮重新获得配额。会话态（槽位、记忆、身份）才跨轮保留——
        否则第 2~3 轮起必然预算耗尽，被误判为需要转人工。
        """
        return BudgetGuard(
            budget=BudgetInfo(
                max_agent_calls=settings.max_agent_calls,
                max_tokens=settings.max_tokens,
                deadline_ms=settings.deadline_ms,
            )
        )

    def _new_session_runtime(self) -> _SessionRuntime:
        runtime = AgentRuntime(
            llm=self.llm,
            tools=self.tools,
            retriever=self.retriever,
            state=self.state,
            budget=self._fresh_budget(),
            tasks=self.tasks,
        )
        agents = build_agents(runtime)
        return _SessionRuntime(runtime=runtime, agents=agents, orchestrator=Orchestrator(runtime, agents))

    def build_orchestrator(self, session_id: str | None = None) -> Orchestrator:
        """取会话运行时；同一会话复用，保证三重预算跨轮累计。"""
        if not session_id:
            return self._new_session_runtime().orchestrator
        if session_id not in self._sessions:
            self._sessions[session_id] = self._new_session_runtime()
        return self._sessions[session_id].orchestrator

    def session_runtime(self, session_id: str) -> _SessionRuntime:
        if session_id not in self._sessions:
            self._sessions[session_id] = self._new_session_runtime()
        return self._sessions[session_id]

    # —— 对外调用 ——
    def chat(self, request: ChatRequest, on_event: Callable[[dict[str, Any]], None] | None = None) -> ChatResponse:
        adapter = self.adapters.get(request.channel, self.adapters["private"])
        session_id = request.session_id or new_session_id(request.user_id, request.channel)
        runtime = self.session_runtime(session_id)
        orchestrator = runtime.orchestrator

        # 每轮重置预算与事件流：同一会话的后续轮次不能被上一轮消耗拖垮
        runtime.runtime.budget = self._fresh_budget()
        runtime.runtime.events.clear()
        runtime.runtime.degraded_reasons.clear()

        ctx = SessionContext(
            user=UserIdentity(id=request.user_id, channel=request.channel),
            channel=adapter.capabilities(),
            budget=runtime.runtime.budget.budget,
        )
        # 复用已有会话态中的槽位与轮次
        stored = self.state.load(session_id)
        if stored:
            ctx.slots = dict(stored.get("slots") or {})
            ctx.turn_count = int(stored.get("turn_count") or 0)
            # 槽位来源元数据：下一轮据此判断旧槽位是否已失效
            # （用户换话题后，旧订单号不该继续生效）
            ctx.slot_meta = dict(stored.get("slot_meta") or {})
            # 追问次数：跨轮累计，用于「同一槽位只问一次」的上限判定
            ctx.ask_counts = dict(stored.get("ask_counts") or {})
            # 恢复最近对话：P1 意图识别的 prompt 依赖 history 做指代消解。
            # 此前只恢复了 slots/turn_count，history 一直是空的 ——
            # 于是用户说「继续」「接着刚才」时，模型面前只有这三个字，
            # 无从知道有未完成的任务，只能当闲聊处理。
            ctx.history.recent_turns = list(stored.get("recent_turns") or [])
            ctx.history.resolved_topics = list(stored.get("resolved_topics") or [])
            # 上一轮意图：用于「继续」这类指代在模型放弃时兜底的辅助线索
            ctx.prev_intent = str(stored.get("prev_intent") or "")
        # 未完成任务栈：从 TaskStore 恢复（SQLite 持久化，跨进程可续接）。
        # 必须放在 `if stored:` 之外 —— 任务栈独立于会话态：
        # 新进程里会话态是空的（内存态丢失），但任务栈在 DB 里，
        # 若跟着会话态一起跳过，重启后用户说「继续」就找不回原任务了。
        ctx.pending_tasks = [t.to_dict() for t in runtime.runtime.tasks.list_pending(session_id)]

        started = time.perf_counter()
        # 每轮独立注入订阅者：同一会话的 runtime 会跨轮复用，
        # 不在结束时清空就会把上一轮的 SSE 队列带到下一轮（串话）。
        runtime.runtime.stream_sink = on_event
        try:
            result = orchestrator.handle(
                text=request.text,
                user_id=request.user_id,
                channel_type=request.channel,
                session_id=session_id,
                is_group=request.is_group,
                session_ctx=ctx,
            )
        finally:
            runtime.runtime.stream_sink = None

        response_text = str(result.get("response", "") or "")
        budget_snapshot = runtime.runtime.budget.snapshot()
        # 可观测性：把这一轮完整落库，事后能按 trace_id 复盘
        # 「用户说了什么 → 意图/路由 → 谁被调用 → 花了多少 token → 最终回复」。
        try:
            traces.put(
                {
                    "trace_id": result.get("trace_id", ""),
                    "session_id": result.get("session_id", session_id),
                    "user_id": request.user_id,
                    "turn": int(result.get("turn", 0)),
                    "ts": time.time(),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "text": request.text,
                    "intent": result.get("intent", ""),
                    "confidence": float(result.get("intent_confidence") or 0.0),
                    "route": result.get("route", ""),
                    "targets": list(result.get("route_targets") or []),
                    "need_handoff": bool(result.get("need_handoff")),
                    # 本轮产生的工单号：让「查这一轮」和「处理这张单」互相可达
                    "ticket_id": str(result.get("ticket_id") or ""),
                    "degraded": bool(result.get("degraded")),
                    "degraded_reasons": list(result.get("degraded_reasons") or []),
                    "quality_score": float(result.get("quality_score") or 0.0),
                    "resolved": bool(result.get("resolved")),
                    "tokens": int(budget_snapshot.get("tokens") or 0),
                    "response": response_text,
                    "events": [event.model_dump() for event in runtime.runtime.events],
                }
            )
        except Exception:  # noqa: BLE001 - 轨迹写入失败不影响对话
            pass
        return ChatResponse(
            trace_id=result.get("trace_id", ""),
            session_id=result.get("session_id", session_id),
            turn=int(result.get("turn", 0)),
            response=response_text,
            intent=str(result.get("intent", "CHITCHAT")),
            confidence=float(result.get("intent_confidence", 0.0)),
            route=str(result.get("route", "")),
            need_handoff=bool(result.get("need_handoff")),
            ticket_id=str(result.get("ticket_id") or ""),
            degraded=bool(result.get("degraded")),
            degraded_reasons=list(result.get("degraded_reasons") or []),
            targets=list(result.get("route_targets") or []),
            quality_score=float(result.get("quality_score") or 0.0),
            resolved=bool(result.get("resolved")),
            events=[e.model_dump() for e in runtime.runtime.events],
            budget=budget_snapshot,
        )

def _sse(event: str, data: dict[str, Any]) -> str:
    """SSE 帧：事件名 + JSON 数据 + 空行分隔。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app() -> FastAPI:
    """FastAPI 应用工厂。"""
    app = FastAPI(title="电商/私域多智能体客服系统", version="0.1.0")
    service = CustomerService()

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """运行态自检，同时是「进程是否最新代码」的探针。

        旧进程只会返回 {"status": "ok"}；有 build / gateway / data_source 字段
        说明已加载本轮修复（按轮重置预算与事件、DeepSeek 网关、SQLite 数据源）。
        """
        from ..registry.data import data_source_info

        return {
            "status": "ok",
            "build": "per-turn-budget+deepseek+sqlite+mcp",
            "gateway": type(service.llm).__name__,
            "model": getattr(getattr(service.llm, "large_model", None), "model_name", None),
            "per_turn_budget": True,
            "max_agent_calls": settings.max_agent_calls,
            "data_source": data_source_info(),
            "mcp": {
                "enabled": settings.mcp_enabled,
                "server_enabled": settings.mcp_server_enabled,
                "tools_via_mcp": settings.tools_via_mcp,
            },
        }

    @app.get("/", response_class=HTMLResponse)
    @app.get("/console", response_class=HTMLResponse)
    def console() -> HTMLResponse:
        """内置可视化控制台：会话窗 + 本轮回放（零依赖单页）。

        no-store：控制台 HTML 会随代码更新，不能被浏览器缓存成旧页面。
        """
        return HTMLResponse(content=CONSOLE_HTML, headers={"Cache-Control": "no-store, max-age=0"})

    @app.get("/admin", response_class=HTMLResponse)
    def admin_console() -> HTMLResponse:
        """人工审核台：列待办、看详情、批准/驳回（零依赖单页）。

        页面只是壳子，权限与幂等由 `/admin/tickets*` 接口保证。
        """
        return HTMLResponse(content=ADMIN_HTML, headers={"Cache-Control": "no-store, max-age=0"})

    @app.get("/tools")
    def tools() -> list[dict[str, Any]]:
        return service.tools.describe()

    @app.get("/agents")
    def agents() -> list[dict[str, Any]]:
        """列出全部处理单元及其性质。

        `kind` 区分单元性质，`uses_llm` 标注是否消耗模型 token——
        全系统只有 4 个单元 `uses_llm=true`。
        """
        return [
            {
                "id": unit.unit_id,
                "domain": unit.domain,
                "description": unit.description,
                "kind": unit.kind,
                "uses_llm": unit.uses_llm,
            }
            for unit in service.session_runtime("__list__").agents.values()
        ]

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        return service.chat(request)

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        """流式对话（SSE）。

        四类事件：
        - `meta`：连接建立（后端已开始处理，前端据此建气泡）；
        - `stage`：流水线阶段进度（感知/路由/执行/生成/校验），
          让用户在文字产出前就知道系统在做什么；
        - `delta`：模型增量文本，边生成边推；
        - `done`：完整 ChatResponse（含意图/路由/事件流/预算），**以它为准**。

        为什么以 `done` 为准：D3 之后还有 Q2/Q3 校验与数字回填，
        流式期间展示的是"草稿"，最终文本必须用 `done` 覆盖，
        否则用户会看到一段被校验否掉的内容。
        """
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(event: dict[str, Any]) -> None:
            # 工作线程 → 事件循环：sink 只做投递，不做任何阻塞操作
            loop.call_soon_threadsafe(queue.put_nowait, ("event", event))

        def work() -> None:
            try:
                response = service.chat(request, on_event=emit)
                loop.call_soon_threadsafe(queue.put_nowait, ("done", response.model_dump()))
            except Exception as exc:  # noqa: BLE001 —— 错误也要通过事件告诉前端
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", f"{type(exc).__name__}: {exc}")
                )

        async def generate() -> Any:
            worker = asyncio.create_task(asyncio.to_thread(work))
            yield _sse("meta", {"user_id": request.user_id, "channel": request.channel})
            try:
                while True:
                    kind, payload = await queue.get()
                    if kind == "event":
                        # 事件名即 type（stage / delta），前端按名分发
                        yield _sse(str(payload.get("type") or "delta"), payload)
                    elif kind == "done":
                        yield _sse("done", payload)
                        break
                    else:
                        yield _sse("error", {"message": payload})
                        break
            finally:
                # 客户端提前断开时 worker 线程仍在跑，这里等它收尾，
                # 保证 traces 落库与预算记账不被流式响应的取消打断
                await worker

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    # —— 可观测性接口 ——
    # 三个接口对应排障时的三个问题：
    #   这一轮到底发生了什么？  → GET /trace/{trace_id}
    #   最近几轮都走了什么路径？ → GET /traces
    #   系统整体在变好还是变差？ → GET /metrics
    @app.get("/trace/{trace_id}")
    def trace_detail(trace_id: str) -> dict[str, Any]:
        record = traces.get(trace_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"未找到轨迹 {trace_id}")
        return record

    @app.get("/traces")
    def trace_list(limit: int = 20, session_id: str | None = None) -> list[dict[str, Any]]:
        return traces.list_recent(limit=limit, session_id=session_id)

    @app.get("/metrics")
    def metrics(limit: int = 200) -> dict[str, Any]:
        return traces.summary(limit=limit)

    # —— 人工审核台 ——
    # 三个接口对应坐席的三个动作：看待办 → 看详情 → 批准/驳回。
    # 审批单批准后会**真正重放写操作**（带原幂等键），这是钱唯一能出去的通路；
    # 接管单批准只是把待办标记为「已处理」，不产生副作用。
    @app.get("/admin/tickets")
    def ticket_list(
        status: str = "pending",
        kind: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """待办列表。默认只看 pending —— 已处理的不该混在待办里。"""
        status_filter = None if status in ("", "all", "*") else status
        items = tickets.list_tickets(status=status_filter, kind=kind, limit=limit)
        return {
            "tickets": [t.to_dict() for t in items],
            "count": len(items),
            "db_available": db.is_available(),
        }

    @app.get("/admin/tickets/{ticket_id}")
    def ticket_detail(ticket_id: str) -> dict[str, Any]:
        ticket = tickets.get(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail=f"未找到工单 {ticket_id}")
        return ticket.to_dict()

    @app.post("/admin/tickets/{ticket_id}/approve")
    def ticket_approve(ticket_id: str, body: TicketDecision) -> dict[str, Any]:
        """批准：审批单会重放写操作，接管单只标记已处理。"""
        ticket = tickets.get(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail=f"未找到工单 {ticket_id}")
        # 幂等：重复点批准不能退两次钱。非 pending 一律拒绝。
        if ticket.status != tickets.PENDING:
            raise HTTPException(
                status_code=409,
                detail=f"工单 {ticket_id} 当前状态为 {ticket.status}，不可重复审批",
            )

        if ticket.kind != "approval":
            updated = tickets.decide(
                ticket_id,
                tickets.APPROVED,
                operator=body.operator,
                note=body.note or "坐席已接管处理",
                result={"replayed": False},
            )
            return {"ticket": updated.to_dict() if updated else None, "replayed": False}

        if not ticket.tool:
            raise HTTPException(status_code=422, detail="审批单缺少可重放的写操作，无法批准")

        # 重放必须用**原来的 trace_id**：幂等键由 trace_id+工具+目标+版本派生，
        # 换一个 trace_id 就等于换了一把键，会被当成全新请求再退一次。
        ctx = ToolCallContext(
            trace_id=ticket.trace_id,
            session_id=ticket.session_id,
            user_id=ticket.user_id,
            channel="admin",
        )
        replay_params = dict(ticket.params or {})
        # 注入审批放行标记，否则重放会再次命中金额闸门，形成死循环
        replay_params["__approval_granted"] = True
        # allow_retry=False：写操作不自动重试，超时一律转「状态未知」人工核对
        result = service.tools.execute(ticket.tool, replay_params, ctx, allow_retry=False)

        if result.status_unknown:
            # 超时：钱可能已经退了，也可能没有。标记 uncertain 交人工去 OMS 侧核对，
            # 绝不能直接判失败或自动重试。
            updated = tickets.decide(
                ticket_id,
                tickets.UNCERTAIN,
                operator=body.operator,
                note=body.note or "重放超时，状态未知，需到 OMS 侧核对",
                result={"error": result.error, "idempotency_key": ticket.idempotency_key},
            )
            return {
                "ticket": updated.to_dict() if updated else None,
                "replayed": True,
                "status_unknown": True,
                "error": result.error,
            }

        if not result.ok:
            # 业务失败（如 4xx）：钱没出去，保持可再处理
            updated = tickets.decide(
                ticket_id,
                tickets.REJECTED,
                operator=body.operator,
                note=body.note or f"重放失败：{result.error}",
                result={"error": result.error},
            )
            return {
                "ticket": updated.to_dict() if updated else None,
                "replayed": True,
                "ok": False,
                "error": result.error,
            }

        updated = tickets.decide(
            ticket_id,
            tickets.APPROVED,
            operator=body.operator,
            note=body.note or "审批通过，写操作已重放执行",
            result={
                "replayed": True,
                "value": result.data.get("value"),
                "idempotency_key": result.idempotency_key,
            },
        )
        return {
            "ticket": updated.to_dict() if updated else None,
            "replayed": True,
            "ok": True,
            "value": result.data.get("value"),
        }

    @app.post("/admin/tickets/{ticket_id}/reject")
    def ticket_reject(ticket_id: str, body: TicketDecision) -> dict[str, Any]:
        """驳回：不产生任何副作用。

        驳回的审批单对应的幂等记录是 rejected 状态，不占用业务锁，
        用户修正后可以同版本重新提交。
        """
        ticket = tickets.get(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail=f"未找到工单 {ticket_id}")
        if ticket.status != tickets.PENDING:
            raise HTTPException(
                status_code=409,
                detail=f"工单 {ticket_id} 当前状态为 {ticket.status}，不可重复审批",
            )
        updated = tickets.decide(
            ticket_id,
            tickets.REJECTED,
            operator=body.operator,
            note=body.note or "审批驳回",
            result={"replayed": False},
        )
        return {"ticket": updated.to_dict() if updated else None, "replayed": False}

    return app


app = create_app()
