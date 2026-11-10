import os, sys, traceback

# 冒烟脚本是「确定性回归」，必须离线跑：强制用内置规则网关，
# 否则接上真实模型后断言会随模型输出漂移（想走真实模型请跑 demo / 控制台）。
os.environ.setdefault("USE_MOCK_LLM", "1")

sys.path.insert(0, 'src')
from cs_agent_system.api import ChatRequest, CustomerService
from cs_agent_system.core.envelope import SessionContext, UserIdentity
from cs_agent_system.knowledge.intent_fewshot import build_fewshot, retrieve_examples

svc = CustomerService()



def run(text, uid="u_1001"):
    orch = svc.build_orchestrator()
    ctx = SessionContext(user=UserIdentity(id=uid, channel="private"))
    r = orch.handle(text=text, user_id=uid, channel_type="private", session_ctx=ctx)
    return orch, r

checks = []
def check(name, cond, extra=""):
    checks.append((name, cond, extra))
    print(("PASS " if cond else "FAIL ") + name + (" | " + str(extra) if extra else ""))

# —— RAG 动态 Few-shot 检索 ——
_hit = retrieve_examples("包裹卡在哪儿了")
check("fewshot.retrieve", bool(_hit) and _hit[0].intent == "LOGISTICS", [e.intent for e in _hit])
# 与示例库毫不相关的句子必须检索不到，否则会把模型带偏（宁缺毋滥）
_noise = build_fewshot("随便聊聊吧")[1]
check("fewshot.no_noise", _noise == [], _noise)

orch, r = run("我的订单状态")
check("order.intent", r["intent"] == "ORDER_STATUS", r["intent"])
check("order.route", r["route"] == "direct_tool", r["route"])
check("order.response_has_order_no", "SO202610280001" in r["response"], r["response"])
check("order.no_handoff", not r["need_handoff"], r.get("need_handoff"))

# 用「我的包裹」而不是「这个订单」：后者是指代无号码，按设计会反问「哪一单」，
# 属于澄清路径，不是物流链路本身的问题。
orch, r = run("我的包裹到哪了")
check("logistics.intent", r["intent"] == "LOGISTICS", r["intent"])
check("logistics.has_facts", bool(r["facts"].get("last_node")), r["facts"])

orch, r = run("订单号 SO202610280001 质量太差了，我要退款")
check("refund.intent", r["intent"] == "RETURN_APPLY", r["intent"])
check("refund.has_e3", "E3" in r["agent_outputs"], list(r["agent_outputs"]))
check("refund.idempotency_key", bool(r["agent_outputs"].get("E3", {}).get("idempotency_key")))
check("refund.sla_from_policy", r["agent_outputs"].get("E3", {}).get("facts", {}).get("refund_sla_days") == 3, r["agent_outputs"].get("E3", {}).get("facts"))

orch, r = run("你们这什么破服务，我要12315投诉！")
check("complaint.route", r["route"] == "handoff", r["route"])
check("complaint.handoff", r["need_handoff"] is True)
check("complaint.ticket", str(r.get("handoff_payload", {}).get("ticket_id", "")).startswith("TK-"), r.get("handoff_payload"))

orch, r = run("那个事情怎么样了")
check("clarify.route", r["route"] == "clarify", r["route"])
check("clarify.conf_low", r["intent_confidence"] < 0.6, r["intent_confidence"])

orch, r = run("这件卫衣有黑色 M 码吗")
check("size.intent", r["intent"] == "SIZE_ADVICE", r["intent"])
check("size.has_price", bool(r["facts"].get("price")), r["facts"])

# 多意图复合请求 → D2 规划-执行（唯一一条此前未覆盖的编排路径）
orch, r = run("订单号 SO202610280001 尺码不合适，我要退款，另外物流到哪了")
check("plan.route", r["route"] == "plan_execute", r["route"])
check("plan.dag_targets", set(r["route_targets"]) >= {"Q1", "E2", "E3"}, r["route_targets"])
check("plan.no_handoff", not r["need_handoff"], r.get("need_handoff"))

# 流式链路：有订阅者时应逐块产出，且与无订阅者时结果一致
def run_stream(text, uid="u_1001"):
    orch = svc.build_orchestrator()
    ctx = SessionContext(user=UserIdentity(id=uid, channel="private"))
    chunks = []
    stages = []
    # stream_sink 现在收结构化事件：stage（阶段进度）/ delta（增量文本）
    orch.rt.stream_sink = lambda ev: (
        chunks.append(ev.get("text") or "") if ev.get("type") == "delta" else stages.append(ev)
    )
    try:
        r = orch.handle(text=text, user_id=uid, channel_type="private", session_ctx=ctx)
    finally:
        orch.rt.stream_sink = None
    return chunks, stages, r, orch

_chunks, _stages, _sr, _orch = run_stream("我的订单状态")
_base = run("我的订单状态")[1]
check("stream.deltas", len(_chunks) >= 2, len(_chunks))
check("stream.non_empty", bool("".join(_chunks).strip()))
check("stream.has_stage", len(_stages) >= 3, [s.get("detail") for s in _stages][:3])
check("stream.same_as_invoke", _sr["response"] == _base["response"], [_sr["response"], _base["response"]])
check("stream.llm_event", any(e.event == "llm_stream" for e in _orch.rt.events), [e.event for e in _orch.rt.events])

failed = [c for c in checks if not c[1]]
print("----")
print(f"TOTAL {len(checks)} PASSED {len(checks)-len(failed)} FAILED {len(failed)}")
sys.exit(1 if failed else 0)
