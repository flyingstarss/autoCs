"""可运行 Demo：打印完整调用 trace。

运行：
    python -m cs_agent_system.demo

无需任何 API Key —— 默认使用内置 MockLLMGateway。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .api import ChatRequest, CustomerService
from .core.envelope import SessionContext, UserIdentity


SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "场景A · 售前咨询（单工具短路）",
        "text": "这件卫衣有黑色 M 码吗",
        "expect": "P1→P2→P3→P4 → D1 短路到 E4 → D3 生成 → Q2/Q3 校验",
    },
    {
        "name": "场景B · 售后退款（写操作 + 策略承诺值 + 幂等键）",
        "text": "订单号 SO202610280001 质量太差了，我要退款",
        "expect": "Q1 取政策证据 → E3 带幂等键发起售后 → 到账时效来自策略服务",
    },
    {
        "name": "场景C · 高风险投诉（直接转人工）",
        "text": "你们这什么破服务，我要12315投诉！",
        "expect": "P3 风险等级=高 → 跳过自动应答链路 → Q5 生成交接摘要与工单",
    },
    {
        "name": "场景D · 意图不明（澄清而非猜测）",
        "text": "那个事情怎么样了",
        "expect": "意图置信度不足 → 澄清提问",
    },
    {
        "name": "场景E · 物流查询（E1+E2 并行扇出）",
        "text": "我的快递到哪了",
        "expect": "订单与物流并行查询 → 事实校验 → 生成",
    },
]


def _render_trace(result: dict[str, Any], events: list[Any]) -> None:
    print("─" * 78)
    print(f"  trace_id : {result.get('trace_id')}")
    print(f"  session  : {result.get('session_id')}  turn={result.get('turn')}")
    print(f"  意图     : {result.get('intent')}  置信度={result.get('intent_confidence')}")
    print(f"  路由     : {result.get('route')}   原因={result.get('route_reason')}")
    print(f"  目标单元: {result.get('route_targets')}")
    print(f"  回复     : {result.get('response')}")
    print(f"  转人工   : {bool(result.get('need_handoff'))}   降级={result.get('degrade_level') or '无'}")
    print(f"  质量分   : {result.get('quality_score')}   自助解决={result.get('resolved')}")
    print("  ── 调用时序 ──")
    for e in events:
        mark = "⊘" if e.skipped else ("✓" if e.verdict in ("pass", "ok") else ("!" if e.verdict else "·"))
        conf = f" conf={e.confidence:.2f}" if e.confidence is not None else ""
        ms = f"{e.cost_ms}ms" if e.cost_ms else "-"
        print(f"    {mark} {e.agent_id:<3} {e.event:<16} {ms:>7}{conf}  {e.detail or e.verdict or ''}")


def run_demo(scenario_indexes: list[int] | None = None) -> int:
    service = CustomerService()
    selected = (
        SCENARIOS
        if not scenario_indexes
        else [SCENARIOS[i] for i in scenario_indexes if 0 <= i < len(SCENARIOS)]
    )

    print("=" * 78)
    print("  电商/私域多智能体客服系统 · 运行 Demo")
    print(f"  模型网关: {type(service.llm).__name__}    工具数: {len(service.tools.names())}")
    if hasattr(service.llm, "large_model"):
        print(f"  生成模型: {getattr(service.llm.large_model, 'model_name', 'unknown')}")
    print("=" * 78)

    for scenario in selected:
        print()
        print(f"▶ {scenario['name']}")
        print(f"  用户输入: {scenario['text']}")
        print(f"  预期链路: {scenario['expect']}")

        orchestrator = service.build_orchestrator()
        ctx = SessionContext(user=UserIdentity(id="u_1001", channel="private"))
        result = orchestrator.handle(
            text=scenario["text"],
            user_id="u_1001",
            channel_type="private",
            session_ctx=ctx,
        )
        _render_trace(result, orchestrator.rt.events)

    print()
    print("=" * 78)
    print("  Demo 结束")
    print("=" * 78)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多智能体客服系统 Demo")
    parser.add_argument("--scenario", type=int, action="append", help="只运行指定场景（0-4），可重复")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = parser.parse_args(argv)

    if args.json:
        service = CustomerService()
        out = []
        for scenario in SCENARIOS:
            resp = service.chat(ChatRequest(text=scenario["text"], user_id="u_1001"))
            out.append(
                {
                    "scenario": scenario["name"],
                    "trace_id": resp.trace_id,
                    "intent": resp.intent,
                    "route": resp.route,
                    "response": resp.response,
                    "need_handoff": resp.need_handoff,
                    # ChatResponse.events 已经是 model_dump 后的 dict，不能再调用 model_dump()
                    "events": resp.events,
                }
            )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    return run_demo(args.scenario)


if __name__ == "__main__":
    sys.exit(main())
