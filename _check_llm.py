"""模型连通性自检：确认 `.env` 里的 DeepSeek 配置真的可用。

运行：
    python _check_llm.py

它用**真实的 Agent Prompt** 做 4 次最小调用（意图 / 槽位 / 情绪 / 话术），
几秒钟出结果，比跑完整个 demo 更适合排查 Key、base_url、模型名和 JSON 解析问题。
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from cs_agent_system.agents.decision import ResponseWriter  # noqa: E402
from cs_agent_system.agents.perception import _PROMPT_EMOTION, _PROMPT_INTENT  # noqa: E402
from cs_agent_system.core.llm import LangChainLLMGateway, MockLLMGateway, build_llm_gateway  # noqa: E402

_SLOT_PROMPT = (
    "你是电商客服的槽位抽取器。当前意图是 RETURN_APPLY，只允许抽取下列槽位：\n"
    "- order_no（订单号，必填）\n- reason（退换原因，必填）\n- type（售后类型，可选）\n\n"
    '只输出 JSON：{"slots": {...}, "missing": ["必填但缺失的槽位名"]}'
)

CASES = [
    ("intent", _PROMPT_INTENT, {"text": "订单号 SO202610280001 质量太差，我要退款"}, "RETURN_APPLY"),
    ("slots", _SLOT_PROMPT, {"text": "订单号 SO202610280001，原因是质量问题"}, None),
    ("emotion", _PROMPT_EMOTION, {"text": "再不处理我就去12315投诉你们"}, None),
    (
        "generate",
        ResponseWriter._PROMPT,
        {"facts_skeleton": "亲，您的订单 SO202610280001 已发货，实付 189 元。"},
        None,
    ),
]


def main() -> int:
    gateway = build_llm_gateway()
    print(f"网关实现 : {type(gateway).__name__}")
    if isinstance(gateway, MockLLMGateway):
        print("！！未读到 LLM_API_KEY（或 langchain-openai 不可用），已回退内置规则网关。")
        print("   请检查工程根目录 .env 是否存在、LLM_API_KEY 是否为空。")
        return 1

    assert isinstance(gateway, LangChainLLMGateway)
    print(f"模型     : {getattr(gateway.large_model, 'model_name', 'unknown')}")
    print(f"base_url : {getattr(gateway.large_model, 'openai_api_base', 'unknown')}")
    print("-" * 60)

    failed = 0
    for item in CASES:
        task, prompt, context = item[0], item[1], item[2]
        expect = item[3] if len(item) > 3 else None
        try:
            result = gateway.invoke(task=task, prompt=prompt, context=context)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {task:<8} {type(exc).__name__}: {exc}")
            continue
        if not result.data:
            failed += 1
            print(f"[FAIL] {task:<8} 未解析出 JSON。原始输出前 80 字：{result.text[:80]!r}")
            continue
        print(f"[ OK ] {task:<8} model={result.model} tokens={result.tokens} data={result.data}")
        if expect and str(result.data.get("intent", "")).strip().upper() != expect:
            print(f"       ！意图与预期不符：期望 {expect}，实际 {result.data.get('intent')}")
            failed += 1

    if gateway.last_error:
        print(f"\n最近一次失败原因：{gateway.last_error}")
    print("-" * 60)
    print("结论：", "全部通过" if failed == 0 and not gateway.last_error else "存在失败，请看上面的原因")
    return 0 if failed == 0 and not gateway.last_error else 1


if __name__ == "__main__":
    sys.exit(main())
