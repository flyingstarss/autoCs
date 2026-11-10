"""处理单元注册中心：按编号装配全部单元。

术语（此前把这里全部称作「智能体」是不准确的）：
本系统共 **21 个处理单元**，其中只有 **4 个调用大模型**（P1/P2/P3/D3），
其余 17 个是确定性能力、决策单元或守卫，零模型调用。

单元性质由 `kind` 标注，`uses_llm` 标注是否消耗 token：
    kind="agent"      调用大模型，输出不可预测       P1 P2 P3 D3
    kind="router"     选路径                         D1
    kind="planner"    选执行顺序与依赖               D2
    kind="capability" 确定性能力，只做取数与投影       E1 E2 E4 E5 E6 E8 P4 Q1 Q4
    kind="writer"     写操作，幂等 + 状态机          E3 E7
    kind="guard"      独立否决权                     Q2 Q3 Q5
    kind="computed"   规则型计算                     E9
"""

from __future__ import annotations

from .base import AgentRuntime, Unit


def build_agents(runtime: AgentRuntime) -> dict[str, Unit]:
    """构造全部处理单元实例，返回 {unit_id: unit}。

    21 个单元是逻辑单元，不等于 21 次模型调用：
    同档位的多个单元共用同一个模型实例，且大多数单元零模型推理。
    """
    from .assurance import (
        ComplianceGuard,
        FactChecker,
        HandoffBuilder,
        KnowledgeRetriever,
        QualityEvaluator,
    )
    from .decision import Planner, ResponseWriter, Router
    from .execution import (
        AfterSaleWriter,
        BenefitCalculator,
        CouponWriter,
        InventoryCapability,
        LogisticsCapability,
        MemberCapability,
        OrderCapability,
        PaymentCapability,
        ProductCapability,
    )
    from .perception import ContextBuilder, EmotionClassifier, IntentClassifier, SlotExtractor

    units: list[Unit] = [
        # 感知域：3 个模型驱动 + 1 个确定性组装
        IntentClassifier(runtime),
        SlotExtractor(runtime),
        EmotionClassifier(runtime),
        ContextBuilder(runtime),
        # 决策域：2 个决策单元 + 1 个模型驱动
        Router(runtime),
        Planner(runtime),
        ResponseWriter(runtime),
        # 执行域：6 个确定性能力 + 2 个写操作 + 1 个规则计算
        OrderCapability(runtime),
        LogisticsCapability(runtime),
        AfterSaleWriter(runtime),
        InventoryCapability(runtime),
        ProductCapability(runtime),
        PaymentCapability(runtime),
        CouponWriter(runtime),
        MemberCapability(runtime),
        BenefitCalculator(runtime),
        # 保障域：2 个守卫 + 2 个确定性能力
        KnowledgeRetriever(runtime),
        FactChecker(runtime),
        ComplianceGuard(runtime),
        QualityEvaluator(runtime),
        HandoffBuilder(runtime),
    ]
    return {u.unit_id: u for u in units}


__all__ = ["AgentRuntime", "Unit", "build_agents"]
