"""置信度的单一事实源。

`confidence` 在本项目里**不是概率**，而是「本次产出可以被信任的程度」。
为了让这个字段只有一种语义，Agent 不再各自写死数字，而是声明两件事：

1. `outcome` —— 这次产出属于哪一类结果状态（决定打折幅度）；
2. `tier`   —— 这类任务本身有多确定（决定上限）。

两者取小值（短板效应）：任一维度弱，整体就不可信。
所有映射关系集中在本文件，可审计、可回归。

历史背景：此前 24 个 Agent 各自硬编码置信度，同一字段混装了
「模型自报概率」「规则加法评分」「人工标注的信任等级」三种语义，
且九成数值不参与任何决策分支。现在统一为「结果状态 + 任务档位」。
"""

from __future__ import annotations

# —— 结果状态：互斥、穷尽 ——
EXACT = "exact"        # 完整命中，事实完整可用
PARTIAL = "partial"    # 部分命中：有内容缺失、或走了兜底路径
EMPTY = "empty"        # 查无此项（合法的空结果）
MISSING = "missing"    # 前置条件不足（缺槽位、缺商品），不是失败
REJECTED = "rejected"  # 核验未通过 / 校验拦截：结论明确，但结果是"不行"
ERROR = "error"        # 工具或依赖报错
UNKNOWN = "unknown"    # 没有给出任何信号

# —— 任务档位：任务本身有多确定 ——
EXACT_MATCH = "exact_match"      # 键精确匹配，命中即真（订单号 → 订单）
DETERMINISTIC = "deterministic"  # 确定性事实查询（库存/会员/支付/物流/商品）
COMPUTED = "computed"            # 规则计算与写操作（含一定不确定性）
GENERATED = "generated"          # 模型组织的话术
PERCEPTION = "perception"        # 槽位抽取
CLASSIFY = "classify"            # 情绪与风险分级
RETRIEVAL = "retrieval"          # 证据检索
VERIFICATION = "verification"    # 事实校验 / 质量评估
POLICY = "policy"                # 合规检查
HANDOFF = "handoff"              # 人工接管摘要
RULE = "rule"                    # 纯规则判定（触达 / 社群）

# 结果状态 → 置信度上限
OUTCOME_CONFIDENCE: dict[str, float] = {
    EXACT: 0.97,
    PARTIAL: 0.75,
    EMPTY: 0.25,
    MISSING: 0.20,
    REJECTED: 0.20,
    ERROR: 0.30,
    UNKNOWN: 0.00,
}

# 任务档位 → 置信度上限
TIER_CONFIDENCE: dict[str, float] = {
    EXACT_MATCH: 0.95,
    DETERMINISTIC: 0.94,
    COMPUTED: 0.90,
    GENERATED: 0.85,
    PERCEPTION: 0.90,
    CLASSIFY: 0.88,
    RETRIEVAL: 0.80,
    VERIFICATION: 0.90,
    POLICY: 0.95,
    HANDOFF: 0.95,
    RULE: 0.85,
}


def resolve(outcome: str, tier: str) -> float:
    """由结果状态与任务档位解析出置信度。

    取二者较小值：结果状态弱（空/错/被拦）或任务本身不确定（生成），
    都会把置信度压低。未知状态/档位直接抛错，避免静默给出错误数值。
    """
    return min(OUTCOME_CONFIDENCE[outcome], TIER_CONFIDENCE[tier])


__all__ = [
    "EXACT",
    "PARTIAL",
    "EMPTY",
    "MISSING",
    "REJECTED",
    "ERROR",
    "UNKNOWN",
    "EXACT_MATCH",
    "DETERMINISTIC",
    "COMPUTED",
    "GENERATED",
    "PERCEPTION",
    "CLASSIFY",
    "RETRIEVAL",
    "VERIFICATION",
    "POLICY",
    "HANDOFF",
    "RULE",
    "OUTCOME_CONFIDENCE",
    "TIER_CONFIDENCE",
    "resolve",
]
