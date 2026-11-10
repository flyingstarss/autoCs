"""事实校验（Q2 的底层）。

Q2 逐句比对生成内容与证据/业务数据，标记幻觉。
核心原则：无证据支撑的事实句一律标记为无依据，不做「看起来合理」的放行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.envelope import Evidence

# 需要有证据支撑的数字型断言（金额、天数、时效）
# 只捕获数值本身，单位不参与比对，否则 float("189.0 元") 会抛异常
_NUM_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:元|块|天|小时|分钟|折|个|件|%)")
# 百分比：必须原样保留（「100% 精梳棉」不能被写成「大概率精梳棉」）
_PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
_SENTENCE_SPLIT = re.compile(r"[。！？!?\n；;]")


@dataclass
class FactCheckResult:
    passed: bool = True
    unsupported: list[str] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "unsupported": self.unsupported,
            "covered": self.covered,
            "reason": self.reason,
        }


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s and s.strip()]


def claim_numbers(sentence: str) -> list[str]:
    """只提取「数值 + 单位」型断言；订单号一类的标识符不算事实断言。"""
    return [m.group(1) for m in _NUM_PATTERN.finditer(sentence)]


def check_facts(
    text: str,
    *,
    evidence: list[Evidence] | None = None,
    facts: dict[str, Any] | None = None,
) -> FactCheckResult:
    """比对生成内容与证据/业务数据。

    - 带具体数值与单位的句子，必须能在证据或业务事实中找到对应数字；
    - 完全没有证据时，所有数值断言都判定为无依据。
    """
    result = FactCheckResult()
    evidence = evidence or []
    facts = facts or {}

    evidence_text = " ".join(e.snippet for e in evidence)
    evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence_text))
    fact_numbers: set[str] = set()
    for value in facts.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fact_numbers.add(f"{value:g}")
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    fact_numbers.add(f"{item:g}")
        elif isinstance(value, str):
            fact_numbers.update(re.findall(r"\d+(?:\.\d+)?", value))

    allowed = evidence_numbers | fact_numbers
    has_evidence = bool(evidence) or bool(facts)

    for sentence in split_sentences(text):
        numbers = claim_numbers(sentence)
        if not numbers:
            continue
        if not has_evidence:
            result.unsupported.append(sentence)
            continue
        # 数字需能在证据/事实中找到（189 与 189.0 视为同一数值）
        unmatched = [n for n in numbers if n not in allowed and f"{float(n):g}" not in allowed]
        if unmatched:
            result.unsupported.append(sentence)
        else:
            result.covered.append(sentence)

    # 反向校验：事实里的百分比必须在输出中原样出现。
    # 正向比对只查「输出有数字但事实没有」，查不出「事实有 100%、输出用
    # 「大概率」把它糊过去」—— 后者一个数字都不剩，正向看完全干净。
    missing_percent: list[str] = []
    normalized = text.replace(" ", "")
    for value in facts.values():
        if isinstance(value, str):
            for token in _PERCENT_PATTERN.findall(value):
                cleaned = token.replace(" ", "")
                if cleaned not in normalized:
                    missing_percent.append(cleaned)
    if missing_percent:
        result.unsupported.append(f"缺失百分比事实：{'、'.join(sorted(set(missing_percent)))}")

    result.passed = not result.unsupported
    if result.unsupported:
        result.reason = f"{len(result.unsupported)} 句缺少证据支撑"
    return result
