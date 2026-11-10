"""保障层（L3）：合规检查与事实校验的底层实现。"""

from .compliance import ComplianceResult, check_compliance
from .factcheck import FactCheckResult, check_facts, split_sentences

__all__ = [
    "check_compliance",
    "ComplianceResult",
    "check_facts",
    "FactCheckResult",
    "split_sentences",
]
