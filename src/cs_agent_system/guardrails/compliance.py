"""合规规则引擎（Q3 的底层）。

保障层有否决权：命中即阻断或改写，不依赖模型自觉。
这里的规则是确定性的、可解释的、可回归的。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ——— 敏感词：监管/舆情风险 ———
SENSITIVE_WORDS: tuple[str, ...] = (
    "12315", "消协", "消费者协会", "曝光", "起诉", "维权", "举报", "工商", "315", "媒体",
    "媒体曝光", "上央视", "315晚会",
)

# ——— 绝对化用语：广告法风险 ———
# 注意：不含「100%」「百分百」—— 成分含量（100% 精梳棉）是事实描述，
# 不是绝对化宣传。把它们当违禁词会触发 _sanitize 改写成「大概率精梳棉」，
# 既丢了事实又变成病句；含量准确性由事实校验（Q2）负责，不靠合规改写。
ABSOLUTE_WORDS: tuple[str, ...] = (
    "最好", "第一", "最佳", "最优", "顶级", "国家级", "世界级", "唯一",
    "永久", "绝对", "根治", "特效", "万能", "无副作用",
)

# ——— 虚假承诺：模型自行承诺时效/金额 ———
FALSE_PROMISE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"一定(到|到账|退|收到|发货|成功)", "确定性承诺"),
    # 允许「保证3个工作日到账」这类中间夹数字的表述
    (r"保证[^。！？，,]{0,10}(到账|到货|退货|退款|收到|发货|成功)", "保证性承诺"),
    (r"马上(到|到账|退|退到)", "无依据时效承诺"),
    (r"明天(一定)?(到|到账|退)", "无依据时效承诺"),
    (r"(最多|只要)\s*\d+\s*(小时|分钟)内(到|到账|退)", "自造时效承诺"),
    (r"全额(退|赔付)", "超出赔付上限的承诺"),
    (r"退你\s*\d+", "自造金额承诺"),
    (r"赔偿\s*\d+", "自造金额承诺"),
    (r"包退包换", "超出政策范围的承诺"),
    (r"(一定能|肯定能|保证能)(退|赔|到)", "越权承诺"),
)

# ——— 诱导话术：引导站外交易 ———
INDUCEMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(加|联系)\s*(我|我们)?\s*(微信|VX|vx|威信)", "诱导站外交易"),
    (r"微信\s*[0-9a-zA-Z_-]{6,}", "诱导站外交易"),
    (r"扫码", "诱导站外交易"),
    (r"转账(给我|给我方|过来)", "诱导站外交易"),
    (r"私下(给|转|付)", "诱导站外交易"),
)

# ——— 竞品贬损 ———
COMPETITOR_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(别家|其他家|同行)(都|全是|一律)", "贬损竞品"),
    (r"我们家(一定|绝对)(比|好于)", "绝对化比较"),
)


@dataclass
class ComplianceIssue:
    code: str
    message: str
    severity: str = "block"  # block / rewrite
    matched: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "severity": self.severity, "matched": self.matched}


@dataclass
class ComplianceResult:
    passed: bool = True
    issues: list[ComplianceIssue] = field(default_factory=list)
    rewritten_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "rewritten_text": self.rewritten_text,
        }


def _sanitize(text: str) -> str:
    """把绝对化用语降级为合规表达。"""
    replacements = {
        "最好": "较为合适",
        "最佳": "较合适",
        "最优": "较合适",
        "最好": "较为合适",
        "顶级": "品质较好",
        "国家级": "较高标准",
        "世界级": "较高标准",
        "永久": "长期",
        "绝对": "基本",
        "特效": "效果较好",
        "万能": "适用面较广",
        "无副作用": "温和",
        "唯一": "少有",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def check_compliance(text: str, *, allow_rewrite: bool = True) -> ComplianceResult:
    """Q3 合规检查：敏感词 / 绝对化用语 / 虚假承诺 / 诱导话术 / 贬损竞品。"""
    result = ComplianceResult()
    if not text:
        return result

    for word in SENSITIVE_WORDS:
        if word in text:
            result.issues.append(ComplianceIssue("sensitive", f"命中敏感词「{word}」", "rewrite", word))

    for word in ABSOLUTE_WORDS:
        if word in text:
            result.issues.append(ComplianceIssue("absolute", f"命中绝对化用语「{word}」", "rewrite", word))

    for pattern, label in FALSE_PROMISE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            result.issues.append(ComplianceIssue("false_promise", f"{label}：{m.group(0)}", "block", m.group(0)))

    for pattern, label in INDUCEMENT_PATTERNS:
        m = re.search(pattern, text)
        if m:
            result.issues.append(ComplianceIssue("inducement", f"{label}：{m.group(0)}", "block", m.group(0)))

    for pattern, label in COMPETITOR_PATTERNS:
        m = re.search(pattern, text)
        if m:
            result.issues.append(ComplianceIssue("competitor", f"{label}：{m.group(0)}", "rewrite", m.group(0)))

    if any(i.severity == "block" for i in result.issues):
        result.passed = False
    elif result.issues and allow_rewrite:
        result.rewritten_text = _sanitize(text)
        result.passed = True  # 改写后放行
    return result
