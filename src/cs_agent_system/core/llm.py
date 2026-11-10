"""统一 LLM 网关 + 多模型分级路由（架构文档 14 / 14.1）。

设计要点：
1. 业务代码只依赖 `LLMGateway.invoke`，不直接依赖任何厂商 SDK；
2. 按任务档位路由模型：判别类（P1 意图 / P2 槽位 / P3 情绪）走小模型，
   生成类（D3 话术）走大模型——全系统仅这 4 处产生模型调用；
3. 内置 `MockLLMGateway` —— 无需任何 API Key 即可跑通全链路与测试。
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# 导入即触发 `.env` 加载：即使调用方只 import 本模块，也能读到 LLM_API_KEY
from ..config import settings  # noqa: F401
from .intents import INTENTS

# 规则网关流式输出的块间隔（毫秒）。设 0 可让测试全速跑完。
# 真实网关不走这个值：它由服务端逐 token 推送决定节奏。
_MOCK_STREAM_DELAY_MS = int(os.getenv("MOCK_STREAM_DELAY_MS", "15").strip() or "15")


class TaskTier(str, Enum):
    """模型档位。"""

    SMALL = "small"  # 判别类：分类、抽取、分级、判定
    LARGE = "large"  # 生成类：话术、文案


@dataclass
class LLMResult:
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    model: str = "mock"
    confidence: float = 0.0
    tier: TaskTier = TaskTier.SMALL
    # 降级标记：真实模型不可用、退回规则网关时置 True。
    # 没有这个标记，调用方无法区分「模型输出」与「兜底输出」——
    # 两者在 text/confidence 上看起来一样，模型挂了系统却表现正常。
    degraded: bool = False
    degraded_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"text": self.text, "data": self.data, "confidence": self.confidence}


class LLMGateway(ABC):
    """模型网关抽象。所有 Agent 只能通过它访问模型。"""

    @abstractmethod
    def invoke(
        self,
        *,
        task: str,
        prompt: str,
        tier: TaskTier = TaskTier.SMALL,
        schema: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> LLMResult:
        """执行一次模型调用。"""

    def classify(self, *, task: str, prompt: str, context: dict[str, Any] | None = None) -> LLMResult:
        return self.invoke(task=task, prompt=prompt, tier=TaskTier.SMALL, schema={"type": "object"}, context=context)

    def generate(self, *, task: str, prompt: str, context: dict[str, Any] | None = None) -> LLMResult:
        return self.invoke(task=task, prompt=prompt, tier=TaskTier.LARGE, schema={"type": "object"}, context=context)

    def stream(
        self,
        *,
        task: str,
        prompt: str,
        tier: TaskTier = TaskTier.SMALL,
        context: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """流式调用：逐个返回文本增量。

        基类默认实现基于 `invoke()` 切分，保证**任何网关（含 Mock）都能流式**；
        真实网关覆写为 LangChain 的 `chain.stream()`。
        `stats` 是输出参数：流结束后回填 tokens / model / degraded，
        让调用方能在同一处完成记账（与 `call_llm` 的收口原则一致）。
        """
        result = self.invoke(task=task, prompt=prompt, tier=tier, context=context)
        if stats is not None:
            stats.update(
                tokens=result.tokens,
                model=result.model,
                degraded=result.degraded,
                degraded_reason=result.degraded_reason,
            )
        text = result.data.get("text") if task == "generate" else result.text
        text = str(text or result.text or "")
        for index in range(0, len(text), 8):
            # 规则网关没有真实的首字延迟：文本是同步算完的。
            # 这里按块之间加一个短间隔，让传输层真的逐块到达 ——
            # 否则所有分片会在同一瞬间涌到前端，看起来就是「一次性输出」。
            # 真实网关覆写本方法走 chain.stream()，不经过这里。
            time.sleep(_MOCK_STREAM_DELAY_MS / 1000.0)
            yield text[index : index + 8]


# —— Mock 实现：确定性、离线可跑 ——

_MOCK_INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    (name, spec.keywords) for name, spec in INTENTS.items() if spec.keywords
]

_ORDER_RE = re.compile(r"(?:订单号|单号)\s*[:：]?\s*([A-Za-z0-9]{6,})", re.IGNORECASE)
_SKU_RE = re.compile(r"(?:sku|商品编码)\s*[:：]?\s*([A-Za-z0-9\-]{4,})", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*元")
# 尺码既可能写作「M 码」也可能写作「码 M」
_SIZE_RE = re.compile(r"([XSML]{1,3}|\d{2})\s*码|码\s*[:：]?\s*([XSML]{1,3}|\d{2})", re.IGNORECASE)
_COLOR_RE = re.compile(r"(黑色|白色|灰色|蓝色|红色|绿色|米色|卡其)")
_DATE_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_QTY_RE = re.compile(r"(\d+)\s*件")

_REASON_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("quality", ("质量", "破损", "坏了", "瑕疵", "开裂")),
    ("size", ("尺码", "大小", "不合适", "偏大", "偏小")),
    ("logistics", ("物流", "快递", "配送", "迟迟", "没到")),
    ("wrong_item", ("发错", "错发", "漏发", "少发")),
    ("no_longer_needed", ("不想要", "不需要", "多余", "退掉")),
)

_RISK_SIGNALS: tuple[tuple[str, str], ...] = (
    ("12315", "监管投诉"),
    ("消协", "监管投诉"),
    ("曝光", "舆情"),
    ("起诉", "法律风险"),
    ("维权", "法律风险"),
    ("举报", "监管投诉"),
    ("工商", "监管投诉"),
    ("媒体", "舆情"),
    ("315", "监管投诉"),
)


class MockLLMGateway(LLMGateway):
    """确定性 Mock：意图分类、槽位抽取、情绪分级、话术生成都有可用实现。

    它不是「假的接口」，而是一个规则驱动的离线模型替身——让整套编排、
    保障、幂等、降级逻辑在没有网络时也能被完整测试。
    """

    def __init__(self, tier_models: dict[str, str] | None = None) -> None:
        self.tier_models = tier_models or {TaskTier.SMALL.value: "mock-small", TaskTier.LARGE.value: "mock-large"}
        self.calls: list[dict[str, Any]] = []

    # —— 公开的规则实现（Agent 也可直接复用，避免逻辑分散）——
    # 多意图连接词：复合请求通常由这些词或明显停顿引出
    _CONNECTIVES = ("同时", "另外", "顺便", "还有", "并且", "以及", "，", ",", "；", ";")

    @staticmethod
    def detect_intent(text: str) -> tuple[str, float, list[str]]:
        lowered = text.lower()
        scores: list[tuple[str, float, int]] = []
        for index, (intent, keywords) in enumerate(_MOCK_INTENT_RULES):
            matched = [kw for kw in keywords if kw in lowered]
            if not matched:
                continue
            # 权重 = 最长命中词长度 + 多关键词加成（长词更具体）
            weight = max(len(kw) for kw in matched) + 0.5 * (len(matched) - 1)
            scores.append((intent, weight, index))
        if not scores:
            return "CHITCHAT", 0.45, []
        scores.sort(key=lambda x: (-x[1], x[2]))
        top, top_weight, _ = scores[0]
        runner_up = scores[1][1] if len(scores) > 1 else 0
        # 置信度：命中强度与领先优势共同决定
        confidence = min(
            0.97,
            0.55 + 0.05 * min(top_weight, 5) + (0.08 if top_weight > runner_up else 0.0),
        )
        # 多意图：需要连接词 + 次优意图有独立强信号，避免把「怎么样」之类泛词误判为复合请求
        has_connective = any(c in text for c in MockLLMGateway._CONNECTIVES)
        multi: list[str] = []
        if has_connective:
            multi = [s[0] for s in scores[1:] if s[1] >= 3 and s[1] >= top_weight * 0.5]
        return top, round(confidence, 3), multi

    @staticmethod
    def extract_slots(text: str) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        m = _ORDER_RE.search(text)
        if m:
            slots["order_no"] = m.group(1)
        m = _SKU_RE.search(text)
        if m:
            slots["sku_id"] = m.group(1)
        m = _AMOUNT_RE.search(text)
        if m:
            slots["amount"] = float(m.group(1))
        m = _SIZE_RE.search(text)
        if m:
            slots["size"] = (m.group(1) or m.group(2)).upper()
        m = _COLOR_RE.search(text)
        if m:
            slots["color"] = m.group(1)
        m = _DATE_RE.search(text)
        if m:
            slots["expect_date"] = f"{m.group(1)}-{int(m.group(2)):02d}"
        m = _QTY_RE.search(text)
        if m:
            slots["quantity"] = int(m.group(1))
        for reason, keywords in _REASON_RULES:
            if any(kw in text for kw in keywords):
                slots["reason"] = reason
                break
        return slots

    @staticmethod
    def assess_emotion(text: str) -> dict[str, Any]:
        signals = [label for kw, label in _RISK_SIGNALS if kw in text]
        anger_words = ("生气", "气死", "太差", "垃圾", "无语", "坑", "骗", "恶心", "什么破")
        anger = sum(1 for w in anger_words if w in text)
        complaint = any(k in text for k in ("投诉", "差评", "举报", "退款", "维权", "骗人"))
        risk = "high" if signals else ("medium" if complaint or anger >= 2 else "low")
        sentiment = "negative" if anger or complaint else ("positive" if "谢谢" in text else "neutral")
        return {
            "sentiment": sentiment,
            "anger_score": min(1.0, anger / 3.0),
            "complaint_intent": complaint,
            "risk_level": risk,
            "escalation_signals": signals,
        }

    def invoke(
        self,
        *,
        task: str,
        prompt: str,
        tier: TaskTier = TaskTier.SMALL,
        schema: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> LLMResult:
        context = context or {}
        text = str(context.get("text") or prompt)
        self.calls.append({"task": task, "tier": tier.value, "text": text[:200]})

        if task == "intent":
            intent, confidence, multi = self.detect_intent(text)
            return LLMResult(
                text=intent,
                data={"intent": intent, "confidence": confidence, "multi": multi},
                tokens=len(text) // 4,
                model=self.tier_models[tier.value],
                confidence=confidence,
                tier=tier,
            )

        if task == "slots":
            slots = self.extract_slots(text)
            missing = list(context.get("missing") or [])
            return LLMResult(
                text=json.dumps(slots, ensure_ascii=False),
                data={"slots": slots, "missing": missing},
                tokens=len(text) // 4,
                model=self.tier_models[tier.value],
                confidence=0.9,
                tier=tier,
            )

        if task == "emotion":
            result = self.assess_emotion(text)
            return LLMResult(
                text=json.dumps(result, ensure_ascii=False),
                data=result,
                tokens=len(text) // 4,
                model=self.tier_models[tier.value],
                confidence=0.88,
                tier=tier,
            )

        if task == "generate":
            # 骨架优先用 facts_skeleton（D3 传的是「事实骨架」），兼容旧的 content 字段
            body = str(context.get("facts_skeleton") or context.get("content") or "")
            template = str(context.get("template") or "{content}")
            text = template.replace("{content}", body) if body else (body or template)
            return LLMResult(
                text=text,
                data={"text": text},
                tokens=len(body) // 3 + 32,
                model=self.tier_models[tier.value],
                confidence=0.85,
                tier=tier,
            )

        if task == "judge":
            # 判定类：事实一致性检查 —— 由 Q2 提供 claims 与 evidence
            claims = context.get("claims") or []
            evidence_text = " ".join(str(e.get("snippet", "")) for e in (context.get("evidence") or []))
            unsupported = [c for c in claims if str(c) not in evidence_text and not context.get("evidence")]
            ok = not unsupported
            return LLMResult(
                text="pass" if ok else "fail",
                data={"pass": ok, "unsupported": unsupported},
                tokens=len(evidence_text) // 4,
                model=self.tier_models[tier.value],
                confidence=0.9,
                tier=tier,
            )

        return LLMResult(text="", data={}, tokens=len(text) // 4, model=self.tier_models[tier.value], tier=tier)


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出里尽力取出一个 JSON 对象。

    真实模型常见的三种「不听话」输出都能容忍：
    1. 裸 JSON；
    2. ```json 围栏包裹；
    3. JSON 前后夹着「好的，结果如下」之类的自然语言。

    取不到就返回空 dict，由调用方走规则兜底——绝不因为解析失败而抛出异常。
    """
    if not text:
        return {}
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)]
    candidates.append(text)
    for raw in candidates:
        chunk = raw.strip()
        start = chunk.find("{")
        end = chunk.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            value = json.loads(chunk[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _usage_tokens(message: Any) -> int:
    """从 LangChain 消息（AIMessage / AIMessageChunk）读实际 token 用量。"""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        try:
            return int(usage.get("total_tokens") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def build_result(
    *,
    text: str,
    task: str,
    tokens: int,
    model: str,
    tier: TaskTier,
    degraded: bool = False,
    degraded_reason: str = "",
) -> LLMResult:
    """把「原始文本 + 用量」整理成业务契约 `LLMResult`。

    非流式 `invoke` 与流式收尾共用这一处解析，避免两条路径对
    JSON/隐藏字段的处理出现分叉（分叉就意味着同一句话在流式与非流式下
    得到不同结果，排障时极难定位）。
    """
    text = (text or "").strip()
    data = extract_json(text)
    if task == "generate" and not data.get("text"):
        # 模型直接给了一段话术（没按 JSON 输出）也要能用，否则会静默退回模板
        data["text"] = text
    try:
        confidence = float(data["confidence"])
    except (KeyError, TypeError, ValueError):
        # 模型未自报置信度 = 没有信号，而不是"应该挺有把握"。
        # 交给调用方按"无信号"处理（意图场景下会走澄清），不给伪精确的默认值。
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return LLMResult(
        text=text,
        data=data,
        tokens=max(0, int(tokens)),
        model=model,
        confidence=confidence,
        tier=tier,
        degraded=degraded,
        degraded_reason=degraded_reason,
    )


class LangChainLLMGateway(LLMGateway):
    """通过 LangChain ChatModel 访问真实模型。

    依赖 `langchain-openai`（或任意 LangChain ChatModel 实现）。
    小模型与大模型分开配置，实现分级路由。
    """

    def __init__(self, small_model: Any, large_model: Any | None = None) -> None:
        self.small_model = small_model
        self.large_model = large_model or small_model
        self._last_error: str = ""
        # 用 LangChain 组件组装调用链：Prompt 模板 → ChatModel → 输出解析。
        # prompt 作为**变量值**传入（不是模板本身），所以 prompt 里的 JSON
        # 花括号不会被当成模板变量解析。
        from langchain_core.output_parsers import StrOutputParser  # type: ignore
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore

        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "{prompt}"),
                ("human", "{context}"),
            ]
        )
        self._parser = StrOutputParser()

    def _model(self, tier: TaskTier) -> Any:
        return self.small_model if tier is TaskTier.SMALL else self.large_model

    def _payload(self, prompt: str, context: dict[str, Any] | None) -> dict[str, str]:
        return {
            "prompt": prompt,
            "context": json.dumps(context, ensure_ascii=False, default=str) if context else "{}",
        }

    @property
    def last_error(self) -> str:
        """最近一次真实调用失败的原因（空字符串表示正常）。"""
        return self._last_error

    @classmethod
    def from_env(cls) -> "LangChainLLMGateway | None":
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            return None
        try:
            from langchain_openai import ChatOpenAI  # type: ignore

            base_url = os.getenv("LLM_BASE_URL") or None
            try:
                timeout = float(os.getenv("LLM_TIMEOUT", "20"))
            except ValueError:
                timeout = 20.0
            small = ChatOpenAI(
                model=os.getenv("LLM_MODEL_SMALL", "deepseek-chat"),
                api_key=api_key,
                base_url=base_url,
                temperature=0,
                timeout=timeout,
                max_retries=1,
            )
            large = ChatOpenAI(
                model=os.getenv("LLM_MODEL_LARGE", "deepseek-chat"),
                api_key=api_key,
                base_url=base_url,
                temperature=0.3,
                timeout=timeout,
                max_retries=1,
            )
            return cls(small, large)
        except Exception:
            # 缺 langchain-openai / langchain-core，或模型客户端构造失败
            # → 交给 build_llm_gateway 回退 Mock，不要因网关初始化失败拖垮服务
            return None

    def invoke(
        self,
        *,
        task: str,
        prompt: str,
        tier: TaskTier = TaskTier.SMALL,
        schema: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> LLMResult:
        model = self._model(tier)
        # LangChain 组件拼链：Prompt 模板 → ChatModel（输出解析在拿到消息后做，
        # 因为 token 计量需要 AIMessage 上的 usage_metadata）
        chain = self._prompt_template | model

        # 网络/鉴权/超时失败一律降级到规则网关：客服链路不能因为模型不可用而中断
        try:
            message = chain.invoke(self._payload(prompt, context))
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            return self._fallback(task=task, prompt=prompt, tier=tier, context=context)

        self._last_error = ""
        text = self._parser.invoke(message).strip()
        tokens = _usage_tokens(message)
        if not tokens:
            tokens = (len(prompt) + len(text)) // 3
        return build_result(
            text=text,
            task=task,
            tokens=tokens,
            model=getattr(model, "model_name", None) or getattr(model, "model", "unknown"),
            tier=tier,
        )

    def stream(
        self,
        *,
        task: str,
        prompt: str,
        tier: TaskTier = TaskTier.SMALL,
        context: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """用 LangChain 的 `chain.stream()` 逐块产出。

        两种模式：
        - 判别类（intent/slots/emotion）：模型输出本身就是答案，直接逐块转发；
        - 生成类（generate）：期望是**话术正文**。若模型不听话、仍返回 JSON，
          先缓冲到流结束再解析出 `text` 字段，绝不让前端看到 JSON 原文。
        """
        model = self._model(tier)
        chain = self._prompt_template | model
        stats = stats if stats is not None else {}
        model_name = getattr(model, "model_name", None) or getattr(model, "model", "unknown")
        pieces: list[str] = []
        tokens = 0
        buffered = ""
        deciding = task == "generate"
        json_mode = False
        try:
            for chunk in chain.stream(self._payload(prompt, context)):
                piece = getattr(chunk, "content", "") or ""
                if not isinstance(piece, str):  # 部分模型返回 content 片段列表
                    piece = "".join(str(part) for part in piece) if isinstance(piece, list) else str(piece)
                if piece:
                    pieces.append(piece)
                chunk_tokens = _usage_tokens(chunk)
                if chunk_tokens:
                    tokens = chunk_tokens
                if deciding:
                    # 首块判定输出形态：`{` 开头当作 JSON，其余按话术直出
                    buffered += piece
                    stripped = buffered.lstrip()
                    if not stripped:
                        continue
                    if stripped.startswith("{"):
                        json_mode = True
                        deciding = False
                        buffered = ""
                    elif len(stripped) >= 6:
                        deciding = False
                        if buffered:
                            yield buffered
                        buffered = ""
                    continue
                if json_mode or not piece:
                    continue
                yield piece
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            fallback = self._fallback(task=task, prompt=prompt, tier=tier, context=context)
            stats.update(
                tokens=fallback.tokens,
                model=fallback.model,
                degraded=True,
                degraded_reason=fallback.degraded_reason,
            )
            # 已经流出去的增量无法收回，只能把兜底文本补在后面
            yield str(fallback.data.get("text") or fallback.text or "")
            return

        full = "".join(pieces)
        if json_mode:
            final_text = str(extract_json(full).get("text") or full)
            if final_text:
                yield final_text
        elif buffered:
            yield buffered
        if not tokens:
            tokens = (len(prompt) + len(full)) // 3
        stats.update(tokens=tokens, model=model_name, degraded=False, degraded_reason="")

    def _fallback(
        self,
        *,
        task: str,
        prompt: str,
        tier: TaskTier,
        context: dict[str, Any] | None,
    ) -> LLMResult:
        """真实模型不可用时，退回内置规则网关。

        必须打上 `degraded` 标记：兜底输出与模型输出在 text/confidence 上
        长得几乎一样，不打标记的话，调用方（乃至最终用户）会把
        「模型挂了」误当成「模型正常回答」。
        """
        result = MockLLMGateway().invoke(task=task, prompt=prompt, tier=tier, context=context)
        result.model = "mock-fallback"
        result.degraded = True
        result.degraded_reason = self._last_error or "真实模型不可用，已退回规则网关"
        return result


def build_llm_gateway() -> LLMGateway:
    """按环境变量选择网关。

    优先级：显式 USE_MOCK_LLM=1 → 强制离线 Mock；
    否则有 LLM_API_KEY 且 langchain-openai 可用 → 真实网关；否则回退 Mock。
    """
    if settings.use_mock_llm:
        return MockLLMGateway()
    gateway = LangChainLLMGateway.from_env()
    return gateway or MockLLMGateway()
