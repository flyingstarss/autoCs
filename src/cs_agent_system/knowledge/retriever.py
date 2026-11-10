"""知识检索（Q1 支撑）：混合检索 + 重排。

政策类查询需要关键词精确匹配，纯向量召回不足，因此这里用
BM25（关键词）+ 向量近似（字符 n-gram 余弦）混合召回，再重排。
证据必须带来源与生效日期，过期证据自动降权。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any

from ..core.envelope import Evidence

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    for chunk in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.extend(list(chunk))  # 中文按字切分，保证关键词召回
        else:
            tokens.append(chunk)
    return tokens


def ngrams(text: str, n: int = 2) -> set[str]:
    cleaned = re.sub(r"\s+", "", text.lower())
    return {cleaned[i : i + n] for i in range(max(0, len(cleaned) - n + 1))}


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    layer: str = "policy"  # policy / script / faq / case / product
    version: str = "v1"
    effective_from: date | None = None
    effective_to: date | None = None
    tags: tuple[str, ...] = ()
    source: str = ""

    def to_evidence(self, score: float) -> Evidence:
        return Evidence(
            source=self.source or self.doc_id,
            snippet=f"{self.title}：{self.content}",
            layer=self.layer,
            version=self.version,
            effective_from=self.effective_from,
            effective_to=self.effective_to,
            score=round(score, 4),
        )


class HybridRetriever:
    """BM25 + 字符 n-gram 向量的混合检索器。"""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokens = {d.doc_id: tokenize(f"{d.title} {d.content} {' '.join(d.tags)}") for d in documents}
        self._ngrams = {d.doc_id: ngrams(f"{d.title} {d.content}") for d in documents}
        self._df: Counter[str] = Counter()
        for tokens in self._tokens.values():
            self._df.update(set(tokens))
        self._avg_len = sum(len(t) for t in self._tokens.values()) / max(1, len(self._tokens))

    def _bm25(self, query_tokens: list[str], doc_id: str) -> float:
        tokens = self._tokens[doc_id]
        if not tokens:
            return 0.0
        tf = Counter(tokens)
        length = len(tokens)
        n_docs = len(self.documents)
        score = 0.0
        for token in set(query_tokens):
            if token not in tf:
                continue
            df = self._df.get(token, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            freq = tf[token]
            denom = freq + self.k1 * (1 - self.b + self.b * length / max(1.0, self._avg_len))
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def _vector(self, query: str, doc_id: str) -> float:
        q = ngrams(query)
        d = self._ngrams[doc_id]
        if not q or not d:
            return 0.0
        inter = len(q & d)
        if inter == 0:
            return 0.0
        return inter / math.sqrt(len(q) * len(d))

    def search(self, query: str, top_k: int = 3, layers: tuple[str, ...] | None = None) -> list[Evidence]:
        candidates = [d for d in self.documents if layers is None or d.layer in layers]
        if not candidates:
            return []
        query_tokens = tokenize(query)
        scored: list[tuple[float, Document]] = []
        for doc in candidates:
            bm25 = self._bm25(query_tokens, doc.doc_id)
            vec = self._vector(query, doc.doc_id)
            if bm25 == 0 and vec == 0:
                continue
            # 混合融合：归一化后加权
            max_bm25 = max((self._bm25(query_tokens, d.doc_id) for d in candidates), default=1.0) or 1.0
            fused = 0.6 * (bm25 / max_bm25) + 0.4 * vec
            scored.append((fused, doc))

        scored.sort(key=lambda x: -x[0])
        results: list[Evidence] = []
        for score, doc in scored[:top_k]:
            evidence = doc.to_evidence(score)
            if evidence.expired:
                # 过期证据降权而不是丢弃：仍可能需提示用户「政策已更新」
                evidence.score = round(evidence.score * 0.3, 4)
                evidence.source = f"{evidence.source}(已过期)"
            results.append(evidence)
        return results


# —— 内置知识库 ——
DEFAULT_DOCUMENTS: list[Document] = [
    Document(
        doc_id="policy-return-15d",
        title="退换货政策",
        content="自签收之日起 15 天内支持无理由退换货，商品需保持吊牌完整、不影响二次销售。定制类商品不支持无理由退换。",
        layer="policy",
        version="v3",
        effective_from=date(2026, 1, 1),
        effective_to=date(2026, 12, 31),
        tags=("退货", "换货", "无理由"),
        source="退换货政策 v3",
    ),
    Document(
        doc_id="policy-refund-sla",
        title="退款到账时效",
        content="退款申请通过后，款项按原支付渠道退回，到账时效为 3 个工作日，具体以支付渠道实际处理时间为准。",
        layer="policy",
        version="v2",
        effective_from=date(2026, 3, 1),
        effective_to=date(2026, 12, 31),
        tags=("退款", "到账", "时效"),
        source="退款时效策略 v2",
    ),
    Document(
        doc_id="policy-compensation",
        title="赔付标准",
        content="因质量问题的退换货可申请运费补偿，单笔最高补偿 200 元；超出部分需人工审批。",
        layer="policy",
        version="v1",
        effective_from=date(2026, 1, 1),
        effective_to=date(2026, 12, 31),
        tags=("赔付", "补偿", "运费"),
        source="赔付策略 v1",
    ),
    Document(
        doc_id="policy-coupon-stack",
        title="优惠券叠加规则",
        content="同一笔订单最多可使用 2 张优惠券，仅服饰与家居类目支持叠加，平台券与店铺券可同时使用，不可叠加的优惠以结算页为准。",
        layer="policy",
        version="v2",
        effective_from=date(2026, 5, 1),
        effective_to=date(2026, 12, 31),
        tags=("优惠券", "叠加", "满减"),
        source="活动规则 v2",
    ),
    Document(
        doc_id="policy-delivery",
        title="配送时效说明",
        content="现货商品默认 48 小时内发出，承诺送达时效为 3 天；遇大促或极端天气顺延，不作为承诺依据。",
        layer="policy",
        version="v2",
        effective_from=date(2026, 2, 1),
        effective_to=date(2026, 12, 31),
        tags=("配送", "时效", "发货"),
        source="配送规则 v2",
    ),
    Document(
        doc_id="policy-invoice",
        title="发票开具规则",
        content="订单完成后可申请电子普通发票或电子专用发票；专票需提供税号与开户行信息，订单完成后 90 天内可申请。",
        layer="policy",
        version="v1",
        effective_from=date(2026, 1, 1),
        effective_to=date(2026, 12, 31),
        tags=("发票", "开票", "抬头"),
        source="发票规则 v1",
    ),
    Document(
        doc_id="script-greeting",
        title="开场话术",
        content="您好～很高兴为您服务，请问有什么可以帮您？",
        layer="script",
        version="v5",
        tags=("开场", "问候"),
        source="话术库 v5",
    ),
    Document(
        doc_id="faq-size",
        title="尺码建议话术",
        content="这款卫衣是宽松版型，建议按平时穿着尺码选择；介于两个尺码之间时建议拍大一码。",
        layer="script",
        version="v3",
        tags=("尺码", "建议"),
        source="话术库 v3",
    ),
    Document(
        doc_id="faq-logistics-exception",
        title="物流异常话术",
        content="包裹目前在派送中，如超过承诺时效仍未收到，可直接申请售后或联系人工核实，无需重复下单。",
        layer="faq",
        version="v2",
        tags=("物流", "异常", "超时"),
        source="FAQ v2",
    ),
]


def build_default_retriever() -> HybridRetriever:
    return HybridRetriever(DEFAULT_DOCUMENTS)
