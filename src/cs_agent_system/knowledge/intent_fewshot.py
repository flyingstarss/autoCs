"""意图识别的 RAG 动态 Few-shot：示例库 + 相似度检索 + Prompt 片段生成。

为什么要有这个模块：
纯零样本（只给意图名和一句描述）时，模型的判断下限取决于它对电商口语的
先验；换个说法（「包裹卡在哪个环节了」）就可能判偏。在 Prompt 里动态塞入
几条「相似说法 → 已标注意图」的示例，等于每次都从示例库里检索出最贴近的
判断依据，效果比静态固定 few-shot 更稳（固定示例只覆盖少数说法）。

检索算法（无需外部依赖、可离线跑）：
- 字符 bigram 的 TF-IDF 向量 + 余弦相似度；
- IDF 由示例库自身统计：越常见的 n-gram（「怎么回事」「在哪里」）权重越低，
  真正有区分度的词（「尺码」「退款」「包裹」）权重越高；
- 低于相似度下限的示例一律不注入（宁缺毋滥）；
- 同一意图最多取 2 条：避免检索结果全是同一个意图，失去区分度。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from ..core.intents import INTENTS
from .retriever import ngrams


@dataclass(frozen=True)
class IntentExample:
    """一条已标注的口语说法。multi 只在确实存在第二个业务链路时填写。"""

    text: str
    intent: str
    multi: tuple[str, ...] = ()


# 示例库：每个意图覆盖多种口语表达（含关键词规则命不中的同义说法）。
# 新增意图时同步补几条，否则该意图在检索里永远排不上去。
EXAMPLES: tuple[IntentExample, ...] = (
    # —— 订单状态 ——
    IntentExample("我的订单到哪一步了", "ORDER_STATUS"),
    IntentExample("我买的东西现在什么情况", "ORDER_STATUS"),
    IntentExample("帮我看看订单进度", "ORDER_STATUS"),
    IntentExample("我的单发货了没有", "ORDER_STATUS"),
    IntentExample("那笔付款之后有动静没", "ORDER_STATUS"),
    IntentExample("我想查一下我的订单", "ORDER_STATUS"),
    IntentExample("我想看看刚才我取消了的那个订单", "ORDER_STATUS"),
    IntentExample("帮我查一下这个订单号", "ORDER_STATUS"),
    IntentExample("最新的一单怎么回事", "ORDER_STATUS"),
    # —— 物流 ——
    IntentExample("包裹卡在哪个环节了", "LOGISTICS"),
    IntentExample("东西寄到哪儿了", "LOGISTICS"),
    IntentExample("快递怎么一直不动", "LOGISTICS"),
    IntentExample("物流信息三天没更新", "LOGISTICS"),
    IntentExample("什么时候能送到", "LOGISTICS"),
    IntentExample("我买的东西还在路上吗", "LOGISTICS"),
    # —— 退换 ——
    IntentExample("这个尺码不合适想退掉", "RETURN_APPLY"),
    IntentExample("收到的东西有破损我要换", "RETURN_APPLY"),
    IntentExample("我不想要了怎么退", "RETURN_APPLY"),
    IntentExample("帮我申请退款", "RETURN_APPLY"),
    IntentExample("质量太差能退吗", "RETURN_APPLY", ("COMPLAINT",)),
    IntentExample("少发了一件怎么办", "RETURN_APPLY"),
    # —— 尺码建议 ——
    IntentExample("我身高165体重55穿什么码", "SIZE_ADVICE"),
    IntentExample("这条裤子选M还是L", "SIZE_ADVICE"),
    IntentExample("鞋子偏大还是偏小", "SIZE_ADVICE"),
    IntentExample("帮我推荐一下尺码", "SIZE_ADVICE"),
    IntentExample("尺码偏小能换吗", "SIZE_ADVICE", ("RETURN_APPLY",)),
    IntentExample("这个版型宽松吗买大一号吗", "SIZE_ADVICE"),
    # —— 库存 ——
    IntentExample("这个还有货吗", "STOCK_CHECK"),
    IntentExample("白色M码有库存吗", "STOCK_CHECK"),
    IntentExample("什么时候补货", "STOCK_CHECK"),
    IntentExample("是不是卖完了", "STOCK_CHECK"),
    IntentExample("现货还有没有", "STOCK_CHECK"),
    # —— 商品参数 ——
    IntentExample("这个面料是什么成分", "PRODUCT_QA"),
    IntentExample("衣服材质怎么样", "PRODUCT_QA"),
    IntentExample("这个能机洗吗", "PRODUCT_QA"),
    IntentExample("参数在哪里看", "PRODUCT_QA"),
    IntentExample("会不会起球缩水", "PRODUCT_QA"),
    # —— 优惠券 ——
    IntentExample("优惠券怎么用不了", "COUPON_USE"),
    IntentExample("满减券能叠加吗", "COUPON_USE"),
    IntentExample("我有券怎么下单", "COUPON_USE"),
    IntentExample("折扣什么时候结束", "COUPON_USE"),
    IntentExample("领的券在哪里用", "COUPON_USE"),
    # —— 支付与发票 ——
    IntentExample("可以开发票吗", "PAYMENT_INVOICE"),
    IntentExample("发票怎么申请", "PAYMENT_INVOICE"),
    IntentExample("付款失败了怎么回事", "PAYMENT_INVOICE"),
    IntentExample("支持哪些支付方式", "PAYMENT_INVOICE"),
    IntentExample("开电子发票要多久", "PAYMENT_INVOICE"),
    # —— 会员权益 ——
    IntentExample("会员有什么权益", "MEMBER_BENEFIT"),
    IntentExample("积分怎么兑换", "MEMBER_BENEFIT"),
    IntentExample("我的等级有什么用", "MEMBER_BENEFIT"),
    IntentExample("成长值怎么涨", "MEMBER_BENEFIT"),
    IntentExample("会员折扣怎么算", "MEMBER_BENEFIT"),
    # —— 投诉 ——
    IntentExample("我要投诉你们", "COMPLAINT"),
    IntentExample("再不解决我就打12315", "COMPLAINT"),
    IntentExample("我要曝光你们", "COMPLAINT"),
    IntentExample("服务太差了我要举报", "COMPLAINT"),
    IntentExample("你们这是欺诈我要维权", "COMPLAINT"),
    # —— 闲聊 ——
    IntentExample("你好", "CHITCHAT"),
    IntentExample("谢谢", "CHITCHAT"),
    IntentExample("在吗", "CHITCHAT"),
    IntentExample("今天天气不错", "CHITCHAT"),
    IntentExample("你是谁", "CHITCHAT"),
)

# 示例里的意图名必须都是合法意图：直接从注册表取，不另抄一份。
# 放在导入期校验，避免示例库写错一个意图名却只表现为「检索不中」。
VALID_INTENTS = frozenset(INTENTS)
_INVALID_EXAMPLES = [ex.text for ex in EXAMPLES if ex.intent not in VALID_INTENTS]
if _INVALID_EXAMPLES:
    raise ValueError(f"意图示例库包含未登记的意图：{_INVALID_EXAMPLES[:3]}")


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    if not common:
        return 0.0
    dot = sum(left[g] * right[g] for g in common)
    norm_l = math.sqrt(sum(w * w for w in left.values()))
    norm_r = math.sqrt(sum(w * w for w in right.values()))
    if not norm_l or not norm_r:
        return 0.0
    return dot / (norm_l * norm_r)


# TF-IDF 向量：字符 bigram 的 IDF 由示例库自身统计。
#
# 为什么必须带 IDF：纯余弦时「怎么回事」「在哪里」这类高频词会主导相似度，
# 实测会把「今天天气不错」召回到「是不是卖完了」。
# 加上 IDF 后，越常见的 n-gram 权重越低，真正有区分度的「尺码/退款/包裹」才有分量。
_DOC_COUNT = len(EXAMPLES)
_DF: dict[str, int] = {}
for _ex in EXAMPLES:
    for _gram in set(ngrams(_ex.text, 2)):
        _DF[_gram] = _DF.get(_gram, 0) + 1


def _tfidf(text: str) -> dict[str, float]:
    grams = ngrams(text, 2)
    return {gram: math.log(1 + _DOC_COUNT / (1 + _DF.get(gram, 0))) for gram in grams}


_EXAMPLE_VECTORS = tuple(_tfidf(example.text) for example in EXAMPLES)

# 相似度下限：低于它的示例不注入 Prompt。
# 宁可零样本，也不给模型看一条不相关的示例 —— 错误示例比没有示例更坏。
# 0.30 是实测调出的线：低于它时「怎么回事」这类高频搭配会把
# 「最新的一单怎么回事」误召回到「付款失败了怎么回事」（PAYMENT_INVOICE）。
_MIN_SIMILARITY = 0.30


def _fewshot_k() -> int:
    try:
        return max(0, int(os.getenv("INTENT_FEWSHOT_K", "4")))
    except ValueError:
        return 4


def _fewshot_enabled() -> bool:
    raw = os.getenv("INTENT_FEWSHOT_ENABLED", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def retrieve_examples(query: str, k: int | None = None, max_per_intent: int = 2) -> list[IntentExample]:
    """检索与用户本轮最相似的历史标注示例。

    返回空列表表示「没有足够相似的示例」——此时 Prompt 等同于零样本，
    不会因为强行塞入不相关示例而把模型带偏（相似度为 0 的示例一律不返回）。
    """
    top_k = _fewshot_k() if k is None else max(0, k)
    if top_k <= 0 or not query:
        return []
    query_vector = _tfidf(query)
    scored = [
        (example, _cosine(query_vector, vector)) for example, vector in zip(EXAMPLES, _EXAMPLE_VECTORS)
    ]
    scored = [(example, score) for example, score in scored if score >= _MIN_SIMILARITY]
    scored.sort(key=lambda item: -item[1])
    picked: list[IntentExample] = []
    per_intent: dict[str, int] = {}
    for example, _ in scored:
        if per_intent.get(example.intent, 0) >= max_per_intent:
            continue
        picked.append(example)
        per_intent[example.intent] = per_intent.get(example.intent, 0) + 1
        if len(picked) >= top_k:
            break
    return picked


def format_examples(examples: list[IntentExample]) -> str:
    """把检索结果拼成 Prompt 里的 few-shot 片段；无示例时返回空串。"""
    if not examples:
        return ""
    lines = ["【相似说法参考】以下是按字面相似度检索到的已标注示例，仅供判断意图时参照，本轮一律以用户原话为准："]
    for example in examples:
        if example.multi:
            extra = "、".join(example.multi)
            lines.append(f"- 「{example.text}」→ 主意图 {example.intent}，同时包含 {extra}")
        else:
            lines.append(f"- 「{example.text}」→ {example.intent}")
    return "\n".join(lines)


def build_fewshot(query: str) -> tuple[str, list[IntentExample]]:
    """对外入口：检索 + 格式化。

    返回 (Prompt 片段, 命中示例)。关闭开关或无相似示例时返回 ("", [])，
    此时调用方按零样本处理 —— 检索不到就绝不硬塞示例。
    """
    if not _fewshot_enabled():
        return "", []
    examples = retrieve_examples(query)
    return format_examples(examples), examples
