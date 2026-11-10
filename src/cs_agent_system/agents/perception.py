"""感知域：P1 意图识别 / P2 槽位抽取 / P3 情绪与风险 / P4 上下文组装。"""

from __future__ import annotations

import re
from typing import Any

from ..core.confidence import CLASSIFY, EXACT, PARTIAL, PERCEPTION, resolve
from ..core.envelope import AgentEnvelope, SessionContext
from ..core.intents import INTENTS, INTENT_SET_VERSION, get_intent, is_revision_request
from ..knowledge.intent_fewshot import build_fewshot
from .base import Unit


# 续接短语：用户中断长任务后回来时说的「继续」「接着」这类纯指代。
#
# 匹配要求「整句就是一个指代」，而不是句中偶然出现这两个字 ——
# 否则「继续看看别的商品多少钱」会被误拉回旧任务。
# 规则：去掉标点后，句子必须**完全等于**某个续接短语（允许带少量修饰）。
_RESUME_PHRASES = frozenset({
    "继续", "接着", "接着说", "继续说", "继续处理", "接着处理",
    "继续刚才", "继续刚才那个", "继续那个", "继续之前的",
    "刚才那个", "刚刚那个", "刚才的", "刚刚的", "还是那个", "之前那个",
    "刚才问的", "回到刚才", "回到那个", "接着刚才",
    "没说完", "刚那个事", "那件事", "刚才那个事",
    "接着刚才那个", "接着那个", "接着之前的", "接着刚才的",
})

# 句尾常见的客套后缀，去掉后仍按整句匹配
_RESUME_TAIL = ("吧", "哈", "啊", "呢", "了", "呀", "哦")


def _is_resume_phrase(text: str) -> bool:
    """判断本轮文本是否是「回到未完成任务」的续接信号。

    只认「整句就是指代」的情况。像「继续看看别的商品」这种带新诉求的句子
    不会被当成续接 —— 否则会把用户已经换了话题的请求错误拉回旧任务。
    """
    stripped = (text or "").strip().strip("？?。！!，,、~～ ")
    if not stripped:
        return False
    if stripped in _RESUME_PHRASES:
        return True
    # 允许「继续吧」「接着哈」这类客套尾巴
    for tail in _RESUME_TAIL:
        if stripped.endswith(tail):
            candidate = stripped[: -len(tail)].strip()
            if candidate in _RESUME_PHRASES:
                return True
    return False


# 各意图的业务词：用户在续接时可能直接说出业务内容（「退款那个」「物流的事」）。
# 命中即可定位到对应任务，不必依赖模型做指代消解。
_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "RETURN_APPLY": ("退款", "退货", "换货", "退换", "退"),
    "LOGISTICS": ("物流", "快递", "运单", "发货", "到哪", "什么时候到"),
    "ORDER_STATUS": ("订单", "单号", "状态", "签收"),
    "SIZE_ADVICE": ("尺码", "大小", "尺寸", "身高", "体重"),
    "STOCK_CHECK": ("库存", "有货", "现货", "还有货"),
    "PRODUCT_QA": ("参数", "材质", "成分", "功能"),
    "COUPON_USE": ("优惠券", "券", "折扣", "满减"),
    "PAYMENT_INVOICE": ("发票", "支付", "付款", "开票"),
    "MEMBER_BENEFIT": ("会员", "积分", "等级", "权益", "成长值"),
    "COMPLAINT": ("投诉", "举报", "12315", "曝光"),
}


# 订单号格式：SO + 数字（与 data.py 的订单号约定一致）
_ORDER_NO_RE = re.compile(r"\bSO\d{6,}\b", re.IGNORECASE)


# 用户对订单状态的筛选诉求 → 具体状态集合。
#
# 为什么需要：用户问「我还有哪些没发货的」时，系统不能只回一单 ——
# 必须按用户要的维度把订单筛出来。而且「没发货」是口语，
# 对应到系统里的状态是「待付款 / 待发货」两种。
#
# 这是单一事实源：感知层用它判断「用户要的是集合还是单据」，
# 执行层用它决定工具层按什么条件筛。
_STATUS_FILTERS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    # 「没发货 / 未发货 / 还没发货 / 待发货」→ 待付款 + 待发货
    (("没发货", "没有发货", "未发货", "还没发货", "没寄出", "没有寄出", "未寄出", "没出库"), ("待付款", "待发货")),
    # 「已取消 / 取消的 / 取消了」→ 已取消
    # 注意要覆盖「取消了」这种带「了」的过去式说法：
    # 用户说「刚才我取消了的那个订单」时，字面里并没有「取消的」三个连续字，
    # 只匹配「取消的」会漏掉 —— 于是又退回到旧槽位那一单。
    (("已取消", "取消的", "取消了", "取消掉", "取消过"), ("已取消",)),
    # 「退款中 / 退款的 / 在退款」→ 退款中
    (("退款中", "退款的", "在退款", "退款了"), ("退款中",)),
    # 「已签收 / 签收了 / 收到了」→ 已签收 + 已完成
    (("已签收", "签收了", "收到了", "签收的"), ("已签收", "已完成")),
    # 「已拒收 / 拒收了」→ 已拒收
    (("已拒收", "拒收了", "拒收的"), ("已拒收",)),
    # 「在途 / 路上 / 还没到 / 运输中」→ 已发货 / 运输中 / 派送中
    (("在途", "路上", "还没到", "运输中"), ("已发货", "运输中", "派送中")),
    # 「已完成 / 完成了」→ 已完成
    (("已完成", "完成的", "完结了"), ("已完成",)),
]

# 价格异议的口语信号：用户抱怨贵，而不是问材质。
# 单一事实源：感知层用它兜底意图，决策层用它选「价格异议」话术模板。
PRICE_OBJECTION_HINTS = ("太贵", "有点贵", "好贵", "贵了", "贵啊", "便宜点", "能便宜", "降点价", "买不起")

# 集合意图词：用户要的是「一批订单」而不是「某一单」。
#
# 这是问题 1 和问题 3 的共同根因所在：
# 用户问「我还有哪些没发货的商品」时，句子里没有任何订单号，
# 问的是集合；如果这时还沿用旧槽位里的某个 order_no，
# 系统就会去查那一单 —— 答非所问。
_COLLECTION_HINTS = ("哪些", "哪些订单", "还有哪些", "所有订单", "全部订单", "所有的", "都", "全部")

# 「查一下我的订单呢」「看看我的订单」：口语里没有「哪些」，但意思一样是看整份清单。
# 光靠上面的提示词识别不到，会被当成「查某一单」而只回最近一单。
_VIEW_VERBS = ("查一下", "查下", "查询一下", "查看", "看下", "看看", "查查", "查一查", "呢")

# 出现这些词说明用户问的是「某一单的状态/细节」，不是要清单。
# 判断顺序上必须先于「我的订单」：否则「我的订单状态」会被误判成清单。
_SINGLE_ORDER_HINTS = (
    "状态", "详情", "进度", "到哪", "物流", "快递", "运单", "签收",
    "取消", "退款", "订单号", "单号", "支付", "付款", "发票", "地址",
)

# 「这个订单」「那个单」「这个订单号」这类指代：用户指向的是某一单，
# 但句子里没给出订单号。
#
# 判断要点：有指代词 + 但没号码 → 系统不知道是哪一单，只能反问。
# 反例是「我想看看刚才我取消了的那个订单」——它有「取消了」这个状态线索，
# 能被状态筛选定位到，所以不反问（见下面的判定条件）。
_DEICTIC_ORDER_RE = re.compile(
    r"(?:这个|那个|这一|那一|该|刚才那个|刚才这个|刚刚那个)[^，。！？]{0,4}(?:订单号|单号|订单|单)"
)

# 「最新/最近/最后一单」：虽然也是指代，但它带明确的排序依据，
# 系统可以自己定位到最新一单，不需要反问 ——
# 用户回答「最新的那一单啊」之后再反问，就成了死循环。
_LATEST_ORDER_HINTS = ("最新", "最近", "最后一单", "最后一个单", "上一单", "最后那个单")

# 同一关键槽位最多追问 1 次；用户拒答则立刻停止追问。
# 不设上限的结果是死循环：「我要退货」→ 问原因 →「不想说！」→ 再问原因 → …
# 拿不到的信息应当交给人工补齐，而不是反复逼问用户。
_MAX_ASKS_PER_SLOT = 1
_DECLINE_HINTS = (
    "不想说", "不想讲", "不方便说", "不方便讲", "别问了", "不告诉你",
    "没什么好说的", "懒得说", "不知道", "随便吧",
)
# 回答追问的典型短句：含指代但不含业务词（「我最近买的那条啊」「就那个」）。
# 这类文本是在「填空槽位」，不续接就会被判成闲聊，任务直接断掉。
_SLOT_ANSWER_HINTS = (
    "最近", "最新", "最后", "那条", "那单", "这个订单", "那个订单",
    "这单", "刚买", "买的那", "买的那个", "就是它", "就这个", "上面那个",
)
# 退换原因的「有依据」判定：原文必须出现原因类词汇，
# 否则模型抽出的 reason 一律视为幻觉丢弃（宁可多问一句，不可凭空开单）。
_REASON_HINTS = (
    "质量", "破", "坏", "瑕疵", "色差", "尺码", "不合身", "不合适", "不喜欢",
    "错发", "漏发", "少发", "发错", "多拍", "买错", "原因", "问题", "没发货",
)


def _mentions_latest_order(text: str) -> bool:
    """判断用户是否用「最新/最近/最后一单」指明了要哪一单。"""
    s = text or ""
    # 必须有排序线索（「最近/最新/最后一单」）：没有它就不是在定位某一单。
    if not any(hint in s for hint in _LATEST_ORDER_HINTS):
        return False
    # 「最近买的那条啊」「最新的那件」：口语里常用「条/件/个」指代商品，
    # 只认「单」字会把最常见的说法排除掉，用户明明说清了是哪一单却被反问。
    return "单" in s or any(
        word in s for word in ("那条", "那件", "那个", "这个", "这件", "这条", "买的", "刚买")
    )


def detect_status_filter(text: str) -> list[str]:
    """从用户本轮文本里识别订单状态筛选诉求。

    命中返回状态列表；没命中返回空列表（= 不筛选，返回全部）。
    只在本轮文本里找 —— 用户第 1 轮说的话不能替第 3 轮做决定。
    """
    stripped = (text or "").strip()
    if not stripped:
        return []
    for keywords, statuses in _STATUS_FILTERS:
        if any(kw in stripped for kw in keywords):
            return list(statuses)
    return []


def wants_collection(text: str) -> bool:
    """判断用户本轮要的是「一批订单」而不是「某一单」。

    优先级：
    1. 本轮报了订单号 → 一定是查某一单（不管还说了别的什么）；
    2. 出现「状态/详情/物流/订单号…」等单笔线索 → 是问某一单的细节；
    3. 「我的订单」「查一下我的订单呢」 → 看清单；
    4. 「哪些/所有/全部」 → 看清单。
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if any(hint in stripped for hint in _COLLECTION_HINTS):
        return True
    if _ORDER_NO_RE.search(stripped):
        return False
    # 指代词（「那个订单」）说明用户心里已经有一张单，不是在要清单
    if _mentions_deictic_order(stripped):
        return False
    # 单笔线索优先于「我的订单」：否则「我的订单状态」会被当成清单
    if any(hint in stripped for hint in _SINGLE_ORDER_HINTS):
        return False
    if "我的订单" in stripped or "我的单" in stripped:
        return True
    return "订单" in stripped and any(verb in stripped for verb in _VIEW_VERBS)


def _mentions_deictic_order(text: str) -> bool:
    """判断本轮是否用指代指向「某一单」，但没给出号码。"""
    return bool(_DEICTIC_ORDER_RE.search(text or ""))


def _extract_order_no(text: str) -> str:
    """从用户本轮文本里直接抠出订单号。

    不依赖模型抽取 —— 订单号是强个人数据，必须由服务端从原文确定，
    模型可能漏抽、错抽（上次实测就出现过模型把 SO202610280001 写成别的）。
    """
    m = _ORDER_NO_RE.search(text or "")
    return m.group(0).upper() if m else ""


def _match_pending_task(
    tasks: list[dict[str, Any]],
    text: str,
) -> dict[str, Any] | None:
    """在未完成任务栈里找出用户这一轮指的是哪个任务。

    两级匹配（先精确后泛化）：
    1. **带业务词**：文本里出现某任务的业务词（「退款那个」「物流的事」）→ 精确命中；
       多个任务都命中时，取栈顶（最近更新的）那个。
    2. **纯指代**：整句就是「继续」「接着」这类 → 取栈顶任务。

    匹配不到返回 None —— 此时不该把用户拉回旧任务。
    """
    if not tasks:
        return None

    stripped = (text or "").strip().strip("？?。！!，,、~～ ")
    if not stripped:
        return None

    # 1) 带业务词：扫描栈，找到第一个（栈顶优先）业务词命中的任务
    for task in tasks:
        intent = str(task.get("intent") or "")
        keywords = _INTENT_KEYWORDS.get(intent, ())
        if any(kw in stripped for kw in keywords):
            return task

    # 2) 报了订单号：用户多半是在补某个需要订单号的任务的槽位。
    #    不能只靠业务词命中 —— 「订单号 SO123，质量太差」里没有「退款」二字，
    #    但它显然是在填退款任务的 order_no。
    if _ORDER_NO_RE.search(stripped):
        for task in tasks:
            if "order_no" in (task.get("missing") or []):
                return task

    # 3) 回答上一轮的追问：文本里出现指代（「我最近买的那条」）或明确拒答
    #    （「不想说」）。这类回答不含业务词，但显然是在填/拒填空槽位；
    #    不续接就会被模型判成 CHITCHAT，回一句寒暄后任务彻底断掉。
    if len(tasks) == 1 and (
        any(hint in stripped for hint in _SLOT_ANSWER_HINTS)
        or any(hint in stripped for hint in _DECLINE_HINTS)
    ):
        return tasks[0]

    # 4) 纯指代：取栈顶（最近更新的那个）
    if _is_resume_phrase(stripped):
        return tasks[0]

    return None


# 取消短语：用户放弃当前未完成任务（「算了不退了」「不查了」「取消」）。
# 命中后要把任务从栈里移除，否则用户说「继续」还会被拉回已放弃的事。
_CANCEL_PHRASES = (
    "算了", "不退了", "不退了不退了", "不查了", "不问了", "不弄了",
    "撤销", "不办了", "先不", "不继续了", "不搞了",
    "不想弄了", "不申请了", "不查物流了", "不要了", "停一下",
)

# 「取消」二字的歧义：既可以是「我要取消订单」（放弃动作），
# 也可以是「我取消了的那个订单」（描述历史状态）。
# 后者是查询诉求，不能被当成放弃当前任务 —— 否则用户想看自己取消过的单，
# 系统却把任务栈清空并回一句寒暄。
_CANCEL_PAST = ("取消的", "取消了", "取消过", "已取消", "取消掉")


def _is_cancel_phrase(text: str) -> bool:
    """判断用户是否在放弃当前任务。

    只认明确带「取消/算了/不弄了」这类信号的句子。
    「先不」「不要了」这类有歧义（「先不退款了」vs「先看看别的」），
    但用户明确说了取消意图时，以用户为准更符合直觉。
    """
    stripped = (text or "").strip().strip("？?。！!，,、~～ ")
    if not stripped:
        return False
    if any(phrase in stripped for phrase in _CANCEL_PHRASES):
        return True
    # 「取消」单独判定：描述已取消状态的句子不算放弃任务
    if "取消" in stripped and not any(past in stripped for past in _CANCEL_PAST):
        return True
    return False


def _as_confidence(value: Any, default: Any = 0.0) -> float:
    """把模型给的置信度安全地收敛到 0~1；脏数据不抛异常。

    默认值是 0.0（未给出信号 = 不可信），不是 0.5：
    0.5 会让"模型没把握"看起来像"刚刚够用"，从而静默放行。
    """
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        try:
            return max(0.0, min(1.0, float(default)))
        except (TypeError, ValueError):
            return 0.0


# 封闭意图菜单：真实模型必须看到枚举，否则会自造意图名 → 被降级成 CHITCHAT
_INTENT_MENU = "\n".join(f"- {name}：{spec.description}" for name, spec in INTENTS.items())

_PROMPT_INTENT = f"""你是电商/私域客服系统的意图识别器。只能从下面的封闭意图集中选择一个主意图，禁止自造意图名：

{_INTENT_MENU}

判定规则：
1. 与业务无关、纯寒暄、或实在判断不出来 → CHITCHAT；
2. 一句话里有多个可独立执行的诉求（如「退款 + 查物流」）→ 主诉求放 intent，其余放 multi；
3. 同一处理能力不要拆成多意图：例如「尺码咨询 + 库存查询」都属于商品规格（同一工具），
   只保留最贴合的主意图，multi 留空；multi 只放真正需要另一条业务链路的诉求；
4. 用户在控诉、威胁监管举报或要曝光 → COMPLAINT；
5. confidence 表示你对 intent 的把握（0~1），不确定就给低分，不要硬猜。

指代消解（重要）：
- 只有当**本轮文本本身没有明确业务对象**时（例如「继续」「接着」「刚才那个」这类纯指代），
  才用最近对话来推断它在说哪件事：如果最近对话里有一个还没办完的业务诉求，就选那个意图；
- 如果本轮文本**自带明确业务对象**（提到价格、尺码、物流、订单、退款、发票等），
  一律以本轮为准，不要被历史里的其他任务带偏 —— 用户很可能只是临时插问一句；
- 只有在最近对话里确实没有任何待续的业务诉求时，才判 CHITCHAT。

只输出 JSON，不要任何解释文字：
{{"intent": "<意图名>", "confidence": <0~1>, "multi": ["<意图名>", ...]}}"""

# 负面情绪线索词：主要是反讽和吐槽的口语。
# 为什么用规则而不是只靠模型：「喜欢个鬼」「呵呵」这类反讽，模型常判成中性，
# 结果就是给正在生气的用户回一句「您好～很高兴为您服务」。
_NEGATIVE_HINTS = (
    "个鬼", "垃圾", "恶心", "坑爹", "离谱", "无语", "呵呵", "敷衍",
    "太差", "好差", "失望", "气死", "烦死", "骗人", "忽悠",
)

_PROMPT_EMOTION = """你是电商客服的情绪与风险分级器。只输出 JSON：
{"sentiment": "positive|neutral|negative", "anger_score": <0~1>, "complaint_intent": <true|false>, "risk_level": "low|medium|high", "escalation_signals": ["监管投诉|舆情|法律风险", ...]}

判定标准：
- 提及 12315 / 消协 / 工商 / 起诉 / 举报 / 曝光 / 维权 → risk_level=high，并写入 escalation_signals；
- 明确投诉、差评、索赔，或情绪强烈负面 → medium 或 high；
- 普通不满、催促 → low / medium；
- 礼貌、感谢 → positive / low。

只输出 JSON，不要解释文字。"""


class IntentClassifier(Unit):
    """P1 意图识别单元：把用户表达映射到封闭意图集，输出意图 + 置信度。

    单元性质：**调用大模型**（全系统仅 4 个此类单元）。
    置信度 < 阈值时不猜测，走澄清。
    """

    unit_id = "P1"
    domain = "perception"
    description = "意图识别"
    kind = "agent"
    uses_llm = True

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        text = str(env.payload.get("text", ""))
        # 最近对话要够长才能做指代消解：用户说「继续」时，模型必须看到
        # 「上一轮用户问了什么、助手问了什么」才能知道该续哪件事。
        # 只给最近 3 条（user/agent 交替）往往只有一轮半，信息不足。
        history = list(session.history.recent_turns[-6:])
        # 续接线索只给「有没有待续任务」这个弱信号，不直接给出旧意图名 ——
        # 直接给意图名会诱导模型把新问题也往旧任务上拽。
        has_pending = bool(session.pending_tasks)
        # RAG 动态 Few-shot：从示例库检索与本轮最相似的说法，拼进 Prompt。
        # 检索不中（相似度为 0）时退回零样本 —— 不硬塞无关示例。
        fewshot_block, fewshot_examples = build_fewshot(text)
        prompt = f"{_PROMPT_INTENT}\n\n{fewshot_block}" if fewshot_block else _PROMPT_INTENT
        result = self.rt.call_llm(
            unit_id=self.unit_id,
            task="intent",
            prompt=prompt,
            context={
                "text": text,
                "history": history,
                "has_pending_task": has_pending,
            },
        )
        data = result.data or {}
        intent = str(data.get("intent") or "CHITCHAT").strip().upper()
        if intent not in INTENTS:  # 意图集封闭，模型输出了未知意图 → 归到闲聊
            intent = "CHITCHAT"
        # 模型自报置信度优先；网关层 confidence 次之；都没有则视为无信号（0.0）
        confidence = _as_confidence(data.get("confidence"), result.confidence or 0.0)
        # multi 要先归一化：下面的续接兜底要拿它判断「模型是否已认出任务意图」
        multi = [str(i).strip().upper() for i in (data.get("multi") or [])]
        multi = [i for i in multi if i in INTENTS and i != intent]

        # —— 续接兜底 ——
        # 模型判成 CHITCHAT（常见于「继续」「接着」这类纯指代），
        # 但任务栈里挂着未完成任务时，应该回到那个任务而不是闲聊。
        # 这是「长任务中断后继续」的关键一步。
        resumed = False
        if session.pending_tasks:
            match = _match_pending_task(session.pending_tasks, text)
            if match:
                matched_intent = str(match.get("intent") or "")
                # 三种情形都以任务意图为准：
                # 1) 模型判成 CHITCHAT —— 它没看出来；
                # 2) 模型把任务意图放进了 multi —— 它其实认出来了，只是排错了主次
                #    （「订单号 xxx，质量太差」常被判成 COMPLAINT + multi=RETURN_APPLY）；
                # 3) 模型判成别的意图且把握不高 —— 用户的槽位会白填。
                # 情形 2/3 用「模型是否已把任务意图列进候选」来判定，
                # 而不是靠置信度阈值 —— 模型的置信度在复合场景下并不可靠。
                model_knows = matched_intent in (multi or []) or matched_intent == intent
                if intent == "CHITCHAT" or model_knows or confidence < 0.75:
                    intent = matched_intent
                    # 续接意图的把握来自「确实有未完成任务」这一事实，
                    # 但仍不能给满信心：用户可能只是想换个话题。
                    confidence = max(confidence, 0.70)
                    resumed = True

        # —— 任务取消 ——
        # 用户放弃当前任务（「算了不退了」）：模型往往会判成 CHITCHAT 或
        # 业务意图（它只看到「不退了」这三个字）。这里显式识别取消信号，
        # 让编排器把任务从栈里摘掉 —— 否则用户说「继续」还会被拉回已放弃的事。
        cancelled = False
        if _is_cancel_phrase(text) and session.pending_tasks:
            cancelled = True
            intent = "CHITCHAT"
            confidence = max(confidence, 0.80)

        # —— 意图兜底（确定性）——
        # 模型偶尔把指向明确的句子判成闲聊：实测「我想看看刚才我取消了的那个订单」
        # 会被判成 CHITCHAT，于是回了句「请问有什么可以帮您」。
        # 句子里有订单号或明说「订单」时，业务对象已经确定，
        # 属于订单查询，不该用寒暄话术打发用户。
        if intent == "CHITCHAT" and not cancelled and not resumed:
            has_order_signal = (
                bool(_ORDER_NO_RE.search(text))
                or "订单" in text
                or bool(detect_status_filter(text))
            )
            if has_order_signal:
                intent = "ORDER_STATUS"
                confidence = max(confidence, 0.80)

        # —— 价格异议兜底 ——
        # 「这个卫衣也太贵了吧」模型给 PRODUCT_QA 但置信度常落在 0.5~0.6，
        # 会被路由判成「意图不明」去反问「问参数还是问尺码」—— 答非所问。
        # 句子里有明确的价格抱怨，业务对象也清楚，直接按商品咨询处理，
        # 由 D3 用价格异议模板回应（共情 + 参数价值 + 优惠/平替出路）。
        if not cancelled and not resumed and any(hint in text for hint in PRICE_OBJECTION_HINTS):
            if intent == "CHITCHAT" or confidence < 0.75:
                intent = "PRODUCT_QA"
                confidence = max(confidence, 0.85)

        out = env.child(self.unit_id, intent=intent, confidence=confidence)
        out.payload = {
            "intent": intent,
            "confidence": confidence,
            "multi": multi,
            "candidates": [intent, *multi],
            "intent_set_version": INTENT_SET_VERSION,
            # 本轮是否走了续接兜底：编排器据此决定是否沿用未完成任务的槽位
            "resumed": resumed,
            # 本轮是否是取消任务：编排器据此清空任务栈
            "cancelled": cancelled,
            # RAG 动态 Few-shot 的可观测信息：检索到了哪些示例，便于排查判偏
            "few_shot_intents": [example.intent for example in fewshot_examples],
            "few_shot_texts": [example.text for example in fewshot_examples],
        }
        out.evidence = []
        return out


class SlotExtractor(Unit):
    """P2 槽位抽取单元：缺什么比抽到什么更重要。

    单元性质：**调用大模型**。
    订单号优先从身份锚定上下文取；对话文本抽取的订单号只作候选，需二次核验。
    """

    unit_id = "P2"
    domain = "perception"
    description = "槽位抽取"
    kind = "agent"
    uses_llm = True

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        text = str(env.payload.get("text", ""))
        primary = env.intent or session.intent.primary
        # 本轮文本里明确出现的订单号：用户口述的号码最算数，
        # 必须优先于会话里的旧槽位（用户换单了就得跟新单走）。
        explicit_order_no = _extract_order_no(text)
        # 复合请求：除主意图外，multi 里的意图声明的槽位也要一起抽。
        # 否则「退款 + 查物流」这类请求里，主意图若是 LOGISTICS，
        # 退换原因会被槽位白名单滤掉，E3 只能回「缺槽位」。
        involved = [primary, *[i for i in (session.intent.multi or []) if i in INTENTS and i != primary]]
        specs: list[Any] = []
        seen: set[str] = set()
        for name in involved:
            spec = get_intent(name)
            for item in (spec.slot_specs if spec else ()):
                if item.name not in seen:
                    seen.add(item.name)
                    specs.append(item)

        slot_menu = "\n".join(
            f"- {s.name}（{s.description}，{'必填' if s.required else '可选'}"
            + (f"，别称：{'/'.join(s.aliases)}" if s.aliases else "")
            + "）"
            for s in specs
        ) or "- （当前意图不需要业务槽位）"
        prompt = (
            f"你是电商客服的槽位抽取器。本轮涉及意图：{'、'.join(involved)}，只允许抽取下列槽位：\n{slot_menu}\n\n"
            "规则：\n"
            "1. 用户没说的槽位不要编造，宁可留空让下一轮追问；\n"
            "2. 金额、数量输出数字；尺码输出 S/M/L/XL 或数字码；日期输出 YYYY-MM-DD；\n"
            '3. 只输出 JSON：{"slots": {...}, "missing": ["必填但缺失的槽位名"]}'
        )
        result = self.rt.call_llm(
            unit_id=self.unit_id,
            task="slots",
            prompt=prompt,
            context={
                "text": text,
                "intent": primary,
                "multi": involved[1:],
                "known_slots": session.slots,
                "missing": session.missing,
            },
        )
        data = result.data or {}
        raw_slots: dict[str, Any] = dict(data.get("slots") or {})
        # 只收本轮意图声明过的槽位，防止模型把自由字段灌进会话态
        allowed = {s.name for s in specs}
        extracted: dict[str, Any] = {k: v for k, v in raw_slots.items() if k in allowed}

        # —— 槽位失效判定（B 方案）——
        # 旧槽位跨轮留存，但用户可能换话题、换对象。判定规则：
        # 1. 本轮文本里明确带了订单号 → 该订单号覆盖旧槽位（用户说的最算数）；
        # 2. 本轮意图与旧槽位的来源意图不同 → 旧槽位失效
        #    （查订单状态时填的 order_no，不能带到「我还有哪些没发货的」里）；
        # 3. 意图相同 → 旧槽位保留（同意图多轮补槽位仍可用）。
        #
        # 判定不了来源意图（旧槽位没有 meta）时，保守保留：
        # 宁可多问一句，也不能凭空丢掉用户已确认的订单号。
        stale_slots: list[str] = []
        # 本轮用户要的是「一批订单」还是「某一单」。
        # 要一批 → 旧槽位里的单个 order_no 不该生效（否则就变成只查那一单了）；
        # 要某一单 → 旧 order_no 可以延续（用户可能只是在补充信息）。
        collection = wants_collection(text)
        for key in list(session.slots.keys()):
            meta = (session.slot_meta or {}).get(key) or {}
            src_intent = str(meta.get("intent") or "")
            src_turn = meta.get("turn")
            if key == "order_no":
                # 规则 0：本轮要的是集合 → 旧单号作废
                if collection:
                    stale_slots.append(key)
                    continue
                # 规则 1：本轮明确说了订单号 → 旧值作废
                if explicit_order_no and explicit_order_no != session.slots.get(key):
                    stale_slots.append(key)
                    continue
                # 规则 2：意图切换 → 旧值作废
                if src_intent and src_intent != primary:
                    stale_slots.append(key)
                    continue
                # 规则 3：本轮表达了状态筛选（「没发货的」「已取消的」）
                #        → 用户在问一类订单，不是问某个具体单 → 旧单号作废
                if detect_status_filter(text):
                    stale_slots.append(key)
                    continue
            elif src_intent and src_intent != primary:
                # 新意图同样声明了这个槽位 → 用户多半在追问同一个对象
                # （问完「蓝牙耳机」详情，紧接着问「库存是多少」）。
                # 此时清掉槽位等于把指代丢了，下一轮工具层只能拿到空参数。
                if key in allowed:
                    continue
                stale_slots.append(key)
                continue
            elif src_turn is not None and isinstance(src_turn, int) and src_turn < session.turn_count:
                # 同意图但隔了很多轮仍算作旧值：用户很可能已经换了单子
                if session.turn_count - int(src_turn) >= 2:
                    stale_slots.append(key)
                    continue

        # 本轮明确给出的订单号：直接采信（还要过下面的身份核验）
        if explicit_order_no:
            extracted["order_no"] = explicit_order_no

        # 失效槽位必须同时从本轮抽取结果里剔除。
        #
        # 为什么：抽取时把 known_slots 给了模型，模型看到会话里有 order_no
        # 常常原样抄回来（「抽取」变成「回显」）。如果只从旧槽位里剔除、
        # 不从 extracted 里剔除，刚判失效的单号又被抄回来，失效判定等于白做。
        #
        # 例外：本轮用户亲口报的单号 —— 旧值失效的原因是「用户换单了」，
        # 新值正是用户要的那一单，不能被一起剔掉。
        for key in stale_slots:
            if key == "order_no" and explicit_order_no:
                continue
            extracted.pop(key, None)

        # 槽位合并：本轮抽取值 > 未失效的旧槽位 > 任务栈底座
        base_slots = {k: v for k, v in session.slots.items() if k not in stale_slots}
        for task in session.pending_tasks or []:
            if str(task.get("intent") or "") == primary:
                for k, v in (task.get("slots") or {}).items():
                    # 任务栈里的槽位同样要过失效判定：否则刚判定作废的 order_no
                    # 会被任务底座重新塞回来，失效判定等于没做。
                    if k in stale_slots:
                        continue
                    base_slots.setdefault(k, v)
                break
        slots = dict(base_slots)
        slots.update({k: v for k, v in extracted.items() if v not in (None, "", [])})

        # 强个人数据：订单号必须经身份核验
        if "order_no" in extracted and any(s.name == "order_no" and s.verify for s in specs):
            from ..registry.data import find_order

            if find_order(session.user.id, str(extracted["order_no"])) is None:
                # 口述订单号不可信：标记为待核验，不直接采信
                slots.pop("order_no", None)
                extracted["order_no_unverified"] = extracted["order_no"]
                # 关键：用户明确给了单号但查不到 → 记录下来。
                # 工具层要据此明确回答「未查询到属于您的订单」，
                # 绝不能回退到「最近一单」—— 那等于回答了另一个问题。
                explicit_order_no = str(extracted["order_no"])

        # 「最新的那一单」「最近那个订单」：指代 + 有确定的排序依据 → 直接定位到最新一单。
        # 用户已经把「是哪一单」说清楚了（按时间最新），再反问就变成死循环 ——
        # 用户答「最新的那一单啊」，系统又问「您说的是哪一单」。
        if (
            slots.get("order_no") in (None, "", [])
            # 用户说「最近那单」但没给完整单号：即使模型抽出过一个号、
            # 且核验失败，也应按最新一单兜底。否则 explicit 标记会让系统
            # 反问一个已经说清楚的问题，最后还可能误判成「追问无果」转人工。
            and (not explicit_order_no or not _ORDER_NO_RE.search(text))
            and _mentions_latest_order(text)
        ):
            from ..registry.data import list_user_orders

            latest_orders = list_user_orders(session.user.id, limit=1)
            if latest_orders:
                slots["order_no"] = latest_orders[0].order_no

        required = [s.name for s in specs if s.required]
        # 自由文本槽位的幻觉防护：模型常把用户压根没提过的「退换原因」凭空填上，
        # 于是 missing 瞬间变空、跳过澄清直接执行 —— 结果是一张没有原因的售后单，
        # 用户还会收到一句风马牛不相及的寒暄。原文里没有原因相关词就不采信。
        #
        # 例外：用户说「把退款金额改一下」这类改额请求时，原因是在指代上一轮，
        # 不是本轮没提 —— 清掉它只会让系统反问一个用户早就答过的问题。
        if (
            slots.get("reason")
            and not is_revision_request(text)
            and not any(hint in text for hint in _REASON_HINTS)
        ):
            slots.pop("reason", None)
            extracted.pop("reason", None)
        missing = [name for name in required if slots.get(name) in (None, "", [])]

        # 指代但没给号码（「帮我查一下这个订单号」）：
        # 用户指的是具体某一单，而系统手上没有可用的订单号。
        # 这时不能猜「最近一单」—— 那很可能不是用户说的那一单；
        # 也不能回一长串列表 —— 用户要的是单笔。
        # 唯一的正确做法是反问「您说的是哪一单」。
        clarify_hint = ""
        if (
            _mentions_deictic_order(text)
            and not explicit_order_no
            and slots.get("order_no") in (None, "", [])
            and not detect_status_filter(text)
        ):
            missing = ["order_no"]
            clarify_hint = "您说的是哪一单？麻烦把订单号发给我，我马上帮您查。"

        # —— 追问策略：拒答或同一槽位已问过一次 → 不再重复追问 ——
        # 换意图后旧意图的追问次数不再适用，先清零。
        if session.prev_intent and session.prev_intent != primary:
            session.ask_counts.clear()
        force_handoff = ""
        if missing:
            slot_name = missing[0]
            asked = int((session.ask_counts or {}).get(slot_name, 0))
            if any(hint in text for hint in _DECLINE_HINTS):
                force_handoff = f"用户拒绝提供关键信息「{slot_name}」"
                missing = []
            elif asked >= _MAX_ASKS_PER_SLOT:
                force_handoff = f"关键信息「{slot_name}」已追问一次仍未获得"
                missing = []
            else:
                session.ask_counts[slot_name] = asked + 1
        else:
            session.ask_counts.clear()

        # 槽位抽取本身是确定性的；「有槽位」还是「缺槽位」都算正常产出。
        # 只有当模型给的槽位全部非法（被白名单滤空）时才是部分命中。
        outcome = EXACT if (extracted or not specs) else PARTIAL
        out = env.child(self.unit_id, intent=primary, confidence=resolve(outcome, PERCEPTION))
        out.slots = slots
        out.missing = missing
        # 追问无果/用户拒答：写进信封字段（而不是只放 payload）。
        # P4 会重建 payload，放 payload 里会在上下文组装时被丢掉。
        out.need_handoff = bool(force_handoff)
        out.payload = {
            "slots": slots,
            "missing": missing,
            "extracted": extracted,
            # 追问无果/被拒答 → 转人工补齐，编排器据此改走人工闸门
            "force_handoff": force_handoff,
            # 本轮明确给出的订单号（含核验失败的）：工具层据此区分
            # 「用户指定了单号但查不到」与「用户没指定单号」
            "explicit_order_no": explicit_order_no,
            # 失效的旧槽位：编排器据此清理会话态里的陈旧值
            "stale_slots": stale_slots,
            # 澄清话术（指代未给号码时用）：编排器澄清节点直接采用
            "clarify_hint": clarify_hint,
        }
        self.rt.log(
            self.unit_id,
            "slots",
            f"missing={missing} handoff={'yes' if force_handoff else 'no'}"
            f" asks={dict(session.ask_counts or {})} txt={text[:12]}",
            confidence=float(out.confidence or 0),
        )
        return out


class EmotionClassifier(Unit):
    """P3 情绪与风险识别单元。

    单元性质：**调用大模型**。
    规则先行：监管关键词命中直接提升风险等级，不等模型。
    """

    unit_id = "P3"
    domain = "perception"
    description = "情绪与风险分级"
    kind = "agent"
    uses_llm = True

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        text = str(env.payload.get("text", ""))
        result = self.rt.call_llm(
            unit_id=self.unit_id,
            task="emotion",
            prompt=_PROMPT_EMOTION,
            context={"text": text, "history_emotion": session.emotion.model_dump()},
        )
        data = result.data or {}
        risk = str(data.get("risk_level", "low")).strip().lower()
        if risk not in ("low", "medium", "high"):
            risk = "low"
        signals = [str(s) for s in (data.get("escalation_signals") or []) if s]
        sentiment = str(data.get("sentiment", "neutral")).strip().lower()
        if sentiment not in ("positive", "neutral", "negative"):
            sentiment = "neutral"
        # 规则兜底：反讽、吐槽这类句子模型容易判成中性（「喜欢个鬼」），
        # 但用户明显在表达不满。命中负面线索词时强制判 negative，
        # 否则下游会用「您好～很高兴为您服务」来回应一个正在生气的用户。
        if any(word in text for word in _NEGATIVE_HINTS):
            sentiment = "negative"
        complaint_intent = bool(data.get("complaint_intent"))

        # 规则前置兜底：监管信号直接判定高风险
        from ..guardrails.compliance import SENSITIVE_WORDS

        for word in SENSITIVE_WORDS:
            if word in text and "监管投诉" not in signals and "舆情" not in signals:
                signals.append("监管投诉" if word not in ("媒体", "曝光") else "舆情")
        if signals:
            risk = "high"
        elif risk == "high" and not any(k in text for k in (*SENSITIVE_WORDS, "投诉", "差评", "索赔")):
            # 风险等级由服务端裁定：模型只给「高」但文本里没有投诉/监管事实 → 收敛为中风险。
            # 否则「质量太差、我要退款」这类普通售后不满会被误判成必须转人工。
            risk = "medium"

        emotion = {
            "sentiment": sentiment,
            "anger_score": _as_confidence(data.get("anger_score"), 0.0),
            "complaint_intent": complaint_intent,
            "risk_level": risk,
            "escalation_signals": signals,
        }
        out = env.child(self.unit_id, confidence=resolve(EXACT, CLASSIFY))
        out.payload = emotion
        if risk == "high":
            out.need_handoff = True
        return out


class ContextBuilder(Unit):
    """P4 上下文组装单元：组装并裁剪统一 SessionContext。

    单元性质：确定性组装，零模型调用。
    裁剪策略按意图相关性：不同意图必装/按需装/不装的字段不同。
    """

    unit_id = "P4"
    domain = "perception"
    description = "上下文组装与裁剪"
    kind = "capability"

    # 各意图的上下文装配策略（对应架构文档 10.3）
    ASSEMBLY: dict[str, dict[str, list[str]]] = {
        "ORDER_STATUS": {"must": ["user", "recent_orders"], "optional": ["active_coupons"], "drop": ["history.summary"]},
        "LOGISTICS": {"must": ["user", "recent_orders"], "optional": ["active_coupons"], "drop": ["history.summary"]},
        "RETURN_APPLY": {"must": ["user", "recent_orders"], "optional": ["history.resolved_topics"], "drop": ["cart_items"]},
        "COUPON_USE": {"must": ["user", "active_coupons"], "optional": ["cart_items"], "drop": ["emotion"]},
        "COMPLAINT": {"must": ["user", "recent_orders", "emotion", "history"], "optional": [], "drop": []},
        "MEMBER_BENEFIT": {"must": ["user"], "optional": ["cart_items"], "drop": ["history.summary"]},
    }

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        intent = env.intent or session.intent.primary
        plan = self.ASSEMBLY.get(intent, {"must": ["user"], "optional": [], "drop": []})

        ctx = session.model_copy(deep=True)
        ctx.intent.primary = intent
        ctx.intent.confidence = float(env.confidence or 0.0)
        # 空字典/空列表是合法结果（P2 判定槽位作废、必填项都齐了），
        # 不能用 `or` 回落 —— `or` 会把「空」当成「没值」，旧槽位又借尸还魂。
        ctx.slots = dict(env.slots if env.slots is not None else session.slots)
        ctx.missing = list(env.missing if env.missing is not None else session.missing)
        ctx.trace_id = env.trace_id
        ctx.session_id = env.session_id

        # 裁剪：丢弃与当前意图无关的字段，控制上下文体积
        for field_name in plan.get("drop", []):
            if field_name == "history.summary":
                ctx.history.summary = ""
            elif field_name == "emotion":
                ctx.emotion.anger_score = 0.0
            elif field_name == "cart_items":
                ctx.cart_items = []

        # 记忆裁剪：只保留最近 3 轮
        ctx.history.recent_turns = ctx.history.recent_turns[-3:]

        out = env.child(self.unit_id, intent=intent, confidence=env.confidence)
        out.payload = {
            "context": ctx.model_dump(),
            "assembly": plan,
            # P4 是 payload 的重建点：P2 判定的「追问无果/用户拒答 → 转人工」
            # 必须显式透传，否则会在上下文组装这一步被静默丢掉，
            # 表现为「用户明明说了不想说，系统还在问同一个问题」。
            "force_handoff": str(env.payload.get("force_handoff") or ""),
        }
        # 信封字段同样要透传：child() 不继承 need_handoff，
        # 只放 payload 会在 P4 这里丢失 → D1 继续走澄清，死循环依旧。
        out.need_handoff = bool(getattr(env, "need_handoff", False))
        return out
