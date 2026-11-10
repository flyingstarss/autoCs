"""决策域：D1 意图路由 / D2 任务规划 / D3 响应生成。"""

from __future__ import annotations

import re
from typing import Any

from ..config import settings
from ..core.confidence import (
    COMPUTED,
    ERROR,
    EXACT,
    GENERATED,
    PARTIAL,
    REJECTED,
    resolve,
)
from ..core.envelope import AgentEnvelope, SessionContext
from ..core.intents import COMPOSITE_HINTS, INTENTS, ROUTE_TABLE, get_intent
from .base import Unit
from .perception import PRICE_OBJECTION_HINTS

_CJK_GAP = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")

# 百分比事实：成分含量（100% 精梳棉）、折扣力度（7 折不算这里）等。
# 模型最易在这里「用模糊词替代精确值」—— 实测把「100% 精梳棉」写成
# 「大概率精梳棉」，数字消失了，Q2 的数字比对也就无从发现。
_PERCENT_TOKEN = re.compile(r"\d+(?:\.\d+)?\s*%")


def _missing_percentages(text: str, skeleton: str) -> list[str]:
    """骨架里出现、但模型输出里消失的百分比。"""
    skeleton_tokens = {token.replace(" ", "") for token in _PERCENT_TOKEN.findall(skeleton)}
    text_tokens = {token.replace(" ", "") for token in _PERCENT_TOKEN.findall(text)}
    return sorted(skeleton_tokens - text_tokens)


def _tidy(text: str) -> str:
    """收拾模型偶发的中文碎空格（如「感谢您的耐心 等待」）。"""
    return _CJK_GAP.sub("", text).strip()


class Router(Unit):
    """D1 意图路由单元：规则优先、配置驱动。

    单元性质：**决策单元**——不调模型，但在选路径（决定后续调用谁）。
    三条短路路径：
    1. 单工具可满足 → direct_tool，不进模型链；
    2. 复合多意图 → plan_execute；
    3. 风险高或意图不明 → handoff / clarify。
    """

    unit_id = "D1"
    domain = "decision"
    description = "意图路由"
    kind = "router"

    # 意图置信度下限：低于此值不猜测，改走澄清。
    # 读取全局配置，使 HANDOFF_CONFIDENCE_THRESHOLD 真正生效（改 .env 即可调参）。
    #
    # 注意：不能写成类属性 `CONFIDENCE_FLOOR = settings.xxx`——那会在模块导入
    # 时刻就把值固化下来，之后改 .env 不重启不生效，与「配置驱动」的承诺不符。
    # 这里改成属性，每次读取都拿到当前的 settings 值。
    @property
    def confidence_floor(self) -> float:
        return float(settings.handoff_confidence_threshold)

    @staticmethod
    def _effective_multi(intent: str, raw: list[str]) -> list[str]:
        """过滤「同能力」的伪多意图。

        真实模型常把「尺码咨询 + 库存查询」都标成多意图，但两者共用 E4 执行器；
        这类请求按单意图走 direct_tool 更省钱，也不会重复调用同一个工具。
        """
        primary = get_intent(intent)
        used = {primary.executor} if primary and primary.executor else set()
        result: list[str] = []
        for name in raw:
            if name not in INTENTS or name == intent:
                continue
            spec = get_intent(name)
            if not spec or not spec.executor or spec.executor in used:
                continue
            used.add(spec.executor)
            result.append(name)
        return result

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        intent = env.intent or session.intent.primary
        confidence = float(env.confidence or 0.0)
        multi = self._effective_multi(intent, list(session.intent.multi or []))
        risk = session.emotion.risk_level

        # 1) 高风险 → 直接接管（不进入自动应答链路）
        if risk == "high":
            out = env.child(self.unit_id, intent=intent, confidence=confidence)
            out.payload = {"route": "handoff", "targets": ["Q5"], "parallel": False, "reason": "风险等级=高"}
            out.need_handoff = True
            return out

        # 2) 意图不明 → 澄清
        if confidence < self.confidence_floor:
            out = env.child(self.unit_id, intent=intent, confidence=confidence)
            out.payload = {
                "route": "clarify",
                "targets": [],
                "parallel": False,
                "reason": f"意图置信度 {confidence:.2f} 低于阈值 {self.confidence_floor}",
                "clarify_options": self._clarify_options(intent),
            }
            return out

        # 3) 复合意图 → 规划执行
        if multi:
            out = env.child(self.unit_id, intent=intent, confidence=confidence)
            out.payload = {
                "route": "plan_execute",
                "targets": list(multi),
                "parallel": False,
                "reason": f"识别到多意图 {multi}",
            }
            return out

        table = ROUTE_TABLE.get(intent)
        if table is None:
            out = env.child(self.unit_id, intent=intent, confidence=confidence)
            out.payload = {"route": "handoff", "targets": ["Q5"], "parallel": False, "reason": "意图未登记"}
            out.need_handoff = True
            return out

        # 4) 拒答 / 同一槽位已追问一次 → 转人工补齐。
        # 继续澄清只会让用户重复「不想说」，把死循环变成一次明确的交接。
        forced = str(env.payload.get("force_handoff") or "")
        if bool(getattr(env, "need_handoff", False)) or forced:
            out = env.child(self.unit_id, intent=intent, confidence=confidence)
            out.payload = {"route": "handoff", "targets": ["Q5"], "parallel": False, "reason": forced}
            out.need_handoff = True
            return out

        # 5) 缺少关键槽位 → 澄清（一次只问一个关键槽位）
        missing = session.missing
        if missing:
            out = env.child(self.unit_id, intent=intent, confidence=confidence)
            out.payload = {
                "route": "clarify",
                "targets": [],
                "parallel": False,
                "reason": f"缺少关键槽位 {missing[0]}",
                "missing": missing,
                "clarify_options": [],
            }
            return out

        out = env.child(self.unit_id, intent=intent, confidence=confidence)
        out.payload = {**table, "intent": intent}
        return out

    @staticmethod
    def _clarify_options(intent: str) -> list[str]:
        """用二选一提问替代长列表。"""
        options = {
            "ORDER_STATUS": ["查订单状态", "查物流到哪了"],
            "LOGISTICS": ["查物流轨迹", "查订单状态"],
            "RETURN_APPLY": ["申请退货", "申请换货"],
            "COUPON_USE": ["用已有券下单", "领一张新券"],
            "PRODUCT_QA": ["问商品参数", "问尺码建议"],
        }
        return options.get(intent, ["继续描述你的问题", "转人工客服"])


class Planner(Unit):
    """D2 任务规划单元：把复合请求拆成有依赖的子任务 DAG。

    单元性质：**决策单元**——不调模型，但在选路径（决定执行顺序与依赖）。
    输出 TaskDAG：每个节点声明目标单元与依赖。
    可执行性校验：依赖必须存在于任务图中，且不得成环。
    """

    unit_id = "D2"
    domain = "decision"
    description = "任务规划"
    kind = "planner"

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        intent = env.intent or session.intent.primary
        multi = list(session.intent.multi or [])
        spec = get_intent(intent)

        steps: list[dict[str, Any]] = []

        def _next_id() -> str:
            return f"s{len(steps) + 1}"

        # 主意图：需要证据的先取证据，再执行
        if spec and spec.needs_knowledge:
            q_id = _next_id()
            steps.append({"id": q_id, "agent": "Q1", "intent": intent, "depends_on": [], "reason": "取政策证据"})
            steps.append(
                {
                    "id": _next_id(),
                    "agent": spec.executor or "E5",
                    "intent": intent,
                    "depends_on": [q_id],
                    "reason": "执行业务",
                }
            )
        elif spec and spec.executor:
            steps.append({"id": _next_id(), "agent": spec.executor, "intent": intent, "depends_on": [], "reason": "单工具短路"})

        # 附加意图：与主任务无依赖可并行；但需要证据的附加意图同样必须先挂 Q1，
        # 否则次意图（如复合请求里的退款）会跳过政策证据直接执行
        extra = multi or [h for h in COMPOSITE_HINTS.get(intent, ()) if h != intent]
        used_agents = {s["agent"] for s in steps if s["agent"] != "Q1"}
        for extra_intent in extra:
            extra_spec = get_intent(extra_intent)
            if not extra_spec or not extra_spec.executor:
                continue
            if extra_spec.executor in used_agents:
                continue  # 同一执行器已在计划里，不重复挂载（如 E4 同时覆盖尺码与库存）
            used_agents.add(extra_spec.executor)
            if extra_spec.needs_knowledge:
                q_id = _next_id()
                steps.append(
                    {"id": q_id, "agent": "Q1", "intent": extra_intent, "depends_on": [], "reason": "追加意图取政策证据"}
                )
                steps.append(
                    {
                        "id": _next_id(),
                        "agent": extra_spec.executor,
                        "intent": extra_intent,
                        "depends_on": [q_id],
                        "reason": "并行子任务",
                    }
                )
            else:
                steps.append(
                    {
                        "id": _next_id(),
                        "agent": extra_spec.executor,
                        "intent": extra_intent,
                        "depends_on": [],
                        "reason": "并行子任务",
                    }
                )

        ok, problem = self._validate(steps)
        out = env.child(
            self.unit_id,
            intent=intent,
            confidence=resolve(EXACT if ok else REJECTED, COMPUTED),
        )
        out.payload = {
            "steps": steps,
            "valid": ok,
            "problem": problem,
            "parallel_groups": self._waves(steps),
        }
        return out

    @staticmethod
    def _validate(steps: list[dict[str, Any]]) -> tuple[bool, str]:
        ids = {s["id"] for s in steps}
        for step in steps:
            for dep in step.get("depends_on", []):
                if dep not in ids:
                    return False, f"依赖 {dep} 不存在"
        # 简易成环检测
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return False
            if node in done:
                return True
            visiting.add(node)
            for step in steps:
                if step["id"] == node:
                    for dep in step.get("depends_on", []):
                        if not visit(dep):
                            return False
            visiting.discard(node)
            done.add(node)
            return True

        return (True, "") if all(visit(s["id"]) for s in steps) else (False, "任务图存在环")

    @staticmethod
    def _waves(steps: list[dict[str, Any]]) -> list[list[str]]:
        """按依赖分层：同一层内可并行执行。"""
        remaining = list(steps)
        waves: list[list[str]] = []
        done: set[str] = set()
        while remaining:
            layer = [s["id"] for s in remaining if all(d in done for d in s.get("depends_on", []))]
            if not layer:
                break
            waves.append(layer)
            done.update(layer)
            remaining = [s for s in remaining if s["id"] not in layer]
        return waves


class ResponseWriter(Unit):
    """D3 响应生成单元：把结构化结果渲染成渠道适配的话术。

    单元性质：**调用大模型**（全系统唯一一个生成类单元）。
    关键设计：事实占位符注入。
    模板里的数字（如 {refund_sla_days}）由服务端事实直接填充，
    模型只负责语气组织，不允许回忆或推算业务数字。
    """

    unit_id = "D3"
    domain = "decision"
    description = "响应生成"
    kind = "agent"
    uses_llm = True

    # 各意图的模板骨架 —— 所有业务数字均为占位符
    TEMPLATES: dict[str, str] = {
        "ORDER_STATUS": "亲，您的订单 {order_no} 当前状态是「{status}」，实付 {amount} 元。",
        "LOGISTICS": "您的包裹由 {carrier} 承运，最新进度是「{last_node}」（{last_time}）：{last_desc}。",
        "RETURN_APPLY": "已为您提交{type}申请（售后单号 {aftersale_no}），退款金额 {refund_amount} 元，按原渠道退回，到账时效约 {refund_sla_days} 个工作日。",
        "SIZE_ADVICE": "{title}（{size}）{stock_desc}，价格 {price} 元。",
        # 库存问答通常不带尺码（「库存是多少」），此时报的是整款各尺码的库存，
        # 再套「（标准 码）」既多余又别扭，所以模板里不写尺码。
        "STOCK_CHECK": "{title}：{stock_desc}，价格 {price} 元。",
        "PRODUCT_QA": "{title} 的{params_desc}，售价 {price} 元。",
        "COUPON_USE": "已为您发放优惠券 {coupon_id}（{amount} 元，满 {threshold} 元可用），{stack_note}。",
        "PAYMENT_INVOICE": "订单 {order_no} 的支付状态：{paid_status}，可开具{invoice_types}。",
        "MEMBER_BENEFIT": "您当前是{level}会员，积分 {points}，本单可享会员权益{member_note}。",
        "COMPLAINT": "非常抱歉给您带来不好的体验，我已为您转接专属客服跟进。",
        "CHITCHAT": "您好～很高兴为您服务，请问有什么可以帮您？",
    }

    # 用户情绪为负（吐槽/反讽/价格不满）时的闲聊回复。
    # 不能再用「您好～很高兴为您服务」—— 那等于无视用户情绪，
    # 实测「喜欢个鬼！」会得到一句热情问候，非常违和。
    CHITCHAT_COMFORT_TEMPLATE = "抱歉让您有不好的体验。您是对价格、商品还是服务不满意？告诉我具体问题，我马上帮您处理。"

    # 用户嫌贵时的回复：先共情价格感受，再给参数价值与可选出路。
    # 只报面料参数会显得答非所问 —— 用户抱怨的是价格，不是材质。
    PRICE_OBJECTION_TEMPLATE = (
        "亲，这款 {title} 售价 {price} 元，{params_desc}。"
        "理解您觉得价格偏高，您可以看看店内优惠券，或告诉我预算，我帮您找同价位的替代款。"
    )

    # 用户问「你们有哪些商品」这类导购问题：E5 定位不到单品时回落到商品目录
    CATALOG_TEMPLATE = "亲，目前在售的有：{catalog}。您想了解哪一款？我可以帮您查参数、尺码和库存～"

    # 用户点名要某个商品、但库里没有（「我想买帽子」）：
    # 先明确说「暂时没有帽子」，再列在售商品。只甩目录等于没听见问题。
    CATALOG_NOT_FOUND_TEMPLATE = (
        "亲，我们暂时没有「{not_found_item}」在售。目前在售的有：{catalog}。"
        "您想了解哪一款？我可以帮您查参数、尺码和库存～"
    )

    # 用户问「我还有哪些没发货的订单」这类列表问题：一次说清命中几单、分别是哪些。
    # 不能只报一单 —— 那是答非所问。
    ORDER_LIST_TEMPLATE = "亲，符合条件的订单共 {order_count} 单：{orders_summary}。需要我帮您处理哪一单吗？"

    # 用户笼统问「我有哪些订单」：同样要列清单，但措辞不能叫「符合条件」——
    # 本轮没有任何筛选，说「符合条件」会让用户以为系统按什么条件筛过。
    MY_ORDERS_TEMPLATE = "亲，您名下共有 {order_count} 单：{orders_summary}。需要我帮您处理哪一单吗？"

    # 用户指定的订单号查不到（不存在或不属于本人）。
    # 必须明确说查不到，绝不能拿「最近一单」顶替 —— 那等于回答了另一个问题。
    ORDER_NOT_FOUND_TEMPLATE = "亲，未查询到属于您的订单 {not_found_order_no}，请核对订单号是否正确。"

    # 用户要某类状态的订单但一单都没匹配上：如实说没有，不拿别的单凑数。
    ORDER_EMPTY_LIST_TEMPLATE = "亲，您目前没有符合条件的订单。"

    _PROMPT = """你是资深电商客服。请把「事实骨架」组织成一条自然、有温度的中文回复。

硬性约束（违反即不可接受）：
1. 事实骨架里的所有数字、单号、时效、价格、库存、积分必须原样保留，不得改写、四舍五入或增删；
2. 不得新增骨架里没有的承诺、时效、补偿、优惠或政策；
3. 骨架里没有数字时，回复里也不得出现任何数字（包括「24小时」「7天」这类套话）；
4. 不得提及「模板」「系统」「模型」「内部」等字眼，不得编造用户没问到的信息；
5. 控制在 120 字以内，口语化，可加一句礼貌收尾。

只输出给用户看到的话术正文，不要 JSON、不要解释、不要加引号包裹。"""

    def _template_for(
        self,
        intent: str,
        facts: dict[str, Any],
        negative_sentiment: bool = False,
        price_objection: bool = False,
    ) -> str:
        """按意图取模板；商品目录类问题没有单品事实，单独用语料模板。"""
        if intent == "CHITCHAT":
            # 情绪为负时不能热情问候，要共情并追问具体问题
            return self.CHITCHAT_COMFORT_TEMPLATE if negative_sentiment else self.TEMPLATES["CHITCHAT"]
        if facts.get("catalog") and not facts.get("title"):
            # 用户点名要的商品不存在 → 先明确回「没有这一款」
            if facts.get("not_found_item"):
                return self.CATALOG_NOT_FOUND_TEMPLATE
            return self.CATALOG_TEMPLATE
        # 嫌贵时不能只报材质参数：用户抱怨的是价格
        if price_objection and facts.get("title") and facts.get("price") is not None:
            return self.PRICE_OBJECTION_TEMPLATE
        if intent == "ORDER_STATUS":
            # 筛选后一单都没有：如实说没有
            if facts.get("empty_order_list"):
                return self.ORDER_EMPTY_LIST_TEMPLATE
            # 指定单号但查不到：明确告知，不能拿别的单顶替
            if facts.get("not_found_order_no"):
                return self.ORDER_NOT_FOUND_TEMPLATE
            # 列表型诉求（用户要「还有哪些没发货的」）：一次说清命中几单
            if facts.get("order_count") and int(facts["order_count"]) > 1:
                # 没有任何筛选条件时用「您名下共有」，否则用「符合条件的」
                if facts.get("list_all_orders"):
                    return self.MY_ORDERS_TEMPLATE
                return self.ORDER_LIST_TEMPLATE
        return self.TEMPLATES.get(intent, "{content}")

    def handle(self, env: AgentEnvelope, session: SessionContext) -> AgentEnvelope:
        intent = env.intent or session.intent.primary
        facts: dict[str, Any] = dict(env.payload.get("facts") or {})
        evidence = env.evidence

        # 复合请求：主意图与 multi 意图各自渲染一段，拼成一条完整回复，
        # 否则「退款 + 查物流」只会答出主意图那一半。
        targets = [intent, *[i for i in (session.intent.multi or []) if i in self.TEMPLATES and i != intent]]
        # 本轮情绪是否为负：闲聊场景要据此改用安抚话术，而不是热情问候
        negative = str(session.emotion.sentiment or "") == "negative"
        # 用户是否在嫌贵（「这个也太贵了吧」）：改善回复要针对价格，而不是只念参数
        price_objection = any(hint in str(env.payload.get("text") or "") for hint in PRICE_OBJECTION_HINTS)
        template = ""
        segments: list[str] = []
        dropped: list[str] = []
        for name in targets:
            tpl = self._template_for(
                name, facts, negative_sentiment=negative, price_objection=price_objection
            )
            filled = self._fill(tpl, facts)
            # 某一段事实缺失就不做该段承诺，只输出能站得住的部分
            if filled.get("unresolved"):
                dropped.append(name)
                continue
            template = template or tpl
            segments.append(str(filled.get("text", "")).strip())
        content = " ".join(s for s in segments if s).strip() or self._template_for("CHITCHAT", {}, negative_sentiment=negative)

        # 让模型只做语气组织：事实骨架由服务端填好，模型不得改动任何数字
        llm_result = None
        try:
            # 流式生成：有 SSE 订阅者时逐块推送；没有订阅者时自动退化为普通调用。
            # 记账/预算仍在 runtime 里统一收口，D3 不自己碰账本。
            llm_result = self.rt.stream_llm(
                unit_id=self.unit_id,
                task="generate",
                prompt=self._PROMPT,
                tier="large",
                context={
                    "intent": intent,
                    "channel": session.channel.type,
                    "facts_skeleton": content,
                },
            )
            text = str(llm_result.data.get("text") or "").strip() or content
        except Exception:  # noqa: BLE001 —— 模型不可用时仍把已填充的事实话术发出去
            # 注意：BudgetExceeded 也会走到这里。token 预算耗尽时降级为
            # 直接输出事实骨架 —— 有事实总比没有回复强。
            text = content
        text = _tidy(text) or content
        # 数字保留兜底：模型把「100% 精梳棉」改写成「大概率精梳棉」这类
        # 丢掉精确值的句子，一律退回事实骨架 —— 骨架是服务端填的，数字一定对。
        restored = False
        if _missing_percentages(text, content):
            text = content
            restored = True

        # 渠道适配：长度截断
        max_len = session.channel.max_len
        if len(text) > max_len:
            text = text[: max_len - 1] + "…"

        # 置信度分级：内容被静默丢弃、模型不可用、事实不全，都会削弱可信度。
        # 此前无论丢了多少段、模型是否降级，都固定报 0.85，等于谎报。
        if dropped:
            outcome = PARTIAL
        elif llm_result is None:
            # 模型不可用，输出的是裸模板骨架：话术成立，但未经模型润色
            outcome = PARTIAL
        elif restored:
            # 模型润色时丢了精确值，已回退骨架：事实可信，但非模型原稿
            outcome = PARTIAL
        elif not segments:
            outcome = REJECTED
        else:
            outcome = EXACT
        out = env.child(self.unit_id, intent=intent, confidence=resolve(outcome, GENERATED))
        out.payload = {
            "text": text,
            "template": template,
            "facts_used": sorted(facts.keys()),
            "evidence_count": len(evidence),
            "dropped_segments": dropped,
            "facts_restored": restored,
        }
        out.evidence = evidence
        return out

    @staticmethod
    def _fill(template: str, facts: dict[str, Any]) -> dict[str, Any]:
        """用服务端事实填充占位符；缺失的事实不臆造，标记为未解析。"""
        text = template
        used: list[str] = []
        unresolved: list[str] = []
        for key, value in facts.items():
            placeholder = "{" + key + "}"
            if placeholder not in text:
                continue
            if value is None or value == "":
                unresolved.append(key)
                continue
            text = text.replace(placeholder, str(value))
            used.append(key)
        # 模板里仍有未提供的占位符 → 视为事实不足
        import re as _re

        leftover = _re.findall(r"\{(\w+)\}", text)
        if leftover:
            unresolved.extend(k for k in leftover if k not in unresolved)
        return {"text": text, "used": used, "unresolved": unresolved}
