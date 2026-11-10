"""对话轨迹存储与聚合指标（可观测性的数据面）。

解决的问题：此前每轮只有内存里的 `runtime.events`，随响应返回给前端；
一旦进程重启或用户事后想复盘，这条轨迹就消失了，只能靠重跑脚本复现。

这里做三件事：
1. **落库**：每轮对话一行（输入、意图、路由、耗时、token、降级原因、事件流）；
2. **查询**：按 trace_id 取单轮详情，按会话/时间取最近 N 轮；
3. **聚合**：把最近若干轮压成一组指标（人工接管率、澄清率、降级率、
   延迟 P95、平均 token、意图分布），用于判断「系统是不是变差了」。

DB 不可用时退化为内存环形缓冲（最近 200 轮），保证接口不 500。
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from typing import Any

from . import db

_MEM: deque[dict[str, Any]] = deque(maxlen=200)

_COLUMNS = (
    "trace_id", "session_id", "user_id", "turn", "ts", "latency_ms", "text",
    "intent", "confidence", "route", "targets", "need_handoff", "degraded",
    "degraded_reasons", "quality_score", "resolved", "tokens", "response", "events",
    # 本轮产生的工单号：从对话可直达待办，反之亦然
    "ticket_id",
)
_JSON_COLUMNS = ("targets", "degraded_reasons", "events")


def _available() -> bool:
    return db.is_available()


def _row_to_dict(row: Any, with_events: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {}
    keys = row.keys()
    for column in _COLUMNS:
        if column not in keys:
            continue
        value = row[column]
        if column in _JSON_COLUMNS:
            try:
                value = json.loads(value or "null")
            except (TypeError, ValueError):
                value = [] if column != "targets" else []
        record[column] = value
    if not with_events:
        record.pop("events", None)
    return record


def put(record: dict[str, Any]) -> None:
    """写入一轮轨迹。任何异常都不允许影响对话主链路。"""
    payload = {
        "trace_id": str(record.get("trace_id") or ""),
        "session_id": str(record.get("session_id") or ""),
        "user_id": str(record.get("user_id") or ""),
        "turn": int(record.get("turn") or 0),
        "ts": float(record.get("ts") or time.time()),
        "latency_ms": float(record.get("latency_ms") or 0.0),
        "text": str(record.get("text") or ""),
        "intent": str(record.get("intent") or ""),
        "confidence": float(record.get("confidence") or 0.0),
        "route": str(record.get("route") or ""),
        "targets": json.dumps(record.get("targets") or [], ensure_ascii=False),
        "need_handoff": 1 if record.get("need_handoff") else 0,
        "degraded": 1 if record.get("degraded") else 0,
        "degraded_reasons": json.dumps(record.get("degraded_reasons") or [], ensure_ascii=False),
        "quality_score": float(record.get("quality_score") or 0.0),
        "resolved": 1 if record.get("resolved") else 0,
        "tokens": int(record.get("tokens") or 0),
        "response": str(record.get("response") or ""),
        "events": json.dumps(record.get("events") or [], ensure_ascii=False),
        "ticket_id": str(record.get("ticket_id") or ""),
    }
    if not payload["trace_id"]:
        return
    if not _available():
        _MEM.append({**payload, "targets": record.get("targets") or [],
                     "degraded_reasons": record.get("degraded_reasons") or [],
                     "events": record.get("events") or []})
        return
    try:
        placeholders = ",".join("?" for _ in _COLUMNS)
        db.execute(
            f"INSERT OR REPLACE INTO traces ({','.join(_COLUMNS)}) VALUES ({placeholders})",
            [payload[column] for column in _COLUMNS],
        )
    except Exception:  # noqa: BLE001 - 可观测性失败不能拖垮业务
        _MEM.append({**payload, "targets": record.get("targets") or [],
                     "degraded_reasons": record.get("degraded_reasons") or [],
                     "events": record.get("events") or []})


def get(trace_id: str) -> dict[str, Any] | None:
    """按 trace_id 取单轮完整轨迹（含事件流）。"""
    if not trace_id:
        return None
    if _available():
        try:
            row = db.query_one("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
            if row is not None:
                return _row_to_dict(row)
        except Exception:  # noqa: BLE001
            pass
    for item in reversed(_MEM):
        if item.get("trace_id") == trace_id:
            return dict(item)
    return None


def list_recent(limit: int = 20, session_id: str | None = None) -> list[dict[str, Any]]:
    """最近若干轮（默认不含事件流，避免响应过大）。"""
    capped = max(1, min(int(limit), 200))
    if _available():
        try:
            if session_id:
                rows = db.query_all(
                    "SELECT * FROM traces WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
                    (session_id, capped),
                )
            else:
                rows = db.query_all("SELECT * FROM traces ORDER BY ts DESC LIMIT ?", (capped,))
            return [_row_to_dict(row, with_events=False) for row in rows]
        except Exception:  # noqa: BLE001
            pass
    items = [item for item in reversed(_MEM) if not session_id or item.get("session_id") == session_id]
    return [{k: v for k, v in item.items() if k != "events"} for item in items[:capped]]


def summary(limit: int = 200) -> dict[str, Any]:
    """把最近若干轮聚合成可判断健康度的指标。"""
    rows = list_recent(limit)
    turns = len(rows)
    if turns == 0:
        return {
            "turns": 0,
            "handoff_rate": 0.0,
            "clarify_rate": 0.0,
            "degraded_rate": 0.0,
            "empty_reply_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_tokens": 0.0,
            "intent_distribution": {},
            "degraded_reasons": {},
        }

    latencies = sorted(float(row.get("latency_ms") or 0.0) for row in rows)
    p95_index = min(turns - 1, int(round(0.95 * (turns - 1))))
    intents = Counter(str(row.get("intent") or "") for row in rows)
    reasons: Counter[str] = Counter()
    for row in rows:
        for reason in row.get("degraded_reasons") or []:
            reasons[str(reason)] += 1
    return {
        "turns": turns,
        "handoff_rate": round(sum(1 for row in rows if row.get("need_handoff")) / turns, 4),
        "clarify_rate": round(sum(1 for row in rows if row.get("route") == "clarify") / turns, 4),
        "degraded_rate": round(sum(1 for row in rows if row.get("degraded")) / turns, 4),
        "empty_reply_rate": round(sum(1 for row in rows if not str(row.get("response") or "").strip()) / turns, 4),
        "avg_latency_ms": round(sum(latencies) / turns, 1),
        "p95_latency_ms": round(latencies[p95_index], 1),
        "avg_tokens": round(sum(float(row.get("tokens") or 0.0) for row in rows) / turns, 1),
        "intent_distribution": dict(intents.most_common()),
        "degraded_reasons": dict(reasons.most_common()),
    }
