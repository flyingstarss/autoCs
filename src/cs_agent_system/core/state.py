"""会话态存储（架构文档 10.1 L-会话态）。

会话态由编排器独占写入（单一写入者原则），Agent 只返回增量。
默认使用内存实现（便于测试与离线运行），配置 REDIS_URL 后切换到 Redis。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

DEFAULT_TTL_SECONDS = 30 * 60  # 会话态闲置 30 分钟过期


class SessionState(ABC):
    """会话态的抽象。字段对应架构文档 10.2 中属于会话态的部分。"""

    @abstractmethod
    def load(self, session_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def save(self, session_id: str, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...

    # —— 通用工具 ——
    def append_turn(self, session_id: str, role: str, text: str, **meta: Any) -> None:
        state = self.load(session_id)
        turns = state.setdefault("recent_turns", [])
        turns.append({"role": role, "text": text, "ts": time.time(), **meta})
        state["recent_turns"] = turns[-20:]
        self.save(session_id, state)

    def apply_delta(self, session_id: str, delta: dict[str, Any]) -> None:
        """把 Agent 返回的增量合并进会话态（唯一写入口）。"""
        if not delta:
            return
        state = self.load(session_id)
        for key, value in delta.items():
            if key in ("slots",) and isinstance(value, dict):
                # 槽位按「当前快照」整体覆盖，不做合并。
                #
                # 编排器是会话态的唯一写入者，它写入的 slots 就是本轮之后
                # 仍然生效的全部槽位。如果这里做合并，被判定失效的旧槽位
                # （例如用户换了订单、改问订单列表）永远删不掉 ——
                # 下一轮又会被恢复出来，用户的新问题继续被旧单号带偏。
                state["slots"] = dict(value)
            elif key == "resolved_topics" and isinstance(value, list):
                merged_topics = list(state.get("resolved_topics", []))
                for topic in value:
                    if topic not in merged_topics:
                        merged_topics.append(topic)
                state["resolved_topics"] = merged_topics
            elif key == "recent_turns" and isinstance(value, list):
                turns = list(state.get("recent_turns", []))
                turns.extend(value)
                state["recent_turns"] = turns[-20:]
            else:
                state[key] = value
        state["updated_at"] = time.time()
        self.save(session_id, state)


class MemorySessionState(SessionState):
    """内存实现，带 TTL 语义。"""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._data: dict[str, dict[str, Any]] = {}

    def _expired(self, session_id: str) -> bool:
        item = self._data.get(session_id)
        if not item:
            return True
        if time.time() - item.get("_ts", 0) > self._ttl:
            self._data.pop(session_id, None)
            return True
        return False

    def load(self, session_id: str) -> dict[str, Any]:
        if self._expired(session_id):
            return {}
        return dict(self._data.get(session_id, {}))

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        payload = dict(state)
        payload["_ts"] = time.time()
        self._data[session_id] = payload

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def reset(self) -> None:
        self._data.clear()


class RedisSessionState(SessionState):
    """Redis 实现。需要在构造时传入已连接的 redis 客户端。"""

    def __init__(self, client: Any, ttl_seconds: int = DEFAULT_TTL_SECONDS, prefix: str = "cs:session:") -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def load(self, session_id: str) -> dict[str, Any]:
        raw = self._client.get(self._key(session_id))
        if not raw:
            return {}
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        self._client.setex(self._key(session_id), self._ttl, json.dumps(state, ensure_ascii=False, default=str))

    def delete(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))


def build_session_state(redis_url: str | None = None) -> SessionState:
    """按配置选择实现；Redis 不可用时自动回退到内存实现。"""
    if redis_url:
        try:
            import redis  # type: ignore

            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            return RedisSessionState(client)
        except Exception:
            pass
    return MemorySessionState()
