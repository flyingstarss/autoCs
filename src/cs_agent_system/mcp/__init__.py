"""MCP 接入层。

- `server`：把业务能力暴露为标准 MCP 工具（数据来自 D 盘 SQLite）；
- `client`：本项目通过 MCP 协议调用这些工具。

子模块**懒加载**：`python -m cs_agent_system.mcp.server` 时若在包 `__init__`
里急切 import server，runpy 会警告「found in sys.modules before execution」，
可能导致 MCP Server 行为异常。懒加载规避这个问题。

用法：
    # 启动 MCP Server（stdio）
    python -m cs_agent_system.mcp.server

    # 启动 MCP Server（streamable-http）
    python -m cs_agent_system.mcp.server --http

    # 在本项目内调用
    from cs_agent_system.mcp import client
    client.call_tool_sync("order_query", {"user_id": "u_1001", "order_no": "SO202610280001"})
"""

from __future__ import annotations

from typing import Any

__all__ = ["client", "server"]


def __getattr__(name: str) -> Any:
    """子模块懒加载（PEP 562）。"""
    if name in ("client", "server"):
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
