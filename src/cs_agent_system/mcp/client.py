"""MCP 客户端：让本项目的工具层能通过 MCP 协议调用业务能力。

两种接入形态：
1. **stdio**（默认）：以子进程方式拉起 `python -m cs_agent_system.mcp.server`，
   进程内通信，无端口占用，最适合「工具在本仓库、能力可独立部署」的场景；
2. **streamable-http**：连接已在运行的 MCP Server（`--http` 启动），
   适合多进程/多机共享同一份业务能力。

关键设计：
- **进程级常驻事件循环**：MCP 会话（`ClientSession` + stdio 子进程）绑定在
  创建它的事件循环上。若每次调用都用 `asyncio.run()` 新建循环，第一次调用
  结束时循环被关闭 → 第二次调用必然 `ClosedResourceError`。
  这里用一个常驻线程 + 一个事件循环承载所有 MCP 调用。
- **会话由单一 task 持有**：anyio 的 cancel scope 要求 enter/exit 落在同一个
  task 内完成。会话的开与关都发生在专属 worker task 上，避免
  "Attempted to exit cancel scope in a different task"。
- **失败即降级**：MCP 不可用（未安装 SDK / 子进程起不来 / 超时）时，
  `McpToolBridge` 抛出 `ToolTimeout` 之类的错误由上层兜底，
  绝不静默返回假数据——假数据比报错更危险。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any

from ..config import settings


# —— 进程级常驻事件循环 ——
class _LoopHost:
    """承载所有 MCP 调用的常驻事件循环。

    MCP 会话绑定在创建它的事件循环上：`asyncio.run()` 每次新建并销毁循环，
    会话在第一次调用后即失效。因此本模块自建一个常驻循环线程，
    在进程生命周期内复用。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever, name="mcp-loop", daemon=True
                )
                thread.start()
                self._loop = loop
                self._thread = thread
            return self._loop

    def submit(self, coro: Any) -> concurrent.futures.Future:
        """把协程排到常驻循环，返回可跨线程等待的 Future。"""
        return asyncio.run_coroutine_threadsafe(coro, self.loop())

    def run(self, coro: Any, timeout: float | None = None) -> Any:
        """在常驻循环上跑完协程并同步返回结果。"""
        fut = self.submit(coro)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise


_host = _LoopHost()


def _server_command() -> list[str]:
    """stdio 模式的子进程命令。用当前解释器，避免环境不一致。"""
    return [sys.executable, "-m", "cs_agent_system.mcp.server"]


def _server_env() -> dict[str, str]:
    """子进程环境变量：确保能 import 到本项目源码。"""
    env = dict(os.environ)
    src = str(settings.data_dir.parent / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    # MCP Server 的 INFO 日志必须走 stderr —— stdout 是 stdio 协议的传输通道，
    # 往 stdout 打印任何非协议内容都会污染 JSON-RPC 帧流。
    env["FASTMCP_LOG_LEVEL"] = env.get("FASTMCP_LOG_LEVEL", "ERROR")
    env["MCP_LOG_LEVEL"] = env.get("MCP_LOG_LEVEL", "ERROR")
    env["PYTHONUNBUFFERED"] = "1"
    return env


@asynccontextmanager
async def _open_stdio() -> Any:
    """建立 stdio MCP 会话。"""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=_server_command()[0],
        args=_server_command()[1:],
        env=_server_env(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def _open_http() -> Any:
    """建立 streamable-http MCP 会话。"""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        f"http://{settings.mcp_server_host}:{settings.mcp_server_port}/mcp"
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def _describe(session: Any) -> list[dict[str, Any]]:
    """列出 MCP Server 提供的工具清单。"""
    result = await session.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "schema": t.inputSchema or {},
        }
        for t in result.tools
    ]


async def _invoke(
    session: Any,
    name: str,
    arguments: dict[str, Any],
    timeout: float | None,
) -> dict[str, Any]:
    """在会话上调用一个 MCP 工具，返回解析后的 JSON。

    MCP 工具返回的是文本内容（JSON 字符串），这里统一反序列化成 dict，
    让上层与进程内调用拿到完全一致的结构。
    """
    wait = timeout if timeout is not None else settings.mcp_client_timeout
    result = await asyncio.wait_for(
        session.call_tool(name, arguments or {}),
        timeout=wait,
    )
    if getattr(result, "isError", False):
        raise RuntimeError(f"MCP 工具 {name} 返回错误：{getattr(result, 'content', '')}")
    for block in result.content or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return {"value": text}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


class _SessionWorker:
    """独占一个 MCP 会话的长驻 task。

    会话的开与关必须发生在同一个 task（anyio cancel scope 的硬要求），
    因此所有对会话的操作都排队交给这个 task 执行，调用方只负责投递命令。
    """

    def __init__(self, transport: str) -> None:
        self.transport = transport
        self._cmd: asyncio.Queue | None = None
        self._ready: asyncio.Future | None = None
        self._task: asyncio.Task | None = None

    # —— 生命周期（只能在常驻循环内调用）——
    def start(self) -> None:
        if self._task is not None:
            return
        self._cmd = asyncio.Queue()
        self._ready = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        opener = _open_stdio if self.transport == "stdio" else _open_http
        assert self._cmd is not None and self._ready is not None
        try:
            async with opener() as session:
                if not self._ready.done():
                    self._ready.set_result(session)
                while True:
                    kind, payload, fut = await self._cmd.get()
                    if kind == "stop":
                        if not fut.done():
                            fut.set_result(None)
                        break
                    try:
                        if kind == "session":
                            value: Any = session
                        elif kind == "list":
                            value = await _describe(session)
                        else:  # "call"
                            name, args, timeout = payload
                            value = await _invoke(session, name, args, timeout)
                        if not fut.done():
                            fut.set_result(value)
                    except Exception as exc:  # noqa: BLE001 —— 错误回传给调用方
                        if not fut.done():
                            fut.set_exception(exc)
        except Exception as exc:  # noqa: BLE001 —— 开会话失败要让等待方知道
            if not self._ready.done():
                self._ready.set_exception(exc)

    # —— 命令接口 ——
    async def dispatch(self, kind: str, payload: Any = None) -> Any:
        self.start()
        assert self._cmd is not None and self._ready is not None
        # 先等会话就绪：开会话失败要立刻抛给调用方
        session = await self._ready
        fut: concurrent.futures.Future = concurrent.futures.Future()
        await self._cmd.put((kind, payload, fut))
        return await asyncio.wrap_future(fut)

    async def shutdown(self) -> None:
        if self._task is None:
            return
        if self._cmd is not None:
            fut: concurrent.futures.Future = concurrent.futures.Future()
            await self._cmd.put(("stop", None, fut))
            try:
                await asyncio.wrap_future(fut)
            except Exception:  # noqa: BLE001 —— 关闭失败不影响主流程
                pass
        try:
            await self._task
        except Exception:  # noqa: BLE001 —— 关闭失败不影响主流程
            pass


_workers: dict[str, _SessionWorker] = {}
_workers_lock = threading.Lock()


def _worker(transport: str) -> _SessionWorker:
    with _workers_lock:
        w = _workers.get(transport)
        if w is None:
            w = _SessionWorker(transport)
            _workers[transport] = w
        return w


# —— 对外异步接口（内部统一调度到常驻循环）——
async def _list_tools(transport: str) -> list[dict[str, Any]]:
    return await _worker(transport).dispatch("list")


async def _get_session(transport: str) -> Any:
    return await _worker(transport).dispatch("session")


async def _call_tool(
    name: str,
    arguments: dict[str, Any] | None,
    transport: str,
    timeout: float | None,
) -> dict[str, Any]:
    return await _worker(transport).dispatch("call", (name, arguments or {}, timeout))


async def get_session(transport: str = "stdio") -> Any:
    """取（并复用）MCP 会话。

    返回的会话绑定在**本模块的常驻事件循环**上，
    只能在 `list_tools` / `call_tool` 这类封装里使用，不要直接跨循环 await。
    """
    return await asyncio.wrap_future(_host.submit(_get_session(transport)))


async def close_session(transport: str = "stdio") -> None:
    """关闭并清除会话。"""
    with _workers_lock:
        worker = _workers.pop(transport, None)
    if worker is None:
        return
    try:
        await asyncio.wrap_future(_host.submit(worker.shutdown()))
    except Exception:  # noqa: BLE001 —— 关闭失败不影响主流程
        pass


async def list_tools(transport: str = "stdio") -> list[dict[str, Any]]:
    """列出 MCP Server 提供的工具清单。"""
    return await asyncio.wrap_future(_host.submit(_list_tools(transport)))


async def call_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    transport: str = "stdio",
    timeout: float | None = None,
) -> dict[str, Any]:
    """调用 MCP 工具，返回解析后的 JSON。"""
    return await asyncio.wrap_future(
        _host.submit(_call_tool(name, arguments, transport, timeout))
    )


# —— 对外同步接口 ——
def list_tools_sync(transport: str = "stdio") -> list[dict[str, Any]]:
    """同步列出 MCP 工具（供 ToolRegistry 启动时注册）。"""
    return _host.run(_list_tools(transport), timeout=settings.mcp_client_timeout + 5)


def call_tool_sync(
    name: str,
    arguments: dict[str, Any] | None = None,
    transport: str = "stdio",
) -> dict[str, Any]:
    """同步调用 MCP 工具（在同步代码里用）。

    编排器执行工具时处于线程池，因此这里不能复用调用方的 loop，
    统一排到本模块的常驻循环上——会话与循环的生命周期始终一致。
    """
    return _host.run(
        _call_tool(name, arguments, transport, None),
        timeout=settings.mcp_client_timeout + 5,
    )


def shutdown_sync() -> None:
    """进程退出前关闭会话（避免 MCP 子进程残留）。"""
    with _workers_lock:
        workers = list(_workers.values())
        _workers.clear()
    for worker in workers:
        try:
            _host.run(worker.shutdown(), timeout=5)
        except Exception:  # noqa: BLE001 —— 关闭失败不影响退出
            pass
