"""全局配置。环境变量优先，其次 `.env` 文件，最后默认值。

`.env` 由本模块自带的极简解析器加载（不引入 python-dotenv 依赖）；
已存在的环境变量优先级更高，便于在 CI / 容器里覆盖。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """加载工程根目录（或当前目录）下的 `.env`，仅填充尚未设置的环境变量。"""
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")
        break


_load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _default_data_dir() -> Path:
    """数据目录：优先 `DB_DATA_DIR`，其次工程根目录下的 `data/`。

    显式要求数据落在 D 盘 —— 绝不写到 C 盘用户目录。
    """
    env_dir = os.getenv("DB_DATA_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    # 源码包路径：src/cs_agent_system/config.py → 工程根目录
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / "data").resolve()


@dataclass
class Settings:
    redis_url: str | None = os.getenv("REDIS_URL") or None
    # 置信度阈值是「调参项」：必须运行时可改，不能在类定义时固化。
    # 私有字段存启动时的兜底值；公开属性（见类末尾）每次实时读环境变量。
    _handoff_confidence_threshold: float = field(default=0.6, repr=False)

    write_approval_amount: float = _float("WRITE_APPROVAL_AMOUNT", 500.0)
    max_agent_calls: int = _int("MAX_AGENT_CALLS", 12)
    max_tokens: int = _int("MAX_TOKENS", 8000)
    deadline_ms: int = _int("DEADLINE_MS", 15000)
    use_mock_llm: bool = os.getenv("USE_MOCK_LLM", "").strip().lower() in ("1", "true", "yes")
    # —— 数据源（SQLite）——
    data_dir: Path = _default_data_dir()
    db_path: Path = Path(os.getenv("DB_PATH", "").strip() or (data_dir / "cs_agent.db")).resolve()
    # 数据源模式：db=真实 SQLite / mock=内存字典（降级兜底）
    data_source: str = os.getenv("DATA_SOURCE", "db").strip().lower() or "db"
    db_enabled: bool = _bool("DB_ENABLED", True)
    db_echo: bool = _bool("DB_ECHO", False)

    # —— MCP 服务 ——
    mcp_enabled: bool = _bool("MCP_ENABLED", True)
    mcp_server_enabled: bool = _bool("MCP_SERVER_ENABLED", True)
    mcp_server_host: str = os.getenv("MCP_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    mcp_server_port: int = _int("MCP_SERVER_PORT", 8765)
    # 业务工具走 MCP Server 时为 true；否则走进程内直连
    tools_via_mcp: bool = _bool("TOOLS_VIA_MCP", False)
    mcp_client_timeout: float = _float("MCP_CLIENT_TIMEOUT", 10.0)

    # —— OMS（写操作后端）——
    # local = 写本地 SQLite；http = 推给真实电商平台/ERP
    oms_backend: str = os.getenv("OMS_BACKEND", "local").strip().lower() or "local"
    oms_base_url: str = os.getenv("OMS_BASE_URL", "").strip()
    oms_token: str = os.getenv("OMS_TOKEN", "").strip()
    oms_timeout: float = _float("OMS_TIMEOUT", 5.0)

    @property
    def handoff_confidence_threshold(self) -> float:
        """转人工置信度阈值：实时读 `HANDOFF_CONFIDENCE_THRESHOLD`。

        为什么不用普通 dataclass 字段：字段默认值在类定义（= 模块导入）时刻
        求值一次，之后改环境变量不生效——与「改 .env 即可调参」的承诺不符。
        """
        return _float("HANDOFF_CONFIDENCE_THRESHOLD", self._handoff_confidence_threshold)


settings = Settings()
