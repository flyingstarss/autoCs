"""pytest 全局夹具：单元测试必须离线、确定性运行。

`.env` 里配置了真实 DeepSeek Key 后，`CustomerService()` 默认会走真实模型，
断言会随模型输出漂移。这里在导入任何业务模块之前强制切到内置规则网关。
"""

import os

os.environ.setdefault("USE_MOCK_LLM", "1")
