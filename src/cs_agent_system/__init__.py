"""电商 / 私域全自动智能客服 · 多单元系统框架。

分层（对应架构文档第 2 章）：
    L0 接入层     cs_agent_system.api
    L1 编排层     cs_agent_system.orchestrator
    L2 处理单元层 cs_agent_system.agents（21 个单元，仅 4 个调用大模型）
    L3 保障层     cs_agent_system.guardrails
    L4 能力层     cs_agent_system.registry / knowledge
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
