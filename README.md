# 电商 / 私域全自动智能客服 · 多智能体系统

基于 LangGraph 构建的电商客服多智能体系统，覆盖售前咨询、订单物流、售后处理、优惠券、投诉转接全流程，带写操作安全闸门与人工审核台。

> **设计原则：回答型任务交给模型生成，事务型任务交给确定性工具执行。**
> 模型只能「申请」写操作，不能「决定」写操作——所有安全约束以代码形式落在保障层，而不是写在 Prompt 里靠模型自觉。

---

## 一、功能概览

**咨询与知识问答**

- 意图识别与路由，覆盖商品咨询、订单查询、物流跟踪、售后、优惠券、发票、投诉七类场景
- 知识检索采用 BM25 + 字符 n-gram 词法召回，千级条目无需向量库
- 多轮对话槽位补全，缺失参数主动追问

**订单与物流**

- 按订单号或用户查订单，返回状态、金额、商品明细
- 物流轨迹查询，异常件识别
- 库存与价格实时校验，活动价与会员价区分

**售后与写操作**

- 退款申请、退款撤销、优惠券发放、订单操作四类写能力
- 退款金额校验：赔付上限封顶、订单状态前置校验
- 退款改额采用「先撤后建」，避免金额篡改
- 四道安全闸：白名单、幂等、超时回查、超阈值审批

**人工接管**

- 四条触发入口：低置信度、高风险情绪、合规拦截、写操作超阈值
- 工单落库，坐席在审核台批准或驳回
- 批准时带原幂等键重放写操作，保证只执行一次
- 重放超时标记「状态未知」，不伪装成功

**可观测性**

- 每条回复带 trace_id，可回放每个处理节点的输入输出
- 聚合指标：响应耗时、模型调用次数、token 消耗、降级次数、转人工率
- 运行控制台支持会话窗与当轮回放

**协议接入**

- 内置 MCP Server，暴露 10 个只读工具供外部客户端调用
- 13 个进程内工具，写操作不对外暴露

---

## 二、架构

```
用户消息
   │
   ▼
Orchestrator（LangGraph 状态图）
  21 个处理单元，其中仅 4 个调用模型
   │
   ├─ 感知层   意图识别 / 槽位抽取 / 情绪判定      ← 调模型
   ├─ 决策层   规则路由 + 置信度闸门 + 政策校验
   ├─ 执行层   工具调用（白名单 + 幂等 + 超时回查）
   └─ 保障层   事实校验 / 合规拦截 / 兜底转人工
   │
   ▼
回复（含 trace_id，全链路可回放）
```

**只有 4 个节点调模型**：意图识别、槽位抽取、情绪判定、话术生成。其余 17 个节点是查表、规则判断或校验，结果确定、可复现、零 token 成本。

---

## 三、核心机制

### 1. 事实占位符注入 —— 防幻觉

不把原始数据丢给模型让它复述，而是先查库拿到确定事实，注入模板占位符，模型只负责措辞。金额、时效、单号由代码填充，模型改不动。

### 2. 写操作四道闸

| 闸门 | 做法 |
| --- | --- |
| 白名单 | 只有 4 个工具可写：`aftersale.apply` / `aftersale.cancel` / `coupon.issue` / `order.action` |
| 幂等 | 幂等键 = `f(trace_id, tool, args)`，叠加业务锁，同一请求不会执行两次 |
| 超时 | 超时不重试，改走幂等键回查——「超时」意味着状态未知，重试可能造成双写 |
| 审批 | 订单金额超阈值（默认 500 元）不直接执行，落工单等人工批准 |

### 3. 三级兜底

`L1` 正常应答 → `L2` 规则应答（模型不可用时） → `L4` 转人工。

### 4. 人工闸门

触发后生成工单落库，坐席在 `/admin` 审核台处理：批准会带原幂等键重放写操作，驳回无副作用。

---

## 四、配置说明

系统启动时自动读取工程根目录的 `.env`（已存在的系统环境变量优先）。**所有变量都有默认值，不写 `.env` 也能以离线 Mock 模式直接运行。**

> 出于安全考虑，仓库未包含 `.env` 与 `.gitignore` 两个文件，需要你在本地手动创建。

### 在项目根目录新建 `.env`

```bash
# ---- 模型网关（OpenAI 兼容，DeepSeek 可用）----
# 不填 Key 则自动使用内置 MockLLMGateway，离线可跑通全流程
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=
LLM_MODEL_SMALL=deepseek-chat
LLM_MODEL_LARGE=deepseek-chat
LLM_TIMEOUT=20
# 填了 Key 但想强制离线时置 1
USE_MOCK_LLM=
MOCK_STREAM_DELAY_MS=15

# ---- 会话态存储（不配置则使用内存实现）----
REDIS_URL=

# ---- 业务闸门阈值 ----
# 置信度低于该值转人工
HANDOFF_CONFIDENCE_THRESHOLD=0.6
# 写操作金额超过该值不执行，转人工审批
WRITE_APPROVAL_AMOUNT=500
MAX_AGENT_CALLS=12
MAX_TOKENS=8000
DEADLINE_MS=30000

# ---- 数据源（SQLite + WAL）----
# 数据目录；留空则用工程根目录下的 data/
DB_DATA_DIR=
# 数据库文件路径；留空则用 <DB_DATA_DIR>/cs_agent.db
DB_PATH=
# db = 真实 SQLite / mock = 内存字典
DATA_SOURCE=db
DB_ENABLED=1
DB_ECHO=0

# ---- MCP 接入 ----
MCP_ENABLED=1
MCP_SERVER_ENABLED=1
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8765
# 业务工具是否走 MCP 协议调用（默认 0：进程内直连更快）
TOOLS_VIA_MCP=0
MCP_CLIENT_TIMEOUT=10

# ---- OMS 写操作后端 ----
# local = 写本地 SQLite（默认，零外部依赖）
# http  = 推给真实电商平台 / ERP
OMS_BACKEND=local
OMS_BASE_URL=
OMS_TOKEN=
OMS_TIMEOUT=5
```

### 变量分组说明

| 分组 | 关键变量 | 作用 |
| --- | --- | --- |
| 模型网关 | `LLM_API_KEY` | 留空走 Mock，填了走真实模型 |
| 业务闸门 | `WRITE_APPROVAL_AMOUNT` | 写操作金额上限，超过转人工审批 |
| 业务闸门 | `HANDOFF_CONFIDENCE_THRESHOLD` | 置信度闸门，低于则转人工 |
| 数据源 | `DB_PATH` | 自定义数据库位置；留空落 `data/cs_agent.db` |
| 数据源 | `DATA_SOURCE` | 置 `mock` 可强制走内存字典 |
| MCP | `TOOLS_VIA_MCP` | 置 `1` 后工具改由 MCP 协议调用 |
| OMS | `OMS_BACKEND` | `local` 写本地库，`http` 推真实平台 |

### 在项目根目录新建 `.gitignore`

```
.env
data/
__pycache__/
*.pyc
.pytest_cache/
.idea/
.vscode/
*.lnk
```

---

## 五、快速开始

```bash
git clone https://github.com/flyingstarss/autoCs
cd autoCs

# 1) 安装（Python 3.10+）
pip install -e ".[dev]"

# 2) 配置模型：在根目录新建 .env，填入 LLM_API_KEY；不填则以 Mock 模式运行

# 3) 初始化演示数据（22 单 / 16 会员 / 7 售后，幂等可重复执行）
cs-agent-seed

# 4) 启动服务
uvicorn cs_agent_system.api:app --port 8000
```

启动后：

| 地址 | 用途 |
| --- | --- |
| `http://127.0.0.1:8000/` | 可视化控制台（会话窗 + 本轮回放） |
| `http://127.0.0.1:8000/admin` | 人工审核台 |

```bash
cs-agent-demo        # 命令行跑一遍完整对话
python _check_llm.py # 4 次最小调用，确认模型连通性
```

---

## 六、接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/chat` | 单轮对话，返回回复 + trace_id |
| POST | `/chat/stream` | SSE 流式输出 |
| GET | `/trace/{id}` | 单条链路回放（每个节点的输入输出） |
| GET | `/traces` `/metrics` | 链路列表 / 聚合指标 |
| GET | `/tools` `/agents` `/healthz` | 自检 |
| GET | `/console` | 运行控制台 |
| GET | `/admin` | 人工审核台 |
| GET | `/admin/tickets` | 待办列表（按状态/类型筛选） |
| POST | `/admin/tickets/{id}/approve` | 批准（审批单会真正执行写操作） |
| POST | `/admin/tickets/{id}/reject` | 驳回 |

---

## 七、目录结构

```
src/cs_agent_system/
├── orchestrator/   LangGraph 状态图、七条编排规则、三级降级
├── agents/         4 类处理单元 + 基类（感知/决策/执行/保障）
├── core/           通信信封、预算控制、置信度、模型网关
├── guardrails/     事实校验、合规拦截
├── knowledge/      知识检索（BM25 + 字符 n-gram）
├── registry/       13 个进程内工具、数据访问、幂等、工单、OMS 抽象
├── mcp/            MCP Server + Client（10 个标准 MCP 工具）
└── api/            FastAPI、运行控制台、审核台
```

数据落 `data/cs_agent.db`（SQLite + WAL），DB 不可用时自动降级到内存字典。

---

## 八、技术栈

Python 3.10+ · LangGraph / LangChain · FastAPI · Pydantic v2 · SQLite(WAL) · MCP

检索为自研 BM25 + 字符 n-gram 词法检索，未引入向量数据库。

---

## 九、测试

```bash
pytest tests -q    # 41 个用例，默认离线可跑
python _smoke.py   # 27 条端到端断言
```

覆盖范围：意图路由、槽位抽取、幻觉拦截、写操作幂等、超时语义、审批与重放、降级链、工单状态机。
