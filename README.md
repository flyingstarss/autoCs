# 电商 / 私域全自动智能客服 · 多智能体系统框架

基于 **LangChain / LangGraph** 的多智能体客服系统实现，对应架构设计文档
`../ecom-privatedomain-agent-architecture/ecom-privatedomain-agent-architecture.html`。

设计原则：**区分「回答型任务」与「事务型任务」**。回答型任务由模型自主生成，
事务型任务由确定性工具执行、模型只负责解释与呈现。所有安全约束以代码形式落在
保障层，模型只能「申请」而不能「决定」。

---

## 快速开始

```bash
cd cs-agent-system

# 1) 安装（需要 Python 3.11+）
pip install -e .
pip install -e ".[dev,openai]"      # 需要 pytest / 真实模型时

# 2) 初始化数据库（拟真数据落 D 盘 data\cs_agent.db，幂等可重复执行）
python -m cs_agent_system.seed
python -m cs_agent_system.seed --check   # 只看各表行数

# 3) 跑 Demo（真实模型：读取根目录 .env 里的 DeepSeek 配置）
python -m cs_agent_system.demo
python -m cs_agent_system.demo --scenario 1     # 只跑某个场景
python -m cs_agent_system.demo --json            # 机器可读输出

# 3.1) 只想确认模型能不能连上（4 次最小调用，几秒出结果）
python _check_llm.py

# 4) 跑测试（离线：_smoke.py / pytest 强制 USE_MOCK_LLM=1，断言确定性）
pytest -q                        # 需要先 pip install -e ".[dev]"
python _smoke.py                 # 没装 pytest 时的冒烟脚本（零依赖）

# 5) 起 MCP Server（把业务能力暴露为标准 MCP 工具）
python -m cs_agent_system.mcp.server            # stdio（供 MCP 客户端拉起）
python -m cs_agent_system.mcp.server --http     # streamable-http（127.0.0.1:8765/mcp）

# 6) 起 HTTP 服务（Windows 推荐一键脚本，见下方「重启服务」）
uvicorn cs_agent_system.api:app --reload --port 8000
# 浏览器打开 http://127.0.0.1:8000/  → 内置可视化控制台（会话窗 + 本轮回放）
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d "{\"text\":\"我的订单到哪了\",\"user_id\":\"u_1001\"}"
```

### 重启服务（Windows，解决 `WinError 10013`）

`WinError 10013` 的含义是「端口被占用（或被 Windows 保留段排除）」，通常就是**上一版 uvicorn 还在跑**，
于是浏览器一直连着旧代码，看起来「结果怎么都没变」。直接双击：

- `restart_console.bat`：杀掉残留 uvicorn（`--reload` 的监督进程 + 真正绑端口的子进程）→
  找个可用端口（8000，被占则自动退到 8010/8020…）→ 起服务 → 等 `/healthz` 带上
  `build=per-turn-budget+deepseek` → 打开控制台
- `restart_console.bat -Stop`：只停服务，不起新的

如果脚本提示端口都被占用，检查是不是 Windows 保留段：`netsh int ipv4 show excludedportrange protocol=tcp`。

### 内置可视化控制台

`http://127.0.0.1:8000/`（或 `/console`）是随服务托管的零依赖单页，不需要 node/npm：

- 左侧会话窗：输入用户消息，可切换 `user_id`（身份锚定按用户查订单）
- 快捷样例：售前尺码 / 售后退款 / 高风险投诉 / 意图澄清 / 物流并行 / 多意图规划
- 右侧本轮回放：意图置信度、路由、目标 Agent、质量分、是否转人工，以及
  P1→P4 / D1→D3 / E* / Q* 的逐事件时序（含工具调用耗时与校验结论）

页面源码在 `api/console.py`（一个 HTML 常量），改完重启服务即生效。

### 流式输出（SSE）

`POST /chat/stream` 按 SSE 推四类事件，控制台已按名分发：

| 事件 | 时机 | 前端行为 |
|---|---|---|
| `meta` | 连接建立 | 建气泡 |
| `stage` | 进入每个节点 / 感知域每步 | 显示「识别意图与槽位…」「抽取槽位…」「查询 E1…」等进度行 |
| `delta` | D3 逐块产出 | 追加正文（撤掉进度行） |
| `done` | 全链路完成 | 用最终 `response` 覆盖草稿（Q2/Q3 可能改写） |

注意两点：
1. **阶段进度是真正的「过程可见」**，D3 之前 P1/P2/P3 是阻塞调用（约 1–2 秒），
   没有 stage 事件时用户只能看到静止的「思考中」。
2. **`delta` 是草稿**：D3 之后还有事实校验与合规改写，最终文本以 `done` 为准。

---

## 分层结构

| 层 | 目录 | 职责 |
|---|---|---|
| L0 接入层 | `api/` | 多渠道适配、统一 ChatRequest/ChatResponse、渠道能力裁剪 |
| L1 编排层 | `orchestrator/` | LangGraph 状态机：路由 / 规划 / 并发扇出 / 反思回边 / 预算 / 降级 |
| L2 处理单元层 | `agents/` | 21 个处理单元，其中**仅 4 个调用大模型**（见下表） |
| L3 保障层 | `guardrails/` | 合规规则引擎、事实校验 |
| L4 能力层 | `registry/` `knowledge/` `mcp/` | 工具注册中心（声明式权限/幂等）、混合检索、**MCP Server**（业务能力对外暴露） |

### 处理单元构成

此前把所有单元统称「智能体」是不准确的：21 个单元里**只有 4 个真的调用大模型**。
每个单元用 `kind` 标注性质、`uses_llm` 标注是否消耗 token（`GET /agents` 可查）：

| kind | 单元 | 是否调模型 | 职责 |
|---|---|---|---|
| `agent` | P1 P2 P3 D3 | ✅ | 意图识别 / 槽位抽取 / 情绪分级 / 响应生成 |
| `router` | D1 | ❌ | 选路径：短路 / 规划 / 澄清 / 转人工 |
| `planner` | D2 | ❌ | 选执行顺序与依赖（任务 DAG） |
| `capability` | E1 E2 E4 E5 E6 E8 P4 Q1 Q4 | ❌ | 确定性能力：取数与事实投影、上下文组装、检索、评分 |
| `writer` | E3 E7 | ❌ | 写操作：幂等键 + 超时回查状态机 |
| `guard` | Q2 Q3 Q5 | ❌ | 独立否决权：事实校验 / 合规 / 人工接管 |
| `computed` | E9 | ❌ | 规则型权益计算 |

**token 只在 4 处产生**：P1、P2、P3 各一次小模型调用，D3 一次大模型调用。
其余单元是纯确定性代码，合并它们省不下任何 token，收益在可维护性与 trace 可读性。

```
src/cs_agent_system/
├── core/
│   ├── envelope.py    AgentEnvelope（from/to/trace_id/confidence…）与 SessionContext
│   ├── intents.py     封闭意图集、槽位规格、路由表（配置驱动）
│   ├── confidence.py  置信度单一事实源：结果状态 × 任务档位 → 数值
│   ├── budget.py      三重预算：调用数 + Token + 墙钟时间
│   ├── state.py       会话态存储（Redis / 内存），单一写入者
│   ├── llm.py         统一 LLM 网关 + 小/大模型分级路由 + Mock 实现
│   └── ids.py         trace/session id、业务锁键、幂等键、写工具白名单
├── registry/
│   ├── tool_registry.py  工具注册中心：只读可重试、写操作禁重试、业务锁 + 幂等键
│   ├── tools.py          全部工具（4 个写工具 + 若干只读工具）
│   ├── facts.py          事实投影层：工具返回 → 话术事实（工具层与表现层解耦）
│   ├── db.py             SQLite 访问层：连接管理、建表、补列迁移、JSON 编解码
│   ├── idempotency.py    幂等存储：尝试记录 + 业务锁 + 状态机（DB 持久化 / 内存兜底）
│   ├── refund_revision.py 退款改额：先撤旧单再下新单（幂等键覆盖不到的那一半）
│   ├── oms.py            OMS 抽象层：写操作后端（local SQLite / http 真实平台）
│   └── data.py           业务数据与策略配置服务（DB 优先，内存兜底）
├── mcp/
│   ├── server.py         MCP Server：10 个标准 MCP 工具（数据源 = D 盘 SQLite）
│   └── client.py         MCP 客户端：stdio / streamable-http 两种接入形态
├── knowledge/retriever.py  BM25 + 字符 n-gram 混合检索 + 重排 + 生效日期
├── guardrails/
│   ├── compliance.py   敏感词 / 绝对化用语 / 虚假承诺 / 诱导话术
│   └── factcheck.py    逐句比对生成内容与证据
├── agents/
│   ├── base.py          Unit 基类（kind / uses_llm 元数据）+ 计时/兜底/异常隔离
│   ├── perception.py    P1 意图 / P2 槽位 / P3 情绪风险 / P4 上下文
│   ├── decision.py     D1 路由 / D2 规划 / D3 生成
│   ├── execution.py    E1–E9 业务域（6 个能力单元 + 2 个写操作 + 1 个规则计算）
│   └── assurance.py    Q1 检索 / Q2 事实 / Q3 合规 / Q4 质量 / Q5 接管
└── orchestrator/
    ├── graph.py        LangGraph 状态机（主编排逻辑）
    ├── state.py        编排器状态
    └── degrade.py      L2 模板兜底（L1/L4 在 graph.py 内）
```

---

## 核心机制

### 1. 统一通信信封

```python
AgentEnvelope(
    trace_id, session_id, turn,
    from_agent, to_agent,     # 发送方 / 接收方
    intent, slots, missing,   # 本轮理解
    payload, evidence,        # 业务产出与证据
    confidence,               # 一等公民，编排器据此裁决
    need_handoff, cost_ms,
)
```

### 2. 编排器七条规则（均有代码落点）

| 规则 | 实现位置 |
|---|---|
| 短路优先 | `ROUTE_TABLE` 中 `direct_tool`；单工具可解不进多单元链 |
| 并行优先 | `orchestrator/graph.py::_execute_agents`，无依赖目标用线程池并行 |
| 三重预算 | `core/budget.py::BudgetGuard`，超限即降级不重试 |
| token 单一记账口 | `agents/base.py::AgentRuntime.call_llm`，全系统只有这一个入口能调模型 |
| 工具超时真实生效 | `registry/tool_registry.py::ToolRegistry._invoke`，按 `timeout_ms` 掐断 |
| 降级可见 | `core/llm.py::LLMResult.degraded`，模型不可用时响应里带 `degraded_reasons` |
| 长任务可续接 | `core/envelope.py::SessionContext.pending_tasks` + `registry/tasks.py`（SQLite 持久化） |
| 任务可取消 | `agents/perception.py` 取消短语识别 → `orchestrator/graph.py::_sync_tasks` 清栈 |
| 置信度裁决 | `agents/decision.py::Router`，低于阈值走澄清 |
| 失败隔离 | `agents/base.py::Unit.run`，单元异常不拖垮链路 |
| 幂等执行 | `registry/tool_registry.py`，写操作幂等键 + 去重 |

> **token 记账必须收口**：模型调用若散落在各单元的 `handle()` 里，必然漏记
> ——历史上 P1/P2/P3 三处调用就没记账，`max_tokens=8000` 形同虚设（只有 D3 记了）。
> 现在所有模型调用统一走 `AgentRuntime.call_llm()`，它负责预检、记账与留痕；
> 每轮真实消耗可在 `POST /chat` 响应的 `budget` 字段中看到。

### 3. 三条不可绕过的硬约束

1. **写操作只有四个**：`aftersale.apply` / `aftersale.cancel` / `coupon.issue` / `order.action`。
   注册时校验，不在白名单直接抛异常。撤销必须是独立写操作 —— 删本地记录不代表钱回来了。
2. **写操作超时禁止自动重试**。超时 ≠ 失败，可能已成功；正确做法是用幂等键
   回查状态（`aftersale.status` / `coupon.status`）。
3. **承诺值一律来自策略服务**。退款时效、赔付上限、优惠叠加规则全部读
   `registry/data.py::POLICY`，模型不得推算。

### 4. 强个人数据二次核验

`find_order(user_id, order_no)` 以用户身份为主键查询。用户在对话里口述的订单号
若不属于当前用户，槽位会被标记为待核验而**不会**被采信（`agents/perception.py::SlotAgent`）。

### 5. 事实占位符注入防幻觉

D3 只做「模板 + 服务端事实」的渲染与语气组织：

```
RETURN_APPLY = "已为您提交{type}申请（售后单号 {aftersale_no}），退款金额 {refund_amount} 元，
                按原渠道退回，到账时效约 {refund_sla_days} 个工作日。"
```

模型不参与任何加减乘除，也不回忆业务数字。Q2 再对生成结果逐句比对证据，
不通过则走一次反思回边（最多 1 轮），仍不过则转人工。

### 6. 兜底三级（不是四级）

```
L1 草稿保留 → L2 模板兜底 → L4 转人工
```

L1 在 `orchestrator/graph.py::_finalize`：模型降级但已有草稿时照用，并标记 degraded。
L2 在 `orchestrator/degrade.py`：按意图给固定话术（覆盖全部意图，未知意图回落 CHITCHAT）。
L4 在 `_handoff` + Q5：带上下文转坐席。

此前文档写的是「L1–L4 降级链」，其中 **L3 规则应答从未被任何代码路径触发**
（L2 的模板已覆盖所有意图），已连同死分支一起删除。

### 7. 人工闸门四条入口

| 入口 | 触发 | 位置 |
|---|---|---|
| 风险 | P3 风险等级 ≥ 高 | `perception.py::EmotionAgent` |
| 置信度 | D1 置信度不足 | `decision.py::Router` |
| 事实 | Q2 连续失败 | `orchestrator/graph.py::_route_after_verify` |
| 金额 | 写操作超阈值 | `registry/tools.py::aftersale_apply` **写入前**判定，超阈值直接返回 `needs_approval`，不产生副作用；阈值由 `WRITE_APPROVAL_AMOUNT` 配置 |

---

## 数据源与 MCP 接入

### 数据落在哪

**全部在 D 盘，不写 C 盘。**

```
D:\项目\cs-agent-system\data\cs_agent.db        # SQLite 主文件
D:\项目\cs-agent-system\data\cs_agent.db-wal    # WAL 日志
D:\项目\cs-agent-system\data\cs_agent.db-shm    # 共享内存索引
```

路径由 `config.Settings.db_path` 决定，默认 `<工程根>/data/cs_agent.db`，
可用 `DB_DATA_DIR` / `DB_PATH` 覆盖。

### 数据流

```
用户提问
   ↓
处理单元（E1/E2/…）
   ↓
ToolRegistry ────→ registry/tools.py ──→ registry/data.py ──→ SQLite（D 盘）
   ↓                      ↓
   │                 facts 投影（registry/facts.py）
   ↓
D3 话术渲染
```

**三层各司其职，互不知情：**

| 层 | 文件 | 职责 |
|---|---|---|
| 工具层 | `registry/tools.py` | 只管调数据、返回原始结构，不知道话术模板 |
| 数据层 | `registry/data.py` + `registry/db.py` | 只管取数与策略值，是业务口径的唯一来源 |
| 投影层 | `registry/facts.py` | 声明「哪些字段进话术、缺字段算不算完整命中」 |

### 自动降级

DB 不可用（未播种 / 文件损坏 / 权限问题）时，`registry/data.py` 自动回落到内存字典，
**客服还能答话**，而不是整条链路崩掉。`/healthz` 的 `data_source.mode` 会显示当前是 `db` 还是 `mock`。

切换：`DATA_SOURCE=db|mock`（`config.py` → `Settings.data_source`）。

### 拟真数据与边界覆盖

数据分两部分：**基础数据**（`seed.py` 内联）+ **扩充数据**（`seed_data.py`）。
`python -m cs_agent_system.seed` 一次性写入，幂等可重复执行。

| 表 | 基础 | 扩充后 | 覆盖维度 |
|---|---|---|---|
| orders | 6 | **22** | 待付款 / 待发货 / 已发货 / 已签收 / 已取消 / 退款中 / 已拒收 / 已完成；已付×未付、有无运单号、有无发票 三维交叉 |
| products | 6 | **28** | 服饰 / 运动 / 家居 / 数码 / 美妆 / 食品 六大品类；全尺码缺货、部分缺货、单尺码、无尺码、高低价、免邮门槛边界（99.0 / 19.9）、无参数商品 |
| members | 4 | **16** | 普通 / 银卡 / 金卡 / 钻石 四级；积分 0、积分极高、成长值与积分不匹配、空标签 |
| coupons | 5 | **22** | unused / used / expired 三态；六品类 + 全品类；无门槛券（0）、等于免邮门槛（99）；品类不匹配场景 |
| trackings | 9 | **29** | 完整在途 / 单节点 / 已签收闭环 / 异常滞留 / 转运中心 |
| aftersales | 0 | **8** | 退货退款 / 换货 / 补发 / 维修 / 仅退款；已受理 / 退款中 / 已完成 / 已驳回；赔付上限边界（200） |
| policies | 11 | 11 | 退款时效、赔付上限、券叠加规则、免邮门槛等承诺值 |

**测试账号**（`u_5001`~`u_5012`）：每个账号对应一类典型场景，
如 `u_5008` 是 1299 元大额客户、`u_5010` 持有「数码券 + 服饰券」用于验证品类不匹配、
`u_5005` 是多商品订单。跨用户越权访问已在 `find_order` 层拦截。

> 基础数据里的 `SO202610280001` / `SKU-10086` / `u_1001` / `CPN-NEW-20` / `SF1234567890`
> 被冒烟测试与单元测试断言依赖，**扩充数据不会覆盖它们** ——
> 新订单一律挂在新用户名下，避免抢占 `u_1001` 的「事实来源单」位置。

### MCP Server

业务能力以标准 MCP 工具暴露，任何 MCP 客户端都能复用：

| MCP 工具 | 作用 |
|---|---|
| `order_query` | 订单详情 / 订单列表（含 `primary_order_no` 事实来源单） |
| `logistics_track` | 物流轨迹 |
| `product_detail` | 商品详情与参数（支持按文本模糊定位） |
| `inventory_check` | 库存与规格 |
| `member_profile` | 会员等级 / 积分 / 成长值 |
| `payment_query` | 支付状态与可开票类型 |
| `coupon_list` | 可用优惠券 |
| `pricing_calc` | 到手价（**唯一计价口径**） |
| `policy_get` | 策略承诺值（退款时效、赔付上限、叠加规则） |
| `data_source_status` | 数据源诊断（DB / mock、文件路径、各表行数） |

启动：

```bash
python -m cs_agent_system.mcp.server            # stdio：供 MCP 客户端拉起
python -m cs_agent_system.mcp.server --http     # streamable-http：127.0.0.1:8765/mcp
```

本项目也能反向调用自己的 MCP Server：

```python
from cs_agent_system.mcp import client
client.call_tool_sync("order_query", {"user_id": "u_1001", "order_no": "SO202610280001"})
```

在 `.env` 里置 `TOOLS_VIA_MCP=1` 后，`ToolRegistry` 会尝试把 MCP 工具一并注册进来
（进程内工具优先，不覆盖；MCP 工具**一律按只读处理**，不允许绕过 `WRITE_TOOLS` 白名单）。

**注意**：MCP 工具名与进程内工具名多数重合（`order_query` ↔ `order.query`），
重名会被跳过；实际只有 `pricing_calc` / `policy_get` / `data_source_status`
这几个本地不存在的工具会被补充注册，而它们目前没有调用方。
这条开关的定位是「补充本仓库外的能力」，不是替换本地实现。

### 三个业务口径（写代码时别踩）

1. **事实来源单跳过「已取消」** —— 用户问「我的订单」时报一单已取消的订单属于答非所问；
2. **物流取最近一单「有轨迹」的** —— 最新单可能是待付款/已取消，没有运单号；
3. **承诺值一律取自 `policy_get`** —— 退款时效、赔付上限不得在单元里硬编码。

---

## 写操作：幂等与 OMS 后端

### 两把锁：幂等键 + 业务锁

写操作同时维护两个键，职责不同：

| 键 | 组成 | 回答的问题 | 唯一性由谁保证 |
|---|---|---|---|
| 业务锁 `business_key` | `sha256(tool + 目标实体 + version)` | 这笔业务事实是不是已经发生过 | `idempotency` 表的部分唯一索引 |
| 幂等键 `key` | `sha256(业务键 + trace_id + user_id + 金额)` | 这次调用是不是刚才那次的重发 | 表主键 |

只有 `business_scoped=True` 的写工具（退款申请 / 撤销）会抢业务锁；
发券这类允许重复领取的操作不启用，避免把正常的多轮领取挡掉。

状态机（`idempotency.status`）：

```
pending ──成功──> executed ──撤销──> voided
   │                  │
   ├──超时/异常──────> uncertain（禁止重发，只能按 key 回查）
   ├──需人工审批──────> rejected（无副作用，不占锁，审批后可重提）
   └──业务失败────────> failed（无副作用，不占锁）
```

**为什么光有幂等键不够**：`trace_id` 每轮重建，用户改一次金额就必然产生新键，
旧的照样会被执行 —— 于是下游出现两笔退款。业务锁把「同一订单同一版本」
锁成唯一；金额变更必须 `version + 1` 变成新事实，并且**先撤旧单再下新单**。

### 退款改额：先撤后建

用户在已有退款申请上要求改金额时，`registry/refund_revision.py::RefundRevisionService`
按固定顺序执行，顺序不可交换：

```
撤销旧单（aftersale.cancel）成功 ──> 才允许提交新单（version+1）
        └── 撤销失败 / 旧单状态未知 ──> 中止并转人工（绝不连下两笔）
```

作废分两层，缺一不可：

1. **下游作废**：调 `aftersale.cancel`，让支付/OMS 侧那笔真的失效；
2. **本地作废**：幂等记录置 `voided`，释放业务锁。

只做第 2 步等于自欺欺人 —— 钱可能已经退出去了，本地删记录不会让它回来。
金额始终由服务端重算（`min(订单实付, 赔付上限)`），**不接受用户指定金额**。

### 幂等记录落 D 盘 SQLite

写操作都必须带幂等键（同上表）。键、业务锁与执行结果存在 `idempotency` 表里：

```
D:\项目\cs-agent-system\data\cs_agent.db  →  表 idempotency
```

**为什么必须落库**：幂等键的用途是「写操作超时后回查这个操作到底执行了没」。
如果只在内存里，服务重启后就查不到了 —— 恰好在最需要它的时候失效。

回查链路：

```
aftersale.apply 超时
   → status_unknown=True（禁止重试）
   → E3 按幂等键调 aftersale.status
   → ToolRegistry.lookup_idempotent(key) → 查 DB idempotency 表
   → 已执行 → 正常告知用户；未确认 → 转人工
```

存储实现在 `registry/idempotency.py`，DB 不可用时自动回落内存实现。

### OMS 后端可切换

写操作的最终落地由 `OmsClient` 抽象层决定，接口只有四个方法：

| 方法 | 作用 |
|---|---|
| `apply_aftersale(...)` | 提交售后单（按 `idempotency_key` 服务端去重） |
| `cancel_aftersale(...)` | 撤销售后单（状态置「已撤销」并恢复订单原状态） |
| `issue_coupon(...)` | 发放优惠券 |
| `update_order(...)` | 订单动作（取消等） |

两个实现，接口完全一致：

| 实现 | 说明 |
|---|---|
| `LocalOmsClient` | 写本地 SQLite（默认，零外部依赖） |
| `HttpOmsClient` | 推给真实电商平台/ERP，幂等键放 `X-Idempotency-Key` 请求头 |

切换只需改 `.env`：

```ini
OMS_BACKEND=http
OMS_BASE_URL=https://oms.internal
OMS_TOKEN=xxx
OMS_TIMEOUT=5
```

**HTTP 后端的错误分级**（决定上层怎么处置）：

| 情况 | 抛出 | 上层行为 |
|---|---|---|
| 4xx 业务拒绝 | `OmsError` | 不重试、不回查，直接告知用户失败 |
| 5xx 服务端错误 | `OmsTimeout` | 状态未知 → 按幂等键回查 |
| 连接失败 / 超时 | `OmsTimeout` | 状态未知 → 按幂等键回查 |

写工具从不自己重试 —— 这是 `ToolRegistry` 层的硬规则，不依赖工具实现自觉。

---

## 接入真实模型

工程根目录的 `.env` 已配置为 **DeepSeek**（OpenAI 兼容协议），无需再复制模板：

```ini
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxxx          # 已写入本地 .env
LLM_MODEL_SMALL=deepseek-chat
LLM_MODEL_LARGE=deepseek-chat
DEADLINE_MS=30000            # 真实模型比 Mock 慢，墙钟预算放宽
```

`.env` 由 `config.py` 自带的解析器自动加载（不引入 python-dotenv）；系统环境变量优先级更高。
`.env` 已被 `.gitignore` 排除，Key 不会进版本库。

- 有 `LLM_API_KEY` 且 `langchain-openai` 可用 → `LangChainLLMGateway`（真实模型）；
- 想临时离线 / 不花额度 → `.env` 里设 `USE_MOCK_LLM=1`；
- `_smoke.py` 与 `pytest` **强制** `USE_MOCK_LLM=1`，保证回归断言确定性；
  想验证真实模型请跑 `python -m cs_agent_system.demo` 或打开控制台对话；
- 真实调用失败（网络/超时/鉴权）会自动降级到规则网关（`model=mock-fallback`），客服链路不中断。

**哪些环节真的用模型（只有 4 处调用点）**

| 环节 | Agent | 方法 | 降级/兜底 |
|---|---|---|---|
| 意图识别 | P1 | `classify(intent)` | 关键词规则打分（Mock 即此实现） |
| 槽位抽取 | P2 | `classify(slots)` | 正则抽取；订单号一律服务端二次核验 |
| 情绪与风险 | P3 | `classify(emotion)` | 监管词/情绪词规则前置，可单独把风险抬到 high |
| 话术生成 | D3 | `generate` | 事实占位符模板；模型只润色语气 |

其余环节**刻意不用模型**：D1 路由查配置表、D2 任务图按依赖推导、Q1 混合检索、Q2 数字型事实比对、
Q3 合规正则、Q4 加权评分、Q5 工单摘要模板。这样路由可审计、校验可复现、延迟可控。

注意：即使接入大模型，金额、时效、库存、优惠叠加等数字仍只来自工具与策略服务；
模型若自行编造数字，会被 Q2 事实校验拦下并触发反思回边（最多 1 次）→ 仍不通过则转人工。

模型分级路由（对应文档 14.1）—— **全系统仅 4 处模型调用**：

| 任务 | 档位 | 处理单元 | 是否调模型 |
|---|---|---|---|
| 意图识别 | 小模型 | P1 | ✅ |
| 槽位抽取 | 小模型 | P2 | ✅ |
| 情绪与风险分级 | 小模型 + 规则前置 | P3 | ✅ |
| 话术生成 | 大模型 | D3 | ✅ |
| 事实校验 / 合规 / 质量评分 | 确定性规则 | Q2 / Q3 / Q4 | ❌ |
| 工具型执行 | 零模型推理 | E1 / E2 / E4 / E5 / E6 / E8 | ❌ |
| 路由 / 规划 | 零模型推理 | D1 / D2 | ❌ |

---

## 扩展新的业务域

架构文档的最终检验标准是「新增业务域不改既有处理单元」。落地方式：

1. 在 `core/intents.py` 登记新意图（名称、描述、槽位规格、执行单元、路由方式）；
2. 在 `core/intents.py::ROUTE_TABLE` 登记路由（配置驱动，不改 Prompt）；
3. 在 `registry/tools.py` 注册只读工具（写操作需先加入 `core/ids.py::WRITE_TOOLS`
   并接受幂等与审批约束）；
4. 在 `registry/facts.py` 写一个事实投影函数，声明「哪些字段进话术、缺字段算不算完整命中」；
5. 在 `agents/execution.py` 新增一个能力单元，继承 `_ProjectedCapability` 并指定
   `tool_name` + `projection_key`；
6. 若需新话术，在 `agents/decision.py::ResponseWriter.TEMPLATES` 增加模板。

感知域、决策域、保障域、编排器、降级链、预算、幂等均无需改动。

---

## 测试

```bash
pytest -q          # 完整用例（需 pytest / pytest-asyncio）
python _smoke.py   # 冒烟脚本：不依赖 pytest，直接跑 6 条端到端断言
```

覆盖：信封契约、三重预算、幂等键、只读/写工具边界（写超时禁重试、幂等去重）、
合规规则、事实校验、混合检索、编排器端到端（短路 / 澄清 / 转人工 / 写操作幂等 /
身份核验 / 会话态写入）。

`_smoke.py` 覆盖的 6 条链路：订单状态短路、物流事实、售后退款（幂等键 + 策略时效）、
高风险投诉转人工、意图不明澄清、尺码咨询。
