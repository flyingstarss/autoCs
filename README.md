cs-agent-system
电商 / 私域全自动智能客服的多智能体系统框架

核心判断：回答型任务交给模型生成，事务型任务交给确定性工具执行。
模型只能「申请」写操作，不能「决定」写操作——所有安全约束以代码形式落在保障层，
而不是写在 Prompt 里靠模型自觉。

一、它解决什么问题
电商客服场景里，模型自由发挥的代价是不对称的：
  说错一句话（运费、时效、退款金额）  →  客诉、赔付
  调错一次工具（重复退款、错发优惠券） →  直接资损

所以本项目不追求「让 Agent 什么都能干」，而是把 90% 的流程做成确定性的规则链路，
只在真正需要语言理解的四类判断上调用模型。

二、架构

用户消息
    ↓
Orchestrator（LangGraph 状态图）
  21 个处理单元，其中仅 4 个调用模型
    ↓
感知层    意图识别 / 槽位抽取 / 情绪判定        ← 调模型
决策层    规则路由 + 置信度闸门 + 政策校验
执行层    工具调用（白名单 + 幂等 + 超时回查）
保障层    事实校验 / 合规拦截 / 兜底转人工
    ↓
回复（含 trace_id，全链路可回放）

只有 4 个节点调模型：意图识别、槽位抽取、情绪判定、话术生成。
其余 17 个节点是查表、规则判断或校验，结果是确定的、可复现的、零 token 成本。

三、四个核心机制

1. 事实占位符注入——防幻觉

不把原始数据丢给模型让它复述，而是先查库拿到确定事实，注入模板占位符，
模型只负责措辞。金额、时效、单号这类数字由代码填充，模型改不动。

2. 写操作四道闸

  白名单   只有 4 个工具能写：
           aftersale.apply / aftersale.cancel / coupon.issue / order.action

  幂等     幂等键 = f(trace_id, tool, args)，叠加业务锁，同一请求不会执行两次

  超时     超时不重试，改走幂等键回查——因为「超时」意味着状态未知，
           重试可能造成双写

  审批     订单金额超阈值（默认 500 元）不直接执行，落工单等人工批准

3. 三级兜底

  L1 正常应答 → L2 规则应答（模型不可用时） → L4 转人工

没有 L3——那一层被证明没有实际触发场景，已删除。
留着只会让降级链看起来更完整、实际更难维护。

4. 人工闸门

四条触发入口：低置信度、高风险情绪、合规拦截、写操作超阈值。

触发后生成工单落库，坐席在 /admin 审核台处理：
批准会带原幂等键重放写操作（保证只退一次），驳回无副作用，
重放超时标记「状态未知」而不是伪装成功。

四、快速开始

git clone <repo>
cd cs-agent-system

# 1) 安装（Python 3.10+）
pip install -e ".[dev]"

# 2) 配置模型（可跳过：不填 Key 会自动使用内置 Mock 网关）
cp .env.example .env          # 填写 LLM_API_KEY
# Windows 下改用： Copy-Item .env.example .env

# 3) 初始化演示数据（22 单 / 16 会员 / 7 售后，幂等可重复执行）
cs-agent-seed

# 4) 启动服务
uvicorn cs_agent_system.api:app --port 8000

打开 http://127.0.0.1:8000/ 是可视化控制台（会话窗 + 本轮回放），
/admin 是人工审核台。

Windows 用户可直接运行 restart_console.ps1，会处理端口占用并自动打开控制台。

其他命令：

  cs-agent-demo           命令行跑一遍完整对话
  python _check_llm.py    4 次最小调用，确认模型连通性

五、接口一览

  POST  /chat                          单轮对话，返回回复 + trace_id
  POST  /chat/stream                   SSE 流式输出
  GET   /trace/{id}                    单条链路回放（每个节点的输入输出）
  GET   /traces   /metrics             链路列表 / 聚合指标
  GET   /tools  /agents  /healthz      自检
  GET   /console                       运行控制台
  GET   /admin                         人工审核台
  GET   /admin/tickets                 待办列表（按状态/类型筛选）
  POST  /admin/tickets/{id}/approve    批准（审批单会真正执行写操作）
  POST  /admin/tickets/{id}/reject     驳回

六、目录结构

src/cs_agent_system/

  orchestrator/   LangGraph 状态图、七条编排规则、三级降级
  agents/         4 类处理单元 + 基类（感知/决策/执行/保障）
  core/           通信信封、预算控制、置信度、模型网关
  guardrails/     事实校验、合规拦截
  knowledge/      知识检索（BM25 + 字符 n-gram）
  registry/       13 个进程内工具、数据访问、幂等、工单、OMS 抽象
  mcp/            MCP Server + Client（10 个标准 MCP 工具）
  api/            FastAPI、运行控制台、审核台

数据落 data/cs_agent.db（SQLite + WAL），DB 不可用时自动降级到内存字典。

七、技术栈

Python 3.10+ · LangGraph / LangChain · FastAPI · Pydantic v2 · SQLite(WAL) · MCP

检索是自研的 BM25 + 字符 n-gram 词法检索，没有引入向量数据库——
知识库规模是千级条目，向量化带来的收益不抵运维成本。

八、测试

  pytest tests -q     41 个用例，默认 USE_MOCK_LLM=1，离线可跑
  python _smoke.py    27 条端到端断言

覆盖范围：意图路由、槽位抽取、幻觉拦截、写操作幂等、超时语义、
审批与重放、降级链、工单状态机。

九、已知不足

  审核台没有鉴权，只强制 operator 留痕；生产需接 SSO 或网关鉴权
  全新环境启动后是空库，需手动执行 cs-agent-seed
  知识检索停在词法层，未接入向量召回
  没有线上流量基线，指标口径尚未校准

详细设计说明见 docs/README-detailed.md
