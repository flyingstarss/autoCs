"""内置可视化控制台（零依赖单页）。

不是独立前端工程：这份 HTML 由 FastAPI 直接托管在 `/` 与 `/console`，
用于观察一次会话里的意图识别、路由决策、工具调用与保障域校验的完整时序。
"""

CONSOLE_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>多智能体客服 · 运行控制台</title>
<style>
  :root{
    --bg:#0e1117; --panel:#161b22; --line:#262d38; --text:#e6edf3;
    --muted:#8b949e; --accent:#2f81f7; --ok:#3fb950; --warn:#d29922;
    --bad:#f85149; --user:#1f6feb; --bot:#21262d;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0; background:var(--bg); color:var(--text);
    font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
    display:flex; flex-direction:column;
  }
  header{
    display:flex; align-items:center; justify-content:space-between;
    padding:12px 20px; border-bottom:1px solid var(--line); background:var(--panel);
  }
  .brand{display:flex; align-items:center; gap:10px; font-weight:600}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--ok)}
  .badges{display:flex; gap:8px; flex-wrap:wrap; align-items:center}
  .badge{
    font-size:12px; color:var(--muted); border:1px solid var(--line);
    padding:2px 8px; border-radius:999px; background:#0d1117;
  }
  .badge.warn{color:var(--warn); border-color:#4a3a12}
  .adminlink{
    font-size:12px; color:var(--accent); text-decoration:none;
    border:1px solid #1d3a63; background:#0d1a2b; padding:3px 10px; border-radius:999px;
  }
  .adminlink:hover{background:#12294a}
  main{
    flex:1; display:grid; grid-template-columns:minmax(360px,1.05fr) minmax(380px,1fr);
    gap:16px; padding:16px; min-height:0;
  }
  section{
    background:var(--panel); border:1px solid var(--line); border-radius:12px;
    display:flex; flex-direction:column; min-height:0; overflow:hidden;
  }
  .panel-title{
    padding:10px 14px; font-size:12px; color:var(--muted); letter-spacing:.08em;
    border-bottom:1px solid var(--line); display:flex; justify-content:space-between;
  }
  .chips{display:flex; gap:8px; flex-wrap:wrap; padding:10px 14px; border-bottom:1px solid var(--line)}
  .chip{
    font-size:12px; color:var(--text); background:#0d1117; border:1px solid var(--line);
    border-radius:999px; padding:4px 10px; cursor:pointer; transition:.15s;
  }
  .chip:hover{border-color:var(--accent); color:var(--accent)}
  .stream{flex:1; overflow-y:auto; padding:16px 14px; display:flex; flex-direction:column; gap:12px}
  .msg{max-width:85%; padding:9px 12px; border-radius:10px; white-space:pre-wrap; word-break:break-word}
  .msg.user{align-self:flex-end;background:var(--user);color:#fff;border-bottom-right-radius:2px}
  .msg.bot{align-self:flex-start;background:var(--bot);border:1px solid var(--line);border-bottom-left-radius:2px}
  .msg .foot{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;font-size:11px;color:var(--muted)}
  .msg .tag{border:1px solid var(--line);border-radius:999px;padding:0 6px}
  /* 阶段进度：文字产出前显示流水线走到哪一步，避免只有一个静止的「思考中」 */
  .msg .stage{font-size:12px;color:var(--muted);margin-bottom:4px}
  .msg .stage::before{content:"▸ ";color:var(--accent)}
  .composer{display:flex; gap:8px; padding:12px 14px; border-top:1px solid var(--line)}
  input{
    flex:1; min-width:0; background:#0d1117; border:1px solid var(--line); color:var(--text);
    border-radius:8px; padding:10px 12px; font:inherit; outline:none;
  }
  input:focus{border-color:var(--accent)}
  #uid{flex:0 0 92px}
  button{
    border:none; border-radius:8px; padding:0 18px; font:inherit; font-weight:600;
    background:var(--accent); color:#fff; cursor:pointer;
  }
  button:disabled{opacity:.5; cursor:not-allowed}
  .meta{padding:12px 14px; display:grid; grid-template-columns:repeat(3,1fr); gap:8px; border-bottom:1px solid var(--line)}
  .card{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:8px 10px}
  .card .k{font-size:11px;color:var(--muted)}
  .card .v{font-size:13px;font-weight:600;margin-top:2px;word-break:break-all}
  .timeline{flex:1; overflow-y:auto; padding:10px 14px 16px}
  .ev{display:grid; grid-template-columns:18px 40px 92px 58px auto; gap:8px; align-items:baseline;
      padding:6px 0; border-bottom:1px dashed #1e242e; font-size:12px}
  .ev:last-child{border-bottom:none}
  .mark{text-align:center;font-weight:700}
  .aid{font-weight:700;font-family:ui-monospace,Consolas,monospace}
  .evname{color:var(--muted)}
  .cost{color:var(--muted);text-align:right;font-family:ui-monospace,Consolas,monospace}
  .detail{color:#c9d1d9;word-break:break-all}
  .p-P{color:#7aa2f7}.p-D{color:#bb9af7}.p-E{color:#2ac3de}.p-G{color:#ff9e64}.p-Q{color:#e0af68}
  .m-ok{color:var(--ok)}.m-handoff{color:var(--warn)}.m-fail{color:var(--bad)}.m-idle{color:#484f58}
  .empty{color:var(--muted);font-size:12px;padding:16px 0}
  @media (max-width:900px){
    main{grid-template-columns:1fr; grid-template-rows:1.2fr 1fr}
  }
</style>
</head>
<body>
<header>
  <div class="brand"><span class="dot"></span>电商 / 私域 多智能体客服 · 运行控制台</div>
  <div class="badges" id="badges"></div>
  <a class="adminlink" href="/admin">人工审核台 →</a>
</header>
<main>
  <section>
    <div class="panel-title"><span>会话</span><span id="session">-</span></div>
    <div class="chips" id="chips"></div>
    <div class="stream" id="stream">
      <div class="msg bot">你好，我是智能客服。可以直接输入问题，或点上面的示例。</div>
    </div>
    <form class="composer" id="composer">
      <input id="uid" value="u_1001" title="用户ID（决定身份锚定与订单归属）" />
      <input id="text" placeholder="输入用户消息，例如：订单号 SO202610280001 质量太差了，我要退款" autocomplete="off" />
      <button id="send" type="submit">发送</button>
    </form>
  </section>
  <section>
    <div class="panel-title"><span>本轮回放</span><span id="traceid">等待请求…</span></div>
    <div class="meta" id="meta"></div>
    <div class="timeline" id="timeline">
      <div class="empty">发送一条消息后，这里会按时间顺序显示 P1→P4 / D1→D3 / E* / Q* 的调用链。</div>
    </div>
  </section>
</main>
<script>
const CHIPS = [
  { label: "售前 · 尺码", text: "这件卫衣有黑色 M 码吗" },
  { label: "售后 · 退款", text: "订单号 SO202610280001 质量太差了，我要退款" },
  { label: "投诉 · 转人工", text: "你们这什么破服务，我要12315投诉！" },
  { label: "意图不明 · 澄清", text: "那个事情怎么样了" },
  { label: "物流 · 并行", text: "我的快递到哪了" },
  { label: "多意图 · 规划执行", text: "订单号 SO202610280001 尺码不合适，我要退款，另外物流到哪了" }
];
const stream = document.getElementById("stream");
const timeline = document.getElementById("timeline");
const meta = document.getElementById("meta");
const chips = document.getElementById("chips");
const form = document.getElementById("composer");
const input = document.getElementById("text");
const uid = document.getElementById("uid");
const sendBtn = document.getElementById("send");
const badges = document.getElementById("badges");

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}
function scroll() { stream.scrollTop = stream.scrollHeight; }

function addFoot(box, data) {
  const foot = el("div", "foot");
  foot.appendChild(el("span", "tag", "意图 " + data.intent));
  foot.appendChild(el("span", "tag", "路由 " + data.route));
  foot.appendChild(el("span", "tag", "质量 " + (data.quality_score || 0).toFixed(2)));
  if (data.need_handoff) foot.appendChild(el("span", "tag", "已转人工"));
  box.appendChild(foot);
}

function addMsg(role, text, data) {
  const box = el("div", "msg " + role);
  const body = el("div", "body", text);
  box.appendChild(body);
  if (data) addFoot(box, data);
  stream.appendChild(box);
  scroll();
  return box;
}

function setMsgText(box, text) {
  const body = box.querySelector(".body");
  if (body) body.textContent = text;
  scroll();
}

// 阶段进度行：传空字符串则移除。首个 delta 到达前用它显示「正在查订单…」
function setStage(box, text) {
  let node = box.querySelector(".stage");
  if (!text) {
    if (node) node.remove();
    return;
  }
  if (!node) {
    node = el("div", "stage");
    box.insertBefore(node, box.firstChild);
  }
  node.textContent = text;
  scroll();
}

CHIPS.forEach(function (c) {
  const chip = el("button", "chip", c.label);
  chip.type = "button";
  chip.onclick = function () { input.value = c.text; input.focus(); };
  chips.appendChild(chip);
});

async function boot() {
  try {
    const [tools, units] = await Promise.all([
      fetch("/tools").then(function (r) { return r.json(); }),
      fetch("/agents").then(function (r) { return r.json(); })
    ]);
    badges.appendChild(el("span", "badge", "工具 " + tools.length));
    badges.appendChild(el("span", "badge", "处理单元 " + units.length));
    // 只有模型驱动单元才消耗 token：如实展示，避免误以为全部单元都在烧钱
    const llmUnits = units.filter(function (u) { return u.uses_llm; });
    const llmBadge = el("span", "badge", "模型单元 " + llmUnits.length + "/" + units.length);
    llmBadge.title = llmUnits.map(function (u) { return u.id + " " + u.kind; }).join(" · ");
    badges.appendChild(llmBadge);
    const writes = tools.filter(function (t) { return t.is_write || t.kind === "write"; }).length;
    badges.appendChild(el("span", "badge", "写操作 " + writes));

    // 运行态探针：模型网关 + build 标记。显示「旧进程」说明服务没重启，跑的还是老代码
    const health = await fetch("/healthz").then(function (r) { return r.json(); }).catch(function () { return {}; });
    const gateway = health.gateway || "旧进程";
    const stale = gateway === "旧进程" || gateway === "MockLLMGateway";
    const modelBadge = el("span", "badge" + (stale ? " warn" : ""), "模型 " + (health.model || gateway));
    modelBadge.title = "网关 " + gateway + " · build " + (health.build || "unknown");
    badges.appendChild(modelBadge);
  } catch (e) {
    badges.appendChild(el("span", "badge", "接口未就绪"));
  }
}

function markOf(ev) {
  if (ev.skipped) return ["⊘", "m-idle"];
  if (ev.verdict === "pass" || ev.verdict === "ok") return ["✓", "m-ok"];
  if (ev.verdict === "handoff") return ["!", "m-handoff"];
  if (ev.verdict === "fail") return ["✕", "m-fail"];
  return ["·", "m-idle"];
}

function renderTrace(data) {
  document.getElementById("traceid").textContent = data.trace_id + " · 第 " + data.turn + " 轮";
  document.getElementById("session").textContent = data.session_id;
  meta.innerHTML = "";
  const cards = [
    ["意图 / 置信度", data.intent + " · " + (data.confidence || 0).toFixed(2)],
    ["路由", data.route],
    ["目标 Agent", (data.targets && data.targets.length ? data.targets.join(" → ") : "无")],
    ["转人工", data.need_handoff ? "是" : "否"],
    ["质量分", (data.quality_score || 0).toFixed(2)],
    ["降级", data.degraded ? "是" : "无"]
  ];
  cards.forEach(function (c) {
    const card = el("div", "card");
    card.appendChild(el("div", "k", c[0]));
    card.appendChild(el("div", "v", String(c[1])));
    meta.appendChild(card);
  });

  timeline.innerHTML = "";
  if (!data.events || !data.events.length) {
    timeline.appendChild(el("div", "empty", "本轮没有事件记录"));
    return;
  }
  data.events.forEach(function (ev) {
    const row = el("div", "ev");
    const m = markOf(ev);
    row.appendChild(el("span", "mark " + m[1], m[0]));
    row.appendChild(el("span", "aid p-" + String(ev.agent_id || "?")[0], ev.agent_id));
    row.appendChild(el("span", "evname", ev.event));
    row.appendChild(el("span", "cost", ev.cost_ms ? ev.cost_ms + "ms" : "-"));
    let detail = ev.detail || ev.verdict || "";
    if (ev.confidence !== null && ev.confidence !== undefined) {
      detail = "conf=" + ev.confidence.toFixed(2) + "  " + detail;
    }
    row.appendChild(el("span", "detail", detail));
    timeline.appendChild(row);
  });
  timeline.scrollTop = 0;
}

async function streamChat(payload, pending) {
  const resp = await fetch("/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!resp.ok) throw new Error("HTTP " + resp.status);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "", draft = "", final = null, sawDelta = false;
  setMsgText(pending, "");
  setStage(pending, "连接中…");

  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\\n\\n")) >= 0) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message", data = "";
      frame.split("\\n").forEach(function (line) {
        if (line.indexOf("event:") === 0) event = line.slice(6).trim();
        else if (line.indexOf("data:") === 0) data += line.slice(5).trim();
      });
      if (!data) continue;
      const obj = JSON.parse(data);
      if (event === "stage") {
        // 阶段进度：只在还没有正文时显示，正文一开始就撤掉
        if (!sawDelta) setStage(pending, obj.detail || obj.stage);
      } else if (event === "delta") {
        if (!sawDelta) {
          sawDelta = true;
          setStage(pending, "");
        }
        draft += obj.text || "";
        setMsgText(pending, draft);
      } else if (event === "done") {
        final = obj;
      } else if (event === "error") {
        throw new Error(obj.message || "流式请求失败");
      }
    }
  }
  if (!final) throw new Error("连接中断，未收到完整回复");
  setStage(pending, "");
  // 校验/回填可能改写文本：以 done 事件的最终回复为准覆盖草稿
  setMsgText(pending, final.response || draft);
  addFoot(pending, final);
  renderTrace(final);
}

form.addEventListener("submit", async function (e) {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  const user = uid.value.trim() || "u_1001";

  addMsg("user", text);
  input.value = "";
  sendBtn.disabled = true;
  const pending = addMsg("bot", "思考中…");

  try {
    await streamChat({ text: text, user_id: user, channel: "private" }, pending);
  } catch (err) {
    setMsgText(pending, "请求失败：" + err.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

boot();
</script>
</body>
</html>
"""
