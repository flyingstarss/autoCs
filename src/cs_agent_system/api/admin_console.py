"""人工审核台（零依赖单页）。

挂在 `/admin`，与运行控制台（`/console`）共用同一套暗色视觉，但不是同一个页面：
控制台看的是「机器人怎么想」，审核台看的是「哪张单子等我处理」。

为什么不做成独立前端工程
------------------------
这里只有「列待办 → 看详情 → 批准/驳回」三个动作，引入 React + 打包链
会让部署多一个构建步骤，收益却近乎为零。单页直出、零依赖，改完刷新即可。

安全边界
--------
页面本身不做任何权限判断——它只是一层壳，真正拦截在接口侧：
非 pending 的工单再次审批一律 409，重复点批准不会退两次钱。
"""

ADMIN_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>人工审核台 · 客服系统</title>
<style>
  :root{
    --bg:#0e1117; --panel:#161b22; --line:#262d38; --text:#e6edf3;
    --muted:#8b949e; --accent:#2f81f7; --ok:#3fb950; --warn:#d29922;
    --bad:#f85149; --mono:ui-monospace,Consolas,"Courier New",monospace;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0; background:var(--bg); color:var(--text);
    font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
    display:flex; flex-direction:column;
  }
  header{
    display:flex; align-items:center; justify-content:space-between; gap:12px;
    padding:12px 20px; border-bottom:1px solid var(--line); background:var(--panel);
  }
  .brand{display:flex; align-items:center; gap:10px; font-weight:600}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--warn)}
  .nav{display:flex; gap:14px; align-items:center; font-size:13px}
  .nav a{color:var(--muted); text-decoration:none}
  .nav a:hover{color:var(--accent)}
  .badges{display:flex; gap:8px; flex-wrap:wrap}
  .badge{
    font-size:12px; color:var(--muted); border:1px solid var(--line);
    padding:2px 8px; border-radius:999px; background:#0d1117;
  }
  .badge.warn{color:var(--warn); border-color:#4a3a12}
  .badge.ok{color:var(--ok); border-color:#1c3a24}
  .badge.bad{color:var(--bad); border-color:#4a2020}
  .toolbar{
    display:flex; gap:8px; align-items:center; flex-wrap:wrap;
    padding:10px 20px; border-bottom:1px solid var(--line); background:#12171f;
  }
  .tab{
    font-size:13px; padding:5px 12px; border-radius:999px; cursor:pointer;
    background:#0d1117; color:var(--muted); border:1px solid var(--line);
  }
  .tab.active{background:var(--accent); color:#fff; border-color:var(--accent)}
  .spacer{flex:1}
  button{
    border:none; border-radius:8px; padding:7px 16px; font:inherit; font-weight:600;
    background:var(--accent); color:#fff; cursor:pointer;
  }
  button.ghost{background:#0d1117; color:var(--text); border:1px solid var(--line)}
  button.ok{background:#238636}
  button.bad{background:#8b2c28}
  button:disabled{opacity:.45; cursor:not-allowed}
  main{flex:1; display:grid; grid-template-columns:minmax(320px,.9fr) minmax(420px,1.1fr); gap:16px; padding:16px; min-height:0}
  section{background:var(--panel); border:1px solid var(--line); border-radius:12px; display:flex; flex-direction:column; min-height:0; overflow:hidden}
  .panel-title{
    padding:10px 14px; font-size:12px; color:var(--muted); letter-spacing:.08em;
    border-bottom:1px solid var(--line); display:flex; justify-content:space-between;
  }
  .list{flex:1; overflow-y:auto}
  .item{padding:12px 14px; border-bottom:1px solid #1e242e; cursor:pointer; transition:.12s}
  .item:hover{background:#1a2029}
  .item.active{background:#1c2534; box-shadow:inset 3px 0 0 var(--accent)}
  .item .row1{display:flex; align-items:center; gap:8px; justify-content:space-between}
  .item .id{font-family:var(--mono); font-size:12px; color:var(--text)}
  .item .sum{font-size:12.5px; color:#c9d1d9; margin-top:5px; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden}
  .item .meta{font-size:11px; color:var(--muted); margin-top:5px; display:flex; gap:10px; flex-wrap:wrap}
  .pill{font-size:11px; padding:1px 7px; border-radius:999px; border:1px solid var(--line); color:var(--muted)}
  .pill.approval{color:#e0af68; border-color:#4a3a12}
  .pill.handoff{color:#7aa2f7; border-color:#1d2f52}
  .pill.pending{color:var(--warn); border-color:#4a3a12}
  .pill.approved{color:var(--ok); border-color:#1c3a24}
  .pill.rejected{color:var(--bad); border-color:#4a2020}
  .pill.uncertain{color:#bb9af7; border-color:#3a2a52}
  .detail{flex:1; overflow-y:auto; padding:16px}
  .detail h2{margin:0 0 4px; font-size:16px; font-family:var(--mono)}
  .kv{display:grid; grid-template-columns:96px 1fr; gap:8px 12px; margin:14px 0; font-size:13px}
  .kv .k{color:var(--muted)}
  .kv .v{word-break:break-all}
  .block{margin:14px 0}
  .block .label{font-size:11px; color:var(--muted); letter-spacing:.06em; margin-bottom:6px}
  .quote{
    background:#0d1117; border:1px solid var(--line); border-radius:8px;
    padding:10px 12px; font-size:13px; color:#c9d1d9; white-space:pre-wrap; word-break:break-word;
  }
  pre{
    background:#0d1117; border:1px solid var(--line); border-radius:8px; margin:0;
    padding:10px 12px; overflow-x:auto; font-family:var(--mono); font-size:12px; color:#c9d1d9;
  }
  .decide{border-top:1px solid var(--line); padding-top:14px; margin-top:16px; display:flex; flex-direction:column; gap:10px}
  .decide input, .decide textarea{
    background:#0d1117; border:1px solid var(--line); color:var(--text);
    border-radius:8px; padding:9px 11px; font:inherit; outline:none; width:100%;
  }
  .decide textarea{resize:vertical; min-height:64px}
  .decide input:focus, .decide textarea:focus{border-color:var(--accent)}
  .actions{display:flex; gap:10px; justify-content:flex-end}
  .banner{border-radius:8px; padding:10px 12px; font-size:13px; white-space:pre-wrap}
  .banner.ok{background:#0f2a17; border:1px solid #1c3a24; color:#7ee2a0}
  .banner.bad{background:#2a1214; border:1px solid #4a2020; color:#ffb4ae}
  .banner.warn{background:#2a2311; border:1px solid #4a3a12; color:#e3c07b}
  .empty{color:var(--muted); font-size:12.5px; padding:24px 16px; text-align:center}
  .hint{font-size:11.5px; color:var(--muted)}
  @media (max-width:900px){ main{grid-template-columns:1fr; grid-template-rows:1fr 1.2fr} }
</style>
</head>
<body>
<header>
  <div class="brand"><span class="dot"></span>人工审核台 · 待办与审批</div>
  <div class="nav">
    <div class="badges" id="badges"></div>
    <a href="/console">← 运行控制台</a>
  </div>
</header>

<div class="toolbar">
  <div class="tab active" data-status="pending">待处理</div>
  <div class="tab" data-status="approved">已批准</div>
  <div class="tab" data-status="rejected">已驳回</div>
  <div class="tab" data-status="uncertain">状态未知</div>
  <div class="tab" data-status="all">全部</div>
  <div class="spacer"></div>
  <div class="tab active" data-kind="">全部类型</div>
  <div class="tab" data-kind="approval">待审批</div>
  <div class="tab" data-kind="handoff">转人工</div>
  <button class="ghost" id="refresh">刷新</button>
</div>

<main>
  <section>
    <div class="panel-title"><span>工单列表</span><span id="listcount">-</span></div>
    <div class="list" id="list"></div>
  </section>
  <section>
    <div class="panel-title"><span>工单详情</span><span id="detailid">-</span></div>
    <div class="detail" id="detail">
      <div class="empty">从左侧选择一张工单查看详情并处理。</div>
    </div>
  </section>
</main>

<script>
var state = { status: "pending", kind: "", tickets: [], selected: null, stats: {} };
var listEl = document.getElementById("list");
var detailEl = document.getElementById("detail");
var listCountEl = document.getElementById("listcount");
var detailIdEl = document.getElementById("detailid");
var badgesEl = document.getElementById("badges");

function el(tag, cls, text) {
  var node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function fmtTime(ts) {
  if (!ts) return "-";
  var d = new Date(ts * 1000);
  return d.toLocaleString("zh-CN", { hour12: false });
}

function fmtMoney(n) {
  if (n === null || n === undefined) return "-";
  return Number(n).toFixed(2);
}

var STATUS_TEXT = { pending: "待处理", approved: "已批准", rejected: "已驳回", uncertain: "状态未知" };
var KIND_TEXT = { approval: "待审批", handoff: "转人工" };

function pill(cls, text) {
  return el("span", "pill " + cls, text);
}

/* ---------- 列表 ---------- */

function renderList() {
  listEl.innerHTML = "";
  if (!state.tickets.length) {
    listEl.appendChild(el("div", "empty", "当前筛选条件下没有工单。"));
    listCountEl.textContent = "0 条";
    return;
  }
  listCountEl.textContent = state.tickets.length + " 条";
  state.tickets.forEach(function (t) {
    var item = el("div", "item" + (t.ticket_id === state.selected ? " active" : ""));
    var row1 = el("div", "row1");
    row1.appendChild(el("span", "id", t.ticket_id));
    var tags = el("span", "");
    tags.appendChild(pill(t.kind, KIND_TEXT[t.kind] || t.kind));
    row1.appendChild(tags);
    item.appendChild(row1);

    item.appendChild(el("div", "sum", t.summary || t.reason || "（无摘要）"));

    var meta = el("div", "meta");
    if (t.order_no) meta.appendChild(el("span", "", "订单 " + t.order_no));
    if (t.amount) meta.appendChild(el("span", "", "¥" + fmtMoney(t.amount)));
    if (t.user_id) meta.appendChild(el("span", "", t.user_id));
    meta.appendChild(el("span", "", fmtTime(t.created_at)));
    item.appendChild(meta);

    item.onclick = function () { select(t.ticket_id); };
    listEl.appendChild(item);
  });
}

/* ---------- 详情 ---------- */

function kv(k, v) {
  var row = el("div", "kv");
  row.appendChild(el("div", "k", k));
  row.appendChild(el("div", "v", v === "" || v === null || v === undefined ? "-" : v));
  return row;
}

function renderDetail(t) {
  detailEl.innerHTML = "";
  detailIdEl.textContent = t.ticket_id;

  var head = el("div", "row1");
  head.appendChild(el("h2", "", t.ticket_id));
  var pills = el("div", "");
  pills.appendChild(pill(t.kind, KIND_TEXT[t.kind] || t.kind));
  pills.appendChild(pill(t.status, STATUS_TEXT[t.status] || t.status));
  head.appendChild(pills);
  detailEl.appendChild(head);

  var kvBox = el("div", "kv");
  [["类型", KIND_TEXT[t.kind] || t.kind],
   ["状态", STATUS_TEXT[t.status] || t.status],
   ["用户", t.user_id],
   ["会话", t.session_id],
   ["意图", t.intent],
   ["订单号", t.order_no],
   ["金额", t.amount ? "¥" + fmtMoney(t.amount) : "-"],
   ["创建时间", fmtTime(t.created_at)]].forEach(function (pair) {
    kvBox.appendChild(el("div", "k", pair[0]));
    kvBox.appendChild(el("div", "v", pair[1] === "" || pair[1] === null || pair[1] === undefined ? "-" : String(pair[1])));
  });
  detailEl.appendChild(kvBox);

  if (t.summary) {
    var sumBlock = el("div", "block");
    sumBlock.appendChild(el("div", "label", "摘要"));
    sumBlock.appendChild(el("div", "quote", t.summary));
    detailEl.appendChild(sumBlock);
  }
  if (t.reason) {
    var reasonBlock = el("div", "block");
    reasonBlock.appendChild(el("div", "label", "转人工 / 待审批原因"));
    reasonBlock.appendChild(el("div", "quote", t.reason));
    detailEl.appendChild(reasonBlock);
  }

  if (t.kind === "approval") {
    var opBlock = el("div", "block");
    opBlock.appendChild(el("div", "label", "待执行的写操作"));
    var payload = { tool: t.tool, params: t.params || {} };
    var pre = el("pre", "", JSON.stringify(payload, null, 2));
    opBlock.appendChild(pre);
    detailEl.appendChild(opBlock);

    if (t.idempotency_key) {
      var keyBlock = el("div", "block");
      keyBlock.appendChild(el("div", "label", "幂等键（批准后据此重放，保证只退一次）"));
      keyBlock.appendChild(el("div", "quote", t.idempotency_key));
      detailEl.appendChild(keyBlock);
    }
  }

  if (t.trace_id) {
    var traceBlock = el("div", "block");
    traceBlock.appendChild(el("div", "label", "来源对话 trace"));
    traceBlock.appendChild(el("div", "quote", t.trace_id));
    detailEl.appendChild(traceBlock);
  }

  /* 已处理：展示决策留痕 */
  if (t.status !== "pending") {
    var resultBox = el("div", "block");
    resultBox.appendChild(el("div", "label", "处理结果"));
    var banner = el("div", "banner " + (t.status === "approved" ? "ok" : (t.status === "rejected" ? "bad" : "warn")));
    var lines = [];
    lines.push("处理人：" + (t.decided_by || "-"));
    lines.push("处理时间：" + fmtTime(t.decided_at));
    if (t.decision_note) lines.push("备注：" + t.decision_note);
    banner.textContent = lines.join("\\n");
    resultBox.appendChild(banner);
    if (t.result && Object.keys(t.result).length) {
      var rBlock = el("div", "block");
      rBlock.appendChild(el("div", "label", "接口返回"));
      rBlock.appendChild(el("pre", "", JSON.stringify(t.result, null, 2)));
      resultBox.appendChild(rBlock);
    }
    detailEl.appendChild(resultBox);
    return;
  }

  /* 待处理：审批操作区 */
  var decide = el("div", "decide");
  var opInput = el("input");
  opInput.placeholder = "操作人（必填，例如 admin_zhang）";
  opInput.value = localStorage.getItem("operator") || "";
  var noteInput = document.createElement("textarea");
  noteInput.placeholder = "审批备注（可选）";
  var actions = el("div", "actions");
  var hint = el("div", "hint");

  if (t.kind === "approval") {
    hint.textContent = "批准会立刻执行退款写操作，且带原幂等键重放，重复点击不会重复退款。";
    var approveBtn = el("button", "ok", "批准并执行");
    approveBtn.onclick = function () {
      var op = readOperator(opInput);
      if (!op) return;
      if (!window.confirm("确认批准 " + t.ticket_id + "？\\n订单 " + (t.order_no || "-") +
                          "，退款 ¥" + fmtMoney(t.amount) + "，批准后将立即执行。")) return;
      decideTicket(t.ticket_id, "approve", op, noteInput.value);
    };
    actions.appendChild(approveBtn);
  } else {
    hint.textContent = "接管单批准仅表示坐席已处理，不会产生任何写操作。";
    var takeoverBtn = el("button", "ok", "标记已处理");
    takeoverBtn.onclick = function () {
      var op = readOperator(opInput);
      if (!op) return;
      decideTicket(t.ticket_id, "approve", op, noteInput.value);
    };
    actions.appendChild(takeoverBtn);
  }

  var rejectBtn = el("button", "bad", "驳回");
  rejectBtn.onclick = function () {
    var op = readOperator(opInput);
    if (!op) return;
    if (!window.confirm("确认驳回 " + t.ticket_id + "？驳回不会产生任何写操作。")) return;
    decideTicket(t.ticket_id, "reject", op, noteInput.value);
  };
  actions.appendChild(rejectBtn);

  decide.appendChild(opInput);
  decide.appendChild(noteInput);
  decide.appendChild(actions);
  decide.appendChild(hint);
  detailEl.appendChild(decide);
}

function readOperator(input) {
  var op = input.value.trim();
  if (!op) {
    input.focus();
    input.style.borderColor = "#f85149";
    return "";
  }
  localStorage.setItem("operator", op);
  return op;
}

/* ---------- 动作 ---------- */

function showMessage(kind, text) {
  detailEl.innerHTML = "";
  detailIdEl.textContent = "-";
  var banner = el("div", "banner " + kind);
  banner.textContent = text;
  detailEl.appendChild(banner);
}

async function decideTicket(id, action, operator, note) {
  try {
    var resp = await fetch("/admin/tickets/" + encodeURIComponent(id) + "/" + action, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operator: operator, note: note || "" })
    });
    var body = await resp.json();
    if (!resp.ok) {
      showMessage("bad", "操作失败：" + (body.detail || ("HTTP " + resp.status)));
      return;
    }
    if (action === "approve" && body.status_unknown) {
      showMessage("warn",
        "写操作重放超时，状态未知。\\n工单已标记为「状态未知」，请到 OMS 侧核对这笔退款是否已经生成，" +
        "切勿直接再次批准。\\n\\n错误：" + (body.error || "-"));
    } else if (action === "approve" && body.ok === false) {
      showMessage("bad", "重放失败，未产生退款：\\n" + (body.error || "-"));
    } else {
      state.selected = id;
      await reload();
      await select(id);
    }
    await loadStats();
  } catch (err) {
    showMessage("bad", "网络异常：" + err.message);
  }
}

async function select(id) {
  state.selected = id;
  renderList();
  try {
    var resp = await fetch("/admin/tickets/" + encodeURIComponent(id));
    if (!resp.ok) {
      showMessage("bad", "工单不存在或已被清理。");
      return;
    }
    renderDetail(await resp.json());
  } catch (err) {
    showMessage("bad", "加载详情失败：" + err.message);
  }
}

/* ---------- 加载 ---------- */

async function loadList() {
  var qs = "?limit=200";
  if (state.status) qs += "&status=" + encodeURIComponent(state.status);
  if (state.kind) qs += "&kind=" + encodeURIComponent(state.kind);
  var resp = await fetch("/admin/tickets" + qs);
  var body = await resp.json();
  state.tickets = body.tickets || [];
  renderList();
  if (!body.db_available) {
    listEl.insertBefore(el("div", "banner warn", "数据库不可用：工单仅存在于进程内存，重启即丢失。"), listEl.firstChild);
  }
}

async function loadStats() {
  try {
    var resp = await fetch("/admin/tickets?status=all&limit=200");
    var body = await resp.json();
    var counts = { pending: 0, approval: 0, handoff: 0, uncertain: 0 };
    (body.tickets || []).forEach(function (t) {
      if (t.status === "pending") counts.pending++;
      if (t.status === "pending" && t.kind === "approval") counts.approval++;
      if (t.status === "pending" && t.kind === "handoff") counts.handoff++;
      if (t.status === "uncertain") counts.uncertain++;
    });
    badgesEl.innerHTML = "";
    badgesEl.appendChild(el("span", "badge warn", "待处理 " + counts.pending));
    badgesEl.appendChild(el("span", "badge", "待审批 " + counts.approval));
    badgesEl.appendChild(el("span", "badge", "转人工 " + counts.handoff));
    if (counts.uncertain) {
      var b = el("span", "badge bad", "状态未知 " + counts.uncertain);
      b.title = "重放超时，需到 OMS 侧人工核对";
      badgesEl.appendChild(b);
    }
  } catch (e) {
    badgesEl.innerHTML = "";
    badgesEl.appendChild(el("span", "badge bad", "接口未就绪"));
  }
}

async function reload() {
  await loadList();
  await loadStats();
}

/* ---------- 交互绑定 ---------- */

document.querySelectorAll(".tab[data-status]").forEach(function (tab) {
  tab.onclick = function () {
    document.querySelectorAll(".tab[data-status]").forEach(function (n) { n.classList.remove("active"); });
    tab.classList.add("active");
    state.status = tab.dataset.status;
    state.selected = null;
    loadList();
  };
});

document.querySelectorAll(".tab[data-kind]").forEach(function (tab) {
  tab.onclick = function () {
    document.querySelectorAll(".tab[data-kind]").forEach(function (n) { n.classList.remove("active"); });
    tab.classList.add("active");
    state.kind = tab.dataset.kind;
    state.selected = null;
    loadList();
  };
});

document.getElementById("refresh").onclick = reload;

reload();
// 轮询：待办是异步产生的（用户在别处聊天），页面不刷新也要能看到新单
setInterval(function () {
  if (document.visibilityState === "visible") reload();
}, 8000);
</script>
</body>
</html>
"""
