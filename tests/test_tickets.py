"""人工审核台测试：工单落库、审批闸门、批准重放、幂等与驳回。

覆盖的是一条**会动钱**的链路，所以断言重点不是「接口返回 200」，而是：

1. 闸门在产生副作用**之前**拦住（没批之前不能有售后单）；
2. 批准后确实重放了写操作，且用的是**同一把幂等键**；
3. 重复批准不会退第二次钱；
4. 驳回不产生任何副作用。

关于订单选择：只有真正会写库的用例才消耗订单（退款后订单状态会变，
不可重复用）。仅建单/驳回的用例复用同一张单，因为它们不动数据。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cs_agent_system import seed as seed_mod
from cs_agent_system.api import ChatRequest, CustomerService, create_app
from cs_agent_system.config import settings
from cs_agent_system.registry import db, tickets

# 1299 元、已签收：超过默认阈值 500。仅用于建单与驳回（不写库）。
GATED_ORDER = "SO202610081115"
GATED_USER = "u_5008"

# 459 元、待发货：默认阈值下不会触发，测试里把阈值降到 300 来触发。
# 单独一张单，是因为批准会真的写库、之后不能再退。
APPROVE_ORDER = "SO202611030512"
APPROVE_USER = "u_2002"

# 189 元：低于默认阈值，正常走完，不产生审批单。
LOW_VALUE_ORDER = "SO202610280001"
LOW_VALUE_USER = "u_1001"


@pytest.fixture(scope="module")
def isolated_db(tmp_path_factory):
    """整模块共用一个临时库并灌种子数据。

    为什么是 module 级而不是每个用例一个库：`db` 的连接是**线程本地**缓存的，
    TestClient 的请求跑在独立线程里，换库后旧线程可能仍握着上一个库的连接，
    于是出现「写进了 A 库、却从 B 库读」的假失败。
    固定一个库文件就不会串；代价是用例共享数据，所以断言一律用前后差值。
    """
    original = db.settings.db_path
    db.settings.db_path = tmp_path_factory.mktemp("tickets") / "tickets.db"
    db._local.conn = None
    tickets._MEM.clear()
    seed_mod.seed()
    try:
        yield
    finally:
        db.settings.db_path = original
        db._local.conn = None
        tickets._MEM.clear()


@pytest.fixture()
def service(isolated_db) -> CustomerService:
    return CustomerService()


@pytest.fixture()
def client(service) -> TestClient:
    return TestClient(create_app())


def _aftersale_count(order_no: str) -> int:
    rows = db.query_all("SELECT aftersale_no FROM aftersales WHERE order_no = ?", (order_no,))
    return len(rows)


def _chat(client: TestClient, text: str, user_id: str) -> dict:
    resp = client.post("/chat", json={"text": text, "user_id": user_id})
    assert resp.status_code == 200
    return resp.json()


def test_gated_refund_creates_approval_ticket_without_side_effect(client) -> None:
    """超阈值退款：必须建审批单，且**此时不能有售后单**（钱还没出去）。"""
    before = _aftersale_count(GATED_ORDER)

    body = _chat(client, f"订单 {GATED_ORDER} 商品破损，我要退款", GATED_USER)

    assert body["need_handoff"] is True
    assert body["ticket_id"].startswith("TK-")

    ticket = tickets.get(body["ticket_id"])
    assert ticket is not None
    assert ticket.kind == "approval"
    assert ticket.status == tickets.PENDING
    assert ticket.order_no == GATED_ORDER
    # 赔付额被 max_compensation（200）封顶，但闸门按订单金额判定
    assert ticket.amount == 200.0
    assert ticket.tool == "aftersale.apply"
    assert ticket.idempotency_key, "审批单必须带上原幂等键，否则批准后无法安全重放"
    assert ticket.params.get("order_no") == GATED_ORDER
    # 闸门在副作用之前：还没批，就不能有售后单
    assert _aftersale_count(GATED_ORDER) == before


def test_low_value_refund_has_no_approval_ticket(client) -> None:
    """低于阈值：正常走完，不产生审批单。"""
    before = _aftersale_count(LOW_VALUE_ORDER)

    body = _chat(client, f"订单 {LOW_VALUE_ORDER} 尺码不合适，我要退款", LOW_VALUE_USER)

    assert body["need_handoff"] is False
    assert body["ticket_id"] == ""
    assert _aftersale_count(LOW_VALUE_ORDER) == before + 1


def test_approve_replays_write_once(client, monkeypatch) -> None:
    """批准后真正执行写操作；重复批准被 409 拦住，不会退两次钱。"""
    # 把阈值压到 300，让 459 元的单进入审批区间
    monkeypatch.setattr(settings, "write_approval_amount", 300.0)

    body = _chat(client, f"订单 {APPROVE_ORDER} 我要退款，原因是不想要了", APPROVE_USER)
    ticket_id = body["ticket_id"]
    assert ticket_id, body
    before = _aftersale_count(APPROVE_ORDER)

    approved = client.post(
        f"/admin/tickets/{ticket_id}/approve",
        json={"operator": "admin_zhang", "note": "核实无误"},
    )
    assert approved.status_code == 200
    result = approved.json()
    assert result["ok"] is True
    assert result["replayed"] is True
    assert result["ticket"]["status"] == tickets.APPROVED
    assert result["ticket"]["decided_by"] == "admin_zhang"
    # 钱真的出去了
    assert _aftersale_count(APPROVE_ORDER) == before + 1

    # 重复批准：不能再退一次
    again = client.post(f"/admin/tickets/{ticket_id}/approve", json={"operator": "admin_zhang"})
    assert again.status_code == 409
    assert _aftersale_count(APPROVE_ORDER) == before + 1


def test_reject_has_no_side_effect(client) -> None:
    """驳回：工单转 rejected，不产生售后单。"""
    body = _chat(client, f"订单 {GATED_ORDER} 商品破损，我要退款", GATED_USER)
    ticket_id = body["ticket_id"]
    before = _aftersale_count(GATED_ORDER)

    rejected = client.post(
        f"/admin/tickets/{ticket_id}/reject",
        json={"operator": "admin_li", "note": "金额与凭证不符"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["ticket"]["status"] == tickets.REJECTED
    assert _aftersale_count(GATED_ORDER) == before

    # 重复驳回同样要拦住
    assert client.post(
        f"/admin/tickets/{ticket_id}/reject", json={"operator": "admin_li"}
    ).status_code == 409


def test_ticket_list_filters_pending(client) -> None:
    """待办列表默认只返回 pending，已处理的不混进来。"""
    body = _chat(client, f"订单 {GATED_ORDER} 商品破损，我要退款", GATED_USER)
    ticket_id = body["ticket_id"]

    pending = client.get("/admin/tickets?status=pending").json()
    assert ticket_id in [t["ticket_id"] for t in pending["tickets"]]

    client.post(f"/admin/tickets/{ticket_id}/reject", json={"operator": "admin_li"})
    pending_after = client.get("/admin/tickets?status=pending").json()
    assert ticket_id not in [t["ticket_id"] for t in pending_after["tickets"]]

    # 按状态查已驳回的，应该能查到
    rejected = client.get("/admin/tickets?status=rejected").json()
    assert ticket_id in [t["ticket_id"] for t in rejected["tickets"]]


def test_handoff_ticket_is_persisted_and_linked_to_trace(service) -> None:
    """转人工工单必须落库，并且 trace 上能查到工单号。"""
    from cs_agent_system.registry import traces

    resp = service.chat(ChatRequest(text="你们这什么破服务，我要12315投诉！", user_id=LOW_VALUE_USER))
    assert resp.need_handoff is True
    assert resp.ticket_id.startswith("TK-")

    ticket = tickets.get(resp.ticket_id)
    assert ticket is not None
    assert ticket.kind == "handoff"
    assert ticket.reason
    # 从对话能反查到工单
    record = traces.get(resp.trace_id)
    assert record is not None
    assert record["ticket_id"] == resp.ticket_id


def test_unknown_ticket_returns_404(client) -> None:
    assert client.get("/admin/tickets/TK-NOT-EXIST").status_code == 404
    assert client.post(
        "/admin/tickets/TK-NOT-EXIST/approve", json={"operator": "x"}
    ).status_code == 404


def test_admin_page_is_served_and_not_cached(client) -> None:
    """审核台页面可访问，且不能被浏览器缓存成旧版本。"""
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert "人工审核台" in resp.text
    assert "/admin/tickets" in resp.text
    assert "no-store" in resp.headers.get("cache-control", "")


def test_console_has_entry_to_admin(client) -> None:
    """控制台上要有跳转到审核台的入口，否则页面藏起来等于没有。"""
    html = client.get("/console").text
    assert 'href="/admin"' in html
