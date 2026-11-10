"""检索与知识层测试。"""

from __future__ import annotations

from cs_agent_system.knowledge.retriever import build_default_retriever


def test_policy_retrieval_returns_evidence_with_version() -> None:
    r = build_default_retriever()
    evidence = r.search("退款多久到账", top_k=3)
    assert evidence
    assert any(e.layer == "policy" for e in evidence)
    assert any(e.version for e in evidence)


def test_expired_evidence_is_downweighted() -> None:
    from datetime import date, timedelta

    from cs_agent_system.knowledge.retriever import Document, HybridRetriever

    doc = Document(
        doc_id="old",
        title="旧政策",
        content="退款到账时效为 7 个工作日",
        layer="policy",
        effective_to=date.today() - timedelta(days=1),
    )
    r = HybridRetriever([doc])
    evidence = r.search("退款到账时效")
    assert evidence
    assert "已过期" in evidence[0].source


def test_hybrid_retrieval_prefers_longer_keyword_match() -> None:
    r = build_default_retriever()
    top = r.search("优惠券能叠加几张", top_k=1)
    assert top
    assert "叠加" in top[0].snippet
