import app.services.retriever as r
from app.services.loaders import Document


def test_company_core_strips_suffixes():
    assert r._company_core("Alpha Trading LLC") == "alpha"
    assert r._company_core("Beta Foods") == "beta foods"
    assert r._company_core("NextEra Consultants Inc") == "nextera consultants"


def test_detect_entity_by_company_name(monkeypatch):
    companies = {"alpha": ("Alpha Trading LLC", "CUST-001")}
    id_to_company = {"CUST-001": "Alpha Trading LLC"}
    monkeypatch.setattr(r, "_entity_cache", (companies, id_to_company))
    cid, company = r._detect_entity("summarize alpha before my meeting")
    assert company == "Alpha Trading LLC"
    assert cid == "CUST-001"


def test_detect_entity_by_customer_id(monkeypatch):
    monkeypatch.setattr(r, "_entity_cache", ({}, {"CUST-042": "Gamma Co"}))
    cid, company = r._detect_entity("show me CUST-042 tickets")
    assert cid == "CUST-042"
    assert company == "Gamma Co"


def test_detect_entity_none_when_unknown(monkeypatch):
    monkeypatch.setattr(r, "_entity_cache", ({"alpha": ("Alpha Trading LLC", "CUST-001")}, {}))
    cid, company = r._detect_entity("what is the refund policy?")
    assert cid is None and company is None


def test_entity_retrieve_forces_customer_records(clean_index):
    docs = [
        Document(text="[customer] Alpha", source="customers.csv", doc_type="customer",
                 record_id="CUST-001", customer_id="CUST-001", company_name="Alpha Trading LLC"),
        Document(text="[ticket] login broken", source="tickets.csv", doc_type="ticket",
                 record_id="TKT-9", customer_id="CUST-001", company_name="Alpha Trading LLC"),
        Document(text="[policy] refund policy text", source="refund_policy.txt",
                 doc_type="policy", split=False),
    ]
    clean_index.ingest_documents(docs, rebuild=True)

    chunks, entity = r.retrieve("summarize Alpha Trading LLC")
    assert entity == "Alpha Trading LLC"
    # both Alpha records pulled via the entity gather, across doc_types
    vias = {c["via"] for c in chunks if c["company_name"] == "Alpha Trading LLC"}
    assert "entity" in vias
    pulled_types = {c["doc_type"] for c in chunks if c["customer_id"] == "CUST-001"}
    assert {"customer", "ticket"} <= pulled_types


def test_hybrid_retrieve_finds_lexical_match(clean_index):
    docs = [
        Document(text="The refund policy allows returns within 30 days.",
                 source="refund_policy.txt", doc_type="policy", split=False),
        Document(text="The support SLA defines response times.",
                 source="support_sla.txt", doc_type="policy", split=False),
    ]
    clean_index.ingest_documents(docs, rebuild=True)
    chunks, _ = r.retrieve("what is the refund policy")
    assert any("refund" in c["source"] for c in chunks)
