"""End-to-end API test with fake embeddings + fake provider (no model, no Ollama)."""

import app.api.routes as routes_mod
import app.main as main_mod
from app.services.loaders import Document


def test_query_endpoint_grounds_and_cites(clean_index, fake_provider, monkeypatch):
    # neutralize startup/real-dataset seeding; we provide our own tiny corpus
    monkeypatch.setattr(main_mod, "ensure_seeded", lambda: None)
    monkeypatch.setattr(routes_mod, "ensure_seeded", lambda: None)

    clean_index.ingest_documents(
        [
            Document(text="The refund policy allows returns within 30 days of purchase.",
                     source="refund_policy.txt", doc_type="policy", split=False),
            Document(text="[customer] Alpha Trading LLC contact ahmed@alpha.ae",
                     source="customers.csv", doc_type="customer", record_id="CUST-001",
                     customer_id="CUST-001", company_name="Alpha Trading LLC"),
        ],
        rebuild=True,
    )

    from fastapi.testclient import TestClient

    with TestClient(main_mod.app) as client:
        r = client.post("/query", data={"question": "what is the refund policy?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["provider"] == "fake"
        assert body["model"] == "fake-model"
        assert body["answer"]                       # non-empty
        assert len(body["citations"]) >= 1          # fake reply cites [1]
        assert body["chunks_used"] >= 1
        assert body["total_ms"] is not None

        stats = client.get("/collection/stats").json()
        assert stats["total_chunks"] >= 2
        assert "policy" in stats["doc_types"]


def test_query_empty_question_is_422(clean_index, fake_provider, monkeypatch):
    monkeypatch.setattr(main_mod, "ensure_seeded", lambda: None)
    monkeypatch.setattr(routes_mod, "ensure_seeded", lambda: None)
    clean_index.ingest_documents(
        [Document(text="something", source="x.txt", doc_type="document", split=False)],
        rebuild=True,
    )
    from fastapi.testclient import TestClient

    with TestClient(main_mod.app) as client:
        r = client.post("/query", data={"question": "   "})
        assert r.status_code == 422
