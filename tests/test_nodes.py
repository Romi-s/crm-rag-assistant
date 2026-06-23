from app.agent.nodes import _build_context, format_response, validate_input
from app.agent.state import RetrievedChunk


def _chunk(i, text, doc_type="ticket", source="t.csv", rid="TKT-1", company="Alpha"):
    return RetrievedChunk(
        id=str(i), text=text, source=source, doc_type=doc_type, record_id=rid,
        customer_id="CUST-1", company_name=company, chunk_index=0, score=1.0, via="hybrid",
    )


def test_validate_input_rejects_empty():
    assert validate_input({"question": "   "})["error"]
    assert validate_input({"question": "x" * 2001})["error"]
    assert validate_input({"question": "ok"}) == {}


def test_build_context_numbers_and_labels_chunks():
    ctx = _build_context([_chunk(1, "first"), _chunk(2, "second")])
    assert "[1]" in ctx and "[2]" in ctx
    assert "type=ticket" in ctx and "source=t.csv" in ctx
    assert "first" in ctx and "second" in ctx


def test_format_response_extracts_only_cited_sources():
    state = {
        "error": None,
        "answer": "The ticket is critical [1]. Unrelated note [3].",
        "retrieved_chunks": [
            _chunk(1, "ticket one", rid="TKT-1"),
            _chunk(2, "ticket two", rid="TKT-2"),
            _chunk(3, "ticket three", rid="TKT-3"),
        ],
    }
    out = format_response(state)
    cited_ids = {c["record_id"] for c in out["citations"]}
    assert cited_ids == {"TKT-1", "TKT-3"}   # [2] not cited -> excluded


def test_format_response_no_citation_when_none_referenced():
    state = {
        "error": None,
        "answer": "I cannot answer this from the available data.",
        "retrieved_chunks": [_chunk(1, "x")],
    }
    assert format_response(state)["citations"] == []


def test_nodes_short_circuit_on_error():
    assert _build_context  # imported
    err = {"error": "boom"}
    assert format_response(err) == {}
