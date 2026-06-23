import io
import json
import zipfile

from app.services.loaders import (
    _csv_doc_type,
    _text_doc_type,
    load_csv,
    load_email_threads,
    load_text,
    load_zip,
)

CUSTOMERS_CSV = (
    "customer_id,company_name,industry,current_status\n"
    "CUST-001,Alpha Trading LLC,Retail,Lead\n"
    "CUST-002,Beta Foods,Food,Customer\n"
)

TICKETS_CSV = (
    "ticket_id,customer_id,company_name,priority,status\n"
    "TKT-001,CUST-001,Alpha Trading LLC,Critical,Open\n"
)


def test_csv_doc_type_mapping():
    assert _csv_doc_type("customers.csv") == "customer"
    assert _csv_doc_type("leads.csv") == "lead"
    assert _csv_doc_type("sales_notes.csv") == "sales_note"
    assert _csv_doc_type("support_tickets_80.csv") == "ticket"


def test_text_doc_type_from_path():
    assert _text_doc_type("documents/policies/support_sla.txt") == "policy"
    assert _text_doc_type("documents/faq/pricing_faq.txt") == "faq"
    assert _text_doc_type("documents/services/crm_platform.txt") == "service"
    assert _text_doc_type("meeting_notes/meeting_note_01.txt") == "meeting_note"


def test_load_customers_csv_sets_join_keys():
    docs = load_csv(CUSTOMERS_CSV, "customers.csv")
    assert len(docs) == 2
    d0 = docs[0]
    assert d0.doc_type == "customer"
    assert d0.record_id == "CUST-001"
    # for the customers file, the record id IS the customer id (join key)
    assert d0.customer_id == "CUST-001"
    assert d0.company_name == "Alpha Trading LLC"
    assert d0.split is False
    assert "Company Name: Alpha Trading LLC" in d0.text


def test_load_tickets_csv_links_to_customer():
    docs = load_csv(TICKETS_CSV, "support_tickets.csv")
    assert docs[0].doc_type == "ticket"
    assert docs[0].customer_id == "CUST-001"
    assert docs[0].record_id == "TKT-001"


def test_load_email_threads():
    data = json.dumps([
        {
            "thread_id": "THREAD-001",
            "customer_id": "CUST-008",
            "company_name": "Sunrise Education Group",
            "subject": "Onboarding",
            "messages": [
                {"from_name": "Huda", "to_name": "Khalid", "timestamp": "2025-04-27", "body": "Hi"},
            ],
        }
    ])
    docs = load_email_threads(data, "email_threads.json")
    assert len(docs) == 1
    assert docs[0].doc_type == "email"
    assert docs[0].customer_id == "CUST-008"
    assert docs[0].company_name == "Sunrise Education Group"
    assert docs[0].split is True


def test_load_text_blank_is_skipped():
    assert load_text("   ", "x.txt") == []
    assert len(load_text("real content", "documents/faq/x.txt")) == 1


def test_load_zip_dispatches_inner_files():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("documents/policies/support_sla.txt", "SLA content here")
        zf.writestr("documents/faq/general_faq.txt", "FAQ content here")
    docs = load_zip(buf.getvalue())
    types = sorted(d.doc_type for d in docs)
    assert types == ["faq", "policy"]
    assert all(d.source.startswith("documents/") for d in docs)
