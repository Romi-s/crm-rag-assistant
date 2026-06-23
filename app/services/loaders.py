"""Turn the raw dataset into a flat list of `Document`s ready for indexing.

The dataset mixes structured and unstructured data:

  crm_records/customers.csv        structured  -> one record per row
  crm_records/leads.csv            structured
  sales/sales_notes.csv            structured  (linked to a customer_id)
  tickets/support_tickets_80.csv   structured  (linked to a customer_id)
  emails/email_threads.json        semi-struct -> one thread, many messages
  documents/**/*.txt  (in zips)    unstructured (faq / policy / service / proposal)
  sales/meeting_notes/*.txt (zip)  unstructured

Everything is normalised to a `Document` carrying the metadata that powers both
**citations** and **entity-aware retrieval**: `doc_type`, `customer_id`,
`company_name`, `record_id`. `customer_id` / `company_name` are the join keys that
let a single question pull a customer's CRM row + sales notes + tickets + emails.

Structured records are kept whole (`split=False`) so a customer/ticket stays a
single coherent, citable unit; prose (`split=True`) is chunked downstream.
"""

import csv
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.services.text_extractor import extract_text_from_pdf

# Files we never index (the assignment brief itself, VCS internals).
_SKIP_NAME_SUBSTRINGS = ("hiring assignment",)


@dataclass
class Document:
    text: str
    source: str                  # shown in citations, e.g. "customers.csv"
    doc_type: str                # customer | lead | sales_note | ticket | email | ...
    record_id: str = ""          # CUST-001 / TKT-001 / THREAD-001 / page no.
    customer_id: str = ""        # join key across sources ("" when not applicable)
    company_name: str = ""
    split: bool = False          # True -> chunk it (prose); False -> keep whole

    def metadata(self) -> dict:
        # Chroma needs scalar, non-None metadata values.
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "record_id": self.record_id or "",
            "customer_id": self.customer_id or "",
            "company_name": self.company_name or "",
        }


# --------------------------------------------------------------------------- #
# doc_type / id helpers
# --------------------------------------------------------------------------- #
def _csv_doc_type(filename: str) -> str:
    name = filename.lower()
    if "customer" in name:
        return "customer"
    if "lead" in name:
        return "lead"
    if "sales_note" in name or "sales-note" in name or "salesnote" in name:
        return "sales_note"
    if "ticket" in name:
        return "ticket"
    return "record"


def _text_doc_type(path: str) -> str:
    p = path.lower()
    if "faq" in p:
        return "faq"
    if "policy" in p or "policies" in p:
        return "policy"
    if "service" in p:
        return "service"
    if "proposal" in p or "feature" in p or "integration_guide" in p or "onboarding" in p:
        return "proposal"
    if "meeting" in p:
        return "meeting_note"
    return "document"


def _humanize(field_name: str) -> str:
    return field_name.replace("_", " ").strip().title()


def _format_record(row: dict, doc_type: str, title: str) -> str:
    """Render a structured row as a labelled block so the embedder, BM25, and the
    LLM all see explicit field context (not a bare CSV line)."""
    lines = [f"[{doc_type}] {title}".strip()]
    for k, v in row.items():
        v = ("" if v is None else str(v)).strip()
        if v and v != "-":
            lines.append(f"{_humanize(k)}: {v}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# per-format loaders
# --------------------------------------------------------------------------- #
def load_csv(text: str, filename: str) -> List[Document]:
    doc_type = _csv_doc_type(filename)
    reader = csv.DictReader(io.StringIO(text))
    docs: List[Document] = []

    # pick the record-id column: prefer the type-specific *_id, else first *_id
    id_cols = [c for c in (reader.fieldnames or []) if c and c.lower().endswith("_id")]
    primary_id = None
    for c in id_cols:
        if doc_type.split("_")[0] in c.lower():
            primary_id = c
            break
    if primary_id is None and id_cols:
        primary_id = id_cols[0]

    for row in reader:
        row = {k: (v or "") for k, v in row.items() if k is not None}
        company = row.get("company_name", "")
        customer_id = row.get("customer_id", "")
        record_id = row.get(primary_id, "") if primary_id else ""
        # For the customers file, the record id *is* the customer id.
        if doc_type == "customer" and not customer_id:
            customer_id = record_id
        title = company or record_id or doc_type
        docs.append(
            Document(
                text=_format_record(row, doc_type, title),
                source=filename,
                doc_type=doc_type,
                record_id=record_id,
                customer_id=customer_id,
                company_name=company,
                split=False,
            )
        )
    return docs


def load_email_threads(text: str, filename: str) -> List[Document]:
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    docs: List[Document] = []
    for thread in data:
        subject = thread.get("subject", "")
        company = thread.get("company_name", "")
        customer_id = thread.get("customer_id", "")
        thread_id = thread.get("thread_id", "")
        parts = [f"[email] Thread: {subject}".strip()]
        if company:
            parts.append(f"Company: {company}")
        for m in thread.get("messages", []):
            ts = m.get("timestamp", "")
            frm = m.get("from_name") or m.get("from_email", "")
            to = m.get("to_name") or m.get("to_email", "")
            body = (m.get("body") or "").strip()
            parts.append(f"\nFrom {frm} to {to} ({ts}):\n{body}")
        docs.append(
            Document(
                text="\n".join(parts),
                source=filename,
                doc_type="email",
                record_id=thread_id,
                customer_id=customer_id,
                company_name=company,
                split=True,   # threads can be long; chunk while keeping metadata
            )
        )
    return docs


def load_text(text: str, source: str) -> List[Document]:
    if not text.strip():
        return []
    return [
        Document(
            text=text,
            source=source,
            doc_type=_text_doc_type(source),
            split=True,
        )
    ]


def load_pdf(pdf_bytes: bytes, source: str, max_pages: Optional[int] = None) -> List[Document]:
    pages = extract_text_from_pdf(pdf_bytes, max_pages=max_pages or settings.max_pdf_pages)
    return [
        Document(
            text=text,
            source=source,
            doc_type=_text_doc_type(source),
            record_id=str(page),
            split=True,
        )
        for page, text in pages
    ]


def load_zip(zip_bytes: bytes) -> List[Document]:
    """Read a zip in-memory and dispatch each inner file by extension. The inner
    path (e.g. 'documents/policies/support_sla.txt') becomes the citation source."""
    docs: List[Document] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            inner = name.replace("\\", "/")
            ext = inner.rsplit(".", 1)[-1].lower() if "." in inner else ""
            raw = zf.read(name)
            try:
                if ext in ("txt", "md"):
                    docs += load_text(raw.decode("utf-8", errors="replace"), inner)
                elif ext == "csv":
                    docs += load_csv(raw.decode("utf-8-sig", errors="replace"),
                                     Path(inner).name)
                elif ext == "json":
                    docs += load_email_threads(raw.decode("utf-8", errors="replace"),
                                               Path(inner).name)
                elif ext == "pdf":
                    docs += load_pdf(raw, inner)
            except Exception:
                continue  # skip a corrupt entry, keep the rest
    return docs


# --------------------------------------------------------------------------- #
# top-level dataset walk
# --------------------------------------------------------------------------- #
def _should_skip(path: Path) -> bool:
    low = path.name.lower()
    if any(s in low for s in _SKIP_NAME_SUBSTRINGS):
        return True
    return False


def load_dataset(root: str) -> List[Document]:
    """Walk the dataset folder and return every Document, ready to ingest."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root_path.resolve()}")

    docs: List[Document] = []
    for path in sorted(root_path.rglob("*")):
        if path.is_dir() or ".git" in path.parts or _should_skip(path):
            continue
        ext = path.suffix.lower().lstrip(".")
        try:
            if ext == "zip":
                docs += load_zip(path.read_bytes())
            elif ext == "csv":
                docs += load_csv(path.read_text(encoding="utf-8-sig", errors="replace"),
                                 path.name)
            elif ext == "json":
                docs += load_email_threads(path.read_text(encoding="utf-8", errors="replace"),
                                           path.name)
            elif ext in ("txt", "md"):
                rel = path.relative_to(root_path).as_posix()
                docs += load_text(path.read_text(encoding="utf-8", errors="replace"), rel)
            elif ext == "pdf":
                rel = path.relative_to(root_path).as_posix()
                docs += load_pdf(path.read_bytes(), rel)
        except Exception:
            continue
    return docs
