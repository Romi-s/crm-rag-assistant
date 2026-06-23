"""Suggested questions for the UI.

Curated, dataset-relevant prompts that exercise the assistant's headline
capabilities (account summary, SLA lookup, open critical tickets, email drafting,
refusal). Kept static on purpose: it avoids a slow CPU LLM call on every page load
and keeps the UI responsive. If the corpus is empty we return a generic set.
"""

from typing import List

from app.services.ingest import get_collection

_SUGGESTIONS = [
    "Summarize the account history for Alpha Trading LLC before my meeting.",
    "Which customers have open critical support tickets?",
    "What is our support SLA for billing issues?",
    "Draft a reply to Sunrise Education Group's onboarding email.",
    "What channels and features does the AllMessage platform support?",
    "Summarize recent sales notes for Apex Legal Advisors.",
]

_DEFAULTS = [
    "What is our refund policy?",
    "What is the support SLA for critical issues?",
    "Which features does the CRM platform support?",
]


def get_suggestions() -> List[str]:
    try:
        if get_collection().count() == 0:
            return list(_DEFAULTS)
    except Exception:
        return list(_DEFAULTS)
    return list(_SUGGESTIONS)
