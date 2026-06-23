"""Lightweight, repeatable evaluation harness.

Runs a fixed question set (eval/eval_questions.json) covering every behaviour the
assignment asks about: answerable, multi-source, refusal, customer summary, email
drafting, structured records, conflicting/incomplete info, and source grounding.

Two modes:

  python -m eval.eval --retrieval-only   # checks retrieval/grounding; NO LLM needed
  python -m eval.eval                     # full pipeline (needs the configured provider)

Retrieval checks (always): did we retrieve the expected sources / doc_types, and
did entity detection fire where expected? Answer checks (full mode): refusal where
required, expected substrings present, forbidden substrings absent, citations present.

It is intentionally simple (substring assertions, not a model-graded rubric) so it
is fast, deterministic, and easy to read during the demo.
"""

import argparse
import json
import sys
from pathlib import Path

from app.config import settings
from app.services.retriever import retrieve

QUESTIONS = json.loads((Path(__file__).parent / "eval_questions.json").read_text("utf-8"))

REFUSAL_MARKERS = [
    "cannot", "can't", "not enough", "no information", "insufficient",
    "couldn't find", "could not find", "isn't enough", "is not enough",
    "don't have", "do not have", "not available", "no relevant",
    "not contain", "doesn't contain", "unable",
]


def _retrieval_checks(q: dict) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    fails = []
    chunks, entity = retrieve(q["question"])
    sources = " ".join(c["source"] for c in chunks).lower()
    rids = " ".join(c["record_id"] for c in chunks).lower()
    haystack = sources + " " + rids
    doc_types = {c["doc_type"] for c in chunks}

    for s in q.get("expect_sources", []):
        if s.lower() not in haystack:
            fails.append(f"missing source ~ '{s}'")
    for dt in q.get("expect_doc_types", []):
        if dt not in doc_types:
            fails.append(f"missing doc_type '{dt}'")
    if "expect_entity" in q:
        if not entity or q["expect_entity"].lower() not in entity.lower():
            fails.append(f"entity not detected (got {entity!r})")
    return fails


def _answer_checks(q: dict, answer: str, citations: list) -> list[str]:
    fails = []
    low = answer.lower()
    if q.get("expect_refusal"):
        if not any(m in low for m in REFUSAL_MARKERS):
            fails.append("expected a refusal but answer looks confident")
    for s in q.get("expect_not_contains", []):
        if s.lower() in low:
            fails.append(f"answer should NOT contain '{s}'")
    if "expect_contains_any" in q:
        if not any(s.lower() in low for s in q["expect_contains_any"]):
            fails.append(f"answer missing any of {q['expect_contains_any']}")
    if q.get("expect_citation") and not citations:
        fails.append("expected at least one citation")
    return fails


def run(retrieval_only: bool) -> int:
    from app.agent.graph import qa_graph  # imported lazily (only needed in full mode)

    passed = total = 0
    print(f"Mode: {'retrieval-only' if retrieval_only else 'full (provider=' + settings.llm_provider + ')'}")
    print("=" * 72)

    for q in QUESTIONS:
        total += 1
        fails = _retrieval_checks(q)

        if not retrieval_only:
            result = qa_graph.invoke({
                "question": q["question"], "retrieved_chunks": [], "answer": "",
                "citations": [], "error": None, "matched_entity": None, "gen_ms": None,
            })
            if result.get("error"):
                # An LLM/provider error is a harness failure, not a refusal.
                fails.append(f"pipeline error: {result['error'][:80]}")
            else:
                fails += _answer_checks(q, result["answer"], result["citations"])

        ok = not fails
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {q['id']:<28} ({q['category']})")
        for f in fails:
            print(f"        - {f}")

    print("=" * 72)
    print(f"{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="eval.eval")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip generation; check retrieval/grounding only (no LLM needed)")
    args = ap.parse_args()
    sys.exit(run(args.retrieval_only))
