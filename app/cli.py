"""Command-line interface for indexing and querying the CRM assistant.

    python -m app.cli ingest [--path DIR] [--rebuild]   # (re)build the index
    python -m app.cli stats                              # show what's indexed
    python -m app.cli search "query"                     # retrieval only (no LLM)
    python -m app.cli ask "question"                     # full RAG answer (needs Ollama)

`search` is handy for verifying retrieval/grounding without a running LLM; `ask`
runs the whole pipeline through the configured provider.
"""

import argparse
import sys

from app.config import settings


def cmd_ingest(args) -> int:
    from app.services.ingest import ingest_dataset

    path = args.path or settings.dataset_dir
    print(f"Indexing dataset: {path}  (rebuild={args.rebuild})")
    result = ingest_dataset(path, rebuild=args.rebuild)
    print(
        f"Done. source_files={result.get('source_files')} "
        f"documents={result.get('documents')} chunks_added={result.get('chunks_added')}"
    )
    return 0


def cmd_stats(_args) -> int:
    from collections import Counter

    from app.services.ingest import get_collection

    col = get_collection()
    count = col.count()
    print(f"Collection '{settings.collection_name}': {count} chunks")
    if count:
        metas = col.get(include=["metadatas"])["metadatas"]
        by_type = Counter(m.get("doc_type", "") for m in metas)
        sources = sorted({m.get("source", "") for m in metas})
        print("By doc_type:")
        for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {t:<12} {n}")
        print(f"Source files ({len(sources)}):")
        for s in sources:
            print(f"  {s}")
    return 0


def cmd_search(args) -> int:
    from app.services.retriever import retrieve

    chunks, entity = retrieve(args.query)
    if entity:
        print(f"[entity matched: {entity}]")
    print(f"Retrieved {len(chunks)} chunks for: {args.query!r}\n")
    for i, c in enumerate(chunks, 1):
        head = f"[{i}] via={c['via']} type={c['doc_type']} src={c['source']}"
        if c["record_id"]:
            head += f" id={c['record_id']}"
        if c["company_name"]:
            head += f" | {c['company_name']}"
        print(head)
        print("    " + c["text"].replace("\n", " ")[:200] + "\n")
    return 0


def cmd_ask(args) -> int:
    from app.agent.graph import qa_graph

    result = qa_graph.invoke(
        {
            "question": args.query,
            "retrieved_chunks": [],
            "answer": "",
            "citations": [],
            "error": None,
            "matched_entity": None,
            "gen_ms": None,
        }
    )
    if result.get("error"):
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return 1

    print(result["answer"])
    print("\nSources:")
    for c in result["citations"]:
        rid = f" {c['record_id']}" if c["record_id"] else ""
        comp = f" — {c['company_name']}" if c["company_name"] else ""
        print(f"  - [{c['doc_type']}] {c['source']}{rid}{comp}")
    if result.get("matched_entity"):
        print(f"\n(entity: {result['matched_entity']})")
    print(f"(generation: {result.get('gen_ms')} ms via {settings.llm_provider})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli", description="CRM RAG Assistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="(re)build the dataset index")
    p_ingest.add_argument("--path", default=None, help="dataset folder (default: DATASET_DIR)")
    p_ingest.add_argument("--rebuild", action="store_true", help="wipe and rebuild the index")
    p_ingest.set_defaults(func=cmd_ingest)

    sub.add_parser("stats", help="show indexed chunk/doc_type counts").set_defaults(func=cmd_stats)

    p_search = sub.add_parser("search", help="retrieval only, no LLM")
    p_search.add_argument("query")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="full RAG answer (needs the configured provider)")
    p_ask.add_argument("query")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
