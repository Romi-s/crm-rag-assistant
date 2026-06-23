"""Index the provided dataset on first boot if the collection is empty.

Runs in a background thread at startup (see app.main) so the server and /health
stay responsive while the corpus is embedded. Idempotent: once the collection has
vectors, this is a no-op. To force a fresh rebuild, use the CLI:
    python -m app.cli ingest --rebuild
"""

import threading

from app.config import settings
from app.services.ingest import get_collection, ingest_dataset

_lock = threading.Lock()
_seeded = False


def ensure_seeded() -> None:
    global _seeded
    if _seeded:
        return
    with _lock:
        if _seeded:
            return
        try:
            if get_collection().count() > 0:
                _seeded = True
                return
            result = ingest_dataset(settings.dataset_dir, rebuild=False)
            if result.get("chunks_added", 0) > 0:
                _seeded = True
        except Exception:
            # Dataset folder missing or embed model still downloading — leave
            # unseeded and let a later call (or the CLI) retry.
            pass
