"""Local embeddings via fastembed (BAAI/bge-small-en-v1.5, ONNX).

Why local + fastembed:
  * The assignment forbids hosted APIs in local mode, and we want embeddings to be
    free and provider-agnostic so the index never needs rebuilding when switching
    local<->Bedrock.
  * fastembed runs the model through onnxruntime — no PyTorch — so it stays light
    and fast on a CPU-only laptop (model is ~130 MB, downloaded once).

The model is loaded lazily as a process-wide singleton: importing this module is
cheap, and tests can monkeypatch `embed_texts` / `embed_query` without ever
downloading the model.
"""

from typing import List, Optional

from app.config import settings

_model = None


def _get_model():
    """Lazily construct the fastembed model (downloads weights on first call)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=settings.embedding_model)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed document/passage texts (no query prefix)."""
    model = _get_model()
    return [vec.tolist() for vec in model.embed(list(texts))]


def embed_query(text: str) -> List[float]:
    """Embed a search query. bge-v1.5 is asymmetric, so the query gets an
    instruction prefix while passages do not — this measurably improves recall."""
    prefixed = settings.embedding_query_prefix + text
    model = _get_model()
    return next(iter(model.embed([prefixed]))).tolist()
