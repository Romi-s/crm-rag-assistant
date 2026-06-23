"""Test fixtures that make the suite hermetic.

No network, no model download, no Ollama:
  * embeddings are replaced by a deterministic token-hash vector (dim 64),
  * the vector store is a throwaway temp Chroma collection,
  * the LLM provider is a Fake that echoes a grounded, citation-bearing answer.
"""

import hashlib
import re
import uuid

import pytest

from app.config import settings
from app.providers.base import LLMProvider

_DIM = 64


def _fake_vector(text: str) -> list[float]:
    vec = [0.0] * _DIM
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def fake_embed_texts(texts):
    return [_fake_vector(t) for t in texts]


def fake_embed_query(text):
    return _fake_vector(text)


class FakeProvider(LLMProvider):
    name = "fake"
    model = "fake-model"

    def __init__(self, reply="Based on the records, the answer is grounded [1]."):
        self.reply = reply

    def generate(self, system_prompt, user_prompt, *, max_tokens, temperature):
        return self.reply


@pytest.fixture
def clean_index(tmp_path, monkeypatch):
    """A fresh temp collection with fake embeddings wired into ingest + retriever."""
    import app.services.ingest as ingest_mod
    import app.services.retriever as retriever_mod

    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "collection_name", f"test_{uuid.uuid4().hex[:8]}")
    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retriever_mod, "embed_query", fake_embed_query)
    retriever_mod.invalidate_caches()
    yield ingest_mod
    retriever_mod.invalidate_caches()


@pytest.fixture
def fake_provider(monkeypatch):
    import app.agent.nodes as nodes_mod
    import app.api.routes as routes_mod

    provider = FakeProvider()
    monkeypatch.setattr(nodes_mod, "get_provider", lambda: provider)
    monkeypatch.setattr(routes_mod, "get_provider", lambda: provider)
    return provider
