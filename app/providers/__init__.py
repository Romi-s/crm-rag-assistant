"""Pluggable LLM generation providers.

The rest of the app only ever talks to the `LLMProvider` interface, so swapping
the local Ollama backend for Amazon Bedrock (Part 2) is a one-line config change
(`LLM_PROVIDER=bedrock`) with no changes to retrieval, grounding, or the pipeline.
"""

from app.providers.base import LLMProvider, ProviderError
from app.providers.factory import get_provider

__all__ = ["LLMProvider", "ProviderError", "get_provider"]
