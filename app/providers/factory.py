"""Select and cache the active generation provider from settings."""

from functools import lru_cache

from app.config import settings
from app.providers.base import LLMProvider, ProviderError


@lru_cache(maxsize=None)
def get_provider() -> LLMProvider:
    """Return the configured provider (cached). Driven by `LLM_PROVIDER`."""
    provider = settings.llm_provider.lower().strip()

    if provider == "local":
        from app.providers.local_ollama import OllamaProvider

        return OllamaProvider(
            host=settings.ollama_host,
            model=settings.ollama_model,
            timeout=settings.ollama_timeout,
        )

    if provider == "bedrock":
        from app.providers.bedrock import BedrockProvider

        return BedrockProvider(
            region=settings.aws_region,
            model_id=settings.bedrock_model_id,
        )

    raise ProviderError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use 'local' or 'bedrock'."
    )
