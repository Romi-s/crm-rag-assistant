"""The single interface every generation backend implements."""

from abc import ABC, abstractmethod


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a response (down, misconfigured, etc.).

    Carries a human-readable, actionable message — surfaced to the user/UI instead
    of a raw stack trace so failures are debuggable during the demo.
    """


class LLMProvider(ABC):
    """Generate a grounded answer from a system + user prompt.

    Implementations must be side-effect free apart from the LLM call and must not
    log prompt contents (which contain customer data). They expose `name` and
    `model` purely for observability (shown in the pipeline telemetry).
    """

    #: short id for telemetry, e.g. "local" or "bedrock"
    name: str = "base"
    #: concrete model id, e.g. "qwen2.5:7b-instruct"
    model: str = ""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Return the model's text completion. Raise ProviderError on failure."""
        raise NotImplementedError
