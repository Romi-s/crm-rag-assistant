"""Local generation via Ollama (open-source model on this machine).

Uses Ollama's native HTTP API (`/api/chat`) over plain `requests` — no hosted-LLM
SDK is imported anywhere in local mode, so there is zero ambiguity about the
"local only, no external API" requirement.
"""

import requests

from app.providers.base import LLMProvider, ProviderError


class OllamaProvider(LLMProvider):
    name = "local"

    def __init__(self, host: str, model: str, timeout: int):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise ProviderError(
                f"Could not reach Ollama at {self.host}. Is it running? "
                "Start it with `ollama serve`, then `ollama pull "
                f"{self.model}`."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise ProviderError(
                f"Ollama timed out after {self.timeout}s generating with "
                f"{self.model}. On CPU, try a smaller model (e.g. llama3.2:3b) "
                "or raise OLLAMA_TIMEOUT."
            ) from exc

        if resp.status_code == 404:
            raise ProviderError(
                f"Model '{self.model}' is not pulled in Ollama. Run "
                f"`ollama pull {self.model}`."
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        content = (data.get("message") or {}).get("content", "")
        if not content.strip():
            raise ProviderError("Ollama returned an empty response.")
        return content.strip()
