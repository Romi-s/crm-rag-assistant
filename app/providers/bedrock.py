"""Amazon Bedrock generation provider.

Stubbed for Part 1 — the interface and wiring are in place so Part 2 only needs to
fill in the boto3 call (no other code changes). Kept here to make the provider seam
explicit and reviewable.
"""

from app.providers.base import LLMProvider, ProviderError


class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(self, region: str, model_id: str):
        self.region = region
        self.model = model_id

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        # Part 2: invoke Bedrock via boto3 `bedrock-runtime` Converse API, e.g.
        #     client = boto3.client("bedrock-runtime", region_name=self.region)
        #     resp = client.converse(modelId=self.model, system=[{"text": system_prompt}],
        #                             messages=[{"role": "user",
        #                                        "content": [{"text": user_prompt}]}],
        #                             inferenceConfig={"maxTokens": max_tokens,
        #                                              "temperature": temperature})
        #     return resp["output"]["message"]["content"][0]["text"]
        raise ProviderError(
            "Bedrock mode is implemented in Part 2. Set LLM_PROVIDER=local to use "
            "the local Ollama backend."
        )
