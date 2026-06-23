"""Amazon Bedrock generation provider (Part 2).

Uses the Bedrock **Converse API**, which gives one uniform request/response shape
across model families (Anthropic Claude, Amazon Nova/Titan, Llama, Mistral) — so the
model is just a config value; no code change to switch models.

Credentials are read from the standard AWS chain (env vars / ~/.aws/credentials), so
no secret ever lives in this repo or in our settings. Only the system prompt + the
already-retrieved excerpts + the question are sent to AWS — nothing else.

Errors are mapped to clear, actionable ProviderError messages (missing creds, model
access not enabled, wrong region, throttling) so failures are debuggable in the demo
instead of dumping a boto stack trace.
"""

import logging

from app.providers.base import LLMProvider, ProviderError

log = logging.getLogger("crm-rag.bedrock")

# Rough on-demand USD pricing per 1K tokens (input, output), for the usage log only.
# Update if you pick a different model; used purely to print an estimated cost.
_PRICING = {
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-3-5-haiku": (0.0008, 0.004),
    "nova-micro": (0.000035, 0.00014),
    "nova-lite": (0.00006, 0.00024),
    "titan-text-express": (0.0002, 0.0006),
}


def _estimate_usd(model: str, in_tok: int, out_tok: int):
    for key, (pin, pout) in _PRICING.items():
        if key in model:
            return (in_tok / 1000) * pin + (out_tok / 1000) * pout
    return None


class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(self, region: str, model_id: str):
        self.region = region
        self.model = model_id
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ProviderError(
                    "boto3 is not installed. Run `pip install boto3` (or "
                    "`pip install -r requirements.txt`) to use Bedrock mode."
                ) from exc
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            NoCredentialsError,
            PartialCredentialsError,
        )

        try:
            # client creation can itself raise (e.g. bad AWS_PROFILE / region)
            client = self._get_client()
            resp = client.converse(
                modelId=self.model,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
        except (NoCredentialsError, PartialCredentialsError) as exc:
            raise ProviderError(
                "AWS credentials not found or incomplete. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY (in .env or via `aws configure`)."
            ) from exc
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("AccessDeniedException", "AccessDenied"):
                raise ProviderError(
                    f"Access denied for model '{self.model}' in {self.region}. Either "
                    "your IAM user is missing `bedrock:InvokeModel`, or model access is "
                    "not enabled — enable it in Bedrock console → Model access."
                ) from exc
            if code in ("ResourceNotFoundException", "ValidationException"):
                raise ProviderError(
                    f"Model '{self.model}' is not available/invokable in {self.region}. "
                    "Check the model id and region (some models need an inference "
                    "profile id, e.g. 'us.amazon.nova-lite-v1:0')."
                ) from exc
            if code in ("ThrottlingException", "TooManyRequestsException"):
                raise ProviderError("Bedrock throttled the request — wait and retry.") from exc
            raise ProviderError(f"Bedrock error [{code}]: {exc}") from exc
        except BotoCoreError as exc:
            raise ProviderError(
                f"Bedrock client/credentials error in region '{self.region}'. Check the "
                f"region name, your AWS profile/credentials, and network. ({exc})"
            ) from exc

        # --- observability: token usage + estimated cost (no customer data logged) ---
        usage = resp.get("usage", {}) or {}
        in_tok, out_tok = usage.get("inputTokens", 0), usage.get("outputTokens", 0)
        est = _estimate_usd(self.model, in_tok, out_tok)
        log.info(
            "Bedrock usage model=%s region=%s input_tokens=%s output_tokens=%s%s",
            self.model, self.region, in_tok, out_tok,
            f" est_usd=${est:.5f}" if est is not None else "",
        )

        try:
            return resp["output"]["message"]["content"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected Bedrock response shape: {resp}") from exc
