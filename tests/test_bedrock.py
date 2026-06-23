"""Hermetic tests for the Bedrock provider — boto client is faked, no AWS calls."""

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from app.providers.base import ProviderError
from app.providers.bedrock import BedrockProvider, _estimate_usd


def _provider(fake_client):
    p = BedrockProvider(region="us-east-1", model_id="anthropic.claude-3-haiku-20240307-v1:0")
    p._client = fake_client  # bypass real boto3 client creation
    return p


class _OkClient:
    def converse(self, **kwargs):
        self.kwargs = kwargs
        return {
            "output": {"message": {"content": [{"text": "Grounded answer [1]."}]}},
            "usage": {"inputTokens": 1200, "outputTokens": 80},
        }


class _RaisingClient:
    def __init__(self, exc):
        self.exc = exc

    def converse(self, **kwargs):
        raise self.exc


def test_bedrock_success_returns_text_and_sends_converse_shape():
    client = _OkClient()
    out = _provider(client).generate("sys", "question?", max_tokens=256, temperature=0.1)
    assert out == "Grounded answer [1]."
    # Converse API shape is correct
    assert client.kwargs["modelId"].startswith("anthropic.claude-3-haiku")
    assert client.kwargs["system"] == [{"text": "sys"}]
    assert client.kwargs["messages"][0]["content"][0]["text"] == "question?"
    assert client.kwargs["inferenceConfig"] == {"maxTokens": 256, "temperature": 0.1}


def test_bedrock_access_denied_maps_to_clear_error():
    err = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "Converse")
    with pytest.raises(ProviderError, match="Access denied"):
        _provider(_RaisingClient(err)).generate("s", "u", max_tokens=10, temperature=0.0)


def test_bedrock_model_not_found_maps_to_clear_error():
    err = ClientError({"Error": {"Code": "ValidationException", "Message": "bad"}}, "Converse")
    with pytest.raises(ProviderError, match="not available"):
        _provider(_RaisingClient(err)).generate("s", "u", max_tokens=10, temperature=0.0)


def test_bedrock_missing_credentials_maps_to_clear_error():
    with pytest.raises(ProviderError, match="credentials"):
        _provider(_RaisingClient(NoCredentialsError())).generate("s", "u", max_tokens=10, temperature=0.0)


def test_estimate_usd_known_model():
    cost = _estimate_usd("anthropic.claude-3-haiku-20240307-v1:0", 1000, 1000)
    assert cost == pytest.approx(0.00025 + 0.00125)


def test_estimate_usd_unknown_model_returns_none():
    assert _estimate_usd("some.unknown-model", 1000, 1000) is None
