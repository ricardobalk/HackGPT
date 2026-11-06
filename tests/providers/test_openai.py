"""Tests for OpenAI model discovery and request translation."""

import json
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from openai import OpenAI

from hackgpt.providers.openai import OpenAIProvider
from hackgpt.providers.types import (
    Capability,
    ChatMessage,
    ChatRequest,
    OptionValidationError,
    ProviderConfigurationError,
)


@pytest.fixture(autouse=True)
def isolate_alias_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL_ALIASES", raising=False)


class FakeModels:
    def __init__(self, model_ids: tuple[str, ...]) -> None:
        self.model_ids = model_ids

    def list(self) -> SimpleNamespace:
        return SimpleNamespace(
            data=[SimpleNamespace(id=model_id) for model_id in self.model_ids]
        )


class FakeResponses:
    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def create(self, **options: Any) -> list[SimpleNamespace]:
        self.options = options
        return [SimpleNamespace(type="response.output_text.delta", delta="Hello")]


class FakeClient:
    def __init__(
        self,
        model_ids: tuple[str, ...] = (
            "text-embedding-3-small",
            "gpt-4.1",
            "gpt-5.6-sol",
        ),
    ) -> None:
        self.models = FakeModels(model_ids)
        self.responses = FakeResponses()


def test_openai_discovers_chat_models_and_model_specific_schemas() -> None:
    provider = OpenAIProvider(client=FakeClient())

    models = {model.id: model for model in provider.discover_models()}

    assert set(models) == {"gpt-4.1", "gpt-5.6-sol"}
    assert {option.key for option in models["gpt-4.1"].options} == {
        "max_tokens",
        "temperature",
        "top_p",
    }
    assert "reasoning_mode" in {option.key for option in models["gpt-5.6-sol"].options}


def test_openai_translates_generic_options_to_responses_request() -> None:
    client = FakeClient()
    provider = OpenAIProvider(client=client)
    model = next(
        model for model in provider.discover_models() if model.id == "gpt-5.6-sol"
    )
    request = ChatRequest(
        model=model,
        system_prompt="Be helpful.",
        messages=(ChatMessage("user", "Hello"),),
        options={
            "max_tokens": 1024,
            "reasoning_effort": "high",
            "reasoning_mode": "pro",
            "verbosity": "low",
        },
        request_id="session-id",
    )

    assert "".join(provider.stream_chat(request)) == "Hello"
    assert client.responses.options == {
        "model": "gpt-5.6-sol",
        "instructions": "Be helpful.",
        "input": [{"role": "user", "content": "Hello"}],
        "max_output_tokens": 1024,
        "safety_identifier": "session-id",
        "stream": True,
        "reasoning": {"effort": "high", "mode": "pro"},
        "text": {"verbosity": "low"},
    }


@pytest.mark.parametrize(
    ("model_id", "supports_none"),
    [
        ("gpt-6-astra", False),
        ("gpt-6-sol", True),
        ("gpt-6-luna", True),
        ("gpt-6.1-sol", False),
        ("gpt-6.1-terra", True),
        ("gpt-6-astra-2026-09-03", False),
        ("gpt-6.1-sol-2026-09-29", False),
        ("openai/gpt-6-luna", True),
    ],
)
def test_gpt6_profiles(model_id: str, supports_none: bool) -> None:
    provider = OpenAIProvider(client=FakeClient((model_id,)))
    model = provider.discover_models()[0]
    schema = {option.key: option for option in model.options}

    assert model.id == model_id
    assert Capability.REASONING in model.capabilities
    assert schema["max_tokens"].maximum == 128_000
    assert schema["reasoning_effort"].default == "medium"
    efforts = {choice.value for choice in schema["reasoning_effort"].choices}
    assert efforts == {"low", "medium", "high", "xhigh", "max"} | (
        {"none"} if supports_none else set()
    )
    assert {choice.value for choice in schema["reasoning_mode"].choices} == {
        "standard",
        "pro",
    }
    assert {choice.value for choice in schema["verbosity"].choices} == {
        "low",
        "medium",
        "high",
    }
    for key in ("temperature", "top_p"):
        assert (key in schema) == supports_none
        if supports_none:
            condition = schema[key].visible_when
            assert condition is not None
            assert (condition.option_key, condition.equals) == (
                "reasoning_effort",
                "none",
            )


@pytest.mark.parametrize("model_id", ["gpt-6-astra", "gpt-6.1-sol"])
@pytest.mark.parametrize("effort", ["none", "minimal", "ultra"])
def test_gpt6_rejects_unsupported_effort_before_request(
    model_id: str, effort: str
) -> None:
    client = FakeClient((model_id,))
    provider = OpenAIProvider(client=client)
    model = provider.discover_models()[0]
    request = ChatRequest(model, "", (), {"reasoning_effort": effort}, "test")

    with pytest.raises(OptionValidationError, match="unsupported value"):
        list(provider.stream_chat(request))
    assert client.responses.options == {}


@pytest.mark.parametrize("maximum", [128_001, 0])
def test_gpt6_rejects_invalid_output_budget(maximum: int) -> None:
    client = FakeClient(("gpt-6-sol",))
    provider = OpenAIProvider(client=client)
    model = provider.discover_models()[0]
    request = ChatRequest(model, "", (), {"max_tokens": maximum}, "test")

    with pytest.raises(OptionValidationError):
        list(provider.stream_chat(request))
    assert client.responses.options == {}


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_gpt6_drops_stale_sampling_values_when_reasoning(effort: str) -> None:
    client = FakeClient(("gpt-6-sol",))
    provider = OpenAIProvider(client=client)
    model = provider.discover_models()[0]
    request = ChatRequest(
        model,
        "",
        (),
        {"reasoning_effort": effort, "temperature": 0.7, "top_p": 0.9},
        "test",
    )

    assert list(provider.stream_chat(request)) == ["Hello"]
    assert client.responses.options["reasoning"] == {"effort": effort}
    assert "temperature" not in client.responses.options
    assert "top_p" not in client.responses.options


@pytest.mark.parametrize("model_id", ["gpt-6", "gpt-6-future", "gpt-6.2-sol"])
def test_unverified_gpt6_variants_do_not_inherit_sampling(model_id: str) -> None:
    provider = OpenAIProvider(client=FakeClient((model_id,)))
    model = provider.discover_models()[0]

    assert {option.key for option in model.options} == {"max_tokens"}
    assert Capability.REASONING not in model.capabilities


@pytest.mark.parametrize("value", ["invalid", "[]", '{"team": 6}', '{"team": ""}'])
def test_invalid_gateway_alias_configuration(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("OPENAI_MODEL_ALIASES", value)
    provider = OpenAIProvider(client=FakeClient())
    with pytest.raises(ProviderConfigurationError, match="OPENAI_MODEL_ALIASES"):
        provider.discover_models()


@pytest.mark.parametrize(
    ("base_url", "model_id", "canonical", "effort"),
    [
        ("https://api.openai.com/v1", "gpt-6-astra", "gpt-6-astra", "max"),
        ("https://litellm.test/v1", "openai/gpt-6-sol", "gpt-6-sol", "none"),
        ("https://litellm.test/v1", "team-reasoner", "gpt-6.1-sol", "high"),
        ("https://gateway.test/openai/v1", "azure/team-chat", "gpt-6-luna", "none"),
        ("https://gateway.test/v1", "team-legacy", "gpt-4.1", None),
        ("https://litellm.test/v1", "team-sol", "gpt-5.6-sol", "high"),
    ],
)
def test_sdk_gateway_discovery_and_streaming_payload(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    model_id: str,
    canonical: str,
    effort: str | None,
) -> None:
    # Exercise the real SDK's URL routing, JSON serialization, and SSE parser.
    if model_id != canonical and not model_id.startswith("openai/"):
        monkeypatch.setenv("OPENAI_MODEL_ALIASES", json.dumps({model_id: canonical}))
    payloads: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            assert str(request.url) == f"{base_url}/models"
            return httpx2.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "test",
                        },
                        {
                            "id": "openai/text-embedding-3-small",
                            "object": "model",
                            "created": 0,
                            "owned_by": "test",
                        },
                    ],
                },
            )
        assert request.method == "POST"
        assert str(request.url) == f"{base_url}/responses"
        payloads.append(json.loads(request.content))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
            "data: [DONE]\n\n",
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    http_client = httpx2.Client(transport=httpx2.MockTransport(handle))

    def make_client(**kwargs: Any) -> OpenAI:
        return OpenAI(http_client=http_client, **kwargs)

    monkeypatch.setattr("hackgpt.providers.openai.OpenAI", make_client)
    provider = OpenAIProvider()
    try:
        models = provider.discover_models()
        assert [model.id for model in models] == [model_id]
        options: dict[str, Any] = {"max_tokens": 1024}
        if effort is not None:
            options.update(
                reasoning_effort=effort, reasoning_mode="pro", verbosity="low"
            )
        if effort == "none" or canonical == "gpt-4.1":
            options.update(temperature=0.7, top_p=0.9)
        request = ChatRequest(
            models[0], "Be helpful.", (ChatMessage("user", "Hi"),), options, "test-id"
        )
        assert "".join(provider.stream_chat(request)) == "Hello"
        expected: dict[str, Any] = {
            "model": model_id,
            "instructions": "Be helpful.",
            "input": [{"role": "user", "content": "Hi"}],
            "max_output_tokens": 1024,
            "safety_identifier": "test-id",
            "stream": True,
        }
        if effort is not None:
            expected.update(
                reasoning={"effort": effort, "mode": "pro"}, text={"verbosity": "low"}
            )
        if effort == "none" or canonical == "gpt-4.1":
            expected.update(temperature=0.7, top_p=0.9)
        assert payloads == [expected]
    finally:
        http_client.close()


@pytest.mark.parametrize("model_id", ["gpt-6-astra", "gpt-6-sol", "gpt-6.1-terra"])
def test_gpt6_defaults_and_maximum_output_translation(model_id: str) -> None:
    client = FakeClient((model_id,))
    provider = OpenAIProvider(client=client)
    model = provider.discover_models()[0]

    list(provider.stream_chat(ChatRequest(model, "", (), {}, "test")))
    assert client.responses.options["max_output_tokens"] == 2048
    assert client.responses.options["reasoning"] == {"effort": "medium"}
    assert client.responses.options["text"] == {"verbosity": "medium"}
    assert "temperature" not in client.responses.options
    assert "top_p" not in client.responses.options

    list(
        provider.stream_chat(
            ChatRequest(model, "", (), {"max_tokens": 128_000}, "test")
        )
    )
    assert client.responses.options["max_output_tokens"] == 128_000


def test_gateway_aliases_only_select_discovered_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OPENAI_MODEL_ALIASES",
        json.dumps(
            {"team-reasoner": "gpt-6.1-sol", "missing-deployment": "gpt-6-luna"}
        ),
    )
    client = FakeClient(("team-reasoner", "unmapped-deployment", "gpt-4.1"))
    provider = OpenAIProvider(client=client)
    models = {model.id: model for model in provider.discover_models()}
    assert set(models) == {"team-reasoner", "gpt-4.1"}
    assert Capability.REASONING in models["team-reasoner"].capabilities
