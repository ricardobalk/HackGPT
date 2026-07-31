"""Tests for OpenAI model discovery and request translation."""

from types import SimpleNamespace
from typing import Any

from hackgpt.providers.openai import OpenAIProvider
from hackgpt.providers.types import ChatMessage, ChatRequest


class FakeModels:
    def list(self) -> SimpleNamespace:
        return SimpleNamespace(
            data=[
                SimpleNamespace(id="text-embedding-3-small"),
                SimpleNamespace(id="gpt-4.1"),
                SimpleNamespace(id="gpt-5.6-sol"),
            ]
        )


class FakeResponses:
    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def create(self, **options: Any) -> list[SimpleNamespace]:
        self.options = options
        return [SimpleNamespace(type="response.output_text.delta", delta="Hello")]


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()
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
