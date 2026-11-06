"""Tests for Ollama discovery and request translation."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from hackgpt.providers.ollama import OllamaProvider
from hackgpt.providers.types import ChatMessage, ChatRequest


class FakeClient:
    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def list(self) -> SimpleNamespace:
        model = SimpleNamespace(
            model="gpt-oss:20b",
            details=SimpleNamespace(family="gptoss"),
        )
        return SimpleNamespace(models=[model])

    def show(self, model: str) -> SimpleNamespace:
        assert model == "gpt-oss:20b"
        return SimpleNamespace(
            capabilities=["completion", "thinking", "tools"],
            modelinfo={"gptoss.context_length": 131_072},
        )

    def chat(self, **options: Any) -> Sequence[SimpleNamespace]:
        self.options = options
        return [SimpleNamespace(message=SimpleNamespace(content="Hello"))]


def test_ollama_uses_show_capabilities_for_schema() -> None:
    provider = OllamaProvider(client=FakeClient())

    model = provider.discover_models()[0]

    think = next(option for option in model.options if option.key == "think")
    assert tuple(choice.value for choice in think.choices) == ("low", "medium", "high")


def test_ollama_translates_generic_chat_request() -> None:
    client = FakeClient()
    provider = OllamaProvider(client=client)
    model = provider.discover_models()[0]
    request = ChatRequest(
        model=model,
        system_prompt="Be helpful.",
        messages=(ChatMessage("user", "Hello"),),
        options={"max_tokens": 1024, "think": "high"},
        request_id="unused-by-ollama",
    )

    assert "".join(provider.stream_chat(request)) == "Hello"
    assert client.options["model"] == "gpt-oss:20b"
    assert client.options["messages"] == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
    ]
    assert client.options["options"]["num_predict"] == 1024
    assert client.options["think"] == "high"
