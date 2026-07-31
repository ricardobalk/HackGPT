"""Tests for Anthropic discovery and request translation."""

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any

from hackgpt.providers.anthropic import AnthropicProvider
from hackgpt.providers.types import ChatMessage, ChatRequest


def support(supported: bool) -> SimpleNamespace:
    return SimpleNamespace(supported=supported)


class FakeModels:
    def list(self, *, limit: int) -> SimpleNamespace:
        assert limit == 1000
        capabilities = SimpleNamespace(
            image_input=support(True),
            structured_outputs=support(True),
            thinking=SimpleNamespace(
                supported=True,
                types=SimpleNamespace(adaptive=support(True), enabled=support(True)),
            ),
            effort=SimpleNamespace(
                supported=True,
                low=support(True),
                medium=support(True),
                high=support(True),
                xhigh=support(False),
                max=support(False),
            ),
        )
        model = SimpleNamespace(
            id="claude-test",
            display_name="Claude Test",
            max_tokens=32_000,
            capabilities=capabilities,
        )
        return SimpleNamespace(data=[model])


class FakeStream(AbstractContextManager[SimpleNamespace]):
    def __enter__(self) -> SimpleNamespace:
        return SimpleNamespace(text_stream=iter(("Hel", "lo")))

    def __exit__(self, *args: object) -> None:
        return None


class FakeMessages:
    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def stream(self, **options: Any) -> FakeStream:
        self.options = options
        return FakeStream()


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()
        self.messages = FakeMessages()


def test_anthropic_uses_discovered_capabilities_for_schema() -> None:
    provider = AnthropicProvider(client=FakeClient())

    model = provider.discover_models()[0]

    assert model.label == "Claude Test"
    assert {option.key for option in model.options} >= {
        "max_tokens",
        "thinking_mode",
        "thinking_budget",
        "effort",
    }


def test_anthropic_translates_adaptive_thinking_request() -> None:
    client = FakeClient()
    provider = AnthropicProvider(client=client)
    model = provider.discover_models()[0]
    request = ChatRequest(
        model=model,
        system_prompt="Be helpful.",
        messages=(ChatMessage("user", "Hello"),),
        options={
            "max_tokens": 8000,
            "thinking_mode": "adaptive",
            "effort": "high",
        },
        request_id="unused-by-anthropic",
    )

    assert "".join(provider.stream_chat(request)) == "Hello"
    assert client.messages.options == {
        "model": "claude-test",
        "system": "Be helpful.",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8000,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
