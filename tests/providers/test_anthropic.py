"""Tests for Anthropic discovery and request translation."""

import json
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic

from hackgpt.providers.anthropic import AnthropicProvider
from hackgpt.providers.types import ChatMessage, ChatRequest, OptionValidationError


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


@pytest.mark.parametrize("key", ["temperature", "top_p", "top_k"])
def test_anthropic_rejects_removed_sampling_controls(key: str) -> None:
    client = FakeClient()
    provider = AnthropicProvider(client=client)
    model = provider.discover_models()[0]
    request = ChatRequest(
        model=model,
        system_prompt="Be precise.",
        messages=(ChatMessage("user", "Hello"),),
        options={key: 0.0},
        request_id="unused-by-anthropic",
    )

    assert key not in {option.key for option in model.options}
    with pytest.raises(OptionValidationError, match="Unsupported generation option"):
        list(provider.stream_chat(request))
    assert client.messages.options == {}


@pytest.mark.parametrize("thinking_mode", ["disabled", "adaptive", "enabled"])
def test_anthropic_v2_sdk_discovery_and_streaming(thinking_mode: str) -> None:
    payloads: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["anthropic-version"] == "2026-10-21"
        assert request.headers["authorization"] == "Bearer test-key"
        if request.method == "GET":
            assert request.url.path == "/v1/models"
            model = FakeModels().list(limit=1000).data[0]
            data = json.loads(json.dumps(model, default=vars))
            data.update(
                type="model", created_at="2026-01-01T00:00:00Z", lifecycle="active"
            )
            return httpx2.Response(
                200,
                json={
                    "data": [data],
                    "has_more": False,
                    "first_id": model.id,
                    "last_id": model.id,
                },
            )
        assert request.method == "POST"
        assert request.url.path == "/v1/messages"
        payloads.append(json.loads(request.content))
        events: list[dict[str, object]] = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            ),
        )

    with httpx2.Client(transport=httpx2.MockTransport(handle)) as http_client:
        with Anthropic(
            api_key="test-key",
            base_url="https://claude.test",
            http_client=http_client,
        ) as client:
            provider = AnthropicProvider(client=client)
            model = provider.discover_models()[0]
            request = ChatRequest(
                model,
                "Be helpful.",
                (ChatMessage("user", "Hi"),),
                {"thinking_mode": thinking_mode, "max_tokens": 8000},
                "test",
            )
            assert "".join(provider.stream_chat(request)) == "Hello"

    expected: dict[str, Any] = {
        "model": "claude-test",
        "system": "Be helpful.",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8000,
        "stream": True,
    }
    if thinking_mode == "adaptive":
        expected.update(
            thinking={"type": "adaptive"}, output_config={"effort": "medium"}
        )
    elif thinking_mode == "enabled":
        expected["thinking"] = {"type": "enabled", "budget_tokens": 4096}
    assert payloads == [expected]
