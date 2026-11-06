"""Tests for provider-neutral chat history state."""

from typing import Any

import hackgpt.state as state_module
from hackgpt.providers.types import ChatMessage


def test_legacy_history_is_normalised_and_preserved(monkeypatch: Any) -> None:
    session_state = {
        "chat_messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
    monkeypatch.setattr(state_module.st, "session_state", session_state)

    messages = state_module.get_chat_messages()

    assert messages == [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi"),
    ]
    assert session_state["chat_messages"] is messages
