"""Tests for schema-driven Streamlit option rendering."""

from typing import Any, cast

import hackgpt.ui.options as options_module
from hackgpt.providers.types import (
    OptionChoice,
    OptionCondition,
    OptionKind,
    OptionSchema,
)
from hackgpt.ui.options import option_widget_key, render_option_fields


class FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.rendered_keys: list[str] = []

    def _value(self, key: str, default: Any) -> Any:
        self.rendered_keys.append(key)
        self.session_state.setdefault(key, default)
        return self.session_state[key]

    def checkbox(self, label: str, *, value: bool, key: str, **_: Any) -> bool:
        return bool(self._value(key, value))

    def number_input(
        self, label: str, *, value: int | float, key: str, **_: Any
    ) -> int | float:
        return cast(int | float, self._value(key, value))

    def selectbox(
        self,
        label: str,
        options: tuple[Any, ...],
        *,
        index: int,
        key: str,
        **_: Any,
    ) -> Any:
        return self._value(key, options[index])

    def text_input(self, label: str, *, value: str, key: str, **_: Any) -> str:
        return str(self._value(key, value))


def test_schema_renderer_uses_scoped_stable_keys(monkeypatch: Any) -> None:
    fake = FakeStreamlit()
    monkeypatch.setattr(options_module, "st", fake)
    schema = (
        OptionSchema(
            key="max_tokens",
            kind=OptionKind.INTEGER,
            label="Maximum tokens",
            default=1024,
            minimum=1,
            maximum=4096,
        ),
        OptionSchema(
            key="effort",
            kind=OptionKind.SELECT,
            label="Effort",
            default="medium",
            choices=(
                OptionChoice("low", "Low"),
                OptionChoice("medium", "Medium"),
            ),
        ),
    )

    values = render_option_fields(schema, provider_id="openai", model_id="gpt/test")

    assert values == {"max_tokens": 1024, "effort": "medium"}
    assert fake.rendered_keys == [
        "generation.openai.gpt/test.max_tokens",
        "generation.openai.gpt/test.effort",
    ]
    assert option_widget_key("ollama", "gemma", "top_p") != fake.rendered_keys[1]


def test_schema_renderer_resets_invalid_and_hidden_values(monkeypatch: Any) -> None:
    fake = FakeStreamlit()
    monkeypatch.setattr(options_module, "st", fake)
    max_key = option_widget_key("provider", "model", "max_tokens")
    budget_key = option_widget_key("provider", "model", "budget")
    fake.session_state[max_key] = 999
    fake.session_state[budget_key] = 2000
    schema = (
        OptionSchema(
            key="enabled",
            kind=OptionKind.BOOLEAN,
            label="Enabled",
            default=False,
        ),
        OptionSchema(
            key="max_tokens",
            kind=OptionKind.INTEGER,
            label="Maximum tokens",
            default=10,
            minimum=1,
            maximum=20,
        ),
        OptionSchema(
            key="budget",
            kind=OptionKind.INTEGER,
            label="Budget",
            default=5,
            minimum=1,
            maximum=10,
            visible_when=OptionCondition("enabled", True),
        ),
    )

    values = render_option_fields(schema, provider_id="provider", model_id="model")

    assert values == {"enabled": False, "max_tokens": 10}
    assert fake.session_state[max_key] == 10
    assert budget_key not in fake.session_state
