"""Tests for provider-neutral capability and option contracts."""

import pytest

from hackgpt.providers.types import (
    OptionChoice,
    OptionCondition,
    OptionKind,
    OptionSchema,
    OptionValidationError,
    validate_options,
)


def test_validate_options_applies_defaults_and_conditions() -> None:
    schema = (
        OptionSchema(
            key="thinking",
            kind=OptionKind.BOOLEAN,
            label="Thinking",
            default=False,
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
            visible_when=OptionCondition("thinking", True),
        ),
    )

    assert validate_options(schema, {}) == {"thinking": False}
    assert validate_options(schema, {"thinking": True}) == {
        "thinking": True,
        "effort": "medium",
    }


@pytest.mark.parametrize("value", [0, 11, 1.5, True])
def test_validate_options_rejects_invalid_integer_values(value: object) -> None:
    schema = (
        OptionSchema(
            key="max_tokens",
            kind=OptionKind.INTEGER,
            label="Maximum tokens",
            default=5,
            minimum=1,
            maximum=10,
        ),
    )

    with pytest.raises(OptionValidationError):
        validate_options(schema, {"max_tokens": value})  # type: ignore[dict-item]


def test_validate_options_rejects_unknown_options() -> None:
    with pytest.raises(OptionValidationError, match="unknown"):
        validate_options((), {"unknown": 1})
