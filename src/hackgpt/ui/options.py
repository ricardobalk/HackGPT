"""Generic Streamlit generation-option rendering."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import streamlit as st

from hackgpt.providers.types import (
    OptionKind,
    OptionSchema,
    OptionValidationError,
    OptionValue,
    validate_options,
)


def option_widget_key(provider_id: str, model_id: str, option_key: str) -> str:
    """Build a stable widget key scoped to a provider and model."""
    return f"generation.{provider_id}.{model_id}.{option_key}"


def _reset_invalid_widget_value(option: OptionSchema, key: str) -> None:
    if key not in st.session_state:
        return
    try:
        option.validate(st.session_state[key])
    except OptionValidationError:
        del st.session_state[key]


def _render_option(option: OptionSchema, key: str) -> OptionValue:
    help_text = option.description or None
    if option.kind is OptionKind.BOOLEAN:
        return st.checkbox(
            option.label,
            value=cast(bool, option.default),
            help=help_text,
            key=key,
        )
    if option.kind is OptionKind.INTEGER:
        return int(
            st.number_input(
                option.label,
                min_value=cast(int | None, option.minimum),
                max_value=cast(int | None, option.maximum),
                value=cast(int, option.default),
                step=cast(int | None, option.step),
                help=help_text,
                key=key,
            )
        )
    if option.kind is OptionKind.FLOAT:
        return float(
            st.number_input(
                option.label,
                min_value=cast(float | None, option.minimum),
                max_value=cast(float | None, option.maximum),
                value=cast(float, option.default),
                step=cast(float | None, option.step),
                help=help_text,
                key=key,
            )
        )
    if option.kind is OptionKind.SELECT:
        choice_labels = {choice.value: choice.label for choice in option.choices}
        values = tuple(choice_labels)
        selected = st.selectbox(
            option.label,
            options=values,
            index=values.index(option.default),
            format_func=choice_labels.__getitem__,
            help=help_text,
            key=key,
        )
        if selected is None:
            raise OptionValidationError(f"{option.label} requires a value.")
        return selected
    return st.text_input(
        option.label,
        value=cast(str, option.default),
        help=help_text,
        key=key,
    )


def render_option_fields(
    schema: Sequence[OptionSchema],
    *,
    provider_id: str,
    model_id: str,
) -> dict[str, OptionValue]:
    """Render and validate provider-defined fields without provider branching."""
    values: dict[str, OptionValue] = {}
    defaults = {option.key: option.default for option in schema}
    for option in schema:
        key = option_widget_key(provider_id, model_id, option.key)
        condition = option.visible_when
        if condition is not None:
            dependency = values.get(
                condition.option_key, defaults[condition.option_key]
            )
            if dependency != condition.equals:
                if key in st.session_state:
                    del st.session_state[key]
                continue
        _reset_invalid_widget_value(option, key)
        values[option.key] = _render_option(option, key)
    return validate_options(schema, values)
