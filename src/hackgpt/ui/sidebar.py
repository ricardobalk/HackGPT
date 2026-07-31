"""Render and process the provider-neutral preferences sidebar."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import streamlit as st

from hackgpt.config import MAX_CUTOFF_DATE, MIN_CUTOFF_DATE
from hackgpt.models import (
    ChatConfiguration,
    PromptCategory,
    PromptSelection,
    SidebarSelection,
)
from hackgpt.prompts import CUSTOM_PROMPT, add_historical_context
from hackgpt.providers.registry import get_model, get_provider
from hackgpt.providers.types import (
    ChatProvider,
    ModelInfo,
    OptionValidationError,
    ProviderError,
)
from hackgpt.state import clear_chat_history, regenerate_session_identifier
from hackgpt.ui.options import render_option_fields


def _render_prompt_picker(
    prompts: dict[str, PromptCategory],
) -> PromptSelection:
    """Render setup-prompt controls and return the selected prompt."""
    category_key = st.selectbox(
        "Select a setup prompt category",
        prompts,
        format_func=lambda key: prompts[key]["label"],
        key="prompt.category",
    )
    if category_key is None:
        category_key = next(iter(prompts))
    options = prompts[category_key]["options"]
    option_key = st.selectbox(
        "Select a setup prompt",
        options,
        format_func=lambda key: options[key]["label"],
        key=f"prompt.option.{category_key}",
    )
    if option_key is None:
        option_key = next(iter(options))
    option = options[option_key]

    value = option["value"]
    if category_key == "custom":
        value = st.text_input(
            "Custom setup prompt",
            value=CUSTOM_PROMPT,
            key="prompt.custom.value",
        )

    return PromptSelection(
        category=prompts[category_key]["label"],
        label=option["label"],
        value=value,
    )


def _render_historical_context(prompt: PromptSelection) -> PromptSelection:
    """Render optional historical context controls for a setup prompt."""
    use_cutoff = st.checkbox(
        "Include knowledge cutoff",
        value=False,
        help="Ask the model to answer from the perspective of a chosen date.",
        key="prompt.cutoff.enabled",
    )
    st.session_state["use_cutoff_date"] = use_cutoff
    if not use_cutoff:
        return prompt

    cutoff = st.date_input(
        "Cutoff date",
        value=st.session_state["date_cutoff"],
        min_value=MIN_CUTOFF_DATE,
        max_value=MAX_CUTOFF_DATE,
        key="prompt.cutoff.date",
    )
    today_value = min(st.session_state["date_cutoff_today"], cutoff)
    today = st.date_input(
        "Today's date",
        value=today_value,
        min_value=MIN_CUTOFF_DATE,
        max_value=cutoff,
        help="Must be on or before the selected knowledge cutoff.",
        key="prompt.cutoff.today",
    )
    st.session_state["date_cutoff"] = cutoff
    st.session_state["date_cutoff_today"] = today
    return PromptSelection(
        category=prompt.category,
        label=prompt.label,
        value=add_historical_context(prompt.value, cutoff, today),
    )


def _discover_models(provider: ChatProvider, *, refresh: bool) -> Sequence[ModelInfo]:
    cache = cast(
        dict[str, Sequence[ModelInfo]], st.session_state["provider_model_cache"]
    )
    if refresh or provider.provider_id not in cache:
        cache[provider.provider_id] = provider.discover_models()
    return cache[provider.provider_id]


def _render_model_configuration(
    providers: Mapping[str, ChatProvider],
) -> ChatConfiguration | None:
    provider_ids = tuple(providers)
    selected_provider_id = st.selectbox(
        "Provider",
        provider_ids,
        format_func=lambda provider_id: providers[provider_id].label,
        key="provider.selection",
    )
    if selected_provider_id is None:
        st.error("No chat provider is selected.")
        return None
    provider = get_provider(providers, selected_provider_id)
    if not provider.is_configured():
        st.error(provider.configuration_help)
        return None

    refresh = st.button(
        "Refresh models",
        key=f"provider.models.refresh.{provider.provider_id}",
        help="Ask the provider for its currently available models.",
    )
    try:
        models = _discover_models(provider, refresh=refresh)
    except ProviderError as error:
        st.error(str(error))
        return None

    model_ids = tuple(model.id for model in models)
    model_key = f"provider.model.selection.{provider.provider_id}"
    if st.session_state.get(model_key) not in {*model_ids, None}:
        del st.session_state[model_key]
    model_labels = {model.id: model.label for model in models}
    selected_model_id = st.selectbox(
        "Model",
        model_ids,
        format_func=model_labels.__getitem__,
        key=model_key,
    )
    if selected_model_id is None:
        st.error("The selected provider has no available chat model.")
        return None
    model = get_model(models, selected_model_id)
    if model.description:
        st.caption(model.description)
    if model.capabilities:
        labels = ", ".join(
            capability.value.replace("_", " ")
            for capability in sorted(model.capabilities, key=lambda item: item.value)
        )
        st.caption(f"Capabilities: {labels}")

    try:
        options = render_option_fields(
            model.options,
            provider_id=provider.provider_id,
            model_id=model.id,
        )
    except OptionValidationError as error:
        st.error(f"Invalid model configuration: {error}")
        return None
    return ChatConfiguration(provider=provider, model=model, options=options)


def render_sidebar(
    prompts: dict[str, PromptCategory],
    providers: Mapping[str, ChatProvider],
) -> SidebarSelection:
    """Render all sidebar controls and return validated chat configuration."""
    with st.sidebar:
        st.title("Preferences")

        with st.expander("Model", expanded=True):
            chat = _render_model_configuration(providers)

        with st.expander("Setup prompt"):
            prompt = _render_prompt_picker(prompts)
            st.write("Tip: Change the setup prompt to shape future responses.")
            prompt = _render_historical_context(prompt)
            st.session_state["setup_prompt"] = prompt.value

        with st.expander("Chat options"):
            st.button(
                "Clear chat",
                help="Clear the chat history",
                use_container_width=True,
                on_click=clear_chat_history,
                key="chat.clear",
            )

        with st.expander("Provider configuration"):
            if chat is not None:
                st.success(f"{chat.provider.label} is ready.")
            else:
                st.info(
                    "Provider credentials and endpoints come from environment "
                    "variables."
                )
            st.text_input(
                "Session identifier",
                value=st.session_state["session_identifier"],
                type="password",
                disabled=True,
                help="A privacy-preserving request identifier.",
                key="provider.session_identifier.display",
            )
            st.button(
                "Generate new session identifier",
                on_click=regenerate_session_identifier,
                key="provider.session_identifier.regenerate",
            )

    return SidebarSelection(prompt=prompt, chat=chat)
