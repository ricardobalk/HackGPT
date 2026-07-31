"""Render and process the preferences sidebar."""

import streamlit as st

from hackgpt.config import (
    DEFAULT_CUTOFF_DATE,
    MAX_CUTOFF_DATE,
    MIN_CUTOFF_DATE,
    MODEL_DESCRIPTIONS,
    MODELS_WITH_ADVANCED_REASONING,
    MODELS_WITH_VERBOSITY,
    REASONING_EFFORTS,
    REASONING_MODES,
    VERBOSITY_LEVELS,
)
from hackgpt.models import PromptCategory, PromptSelection
from hackgpt.prompts import CUSTOM_PROMPT, add_historical_context
from hackgpt.state import clear_chat_history, regenerate_session_identifier


def _render_prompt_picker(
    prompts: dict[str, PromptCategory],
) -> PromptSelection:
    """Render setup-prompt controls and return the selected prompt."""
    category_key = st.selectbox(
        "Select a setup prompt category",
        prompts,
        format_func=lambda key: prompts[key]["label"],
    )
    options = prompts[category_key]["options"]
    option_key = st.selectbox(
        "Select a setup prompt",
        options,
        format_func=lambda key: options[key]["label"],
    )
    option = options[option_key]

    value = option["value"]
    if category_key == "custom":
        value = st.text_input("Custom setup prompt", value=CUSTOM_PROMPT)

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
    )
    st.session_state["use_cutoff_date"] = use_cutoff
    if not use_cutoff:
        return prompt

    cutoff = st.date_input(
        "Cutoff date",
        value=st.session_state["date_cutoff"],
        min_value=MIN_CUTOFF_DATE,
        max_value=MAX_CUTOFF_DATE,
    )
    today_value = min(st.session_state["date_cutoff_today"], cutoff)
    today = st.date_input(
        "Today's date",
        value=today_value,
        min_value=MIN_CUTOFF_DATE,
        max_value=cutoff,
        help="Must be on or before the selected knowledge cutoff.",
    )
    st.session_state["date_cutoff"] = cutoff
    st.session_state["date_cutoff_today"] = today
    return PromptSelection(
        category=prompt.category,
        label=prompt.label,
        value=add_historical_context(prompt.value, cutoff, today),
    )


def render_sidebar(prompts: dict[str, PromptCategory]) -> PromptSelection:
    """Render all sidebar controls and return the active setup prompt."""
    with st.sidebar:
        st.title("Preferences")

        with st.expander("Model"):
            model = st.selectbox("Select a model", MODEL_DESCRIPTIONS)
            st.session_state["openai_model"] = model
            st.write(MODEL_DESCRIPTIONS[model])

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
            )

        with st.expander("Response options"):
            st.session_state["max_tokens"] = st.slider(
                "Max tokens",
                min_value=512,
                max_value=8192,
                value=2048,
                step=16,
                help="The maximum number of tokens to generate in the response.",
            )
            if model in MODELS_WITH_ADVANCED_REASONING:
                st.session_state["reasoning_effort"] = st.select_slider(
                    "Reasoning effort",
                    options=REASONING_EFFORTS,
                    value="medium",
                    help=(
                        "Higher effort can improve difficult answers but increases "
                        "latency and token use."
                    ),
                )
                st.session_state["reasoning_mode"] = st.selectbox(
                    "Reasoning mode",
                    options=REASONING_MODES,
                    help=(
                        "Pro mode can improve difficult answers, but increases "
                        "latency and token usage."
                    ),
                )
            else:
                st.session_state["reasoning_effort"] = None
                st.session_state["reasoning_mode"] = None

            if model in MODELS_WITH_VERBOSITY:
                st.session_state["verbosity"] = st.select_slider(
                    "Response detail",
                    options=VERBOSITY_LEVELS,
                    value="medium",
                    help="Controls how detailed the model's final response is.",
                )
            else:
                st.session_state["verbosity"] = None

        with st.expander("API configuration"):
            st.text_input(
                "API token",
                value=st.secrets["OPENAI_API_KEY"],
                type="password",
                help="Manage API keys in the OpenAI API dashboard.",
                disabled=True,
            )
            st.text_input(
                "Session identifier",
                value=st.session_state["session_identifier"],
                type="password",
                disabled=True,
                help="A privacy-preserving identifier used by OpenAI safety systems.",
            )
            st.button(
                "Generate new session identifier",
                on_click=regenerate_session_identifier,
            )

    return prompt
