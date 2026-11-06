"""Render the provider-neutral conversation and streamed responses."""

import streamlit as st

from hackgpt.config import APP_FOOTER, APP_NAME
from hackgpt.models import ChatConfiguration, PromptSelection
from hackgpt.providers.types import ChatMessage, ChatRequest, ProviderError
from hackgpt.state import get_chat_messages


def _render_setup_prompt(prompt: str) -> None:
    """Show the effective setup prompt above the conversation."""
    with st.chat_message("system"):
        st.caption("Setup prompt")
        st.markdown(prompt)


def _render_history(messages: list[ChatMessage]) -> None:
    """Render all persisted user and assistant messages."""
    for message in messages:
        with st.chat_message(message.role):
            st.markdown(message.content)


def _stream_assistant_response(
    configuration: ChatConfiguration,
    messages: list[ChatMessage],
) -> str:
    """Stream one provider response into the current chat container."""
    placeholder = st.empty()
    full_response = ""
    request = ChatRequest(
        model=configuration.model,
        system_prompt=st.session_state["setup_prompt"],
        messages=tuple(messages),
        options=configuration.options,
        request_id=st.session_state["session_identifier"],
    )
    try:
        for delta in configuration.provider.stream_chat(request):
            full_response += delta
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)
    except ProviderError as error:
        placeholder.error(str(error))
    return full_response


def render_chat(
    configuration: ChatConfiguration,
    prompt_selection: PromptSelection,
) -> None:
    """Render chat history, accept a prompt, and stream its response."""
    with st.container():
        st.header(f"{APP_NAME}: {prompt_selection.category}")
        st.caption(f"{configuration.provider.label} · {configuration.model.label}")
        st.write(APP_FOOTER)

        messages = get_chat_messages()
        _render_setup_prompt(st.session_state["setup_prompt"])
        _render_history(messages)

        if prompt := st.chat_input("What is up?"):
            user_message = ChatMessage(role="user", content=prompt)
            messages.append(user_message)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = _stream_assistant_response(configuration, messages)
            if response:
                messages.append(ChatMessage(role="assistant", content=response))


def render_unconfigured_chat(prompt_selection: PromptSelection) -> None:
    """Render a useful empty state when no provider configuration is usable."""
    st.header(f"{APP_NAME}: {prompt_selection.category}")
    st.write(APP_FOOTER)
    st.info("Configure a provider in the environment to start chatting.")
