"""Render the conversation and stream assistant responses."""

import streamlit as st

from hackgpt.config import APP_FOOTER
from hackgpt.models import ChatMessage, PromptSelection
from hackgpt.openai_service import ChatServiceError, OpenAIChatService
from hackgpt.state import get_chat_messages


def _render_history(messages: list[ChatMessage]) -> None:
    """Render all persisted user and assistant messages."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _stream_assistant_response(
    service: OpenAIChatService,
    messages: list[ChatMessage],
) -> str:
    """Stream one assistant response into the current chat container."""
    placeholder = st.empty()
    full_response = ""
    try:
        for delta in service.stream_response(
            model=st.session_state["openai_model"],
            instructions=st.session_state["setup_prompt"],
            messages=messages,
            max_output_tokens=st.session_state["max_tokens"],
            reasoning_effort=st.session_state["reasoning_effort"],
            safety_identifier=st.session_state["session_identifier"],
        ):
            full_response += delta
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)
    except ChatServiceError as error:
        placeholder.error(f"OpenAI API error: {error}")
    return full_response


def render_chat(
    service: OpenAIChatService,
    prompt_selection: PromptSelection,
) -> None:
    """Render chat history, accept a prompt, and stream its response."""
    with st.container():
        st.header(prompt_selection.label)
        st.write(APP_FOOTER)

        messages = get_chat_messages()
        _render_history(messages)

        if prompt := st.chat_input("What is up?"):
            user_message: ChatMessage = {"role": "user", "content": prompt}
            messages.append(user_message)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = _stream_assistant_response(service, messages)
            if response:
                messages.append({"role": "assistant", "content": response})
