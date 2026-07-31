"""Streamlit entry point for HackGPT."""

import streamlit as st

from hackgpt.config import PAGE_CONFIG
from hackgpt.openai_service import OpenAIChatService
from hackgpt.prompts import load_setup_prompts
from hackgpt.state import initialize_session_state
from hackgpt.ui.chat import render_chat
from hackgpt.ui.sidebar import render_sidebar


def main() -> None:
    """Configure and render the HackGPT application."""
    st.set_page_config(**PAGE_CONFIG)
    initialize_session_state()

    prompt_selection = render_sidebar(load_setup_prompts())
    chat_service = OpenAIChatService(api_key=st.secrets["OPENAI_API_KEY"])
    render_chat(chat_service, prompt_selection)


if __name__ == "__main__":
    main()
