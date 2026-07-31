"""Streamlit entry point for HackGPT."""

from collections.abc import Mapping

import streamlit as st

from hackgpt.config import PAGE_CONFIG
from hackgpt.prompts import load_setup_prompts
from hackgpt.providers.registry import build_provider_registry
from hackgpt.providers.types import ChatProvider
from hackgpt.state import initialize_session_state
from hackgpt.ui.chat import render_chat, render_unconfigured_chat
from hackgpt.ui.sidebar import render_sidebar


@st.cache_resource
def get_provider_registry() -> Mapping[str, ChatProvider]:
    """Reuse provider clients and their discovered model state across reruns."""
    return build_provider_registry()


def main() -> None:
    """Configure and render the HackGPT application."""
    st.set_page_config(**PAGE_CONFIG)
    initialize_session_state()

    selection = render_sidebar(load_setup_prompts(), get_provider_registry())
    if selection.chat is None:
        render_unconfigured_chat(selection.prompt)
    else:
        render_chat(selection.chat, selection.prompt)


if __name__ == "__main__":
    main()
