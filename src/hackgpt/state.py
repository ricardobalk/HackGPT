"""Helpers for initializing and mutating Streamlit session state."""

from uuid import uuid4

import streamlit as st

from hackgpt.config import DEFAULT_CUTOFF_DATE
from hackgpt.models import ChatMessage


def initialize_session_state() -> None:
    """Create all session values that must persist across Streamlit reruns."""
    defaults = {
        "chat_messages": [],
        "session_identifier": str(uuid4()),
        "use_cutoff_date": False,
        "date_cutoff": DEFAULT_CUTOFF_DATE,
        "date_cutoff_today": DEFAULT_CUTOFF_DATE,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_chat_messages() -> list[ChatMessage]:
    """Return the current conversation history."""
    return st.session_state["chat_messages"]


def clear_chat_history() -> None:
    """Remove all user and assistant messages from the current session."""
    st.session_state["chat_messages"] = []


def regenerate_session_identifier() -> None:
    """Replace the privacy-preserving API safety identifier."""
    st.session_state["session_identifier"] = str(uuid4())
