"""Shared data structures used by the application."""

from dataclasses import dataclass
from typing import Literal, TypedDict


class ChatMessage(TypedDict):
    """A user or assistant message accepted by the Responses API."""

    role: Literal["user", "assistant"]
    content: str


class PromptOption(TypedDict):
    """A selectable setup prompt."""

    label: str
    value: str


class PromptCategory(TypedDict):
    """A named group of setup prompt options."""

    label: str
    options: dict[str, PromptOption]


@dataclass(frozen=True)
class PromptSelection:
    """The setup prompt currently selected in the sidebar."""

    label: str
    value: str
