"""Shared data structures used by the application."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from hackgpt.providers.types import ChatProvider, ModelInfo, OptionValue


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

    category: str
    label: str
    value: str


@dataclass(frozen=True)
class ChatConfiguration:
    """Validated provider, model, and generation options for one chat rerun."""

    provider: ChatProvider
    model: ModelInfo
    options: Mapping[str, OptionValue]


@dataclass(frozen=True)
class SidebarSelection:
    """Complete result of rendering the application sidebar."""

    prompt: PromptSelection
    chat: ChatConfiguration | None
