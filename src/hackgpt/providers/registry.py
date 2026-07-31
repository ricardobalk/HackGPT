"""Construct the application provider registry."""

from collections.abc import Mapping

from hackgpt.providers.anthropic import AnthropicProvider
from hackgpt.providers.ollama import OllamaProvider
from hackgpt.providers.openai import OpenAIProvider
from hackgpt.providers.types import ChatProvider


def build_provider_registry() -> Mapping[str, ChatProvider]:
    """Return all supported providers in display order."""
    providers: tuple[ChatProvider, ...] = (
        OpenAIProvider(),
        AnthropicProvider(),
        OllamaProvider(),
    )
    return {provider.provider_id: provider for provider in providers}
