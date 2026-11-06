"""Construct the application provider registry."""

from collections.abc import Mapping, Sequence

from hackgpt.providers.anthropic import AnthropicProvider
from hackgpt.providers.ollama import OllamaProvider
from hackgpt.providers.openai import OpenAIProvider
from hackgpt.providers.types import (
    ChatProvider,
    ModelInfo,
    ModelUnavailableError,
    ProviderConfigurationError,
)


def build_provider_registry() -> Mapping[str, ChatProvider]:
    """Return all supported providers in display order."""
    providers: tuple[ChatProvider, ...] = (
        OpenAIProvider(),
        AnthropicProvider(),
        OllamaProvider(),
    )
    return {provider.provider_id: provider for provider in providers}


def get_provider(
    providers: Mapping[str, ChatProvider], provider_id: str
) -> ChatProvider:
    """Return a selected provider or a clear user-facing error."""
    try:
        return providers[provider_id]
    except KeyError as error:
        raise ProviderConfigurationError(
            f"Provider '{provider_id}' is not available in this application."
        ) from error


def get_model(models: Sequence[ModelInfo], model_id: str) -> ModelInfo:
    """Return a selected discovered model or a clear user-facing error."""
    try:
        return next(model for model in models if model.id == model_id)
    except StopIteration as error:
        raise ModelUnavailableError(
            f"Model '{model_id}' is no longer available from this provider."
        ) from error
