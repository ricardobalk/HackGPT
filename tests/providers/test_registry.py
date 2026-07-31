"""Tests for provider and model selection helpers."""

import pytest

from hackgpt.providers.registry import build_provider_registry, get_model, get_provider
from hackgpt.providers.types import (
    ModelInfo,
    ModelUnavailableError,
    ProviderConfigurationError,
)


def test_registry_contains_all_supported_providers() -> None:
    registry = build_provider_registry()

    assert tuple(registry) == ("openai", "anthropic", "ollama")
    assert get_provider(registry, "anthropic").label == "Anthropic Claude"


def test_provider_selection_rejects_unknown_provider() -> None:
    with pytest.raises(ProviderConfigurationError, match="unknown"):
        get_provider(build_provider_registry(), "unknown")


def test_model_selection_rejects_stale_model() -> None:
    with pytest.raises(ModelUnavailableError, match="removed-model"):
        get_model((ModelInfo(id="available", label="Available"),), "removed-model")
