"""Anthropic Claude Messages API provider adapter."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any

from anthropic import Anthropic, APIError

from hackgpt.providers.types import (
    Capability,
    ChatRequest,
    ModelDiscoveryError,
    ModelInfo,
    ModelUnavailableError,
    OptionChoice,
    OptionCondition,
    OptionKind,
    OptionSchema,
    ProviderConfigurationError,
    ProviderError,
    validate_options,
)


def _choice(value: str) -> OptionChoice:
    return OptionChoice(value, value.replace("_", " ").title())


def _supported(value: Any) -> bool:
    return bool(value is not None and getattr(value, "supported", False))


def _model_capabilities(model: Any) -> frozenset[Capability]:
    capabilities = {
        Capability.STREAMING,
        Capability.SYSTEM_PROMPT,
        Capability.TOOLS,
    }
    details = getattr(model, "capabilities", None)
    if details is None:
        return frozenset(capabilities)
    if _supported(details.image_input):
        capabilities.add(Capability.IMAGE_INPUT)
    if _supported(details.structured_outputs):
        capabilities.add(Capability.STRUCTURED_OUTPUT)
    if _supported(details.thinking):
        capabilities.add(Capability.REASONING)
    return frozenset(capabilities)


def _thinking_choices(model: Any) -> tuple[OptionChoice, ...]:
    details = getattr(model, "capabilities", None)
    thinking = getattr(details, "thinking", None)
    if thinking is None or not _supported(thinking):
        return ()
    choices = [_choice("disabled")]
    types = thinking.types
    if _supported(types.adaptive):
        choices.append(_choice("adaptive"))
    if _supported(types.enabled):
        choices.append(_choice("enabled"))
    return tuple(choices)


def _effort_choices(model: Any) -> tuple[OptionChoice, ...]:
    details = getattr(model, "capabilities", None)
    effort = getattr(details, "effort", None)
    if effort is None or not effort.supported:
        return ()
    return tuple(
        _choice(level)
        for level in ("low", "medium", "high", "xhigh", "max")
        if _supported(getattr(effort, level, None))
    )


def _option_schema(model: Any) -> tuple[OptionSchema, ...]:
    reported_maximum = getattr(model, "max_tokens", None)
    maximum = reported_maximum if reported_maximum and reported_maximum > 0 else 64_000
    options = [
        OptionSchema(
            key="max_tokens",
            kind=OptionKind.INTEGER,
            label="Maximum output tokens",
            default=min(4096, maximum),
            description="Maximum number of tokens generated for the response.",
            minimum=1,
            maximum=maximum,
            step=128,
        )
    ]
    thinking_choices = _thinking_choices(model)
    if thinking_choices:
        options.append(
            OptionSchema(
                key="thinking_mode",
                kind=OptionKind.SELECT,
                label="Thinking mode",
                default="disabled",
                description="Enable Claude's model-native thinking controls.",
                choices=thinking_choices,
            )
        )
        if any(choice.value == "enabled" for choice in thinking_choices):
            options.append(
                OptionSchema(
                    key="thinking_budget",
                    kind=OptionKind.INTEGER,
                    label="Thinking token budget",
                    default=min(4096, maximum - 1),
                    description="Token budget reserved for enabled thinking.",
                    minimum=1024,
                    maximum=maximum - 1,
                    step=1024,
                    visible_when=OptionCondition("thinking_mode", "enabled"),
                )
            )
        effort_choices = _effort_choices(model)
        if effort_choices:
            default_effort = next(
                (choice.value for choice in effort_choices if choice.value == "medium"),
                effort_choices[0].value,
            )
            options.append(
                OptionSchema(
                    key="effort",
                    kind=OptionKind.SELECT,
                    label="Thinking effort",
                    default=default_effort,
                    choices=effort_choices,
                    visible_when=OptionCondition("thinking_mode", "adaptive"),
                )
            )

    return tuple(options)


class AnthropicProvider:
    """Discover Claude models and stream Messages API completions."""

    provider_id = "anthropic"
    label = "Anthropic Claude"
    configuration_help = "Set ANTHROPIC_API_KEY in the application environment."

    def __init__(self, client: Any | None = None) -> None:
        self._api_key = os.getenv("ANTHROPIC_API_KEY")
        self._base_url = os.getenv("ANTHROPIC_BASE_URL")
        self._client = client
        self._available_model_ids: set[str] = set()

    def is_configured(self) -> bool:
        return self._client is not None or bool(self._api_key)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise ProviderConfigurationError(self.configuration_help)
        if self._base_url:
            self._client = Anthropic(api_key=self._api_key, base_url=self._base_url)
        else:
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def discover_models(self) -> Sequence[ModelInfo]:
        try:
            page = self._get_client().models.list(limit=1000)
            discovered = sorted(page.data, key=lambda model: model.id)
        except ProviderConfigurationError:
            raise
        except APIError as error:
            raise ModelDiscoveryError(
                f"Anthropic model discovery failed: {error}"
            ) from error

        if not discovered:
            raise ModelDiscoveryError(
                "Anthropic returned no Claude models for this account."
            )
        self._available_model_ids = {model.id for model in discovered}
        return tuple(
            ModelInfo(
                id=model.id,
                label=model.display_name,
                description=f"Anthropic model {model.id}.",
                capabilities=_model_capabilities(model),
                options=_option_schema(model),
            )
            for model in discovered
        )

    def stream_chat(self, request: ChatRequest) -> Iterator[str]:
        if (
            self._available_model_ids
            and request.model.id not in self._available_model_ids
        ):
            raise ModelUnavailableError(
                f"Anthropic model '{request.model.id}' is no longer available."
            )
        options = validate_options(request.model.options, request.options)
        max_tokens = options.pop("max_tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise ProviderError("Maximum output tokens must be an integer.")
        request_options: dict[str, object] = {
            "model": request.model.id,
            "system": request.system_prompt,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "max_tokens": max_tokens,
        }
        thinking_mode = options.pop("thinking_mode", "disabled")
        if thinking_mode == "adaptive":
            request_options["thinking"] = {"type": "adaptive"}
            effort = options.pop("effort", None)
            if effort is not None:
                request_options["output_config"] = {"effort": effort}
        elif thinking_mode == "enabled":
            budget = options.pop("thinking_budget")
            if not isinstance(budget, int) or isinstance(budget, bool):
                raise ProviderError("Thinking token budget must be an integer.")
            if budget >= max_tokens:
                raise ProviderError(
                    "Maximum output tokens must exceed the thinking token budget."
                )
            request_options["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
        request_options.update(options)

        try:
            with self._get_client().messages.stream(**request_options) as stream:
                yield from stream.text_stream
        except ProviderConfigurationError:
            raise
        except APIError as error:
            raise ProviderError(f"Anthropic request failed: {error}") from error
