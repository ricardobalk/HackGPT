"""OpenAI Responses API provider adapter."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any, Final

from openai import OpenAI, OpenAIError

from hackgpt.providers.types import (
    Capability,
    ChatRequest,
    ModelDiscoveryError,
    ModelInfo,
    ModelUnavailableError,
    OptionChoice,
    OptionKind,
    OptionSchema,
    ProviderConfigurationError,
    ProviderError,
    validate_options,
)

_CHAT_MODEL_PREFIXES: Final = ("gpt-", "chatgpt-", "o1", "o3", "o4")
_NON_CHAT_MARKERS: Final = (
    "audio",
    "codex",
    "computer-use",
    "embedding",
    "image",
    "moderation",
    "realtime",
    "search-preview",
    "transcribe",
    "tts",
)


def _choices(*values: str) -> tuple[OptionChoice, ...]:
    return tuple(
        OptionChoice(value, value.replace("_", " ").title()) for value in values
    )


def _is_chat_model(model_id: str) -> bool:
    return model_id.startswith(_CHAT_MODEL_PREFIXES) and not any(
        marker in model_id for marker in _NON_CHAT_MARKERS
    )


def _max_output_tokens(model_id: str) -> int:
    if model_id.startswith("gpt-4o"):
        return 16_384
    if model_id.startswith("gpt-4.1"):
        return 32_768
    if model_id.startswith(("gpt-5", "o1", "o3", "o4")):
        return 128_000
    return 32_768


def _model_capabilities(model_id: str) -> frozenset[Capability]:
    capabilities = {
        Capability.STREAMING,
        Capability.SYSTEM_PROMPT,
        Capability.TOOLS,
        Capability.STRUCTURED_OUTPUT,
        Capability.IMAGE_INPUT,
    }
    if model_id.startswith(("gpt-5", "o1", "o3", "o4")):
        capabilities.add(Capability.REASONING)
    return frozenset(capabilities)


def _option_schema(model_id: str) -> tuple[OptionSchema, ...]:
    maximum = _max_output_tokens(model_id)
    options = [
        OptionSchema(
            key="max_tokens",
            kind=OptionKind.INTEGER,
            label="Maximum output tokens",
            default=min(2048, maximum),
            description="Maximum number of tokens generated for the response.",
            minimum=1,
            maximum=maximum,
            step=128,
        )
    ]
    if model_id.startswith("gpt-5.6"):
        options.extend(
            [
                OptionSchema(
                    key="reasoning_effort",
                    kind=OptionKind.SELECT,
                    label="Reasoning effort",
                    default="medium",
                    description="Higher effort can improve difficult answers.",
                    choices=_choices("none", "low", "medium", "high", "xhigh", "max"),
                ),
                OptionSchema(
                    key="reasoning_mode",
                    kind=OptionKind.SELECT,
                    label="Reasoning mode",
                    default="standard",
                    description="Pro mode uses more compute and can be slower.",
                    choices=_choices("standard", "pro"),
                ),
                OptionSchema(
                    key="verbosity",
                    kind=OptionKind.SELECT,
                    label="Response detail",
                    default="medium",
                    description="Default level of detail in the final response.",
                    choices=_choices("low", "medium", "high"),
                ),
            ]
        )
    elif model_id.startswith("gpt-5"):
        options.extend(
            [
                OptionSchema(
                    key="reasoning_effort",
                    kind=OptionKind.SELECT,
                    label="Reasoning effort",
                    default="medium",
                    choices=_choices("none", "low", "medium", "high", "xhigh"),
                ),
                OptionSchema(
                    key="verbosity",
                    kind=OptionKind.SELECT,
                    label="Response detail",
                    default="medium",
                    choices=_choices("low", "medium", "high"),
                ),
            ]
        )
    elif model_id.startswith(("o1", "o3", "o4")):
        options.append(
            OptionSchema(
                key="reasoning_effort",
                kind=OptionKind.SELECT,
                label="Reasoning effort",
                default="medium",
                choices=_choices("low", "medium", "high"),
            )
        )
    else:
        options.extend(
            [
                OptionSchema(
                    key="temperature",
                    kind=OptionKind.FLOAT,
                    label="Temperature",
                    default=1.0,
                    description="Controls response randomness.",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                ),
                OptionSchema(
                    key="top_p",
                    kind=OptionKind.FLOAT,
                    label="Top P",
                    default=1.0,
                    description="Limits sampling to a probability mass.",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                ),
            ]
        )
    return tuple(options)


class OpenAIProvider:
    """Discover OpenAI models and stream Responses API completions."""

    provider_id = "openai"
    label = "OpenAI"
    configuration_help = "Set OPENAI_API_KEY in the application environment."

    def __init__(self, client: Any | None = None) -> None:
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._base_url = os.getenv("OPENAI_BASE_URL")
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
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        else:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def discover_models(self) -> Sequence[ModelInfo]:
        try:
            result = self._get_client().models.list()
            model_ids = sorted(
                model.id for model in result.data if _is_chat_model(model.id)
            )
        except ProviderConfigurationError:
            raise
        except OpenAIError as error:
            raise ModelDiscoveryError(
                f"OpenAI model discovery failed: {error}"
            ) from error

        if not model_ids:
            raise ModelDiscoveryError(
                "OpenAI returned no compatible text chat models for this account."
            )
        self._available_model_ids = set(model_ids)
        return tuple(
            ModelInfo(
                id=model_id,
                label=model_id,
                description="Available through the OpenAI Responses API.",
                capabilities=_model_capabilities(model_id),
                options=_option_schema(model_id),
            )
            for model_id in model_ids
        )

    def stream_chat(self, request: ChatRequest) -> Iterator[str]:
        if (
            self._available_model_ids
            and request.model.id not in self._available_model_ids
        ):
            raise ModelUnavailableError(
                f"OpenAI model '{request.model.id}' is no longer available."
            )
        options = validate_options(request.model.options, request.options)
        request_options: dict[str, object] = {
            "model": request.model.id,
            "instructions": request.system_prompt,
            "input": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "max_output_tokens": options.pop("max_tokens"),
            "safety_identifier": request.request_id,
            "stream": True,
        }
        reasoning_effort = options.pop("reasoning_effort", None)
        reasoning_mode = options.pop("reasoning_mode", None)
        if reasoning_effort is not None:
            reasoning = {"effort": reasoning_effort}
            if reasoning_mode == "pro":
                reasoning["mode"] = "pro"
            request_options["reasoning"] = reasoning
        verbosity = options.pop("verbosity", None)
        if verbosity is not None:
            request_options["text"] = {"verbosity": verbosity}
        request_options.update(options)

        try:
            stream = self._get_client().responses.create(**request_options)
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
        except ProviderConfigurationError:
            raise
        except OpenAIError as error:
            raise ProviderError(f"OpenAI request failed: {error}") from error
