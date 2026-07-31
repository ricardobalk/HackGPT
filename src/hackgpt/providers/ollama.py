"""Ollama chat provider adapter."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any

from ollama import Client, RequestError, ResponseError

from hackgpt.providers.types import (
    Capability,
    ChatRequest,
    ModelDiscoveryError,
    ModelInfo,
    ModelUnavailableError,
    OptionChoice,
    OptionKind,
    OptionSchema,
    ProviderError,
    validate_options,
)


def _context_length(model_info: dict[str, Any] | Any) -> int:
    if not model_info:
        return 32_768
    lengths = [
        value
        for key, value in model_info.items()
        if key.endswith(".context_length") and isinstance(value, int)
    ]
    return max(lengths, default=32_768)


def _model_capabilities(capabilities: Sequence[str]) -> frozenset[Capability]:
    mapped = {Capability.STREAMING, Capability.SYSTEM_PROMPT}
    if "tools" in capabilities:
        mapped.add(Capability.TOOLS)
    if "vision" in capabilities:
        mapped.add(Capability.IMAGE_INPUT)
    if "thinking" in capabilities:
        mapped.add(Capability.REASONING)
    if "completion" in capabilities:
        mapped.add(Capability.STRUCTURED_OUTPUT)
    return frozenset(mapped)


def _option_schema(
    model_id: str,
    capabilities: Sequence[str],
    context_length: int,
) -> tuple[OptionSchema, ...]:
    maximum = max(512, context_length)
    options = [
        OptionSchema(
            key="max_tokens",
            kind=OptionKind.INTEGER,
            label="Maximum output tokens",
            default=min(2048, maximum),
            minimum=1,
            maximum=maximum,
            step=128,
        ),
        OptionSchema(
            key="temperature",
            kind=OptionKind.FLOAT,
            label="Temperature",
            default=0.8,
            minimum=0.0,
            maximum=2.0,
            step=0.1,
        ),
        OptionSchema(
            key="top_p",
            kind=OptionKind.FLOAT,
            label="Top P",
            default=0.9,
            minimum=0.0,
            maximum=1.0,
            step=0.05,
        ),
        OptionSchema(
            key="top_k",
            kind=OptionKind.INTEGER,
            label="Top K",
            default=40,
            minimum=1,
            maximum=200,
            step=1,
        ),
        OptionSchema(
            key="repeat_penalty",
            kind=OptionKind.FLOAT,
            label="Repeat penalty",
            default=1.1,
            minimum=0.0,
            maximum=2.0,
            step=0.05,
        ),
        OptionSchema(
            key="seed",
            kind=OptionKind.INTEGER,
            label="Seed",
            default=0,
            description="Zero lets Ollama choose a seed.",
            minimum=0,
            maximum=2_147_483_647,
            step=1,
        ),
    ]
    if "thinking" in capabilities:
        if model_id.lower().startswith("gpt-oss"):
            options.append(
                OptionSchema(
                    key="think",
                    kind=OptionKind.SELECT,
                    label="Thinking effort",
                    default="medium",
                    choices=tuple(
                        OptionChoice(value, value.title())
                        for value in ("low", "medium", "high")
                    ),
                )
            )
        else:
            options.append(
                OptionSchema(
                    key="think",
                    kind=OptionKind.BOOLEAN,
                    label="Thinking",
                    default=False,
                    description="Enable the model's separate thinking output.",
                )
            )
    return tuple(options)


class OllamaProvider:
    """Discover locally available Ollama models and stream chat completions."""

    provider_id = "ollama"
    label = "Ollama"
    configuration_help = (
        "Start Ollama locally or set OLLAMA_HOST for a different endpoint."
    )

    def __init__(self, client: Any | None = None) -> None:
        self._host = os.getenv("OLLAMA_HOST") or None
        self._api_key = os.getenv("OLLAMA_API_KEY") or None
        self._client = client
        self._available_model_ids: set[str] = set()

    def is_configured(self) -> bool:
        return True

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        headers = (
            {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        )
        self._client = Client(host=self._host, headers=headers)
        return self._client

    def discover_models(self) -> Sequence[ModelInfo]:
        try:
            listed = self._get_client().list().models
            discovered: list[ModelInfo] = []
            for listed_model in listed:
                model_id = listed_model.model
                if not model_id:
                    continue
                details = self._get_client().show(model_id)
                capabilities = tuple(details.capabilities or ())
                if "completion" not in capabilities:
                    continue
                context_length = _context_length(details.modelinfo)
                family = getattr(listed_model.details, "family", None)
                discovered.append(
                    ModelInfo(
                        id=model_id,
                        label=model_id,
                        description=(
                            f"Ollama model from the {family} family."
                            if family
                            else "Available through Ollama."
                        ),
                        capabilities=_model_capabilities(capabilities),
                        options=_option_schema(model_id, capabilities, context_length),
                    )
                )
        except (RequestError, ResponseError) as error:
            raise ModelDiscoveryError(
                f"Ollama model discovery failed: {error}. {self.configuration_help}"
            ) from error

        if not discovered:
            raise ModelDiscoveryError(
                "Ollama returned no installed chat models. Pull a model first."
            )
        discovered.sort(key=lambda model: model.id)
        self._available_model_ids = {model.id for model in discovered}
        return tuple(discovered)

    def stream_chat(self, request: ChatRequest) -> Iterator[str]:
        if (
            self._available_model_ids
            and request.model.id not in self._available_model_ids
        ):
            raise ModelUnavailableError(
                f"Ollama model '{request.model.id}' is no longer installed."
            )
        options = validate_options(request.model.options, request.options)
        think = options.pop("think", None)
        ollama_options = {
            "num_predict": options.pop("max_tokens"),
            **options,
        }
        messages = [
            {"role": "system", "content": request.system_prompt},
            *(
                {"role": message.role, "content": message.content}
                for message in request.messages
            ),
        ]
        try:
            stream = self._get_client().chat(
                model=request.model.id,
                messages=messages,
                options=ollama_options,
                think=think,
                stream=True,
            )
            for part in stream:
                if part.message.content:
                    yield part.message.content
        except (RequestError, ResponseError) as error:
            raise ProviderError(f"Ollama request failed: {error}") from error
