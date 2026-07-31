"""Typed contracts shared by chat providers and the user interface."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

type OptionValue = bool | int | float | str


class Capability(StrEnum):
    """Features a provider model can expose to provider-neutral consumers."""

    STREAMING = "streaming"
    SYSTEM_PROMPT = "system_prompt"
    TOOLS = "tools"
    STRUCTURED_OUTPUT = "structured_output"
    IMAGE_INPUT = "image_input"
    REASONING = "reasoning"


class OptionKind(StrEnum):
    """Input controls supported by the generic option renderer."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    SELECT = "select"
    TEXT = "text"


@dataclass(frozen=True)
class OptionChoice:
    """A labelled value accepted by a select option."""

    value: OptionValue
    label: str


@dataclass(frozen=True)
class OptionCondition:
    """Show an option only when another option has the expected value."""

    option_key: str
    equals: OptionValue


@dataclass(frozen=True)
class OptionSchema:
    """Provider-supplied schema for one generation option."""

    key: str
    kind: OptionKind
    label: str
    default: OptionValue
    description: str = ""
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    choices: tuple[OptionChoice, ...] = ()
    visible_when: OptionCondition | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Option keys cannot be empty.")
        if self.kind is OptionKind.SELECT and not self.choices:
            raise ValueError(f"Select option '{self.key}' requires choices.")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError(f"Option '{self.key}' has an invalid range.")
        self.validate(self.default)

    def validate(self, value: OptionValue) -> OptionValue:
        """Validate and return one value according to this schema."""
        expected_types: dict[OptionKind, tuple[type[object], ...]] = {
            OptionKind.BOOLEAN: (bool,),
            OptionKind.INTEGER: (int,),
            OptionKind.FLOAT: (float,),
            OptionKind.SELECT: (bool, int, float, str),
            OptionKind.TEXT: (str,),
        }
        if type(value) not in expected_types[self.kind]:
            raise OptionValidationError(
                f"{self.label} must be a valid {self.kind.value} value."
            )
        if self.choices and value not in {choice.value for choice in self.choices}:
            raise OptionValidationError(f"{self.label} has an unsupported value.")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.minimum is not None and value < self.minimum:
                raise OptionValidationError(
                    f"{self.label} must be at least {self.minimum}."
                )
            if self.maximum is not None and value > self.maximum:
                raise OptionValidationError(
                    f"{self.label} must be at most {self.maximum}."
                )
        return value


@dataclass(frozen=True)
class ModelInfo:
    """A discovered chat model and its provider-specific configuration schema."""

    id: str
    label: str
    description: str = ""
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    options: tuple[OptionSchema, ...] = ()


@dataclass(frozen=True)
class ChatMessage:
    """A provider-neutral text chat message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {self.role}")
        if not self.content:
            raise ValueError("Chat message content cannot be empty.")


@dataclass(frozen=True)
class ChatRequest:
    """Normalised input for a provider streaming request."""

    model: ModelInfo
    system_prompt: str
    messages: tuple[ChatMessage, ...]
    options: Mapping[str, OptionValue]
    request_id: str


class ProviderError(RuntimeError):
    """Base error suitable for display to an application user."""


class ProviderConfigurationError(ProviderError):
    """Raised when required provider configuration is missing."""


class ModelDiscoveryError(ProviderError):
    """Raised when a provider cannot discover its available models."""


class ModelUnavailableError(ProviderError):
    """Raised when a requested model is not in the provider model list."""


class OptionValidationError(ProviderError, ValueError):
    """Raised when generation options do not match their schema."""


@runtime_checkable
class ChatProvider(Protocol):
    """Minimal interface implemented by every chat provider adapter."""

    provider_id: str
    label: str
    configuration_help: str

    def is_configured(self) -> bool:
        """Return whether environment configuration is sufficient for discovery."""
        ...

    def discover_models(self) -> Sequence[ModelInfo]:
        """Return currently available models with capabilities and option schemas."""
        ...

    def stream_chat(self, request: ChatRequest) -> Iterator[str]:
        """Yield text response deltas for a validated request."""
        ...


def normalise_messages(messages: Sequence[ChatMessage]) -> tuple[ChatMessage, ...]:
    """Validate provider-neutral messages and freeze them for a request."""
    return tuple(ChatMessage(message.role, message.content) for message in messages)


def validate_options(
    schema: Sequence[OptionSchema],
    values: Mapping[str, OptionValue],
) -> dict[str, OptionValue]:
    """Apply defaults and validate values against a model option schema."""
    definitions = {option.key: option for option in schema}
    unknown = set(values).difference(definitions)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise OptionValidationError(f"Unsupported generation option(s): {names}.")

    validated: dict[str, OptionValue] = {}
    for option in schema:
        condition = option.visible_when
        if condition is not None:
            dependency = values.get(
                condition.option_key,
                definitions[condition.option_key].default,
            )
            if dependency != condition.equals:
                continue
        value = values.get(option.key, option.default)
        validated[option.key] = option.validate(value)
    return validated
