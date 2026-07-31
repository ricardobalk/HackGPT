"""OpenAI Responses API integration."""

from collections.abc import Iterator, Sequence

from openai import OpenAI, OpenAIError

from hackgpt.models import ChatMessage


class ChatServiceError(RuntimeError):
    """Raised when OpenAI cannot generate a chat response."""


class OpenAIChatService:
    """Generate streamed chat responses with the OpenAI Responses API."""

    def __init__(self, api_key: str) -> None:
        """Create a service using the supplied OpenAI API key."""
        self._client = OpenAI(api_key=api_key)

    def stream_response(
        self,
        *,
        model: str,
        instructions: str,
        messages: Sequence[ChatMessage],
        max_output_tokens: int,
        reasoning_effort: str,
        safety_identifier: str,
    ) -> Iterator[str]:
        """Yield text deltas for a single assistant response."""
        try:
            stream = self._client.responses.create(
                model=model,
                instructions=instructions,
                input=list(messages),
                max_output_tokens=max_output_tokens,
                reasoning={"effort": reasoning_effort},
                safety_identifier=safety_identifier,
                stream=True,
            )
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
        except OpenAIError as error:
            raise ChatServiceError(str(error)) from error
