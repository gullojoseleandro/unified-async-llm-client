from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from anthropic import (
    APIConnectionError as AnthropicConnectionError,
    APITimeoutError as AnthropicTimeoutError,
    AsyncAnthropic,
    AuthenticationError as AnthropicAuthenticationError,
    RateLimitError as AnthropicRateLimitError,
)
from openai import (
    APIConnectionError as OpenAIConnectionError,
    APITimeoutError as OpenAITimeoutError,
    AsyncOpenAI,
    AuthenticationError as OpenAIAuthenticationError,
    RateLimitError as OpenAIRateLimitError,
)

from schemas import ChatMessage, ModelConfig, ModelResponse


class LLMClientError(Exception):
    """A provider failure with a stable, serializable error shape."""

    def __init__(self, provider: str, code: str, message: str, retryable: bool):
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.message = message
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


class LLMStreamError(LLMClientError):
    """Controlled failure raised while creating or consuming a text stream."""


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Generate a complete response without blocking the event loop."""

    @abstractmethod
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Yield response text chunks asynchronously."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release the provider SDK's underlying HTTP resources."""

    async def __aenter__(self) -> "BaseLLMClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.aclose()

    def _error(self, error: Exception, *, streaming: bool) -> LLMClientError:
        error_type = LLMStreamError if streaming else LLMClientError
        status_code = getattr(error, "status_code", None)
        if isinstance(error, (OpenAIRateLimitError, AnthropicRateLimitError)) or status_code == 429:
            code, retryable = "rate_limit", True
        elif isinstance(error, (OpenAIAuthenticationError, AnthropicAuthenticationError)) or status_code == 401:
            code, retryable = "authentication_error", False
        elif isinstance(
            error,
            (
                OpenAIConnectionError,
                OpenAITimeoutError,
                AnthropicConnectionError,
                AnthropicTimeoutError,
                ConnectionError,
                TimeoutError,
            ),
        ):
            code, retryable = "network_error", True
        else:
            code = "provider_error"
            retryable = status_code in {408, 500, 502, 503, 504}
        return error_type(
            provider=self.config.provider,
            code=code,
            message=str(error) or error.__class__.__name__,
            retryable=retryable,
        )


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, sdk_client: Any | None = None):
        super().__init__(config)
        self._client = sdk_client or AsyncOpenAI(**_api_key_kwargs(config))

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[message.model_dump() for message in messages],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as error:
            raise self._error(error, streaming=False) from error

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return ModelResponse(
            content=choice.message.content or "",
            model=response.model,
            provider="openai",
            usage=_usage(getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)),
        )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        response = None
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[message.model_dump() for message in messages],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            async for chunk in response:
                for choice in chunk.choices:
                    if choice.delta.content:
                        yield choice.delta.content
        except Exception as error:
            raise self._error(error, streaming=True) from error
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if close is not None:
                    result = close()
                    if hasattr(result, "__await__"):
                        await result

    async def aclose(self) -> None:
        await self._client.close()


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, sdk_client: Any | None = None):
        super().__init__(config)
        self._client = sdk_client or AsyncAnthropic(**_api_key_kwargs(config))

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        system, provider_messages = _anthropic_messages(messages)
        try:
            response = await self._client.messages.create(
                model=self.config.model,
                messages=provider_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **({"system": system} if system else {}),
            )
        except Exception as error:
            raise self._error(error, streaming=False) from error

        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return ModelResponse(
            content=text,
            model=response.model,
            provider="anthropic",
            usage=_usage(getattr(response.usage, "input_tokens", None), getattr(response.usage, "output_tokens", None)),
        )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        system, provider_messages = _anthropic_messages(messages)
        try:
            async with self._client.messages.stream(
                model=self.config.model,
                messages=provider_messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **({"system": system} if system else {}),
            ) as response:
                async for text in response.text_stream:
                    yield text
        except Exception as error:
            raise self._error(error, streaming=True) from error

    async def aclose(self) -> None:
        await self._client.close()


class AsyncLLMManager:
    """Select one async provider implementation from validated configuration."""

    def __init__(self, config: ModelConfig):
        clients = {"openai": OpenAIClient, "anthropic": AnthropicClient}
        try:
            self.client = clients[config.provider](config)
        except Exception as error:
            raise LLMClientError(config.provider, "configuration_error", str(error), False) from error

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return await self.client.generate(messages)

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        return self.client.stream(messages)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncLLMManager":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.aclose()


def _api_key_kwargs(config: ModelConfig) -> dict[str, str]:
    return {"api_key": config.api_key} if config.api_key else {}


def _usage(input_tokens: int | None, output_tokens: int | None) -> dict[str, int]:
    usage: dict[str, int] = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    return usage


def _anthropic_messages(messages: Sequence[ChatMessage]) -> tuple[str, list[dict[str, str]]]:
    """Anthropic accepts system instructions separately from user/assistant turns."""
    system_text = "\n\n".join(message.content for message in messages if message.role == "system")
    conversation = [
        {"role": message.role, "content": message.content}
        for message in messages
        if message.role != "system"
    ]
    return system_text, conversation
