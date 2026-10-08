import unittest
import json
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import openai
from pydantic import ValidationError

from llm_client import (
    AnthropicClient,
    AsyncLLMManager,
    LLMClientError,
    LLMStreamError,
    OpenAIClient,
)
from schemas import ChatMessage, ModelConfig, ModelResponse
from main import _config, _parser, main, run


def sdk_failure(provider, failure_kind):
    sdk = openai if provider == "openai" else anthropic
    if failure_kind == "rate_limit":
        return sdk.RateLimitError("too many requests", response=MagicMock(status_code=429), body=None)
    if failure_kind == "network":
        return sdk.APIConnectionError(message="connection failed", request=MagicMock())
    if failure_kind == "timeout":
        return sdk.APITimeoutError(request=MagicMock())
    return sdk.AuthenticationError("invalid API key", response=MagicMock(status_code=401), body=None)


class SchemaTests(unittest.TestCase):
    def test_models_validate_roles_and_provider(self):
        self.assertEqual(ChatMessage(role="user", content="hello").role, "user")
        with self.assertRaises(ValidationError):
            ChatMessage(role="unknown", content="hello")
        with self.assertRaises(ValidationError):
            ModelConfig(provider="unknown", model="x")
        response = ModelResponse(content="ok", model="model-x", provider="openai")
        self.assertEqual(response.content, "ok")

    def test_temperature_limits_depend_on_provider(self):
        self.assertEqual(ModelConfig(provider="openai", model="gpt-test", temperature=2).temperature, 2)
        with self.assertRaises(ValidationError):
            ModelConfig(provider="anthropic", model="claude-test", temperature=1.1)

    def test_anthropic_default_model_uses_haiku_4_5(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_config("anthropic").model, "claude-haiku-4-5-20251001")

    def test_cli_defaults_to_both_modes(self):
        self.assertEqual(_parser().parse_args([]).mode, "both")

    def test_cli_reports_invalid_model_configuration_as_json(self):
        error_output = StringIO()
        with patch("main.load_dotenv"), patch.dict("os.environ", {"OPENAI_MODEL": ""}, clear=True):
            with redirect_stderr(error_output):
                result = main(["--mode", "normal"])
        self.assertEqual(result, 1)
        payload = json.loads(error_output.getvalue())
        self.assertEqual(payload["code"], "configuration_error")
        self.assertFalse(payload["retryable"])


class OpenAIClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_maps_response_and_messages(self):
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                model="actual-model",
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            )
        ))), close=AsyncMock())
        client = OpenAIClient(ModelConfig(provider="openai", model="test-model"), sdk_client=sdk)
        response = await client.generate([ChatMessage(role="user", content="question")])
        self.assertEqual(response.content, "answer")
        self.assertEqual(response.usage, {"input_tokens": 3, "output_tokens": 2})
        self.assertEqual(sdk.chat.completions.create.await_args.kwargs["messages"], [
            {"role": "user", "content": "question"}
        ])

    async def test_stream_yields_text_and_closes_stream(self):
        class Stream:
            def __init__(self):
                self.closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if hasattr(self, "sent"):
                    raise StopAsyncIteration
                self.sent = True
                return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])

            async def close(self):
                self.closed = True

        stream = Stream()
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=stream))), close=AsyncMock())
        client = OpenAIClient(ModelConfig(provider="openai", model="test-model"), sdk_client=sdk)
        self.assertEqual([part async for part in client.stream([ChatMessage(role="user", content="hi")])], ["hi"])
        self.assertTrue(stream.closed)

    async def test_generate_wraps_provider_failure(self):
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("offline")))), close=AsyncMock())
        client = OpenAIClient(ModelConfig(provider="openai", model="test-model"), sdk_client=sdk)
        with self.assertRaises(LLMClientError) as raised:
            await client.generate([ChatMessage(role="user", content="hi")])
        self.assertEqual(raised.exception.provider, "openai")
        self.assertEqual(raised.exception.code, "provider_error")


class AnthropicClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_moves_system_messages_to_system_parameter(self):
        sdk = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="answer")], model="claude-test",
            usage=SimpleNamespace(input_tokens=4, output_tokens=1),
        ))), close=AsyncMock())
        client = AnthropicClient(ModelConfig(provider="anthropic", model="claude-test"), sdk_client=sdk)
        await client.generate([
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="question"),
            ChatMessage(role="system", content="Use plain language."),
        ])
        args = sdk.messages.create.await_args.kwargs
        self.assertEqual(args["system"], "Be concise.\n\nUse plain language.")
        self.assertEqual(args["messages"], [{"role": "user", "content": "question"}])

    async def test_stream_wraps_failures_in_typed_error(self):
        class Manager:
            async def __aenter__(self):
                raise RuntimeError("offline")

            async def __aexit__(self, *args):
                return False

        sdk = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: Manager()), close=AsyncMock())
        client = AnthropicClient(ModelConfig(provider="anthropic", model="claude-test"), sdk_client=sdk)
        with self.assertRaises(LLMStreamError) as raised:
            async for _ in client.stream([ChatMessage(role="user", content="hi")]):
                pass
        self.assertEqual(raised.exception.provider, "anthropic")


    async def test_stream_yields_text_and_exits_stream_context(self):
        class Stream:
            async def __aenter__(self):
                self.text_stream = self._texts()
                self.closed = False
                return self

            async def __aexit__(self, *args):
                self.closed = True
                return False

            async def _texts(self):
                yield "chunk"

        stream = Stream()
        sdk = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: stream), close=AsyncMock())
        client = AnthropicClient(ModelConfig(provider="anthropic", model="claude-test"), sdk_client=sdk)
        self.assertEqual([part async for part in client.stream([ChatMessage(role="user", content="hi")])], ["chunk"])
        self.assertTrue(stream.closed)


class ManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_mode_generates_and_streams_in_order(self):
        manager = AsyncMock()
        manager.generate.return_value = ModelResponse(content="answer", model="gpt-test", provider="openai")
        stream_calls = []

        async def chunks(messages):
            stream_calls.append(messages)
            yield "streamed"

        manager.stream = chunks
        with patch("main.AsyncLLMManager", return_value=manager):
            self.assertEqual(await run("openai", "both"), 0)
        manager.generate.assert_awaited_once()
        self.assertEqual(len(stream_calls), 1)
        manager.aclose.assert_awaited_once()

    async def test_selects_provider_and_closes_client(self):
        with patch("llm_client.OpenAIClient") as openai:
            openai.return_value.aclose = AsyncMock()
            manager = AsyncLLMManager(ModelConfig(provider="openai", model="gpt-test"))
            self.assertIs(manager.client, openai.return_value)
            await manager.aclose()
            openai.return_value.aclose.assert_awaited_once()

    async def test_nonstreaming_errors_are_structured(self):
        error = LLMClientError(provider="openai", code="provider_error", message="failed", retryable=False)
        self.assertEqual(error.to_dict(), {
            "provider": "openai", "code": "provider_error", "message": "failed", "retryable": False
        })


class ProviderFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_classifies_rate_network_and_auth_failures(self):
        for provider, client_type in (("openai", OpenAIClient), ("anthropic", AnthropicClient)):
            for failure_kind, expected_code, retryable in (
                ("rate_limit", "rate_limit", True),
                ("network", "network_error", True),
                ("timeout", "network_error", True),
                ("authentication", "authentication_error", False),
            ):
                with self.subTest(provider=provider, failure_kind=failure_kind):
                    failure = sdk_failure(provider, failure_kind)
                    if provider == "openai":
                        sdk_client = SimpleNamespace(
                            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=failure))),
                            close=AsyncMock(),
                        )
                    else:
                        sdk_client = SimpleNamespace(
                            messages=SimpleNamespace(create=AsyncMock(side_effect=failure)),
                            close=AsyncMock(),
                        )
                    client = client_type(ModelConfig(provider=provider, model="test-model"), sdk_client=sdk_client)
                    with self.assertRaises(LLMClientError) as raised:
                        await client.generate([ChatMessage(role="user", content="hello")])
                    self.assertEqual(raised.exception.code, expected_code)
                    self.assertEqual(raised.exception.retryable, retryable)

    async def test_stream_classifies_rate_network_and_auth_failures(self):
        class FailingStream:
            def __init__(self, failure):
                self.failure = failure

            async def __aenter__(self):
                raise self.failure

            async def __aexit__(self, *args):
                return False

        for provider, client_type in (("openai", OpenAIClient), ("anthropic", AnthropicClient)):
            for failure_kind, expected_code, retryable in (
                ("rate_limit", "rate_limit", True),
                ("network", "network_error", True),
                ("timeout", "network_error", True),
                ("authentication", "authentication_error", False),
            ):
                with self.subTest(provider=provider, failure_kind=failure_kind):
                    failure = sdk_failure(provider, failure_kind)
                    if provider == "openai":
                        sdk_client = SimpleNamespace(
                            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=failure))),
                            close=AsyncMock(),
                        )
                    else:
                        sdk_client = SimpleNamespace(
                            messages=SimpleNamespace(stream=lambda **kwargs: FailingStream(failure)),
                            close=AsyncMock(),
                        )
                    client = client_type(ModelConfig(provider=provider, model="test-model"), sdk_client=sdk_client)
                    with self.assertRaises(LLMStreamError) as raised:
                        async for _ in client.stream([ChatMessage(role="user", content="hello")]):
                            pass
                    self.assertEqual(raised.exception.code, expected_code)
                    self.assertEqual(raised.exception.retryable, retryable)

    async def test_invalid_environment_config_is_controlled(self):
        error_output = StringIO()
        with patch.dict("os.environ", {"OPENAI_MODEL": ""}, clear=True), redirect_stderr(error_output):
            result = await run("openai", "normal")
        self.assertEqual(result, 1)
        payload = json.loads(error_output.getvalue())
        self.assertEqual(payload["provider"], "openai")
        self.assertEqual(payload["code"], "configuration_error")


if __name__ == "__main__":
    unittest.main()
