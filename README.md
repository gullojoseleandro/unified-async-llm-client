# Unified Async LLM Client

A small Python 3.12 example that sends the same prompt through OpenAI or Anthropic using their official asynchronous SDKs. The default run prints a normal answer and then streams a second answer; automated tests mock both SDKs and need no credentials.

## Quick start

From PowerShell, create the environment, install dependencies, and prepare local credentials:

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
Copy-Item .env.example .env
```

Add the API key for your chosen provider to `.env`. The default provider is OpenAI. To use Anthropic instead, set `LLM_PROVIDER=anthropic` or pass `--provider anthropic`. Keep `.env` private; it is excluded from Git.

Run the example:

```powershell
python main.py
```

This asks **¿Qué es la entropía?** once for a complete response and again as a text stream. Choose a single mode when needed:

```powershell
python main.py --provider openai --mode normal
python main.py --provider anthropic --mode stream
```

## Configuration and errors

`OPENAI_MODEL` and `ANTHROPIC_MODEL` override the model IDs. The default Anthropic model is Claude Haiku 4.5 (`claude-haiku-4-5-20251001`). Temperature is validated from 0 to 2 for OpenAI and 0 to 1 for Anthropic. The configured value is forwarded to the provider; newer Anthropic models may reject the parameter, so use a compatible model if that occurs.

Expected provider and configuration failures are printed to stderr as one JSON object with `provider`, `code`, `message`, and `retryable` fields. Codes are stable across both provider adapters:

| Code | Meaning | Retryable |
| --- | --- | --- |
| `configuration_error` | Invalid local model configuration or missing SDK credentials | No |
| `authentication_error` | Provider rejected the API key | No |
| `rate_limit` | Provider rate limit reached | Yes |
| `network_error` | Connection or timeout failure | Yes |
| `provider_error` | Other provider error; selected transient HTTP statuses are retryable | Depends on status |

Streaming failures use the typed `LLMStreamError` exception and the same JSON fields. Retries are reported, not automatically performed.

## Tests

Run offline tests and a syntax check with the Python 3.12 environment:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q schemas.py llm_client.py main.py tests
python main.py --help
```

Tests use mocked SDK objects; no provider credentials or network access are needed.

## Design notes

- `ChatMessage`, `ModelConfig`, and `ModelResponse` are Pydantic models. Temperature validation is provider-aware.
- `BaseLLMClient` defines async generation, text streaming, and resource cleanup. `AsyncLLMManager` selects the configured provider.
- Anthropic receives system instructions through its separate `system` argument, not as a conversation message. Its stream helper is an async context manager; OpenAI's async stream is closed in `finally`. Both SDK clients are closed by the manager.
