# Unified Async LLM Client

A Python 3.12 example for sending one prompt to OpenAI or Anthropic with official asynchronous SDKs. Run one initializer to prepare the environment, configure a provider, and launch both a normal response and a text stream.

## Quick start

### Windows

Double-click `start-project.bat`, or run this from the project directory:

```powershell
python bootstrap.py
```

The initializer uses an existing provider in `.env` when available; otherwise it asks you to select OpenAI or Anthropic. It uses `uv` to provision Python 3.12 when installed, or an existing Python 3.12 interpreter and `pip`. It installs dependencies, securely prompts for the selected provider key when missing, stores it in the ignored `.env`, and starts `main.py`. The key is never placed in a command argument or printed.

The default run asks **¿Qué es la entropía?** once for a complete response and again as a text stream. To select a provider or one mode explicitly:

```powershell
python bootstrap.py --provider anthropic
python bootstrap.py --provider openai --mode normal
```

Use `--setup-only` to prepare Python, install dependencies, and configure the key without making an API call. Use `--non-interactive` for scripts; supply the provider and have its API key in the environment or `.env`, or the initializer exits with guidance instead of prompting.

## Manual setup

If you prefer to manage the environment yourself:

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
Copy-Item .env.example .env
python main.py --provider openai
```

Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env`. You can also set `LLM_PROVIDER`, `OPENAI_MODEL`, and `ANTHROPIC_MODEL`. The default provider is OpenAI; the default Anthropic model is Claude Haiku 4.5 (`claude-haiku-4-5-20251001`). Keep real credentials in `.env`; it is excluded from Git.

## Configuration and errors

Temperature is validated from 0 to 2 for OpenAI and 0 to 1 for Anthropic. The configured value is forwarded to the provider; newer Anthropic models may reject this parameter, so choose a compatible model if that occurs.

Expected provider and configuration failures are printed to stderr as one JSON object with `provider`, `code`, `message`, and `retryable` fields:

| Code | Meaning | Retryable |
| --- | --- | --- |
| `configuration_error` | Invalid local settings or missing SDK credentials | No |
| `authentication_error` | Provider rejected the API key | No |
| `rate_limit` | Provider rate limit reached | Yes |
| `network_error` | Connection or timeout failure | Yes |
| `provider_error` | Other provider response; transient HTTP statuses may be retryable | Depends on status |

Streaming failures use the typed `LLMStreamError` with the same structured fields. The client reports retryability but does not automatically retry requests.

## Tests

Run the offline suite and syntax checks with Python 3.12:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q bootstrap.py schemas.py llm_client.py main.py tests
python bootstrap.py --help
```

Tests mock installers, terminal prompts, and SDK clients; they need neither a real API key nor provider network access.
