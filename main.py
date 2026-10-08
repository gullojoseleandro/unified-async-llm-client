import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client import AsyncLLMManager, LLMClientError, LLMStreamError
from schemas import ChatMessage, ModelConfig

QUESTION = "¿Qué es la entropía?"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask an async OpenAI or Anthropic model.")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default=os.getenv("LLM_PROVIDER", "openai"))
    parser.add_argument("--mode", choices=("both", "normal", "stream"), default="both")
    return parser


def _config(provider: str) -> ModelConfig:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini") if provider == "openai" else os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    return ModelConfig(provider=provider, model=model, api_key=os.getenv(key_name))


async def run(provider: str, mode: str) -> int:
    manager: AsyncLLMManager | None = None
    try:
        manager = AsyncLLMManager(_config(provider))
        messages = [ChatMessage(role="user", content=QUESTION)]
        if mode in {"both", "normal"}:
            response = await manager.generate(messages)
            if mode == "both":
                print("Normal response:")
            print(response.content)
        if mode in {"both", "stream"}:
            if mode == "both":
                print("Streaming response:")
            async for text in manager.stream(messages):
                print(text, end="", flush=True)
            print()
        return 0
    except LLMClientError as error:
        print(json.dumps(error.to_dict(), ensure_ascii=False), file=sys.stderr)
        return 1
    except ValidationError as error:
        payload = {
            "provider": provider,
            "code": "configuration_error",
            "message": str(error),
            "retryable": False,
        }
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        if manager is not None:
            await manager.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(run(args.provider, args.mode))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
