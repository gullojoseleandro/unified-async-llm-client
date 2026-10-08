from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ChatRole = Literal["system", "user", "assistant"]
ProviderName = Literal["openai", "anthropic"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    model: str = Field(min_length=1)
    api_key: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)

    @model_validator(mode="after")
    def validate_provider_temperature(self) -> "ModelConfig":
        if self.provider == "anthropic" and self.temperature > 1.0:
            raise ValueError("Anthropic temperature must be between 0 and 1")
        return self


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    model: str
    provider: ProviderName
    usage: dict[str, int] = Field(default_factory=dict)
