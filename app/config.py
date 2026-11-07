"""Application settings, loaded from environment variables / .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Chatbot with Conversational Memory"

    # --- Storage -----------------------------------------------------------
    database_url: str = "sqlite:///./chatbot.db"

    # --- LLM ---------------------------------------------------------------
    # hf_endpoint : Hugging Face Inference Providers (serverless, needs HF_TOKEN)
    # hf_local    : local transformers pipeline (needs `pip install -r requirements-local.txt`)
    # fake        : deterministic offline model for tests / demos (no network)
    llm_provider: Literal["hf_endpoint", "hf_local", "fake"] = "hf_endpoint"
    # Comma-separated, tried in order: if a model isn't served by any provider you
    # have enabled (or errors), the next one is used. Availability on Hugging Face
    # Inference Providers changes over time, so a fallback chain keeps the app up.
    hf_model_id: str = (
        "meta-llama/Llama-3.1-8B-Instruct,"
        "deepseek-ai/DeepSeek-V3-0324,"
        "moonshotai/Kimi-K2-Instruct,"
        "openai/gpt-oss-20b"
    )
    hf_local_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")
    hf_inference_provider: str = "auto"
    temperature: float = 0.7
    max_new_tokens: int = 1024

    # --- Memory ------------------------------------------------------------
    # Number of most recent history messages sent to the LLM on each turn.
    # Everything is persisted; only the prompt window is trimmed.
    memory_window: int = 20

    # --- Auth --------------------------------------------------------------
    # Secret used to sign login tokens (JWT). If unset, one is generated once and
    # saved to `secret_key_file` so logins survive restarts.
    jwt_secret: str | None = None
    secret_key_file: str = ".secret_key"
    token_ttl_hours: int = 24 * 7

    @property
    def hf_model_ids(self) -> list[str]:
        return [m.strip() for m in self.hf_model_id.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
