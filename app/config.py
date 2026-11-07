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
    hf_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_local_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")
    hf_inference_provider: str = "auto"
    temperature: float = 0.7
    max_new_tokens: int = 512

    # --- Memory ------------------------------------------------------------
    # Number of most recent history messages sent to the LLM on each turn.
    # Everything is persisted; only the prompt window is trimmed.
    memory_window: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
