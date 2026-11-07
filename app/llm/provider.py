"""LLM factory: returns a LangChain chat model (runnable) for the configured provider."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.config import Settings

log = logging.getLogger(__name__)


def _hf_endpoint_model(settings: Settings, model_id: str) -> BaseChatModel:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        provider=settings.hf_inference_provider,
        huggingfacehub_api_token=settings.hf_token,
        task="conversational",
        temperature=settings.temperature,
        max_new_tokens=settings.max_new_tokens,
        streaming=True,
    )
    return ChatHuggingFace(llm=endpoint, model_id=model_id)


def build_llm(settings: Settings) -> Runnable:
    provider = settings.llm_provider

    if provider == "fake":
        from app.llm.fake import EchoMemoryChatModel

        return EchoMemoryChatModel()

    if provider == "hf_endpoint":
        if not settings.hf_token:
            raise RuntimeError(
                "LLM_PROVIDER=hf_endpoint requires HF_TOKEN. Create one at "
                "https://huggingface.co/settings/tokens, or set LLM_PROVIDER=fake to run offline."
            )
        model_ids = settings.hf_model_ids
        if not model_ids:
            raise RuntimeError("HF_MODEL_ID is empty")
        log.info("Hugging Face model chain (first that works is used): %s", " -> ".join(model_ids))
        primary, *fallbacks = [_hf_endpoint_model(settings, m) for m in model_ids]
        # LangChain fallbacks: if a model is unavailable or errors, try the next one.
        return primary.with_fallbacks(fallbacks) if fallbacks else primary

    if provider == "hf_local":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        pipeline = HuggingFacePipeline.from_model_id(
            model_id=settings.hf_local_model_id,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": settings.max_new_tokens,
                "temperature": settings.temperature,
                "do_sample": settings.temperature > 0,
                "return_full_text": False,
            },
        )
        return ChatHuggingFace(llm=pipeline, model_id=settings.hf_local_model_id)

    raise ValueError(f"Unknown LLM provider: {provider}")
