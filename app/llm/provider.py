"""LLM factory: returns a LangChain chat model for the configured provider."""

from langchain_core.language_models import BaseChatModel

from app.config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider

    if provider == "fake":
        from app.llm.fake import EchoMemoryChatModel

        return EchoMemoryChatModel()

    if provider == "hf_endpoint":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        if not settings.hf_token:
            raise RuntimeError(
                "LLM_PROVIDER=hf_endpoint requires HF_TOKEN. Create one at "
                "https://huggingface.co/settings/tokens, or set LLM_PROVIDER=fake to run offline."
            )
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.hf_model_id,
            provider=settings.hf_inference_provider,
            huggingfacehub_api_token=settings.hf_token,
            task="conversational",
            temperature=settings.temperature,
            max_new_tokens=settings.max_new_tokens,
            streaming=True,
        )
        return ChatHuggingFace(llm=endpoint, model_id=settings.hf_model_id)

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
