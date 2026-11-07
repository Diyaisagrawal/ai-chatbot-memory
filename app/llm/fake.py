"""A deterministic, offline chat model.

Used by the test suite and for running the app without a Hugging Face token.
Its replies are derived from the prompt it receives, which makes it a good
probe for memory: you can see exactly how much history reached the model.
"""

import re
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

_NAME_RE = re.compile(r"\bmy name is ([A-Za-z][\w-]*)", re.IGNORECASE)


class EchoMemoryChatModel(BaseChatModel):
    """Echoes the latest user message and reports what it remembers."""

    @property
    def _llm_type(self) -> str:
        return "echo-memory"

    def _reply(self, messages: list[BaseMessage]) -> str:
        humans = [m for m in messages if m.type == "human"]
        latest = str(humans[-1].content) if humans else ""
        earlier = humans[:-1]
        name = None
        for m in humans:
            if match := _NAME_RE.search(str(m.content)):
                name = match.group(1)
        parts = [f'You said: "{latest}".']
        parts.append(f"I can see {len(earlier)} earlier message(s) from you in this conversation.")
        if earlier:
            parts.append(f'The oldest one in my context is: "{earlier[0].content}".')
        if name:
            parts.append(f"Your name is {name}.")
        return " ".join(parts)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._reply(messages)))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        words = self._reply(messages).split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
