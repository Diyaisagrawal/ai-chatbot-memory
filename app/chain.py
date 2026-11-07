"""Conversation pipeline: persistent history -> trimmed window -> prompt template -> LLM.

    load history (SQL)  ->  trim to last N messages  ->  ChatPromptTemplate
         ^                                                     |
         |                                                     v
    persist (human, ai) turn  <----------------------------  LLM (Hugging Face)

The turn is saved only after the model succeeds, so a failed or aborted LLM
call never leaves a dangling user message without a reply in memory.
"""

from collections.abc import AsyncIterator
from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy.orm import sessionmaker

from app.memory.history import SQLChatHistory
from app.prompts import PERSONAS, chat_prompt


class ChatService:
    def __init__(self, llm: BaseChatModel, session_factory: sessionmaker, memory_window: int):
        self.session_factory = session_factory
        self.chain = chat_prompt | llm | StrOutputParser()
        # Keep the last `memory_window` messages, always starting on a human turn
        # so the model never sees an orphaned assistant reply.
        self.trimmer = trim_messages(
            max_tokens=memory_window,
            token_counter=len,  # counts messages, not tokens: works for any model
            strategy="last",
            start_on="human",
        )

    def get_history(self, session_id: str) -> SQLChatHistory:
        return SQLChatHistory(session_id, self.session_factory)

    def _build_inputs(self, history: SQLChatHistory, message: str, user_name: str, persona: str) -> dict:
        return {
            "persona_instructions": PERSONAS.get(persona, PERSONAS["assistant"]),
            "user_name": user_name,
            "today": date.today().isoformat(),
            "history": self.trimmer.invoke(history.messages),
            "input": message,
        }

    async def respond(self, session_id: str, message: str, user_name: str, persona: str) -> str:
        history = self.get_history(session_id)
        reply = await self.chain.ainvoke(self._build_inputs(history, message, user_name, persona))
        history.add_messages([HumanMessage(content=message), AIMessage(content=reply)])
        return reply

    async def stream(
        self, session_id: str, message: str, user_name: str, persona: str
    ) -> AsyncIterator[str]:
        history = self.get_history(session_id)
        parts: list[str] = []
        async for token in self.chain.astream(self._build_inputs(history, message, user_name, persona)):
            parts.append(token)
            yield token
        history.add_messages([HumanMessage(content=message), AIMessage(content="".join(parts))])
