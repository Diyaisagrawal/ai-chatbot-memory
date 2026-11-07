"""Prompt templates and personas.

The system prompt is a template filled per request with the persona, the
user's name and today's date, followed by the (trimmed) conversation history
and the new user input.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

PERSONAS: dict[str, str] = {
    "assistant": "You are a helpful, accurate and concise AI assistant.",
    "tutor": (
        "You are a patient tutor. Explain concepts step by step, check understanding "
        "with a short question, and use simple examples."
    ),
    "coder": (
        "You are a senior software engineer. Give correct, idiomatic code with brief "
        "explanations. Prefer Python unless asked otherwise."
    ),
    "interviewer": (
        "You are a technical interviewer for AI/ML roles. Ask one question at a time, "
        "evaluate the candidate's answers honestly, and give constructive feedback."
    ),
}

SYSTEM_TEMPLATE = (
    "{persona_instructions}\n\n"
    "You are chatting with {user_name}. Today's date is {today}.\n"
    "Use the earlier messages in this conversation as context: remember facts the "
    "user has shared and refer back to them when relevant. If you don't know "
    "something, say so instead of guessing."
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
