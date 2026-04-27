from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

import os
print(os.getenv("OPENAI_API_KEY"))

model="gpt-4.1-mini"
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

def chat_with_history(user_input, session_id):
    return "Mock response for testing"

def chat_with_history(user_input: str, session_id: str) -> str:
    config = {"configurable": {"session_id": session_id}}
    response = with_message_history.invoke([HumanMessage(content=user_input)], config=config)
    return response.content

def chat_with_history(user_input: str, session_id: str, role="helpful assistant"):
    custom_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a {role}."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = custom_prompt | model
    with_history = RunnableWithMessageHistory(chain, get_session_history)

    config = {"configurable": {"session_id": session_id}}
    response = with_history.invoke(
        [HumanMessage(content=user_input)],
        config=config
    )

    return response.content

print(chat_with_history("Hi", "user1"))
print(chat_with_history("What did I just say?", "user1"))