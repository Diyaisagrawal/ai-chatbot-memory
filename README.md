# 🧠 AI Chatbot with Conversational Memory

A multi-user, context-aware chatbot built with **LangChain**, **Hugging Face** and **FastAPI**.
Each user can have many chat sessions. Every session keeps its own persistent memory, so the bot
remembers what you said earlier in a conversation, even after the server restarts.

![screenshot](docs/screenshot.png)

**Tech:** Python · LangChain · Hugging Face (Inference Providers / Transformers) · FastAPI · SQLAlchemy · REST + Server-Sent Events · pytest · Docker

## Features

- **Session-based conversational memory**: multi-turn, context-aware replies, persisted in SQL.
- **Multi-user sessions**: each user owns their sessions; cross-user access is blocked.
- **Prompt templating** with LangChain `ChatPromptTemplate`: persona + user name + date + history.
- **Message-history management**: full history stored, sliding window (`trim_messages`) sent to the model.
- **Real-time streaming** responses over SSE, plus a standard JSON endpoint.
- **Pluggable LLM backends**: Hugging Face Inference Providers, local `transformers` pipeline, or an offline fake model.
- **Personas**: assistant, tutor, coder, interviewer (switchable per session).
- Auto-titled sessions, rename / clear / delete, a dark-mode web UI, an OpenAPI docs page, CI and Docker.

## Architecture

```mermaid
flowchart LR
    UI[Web UI / REST client] -->|X-User-Id| API[FastAPI routes<br/>app/api]
    API --> SVC[ChatService<br/>app/chain.py]
    SVC -->|load| MEM[SQLChatHistory<br/>BaseChatMessageHistory]
    MEM <--> DB[(SQLite / Postgres<br/>users · sessions · messages)]
    SVC --> TRIM[trim_messages<br/>last N msgs]
    TRIM --> PROMPT[ChatPromptTemplate<br/>system + history + input]
    PROMPT --> LLM{{LLM provider<br/>HF endpoint · HF local · fake}}
    LLM -->|tokens| API
    SVC -->|save turn| MEM
```

**Request flow for one turn:**
1. FastAPI authenticates the user (`X-User-Id`) and checks they own the session.
2. `ChatService` loads that session's history from the DB through a custom LangChain `BaseChatMessageHistory`.
3. `trim_messages` keeps the last `MEMORY_WINDOW` messages, always starting on a user turn.
4. The `ChatPromptTemplate` renders the system prompt (persona, user name, date), the history and the new input.
5. The LLM generates a response (streamed token by token on `/api/chat/stream`).
6. The human and AI messages are persisted together, **only after the model succeeds**, so a failed call never leaves half a turn in memory.

```
app/
├── main.py            # App factory, lifespan (DB + LLM wiring), static UI
├── config.py          # Typed settings from env / .env (pydantic-settings)
├── db.py              # SQLAlchemy models: User, ChatSession, Message
├── memory/history.py  # SQL-backed LangChain chat history (per session)
├── chain.py           # ChatService: history -> trim -> prompt -> LLM -> persist
├── prompts.py         # ChatPromptTemplate + personas
├── llm/provider.py    # LLM factory (hf_endpoint | hf_local | fake)
├── llm/fake.py        # Deterministic offline model (tests / demos)
├── schemas.py         # Pydantic request/response models
├── api/routes.py      # REST + SSE endpoints
├── api/deps.py        # Dependencies: DB, current user, ownership checks
└── static/index.html  # Web UI
tests/test_api.py      # 15 end-to-end API tests
```

## Quickstart

```bash
git clone https://github.com/Diyaisagrawal/ai-chatbot-memory.git
cd ai-chatbot-memory
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then put your HF_TOKEN in .env
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the chat UI, or http://127.0.0.1:8000/docs for interactive API docs.

### Choosing an LLM backend (`LLM_PROVIDER`)

| Value | What it does | Needs |
|---|---|---|
| `hf_endpoint` *(default)* | Hugging Face Inference Providers via `ChatHuggingFace` + `HuggingFaceEndpoint` | Free `HF_TOKEN` ([create one](https://huggingface.co/settings/tokens) with "Make calls to Inference Providers") |
| `hf_local` | Runs `HF_LOCAL_MODEL_ID` locally with a `transformers` pipeline | `pip install -r requirements-local.txt` (a 0.5B model runs on CPU) |
| `fake` | Deterministic offline model that reports what's in its context | Nothing. Great for demos of the memory mechanism. |

Change the model with `HF_MODEL_ID` (any chat model served by HF Inference Providers, e.g. `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`).

### Docker

```bash
docker build -t memory-chat .
docker run -p 8000:8000 --env-file .env -v chat-data:/app/data memory-chat
```

## REST API

All endpoints except `POST /api/users`, `/api/personas` and `/health` require the `X-User-Id` header.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/users` | Sign in / create user by `username` → returns `id` |
| `GET` | `/api/users/me` | Current user |
| `GET` | `/api/personas` | Available personas |
| `POST` | `/api/sessions` | Create a session `{persona?, title?}` |
| `GET` | `/api/sessions` | List your sessions (most recent first) |
| `PATCH` | `/api/sessions/{id}` | Rename / change persona |
| `DELETE` | `/api/sessions/{id}` | Delete session and its memory |
| `GET` | `/api/sessions/{id}/messages` | Full stored history |
| `DELETE` | `/api/sessions/{id}/messages` | Clear memory, keep the session |
| `POST` | `/api/chat` | `{session_id, message}` → `{reply}` |
| `POST` | `/api/chat/stream` | Same, streamed as SSE: `data: {"token": ...}` … `data: {"done": true}` |

```bash
USER_ID=$(curl -s -X POST localhost:8000/api/users -H 'Content-Type: application/json' -d '{"username":"diya"}' | jq -r .id)
SID=$(curl -s -X POST localhost:8000/api/sessions -H "X-User-Id: $USER_ID" -H 'Content-Type: application/json' -d '{"persona":"tutor"}' | jq -r .id)
curl -s -X POST localhost:8000/api/chat -H "X-User-Id: $USER_ID" -H 'Content-Type: application/json' \
     -d "{\"session_id\":\"$SID\",\"message\":\"Hi, I'm Diya and I'm studying transformers\"}"
curl -s -X POST localhost:8000/api/chat -H "X-User-Id: $USER_ID" -H 'Content-Type: application/json' \
     -d "{\"session_id\":\"$SID\",\"message\":\"What am I studying?\"}"
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite runs fully offline against the `fake` model and a temporary SQLite DB. It covers multi-turn memory, session isolation, cross-user access control, window trimming vs. full persistence, memory surviving a restart, SSE streaming, failure handling (no orphaned messages) and prompt rendering. CI runs it on every push (GitHub Actions).

## Design decisions

- **Store everything, prompt a window.** Persisting all messages keeps history auditable and lets the UI show the full chat, while the sliding window bounds prompt size and latency. Counting *messages* instead of tokens keeps it model-agnostic.
- **Explicit history management instead of `RunnableWithMessageHistory`.** That helper is deprecated in LangChain 1.x. Loading, trimming and saving explicitly makes the memory flow easy to follow and lets us save a turn only after the LLM succeeds.
- **Custom `BaseChatMessageHistory` over SQLAlchemy.** It plugs into any LangChain component and works with SQLite locally or Postgres in production (just change `DATABASE_URL`).
- **Ownership returns 404, not 403.** Session IDs can't be probed across users.
- **Provider factory.** Swapping between hosted, local and fake models is a config change; tests never hit the network.

## Roadmap

- Summary memory: summarize messages that fall out of the window into a running summary.
- Long-term semantic memory: embed past messages (`HuggingFaceEmbeddings`) and retrieve relevant ones (RAG over history).
- Real authentication (JWT / OAuth) instead of the `X-User-Id` header.
- Token-based trimming using the model's tokenizer; rate limiting.

## License

MIT
