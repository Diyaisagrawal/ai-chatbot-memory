# 💬 Chatbot with Memory using LangChain, FastAPI & OpenAI

This project is a simple yet powerful chatbot built with **FastAPI** and **LangChain**, integrating OpenAI's GPT-3.5-Turbo model. It supports session-based message memory, a customizable user interface with HTML/JS, dark mode, and persistent chat history using localStorage.

## 🚀 Features

- ✅ Session-based memory using LangChain
- ✅ FastAPI-powered backend
- ✅ Lightweight HTML + JavaScript frontend
- ✅ Username prompt on first visit
- ✅ Persistent chat history (localStorage)
- ✅ Dark mode toggle
- ✅ Clear chat button


## 🛠 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/iremhazald/chatbot-memory-app.git
cd chatbot-memory-app
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create a `.env` file** and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_key_here
```

5. **Run the server:**
```bash
uvicorn main:app --reload
```

6. **Open in your browser:**
```
http://127.0.0.1:8000
```

## 💻 Usage

- When the page loads, you’ll be prompted to enter your name.
- Type your message and click **Send**.
- Messages will appear in the chat box along with AI responses.
- You can toggle dark mode or clear chat history at any time.

## 📁 Project Structure

```
├── main.py               # FastAPI app and routes
├── chat_logic.py         # LangChain logic for session memory
├── templates/
│   └── index.html        # Frontend user interface
├── requirements.txt      # Python dependencies
└── .env                  # Your OpenAI API key (not tracked)
```

## 🛠 Technologies Used

- [LangChain](https://www.langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI GPT-3.5 Turbo](https://platform.openai.com/)
- HTML + Vanilla JavaScript
- Python 3.9+

## 👩‍💻 Developer

**İrem Hazal Deveci**  
[LinkedIn](https://www.linkedin.com/in/iremhazaldeveci) • [GitHub](https://github.com/iremhazald))

---

## 📃 License

This project is licensed under the MIT License. Feel free to use, modify, and share.