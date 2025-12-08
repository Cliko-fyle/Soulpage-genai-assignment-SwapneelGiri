# Conversational Knowledge Bot

**A Langchain-powered AI assistant with web search, memory, and conversational rewriting**

**OVERVIEW**

This project implements a Conversational Knowledge Bot using Streamlit, LangChain, Groq LLaMA 3.1, DuckDuckGo web search, and a conversation-aware retriever pipeline.

The assistant can:
- Rewrite user questions using chat history
- Perform real-time web search
- Generate concise, accurate answers
- Maintain conversation memory
- Provide a clean chat interface like ChatGPT


**FEATURES**

- **Conversational Memory**: Stores and uses previous chat history to resolve pronouns and context.

- **Real-Time Web Search**: Uses DuckDuckGo API via LangChain wrappers.

- **LLM-Powered Reasoning**: LLaMA-3.1-8B-Instant (Groq API) for fast inference.

- **Chat Interface with Streamlit**: Interactive, clean UI with chat bubbles.

- **Rewritten Query Pipeline**: Converts vague user questions into explicit rewritten versions for better search accuracy.


**INSTALLATION & SETUP**

- **Clone the repository**
  
  ```
  git clone https://github.com/Cliko-Fyle/conversational-knowledge-bot.git
  cd conversational-knowledge-bot
- **Create a virtual environment**
  
  ```
  python -m venv venv
  venv\Scripts\activate
- **Install dependencies**
  
  ```
  groq_api_key = "YOUR_GROQ_API_KEY"
**SYSTEM ARCHITECTURE**
```
                   ┌───────────────────────────────┐
                   │             User              │
                   └───────────────────────────────┘
                                 │  Query
                                 ▼
      ┌────────────────────────────────────────────────────────┐
      │                    Streamlit Frontend                  │
      │  - Chat UI                                             │
      │  - Message state management                            │
      └────────────────────────────────────────────────────────┘
                                 │
                                 ▼
      ┌────────────────────────────────────────────────────────┐
      │                web_qa_chain() Pipeline                 │
      │                                                        │
      │ 1. Rewrite Question (LLM)                              │
      │    → resolves pronouns using chat history              │
      │                                                        │
      │ 2. Web Search Tool (DuckDuckGo)                        │
      │    → retrieves 3 relevant search results               │
      │                                                        │
      │ 3. Answer LLM (Groq LLaMA 3.1)                         │
      │    → synthesizes a concise answer using the results    │
      │                                                        │
      │ 4. Memory Saver                                        │
      │    → stores interaction in ConversationBufferMemory    │
      └────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                         Final Answer Shown in UI
```
**EXAMPLE USAGE**

<img width="500" height="400" alt="Screenshot 2025-12-08 230737" src="https://github.com/user-attachments/assets/2f4ea57e-3702-4b7d-ab0c-018b6dec1a24" />
