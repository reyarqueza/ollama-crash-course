# Agent Notes

This repository contains a small Streamlit RAG app that answers questions about `constitution.txt` using LangChain, Pinecone, Google Gemini embeddings, and an Ollama Cloud-hosted chat model.

## Project Overview

- `chatdoc.py` is the main Streamlit app.
- `constitution.txt` is the source document used for retrieval.
- `requirements.txt` contains the Python dependencies.
- `.vscode/settings.json` points VS Code at the local `.venv` interpreter.

## Local Setup

Use the project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The current app does not require a local Ollama server or locally pulled Ollama models. It calls hosted APIs instead:

- Chat model: `gpt-oss:120b` through the Ollama Cloud API at `https://ollama.com`.
- Embedding model: Google `models/gemini-embedding-001` through the Gemini API.
- Vector database: Pinecone cloud.

Provide API keys either as environment variables or in `.streamlit/secrets.toml`:

```toml
OLLAMA_API_KEY = "your_ollama_api_key"
GEMINI_API_KEY = "your_gemini_api_key"
PINECONE_API_KEY = "your_pinecone_api_key"
```

`GOOGLE_API_KEY` can be used instead of `GEMINI_API_KEY` because `chatdoc.py` falls back to it for Gemini embeddings.

Run the app with:

```bash
streamlit run chatdoc.py
```

## RAG Behavior

The prompt in `chatdoc.py` intentionally tells the model to answer only from retrieved context. If the answer is not in the retrieved context, the app should say it does not know based on the provided document.

The app uses hybrid retrieval:

- Vector retrieval with Pinecone and Google Gemini embeddings.
- Multi-query retrieval with the Ollama Cloud chat model to generate alternate search queries.
- A small keyword backup that pulls in chunks containing exact terms from the question.

Current retrieval settings:

- `constitution.txt` is split into chunks of `1000` characters with `200` characters of overlap.
- Pinecone index name: `constitution-rag`.
- Pinecone namespace: `constitution`.
- Pinecone vector dimension: `3072`, matching the default output dimension of `models/gemini-embedding-001`.
- Pinecone retrieves `k=6` vector results.
- The multi-query prompt asks the chat model to generate five alternate versions of the user's question.
- The keyword backup adds up to the top three exact-term chunk matches after filtering common stop words.

This keyword backup is important for questions like:

```text
What is the age requirement to be a senator?
```

Plain vector search may retrieve the similar Representative age chunk before the Senator chunk. The keyword backup helps include chunks containing exact words like `senator`.

## Useful Test Questions

These should be answerable from `constitution.txt`:

- What is the age requirement to be a Senator?
- What powers does Congress have?
- What does the document say about the President?

This should not be answered from model memory:

- What does the 22nd Amendment say about presidential term limits?

Expected behavior: the app should say it does not know based on the provided document because this file does not include the 22nd Amendment.

## Development Notes

- Keep the retrieved context expander while debugging RAG behavior; it shows which chunks were passed to the model.
- Be careful when changing chunk size, retrieval `k`, or the prompt, because small changes can affect whether the model answers or refuses.
- Be careful when changing `CHAT_MODEL`, `EMBEDDING_MODEL`, `OLLAMA_BASE_URL`, or secret handling because the app currently depends on hosted Ollama and Gemini APIs.
- Prefer clear, beginner-friendly code and comments. This project is part of a learning-oriented crash course.
- Do not commit `.venv`, caches, `.streamlit/secrets.toml`, or other local environment files.
