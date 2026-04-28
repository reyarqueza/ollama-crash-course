# Agent Notes

This repository contains a small Streamlit RAG app that answers questions about `constitution.txt` using Ollama, LangChain, and Chroma.

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

The app expects Ollama to be installed and running with these models available:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Run the app with:

```bash
streamlit run chatdoc.py
```

## RAG Behavior

The prompt in `chatdoc.py` intentionally tells the model to answer only from retrieved context. If the answer is not in the retrieved context, the app should say it does not know based on the provided document.

The app uses hybrid retrieval:

- Vector retrieval with Chroma and Ollama embeddings.
- Multi-query retrieval to generate alternate search queries.
- A small keyword backup that pulls in chunks containing exact terms from the question.

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
- Prefer clear, beginner-friendly code and comments. This project is part of a learning-oriented crash course.
- Do not commit `.venv`, caches, or local environment files.
