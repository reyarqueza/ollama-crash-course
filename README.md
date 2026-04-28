# Ollama Crash Course: Chat with a Document

This project is a small Streamlit app that lets you ask questions about a local text document using Ollama, LangChain, and Chroma.

The app currently loads `constitution.txt`, splits it into chunks, creates embeddings with `nomic-embed-text`, stores them in a Chroma vector store, and answers questions with `llama3.2:3b`.

Streamlit is a Python framework for quickly building interactive web apps for data and AI projects. Learn more at https://streamlit.io/.

## Tools

| Tool | URL |
| --- | --- |
| <img src="https://www.google.com/s2/favicons?domain=ollama.com&sz=32" alt="Ollama logo" width="20" height="20"> Ollama | https://ollama.com/ |
| <img src="https://www.google.com/s2/favicons?domain=langchain.com&sz=32" alt="LangChain logo" width="20" height="20"> LangChain | https://www.langchain.com/ |
| <img src="https://www.google.com/s2/favicons?domain=trychroma.com&sz=32" alt="Chroma logo" width="20" height="20"> Chroma | https://www.trychroma.com/ |
| <img src="https://www.google.com/s2/favicons?domain=streamlit.io&sz=32" alt="Streamlit logo" width="20" height="20"> Streamlit | https://streamlit.io/ |

## Requirements

Before running the app, make sure you have:

- Python 3.11 or newer
- Ollama installed and running

## 1. Install Ollama

Download and install Ollama from:

https://ollama.com/download

After installing, make sure Ollama is running. On macOS, opening the Ollama app usually starts the local server automatically.

You can confirm Ollama is available with:

```bash
ollama --version
```

## 2. Clone or Open the Project

If you already have this project locally, open a terminal in the project directory:

```bash
cd /path/to/ollama-crash-course
```

## 3. Create a Virtual Environment

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

On Windows PowerShell, use:

```powershell
.venv\Scripts\Activate.ps1
```

## 4. Install Python Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## VS Code Setup

This project includes `.vscode/settings.json`, so VS Code should automatically use the local virtual environment at `.venv/bin/python` when you open the folder.

If VS Code still shows squiggly lines under imports after installing dependencies, reload the window:

1. Open the Command Palette again.
2. Search for and select `Developer: Reload Window`.

If the squiggly lines remain, open the Command Palette, select `Python: Select Interpreter`, and choose `.venv/bin/python`.

## 5. Download the Ollama Models

The app uses two Ollama models:

- `nomic-embed-text` for document embeddings
- `llama3.2:3b` for answering questions

Pull them before running the app:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Note: `chatdoc.py` also calls `ollama.pull('nomic-embed-text')`, but pulling the models ahead of time makes startup smoother.

## 6. Run the App

Start the Streamlit app:

```bash
streamlit run chatdoc.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser, type a question, and the app will answer using the contents of `constitution.txt`.

## Example Questions

Try asking:

- What is the purpose of the Constitution?
- What powers does Congress have?
- How can the Constitution be amended?
- What does the document say about the President?

You can also test whether the app avoids answering from the model's general knowledge:

| Question | Expected Answer |
| --- | --- |
| What does the 22nd Amendment say about presidential term limits? | The app should say it does not know based on the provided document, because this `constitution.txt` file does not include the 22nd Amendment. |

## Project Files

- `chatdoc.py` - Main Streamlit application
- `constitution.txt` - Source document used by the app
- `requirements.txt` - Python dependencies
- `CDOC-110hdoc50.pdf` - Original PDF source included in the project

## Troubleshooting

If you see an error about connecting to Ollama, make sure Ollama is installed and running.

If a model is missing, run:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

If Python packages are missing, make sure your virtual environment is activated, then run:

```bash
pip install -r requirements.txt
```

If Streamlit is not found, run it through Python:

```bash
python -m streamlit run chatdoc.py
```
