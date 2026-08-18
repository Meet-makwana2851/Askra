# Askra 📚

A local, private RAG (Retrieval-Augmented Generation) assistant that answers questions grounded in your own documents — runs entirely on your machine, no API keys, no data leaving your computer.

## Architecture

## Stack
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Re-ranking:** cross-encoder (`ms-marco-MiniLM-L-6-v2`)
- **Vector store:** ChromaDB (local, persistent)
- **LLM:** Llama 3.2 via Ollama (local inference)
- **UI:** Streamlit

## Setup

1. Install [Ollama](https://ollama.com) and pull a model:
```bash
   ollama pull llama3.2
   brew services start ollama
```

2. Create a virtual environment and install dependencies:
```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```

3. Add your documents (`.pdf` or `.txt`) to the `docs/` folder.

4. Build the vector index:
```bash
   python3 index_docs.py
```

5. Run the app:
```bash
   streamlit run app.py
```

## Why RAG?

LLMs can't answer questions about private/local documents they were never trained on. Askra retrieves the most relevant chunks from your own documents and feeds them to the LLM as context — so answers are grounded in real content instead of the model guessing or hallucinating.
