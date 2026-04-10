# Lab Day 7 — Running the RAG Pipeline from Scratch

This guide explains how to process your raw data, generate embeddings, and store them in a persistent vector database using the tools implemented in this repository.

## 1. Setup Environment

First, ensure you have all necessary dependencies installed. The pipeline now supports persistent **ChromaDB** and **Local Embeddings** (sentence-transformers).

```bash
# Activate your virtual environment
.\env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

---

## 2. Step 1: Data Chunking

Use the `process_data.py` script to split your raw markdown files into smaller chunks using multiple strategies (`fixed_size`, `by_sentences`, and `recursive`).

```bash
python src/process_data.py --input_dir data/raw_data --output_dir data/processed_data --chunk_size 500
```

- **Input:** Raw `.md` or `.txt` files in `data/raw_data/`.
- **Output:** Structured `.json` files in `data/processed_data/`.

---

## 3. Step 2: Vector Indexing (Local Embeddings + ChromaDB)

The `index_data.py` script generates real vector embeddings for your chunks and stores them in a persistent ChromaDB database.

Run this command to index using the most effective strategy (**recursive**) and a **local transformer model**:

```bash
python src/index_data.py --all --use_chroma --embedder local --output_dir data/vector_stores
```

- **Strategies:** `--all` creates separate indices for all 3 strategies.
- **Embedder:** `--embedder local` uses the `all-MiniLM-L6-v2` model.
- **Persistence:** `--use_chroma` saves the database to `data/vector_stores/chroma_db/`.

---

## 4. Step 3: Verify the System

Run the test suite to ensure all components (chunking, store, agent) are working correctly with the new updates.

```bash
python -m pytest tests/ -v
```

---

## 5. Usage in Code

Once indexed, you can use the `EmbeddingStore` and `KnowledgeBaseAgent` in your application like this:

```python
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.embeddings import LocalEmbedder

# 1. Initialize the persistent store
store = EmbeddingStore(
    collection_name="rap_viet_recursive", 
    persist_directory="data/vector_stores/chroma_db",
    embedding_fn=LocalEmbedder()
)

# 2. Setup the Knowledge Base Agent
agent = KnowledgeBaseAgent(
    store=store, 
    llm_fn=lambda prompt: "Kết quả mô phỏng trả về..." # Replace with your LLM call
)

# 3. Ask a question
answer = agent.answer("Minh Lai là ai?")
print(f"Câu trả lời: {answer}")
```

---

## Command Reference Summary

| Task | Command |
|------|---------|
| **Install** | `pip install -r requirements.txt` |
| **Chunk Data** | `python src/process_data.py` |
| **Index (Mock)** | `python src/index_data.py --all` |
| **Index (Real)** | `python src/index_data.py --all --use_chroma --embedder local` |
| **Test** | `python -m pytest tests/` |
