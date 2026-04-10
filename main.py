import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import LocalEmbedder
from src.store import EmbeddingStore


def build_llm_fn():
    """
    Build a real LLM call function using the NVIDIA API.
    NVIDIA's API is OpenAI-compatible, so we use the openai client
    pointed at NVIDIA_BASE_URL.
    """
    load_dotenv()
    
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_BASE_URL")
    model = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-20b")

    if not api_key:
        print("Warning: NVIDIA_API_KEY not set. Falling back to mock LLM.")
        return lambda prompt: f"[MOCK LLM] Prompt preview: {prompt[:100]}..."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)

        def llm_fn(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý thông minh. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp bằng tiếng Việt. Nếu ngữ cảnh không đủ thông tin, hãy nói thẳng điều đó."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=512,
            )
            return response.choices[0].message.content

        print(f"LLM Backend: NVIDIA API ({model})")
        return llm_fn

    except ImportError:
        print("Warning: openai package not installed. Falling back to mock LLM.")
        return lambda prompt: f"[MOCK LLM] Prompt preview: {prompt[:100]}..."


def main():
    load_dotenv()

    # Configuration
    strategy = os.getenv("RAG_STRATEGY", "recursive")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/vector_stores/chroma_db")

    print("=== Rap Viet RAG Agent ===")
    print(f"Strategy  : {strategy}")
    print(f"Database  : {persist_dir}")

    # Initialize persistent ChromaDB store with LocalEmbedder
    try:
        store = EmbeddingStore(
            collection_name=f"rap_viet_{strategy}",
            persist_directory=persist_dir,
            embedding_fn=LocalEmbedder()
        )
        size = store.get_collection_size()
        print(f"Collection: rap_viet_{strategy} ({size} chunks)")
        if size == 0:
            print("\n[!] Collection is empty. Please run:")
            print("    python src/process_data.py")
            print("    python src/index_data.py --all --use_chroma --embedder local")
            return
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return

    # Build LLM function (NVIDIA API)
    llm_fn = build_llm_fn()

    # Setup Agent
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)

    # Get query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Minh Lai là ai?"

    print(f"\nQuery: {query}")

    # Show retrieved chunks for transparency
    print("\n--- Retrieved Chunks ---")
    results = store.search(query, top_k=3)
    for i, res in enumerate(results):
        print(f"[{i+1}] Score={res['score']:.3f} | File={res['metadata'].get('original_file', 'N/A')}")
        print(f"    {res['content'][:150].strip()}...")
        print()

    # Final answer
    print("--- Answer ---")
    print(agent.answer(query, top_k=3))


if __name__ == "__main__":
    main()
