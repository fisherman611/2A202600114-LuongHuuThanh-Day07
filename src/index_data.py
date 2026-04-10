import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow imports if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.store import EmbeddingStore
from src.models import Document
from src.embeddings import LocalEmbedder, _mock_embed

def index_chunks(processed_dir: Path, output_dir: Path, strategy: str, use_chroma: bool = False, embedder_type: str = "mock"):
    """Load chunks for a specific strategy, embed them, and save to the store (Chroma or JSON)."""
    
    embedding_fn = _mock_embed
    if embedder_type == "local":
        print("Initializing LocalEmbedder (this might download/load a model)...")
        embedding_fn = LocalEmbedder()

    if use_chroma:
        # Chroma handles its own persistence in the directory
        chroma_path = str(output_dir / "chroma_db")
        store = EmbeddingStore(
            collection_name=f"rap_viet_{strategy}", 
            persist_directory=chroma_path,
            embedding_fn=embedding_fn
        )
        print(f"Indexing strategy '{strategy}' using {embedder_type} embeddings into ChromaDB at {chroma_path}...")
    else:
        output_file = output_dir / f"vector_store_{strategy}_{embedder_type}.json"
        store = EmbeddingStore(
            collection_name=f"kb_{strategy}", 
            persist_directory=None,
            embedding_fn=embedding_fn
        )
        print(f"Indexing strategy '{strategy}' using {embedder_type} embeddings into JSON store...")
    
    json_files = list(processed_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No processed JSON files found in {processed_dir}")
        return
    
    total_docs = []
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if strategy not in data:
                continue
            
            chunks = data[strategy].get("chunks", [])
            doc_id_base = json_file.stem.replace("_chunks", "")
            
            for i, chunk_text in enumerate(chunks):
                doc = Document(
                    id=f"{doc_id_base}_{strategy}_c{i}",
                    content=chunk_text,
                    metadata={
                        "original_file": json_file.name,
                        "strategy": strategy,
                        "chunk_index": i
                    }
                )
                total_docs.append(doc)
        except Exception as e:
            print(f"  Failed to read {json_file.name}: {e}")
    
    if total_docs:
        store.add_documents(total_docs)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not use_chroma:
            store.save(str(output_file))
            print(f"  Success: {len(total_docs)} chunks -> {output_file.name}")
        else:
            print(f"  Success: {len(total_docs)} chunks indexed into ChromaDB.")
    else:
        print(f"  No chunks found for strategy '{strategy}'")

def main():
    parser = argparse.ArgumentParser(description="Embed chunked text and save to separate vector stores.")
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="data/processed_data", 
        help="Directory with processed JSON chunk files."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/vector_stores", 
        help="Directory to save the separate vector store files."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Index all available strategies (fixed_size, by_sentences, recursive)."
    )
    parser.add_argument(
        "--use_chroma",
        action="store_true",
        help="Use ChromaDB instead of the in-memory fallback store."
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="mock",
        choices=["mock", "local"],
        help="Which embedding model to use."
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="recursive", 
        choices=["fixed_size", "by_sentences", "recursive"], 
        help="Strategy to index if --all is not set."
    )
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    
    strategies = ["fixed_size", "by_sentences", "recursive"] if args.all else [args.strategy]
    
    print(f"Starting indexing process in: {processed_dir}")
    for strat in strategies:
        index_chunks(processed_dir, output_dir, strat, use_chroma=args.use_chroma, embedder_type=args.embedder)
    
    print(f"\nAll operations complete. Files stored in: {output_dir}")

if __name__ == "__main__":
    main()
