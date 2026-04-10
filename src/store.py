from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except ImportError:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": dict(doc.metadata, doc_id=doc.id),
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_vec = self._embedding_fn(query)
        scored = []
        for r in records:
            score = _dot(query_vec, r["embedding"])
            scored.append(
                {
                    "id": r["id"],
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "score": score,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        if self._use_chroma and self._collection:
            count = len(docs)
            ids = [f"{doc.id}_{self._next_index + i}" for i, doc in enumerate(docs)]
            contents = [doc.content for doc in docs]
            embeddings = [self._embedding_fn(c) for c in contents]
            metadatas = [dict(doc.metadata, doc_id=doc.id) for doc in docs]
            self._collection.add(ids=ids, documents=contents, embeddings=embeddings, metadatas=metadatas)
            self._next_index += count
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._use_chroma and self._collection:
            results = self._collection.query(query_embeddings=[self._embedding_fn(query)], n_results=top_k)
            if not results["ids"] or not results["ids"][0]:
                return []

            out = []
            for i in range(len(results["ids"][0])):
                out.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - (results["distances"][0][i] if results.get("distances") else 0),
                    }
                )
            return out
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        if self._use_chroma and self._collection:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        if self._use_chroma and self._collection:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)], n_results=top_k, where=metadata_filter
            )
            if not results["ids"] or not results["ids"][0]:
                return []

            out = []
            for i in range(len(results["ids"][0])):
                out.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - (results["distances"][0][i] if results.get("distances") else 0),
                    }
                )
            return out

        candidates = self._store
        if metadata_filter:
            candidates = [
                r for r in self._store if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        if self._use_chroma and self._collection:
            size_before = self._collection.count()
            self._collection.delete(where={"doc_id": doc_id})
            return self._collection.count() < size_before

        size_before = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        return len(self._store) < size_before

    def save(self, file_path: str) -> None:
        """Save the in-memory store to a JSON file."""
        if self._use_chroma:
            print("Notice: 'save' is only for in-memory store. ChromaDB is already persistent.")
            return

        import json

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)
        print(f"Store saved to {file_path}")

    def load(self, file_path: str) -> None:
        """Load the in-memory store from a JSON file."""
        if self._use_chroma:
            print("Notice: 'load' is only for in-memory store.")
            return

        import json
        import os

        if not os.path.exists(file_path):
            print(f"No store file found at {file_path}")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            self._store = json.load(f)
        print(f"Store loaded from {file_path} ({len(self._store)} records)")
