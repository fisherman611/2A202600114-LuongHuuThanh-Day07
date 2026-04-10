from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Split on ". ", "! ", "? " or ".\n"
        sentences = re.split(r"(?<=\. |! |\? |\.\n)", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            # Fallback to fixed size if no separators left
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        # Split current text by this separator
        if sep == "":
            splits = list(current_text)
        else:
            parts = current_text.split(sep)
            splits = []
            for i, p in enumerate(parts):
                if i < len(parts) - 1:
                    p += sep
                splits.append(p)

        # Process fragments that are still too large
        final_fragments = []
        for fragment in splits:
            if len(fragment) <= self.chunk_size:
                final_fragments.append(fragment)
            else:
                final_fragments.extend(self._split(fragment, next_seps))

        # Merge segments into final chunks
        merged_chunks = []
        current_chunk = ""
        for frag in final_fragments:
            if not current_chunk:
                current_chunk = frag
            elif len(current_chunk) + len(frag) <= self.chunk_size:
                current_chunk += frag
            else:
                merged_chunks.append(current_chunk)
                current_chunk = frag
        if current_chunk:
            merged_chunks.append(current_chunk)
        return merged_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        results = {}
        
        # Fixed Size Chunker
        fixed = FixedSizeChunker(chunk_size=chunk_size)
        f_chunks = fixed.chunk(text)
        results["fixed_size"] = {
            "count": len(f_chunks),
            "avg_length": sum(len(c) for c in f_chunks) / len(f_chunks) if f_chunks else 0,
            "chunks": f_chunks,
        }

        # Sentence Chunker
        sentence = SentenceChunker(max_sentences_per_chunk=3)
        s_chunks = sentence.chunk(text)
        results["by_sentences"] = {
            "count": len(s_chunks),
            "avg_length": sum(len(c) for c in s_chunks) / len(s_chunks) if s_chunks else 0,
            "chunks": s_chunks,
        }

        # Recursive Chunker
        recursive = RecursiveChunker(chunk_size=chunk_size)
        r_chunks = recursive.chunk(text)
        results["recursive"] = {
            "count": len(r_chunks),
            "avg_length": sum(len(c) for c in r_chunks) / len(r_chunks) if r_chunks else 0,
            "chunks": r_chunks,
        }

        return results
