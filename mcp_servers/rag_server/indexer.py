"""
Document Indexer
Load documentation, build semantic chunks, create embeddings, and store in
ChromaDB. A small BM25 artifact bundle is also written for the retrieval phase.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from chromadb import PersistentClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "python_docs"


def load_documents(docs_path: str) -> list[tuple[str, str]]:
    """Load supported documentation files recursively from docs_path."""
    root = Path(docs_path)
    if not root.exists():
        raise FileNotFoundError(f"docs_path not found: {docs_path}")
    if not root.is_dir():
        raise ValueError(f"docs_path must be a directory: {docs_path}")

    exts = {".md", ".rst", ".txt", ".py"}
    docs: list[tuple[str, str]] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in exts:
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        text = _normalize_text(text)
        if text:
            docs.append((str(file_path), text))
    return docs


def index_documents(docs_path: str, db_path: str):
    """
    One-time ingestion:
    1) load docs
    2) semantic chunking
    3) embed chunks with sentence-transformers
    4) persist in ChromaDB
    5) persist BM25 tokenized corpus/index
    """
    if is_indexed(db_path):
        return {
            "status": "already_indexed",
            "count": _collection_count(db_path),
            "db_path": str(Path(db_path)),
        }

    docs = load_documents(docs_path)
    if not docs:
        raise ValueError(f"No supported docs found in {docs_path}")

    model = SentenceTransformer(DEFAULT_MODEL)
    chunk_records = _chunk_documents(docs, model)
    if not chunk_records:
        raise ValueError("No chunks created from documents.")

    chunks = [c["text"] for c in chunk_records]
    embeddings = model.encode(
        chunks,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    db_dir = Path(db_path)
    db_dir.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(name=DEFAULT_COLLECTION)

    collection.add(
        ids=[c["id"] for c in chunk_records],
        documents=chunks,
        embeddings=[e.astype(float).tolist() for e in embeddings],
        metadatas=[c["metadata"] for c in chunk_records],
    )

    try:
        _persist_bm25_artifacts(chunk_records, db_dir)
    except Exception:
        # BM25 persistence is optional for step 1.
        # Retrieval can rebuild from chunk metadata if needed.
        pass

    return {
        "status": "indexed",
        "count": len(chunk_records),
        "db_path": str(db_dir),
        "collection": DEFAULT_COLLECTION,
    }


def _persist_bm25_artifacts(chunk_records: list[dict], db_dir: Path) -> None:
    """
    Persist BM25 artifacts so retrieval can load them without reparsing docs.
    """
    tokenized = [_tokenize_for_bm25(record["text"]) for record in chunk_records]
    bm25 = BM25Okapi(tokenized)
    (db_dir / "bm25").mkdir(exist_ok=True)

    with (db_dir / "bm25" / "index.pkl").open("wb") as fp:
        pickle.dump(bm25, fp)

    with (db_dir / "bm25" / "chunks.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "chunk_ids": [record["id"] for record in chunk_records],
                "chunk_texts": [record["text"] for record in chunk_records],
                "chunk_sources": [record["metadata"].get("source", "") for record in chunk_records],
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )


def _chunk_documents(
    docs: list[tuple[str, str]],
    model: SentenceTransformer,
) -> list[dict]:
    chunk_records: list[dict] = []
    for source, text in docs:
        chunks = _semantic_chunks(text, model)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{_stable_id(source)}-{chunk_idx}"
            chunk_records.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "source": source,
                        "chunk_index": chunk_idx,
                        "chars": len(chunk),
                    },
                }
            )
    return chunk_records


def _stable_id(value: str) -> str:
    # Avoid python's random hash seed issues by making IDs deterministic.
    import hashlib

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return digest


def _semantic_chunks(
    text: str,
    model: SentenceTransformer,
    *,
    min_chunk_chars: int = 700,
    max_chunk_chars: int = 2400,
    threshold: float = 0.78,
    sentence_batch: int = 32,
) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()]

    sentence_embs = model.encode(
        sentences,
        batch_size=sentence_batch,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    chunks: list[str] = []
    current_sents: list[str] = []
    current_embs: list[np.ndarray] = []
    current_chars = 0

    for sentence, emb in zip(sentences, sentence_embs):
        if not current_sents:
            current_sents = [sentence]
            current_embs = [emb]
            current_chars = len(sentence)
            continue

        centroid = np.mean(np.stack(current_embs, axis=0), axis=0)
        similarity = float(np.dot(centroid, emb))
        next_chars = current_chars + 1 + len(sentence)

        if (similarity < threshold and current_chars >= min_chunk_chars) or (
            next_chars > max_chunk_chars and current_sents
        ):
            chunk_text = " ".join(current_sents).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_sents = [sentence]
            current_embs = [emb]
            current_chars = len(sentence)
        else:
            current_sents.append(sentence)
            current_embs.append(emb)
            current_chars = next_chars

    if current_sents:
        chunk_text = " ".join(current_sents).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n{2,}", text)
    if len(parts) <= 1:
        parts = [line.strip() for line in text.split("\n") if line.strip()]
    return [part.strip() for part in parts if part.strip()]


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _tokenize_for_bm25(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [t for t in tokens if len(t) > 1]


def _collection_count(db_path: str, collection_name: str = DEFAULT_COLLECTION) -> int:
    try:
        client = PersistentClient(path=str(Path(db_path)))
        collection = client.get_collection(name=collection_name)
        return collection.count()
    except Exception:
        return 0


def is_indexed(db_path: str) -> bool:
    """Check if Chroma already has at least one vector in the target collection."""
    return _collection_count(db_path) > 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Python docs into Chroma + BM25.")
    parser.add_argument(
        "--docs-path",
        required=True,
        help="Directory containing documentation files",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Directory for persistent Chroma/BM25 artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = index_documents(docs_path=args.docs_path, db_path=args.db_path)
    except Exception as exc:  # intentionally broad: CLI should be explicit and simple
        print(f"Failed to run ingestion: {exc}")
        sys.exit(1)

    print(
        "Indexing complete: "
        f"status={result['status']} count={result['count']} db_path={result['db_path']}"
    )


if __name__ == "__main__":
    main()
