"""
Fusion Retriever
Loads persisted RAG artifacts and returns fused BM25 + vector results
using Reciprocal Rank Fusion.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from chromadb import PersistentClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .fusion import order_fusion_scores, reciprocal_rank_fusion


DEFAULT_DB_PATH = "data"
DEFAULT_COLLECTION = "python_docs"
DEFAULT_TOP_K = 5
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RETRIEVER_K = 60


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Run fused retrieval and return JSON-like payload expected by the MCP tool.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    db_path = Path(DEFAULT_DB_PATH)
    bm25_data = _load_bm25_state(db_path)
    collection = _load_chroma_collection(db_path)

    vector_k = max(top_k, 10)
    bm25_k = max(top_k, 10)

    bm25_results = _bm25_retrieve(query, bm25_data, top_k=bm25_k)
    vector_results = _vector_retrieve(query, collection, top_k=vector_k)

    fused = reciprocal_rank_fusion(
        [ [r["id"] for r in bm25_results], [r["id"] for r in vector_results] ],
        k=DEFAULT_RETRIEVER_K,
    )
    ranked = order_fusion_scores(fused, top_k=top_k)

    chunk_text_by_id = {}
    chunk_source_by_id = {}

    for item in bm25_results:
        chunk_text_by_id[item["id"]] = item["chunk"]
        chunk_source_by_id[item["id"]] = item["source"]

    for item in vector_results:
        if item["id"] not in chunk_text_by_id:
            chunk_text_by_id[item["id"]] = item["chunk"]
        if item["id"] not in chunk_source_by_id:
            chunk_source_by_id[item["id"]] = item["source"]

    results: list[dict] = []
    for chunk_id, score in ranked:
        chunk = chunk_text_by_id.get(chunk_id, "")
        source = chunk_source_by_id.get(chunk_id, "")
        if not chunk:
            continue
        results.append(
            {
                "chunk": chunk,
                "source": source,
                "score": float(round(score, 6)),
            }
        )

    return {"results": results}


def fusion_retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Backward-compatible alias for retrieve.
    """
    return retrieve(query=query, top_k=top_k)


def _load_bm25_state(db_path: Path) -> dict:
    index_path = db_path / "bm25" / "index.pkl"
    corpus_path = db_path / "bm25" / "chunks.json"

    if not index_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            "BM25 artifacts are missing. Please run indexer.py ingestion first "
            f"to generate bm25 artifacts under {db_path / 'bm25'}."
        )

    with index_path.open("rb") as fp:
        bm25 = pickle.load(fp)

    if not isinstance(bm25, BM25Okapi):
        raise TypeError(f"BM25 artifact is not a BM25Okapi object: {type(bm25)}")

    with corpus_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    required_keys = {"chunk_ids", "chunk_texts", "chunk_sources"}
    if not required_keys.issubset(data.keys()):
        raise ValueError(
            f"BM25 chunks.json is missing expected keys: {sorted(required_keys)}"
        )

    return {
        "bm25": bm25,
        "chunk_ids": data["chunk_ids"],
        "chunk_texts": data["chunk_texts"],
        "chunk_sources": data["chunk_sources"],
    }


def _load_chroma_collection(db_path: Path):
    try:
        client = PersistentClient(path=str(db_path))
        collection = client.get_collection(name=DEFAULT_COLLECTION)
        count = collection.count()
    except Exception as exc:
        raise FileNotFoundError(
            "Chroma collection is not available. Please run indexer.py ingestion first "
            f"to create collection '{DEFAULT_COLLECTION}' in {db_path}."
        ) from exc

    if count <= 0:
        raise FileNotFoundError(
            f"Chroma collection '{DEFAULT_COLLECTION}' is empty at {db_path}."
        )

    return collection


def _bm25_retrieve(
    query: str,
    bm25_state: dict,
    top_k: int,
) -> list[dict]:
    bm25 = bm25_state["bm25"]
    chunk_ids = bm25_state["chunk_ids"]
    chunk_texts = bm25_state["chunk_texts"]
    chunk_sources = bm25_state["chunk_sources"]

    tokens = _tokenize_for_bm25(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    scored = list(enumerate(scores))
    scored.sort(key=lambda item: (-item[1], item[0]))

    results: list[dict] = []
    for idx, score in scored[:top_k]:
        if idx >= len(chunk_ids):
            continue
        results.append(
            {
                "id": chunk_ids[idx],
                "chunk": chunk_texts[idx],
                "source": chunk_sources[idx],
                "score": float(score),
            }
        )

    return results


def _vector_retrieve(
    query: str,
    collection,
    top_k: int,
) -> list[dict]:
    query_embedding = _embed_query(query)
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    output: list[dict] = []
    if not ids:
        return output

    for idx, chunk_id in enumerate(ids):
        distance = float(distances[idx]) if idx < len(distances) else 0.0
        score = 1.0 / (1.0 + max(0.0, distance))
        source = ""
        if metas and idx < len(metas) and isinstance(metas[idx], dict):
            source = metas[idx].get("source", "")
        chunk = docs[idx] if docs and idx < len(docs) else ""
        output.append(
            {
                "id": chunk_id,
                "chunk": chunk,
                "source": source,
                "score": float(score),
            }
        )
    return output


def _embed_query(query: str) -> np.ndarray:
    model = SentenceTransformer(DEFAULT_EMBED_MODEL)
    embedding = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding[0]


def _tokenize_for_bm25(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [token for token in tokens if len(token) > 1]
