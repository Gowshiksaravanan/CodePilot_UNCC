import numpy as np
from pathlib import Path

import pytest

from mcp_servers.rag_server import indexer, retriever


def test_load_documents_only_reads_supported_extensions(tmp_path: Path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "readme.md").write_text("Python docs content", encoding="utf-8")
    (docs_root / "index.py").write_text("def hello():\n    pass", encoding="utf-8")
    (docs_root / "ignore.bin").write_text("ignore", encoding="utf-8")

    docs = indexer.load_documents(str(docs_root))
    sources = {Path(source).name for source, _ in docs}

    assert sources == {"readme.md", "index.py"}
    assert all(text for _, text in docs)


class _DummyCollection:
    def __init__(self):
        self.ids = []
        self.documents = []
        self.metadatas = []

    def add(self, ids, documents, embeddings, metadatas):
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def count(self):
        return len(self.ids)


class _DummyClient:
    def __init__(self, _path: str):
        self.collection = _DummyCollection()

    def get_or_create_collection(self, name: str):
        return self.collection


class _DummyEmbedModel:
    def encode(self, sentences, **_kwargs):
        # Return stable small embeddings without loading a real sentence-transformer model.
        return np.tile(np.array([1.0, 0.0], dtype=float), (len(sentences), 1))


def test_index_documents_creates_artifacts(tmp_path, monkeypatch):
    docs_root = tmp_path / "docs"
    db_root = tmp_path / "db"
    docs_root.mkdir()
    db_root.mkdir()
    (docs_root / "stdlib.md").write_text(
        "Python supports decorators. Decorators change function behavior.",
        encoding="utf-8",
    )

    dummy_client = _DummyClient(str(db_root))
    monkeypatch.setattr(indexer, "PersistentClient", lambda path: dummy_client)
    monkeypatch.setattr(indexer, "SentenceTransformer", lambda model_name: _DummyEmbedModel())

    result = indexer.index_documents(str(docs_root), str(db_root))

    assert result["status"] == "indexed"
    assert result["count"] >= 1
    assert result["db_path"] == str(db_root)
    assert (db_root / "bm25" / "chunks.json").exists()
    assert (db_root / "bm25" / "index.pkl").exists()


def test_retrieve_shapes_and_order_via_fusion(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "_bm25_retrieve",
        lambda query, bm25_state, top_k: [
            {"id": "b1", "chunk": "bm25 one", "source": "bm25-src", "score": 0.8},
            {"id": "b2", "chunk": "bm25 two", "source": "bm25-src", "score": 0.4},
        ],
    )
    monkeypatch.setattr(
        retriever,
        "_vector_retrieve",
        lambda query, collection, top_k: [
            {"id": "b2", "chunk": "vector two", "source": "vector-src", "score": 0.9},
            {"id": "v3", "chunk": "vector three", "source": "vector-src", "score": 0.7},
        ],
    )
    monkeypatch.setattr(retriever, "_load_bm25_state", lambda db_path: {})
    monkeypatch.setattr(retriever, "_load_chroma_collection", lambda db_path: None)

    payload = retriever.retrieve("function", top_k=3)

    assert payload.keys() == {"results"}
    assert len(payload["results"]) == 3
    assert payload["results"][0]["chunk"] == "bm25 two"
    assert {"chunk", "source", "score"} <= set(payload["results"][0].keys())
    assert isinstance(payload["results"][0]["score"], float)


def test_retrieve_raises_on_empty_query():
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_retrieve_reports_missing_bm25_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever, "DEFAULT_DB_PATH", str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError):
        retriever.retrieve("decorator")
