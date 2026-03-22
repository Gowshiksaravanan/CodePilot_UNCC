from mcp_servers.rag_server import server


def test_query_python_docs_success(monkeypatch):
    expected = {
        "results": [
            {"chunk": "def hello(): ...", "source": "stdlib.md", "score": 0.87}
        ]
    }
    monkeypatch.setattr(server, "retrieve", lambda query, top_k: expected)

    result = server.query_python_docs("decorator")

    assert result == expected


def test_query_python_docs_invalid_query():
    result = server.query_python_docs("   ")

    assert result["results"] == []
    assert "error" in result
    assert "non-empty" in result["error"]


def test_query_python_docs_missing_artifacts(monkeypatch):
    monkeypatch.setattr(server, "retrieve", lambda query, top_k: (_ for _ in ()).throw(
        FileNotFoundError("BM25 artifacts are missing")
    ))

    result = server.query_python_docs("list")

    assert result["results"] == []
    assert result["error"] == "BM25 artifacts are missing"


def test_query_python_docs_unexpected_error(monkeypatch):
    monkeypatch.setattr(server, "retrieve", lambda query, top_k: (_ for _ in ()).throw(
        RuntimeError("boom")
    ))

    result = server.query_python_docs("list")

    assert result["results"] == []
    assert result["error"] == "retrieval failed: boom"
