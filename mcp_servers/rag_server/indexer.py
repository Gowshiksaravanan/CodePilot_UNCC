"""
Document Indexer
Loads Python documentation, chunks it, embeds with sentence-transformers,
and stores in ChromaDB. Only runs once — skips if vector DB already exists.
"""


def index_documents(docs_path: str, db_path: str):
    """TODO: Implement document loading, chunking, embedding, and storage."""
    pass


def is_indexed(db_path: str) -> bool:
    """TODO: Check if ChromaDB already has indexed documents."""
    return False
