"""
Checkpointing
Persistent checkpointer built on LangGraph's in-memory saver, persisted to disk.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver


class PersistentMemorySaver(InMemorySaver):
    """
    A simple persistent checkpointer that pickles the in-memory checkpoint store to disk.

    This keeps us aligned with the design doc's "pause/resume across sessions" intent
    without requiring external DB dependencies.
    """

    def __init__(self, path: str):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = pickle.loads(self.path.read_bytes())
            storage = data.get("storage")
            writes = data.get("writes")
            if storage is not None:
                self.storage = storage
            if writes is not None:
                self.writes = writes
        except Exception:
            # Corrupt checkpoints should not break the agent
            return

    def _save(self) -> None:
        try:
            payload = {"storage": self.storage, "writes": self.writes}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(payload))
            os.replace(tmp, self.path)
        except Exception:
            return

    def put(self, *args, **kwargs):
        out = super().put(*args, **kwargs)
        self._save()
        return out

    def put_writes(self, *args, **kwargs):
        out = super().put_writes(*args, **kwargs)
        self._save()
        return out

