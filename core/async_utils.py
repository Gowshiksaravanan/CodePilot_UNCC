"""
Async utilities — safe wrapper for running coroutines from sync code.
Uses a dedicated background thread with its own event loop for MCP operations.
"""

import asyncio
import threading


_loop = None
_thread = None
_lock = threading.Lock()


def _ensure_loop():
    """Ensure a background event loop thread is running."""
    global _loop, _thread
    if _loop is not None and _loop.is_running():
        return
    with _lock:
        if _loop is not None and _loop.is_running():
            return
        _loop = asyncio.new_event_loop()
        _thread = threading.Thread(target=_loop.run_forever, daemon=True)
        _thread.start()


def run_async(coro):
    """Run an async coroutine safely from sync code.
    Uses a dedicated background event loop thread to avoid cancel scope issues."""
    _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()
