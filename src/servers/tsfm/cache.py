"""Preprocessing cache for the TSFM MCP server.

Two-tier cache (in-memory LRU + optional on-disk pickle) used to skip
deterministic, expensive preprocessing steps (data quality filter,
TimeSeriesPreprocessor + get_datasets).

Toggled at runtime via environment variables so the benchmarking harness
can compare baseline / cache-only / combined modes without code changes.

Env vars:
  TSFM_CACHE_ENABLED   "1" enables cache; any other value disables it.
  TSFM_CACHE_MAX_ITEMS Max in-memory entries before LRU eviction (default 32).
  TSFM_CACHE_DIR       Optional directory for on-disk pickle persistence.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger("tsfm-mcp-server.cache")

_LOCK = threading.Lock()
_MEM: "OrderedDict[str, Any]" = OrderedDict()


def enabled() -> bool:
    return os.environ.get("TSFM_CACHE_ENABLED", "0") == "1"


def _max_items() -> int:
    try:
        return int(os.environ.get("TSFM_CACHE_MAX_ITEMS", "32"))
    except ValueError:
        return 32


def _disk_dir() -> Optional[Path]:
    d = os.environ.get("TSFM_CACHE_DIR")
    return Path(d) if d else None


def _stable_repr(obj: Any) -> bytes:
    """Deterministic byte representation for hashing config dicts."""
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return b"{" + b",".join(_stable_repr(k) + b":" + _stable_repr(v) for k, v in items) + b"}"
    if isinstance(obj, (list, tuple)):
        return b"[" + b",".join(_stable_repr(x) for x in obj) + b"]"
    return repr(obj).encode()


def make_key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(_stable_repr(p))
        h.update(b"|")
    return h.hexdigest()[:24]


def file_fingerprint(path: str) -> Tuple[str, int, int]:
    """Cheap content fingerprint: path, mtime_ns, size."""
    st = os.stat(path)
    return (path, st.st_mtime_ns, st.st_size)


def df_fingerprint(df) -> Tuple[Any, ...]:
    """Fingerprint a DataFrame by shape, columns, and head/tail values.

    Used when an upstream cache miss already produced a DataFrame and we
    want to key the next stage on its content without rehashing every row.
    """
    try:
        head = tuple(df.head(1).to_records(index=False).tolist())
        tail = tuple(df.tail(1).to_records(index=False).tolist())
    except Exception:
        head = tail = ()
    return (df.shape, tuple(df.columns), head, tail)


def get(key: str) -> Any:
    if not enabled():
        return None
    with _LOCK:
        if key in _MEM:
            _MEM.move_to_end(key)
            return _MEM[key]
    d = _disk_dir()
    if d is not None:
        p = d / f"{key}.pkl"
        if p.exists():
            try:
                with open(p, "rb") as f:
                    v = pickle.load(f)
                with _LOCK:
                    _MEM[key] = v
                return v
            except Exception as exc:
                logger.warning("cache disk read failed for %s: %s", key, exc)
    return None


def put(key: str, value: Any) -> None:
    if not enabled():
        return
    with _LOCK:
        _MEM[key] = value
        while len(_MEM) > _max_items():
            _MEM.popitem(last=False)
    d = _disk_dir()
    if d is not None:
        try:
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{key}.pkl", "wb") as f:
                pickle.dump(value, f)
        except Exception as exc:
            logger.warning("cache disk write failed for %s: %s", key, exc)


def clear() -> None:
    """Drop all in-memory entries. Disk entries left intact."""
    with _LOCK:
        _MEM.clear()


def stats() -> dict:
    with _LOCK:
        return {"mem_entries": len(_MEM), "max_items": _max_items(), "enabled": enabled()}
