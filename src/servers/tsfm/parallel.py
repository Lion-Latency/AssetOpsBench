"""Parallel-execution helper for TSFM preprocessing fan-out.

Provides an `executor()` context manager and a `map_or_serial()` helper so
call sites can use the same code path whether parallelism is enabled or
not. Toggled via env vars so the benchmarking harness flips modes without
code changes.

Env vars:
  TSFM_PREPROCESS_OPT       "1" enables parallel execution.
  TSFM_PREPROCESS_WORKERS   Worker count (default 4).
  TSFM_PREPROCESS_EXECUTOR  "thread" (default) or "process".
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Iterable, List, Optional

logger = logging.getLogger("tsfm-mcp-server.parallel")


def opt_enabled() -> bool:
    return os.environ.get("TSFM_PREPROCESS_OPT", "0") == "1"


def workers() -> int:
    try:
        return max(1, int(os.environ.get("TSFM_PREPROCESS_WORKERS", "4")))
    except ValueError:
        return 4


def executor_kind() -> str:
    return os.environ.get("TSFM_PREPROCESS_EXECUTOR", "thread").lower()


@contextmanager
def executor():
    """Yield a configured Executor or None when parallelism disabled.

    Callers branch on the yielded value; `map_or_serial` is the preferred
    consumer.
    """
    if not opt_enabled() or workers() <= 1:
        yield None
        return
    cls = ProcessPoolExecutor if executor_kind() == "process" else ThreadPoolExecutor
    with cls(max_workers=workers()) as ex:
        yield ex


def map_or_serial(fn: Callable[[Any], Any], items: Iterable[Any], ex) -> List[Any]:
    """Apply fn to each item; parallel if `ex` is not None, serial otherwise.

    Preserves item order. Exceptions propagate from the worker.
    """
    items = list(items)
    if ex is None:
        return [fn(x) for x in items]
    return list(ex.map(fn, items))


def mode_tag() -> str:
    """Cache-key namespace tag — keeps parallel-path artifacts separate
    from serial-path artifacts when the produced object differs."""
    return f"opt{int(opt_enabled())}"
