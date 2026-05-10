"""MCP stdio client wrapper for the L2 transport benchmark path.

Spawns `tsfm-mcp-server` as a subprocess and routes tool calls to it via
the official MCP client. Used by `benchmark_runner.py --transport stdio`
to capture FastMCP dispatch + pydantic + JSON-RPC framing overhead that
the in-process path skips entirely.

Per-call server-side stage breakdowns flow back via a JSONL file under
`TSFM_BENCH_REPORT_DIR` that `_emit_metrics` appends to and this client
tails after each call. Result objects are wrapped in `SimpleNamespace`
so existing bench code reads `.status`/`.results_file`/`.error` unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


def shim_result(data: Dict[str, Any]) -> SimpleNamespace:
    """Wrap a tool-response dict so existing bench code can use attr access."""
    return SimpleNamespace(**data)


class StdioBenchClient:
    """Persistent MCP-stdio session against a spawned `tsfm-mcp-server`.

    Lifetime should match a single benchmark mode: opt-flag env vars are
    captured at server-process startup, and lru_cache state carries across
    calls within the session (matching the L1 path's `apply_mode`->cache.clear
    behavior, which fires on each new mode).
    """

    def __init__(
        self,
        env_overrides: Optional[Dict[str, str]] = None,
        server_cmd: str = "tsfm-mcp-server",
        server_args: Optional[List[str]] = None,
    ):
        self.env_overrides = env_overrides or {}
        self.server_cmd = server_cmd
        self.server_args = server_args or []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session = None
        self._stdio_cm = None
        self._client_cm = None
        self._report_dir: Optional[Path] = None
        self._jsonl_path: Optional[Path] = None
        self._jsonl_pos: int = 0
        self.init_ms: float = 0.0

    def __enter__(self) -> "StdioBenchClient":
        self._report_dir = Path(tempfile.mkdtemp(prefix="tsfm_bench_reports_"))
        self._jsonl_path = self._report_dir / "reports.jsonl"

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = dict(os.environ)
        env.update(self.env_overrides)
        env["TSFM_BENCH_REPORT_DIR"] = str(self._report_dir)

        params = StdioServerParameters(
            command=self.server_cmd,
            args=self.server_args,
            env=env,
        )

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _start():
            self._stdio_cm = stdio_client(params)
            reader, writer = await self._stdio_cm.__aenter__()
            self._client_cm = ClientSession(reader, writer)
            self._session = await self._client_cm.__aenter__()
            t0 = time.perf_counter()
            await self._session.initialize()
            return (time.perf_counter() - t0) * 1000.0

        self.init_ms = self._loop.run_until_complete(_start())
        return self

    def __exit__(self, exc_type, exc, tb):
        async def _stop():
            try:
                if self._client_cm is not None:
                    await self._client_cm.__aexit__(None, None, None)
            finally:
                if self._stdio_cm is not None:
                    await self._stdio_cm.__aexit__(None, None, None)

        try:
            if self._loop is not None:
                self._loop.run_until_complete(_stop())
        except Exception:
            pass
        finally:
            if self._loop is not None:
                self._loop.close()
            if self._report_dir is not None:
                shutil.rmtree(self._report_dir, ignore_errors=True)

    def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Send one MCP tool call. Blocks the calling thread."""
        async def _do():
            resp = await self._session.call_tool(tool_name, args)
            ok = not getattr(resp, "isError", False)
            text = resp.content[0].text if resp.content else "{}"
            return ok, text

        ok, text = self._loop.run_until_complete(_do())
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"status": "error", "error": text[:400]}
        if not ok and "status" not in data:
            data["status"] = "error"
        return data

    def read_latest_report(self) -> Dict[str, Any]:
        """Tail the per-call JSONL, returning the most-recent server report.

        Returns {} when the server didn't write one (e.g. tool errored
        before `_emit_metrics` was called).
        """
        if self._jsonl_path is None or not self._jsonl_path.exists():
            return {}
        try:
            with open(self._jsonl_path) as f:
                f.seek(self._jsonl_pos)
                lines = f.readlines()
                self._jsonl_pos = f.tell()
            if not lines:
                return {}
            return json.loads(lines[-1])
        except Exception:
            return {}
