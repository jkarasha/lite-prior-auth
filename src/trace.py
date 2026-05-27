"""A tiny tracing helper so you can *watch* the agent work.

Every key step in the loop calls `trace()`. Lines are tagged by component
(`[agent]`, `[retrieval]`, ...) and printed to **stderr**, which keeps them
separate from the agent's actual answer (that goes to stdout) — so you can pipe
the answer somewhere and still see the trace, or vice versa.

Tracing is controlled by `config.verbose`: on for the CLI so you can follow the
logic, off in the eval harness so its results table stays clean.
"""
from __future__ import annotations

import sys


def trace(enabled: bool, tag: str, msg: str) -> None:
    """Print a tagged trace line to stderr when enabled; a no-op otherwise."""
    if enabled:
        print(f"[{tag}] {msg}", file=sys.stderr)
