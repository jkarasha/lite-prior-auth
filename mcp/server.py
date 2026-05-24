"""MCP server wrapping the retrieval tool — the Sunday stretch goal.

This is the highest-leverage piece: it lets you say "I use Claude Code daily,
AND I exposed my retrieval layer as an MCP server it consumes." That single
sentence closes the Claude-experience gap in an interview.

Run as an stdio MCP server:  python -m mcp.server
Then add it to Claude Code via:  claude mcp add lite-prior-auth -- python -m mcp.server
(run `claude mcp add --help` to confirm the exact flag syntax for your version)

Requires: pip install "mcp[cli]"
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.config import load
from src.retrieval import Retriever

mcp = FastMCP("lite-prior-auth")

# Built once at startup — embedding the corpus is expensive, do it lazily on first call.
_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever.from_corpus(load())
    return _retriever


@mcp.tool()
def search_guidelines(query: str) -> str:
    """Search the clinical guideline corpus and return the most relevant passages."""
    hits = _get_retriever().search(query)
    if not hits:
        return "No matching guidelines found."
    return "\n\n".join(f"[{h.chunk.doc_id}] {h.chunk.text}" for h in hits)


if __name__ == "__main__":
    mcp.run()
