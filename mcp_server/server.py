"""MCP server wrapping the retrieval tool — save this for last.

Doing this is what makes MCP click: instead of consuming someone else's server,
you publish your own. It exposes the same `Retriever.search` this project uses
internally as an MCP tool that any client — including Claude Code — can call.

Run as an stdio MCP server:  python -m mcp_server.server
Then add it to Claude Code via:  claude mcp add lite-prior-auth -- python -m mcp_server.server
(run `claude mcp add --help` to confirm the exact flag syntax for your version)

Note the package is named `mcp_server`, not `mcp`: a local `mcp/` directory would
shadow the installed `mcp` PyPI package, so `python -m mcp.server` would silently run
the library's entrypoint instead of this file. Never name a package after a dependency.

Requires: pip install "mcp[cli]"
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.config import load
from src.retrieval import Retriever

mcp = FastMCP("lite-prior-auth")

# Embedding the corpus is expensive, so build the retriever lazily on the first
# tool call and reuse it for the lifetime of the server process.
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
