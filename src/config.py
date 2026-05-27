"""Central configuration. No hardcoded values elsewhere — read from here."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Agent loop model. claude-opus-4-7 is the most capable; sonnet-4-6 is the
    # cost/latency sweet spot for a customer-facing assistant. Try swapping it
    # to feel the quality/cost tradeoff for yourself.
    agent_model: str = "claude-sonnet-4-6"
    # Cheap model for the eval judge — judging is a simpler task than the agent loop.
    judge_model: str = "claude-haiku-4-5"
    # Voyage embedding model (Anthropic's recommended embeddings provider).
    embed_model: str = "voyage-3"

    max_tokens: int = 4096
    max_agent_turns: int = 8  # hard stop on the tool-use loop to prevent runaways
    top_k: int = 4  # retrieved chunks per query

    # When True, narrate each key step of the agent loop and retrieval to stderr
    # so you can follow the logic. On for the CLI; the eval harness turns it off.
    verbose: bool = True

    corpus_dir: str = os.path.join(os.path.dirname(__file__), "..", "data", "guidelines")


def load() -> Config:
    """Validate environment at the boundary, then return frozen config."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set — see .env.example")
    if not os.environ.get("VOYAGE_API_KEY"):
        raise RuntimeError("VOYAGE_API_KEY not set — see .env.example")
    return Config()
