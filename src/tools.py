"""Tool definitions (JSON schema) and client-side dispatch.

Three tools mirror a real prior-auth assistant:
  - search_guidelines: grounding via RAG (the safety-critical one)
  - check_formulary:   structured lookup into a mock payer system
  - escalate_to_human: the refusal/handoff path when the model can't ground an answer
"""
from __future__ import annotations

import json

from .retrieval import Retriever

# Sent to Claude as the `tools` parameter. Descriptions matter — Claude uses
# them to decide when to call each tool, so write them carefully.
TOOL_DEFS = [
    {
        "name": "search_guidelines",
        "description": (
            "Search the clinical guideline and prior-authorization corpus. "
            "ALWAYS call this before answering a clinical or coverage question — "
            "answers must be grounded in retrieved text, not prior knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_formulary",
        "description": "Check whether a drug is on a given plan's formulary and its tier.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drug": {"type": "string"},
                "plan_id": {"type": "string"},
            },
            "required": ["drug", "plan_id"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Hand off to a human reviewer when the answer cannot be grounded in the "
            "guidelines or the request is outside scope. Prefer this over guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why a human is needed"}
            },
            "required": ["reason"],
        },
    },
]

# Mock payer data — stand-in for an EHR/formulary integration.
_MOCK_FORMULARY = {
    ("adalimumab", "PLAN_A"): {"covered": True, "tier": 3, "prior_auth_required": True},
    ("metformin", "PLAN_A"): {"covered": True, "tier": 1, "prior_auth_required": False},
}


def dispatch(name: str, tool_input: dict, retriever: Retriever) -> str:
    """Execute one tool call, returning a string for the tool_result block.

    Returns an error string (not raises) so the agent can recover in-loop.
    """
    try:
        if name == "search_guidelines":
            hits = retriever.search(tool_input["query"])
            if not hits:
                return "No matching guidelines found."
            return "\n\n".join(f"[{h.chunk.doc_id}] {h.chunk.text}" for h in hits)

        if name == "check_formulary":
            key = (tool_input["drug"].lower(), tool_input["plan_id"])
            entry = _MOCK_FORMULARY.get(key)
            return json.dumps(entry) if entry else "Drug not found on this plan's formulary."

        if name == "escalate_to_human":
            return f"ESCALATED: {tool_input['reason']}"

        return f"Error: unknown tool '{name}'"
    except KeyError as e:
        return f"Error: missing required argument {e}"
