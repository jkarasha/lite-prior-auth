"""The manual tool-use loop — the credibility piece.

This is the loop the Claude API runs for you under the hood. Writing it by hand
is what turns "I use Claude Code daily" into "I've built tool use into a stack."

Pattern (from the Anthropic SDK docs):
  1. call messages.create with tools
  2. if stop_reason != "tool_use", we're done — return the text
  3. otherwise: append the assistant turn, execute each tool_use block,
     append the tool_result blocks as a user turn, loop
"""
from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from .config import Config
from .retrieval import Retriever
from .tools import TOOL_DEFS, dispatch

SYSTEM_PROMPT = """You are a prior-authorization assistant for a health plan.

Rules:
- Ground every clinical or coverage claim in retrieved guidelines. Call \
search_guidelines before answering; cite the [doc_id] you relied on.
- Use check_formulary for any drug-coverage question.
- If you cannot ground an answer in the guidelines, call escalate_to_human \
instead of guessing. Patient safety depends on not fabricating coverage rules.
- Be concise. State the decision, then the citation."""


@dataclass(frozen=True)
class AgentResult:
    answer: str
    tool_calls: list[str] = field(default_factory=list)  # ordered tool names invoked
    escalated: bool = False


def run(question: str, config: Config, retriever: Retriever) -> AgentResult:
    client = anthropic.Anthropic()

    # cache_control on the system prompt: it's stable across every request, so
    # cache it once and pay ~0.1x on subsequent calls. Verify via
    # response.usage.cache_read_input_tokens.
    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    messages: list[dict] = [{"role": "user", "content": question}]
    tool_calls: list[str] = []
    escalated = False

    for _ in range(config.max_agent_turns):
        response = client.messages.create(
            model=config.agent_model,
            max_tokens=config.max_tokens,
            system=system,
            tools=TOOL_DEFS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            answer = next((b.text for b in response.content if b.type == "text"), "")
            return AgentResult(answer=answer, tool_calls=tool_calls, escalated=escalated)

        # Preserve the full assistant turn (text + tool_use blocks) in history.
        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_calls.append(block.name)
            if block.name == "escalate_to_human":
                escalated = True
            output = dispatch(block.name, block.input, retriever)
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )
        messages.append({"role": "user", "content": results})

    return AgentResult(
        answer="Stopped: exceeded max agent turns without a final answer.",
        tool_calls=tool_calls,
        escalated=escalated,
    )
