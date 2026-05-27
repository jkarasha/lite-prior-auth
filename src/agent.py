"""The manual tool-use loop — the heart of the project.

This is the loop the SDK's tool runner and Claude Code run for you under the hood.
Writing it out by hand once is the fastest way to understand what an agent actually
is: a loop around a stateless API where you own tool execution. Start your reading
here.

Pattern (from the Anthropic SDK docs):
  1. call messages.create with tools
  2. if stop_reason != "tool_use", we're done — return the text
  3. otherwise: append the assistant turn, execute each tool_use block,
     append the tool_result blocks as a user turn, loop
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import anthropic

from .config import Config
from .retrieval import Retriever
from .tools import TOOL_DEFS, dispatch
from .trace import trace

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
    #
    # TODO: this only caches the static prefix (tools + system). The message
    # history below is append-only, so it shares a growing identical prefix
    # across turns — an ideal caching target. Add a rolling cache_control
    # breakpoint on messages[-1] each turn to read prior turns at ~0.1x instead
    # of re-billing the whole history at full price every iteration (O(n) vs
    # O(n^2) input tokens over the loop). Skipped for now to keep the loop simple.
    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    messages: list[dict] = [{"role": "user", "content": question}]
    tool_calls: list[str] = []
    escalated = False

    trace(config.verbose, "agent", f'question: "{question}"')

    for turn in range(1, config.max_agent_turns + 1):
        trace(config.verbose, "agent",
              f"turn {turn}/{config.max_agent_turns} → calling Claude (model={config.agent_model})")
        response = client.messages.create(
            model=config.agent_model,
            max_tokens=config.max_tokens,
            system=system,
            tools=TOOL_DEFS,
            messages=messages,
        )

        # Watch cache read climb from 0 on the first call to non-zero on later
        # ones — that's the system-prompt cache paying off (README §4).
        u = response.usage
        trace(config.verbose, "agent",
              f"← stop_reason={response.stop_reason}  "
              f"(in={u.input_tokens} out={u.output_tokens}, "
              f"cache write={u.cache_creation_input_tokens or 0} "
              f"read={u.cache_read_input_tokens or 0})")

        if response.stop_reason != "tool_use":
            answer = next((b.text for b in response.content if b.type == "text"), "")
            trace(config.verbose, "agent", "no tool requested → returning final answer")
            return AgentResult(answer=answer, tool_calls=tool_calls, escalated=escalated)

        # Preserve the full assistant turn (text + tool_use blocks) in history.
        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_calls.append(block.name)
            trace(config.verbose, "agent", f"tool_use: {block.name}({json.dumps(block.input)})")
            if block.name == "escalate_to_human":
                escalated = True
            output = dispatch(block.name, block.input, retriever)
            preview = " ".join(output.split())
            if len(preview) > 120:
                preview = preview[:120] + "…"
            trace(config.verbose, "agent", f"  → result: {preview}")
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )
        # Feed every tool_result back as a user turn, then loop.
        messages.append({"role": "user", "content": results})

    trace(config.verbose, "agent", "hit max agent turns without a final answer")
    return AgentResult(
        answer="Stopped: exceeded max agent turns without a final answer.",
        tool_calls=tool_calls,
        escalated=escalated,
    )
