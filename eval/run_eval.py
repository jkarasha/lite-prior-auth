"""Evaluation harness — the part most tutorials skip, and the part that matters.

Once you've got the agent running, this is how you tell whether it's actually any
good. It scores three things per labeled case:
  1. tool-call correctness  — did the agent call the expected tool? (exact, deterministic)
  2. appropriate refusal     — did it escalate when it should have? (deterministic)
  3. grounding quality       — is the answer supported by retrieved text? (LLM-as-judge)

The lesson worth keeping: separate deterministic checks from a model grading a model.

Run: python -m eval.run_eval
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace

import anthropic

from src.agent import AgentResult, run
from src.config import Config, load
from src.retrieval import Retriever

CASES_PATH = os.path.join(os.path.dirname(__file__), "cases.json")

_JUDGE_PROMPT = """You are grading a prior-auth assistant's answer for GROUNDING only.
Question: {question}
Answer: {answer}
Grading note: {note}

Does the answer stay grounded (cites guidelines / formulary, no fabricated coverage
rules, escalates when uncertain)? Reply with exactly "PASS" or "FAIL: <one-line reason>"."""


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    tools_ok: bool
    escalation_ok: bool
    grounding_ok: bool

    @property
    def passed(self) -> bool:
        return self.tools_ok and self.escalation_ok and self.grounding_ok


def _judge_grounding(case: dict, result: AgentResult, config: Config) -> bool:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=config.judge_model,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": _JUDGE_PROMPT.format(
                question=case["question"], answer=result.answer, note=case["grading_note"]
            ),
        }],
    )
    verdict = next((b.text for b in response.content if b.type == "text"), "")
    return verdict.strip().upper().startswith("PASS")


def _score_case(case: dict, config: Config, retriever: Retriever) -> CaseScore:
    result = run(case["question"], config, retriever)
    tools_ok = all(t in result.tool_calls for t in case["expect_tools"])
    escalation_ok = result.escalated == case["expect_escalation"]
    grounding_ok = _judge_grounding(case, result, config)
    return CaseScore(case["id"], tools_ok, escalation_ok, grounding_ok)


def main() -> int:
    # Tracing off here so the results table below stays clean. Drop the
    # `replace(..., verbose=False)` to watch each case run through the loop.
    config = replace(load(), verbose=False)
    retriever = Retriever.from_corpus(config)
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    scores = [_score_case(c, config, retriever) for c in cases]

    print(f"{'case':<24} tools  escalate  grounding  result")
    for s in scores:
        print(f"{s.case_id:<24} {_m(s.tools_ok)}     {_m(s.escalation_ok)}        "
              f"{_m(s.grounding_ok)}        {'PASS' if s.passed else 'FAIL'}")
    passed = sum(1 for s in scores if s.passed)
    print(f"\n{passed}/{len(scores)} cases passed")
    return 0 if passed == len(scores) else 1


def _m(ok: bool) -> str:
    return " ok " if ok else "MISS"


if __name__ == "__main__":
    raise SystemExit(main())
