"""CLI entry point. Usage: python -m src.main "is adalimumab covered on PLAN_A?" """
from __future__ import annotations

import sys

from .agent import run
from .config import load
from .retrieval import Retriever


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print('Usage: python -m src.main "<question>"', file=sys.stderr)
        return 1

    config = load()
    retriever = Retriever.from_corpus(config)
    result = run(argv[1], config, retriever)

    print(result.answer)
    print(f"\n[tools: {', '.join(result.tool_calls) or 'none'}"
          f"{' | ESCALATED' if result.escalated else ''}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
