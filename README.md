# lite-prior-auth — a Claude API learning project

A small, runnable agent that answers clinical prior-authorization and formulary
questions by **retrieving** from a guideline corpus, **calling tools**, and
**refusing/escalating** when it can't ground an answer. It mirrors the kind of
grounded, safety-conscious assistant a pre-sales architect would prototype for a
healthcare or payer customer.

The point of the project is **learning the core Claude API building blocks** by
hand — not the domain. Each concept is implemented in the smallest honest form so
you can read the whole thing in an hour.

---

## The five concepts (and where each lives)

Read them in this order — each builds on the last.

### 1. The Messages API + the manual tool-use loop
`src/agent.py` → `run()`

Everything goes through one endpoint: `client.messages.create()`. You pass
`tools=[...]`; Claude replies with either a final answer or a request to call a
tool. The loop is:

```
call create() → stop_reason == "tool_use"? → run the tool →
append the result as a user turn → call create() again → repeat until end_turn
```

This is the loop the SDK's "tool runner" and Claude Code run *for you*. Writing it
by hand is the fastest way to understand what an agent actually is: a `while` loop
around a stateless API, where you own tool execution.

Two details that trip people up, both visible in `run()`:
- You must append the **entire** assistant turn (`response.content`, including the
  `tool_use` blocks) back into `messages` before sending results — the API is
  stateless and reconstructs context from what you send.
- Every `tool_result` must carry the matching `tool_use_id`, or the next call 400s.

**Go deeper:** https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview.md

### 2. Tool design
`src/tools.py`

A tool is a name + description + JSON-Schema input. Claude chooses tools based
almost entirely on the **descriptions** — so they're written like instructions, not
docstrings (note `search_guidelines` says "ALWAYS call this before answering"). The
three tools model the real shapes you'll see: a retrieval tool, a structured-system
lookup (`check_formulary`), and a **handoff/refusal** tool (`escalate_to_human`).
The escalate tool is the safety story — the model hands off instead of guessing.

`dispatch()` returns error *strings* rather than raising, so the model can read the
error in the next turn and recover — a key agent-design pattern.

**Go deeper:** https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools.md

### 3. RAG (retrieval-augmented generation)
`src/retrieval.py`

"Grounding" means the model answers from retrieved text, not its training data. The
mechanics: embed every document once (`Retriever.from_corpus`), embed the query at
search time, rank by cosine similarity, return the top-k. We use **Voyage**
embeddings (Anthropic's recommended provider) and an in-memory store — deliberately
no vector DB, because the *concept* is identical and the DB is just an
implementation swap. The whole interface the rest of the app depends on is
`Retriever.search(query) -> list[Hit]`.

The honest TODO here: real RAG splits long documents into ~500-token **chunks**
before embedding. This demo does one chunk per file — see `_load_chunks`.

**Go deeper:** embeddings overview at https://docs.voyageai.com — and the chunking/
hybrid-search ideas you already know from Azure AI Search map 1:1.

### 4. Prompt caching
`src/agent.py` → the `system=[{... "cache_control": {"type": "ephemeral"}}]`

The system prompt is identical on every request, so cache it: the first call writes
the cache (~1.25× cost), every later call reads it (~0.1× cost). Caching is a
**prefix match** — anything stable goes early (system prompt, tool list), anything
that changes per request goes late. Verify it's working by reading
`response.usage.cache_read_input_tokens` — if it's zero across repeated calls, a
"silent invalidator" (a timestamp or UUID in the prefix) is breaking it.

This is a direct **TCO lever** — exactly the kind of cost/latency tradeoff a
pre-sales architect speaks to.

**Go deeper:** https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md

### 5. Evaluation
`eval/run_eval.py`

The piece most demos skip and the one the role explicitly asks for: "help customers
develop evaluation frameworks for Claude performance." It scores three things per
labeled case (`eval/cases.json`):
- **tool-call correctness** — did the expected tool get called? (deterministic)
- **appropriate refusal** — did it escalate when it should? (deterministic)
- **grounding quality** — is the answer supported? (LLM-as-judge, using the cheaper
  Haiku model — see `_judge_grounding`)

The pattern to internalize: split deterministic checks (exact assertions) from
fuzzy ones (a model grading a model). That separation is what makes an eval
trustworthy.

**Go deeper:** Anthropic's guidance on building evals — see the "Test & evaluate"
section at https://platform.claude.com/docs (the in-Console **Evaluation tool**
lets you run test cases against prompts without writing harness code).

### Bonus — MCP (Model Context Protocol)
`mcp/server.py`

MCP is the open standard for exposing tools/data to any LLM client over a uniform
protocol. `server.py` wraps `Retriever.search` as an MCP tool with `FastMCP`. Run it
and register it with Claude Code (`claude mcp add lite-prior-auth -- python -m mcp.server`)
and Claude Code itself can call your retrieval layer — the same primitive this
project uses internally, now exposed as a reusable server.

**Go deeper:** https://modelcontextprotocol.io

---

## Architecture / data flow

```
                  ┌──────────────────────────────────────────────┐
   question ─────▶│  agent.run()  (src/agent.py)                  │
                  │  ┌────────────────────────────────────────┐  │
                  │  │ loop:                                   │  │
                  │  │   messages.create(model, tools, system) │──┼──▶ Claude API
                  │  │   stop_reason == "tool_use"?            │  │
                  │  │     ├─ search_guidelines ──▶ Retriever ─┼──┼──▶ Voyage (embeddings)
                  │  │     ├─ check_formulary  ──▶ mock payer  │  │
                  │  │     └─ escalate_to_human ─▶ handoff      │  │
                  │  │   else: return answer + citations        │  │
                  │  └────────────────────────────────────────┘  │
                  └──────────────────────────────────────────────┘
                                       ▲
   eval/run_eval.py ───────────────────┘  (drives run() over labeled cases,
                                            judges grounding with Haiku)
```

Config (models, top-k, turn limit) is centralized in `src/config.py`; nothing else
hardcodes a model ID or setting.

---

## Layout

| Path | Role |
|---|---|
| `src/config.py` | models + settings; validates env at the boundary |
| `src/retrieval.py` | Voyage embeddings + in-memory cosine search |
| `src/tools.py` | tool schemas + client-side dispatch |
| `src/agent.py` | the manual tool-use loop (start here) |
| `src/main.py` | CLI entry point |
| `eval/cases.json` | labeled evaluation cases |
| `eval/run_eval.py` | the eval harness |
| `mcp/server.py` | MCP server wrapping retrieval |
| `data/guidelines/` | the sample corpus (add more docs here) |

---

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY and VOYAGE_API_KEY

python -m src.main "Is adalimumab covered on PLAN_A and does it need prior auth?"
python -m eval.run_eval
```

---

## Suggested learning path

1. Read `src/agent.py` end to end — it's the spine. Trace one full loop.
2. Read `src/tools.py` — see how descriptions drive tool selection.
3. Run the CLI with a question, then with a deliberately out-of-scope one (e.g. a
   dosing question) and watch it escalate.
4. Add `print(response.usage)` in the loop and watch `cache_read_input_tokens` go
   from 0 (first call) to non-zero (later calls).
5. Read `eval/run_eval.py`, then add a 4th case to `eval/cases.json` and run it.
6. Stretch: split the corpus into chunks (`retrieval._load_chunks`), then run the
   MCP server and call it from Claude Code.

---

## Models

Set in `src/config.py`. Defaults: **Sonnet 4.6** for the agent loop (cost/latency
balance), **Haiku 4.5** for the eval judge (a cheaper model is fine for grading).
**Opus 4.7** is the most capable if you want to A/B for quality — change one string.

**Model reference:** https://platform.claude.com/docs/en/about-claude/models/overview.md
