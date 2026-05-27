# lite-prior-auth — a Claude API learning project

I built this to learn the core Claude API building blocks by writing them out by
hand, and I'm sharing it because it's the version I wish I'd had when I started. If
you're coming from using Claude (Claude Code, the chat apps) and want to understand
what's actually happening underneath — how an agent calls tools, grounds answers,
and gets evaluated — this is a short on-ramp.

It's a small, runnable agent that answers clinical prior-authorization and formulary
questions by **retrieving** from a guideline corpus, **calling tools**, and
**escalating** when it can't ground an answer. Don't worry about the healthcare
domain — it's just a realistic shape to hang the concepts on (you'll recognize it if
you've done enterprise or field work). Every piece is written in the smallest honest
form I could manage, so you can read the whole thing in an hour.

My advice: clone it, get it running, then read the five sections below in order.
Each one builds on the last.

---

## The five concepts (and where each lives)

### 1. The Messages API + the manual tool-use loop
`src/agent.py` → `run()`

Start here — this is the spine, and once it clicks the rest is easy. Everything goes
through one endpoint: `client.messages.create()`. You pass `tools=[...]`; Claude
replies with either a final answer or a request to call a tool. The loop is:

```
call create() → stop_reason == "tool_use"? → run the tool →
append the result as a user turn → call create() again → repeat until end_turn
```

The SDK's "tool runner" and Claude Code run this loop *for you*. Writing it by hand
is the fastest way I know to see what an agent really is: a `while` loop around a
stateless API where you own tool execution. Nothing magic.

Two things that tripped me up, both visible in `run()` — watch for them:
- You have to append the **entire** assistant turn (`response.content`, including the
  `tool_use` blocks) back into `messages` before sending results. The API is
  stateless; it rebuilds context only from what you send.
- Every `tool_result` must carry the matching `tool_use_id`, or the next call 400s.

**Go deeper:** https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview.md

### 2. Tool design
`src/tools.py`

A tool is just a name + description + JSON-Schema input. The thing to internalize:
Claude decides *when* to call a tool almost entirely from the **description**, so
write descriptions like instructions, not docstrings (notice `search_guidelines`
literally says "ALWAYS call this before answering"). The three tools here are the
shapes you'll meet everywhere: a retrieval tool, a structured-system lookup
(`check_formulary`), and a **handoff/refusal** tool (`escalate_to_human`). Pay
attention to that last one — letting the model hand off instead of guess is most of
your safety story.

One pattern worth copying: `dispatch()` returns error *strings* instead of raising,
so the model reads the error on the next turn and recovers on its own.

**Go deeper:** https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools.md

### 3. RAG (retrieval-augmented generation)
`src/retrieval.py`

"Grounding" just means the model answers from retrieved text instead of its training
data. The mechanics are simpler than the acronym suggests: embed every document once
(`Retriever.from_corpus`), embed the question at search time, rank by cosine
similarity, return the top-k. I used **Voyage** embeddings (Anthropic's recommended
provider) and an in-memory store on purpose — skip the vector database while you're
learning, because the concept is identical and the DB is just an implementation swap
later. The only interface the rest of the app leans on is
`Retriever.search(query) -> list[Hit]`.

Be honest with yourself about the shortcut here: real RAG splits long documents into
~500-token **chunks** before embedding. This demo does one chunk per file (see
`_load_chunks`) — fixing that is your first good exercise.

**Go deeper:** embeddings overview at https://docs.voyageai.com. If you've used Azure
AI Search or similar, the chunking and hybrid-search ideas map over almost directly.

### 4. Prompt caching
`src/agent.py` → the `system=[{... "cache_control": {"type": "ephemeral"}}]`

The system prompt is identical on every request, so cache it: the first call writes
the cache (~1.25× cost), every later call reads it (~0.1× cost). The mental model to
hold onto is **prefix match** — stable content goes early (system prompt, tool list),
anything that changes per request goes late. Prove to yourself it's working by
reading `response.usage.cache_read_input_tokens`; if it's stuck at zero across
repeated calls, something in your prefix is changing (a timestamp or UUID is the
usual culprit).

This is the first lever I reach for when a bill looks high — worth understanding
early, not as an afterthought.

**Go deeper:** https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md

### 5. Evaluation
`eval/run_eval.py`

This is the piece most demos skip, and in my experience it's what separates a toy
from something you'd actually put in front of a user. It scores three things per
labeled case (`eval/cases.json`):
- **tool-call correctness** — did the expected tool get called? (deterministic)
- **appropriate refusal** — did it escalate when it should? (deterministic)
- **grounding quality** — is the answer supported? (LLM-as-judge, using the cheaper
  Haiku model — see `_judge_grounding`)

The lesson to take away: keep your deterministic checks (exact assertions) separate
from your fuzzy ones (a model grading a model). That separation is what makes an eval
you can trust instead of one you argue with.

**Go deeper:** Anthropic's guidance on building evals — see the "Test & evaluate"
section at https://platform.claude.com/docs (the in-Console **Evaluation tool**
lets you run test cases against prompts without writing harness code).

### Bonus — MCP (Model Context Protocol)
`mcp_server/server.py`

Save this one for last. MCP is the open standard for exposing tools/data to any LLM
client over a uniform protocol. `server.py` wraps `Retriever.search` as an MCP tool
with `FastMCP`. Run it, register it with Claude Code
(`claude mcp add lite-prior-auth -- python -m mcp_server.server`), and Claude Code itself can
call your retrieval layer — the same primitive this project uses internally, now
exposed as a reusable server. Doing this is what made MCP finally click for me: you're
not consuming someone else's server, you're publishing your own.

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

Config (models, top-k, turn limit) lives in `src/config.py` — nothing else hardcodes
a model ID or setting, so that's the one file to look at when you want to change
behavior.

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
| `mcp_server/server.py` | MCP server wrapping retrieval |
| `data/guidelines/` | the sample corpus (add more docs here) |

---

## Run it

You'll need an Anthropic API key and a Voyage API key.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY and VOYAGE_API_KEY

python -m src.main "Is adalimumab covered on PLAN_A and does it need prior auth?"
python -m eval.run_eval
```

---

## How I'd work through it

This is the order I'd recommend — resist the urge to read everything at once.

1. Read `src/agent.py` end to end and trace one full loop. Everything else makes more
   sense once the loop is in your head.
2. Read `src/tools.py` and notice how the descriptions — not the code — drive which
   tool Claude picks.
3. Run the CLI with a normal question, then with a deliberately out-of-scope one (ask
   it for a specific patient's drug dose) and watch it escalate instead of guessing.
4. Drop a `print(response.usage)` into the loop and watch `cache_read_input_tokens`
   jump from 0 on the first call to non-zero on later ones. Seeing it is worth more
   than reading about it.
5. Read `eval/run_eval.py`, then write a 4th case in `eval/cases.json` and run it.
6. When you're ready to stretch: split the corpus into real chunks
   (`retrieval._load_chunks`), then run the MCP server and call it from Claude Code.

If you only do three of these, do 1, 3, and 4.

---

## Models

Set in `src/config.py`. I defaulted to **Sonnet 4.6** for the agent loop (a good
speed/cost balance) and **Haiku 4.5** for the eval judge (grading is a simpler job, so
a cheaper model is fine). **Opus 4.7** is the most capable if you want to A/B for
quality — it's a one-string change. Getting comfortable choosing a model per job is a
skill in itself; this is a low-stakes place to practice it.

**Model reference:** https://platform.claude.com/docs/en/about-claude/models/overview.md

---

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, learn from it.
