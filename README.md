# langgraph-factory

A reference implementation showing how to use [LangGraph](https://github.com/langchain-ai/langgraph) to build a multi-agent code generation pipeline with locally-hosted LLMs.

The pipeline generates complete Next.js App Router projects from a natural-language spec. It demonstrates several patterns that are useful when building LLM-powered code generation systems:

- **Multi-model orchestration** — a "foreman" model (gpt-oss 20B) handles planning and review, while a "coder" model (qwen3-coder) handles generation and fixes
- **Decomposed generation** — instead of asking one model to produce an entire project in a single prompt, the pipeline breaks the task into discrete nodes: architecture policy, file manifest, manifest review, code generation
- **Automated build-fix loops** — when `pnpm build` fails, a reviewer diagnoses the error and a coder applies targeted fixes, with spin detection to avoid infinite loops
- **Mechanical error recovery** — common build failures (missing packages, unused imports, config file issues) are fixed deterministically without burning LLM calls
- **Failure-informed regeneration** — when fixes aren't enough, the pipeline regenerates with error context from the failed attempt

Everything runs locally on [Docker Model Runner](https://docs.docker.com/model-runner/) — no API keys, no cloud dependencies, no per-token costs.

## How it works

```
policy (gpt-oss) → scaffold (create-next-app) → manifest (qwen3-coder) → review (gpt-oss)
    → generate (qwen3-coder) → write → install → build
                                                    ↓
                                            review (gpt-oss)
                                              ↓         ↓
                                       fix (qwen3) → regenerate
```

Each box is a LangGraph node. The pipeline is defined in `factory.py` using `StateGraph` — edges between nodes are explicit, and conditional routing handles the build-fix-regenerate loop. State flows through the graph as a typed dict.

| Node | What it does |
|------|-------------|
| **policy** | Foreman produces an architecture contract: layout, entities, routes, library usage notes |
| **scaffold** | Runs `npx create-next-app@14` to create a working skeleton with correct config |
| **manifest** | Coder plans which app-specific files to generate (pages, API routes, lib) |
| **review_manifest** | Foreman reviews the plan, trims unnecessary files |
| **generate** | Coder produces all app files in one call using a fence format, merged on top of scaffold |
| **write / install / build** | Deterministic steps: write to disk, `pnpm install`, `pnpm build` |
| **review** | Foreman evaluates build failures — decides: fix, regenerate, or give up |
| **fix** | Coder applies targeted patches guided by the reviewer's diagnosis |

### Recovery layers

The pipeline has several layers that handle failures, applied in order:

1. **Post-generate sanitizer** — deterministic fixes before the first build (next.config.ts rename, missing root layout, import reconciliation)
2. **Mechanical fixes** — pattern-matched build errors fixed without LLM calls (missing npm packages, unused imports/variables, incomplete prop interfaces)
3. **Reviewer-guided LLM fixes** — foreman diagnoses the error, coder applies the fix
4. **Spin detection** — if the same file is patched 2+ times, forces regeneration
5. **Failure-informed regeneration** — error patterns from failed attempts are fed into the next generation prompt

## Sample output

From `examples/blog_markdown.py` — blog platform with markdown rendering, admin CRUD, and comments:

```
Total pipeline time: 266.3s
Generate attempts: 1  |  Fix attempts: 1  |  Build attempts: 2
Result: BUILD OK

Step                     Time   Tokens    tok/s Model              Notes
--------------------------------------------------------------------------
policy                  27.1s      361     13.3 gpt-oss
scaffold                 3.1s                                      13 files
manifest                68.0s      640      9.4 qwen3-coder-next   14 planned
review_manifest         29.5s       86      2.9 gpt-oss
generate                95.5s     4010     42.0 qwen3-coder-next   24 files
install                  1.6s                                      OK
build                    6.4s                                      FAILED
review                   3.6s      111     31.1 gpt-oss
fix                     24.2s      716     29.5 qwen3-coder-next   2 files patched
build                    7.3s                                      OK
```

Build failure was 2 ESLint errors (`let` → `const`). Reviewer diagnosed in 3.6s, coder fixed in 24s.

Full sample output: [`examples/sample_output_blog.txt`](examples/sample_output_blog.txt)

## Running it yourself

### Prerequisites

- **Docker Model Runner** with GPU support (or any OpenAI-compatible endpoint at `localhost:12434`)
- **Node.js** and **pnpm**
- **Python 3.12+**

Pull the models:
```bash
docker model pull ai/gpt-oss:20B
docker model pull ai/qwen3-coder-next:latest
```

Verify your setup:
```bash
docker model list
pnpm --version
curl http://localhost:12434/engines/v1/models
```

### Install and run

```bash
git clone <this-repo> langgraph-factory
cd langgraph-factory
pip install -r requirements.txt
```

```python
from langgraph_factory import build_factory_graph

graph = build_factory_graph()
result = graph.invoke({"spec": "A CRUD app for managing products...", "project_dir": "runs/my_run"})
```

See `examples/` for complete specs. Run all of them:

```bash
./run_all_specs.sh
```

### Configuration

All config is via environment variables — see `config.py` for defaults:

| Variable | Default | Notes |
|----------|---------|-------|
| `DMR_BASE_URL` | `http://localhost:12434/engines/v1` | Docker Model Runner endpoint |
| `FOREMAN_MODEL` | `docker.io/ai/gpt-oss:20B` | Planning + review model |
| `CODER_MODEL` | `docker.io/ai/qwen3-coder-next:latest` | Code generation + fixes |
| `MAX_GENERATE_ATTEMPTS` | `2` | Full regeneration budget |
| `MAX_FIX_ATTEMPTS` | `4` | Fix cycles per generation attempt |

## Project structure

```
langgraph_factory/
├── factory.py       # The pipeline — all nodes, routing, and recovery logic
├── llm.py           # LLM client (streaming, retry, token stats)
├── utils.py         # Fence parser, JSON extraction, logging
├── config.py        # Environment-based configuration
└── __init__.py
examples/
├── blog_markdown.py          # Blog with markdown rendering spec
├── crud_products.py          # Products CRUD spec
└── sample_output_blog.txt    # What a successful run looks like
tests/
├── test_factory_functions.py # Unit tests for recovery/validation logic
└── test_utils.py             # Unit tests for parsing and extraction
run_all_specs.sh              # Run all example specs and collect results
```

## What this is (and isn't)

This is a **reference implementation** — a worked example of how to wire up LangGraph, local LLMs, and deterministic tooling into a pipeline that produces working code. It demonstrates the patterns; it's not a production platform.

Things worth studying:
- How `factory.py` decomposes a complex task into bounded LLM calls
- How the build-fix loop uses a reviewer model to guide a coder model
- How mechanical fixes avoid wasting LLM calls on deterministic problems
- How the scaffold node anchors generation on a known-good starting point
- How failure context flows back into regeneration prompts

The hardware used for development was an NVIDIA DGX Spark (GB10 GPU, 128GB unified RAM) running Docker Model Runner. The models are large — qwen3-coder is 80B parameters (~45GB Q4) and gpt-oss is 20B (~11GB Q4) — so you need substantial GPU memory to run both. The pipeline just talks to an OpenAI-compatible HTTP endpoint, so you can swap in different models or backends (Ollama, LM Studio, etc.) by pointing `DMR_BASE_URL` at it in `config.py`.
