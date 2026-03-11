# langgraph-factory

LangGraph pipeline that generates complete Next.js App Router projects from a natural-language spec, using locally-hosted LLMs via Docker Model Runner. No external API dependencies.

Addresses a known failure mode of AI coding tools — context window breakdown on complex, multi-file tasks — by decomposing generation into discrete, bounded agent nodes with an automated build-fix-regenerate loop that recovers from failures without human intervention.

## Pipeline

```
policy (gpt-oss) → manifest (qwen3-coder) → generate → write → install → build
                                                                           ↓
                                                                   review (gpt-oss)
                                                                     ↓         ↓
                                                              fix (qwen3) → regenerate
```

### Nodes

| Node | Model | Purpose |
|------|-------|---------|
| **policy** | gpt-oss 20B | Produces an architecture contract (project layout, entity schema, routes, UI strategy) |
| **manifest** | qwen3-coder | Plans the file list with descriptions and import dependencies |
| **generate** | qwen3-coder | Generates all project files in a single call using fence format |
| **write** | — | Writes files to disk, removes stale files from prior runs |
| **install** | — | Runs `pnpm install` with auto-recovery from 404 packages |
| **build** | — | Runs `pnpm build` |
| **review** | gpt-oss 20B | Evaluates build failures and decides strategy: fix, regenerate, or fail |
| **fix** | qwen3-coder | Applies targeted fixes guided by the reviewer's diagnosis |

### Recovery layers

1. **Post-generate sanitizer** — deterministic fixes applied before the first build (next.config.ts rename, missing root layout injection, import reconciliation)
2. **Mechanical fixes** — pattern-matched build errors fixed instantly without LLM calls (missing npm packages, missing type imports, incomplete prop interfaces)
3. **Reviewer-guided LLM fixes** — gpt-oss diagnoses the error and gives specific guidance to the coder
4. **Spin detection** — if the same file is patched 2+ times, forces regeneration instead of continuing to fix
5. **Failure-informed regeneration** — error patterns from failed attempts are fed into the next generation prompt

## Setup

```bash
pip install -r requirements.txt
```

Requires:
- [Docker Model Runner](https://docs.docker.com/desktop/features/model-runner/) or any OpenAI-compatible endpoint
- `pnpm` for the build step

### Models

| Role | Default | Size | Notes |
|------|---------|------|-------|
| Foreman | `docker.io/ai/gpt-oss:20B` | ~11GB Q4 | Architecture policy, build failure review |
| Coder | `docker.io/ai/qwen3-coder-next:latest` | ~45GB Q4 | Manifest, generation, fixes. 128k context. |

## Usage

```python
from langgraph_factory import build_factory_graph

graph = build_factory_graph()
result = graph.invoke({"spec": "A CRUD app for managing products...", "project_dir": "runs/my_run"})
```

See `examples/` for complete examples. Run all specs:

```bash
./run_all_specs.sh
```

## Configuration

| Variable | Default | Notes |
|----------|---------|-------|
| `DMR_BASE_URL` | `http://localhost:12434/engines/v1` | Docker Model Runner endpoint |
| `DMR_API_KEY` | `local-dummy` | |
| `FOREMAN_MODEL` | `docker.io/ai/gpt-oss:20B` | Architecture policy + review |
| `CODER_MODEL` | `docker.io/ai/qwen3-coder-next:latest` | Code generation and fixes |
| `FACTORY_RUNS_DIR` | `./runs` | Output directory for runs |
| `MAX_GENERATE_ATTEMPTS` | `2` | Full regeneration attempts |
| `MAX_FIX_ATTEMPTS` | `4` | Build-fix loop iterations per generation |
| `LINT_MAX_LOOPS` | `10` | Max total build attempts across all generations |

## Project structure

```
langgraph_factory/
├── __init__.py      # Package exports
├── config.py        # Environment-based configuration
├── llm.py           # LLM client wrappers (streaming, stats tracking)
├── utils.py         # Shared utilities (fence parser, logging)
├── mvp.py           # MVP pipeline (simpler, single-pass)
└── factory.py       # Factory pipeline (full build-fix-review loop)
examples/
├── crud_products.py      # Products CRUD spec
├── blog_markdown.py      # Blog with markdown rendering spec
└── crud_products_mvp.py  # MVP pipeline example
tests/
└── test_generate_only.py # Targeted test: policy + generate without build
run_all_specs.sh          # Run all example specs and collect results
```

## Run output

Each run creates a timestamped directory under `runs/` containing the generated project and a `summary.txt` with per-step timing, model stats, and build attempt history.
