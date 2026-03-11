# langgraph-factory

LangGraph pipeline that generates complete Next.js App Router projects from a natural-language spec, using locally-hosted LLMs via Docker Model Runner. No external API dependencies — runs entirely on local hardware.

Addresses a known failure mode of AI coding tools — context window breakdown on complex, multi-file tasks — by decomposing generation into discrete, bounded agent nodes with an automated build-fix-regenerate loop that recovers from failures without human intervention.

## Pipeline

```
policy (gpt-oss) → scaffold (create-next-app) → manifest (qwen3-coder) → review (gpt-oss)
    → generate (qwen3-coder) → write → install → build
                                                    ↓
                                            review (gpt-oss)
                                              ↓         ↓
                                       fix (qwen3) → regenerate
```

### Nodes

| Node | Model | Purpose |
|------|-------|---------|
| **policy** | gpt-oss 20B | Produces architecture contract (layout, entities, routes, library usage notes) |
| **scaffold** | — | Runs `npx create-next-app@14` to create a working project skeleton |
| **manifest** | qwen3-coder | Plans app-specific files only (scaffold provides config/boilerplate) |
| **review_manifest** | gpt-oss 20B | Reviews file plan, trims unnecessary files before generation |
| **generate** | qwen3-coder | Generates app files in a single call, merges on top of scaffold |
| **write** | — | Writes files to disk, removes stale files from prior runs |
| **install** | — | Runs `pnpm install` with auto-recovery from 404 packages |
| **build** | — | Runs `pnpm build` |
| **review** | gpt-oss 20B | Evaluates build failures and decides strategy: fix, regenerate, or fail |
| **fix** | qwen3-coder | Applies targeted fixes guided by the reviewer's diagnosis |

### Recovery layers

1. **Post-generate sanitizer** — deterministic fixes before the first build (next.config.ts rename, missing root layout, import reconciliation)
2. **Mechanical fixes** — pattern-matched build errors fixed instantly without LLM calls (missing npm packages, missing type imports, incomplete prop interfaces)
3. **Reviewer-guided LLM fixes** — gpt-oss diagnoses the error and gives specific guidance to the coder
4. **Spin detection** — if the same file is patched 2+ times, forces regeneration instead of continuing to fix
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

## Prerequisites

- **Docker Model Runner** with GPU support, or any OpenAI-compatible endpoint at `localhost:12434`
- **Models pulled** (these are publicly available via Docker Model Runner on any platform):
  ```bash
  docker model pull ai/gpt-oss:20B
  docker model pull ai/qwen3-coder-next:latest
  ```
- **Node.js** and **pnpm** (for scaffold and build steps)
- **Python 3.12+**

To verify your setup:
```bash
# Check Docker Model Runner is running with GPU
docker model list
# Check pnpm is available
pnpm --version
# Check models respond
curl http://localhost:12434/engines/v1/models
```

## Setup

```bash
pip install -r requirements.txt
```

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

## Project structure

```
langgraph_factory/
├── __init__.py      # Package exports
├── config.py        # Environment-based configuration
├── llm.py           # LLM client wrappers (streaming, retry, stats)
├── utils.py         # Shared utilities (fence parser, logging)
├── mvp.py           # MVP pipeline (simpler single-pass, historical)
└── factory.py       # Full pipeline (scaffold → policy → manifest → review → generate → build → fix)
examples/
├── crud_products.py          # Products CRUD spec
├── blog_markdown.py          # Blog with markdown rendering spec
├── crud_products_mvp.py      # MVP pipeline example (historical)
└── sample_output_blog.txt    # Sample run output
tests/
└── test_generate_only.py     # Smoke test: policy + generate without build
run_all_specs.sh              # Run all example specs and collect results
```

> **Note on `mvp.py`**: This was the original single-pass pipeline used during early development. It's preserved for reference but `factory.py` is the active pipeline with the full build-fix-review loop.

## Run output

Each run creates a timestamped directory under `runs/` containing:
- The complete generated Next.js project (buildable with `pnpm build`)
- `run.log` — full pipeline log (LLM calls, timings, reviewer verdicts, fix details)
- `summary.txt` — per-step timing, model stats, and build attempt history
