# langgraph-factory

LangGraph pipelines that generate complete Next.js App Router projects from a natural-language spec, using locally-hosted LLMs via Docker Model Runner. Addresses a known failure mode of AI coding tools — context window breakdown on complex, multi-file tasks — by decomposing generation into discrete, bounded agent nodes with an automated build-fix-regenerate loop that recovers from failures without human intervention. No external API dependencies.

## Pipelines

**MVP** (`build_mvp_graph`) — Single-pass generation with manifest planning and retry on missing files.

**Factory** (`build_factory_graph`) — Full pipeline with architecture policy, manifest planning, build-fix loop, and regeneration fallback. Flow:

```
warmup → policy → manifest → generate → write → install → build
                                                            ↓
                                                    fix ← (fail?)
                                                            ↓
                                                    regenerate (if fixes exhausted)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # edit if needed
```

Requires [Docker Model Runner](https://docs.docker.com/desktop/features/model-runner/) or any OpenAI-compatible endpoint. Default models:

- **Foreman** (architecture policy): `deepseek-r1-distill-llama:70B-Q4_K_M`
- **Coder** (manifest, generation, fixes): `qwen3-coder-next`

Also requires `pnpm` for the build step.

## Usage

```python
from langgraph_factory import build_factory_graph

graph = build_factory_graph()
result = graph.invoke({"spec": "A CRUD app for managing products..."})
```

See `examples/` for complete examples.

## Configuration

| Variable | Default | Notes |
|----------|---------|-------|
| `DMR_BASE_URL` | `http://localhost:12434/engines/v1` | Docker Model Runner endpoint |
| `DMR_API_KEY` | `local-dummy` | |
| `FOREMAN_MODEL` | `docker.io/ai/deepseek-r1-distill-llama:70B-Q4_K_M` | Architecture policy |
| `CODER_MODEL` | `docker.io/ai/qwen3-coder-next:latest` | Code generation and fixes |
| `FACTORY_OUTPUT_DIR` | `./factory_out` | |
| `MAX_GENERATE_ATTEMPTS` | `2` | Full regeneration attempts |
| `MAX_FIX_ATTEMPTS` | `4` | Build-fix loop iterations |

## Project Structure

```
langgraph_factory/
├── __init__.py      # Package exports
├── config.py        # Environment-based configuration
├── llm.py           # LLM client wrappers (streaming, progress output)
├── utils.py         # Shared utilities
├── mvp.py           # MVP pipeline (simpler, single-pass)
└── factory.py       # Factory pipeline (policy → manifest → generate → build-fix loop)
examples/
├── crud_products.py      # Factory pipeline example
└── crud_products_mvp.py  # MVP pipeline example
tests/
└── test_generate_only.py # Targeted test: policy + generate without build
```

## History

The original flat scripts (`langgraph_mvp.py`, `langgraph_nextjs_factory.py`) are preserved in the repo root to show the evolution from prototype to package.
