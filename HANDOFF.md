# langgraph-factory — Handoff Notes

## What this is

LangGraph pipeline that generates complete Next.js App Router projects from a natural-language spec using locally-hosted LLMs via Docker Model Runner on an NVIDIA DGX Spark.

## Current state (2026-03-11)

**Pipeline is working.** Both specs (products CRUD, blog with markdown) build successfully. The blog spec typically succeeds in 1-2 generate attempts with 1-4 fix cycles. Products CRUD is more reliable (usually first attempt).

**Two-model architecture:**
- **Foreman** (gpt-oss:20B) — architecture policy, manifest review, build failure review
- **Coder** (qwen3-coder-next) — manifest generation, code generation, code fixes

### Pipeline flow

```
policy → scaffold → manifest → review_manifest → generate → write → install → build
                                                                         ↓
                                                              [fail] ← review ← [build failed]
                                                                         ↓
                                                                    fix → write → install → build
                                                                         ↓ (spin detected)
                                                                    regenerate → write → ...
```

### Recovery layers (in order)

1. **Post-generate sanitizer** — deterministic fixes (next.config.ts→mjs, missing layout)
2. **Import reconciliation** — scans source for npm imports not in package.json, adds them
3. **Mechanical fixes** — pattern-matched build errors fixed without LLM (missing props, missing imports, missing packages)
4. **Reviewer-guided LLM fixes** — gpt-oss diagnoses the error, provides guidance, coder applies fix
5. **Spin detection** — if same file patched 2+ times, forces full regeneration
6. **Failure-informed regeneration** — retry includes error patterns and reviewer reasoning from previous attempt

### Key features added this session

- **Scaffold node**: Runs `npx create-next-app@14` before manifest/generate. Coder builds on top of a working skeleton instead of generating config from scratch.
- **Manifest review**: gpt-oss reviews the file plan, trims unnecessary files (e.g. UI wrappers when scaffold exists).
- **Manifest-constrained generation**: Generate prompt includes the exact file list — "generate EXACTLY these files, no more, no fewer."
- **Scaffold-aware manifest**: When scaffold exists, manifest only plans app-specific files (pages, API routes, lib, components). No config files, no generic UI wrappers.
- **Full logging**: All stdout output also written to `{project_dir}/run.log` via `tee_print()`.
- **HTTP retry**: LLM calls retry up to 3x with backoff on HTTP/connection errors.
- **Markdown fence stripping**: Coder sometimes wraps output in ` ```tsx ` inside `===FILE:===` blocks — now stripped automatically.
- **Protected files**: Fix node cannot delete package.json, tsconfig.json, next.config.mjs.
- **Install recovery**: On npm 404 errors, removes bad packages from package.json, strips their imports from source, retries.

### Recurring issues (model behavior)

- **shadcn/Radix muscle memory**: Qwen3-Coder has a strong prior toward shadcn patterns — `asChild` prop, `@radix-ui/*` imports, `components/ui/` wrappers. Addressed via prompt constraints and manifest enforcement.
- **`marked()` returns `Promise<string>`**: The `marked` library's API changed; coder consistently generates synchronous usage. Usually fixed in one reviewer-guided cycle.
- **Markdown code fences in output**: Coder wraps file contents in ` ```tsx ` inside the fence format. Now stripped by `parse_fenced_files()`.
- **Stray characters at file start**: Related to fence leakage — causes syntax errors on first build.

## Infrastructure

- **Machine**: NVIDIA DGX Spark — 20-core ARM (Grace), GB10 GPU (6,000+ CUDA cores), 128GB unified/shared RAM
- **OS**: Linux (Ubuntu-based), kernel 6.11.0-1016-nvidia
- **Docker**: nvidia set as default runtime (`/etc/docker/daemon.json`) for GPU persistence
- **Docker Model Runner**: llama.cpp backend at `http://localhost:12434/engines/v1`, OpenAI-compatible API
- **GPU throughput**: ~35-44 tok/s (qwen3-coder), ~9-35 tok/s (gpt-oss, variable due to thinking tokens)
- **pnpm**: Required for build step
- **Node.js**: Required for `npx create-next-app` scaffold

## File map

```
langgraph_factory/
├── __init__.py       # Exports build_mvp_graph, build_factory_graph
├── config.py         # Env-based config (DMR_BASE_URL, models, retry limits)
├── llm.py            # dmr_chat_json(), dmr_chat_raw() — streaming with progress & retry
├── utils.py          # parse_fenced_files, extract_json, tee_print, log_step/log_detail
├── mvp.py            # MVP pipeline graph (simpler, uses langchain_chat)
└── factory.py        # Full pipeline: policy → scaffold → manifest → review → generate → build → fix loop
examples/
├── crud_products.py      # Products CRUD spec (passing)
├── crud_products_mvp.py  # MVP pipeline example
└── blog_markdown.py      # Blog with markdown + comments spec (passing)
run_all_specs.sh          # Runs all example specs, reports pass/fail
```

## Config (env vars)

| Variable | Default | Notes |
|----------|---------|-------|
| `DMR_BASE_URL` | `http://localhost:12434/engines/v1` | Docker Model Runner |
| `DMR_API_KEY` | `local-dummy` | |
| `FOREMAN_MODEL` | `docker.io/ai/gpt-oss:20B` | Policy, review |
| `CODER_MODEL` | `docker.io/ai/qwen3-coder-next:latest` | Manifest, generate, fix |
| `FACTORY_RUNS_DIR` | `./runs` | Parent dir for timestamped run outputs |
| `MAX_GENERATE_ATTEMPTS` | `2` | Full regeneration budget |
| `MAX_FIX_ATTEMPTS` | `4` | Fix cycles per generation attempt |

## Token limits (current, in factory.py)

| Node | max_tokens | Model |
|------|-----------|-------|
| policy | 3,000 | gpt-oss |
| manifest | 4,000 | qwen3-coder |
| manifest review | 8,000 | gpt-oss |
| generate | 65,536 | qwen3-coder |
| build review | 2,000 | gpt-oss |
| fix | 16,000 | qwen3-coder |

## Best run (blog_markdown)

```
Total pipeline time: 286.7s
Generate attempts: 1  |  Fix attempts: 2  |  Build attempts: 3
Result: BUILD OK
```

## Next priorities

1. **Scaffold node**: Working but needs more testing. Eliminates config file generation errors.
2. **More specs**: Need 3-5 specs passing to prove generalization. Currently: products CRUD (passing), blog markdown (passing).
3. **Scaffold CLI node**: User wants `npx create-next-app` + optionally `npx shadcn@latest init` as a graph node. Scaffold node is implemented; shadcn init is not yet added.
