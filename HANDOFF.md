# langgraph-factory — Handoff Notes

## What this is

LangGraph pipelines that generate complete Next.js App Router projects from a natural-language spec using locally-hosted LLMs via Docker Model Runner.

Two pipelines:
- **MVP** (`mvp.py`) — single-pass: plan manifest → generate → verify → write → build check
- **Factory** (`factory.py`) — full pipeline: warmup → policy → manifest → generate → write → install → build, with fix loop and regeneration fallback

## Current state (2026-03-10)

**Truncation bug: resolved.** The JSON truncation described in the original handoff was a transient DMR issue. Diagnostic testing confirmed all API calls return `finish_reason=stop` with valid JSON. The `dmr_chat_json()` function now uses streaming with live progress output.

**Pipeline: functional but needs end-to-end validation.** The generate step produces valid output (13 files, `next.config.mjs`, all correct). The fix loop, file-rename support, and stale file cleanup have been added but the full pipeline hasn't completed a successful `pnpm build` yet with the new code.

**Foreman model: upgrading.** The 8B deepseek-r1 echoes input back verbatim in JSON mode — useless for policy generation. The 70B model (`deepseek-r1-distill-llama:70B-Q4_K_M`, 40.5GB) is being pulled. Config already points to it.

### What was fixed
1. **Streaming progress** — `dmr_chat_json()` streams responses with live token count, tok/s, elapsed time.
2. **`next.config.ts` → `.mjs`** — Added constraints to generate/fix prompts. Model was generating `.ts` config which Next.js 14 rejects. Feb 28 working runs used `.mjs`.
3. **Fix node file renames** — Fix schema now supports `"deletions": ["path"]` alongside patches, enabling file renames.
4. **Stale file cleanup** — `write_node` removes files on disk that are no longer in the file map (preserves node_modules, .next, pnpm-lock.yaml).
5. **Manifest step** — New node between policy and generate that plans the file list with descriptions, `imports_from` dependencies, and categories. Includes validation (checks for required files) with retry. Manifest is passed to the monolithic generate step as a structured file plan.
6. **Removed parallel generation** — Investigated parallel file generation (ThreadPoolExecutor, wave-based ordering) but removed it. On a single GPU, llama.cpp parallel slots share compute with no throughput gain, and files generated without seeing each other's content cause import mismatches that cost more to fix than they save.

### Known issues / TODO
- **70B foreman model**: Not yet tested. Need to verify it produces real architecture contracts in JSON mode (unlike the 8B which just echoes input).
- **End-to-end test**: Need a full pipeline run through to `pnpm build` success with the new code.

## Infrastructure

- **Machine**: DGX Spark, NVIDIA GB10 (Grace Hopper), 128GB unified memory, 20 cores
- **Docker**: Engine 28.5.1, linux/arm64 (NOT Docker Desktop)
- **Docker Model Runner**: llama.cpp backend (c55bce4), serves on `http://localhost:12434/engines/v1`
- **Models**:
  - Foreman: `deepseek-r1-distill-llama:70B-Q4_K_M` (40.5GB, 70B params) — upgrading from 8B
  - Coder: `qwen3-coder-next:latest` (45GB, 79.6B params, MoE 512x2.5B)
- **llama.cpp config**: `-ngl 999 --threads 10 --ctx-size 32768 --jinja`, 4 parallel slots, flash_attn enabled
- **pnpm**: Required for build step
- **Memory budget**: 70B (40.5GB) + qwen3-coder (45GB) = 85.5GB of 128GB. DMR loads/unloads on demand.

## File map

```
langgraph_factory/
├── __init__.py       # Exports build_mvp_graph, build_factory_graph
├── config.py         # Env-based config (DMR_BASE_URL, models, retry limits)
├── llm.py            # dmr_chat_json() — streaming requests with progress; langchain_chat() — LangChain wrapper
├── utils.py          # extract_json, normalize_path, log_step, extract_referenced_paths
├── mvp.py            # MVP pipeline graph (simpler, uses langchain_chat)
└── factory.py        # Factory pipeline: policy → manifest → generate → build → fix loop
examples/
├── crud_products.py      # Factory pipeline example
└── crud_products_mvp.py  # MVP pipeline example
tests/
└── test_generate_only.py # Targeted test: policy + generate without build
langgraph_mvp.py              # Original flat MVP script (historical)
langgraph_nextjs_factory.py   # Original flat factory script (historical)
```

## Config (env vars)

| Variable | Default | Notes |
|----------|---------|-------|
| `DMR_BASE_URL` | `http://localhost:12434/engines/v1` | |
| `DMR_API_KEY` | `local-dummy` | |
| `FOREMAN_MODEL` | `docker.io/ai/deepseek-r1-distill-llama:70B-Q4_K_M` | Upgraded from 8B |
| `CODER_MODEL` | `docker.io/ai/qwen3-coder-next:latest` | |
| `FACTORY_OUTPUT_DIR` | `./factory_out` | |
| `MAX_GENERATE_ATTEMPTS` | `2` | |
| `MAX_FIX_ATTEMPTS` | `4` | |

## Token limits (current, in factory.py)

| Node | max_tokens |
|------|-----------|
| warmup | 32 |
| policy | 2000 |
| manifest | 2000 |
| generate | 16000 |
| fix | 8000 |

## Working runs

### Feb 28 (original flat script)
Five runs in `~/lg_factory_out/`. The last one (`crud-products-1772340610`) produced 13 source files (15KB total) and a successful `pnpm build`. Used max_tokens=6000, `next.config.mjs`, `no_ui_imports` strategy.

### Mar 10 (restructured package)
- `test_generate_only.py` — successful: policy + generate produced 13 files with `next.config.mjs`, valid JSON, 4,296 tokens in 396s (10.8 tok/s)
- Full pipeline — not yet successful. First run hit `next.config.ts` fix loop (now fixed). Second run hung due to llama.cpp request queue contention (concurrent runs).
