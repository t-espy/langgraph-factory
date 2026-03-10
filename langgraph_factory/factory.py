"""Full factory pipeline: policy -> manifest -> generate -> build -> fix loop.

Evolution of the MVP pipeline. Adds:
- Architecture policy step (foreman produces a contract before generation)
- Model warmup (avoid cold-start latency during generation)
- Manifest step (plan file list with validation/retry before generating)
- Build-fix loop (iterative patching from build errors)
- Package.json hash tracking (skip reinstall when deps unchanged)
- Full regeneration fallback when fixes aren't enough
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from langgraph_factory.config import (
    CODER_MODEL,
    FOREMAN_MODEL,
    MAX_FIX_ATTEMPTS,
    MAX_GENERATE_ATTEMPTS,
    OUTPUT_DIR,
)
from langgraph_factory.llm import dmr_chat_json, dmr_chat_raw
from langgraph_factory.utils import (
    extract_referenced_paths,
    log_detail,
    log_step,
    normalize_path,
    parse_fenced_files,
)


class FactoryState(TypedDict, total=False):
    spec: str
    architecture_contract: dict
    acceptance: list[str]
    manifest: list[dict]
    files: dict[str, str]
    last_build_ok: bool
    last_build_log: str
    generate_attempt: int
    fix_attempt: int
    project_dir: str
    build_attempt: int
    package_json_hash: str
    last_installed_package_json_hash: str
    last_patched_files: list[str]


def _require_spec(state: FactoryState) -> str:
    spec = state.get("spec")
    if not spec:
        raise RuntimeError("Missing spec in state")
    return spec


def _require_project_dir(state: FactoryState) -> str:
    project_dir = state.get("project_dir")
    if not project_dir:
        raise RuntimeError("Missing project_dir in state")
    return project_dir


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_cmd(cmd: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )


def _log_build_failure(
    build_log: str, build_attempt: int, patched_files: list[str],
) -> None:
    log_detail(f"Build attempt {build_attempt} failed")
    lines = (build_log or "").splitlines()
    head = lines[:60]
    tail = lines[-200:] if len(lines) > 200 else lines
    print("[detail] build log (first 60 lines)")
    print("\n".join(head))
    print("[detail] build log (last 200 lines)")
    print("\n".join(tail))
    if patched_files:
        log_detail(f"Patched files: {', '.join(patched_files)}")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def warmup_node(state: FactoryState) -> dict:
    """Warm the coder model to avoid cold-start latency during generation."""
    log_step("Warmup model")
    try:
        dmr_chat_json(
            model=CODER_MODEL, system="Return JSON only.",
            user='{"ok":"warmup"}', max_tokens=32, temperature=0.0,
            label="warmup",
        )
    except Exception:
        pass  # warmup is opportunistic
    return {}


def policy_node(state: FactoryState) -> dict:
    """Generate an architecture contract from the app spec."""
    log_step("Generate architecture policy")
    system = (
        "You are a senior software architect. "
        "Return STRICT JSON only."
    )
    user = json.dumps({
        "task": "Produce an architecture_contract for a Next.js App Router CRUD app generator.",
        "app_spec": _require_spec(state),
        "requirements": [
            "Do NOT output a file list. Output only the architecture contract.",
            "Prefer Next.js App Router + TypeScript.",
            "Package manager must be pnpm.",
            "In-memory storage acceptable for MVP.",
            "Goal: pnpm build must pass.",
        ],
        "output_schema": {
            "architecture_contract": {
                "project_layout": "src_app_router | app_router",
                "typescript": "boolean",
                "package_manager": "pnpm",
                "import_aliases": {"@/*": "./src/*"},
                "ui_strategy": {
                    "mode": "no_ui_imports | local_primitives",
                    "allowed_import_prefixes": ["@/components/ui/"],
                    "primitives": [
                        "button", "input", "label", "card",
                        "table", "textarea", "select",
                    ],
                },
                "entity": {
                    "name": "Product",
                    "fields": [
                        {"name": "id", "type": "string"},
                        {"name": "name", "type": "string"},
                        {"name": "price", "type": "number"},
                        {"name": "createdAt", "type": "string"},
                    ],
                },
                "routes": ["/products", "/products/new", "/products/[id]"],
                "server_client_rules": {
                    "prefer_server_components": "boolean",
                    "use_client_only_for_forms": "boolean",
                },
                "acceptance": ["pnpm build"],
            },
        },
    })
    out = dmr_chat_json(
        model=FOREMAN_MODEL, system=system, user=user,
        max_tokens=1200, temperature=0.4,
        label="policy",
    )
    contract = out.get("architecture_contract", {})
    acceptance = contract.get("acceptance", []) if isinstance(contract, dict) else []
    return {"architecture_contract": contract, "acceptance": acceptance}


_REQUIRED_FILES = {"package.json", "tsconfig.json"}
_REQUIRED_PATTERNS = {
    "next.config": lambda p: p.startswith("next.config.") and not p.endswith(".ts"),
    "layout": lambda p: "layout" in p and p.endswith((".tsx", ".jsx")),
    "page": lambda p: p.endswith(("page.tsx", "page.jsx")),
}

MAX_MANIFEST_ATTEMPTS = 2


def _validate_manifest(manifest: list[dict]) -> list[str]:
    """Check manifest for required files. Returns list of issues."""
    paths = {e.get("path", "") for e in manifest}
    issues = []

    for required in _REQUIRED_FILES:
        if required not in paths:
            issues.append(f"missing {required}")

    for name, check in _REQUIRED_PATTERNS.items():
        if not any(check(p) for p in paths):
            issues.append(f"missing {name} file")

    # Check for the bad next.config.ts
    if any(p == "next.config.ts" for p in paths):
        issues.append("next.config.ts present — must be .mjs or .js (Next.js 14)")

    return issues


def _build_manifest_prompt(state: FactoryState, issues: list[str] | None = None):
    """Build manifest system/user prompts, optionally including prior issues."""
    system = (
        "You are a senior software architect planning a Next.js App Router project. "
        "Return STRICT JSON only. "
        'Schema: {"files": [{"path": "relative/path", "description": "what this file does", '
        '"imports_from": ["other/project/paths"], '
        '"category": "config|lib|style|component|page|api"}]} '
        "List EVERY file needed for pnpm install and pnpm build to pass. "
        "imports_from lists other files IN THIS PROJECT that this file will import. "
        "IMPORTANT: Use next.config.mjs (NOT next.config.ts)."
    )
    constraints = [
        "Include all config files: package.json, tsconfig.json, next.config.mjs.",
        "Include all source files: pages, lib, API routes, styles.",
        "Use Tailwind CSS — do NOT plan separate UI primitive components (Button, Input, etc.).",
        "Only plan a shared component file if it contains real logic reused across 3+ pages.",
        "Keep the file count minimal. An experienced developer targets ~10-15 files for a CRUD app.",
        "imports_from should only reference paths within this project.",
        "Do NOT include file contents — only paths, descriptions, and metadata.",
    ]
    if issues:
        constraints.append(
            f"PREVIOUS ATTEMPT HAD ISSUES — fix these: {'; '.join(issues)}"
        )
    user = json.dumps({
        "app_spec": _require_spec(state),
        "architecture_contract": state.get("architecture_contract", {}),
        "constraints": constraints,
    })
    return system, user


def manifest_node(state: FactoryState) -> dict:
    """Generate and validate a file manifest. Retries once on validation failure."""
    log_step("Generate file manifest")

    issues: list[str] = []
    for attempt in range(1, MAX_MANIFEST_ATTEMPTS + 1):
        system, user = _build_manifest_prompt(
            state, issues=issues if attempt > 1 else None,
        )
        out = dmr_chat_json(
            model=CODER_MODEL, system=system, user=user,
            max_tokens=2000, temperature=0.3,
            label=f"manifest-{attempt}",
        )
        manifest = out.get("files", [])
        if not isinstance(manifest, list) or not manifest:
            raise RuntimeError(
                f"Manifest step returned no files. Keys: {list(out.keys())}"
            )

        log_detail(f"Manifest attempt {attempt}: {len(manifest)} files planned")
        for entry in manifest:
            log_detail(f"  {entry.get('category', '?'):10s}  {entry.get('path', '?')}")

        issues = _validate_manifest(manifest)
        if not issues:
            log_detail("Manifest validation passed")
            return {"manifest": manifest}

        log_detail(f"Manifest validation failed: {'; '.join(issues)}")
        if attempt < MAX_MANIFEST_ATTEMPTS:
            log_detail("Retrying manifest generation...")

    # Last attempt still had issues — proceed with warning, build-fix loop is the fallback
    log_detail(
        f"WARNING: manifest has unresolved issues after {MAX_MANIFEST_ATTEMPTS} attempts: "
        f"{'; '.join(issues)}. Proceeding anyway — build-fix loop will handle gaps."
    )
    return {"manifest": manifest}


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def generate_node(state: FactoryState) -> dict:
    """Generate the complete project in a single model call."""
    gen_attempt = state.get("generate_attempt", 0) + 1
    log_step(f"Generate project — monolithic (attempt {gen_attempt})")

    system = (
        "You are a meticulous senior engineer generating a complete runnable project.\n"
        "Output each file using this EXACT fence format (no JSON, no markdown):\n\n"
        "===FILE: relative/path.ext===\n"
        "file contents here\n"
        "===END FILE===\n\n"
        "Do not omit required config files; ensure pnpm build will succeed.\n"
        "Obey the architecture_contract strictly.\n"
        "If ui_strategy.mode is no_ui_imports, do NOT import @/components/ui/* "
        "and use plain HTML + minimal CSS (or Tailwind only if included).\n"
        "If ui_strategy.mode is local_primitives, create the required primitive "
        "files and import them.\n"
        "Do not depend on shadcn CLI."
    )

    user = json.dumps({
        "app_spec": _require_spec(state),
        "architecture_contract": state.get("architecture_contract", {}),
        "constraints": [
            "Project must be Next.js App Router + TypeScript.",
            "Provide all files required for pnpm install and pnpm build.",
            "Keep it minimal but complete.",
            "CRUD entity: Products (id, name, price, createdAt).",
            "Pages: /products (list), /products/new (create), /products/[id] (detail).",
            "In-memory store is fine; prefer server components + route handlers or server actions.",
            "Avoid external DB for MVP.",
        ],
        "output_requirements": [
            "Output ONLY ===FILE: path=== ... ===END FILE=== blocks.",
            "Every file must contain full, complete code (no placeholders).",
            "Do not reference files that you did not include.",
            "Honor ui_strategy mode in architecture_contract.",
        ],
        "attempt": gen_attempt,
    })

    start = time.monotonic()
    raw = dmr_chat_raw(
        model=CODER_MODEL, system=system, user=user,
        max_tokens=6000, temperature=0.2,
        label="generate-all",
    )
    elapsed = time.monotonic() - start
    log_detail(f"Model response received in {elapsed:.1f}s")

    files = parse_fenced_files(raw)
    if not files:
        # Dump first 500 chars for debugging
        log_detail(f"No fenced files found. Response start: {raw[:500]!r}")
        raise RuntimeError("Generate step returned no files in fence format")

    log_detail(f"Generated {len(files)} files")
    return {"files": files, "generate_attempt": gen_attempt, "fix_attempt": 0}


# ---------------------------------------------------------------------------
# Write / Install / Build / Fix
# ---------------------------------------------------------------------------

_PRESERVE_DIRS = {"node_modules", ".next", ".pnpm-store"}


def write_node(state: FactoryState) -> dict:
    """Write generated files to disk, removing stale files from prior runs."""
    log_step("Write files to disk")
    project_dir = state.get("project_dir") or OUTPUT_DIR
    os.makedirs(project_dir, exist_ok=True)

    files = state.get("files", {})
    if not files:
        raise RuntimeError("No files to write.")

    # Normalise expected paths
    expected = {p.lstrip("/").strip() for p in files}

    # Remove files on disk that are no longer in the file map
    for dirpath, dirnames, filenames in os.walk(project_dir, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in _PRESERVE_DIRS]
        for fname in filenames:
            abs_path = os.path.join(dirpath, fname)
            rel = os.path.relpath(abs_path, project_dir)
            if rel not in expected and fname != "pnpm-lock.yaml":
                os.remove(abs_path)

    for rel_path, content in files.items():
        rel_path = rel_path.lstrip("/").strip()
        if not rel_path:
            continue
        abs_path = os.path.join(project_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

    package_json = files.get("package.json")
    package_json_hash = _hash_text(package_json) if isinstance(package_json, str) else ""
    return {"project_dir": project_dir, "package_json_hash": package_json_hash}


def install_node(state: FactoryState) -> dict:
    """Install dependencies, skipping if package.json hasn't changed."""
    log_step("Install dependencies")
    project_dir = _require_project_dir(state)
    pkg_hash = state.get("package_json_hash")
    last_hash = state.get("last_installed_package_json_hash")

    node_modules = os.path.join(project_dir, "node_modules")
    if os.path.isdir(node_modules) and pkg_hash and pkg_hash == last_hash:
        log_detail("deps already installed, skipping")
        return {}

    if shutil.which("pnpm") is None:
        raise RuntimeError("pnpm not found in PATH")

    p = _run_cmd(["pnpm", "install"], cwd=project_dir)
    if p.returncode != 0:
        return {"last_build_ok": False, "last_build_log": f"pnpm install failed:\n{p.stdout}"}
    return {"last_installed_package_json_hash": pkg_hash or last_hash}


def build_node(state: FactoryState) -> dict:
    """Run pnpm build."""
    log_step("Run pnpm build")
    project_dir = _require_project_dir(state)
    build_attempt = state.get("build_attempt", 0) + 1

    if shutil.which("pnpm") is None:
        raise RuntimeError("pnpm not found in PATH")

    p = _run_cmd(["pnpm", "build"], cwd=project_dir)
    ok = p.returncode == 0
    if not ok:
        _log_build_failure(p.stdout or "", build_attempt, state.get("last_patched_files", []))
    return {"last_build_ok": ok, "last_build_log": p.stdout, "build_attempt": build_attempt}


def fix_node(state: FactoryState) -> dict:
    """Ask the coder model to fix build errors with minimal patches."""
    fix_attempt = state.get("fix_attempt", 0) + 1
    log_step(f"Apply fix (attempt {fix_attempt})")

    system = (
        "You are a senior engineer fixing a broken Next.js project.\n"
        "Do NOT refactor the project structure.\n"
        "Do NOT rewrite the entire project.\n"
        "Do NOT introduce new dependencies unless required by the build error.\n"
        "Prefer minimal edits to the smallest number of files.\n\n"
        "Output format:\n"
        "1. For each file to patch, output its FULL new content using fence format:\n"
        "===FILE: relative/path.ext===\n"
        "full new file contents\n"
        "===END FILE===\n\n"
        "2. To delete a file, use:\n"
        "===DELETE: relative/path.ext===\n\n"
        "3. End with a brief explanation on its own line starting with EXPLANATION:\n\n"
        "IMPORTANT: Use next.config.mjs (NOT next.config.ts) — Next.js 14 does not support TypeScript config files."
    )

    files = state.get("files", {})
    project_dir = state.get("project_dir")
    build_log = state.get("last_build_log", "")

    # Include relevant file snapshots for context
    snapshots: dict[str, str] = {}
    for key in ("package.json", "tsconfig.json"):
        if key in files:
            snapshots[key] = files[key]
    for path in files:
        if path.startswith("next.config."):
            snapshots[path] = files[path]

    referenced = extract_referenced_paths(build_log, project_dir)
    added = 0
    for path in referenced:
        if path in files and path not in snapshots:
            snapshots[path] = files[path]
            added += 1
            if added >= 5:
                break

    # Format snapshots as fenced content for the model
    snapshot_text = ""
    for path, content in snapshots.items():
        snapshot_text += f"===CURRENT FILE: {path}===\n{content}\n===END CURRENT FILE===\n\n"

    user = (
        f"Build log (last 12000 chars):\n{build_log[-12000:]}\n\n"
        f"Architecture contract:\n{json.dumps(state.get('architecture_contract', {}))}\n\n"
        f"All project files: {', '.join(sorted(files.keys()))}\n\n"
        f"Current file contents:\n{snapshot_text}\n"
        f"Fix attempt: {fix_attempt}\n\n"
        "Instructions:\n"
        "- Output patched files using ===FILE: path=== ... ===END FILE=== format.\n"
        "- Only include files that need changes.\n"
        "- Each file must contain the FULL new content.\n"
        "- Keep changes minimal.\n"
        "- Use next.config.mjs (NOT next.config.ts).\n"
        "- To delete a file, use ===DELETE: path===\n"
    )

    raw = dmr_chat_raw(
        model=CODER_MODEL, system=system, user=user,
        max_tokens=8000, temperature=0.2,
        label="fix",
    )

    patches = parse_fenced_files(raw)

    # Parse deletions
    deletion_pattern = re.compile(r"===DELETE:\s*(.+?)\s*===")
    deletions = deletion_pattern.findall(raw)

    if not patches and not deletions:
        log_detail(f"Fix produced no patches or deletions. Response start: {raw[:500]!r}")
        raise RuntimeError("Fix step produced no patches")

    patched_files = sorted(patches.keys())
    log_detail(f"Patched files: {', '.join(patched_files)}")
    if deletions:
        log_detail(f"Deleted files: {', '.join(deletions)}")

    merged = dict(files)
    for path in deletions:
        merged.pop(path, None)
    merged.update(patches)
    return {"files": merged, "fix_attempt": fix_attempt, "last_patched_files": patched_files}


def done_node(state: FactoryState) -> dict:
    return {}


def fail_node(state: FactoryState) -> dict:
    raise RuntimeError(
        "Failed to reach a successful pnpm build within retry limits.\n\n"
        f"Project dir: {state.get('project_dir')}\n\n"
        f"Last build log:\n{state.get('last_build_log', '')}"
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def decide_next(state: FactoryState) -> str:
    if state.get("last_build_ok"):
        return "done"

    log = state.get("last_build_log", "")
    if "pnpm install failed" in log:
        if state.get("generate_attempt", 0) < MAX_GENERATE_ATTEMPTS:
            return "regenerate"
        return "fail"

    if state.get("fix_attempt", 0) < MAX_FIX_ATTEMPTS:
        return "fix"

    if state.get("generate_attempt", 0) < MAX_GENERATE_ATTEMPTS:
        return "regenerate"

    return "fail"


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------


def build_factory_graph():
    """Build the full factory pipeline graph.

    Pipeline: warmup → policy → generate → write → install → build
    with fix loop and regeneration fallback on build failure.
    """
    g = StateGraph(FactoryState)

    g.add_node("warmup", warmup_node)
    g.add_node("policy", policy_node)
    g.add_node("generate", generate_node)
    g.add_node("write", write_node)
    g.add_node("install", install_node)
    g.add_node("build", build_node)
    g.add_node("fix", fix_node)
    g.add_node("done", done_node)
    g.add_node("fail", fail_node)

    g.add_edge(START, "warmup")
    g.add_edge("warmup", "policy")
    g.add_edge("policy", "generate")

    g.add_edge("generate", "write")
    g.add_edge("write", "install")
    g.add_edge("install", "build")

    g.add_conditional_edges(
        "build",
        decide_next,
        {
            "fix": "fix",
            "regenerate": "generate",
            "done": "done",
            "fail": "fail",
        },
    )

    g.add_edge("fix", "write")
    g.add_edge("done", END)

    return g.compile()
