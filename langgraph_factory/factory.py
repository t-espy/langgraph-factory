"""Full factory pipeline: policy -> manifest -> generate -> build -> fix loop.

Evolution of the MVP pipeline. Adds:
- Architecture policy step (foreman produces a contract before generation)
- Manifest step (plan file list with validation/retry before generating)
- Build-fix loop with mechanical error classification and LLM fallback
- Package.json hash tracking (skip reinstall when deps unchanged)
- Full regeneration fallback when fixes aren't enough
- Per-step timing and end-of-run summary
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
from langgraph_factory.llm import LLMStats, dmr_chat_json, dmr_chat_raw
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
    fix_history: list[dict]
    review_verdict: dict  # from gpt-oss reviewer: {action, guidance, reasoning}
    step_timings: list[dict]
    pipeline_start: float


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


def _record_step(state: FactoryState, node: str, elapsed_s: float,
                  llm_stats: LLMStats | None = None, **extra) -> list[dict]:
    """Append a step timing record and return the updated list."""
    timings = list(state.get("step_timings", []))
    entry: dict = {"node": node, "elapsed_s": round(elapsed_s, 1)}
    if llm_stats:
        entry["model"] = llm_stats.model
        entry["tokens"] = llm_stats.tokens
        entry["tok_s"] = round(llm_stats.tok_s, 1)
        entry["prompt_chars"] = llm_stats.prompt_chars
        entry["finish_reason"] = llm_stats.finish_reason
    entry.update(extra)
    timings.append(entry)
    return timings


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


def policy_node(state: FactoryState) -> dict:
    """Generate an architecture contract from the app spec."""
    log_step("Generate architecture policy")
    t0 = time.monotonic()

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
    out, stats = dmr_chat_json(
        model=FOREMAN_MODEL, system=system, user=user,
        max_tokens=3000, temperature=0.4,
        label="policy",
    )
    contract = out.get("architecture_contract", {})
    acceptance = contract.get("acceptance", []) if isinstance(contract, dict) else []
    elapsed = time.monotonic() - t0
    timings = _record_step(state, "policy", elapsed, stats)
    return {
        "architecture_contract": contract, "acceptance": acceptance,
        "step_timings": timings, "pipeline_start": state.get("pipeline_start", t0),
    }


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
        "Include a root layout file (e.g. src/app/layout.tsx or app/layout.tsx) — Next.js requires this.",
        "Include all source files: pages, lib, API routes, styles.",
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
    t0 = time.monotonic()
    last_stats: LLMStats | None = None

    issues: list[str] = []
    for attempt in range(1, MAX_MANIFEST_ATTEMPTS + 1):
        system, user = _build_manifest_prompt(
            state, issues=issues if attempt > 1 else None,
        )
        out, last_stats = dmr_chat_json(
            model=CODER_MODEL, system=system, user=user,
            max_tokens=4000, temperature=0.3,
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
            elapsed = time.monotonic() - t0
            timings = _record_step(state, "manifest", elapsed, last_stats,
                                   files_planned=len(manifest), attempts=attempt)
            return {"manifest": manifest, "step_timings": timings}

        log_detail(f"Manifest validation failed: {'; '.join(issues)}")
        if attempt < MAX_MANIFEST_ATTEMPTS:
            log_detail("Retrying manifest generation...")

    # Last attempt still had issues — proceed with warning
    log_detail(
        f"WARNING: manifest has unresolved issues after {MAX_MANIFEST_ATTEMPTS} attempts: "
        f"{'; '.join(issues)}. Proceeding anyway — build-fix loop will handle gaps."
    )
    elapsed = time.monotonic() - t0
    timings = _record_step(state, "manifest", elapsed, last_stats,
                           files_planned=len(manifest), attempts=MAX_MANIFEST_ATTEMPTS,
                           issues=issues)
    return {"manifest": manifest, "step_timings": timings}


# ---------------------------------------------------------------------------
# Import reconciliation
# ---------------------------------------------------------------------------

# Packages that are built into Node.js or Next.js — never add to package.json
_BUILTIN_MODULES = frozenset({
    "react", "react-dom", "next", "fs", "path", "os", "url", "util",
    "stream", "crypto", "http", "https", "events", "buffer", "querystring",
    "child_process", "cluster", "dgram", "dns", "net", "readline", "tls",
    "zlib", "assert", "constants", "module", "process", "timers", "tty",
    "v8", "vm", "worker_threads", "perf_hooks",
})

# Import patterns for JS/TS
_IMPORT_PATTERN = re.compile(
    r"""(?:import\s+.*?\s+from\s+|import\s+|require\s*\(\s*)['"]([^'"]+)['"]""",
)


def _extract_npm_packages(files: dict[str, str]) -> set[str]:
    """Scan source files for bare npm package imports (not relative or alias)."""
    packages: set[str] = set()
    for path, content in files.items():
        if not path.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")):
            continue
        for match in _IMPORT_PATTERN.finditer(content):
            specifier = match.group(1)
            # Skip relative imports and alias imports (@/ is project alias)
            if specifier.startswith((".")) or specifier.startswith("@/"):
                continue
            # Extract package name: 'lodash/merge' -> 'lodash', '@radix-ui/react-slot' -> '@radix-ui/react-slot'
            if specifier.startswith("@"):
                parts = specifier.split("/")
                pkg_name = "/".join(parts[:2]) if len(parts) >= 2 else specifier
            else:
                pkg_name = specifier.split("/")[0]
            if pkg_name and pkg_name not in _BUILTIN_MODULES:
                packages.add(pkg_name)
    return packages


def _reconcile_imports(files: dict[str, str]) -> dict[str, str]:
    """Ensure every npm package imported in source files is in package.json."""
    pkg_json_str = files.get("package.json")
    if not pkg_json_str:
        return files

    try:
        pkg = json.loads(pkg_json_str)
    except json.JSONDecodeError:
        return files

    deps = pkg.get("dependencies", {})
    dev_deps = pkg.get("devDependencies", {})
    all_declared = set(deps) | set(dev_deps)

    imported = _extract_npm_packages(files)
    missing = imported - all_declared

    if not missing:
        return files

    # Add missing packages to dependencies
    for pkg_name in sorted(missing):
        deps[pkg_name] = "latest"
    pkg["dependencies"] = deps

    log_detail(f"Import reconciliation: added missing packages to package.json: {', '.join(sorted(missing))}")
    files = dict(files)
    files["package.json"] = json.dumps(pkg, indent=2)
    return files


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def generate_node(state: FactoryState) -> dict:
    """Generate the complete project in a single model call."""
    gen_attempt = state.get("generate_attempt", 0) + 1
    log_step(f"Generate project — monolithic (attempt {gen_attempt})")
    t0 = time.monotonic()

    system = (
        "You are a meticulous senior engineer generating a complete runnable project.\n"
        "Output each file using this EXACT fence format (no JSON, no markdown):\n\n"
        "===FILE: relative/path.ext===\n"
        "file contents here\n"
        "===END FILE===\n\n"
        "Do not omit required config files; ensure pnpm build will succeed.\n"
        "Obey the architecture_contract strictly.\n"
        "Every npm package you import MUST appear in package.json dependencies.\n"
        "Every file you import from within the project MUST be included in your output.\n"
        "If you create component files, they must export everything that other files import from them.\n"
        "Put shared TypeScript types and in-memory stores in a dedicated file (e.g. lib/types.ts, lib/store.ts) "
        "and import from there — NEVER duplicate type definitions or stores across files."
    )

    user = json.dumps({
        "app_spec": _require_spec(state),
        "architecture_contract": state.get("architecture_contract", {}),
        "constraints": [
            "Project must be Next.js App Router + TypeScript.",
            "Provide all files required for pnpm install and pnpm build.",
            "Keep it minimal but complete.",
            "Derive ALL entity fields, pages, and routes from the app_spec and architecture_contract — do not hardcode or omit fields.",
            "Sample/seed data must include every field defined in the entity type.",
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

    raw, stats = dmr_chat_raw(
        model=CODER_MODEL, system=system, user=user,
        max_tokens=32000, temperature=0.2,
        label="generate-all",
    )
    elapsed = time.monotonic() - t0
    log_detail(f"Model response received in {elapsed:.1f}s")

    files = parse_fenced_files(raw)
    if not files:
        log_detail(f"No fenced files found. Response start: {raw[:500]!r}")
        raise RuntimeError("Generate step returned no files in fence format")

    log_detail(f"Generated {len(files)} files")

    # Reconcile imports: ensure every npm package used in source files
    # is listed in package.json dependencies
    files = _reconcile_imports(files)

    total_chars = sum(len(c) for c in files.values())
    timings = _record_step(state, "generate", elapsed, stats,
                           files=len(files), total_chars=total_chars,
                           attempt=gen_attempt)
    return {
        "files": files, "generate_attempt": gen_attempt, "fix_attempt": 0,
        "step_timings": timings,
    }


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
    t0 = time.monotonic()
    project_dir = _require_project_dir(state)
    pkg_hash = state.get("package_json_hash")
    last_hash = state.get("last_installed_package_json_hash")

    node_modules = os.path.join(project_dir, "node_modules")
    if os.path.isdir(node_modules) and pkg_hash and pkg_hash == last_hash:
        log_detail("deps already installed, skipping")
        elapsed = time.monotonic() - t0
        timings = _record_step(state, "install", elapsed, skipped=True)
        return {"step_timings": timings}

    if shutil.which("pnpm") is None:
        raise RuntimeError("pnpm not found in PATH")

    p = _run_cmd(["pnpm", "install"], cwd=project_dir)
    elapsed = time.monotonic() - t0
    if p.returncode != 0:
        timings = _record_step(state, "install", elapsed, ok=False)
        return {
            "last_build_ok": False,
            "last_build_log": f"pnpm install failed:\n{p.stdout}",
            "step_timings": timings,
        }
    log_detail(f"Install completed in {elapsed:.1f}s")
    timings = _record_step(state, "install", elapsed, ok=True)
    return {
        "last_installed_package_json_hash": pkg_hash or last_hash,
        "step_timings": timings,
    }


def build_node(state: FactoryState) -> dict:
    """Run pnpm build."""
    log_step("Run pnpm build")
    t0 = time.monotonic()
    project_dir = _require_project_dir(state)
    build_attempt = state.get("build_attempt", 0) + 1

    if shutil.which("pnpm") is None:
        raise RuntimeError("pnpm not found in PATH")

    p = _run_cmd(["pnpm", "build"], cwd=project_dir)
    elapsed = time.monotonic() - t0
    ok = p.returncode == 0
    if ok:
        log_detail(f"Build succeeded in {elapsed:.1f}s")
    else:
        _log_build_failure(p.stdout or "", build_attempt, state.get("last_patched_files", []))
        log_detail(f"Build failed in {elapsed:.1f}s")
    timings = _record_step(state, "build", elapsed, ok=ok, attempt=build_attempt)
    return {
        "last_build_ok": ok, "last_build_log": p.stdout,
        "build_attempt": build_attempt, "step_timings": timings,
    }


def _try_mechanical_fix(build_log: str, files: dict[str, str]) -> dict[str, str] | None:
    """Try to fix build errors mechanically without calling the LLM.

    Returns a dict of {path: new_content} for patched files, or None if
    no mechanical fix applies.
    """
    patches: dict[str, str] = {}

    # --- Fix 1: missing npm packages ---
    missing_modules = re.findall(
        r"Module not found: Can't resolve '([^./][^']*)'", build_log,
    )
    if missing_modules:
        pkg_json_str = files.get("package.json")
        if pkg_json_str:
            try:
                pkg = json.loads(pkg_json_str)
                deps = pkg.get("dependencies", {})
                added = []
                for mod in missing_modules:
                    pkg_name = mod if mod.startswith("@") else mod.split("/")[0]
                    if pkg_name not in deps:
                        deps[pkg_name] = "latest"
                        added.append(pkg_name)
                if added:
                    pkg["dependencies"] = deps
                    log_detail(f"Mechanical fix: adding missing packages: {', '.join(added)}")
                    patches["package.json"] = json.dumps(pkg, indent=2)
            except json.JSONDecodeError:
                pass

    # --- Fix 2: "Cannot find name 'X'" — find the type/interface in another
    #     file and add an import statement to the broken file ---
    missing_names = re.findall(
        r"^(\./[^:]+):.*Cannot find name '(\w+)'", build_log, re.MULTILINE,
    )
    if missing_names:
        # Build index: which files export/define which names
        _TYPE_DEF_RE = re.compile(
            r"(?:export\s+)?(?:type|interface|enum|class)\s+(\w+)"
        )
        name_to_file: dict[str, str] = {}
        for fpath, content in files.items():
            if not fpath.endswith((".ts", ".tsx")):
                continue
            for m in _TYPE_DEF_RE.finditer(content):
                name_to_file[m.group(1)] = fpath

        for error_path, missing_name in missing_names:
            # Normalize the error path (strip leading ./)
            norm_path = error_path.lstrip("./")
            if norm_path not in files:
                continue
            if missing_name not in name_to_file:
                continue
            source_file = name_to_file[missing_name]
            if source_file == norm_path:
                continue  # defined in same file, different issue

            # Compute relative import path
            from_dir = os.path.dirname(norm_path)
            rel = os.path.relpath(source_file, from_dir)
            # Remove extension for TS imports
            rel = re.sub(r"\.(tsx?|jsx?)$", "", rel)
            if not rel.startswith("."):
                rel = "./" + rel

            content = files[norm_path]
            # Check if already imported from that path
            if rel in content:
                continue

            # Also check @/ alias path
            alias_path = "@/" + source_file
            alias_path = re.sub(r"\.(tsx?|jsx?)$", "", alias_path)
            if alias_path in content:
                continue

            # Ensure the source file actually exports the name
            source_content = files[source_file]
            if f"export " not in source_content:
                # Add export to the type definition in the source file
                source_content = re.sub(
                    rf"^(type|interface|enum|class)\s+{re.escape(missing_name)}\b",
                    rf"export \1 {missing_name}",
                    source_content,
                    count=1,
                    flags=re.MULTILINE,
                )
                patches[source_file] = source_content

            import_line = f'import {{ {missing_name} }} from "{rel}";\n'
            content = import_line + content
            patches[norm_path] = content
            log_detail(f"Mechanical fix: added import of {missing_name} from {rel} in {norm_path}")

    return patches if patches else None


def _build_fix_history_text(fix_history: list[dict]) -> str:
    """Format prior fix attempts for inclusion in the LLM prompt."""
    if not fix_history:
        return ""
    parts = []
    for entry in fix_history:
        kind = "mechanical" if entry.get("mechanical") else "LLM"
        parts.append(
            f"- Attempt {entry['attempt']} ({kind}): "
            f"patched {', '.join(entry['patches'])}. "
            f"Error: {entry['error_summary']}"
        )
    return "Prior fix attempts (DO NOT repeat these):\n" + "\n".join(parts) + "\n\n"


def fix_node(state: FactoryState) -> dict:
    """Fix build errors: apply mechanical or LLM fixes based on reviewer verdict."""
    fix_attempt = state.get("fix_attempt", 0) + 1
    log_step(f"Apply fix (attempt {fix_attempt})")
    t0 = time.monotonic()

    files = state.get("files", {})
    build_log = state.get("last_build_log", "")
    fix_history = list(state.get("fix_history", []))
    verdict = state.get("review_verdict", {})

    # --- Mechanical fix (reviewer already detected it) ---
    if verdict.get("mechanical"):
        mechanical_patches = _try_mechanical_fix(build_log, files)
        if mechanical_patches:
            patched_files = sorted(mechanical_patches.keys())
            log_detail(f"Mechanical fix patched: {', '.join(patched_files)}")
            merged = dict(files)
            merged.update(mechanical_patches)
            fix_history.append({
                "attempt": fix_attempt,
                "error_summary": build_log.strip().splitlines()[-1][:200] if build_log.strip() else "",
                "patches": patched_files,
                "mechanical": True,
            })
            elapsed = time.monotonic() - t0
            timings = _record_step(state, "fix", elapsed, mechanical=True,
                                   patches=patched_files)
            return {
                "files": merged, "fix_attempt": fix_attempt,
                "last_patched_files": patched_files, "fix_history": fix_history,
                "step_timings": timings,
            }

    # --- LLM fix ---
    reviewer_guidance = verdict.get("guidance", "")

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

    # Send all project files as context — the model has 128k context, use it
    snapshot_text = ""
    for path in sorted(files.keys()):
        snapshot_text += f"===CURRENT FILE: {path}===\n{files[path]}\n===END CURRENT FILE===\n\n"

    history_text = _build_fix_history_text(fix_history)

    # Include reviewer guidance when available
    guidance_text = ""
    if reviewer_guidance:
        guidance_text = (
            f"REVIEWER GUIDANCE (from senior engineering lead):\n"
            f"{reviewer_guidance}\n\n"
        )

    user = (
        f"Build log (last 12000 chars):\n{build_log[-12000:]}\n\n"
        f"{guidance_text}"
        f"Architecture contract:\n{json.dumps(state.get('architecture_contract', {}))}\n\n"
        f"All project files: {', '.join(sorted(files.keys()))}\n\n"
        f"{history_text}"
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

    raw, stats = dmr_chat_raw(
        model=CODER_MODEL, system=system, user=user,
        max_tokens=16000, temperature=0.2,
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

    # Build a concise error summary from the build log
    error_lines = [l for l in build_log.splitlines() if "error" in l.lower() or "Error" in l]
    error_summary = "; ".join(error_lines[:3])[:300] if error_lines else build_log.strip().splitlines()[-1][:200]

    fix_history.append({
        "attempt": fix_attempt,
        "error_summary": error_summary,
        "patches": patched_files + [f"DELETE:{d}" for d in deletions],
        "mechanical": False,
    })
    elapsed = time.monotonic() - t0
    timings = _record_step(state, "fix", elapsed, stats, mechanical=False,
                           patches=patched_files)
    return {
        "files": merged, "fix_attempt": fix_attempt,
        "last_patched_files": patched_files, "fix_history": fix_history,
        "step_timings": timings,
    }


def _build_summary(state: FactoryState) -> str:
    """Build a run summary string with per-step timings and model stats."""
    timings = state.get("step_timings", [])
    pipeline_start = state.get("pipeline_start", 0)
    total_elapsed = time.monotonic() - pipeline_start if pipeline_start else 0

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("RUN SUMMARY")
    lines.append("=" * 70)

    # Per-step table
    lines.append(f"\n{'Step':<20} {'Time':>8} {'Tokens':>8} {'tok/s':>8} {'Model':<25} {'Notes'}")
    lines.append("-" * 90)
    for t in timings:
        node = t["node"]
        elapsed = f"{t['elapsed_s']:.1f}s"
        tokens = str(t.get("tokens", "")) if t.get("tokens") else ""
        tok_s = f"{t['tok_s']:.1f}" if t.get("tok_s") else ""
        model = t.get("model", "")
        notes_parts = []
        if t.get("files"):
            notes_parts.append(f"{t['files']} files")
        if t.get("total_chars"):
            notes_parts.append(f"{t['total_chars']:,} chars")
        if t.get("attempt") and t["attempt"] > 1:
            notes_parts.append(f"attempt {t['attempt']}")
        if t.get("attempts") and t["attempts"] > 1:
            notes_parts.append(f"{t['attempts']} attempts")
        if t.get("files_planned"):
            notes_parts.append(f"{t['files_planned']} planned")
        if t.get("mechanical"):
            notes_parts.append("mechanical")
        if t.get("patches"):
            notes_parts.append(f"patched: {', '.join(t['patches'])}")
        if t.get("skipped"):
            notes_parts.append("skipped")
        if "ok" in t:
            notes_parts.append("OK" if t["ok"] else "FAILED")
        if t.get("finish_reason") and t["finish_reason"] != "stop":
            notes_parts.append(f"finish={t['finish_reason']}")
        notes = ", ".join(notes_parts)
        lines.append(f"{node:<20} {elapsed:>8} {tokens:>8} {tok_s:>8} {model:<25} {notes}")

    # Aggregate model stats
    model_stats: dict[str, dict] = {}
    for t in timings:
        model = t.get("model", "")
        if not model or not t.get("tokens"):
            continue
        if model not in model_stats:
            model_stats[model] = {"total_tokens": 0, "total_time": 0.0, "calls": 0}
        model_stats[model]["total_tokens"] += t["tokens"]
        model_stats[model]["total_time"] += t["elapsed_s"]
        model_stats[model]["calls"] += 1

    if model_stats:
        lines.append(f"\n{'Model':<30} {'Calls':>6} {'Tokens':>10} {'Total Time':>12} {'Avg tok/s':>10}")
        lines.append("-" * 70)
        for model, ms in model_stats.items():
            avg_tok_s = ms["total_tokens"] / ms["total_time"] if ms["total_time"] > 0 else 0
            lines.append(f"{model:<30} {ms['calls']:>6} {ms['total_tokens']:>10,} {ms['total_time']:>11.1f}s {avg_tok_s:>9.1f}")

    # Overall
    build_attempts = sum(1 for t in timings if t["node"] == "build")
    fix_attempts = sum(1 for t in timings if t["node"] == "fix")
    gen_attempts = sum(1 for t in timings if t["node"] == "generate")
    lines.append(f"\nTotal pipeline time: {total_elapsed:.1f}s")
    lines.append(f"Generate attempts: {gen_attempts}  |  Fix attempts: {fix_attempts}  |  Build attempts: {build_attempts}")
    lines.append(f"Result: {'BUILD OK' if state.get('last_build_ok') else 'FAILED'}")
    lines.append(f"Project: {state.get('project_dir', 'N/A')}")
    lines.append("=" * 70)

    return "\n".join(lines)


def _emit_summary(state: FactoryState) -> None:
    """Print run summary to stdout and write to summary.txt in the project dir."""
    summary = _build_summary(state)
    print("\n" + summary)

    project_dir = state.get("project_dir")
    if project_dir:
        summary_path = os.path.join(project_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        log_detail(f"Summary written to {summary_path}")


def done_node(state: FactoryState) -> dict:
    _emit_summary(state)
    return {}


def fail_node(state: FactoryState) -> dict:
    _emit_summary(state)
    raise RuntimeError(
        "Failed to reach a successful pnpm build within retry limits.\n\n"
        f"Project dir: {state.get('project_dir')}\n\n"
        f"Last build log:\n{state.get('last_build_log', '')}"
    )


# ---------------------------------------------------------------------------
# Review (gpt-oss as build-failure reviewer)
# ---------------------------------------------------------------------------


def review_node(state: FactoryState) -> dict:
    """Have the foreman model review build failures and decide the strategy.

    Returns a verdict: fix (with guidance), regenerate, or fail.
    Mechanical fixes bypass the reviewer — they go straight to fix_node.
    """
    build_log = state.get("last_build_log", "")
    fix_attempt = state.get("fix_attempt", 0)
    gen_attempt = state.get("generate_attempt", 0)
    fix_history = state.get("fix_history", [])

    log_step(f"Review build failure (fix={fix_attempt}, gen={gen_attempt})")
    t0 = time.monotonic()

    # Check for mechanical fix first — skip the LLM reviewer entirely
    files = state.get("files", {})
    mechanical_patches = _try_mechanical_fix(build_log, files)
    if mechanical_patches:
        log_detail("Reviewer skipped — mechanical fix available")
        elapsed = time.monotonic() - t0
        timings = _record_step(state, "review", elapsed, skipped="mechanical")
        return {
            "review_verdict": {
                "action": "fix",
                "guidance": "Mechanical fix available — apply automatically.",
                "reasoning": "Deterministic pattern match.",
                "mechanical": True,
            },
            "step_timings": timings,
        }

    history_text = _build_fix_history_text(fix_history)

    system = (
        "You are a senior engineering lead reviewing a build failure.\n"
        "Your job is to decide the best recovery strategy — NOT to write code.\n\n"
        "Return STRICT JSON with these fields:\n"
        '  "action": one of "fix", "regenerate", or "fail"\n'
        '  "reasoning": 1-2 sentences on why you chose this action\n'
        '  "guidance": if action is "fix", specific guidance for the engineer '
        "on what to change and in which files (2-4 sentences max)\n\n"
        "Decision criteria:\n"
        '- "fix": the errors are localized (type mismatches, missing imports, '
        "small logic errors in 1-3 files). Most build failures should be fixable.\n"
        '- "regenerate": the errors are systemic (wrong project structure, '
        "fundamentally broken architecture, many files affected, or prior fix "
        "attempts have failed to make progress).\n"
        '- "fail": you believe the spec or architecture contract is contradictory '
        "or impossible to satisfy, or all retry budgets are exhausted.\n"
    )

    user = json.dumps({
        "build_log_tail": build_log[-6000:],
        "fix_attempt": fix_attempt,
        "max_fix_attempts": MAX_FIX_ATTEMPTS,
        "generate_attempt": gen_attempt,
        "max_generate_attempts": MAX_GENERATE_ATTEMPTS,
        "fix_history": fix_history[-5:],  # last 5 attempts
        "architecture_contract": state.get("architecture_contract", {}),
        "file_list": sorted(files.keys()),
    })

    verdict, stats = dmr_chat_json(
        model=FOREMAN_MODEL, system=system, user=user,
        max_tokens=2000, temperature=0.3,
        label="review",
    )

    action = verdict.get("action", "fix")
    # Enforce retry budget limits regardless of what the model says
    if action == "fix" and fix_attempt >= MAX_FIX_ATTEMPTS:
        if gen_attempt < MAX_GENERATE_ATTEMPTS:
            action = "regenerate"
            verdict["reasoning"] = (verdict.get("reasoning", "") +
                                    " (overridden: fix budget exhausted)")
        else:
            action = "fail"
            verdict["reasoning"] = (verdict.get("reasoning", "") +
                                    " (overridden: all budgets exhausted)")
    elif action == "regenerate" and gen_attempt >= MAX_GENERATE_ATTEMPTS:
        action = "fail"
        verdict["reasoning"] = (verdict.get("reasoning", "") +
                                " (overridden: regenerate budget exhausted)")

    verdict["action"] = action
    log_detail(f"Reviewer verdict: {action} — {verdict.get('reasoning', '')}")
    if action == "fix" and verdict.get("guidance"):
        log_detail(f"Reviewer guidance: {verdict['guidance']}")

    elapsed = time.monotonic() - t0
    timings = _record_step(state, "review", elapsed, stats)
    return {
        "review_verdict": verdict,
        "step_timings": timings,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def decide_after_build(state: FactoryState) -> str:
    """Route after build: success → done, failure → review."""
    if state.get("last_build_ok"):
        return "done"
    return "review"


def decide_after_review(state: FactoryState) -> str:
    """Route based on the reviewer's verdict."""
    verdict = state.get("review_verdict", {})
    action = verdict.get("action", "fix")
    if action == "regenerate":
        return "regenerate"
    if action == "fail":
        return "fail"
    return "fix"


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------


def build_factory_graph():
    """Build the full factory pipeline graph.

    Pipeline: policy -> manifest -> generate -> write -> install -> build
    with fix loop (mechanical + LLM) and regeneration fallback.
    """
    g = StateGraph(FactoryState)

    g.add_node("policy", policy_node)
    g.add_node("manifest", manifest_node)
    g.add_node("generate", generate_node)
    g.add_node("write", write_node)
    g.add_node("install", install_node)
    g.add_node("build", build_node)
    g.add_node("review", review_node)
    g.add_node("fix", fix_node)
    g.add_node("done", done_node)
    g.add_node("fail", fail_node)

    g.add_edge(START, "policy")
    g.add_edge("policy", "manifest")
    g.add_edge("manifest", "generate")

    g.add_edge("generate", "write")
    g.add_edge("write", "install")
    g.add_edge("install", "build")

    # Build outcome: success → done, failure → review
    g.add_conditional_edges(
        "build",
        decide_after_build,
        {
            "done": "done",
            "review": "review",
        },
    )

    # Review outcome: fix, regenerate, or fail
    g.add_conditional_edges(
        "review",
        decide_after_review,
        {
            "fix": "fix",
            "regenerate": "generate",
            "fail": "fail",
        },
    )

    g.add_edge("fix", "write")
    g.add_edge("done", END)

    return g.compile()
