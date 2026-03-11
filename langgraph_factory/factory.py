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
    close_log_file,
    extract_referenced_paths,
    init_log_file,
    log_detail,
    log_step,
    normalize_path,
    parse_fenced_files,
    tee_print,
)


class FactoryState(TypedDict, total=False):
    spec: str
    architecture_contract: dict
    acceptance: list[str]
    manifest: list[dict]
    scaffold_files: dict[str, str]  # pristine scaffold from create-next-app
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
    tee_print("[detail] build log (first 60 lines)")
    tee_print("\n".join(head))
    tee_print("[detail] build log (last 200 lines)")
    tee_print("\n".join(tail))
    if patched_files:
        log_detail(f"Patched files: {', '.join(patched_files)}")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def policy_node(state: FactoryState) -> dict:
    """Generate an architecture contract from the app spec."""
    # Initialize log file at the start of the pipeline
    project_dir = state.get("project_dir")
    if project_dir:
        init_log_file(project_dir)
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
            "Include a library_notes section with correct usage patterns for any "
            "third-party npm packages the app will need. This is critical — the "
            "coder model will follow these notes exactly.",
        ],
        "output_schema": {
            "architecture_contract": {
                "project_layout": "src_app_router | app_router",
                "typescript": "boolean",
                "package_manager": "pnpm",
                "import_aliases": {"@/*": "./src/*"},
                "ui_strategy": {
                    "mode": "no_ui_imports",
                    "note": "Use plain HTML elements with Tailwind CSS. No component wrappers.",
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
                "library_notes": {
                    "marked": "marked() returns string | Promise<string>. Always cast: marked(content) as string. Never use async/await for marked.",
                    "<pkg>": "usage notes for any other third-party packages",
                },
                "acceptance": ["pnpm build"],
            },
        },
    })
    out, stats = dmr_chat_json(
        model=FOREMAN_MODEL, system=system, user=user,
        max_tokens=8000, temperature=0.4,
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


# Files provided by the scaffold that the coder should NOT regenerate
_SCAFFOLD_ONLY_FILES = frozenset({
    "package.json", "tsconfig.json", "next.config.mjs", "next.config.ts",
    "tailwind.config.ts", "postcss.config.mjs", ".eslintrc.json",
})

_REQUIRED_FILES = {"package.json", "tsconfig.json"}
_REQUIRED_PATTERNS = {
    "next.config": lambda p: p.startswith("next.config.") and not p.endswith(".ts"),
    "layout": lambda p: "layout" in p and p.endswith((".tsx", ".jsx")),
    "page": lambda p: p.endswith(("page.tsx", "page.jsx")),
}

MAX_MANIFEST_ATTEMPTS = 2


def _validate_manifest(manifest: list[dict], has_scaffold: bool = False) -> list[str]:
    """Check manifest for required files. Returns list of issues.

    When a scaffold exists, config files and layout are already provided,
    so we only validate app-specific files.
    """
    paths = {e.get("path", "") for e in manifest}
    issues = []

    if not has_scaffold:
        for required in _REQUIRED_FILES:
            if required not in paths:
                issues.append(f"missing {required}")

        for name, check in _REQUIRED_PATTERNS.items():
            if not any(check(p) for p in paths):
                issues.append(f"missing {name} file")

        # Check for the bad next.config.ts
        if any(p == "next.config.ts" for p in paths):
            issues.append("next.config.ts present — must be .mjs or .js (Next.js 14)")

    # Always require at least one page
    if not any(p.endswith(("page.tsx", "page.jsx")) for p in paths):
        issues.append("missing page file")

    # Warn if scaffold exists but manifest includes config files
    if has_scaffold:
        config_in_manifest = [p for p in paths if p in _SCAFFOLD_ONLY_FILES]
        if config_in_manifest:
            issues.append(
                f"scaffold already provides these — remove from manifest: {', '.join(config_in_manifest)}"
            )

    return issues


def _build_manifest_prompt(state: FactoryState, issues: list[str] | None = None):
    """Build manifest system/user prompts, optionally including prior issues."""
    scaffold_files = state.get("scaffold_files", {})
    has_scaffold = bool(scaffold_files)

    system = (
        "You are a senior software architect planning a Next.js App Router project. "
        "Return STRICT JSON only. "
        'Schema: {"files": [{"path": "relative/path", "description": "what this file does", '
        '"imports_from": ["other/project/paths"], '
        '"category": "config|lib|style|component|page|api"}]} '
    )

    if has_scaffold:
        scaffold_listing = ", ".join(sorted(scaffold_files.keys()))
        system += (
            "A project scaffold has already been created via create-next-app. "
            f"These files ALREADY EXIST and must NOT appear in your manifest: {scaffold_listing}. "
            "Plan ONLY the app-specific files that need to be ADDED or REPLACED: "
            "pages, API routes, lib utilities, components, and styles. "
            "You may include src/app/layout.tsx and src/app/globals.css if the app needs "
            "custom layout or styles (they will replace the scaffold defaults). "
            "Do NOT plan config files (package.json, tsconfig.json, next.config.mjs, "
            "tailwind.config.ts, postcss.config.mjs, .eslintrc.json). "
            "Do NOT plan generic UI wrapper components (Button, Input, Card, etc.) — "
            "use plain HTML elements with Tailwind CSS classes directly."
        )
    else:
        system += (
            "List EVERY file needed for pnpm install and pnpm build to pass. "
            "IMPORTANT: Use next.config.mjs (NOT next.config.ts)."
        )

    system += " imports_from lists other files IN THIS PROJECT that this file will import."

    constraints = []
    if not has_scaffold:
        constraints.append("Include all config files: package.json, tsconfig.json, next.config.mjs.")
        constraints.append("Include a root layout file (e.g. src/app/layout.tsx or app/layout.tsx) — Next.js requires this.")
    constraints.extend([
        "Include all source files: pages, lib, API routes, styles.",
        "imports_from should only reference paths within this project.",
        "Do NOT include file contents — only paths, descriptions, and metadata.",
    ])
    if has_scaffold:
        constraints.append(
            "Do NOT include generic UI primitive wrappers (components/ui/button.tsx, etc.). "
            "Use plain <button>, <input>, <select>, <table> HTML elements with Tailwind classes."
        )
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
    has_scaffold = bool(state.get("scaffold_files"))

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

        issues = _validate_manifest(manifest, has_scaffold=has_scaffold)
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
# Manifest review (foreman)
# ---------------------------------------------------------------------------


def review_manifest_node(state: FactoryState) -> dict:
    """Have the foreman (gpt-oss) review the manifest for quality and relevance.

    Catches problems like unnecessary UI wrapper components, missing files,
    or patterns that don't match the architecture policy.
    """
    log_step("Review manifest")
    t0 = time.monotonic()

    manifest = state.get("manifest", [])
    contract = state.get("architecture_contract", {})
    spec = _require_spec(state)

    # Format manifest for the reviewer
    manifest_summary = "\n".join(
        f"  {e.get('category', '?'):10s}  {e.get('path', '?')}  — {e.get('description', '')}"
        for e in manifest
    )

    has_scaffold = bool(state.get("scaffold_files"))
    scaffold_note = ""
    if has_scaffold:
        scaffold_listing = ", ".join(sorted(state.get("scaffold_files", {}).keys()))
        scaffold_note = (
            f"\n\nIMPORTANT: A project scaffold (create-next-app) already provides these files: "
            f"{scaffold_listing}. "
            "The manifest should contain ONLY app-specific files (pages, API routes, lib, components). "
            "If the manifest includes config files or generic UI wrappers that the scaffold already "
            "provides, flag them for removal."
        )

    system = (
        "You are a senior software architect reviewing a file manifest for a Next.js project.\n"
        "Your job is to ensure the planned files are appropriate, necessary, and consistent with "
        "the architecture policy.\n\n"
        "Return STRICT JSON with this schema:\n"
        '{"action": "approve" | "trim", "remove_paths": ["path/to/remove", ...], '
        '"reasoning": "why", "warnings": ["optional notes for the coder"]}\n\n'
        "Guidelines:\n"
        "- Remove any files that duplicate framework functionality or create unnecessary abstractions.\n"
        "- Remove generic UI wrapper components (components/ui/button.tsx, etc.) — "
        "use plain HTML elements with Tailwind CSS classes directly.\n"
        "- Keep files that are genuinely needed: pages, API routes, lib utilities, domain components.\n"
        "- Config files should NOT be in the manifest if a scaffold provides them.\n"
        "- When in doubt, keep the file — the build-fix loop can handle extras better than gaps."
        + scaffold_note
    )
    user = json.dumps({
        "app_spec": spec,
        "architecture_contract": contract,
        "manifest": manifest_summary,
        "file_count": len(manifest),
    })

    out, stats = dmr_chat_json(
        model=FOREMAN_MODEL, system=system, user=user,
        max_tokens=8000, temperature=0.3,
        label="manifest-review",
    )

    action = out.get("action", "approve")
    remove_paths = out.get("remove_paths", [])
    reasoning = out.get("reasoning", "")
    warnings = out.get("warnings", [])

    if action == "trim" and remove_paths:
        # Filter out the removed paths
        original_count = len(manifest)
        protected = {"package.json", "tsconfig.json", "next.config.mjs", "next.config.js"}
        safe_removals = [p for p in remove_paths if p not in protected]
        manifest = [e for e in manifest if e.get("path") not in safe_removals]
        log_detail(
            f"Manifest review: trimmed {original_count - len(manifest)} files "
            f"({original_count} → {len(manifest)})"
        )
        for p in safe_removals:
            if p in {e.get("path") for e in state.get("manifest", [])}:
                log_detail(f"  removed: {p}")
        if set(remove_paths) - set(safe_removals):
            log_detail(f"  protected (kept): {', '.join(set(remove_paths) - set(safe_removals))}")
    else:
        log_detail(f"Manifest review: approved ({len(manifest)} files)")

    if reasoning:
        log_detail(f"Reviewer reasoning: {reasoning}")
    for w in warnings:
        log_detail(f"Reviewer warning: {w}")

    elapsed = time.monotonic() - t0
    timings = _record_step(state, "review_manifest", elapsed, stats,
                           action=action, removed=len(remove_paths) if remove_paths else 0)
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


def _sanitize_generated_files(files: dict[str, str]) -> dict[str, str]:
    """Fix known generation issues before the first build.

    Deterministic fixes applied post-generate to avoid wasting build cycles
    on problems we've seen the model repeat.
    """
    files = dict(files)

    # --- Fix: next.config.ts → next.config.mjs ---
    # Next.js 14 does not support TypeScript config files.
    if "next.config.ts" in files and "next.config.mjs" not in files:
        content = files.pop("next.config.ts")
        # Convert TS export syntax to ESM if needed
        # e.g. "export default { ... }" is already valid ESM
        # but "const config: NextConfig = ..." needs the type annotation stripped
        content = re.sub(r":\s*NextConfig\b", "", content)
        content = re.sub(r"import\s+type\s*\{[^}]*\}\s*from\s*['\"]next['\"];?\n?", "", content)
        files["next.config.mjs"] = content
        log_detail("Sanitize: renamed next.config.ts → next.config.mjs")

    # --- Fix: missing root layout ---
    # Next.js App Router requires a root layout.tsx in the app directory.
    layout_paths = [p for p in files if re.match(r"^(src/)?app/layout\.(tsx?|jsx?)$", p)]
    if not layout_paths:
        # Determine if project uses src/ prefix
        uses_src = any(p.startswith("src/") for p in files)
        layout_path = "src/app/layout.tsx" if uses_src else "app/layout.tsx"
        files[layout_path] = (
            'import type {{ Metadata }} from "next";\n'
            'import "./globals.css";\n'
            "\n"
            "export const metadata: Metadata = {{\n"
            '  title: "App",\n'
            '  description: "Generated by langgraph-factory",\n'
            "}};\n"
            "\n"
            "export default function RootLayout({{\n"
            "  children,\n"
            "}}: {{\n"
            "  children: React.ReactNode;\n"
            "}}) {{\n"
            "  return (\n"
            '    <html lang="en">\n'
            "      <body>{{children}}</body>\n"
            "    </html>\n"
            "  );\n"
            "}}\n"
        ).replace("{{", "{").replace("}}", "}")

        # Also ensure globals.css exists if referenced
        css_path = "src/app/globals.css" if uses_src else "app/globals.css"
        if css_path not in files:
            files[css_path] = "/* Global styles */\n"

        log_detail(f"Sanitize: injected missing root layout at {layout_path}")

    return files


# ---------------------------------------------------------------------------
# Scaffold (npx create-next-app)
# ---------------------------------------------------------------------------


def scaffold_node(state: FactoryState) -> dict:
    """Run npx create-next-app to create a working project skeleton.

    This gives us correct config files (package.json, tsconfig.json,
    next.config.mjs, layout.tsx, globals.css, tailwind.config.ts) so
    the coder only needs to add the actual app code on top.
    """
    log_step("Scaffold project via create-next-app")
    t0 = time.monotonic()
    project_dir = _require_project_dir(state)

    # Use a temp directory for the scaffold, then read files
    scaffold_dir = os.path.join(project_dir, "scaffold-tmp")
    if os.path.exists(scaffold_dir):
        shutil.rmtree(scaffold_dir)

    cmd = [
        "npx", "create-next-app@14",
        scaffold_dir,
        "--typescript",
        "--tailwind",
        "--app",
        "--src-dir",
        "--use-pnpm",
        "--eslint",
        "--import-alias", "@/*",
    ]
    log_detail(f"Running: {' '.join(cmd)}")
    result = _run_cmd(cmd, cwd=project_dir)

    if result.returncode != 0:
        log_detail(f"Scaffold failed (exit {result.returncode}): {result.stdout[-500:]}")
        # Fall through — generate node will create everything from scratch
        elapsed = time.monotonic() - t0
        timings = _record_step(state, "scaffold", elapsed, status="FAILED")
        return {"step_timings": timings}

    # Read all scaffold files into the files dict
    scaffold_files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(scaffold_dir, topdown=True):
        # Skip node_modules and .git
        dirnames[:] = [d for d in dirnames if d not in {"node_modules", ".git", ".next"}]
        for fname in filenames:
            abs_path = os.path.join(dirpath, fname)
            rel = os.path.relpath(abs_path, scaffold_dir)
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    scaffold_files[rel] = f.read()
            except (UnicodeDecodeError, OSError):
                continue  # skip binary files

    # Clean up temp scaffold
    shutil.rmtree(scaffold_dir, ignore_errors=True)

    log_detail(f"Scaffold created {len(scaffold_files)} files")
    for p in sorted(scaffold_files):
        log_detail(f"  scaffold: {p}")

    elapsed = time.monotonic() - t0
    timings = _record_step(state, "scaffold", elapsed, files=len(scaffold_files))
    return {"scaffold_files": scaffold_files, "files": scaffold_files, "step_timings": timings}


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def generate_node(state: FactoryState) -> dict:
    """Generate the app-specific project files in a single model call.

    Builds on top of the scaffold (if present) — the coder only needs to
    produce pages, API routes, lib, and components. Config files come from
    the scaffold and are merged afterward.
    """
    gen_attempt = state.get("generate_attempt", 0) + 1
    log_step(f"Generate project — monolithic (attempt {gen_attempt})")
    t0 = time.monotonic()

    scaffold_files = state.get("scaffold_files", {})
    has_scaffold = bool(scaffold_files) and "package.json" in scaffold_files

    if has_scaffold:
        # List scaffold files so the coder knows what already exists
        scaffold_listing = "\n".join(f"  {p}" for p in sorted(scaffold_files) if p not in _SCAFFOLD_ONLY_FILES)
        scaffold_context = (
            "A project scaffold has already been created via create-next-app with these files:\n"
            f"{scaffold_listing}\n\n"
            "IMPORTANT scaffold rules:\n"
            "- Do NOT output package.json, tsconfig.json, next.config.mjs, tailwind.config.ts, "
            "postcss.config.mjs, or .eslintrc.json — these already exist from the scaffold.\n"
            "- You MUST output src/app/layout.tsx and src/app/globals.css to replace the scaffold defaults.\n"
            "- You CAN add new npm dependencies — output a file called EXTRA_DEPS.json with "
            '{"dependencies": {"pkg": "version"}, "devDependencies": {"pkg": "version"}} '
            "and they will be merged into the scaffold package.json.\n"
            "- All your files will be added ON TOP of the scaffold.\n"
        )
    else:
        scaffold_context = ""

    system = (
        "You are a meticulous senior engineer generating a complete runnable project.\n"
        "Output each file using this EXACT fence format (no JSON, no markdown):\n\n"
        "===FILE: relative/path.ext===\n"
        "file contents here\n"
        "===END FILE===\n\n"
    )
    if has_scaffold:
        system += scaffold_context
    else:
        system += "Do not omit required config files; ensure pnpm build will succeed.\n"

    system += (
        "Obey the architecture_contract strictly.\n"
        "Every npm package you import MUST appear in package.json dependencies "
        "(or in EXTRA_DEPS.json if a scaffold is present).\n"
        "Every file you import from within the project MUST be included in your output.\n"
        "If you create component files, they must export everything that other files import from them.\n"
        "Put shared TypeScript types and in-memory stores in a dedicated file (e.g. lib/types.ts, lib/store.ts) "
        "and import from there — NEVER duplicate type definitions or stores across files.\n"
        "CRITICAL UI rules:\n"
        "- Do NOT import from @radix-ui/* or use shadcn patterns unless the architecture_contract explicitly requires them.\n"
        "- Do NOT create mock or wrapper files for third-party component libraries.\n"
        "- UI components should use plain HTML elements (button, select, input, table) styled with Tailwind CSS.\n"
        "- Keep component files short and simple — no component needs forwardRef, Slot, or Primitive patterns for an MVP.\n"
        "KNOWN LIBRARY PITFALLS:\n"
        "- The 'marked' package: marked() returns string | Promise<string>. "
        "Always cast the result: `marked(markdown) as string` or use `marked.parse(markdown) as string`. "
        "Do NOT declare a synchronous return type `: string` on a function that calls marked() without casting.\n"
        "- ESLint: Use `const` instead of `let` when the variable is never reassigned. "
        "For unused catch parameters, omit the parameter entirely: `catch {` (NOT `catch (error)` or `catch (_error)`). "
        "Remove unused imports and variables.\n"
        "- Next.js 14 params: In Next.js 14, route params are a plain object — access `params.id` directly. "
        "Do NOT use `React.use(params)` or `use(params.id)` — that is a Next.js 15 pattern and will cause type errors.\n"
        "- ESLint no-explicit-any: NEVER use `any` as a type. Use specific types: "
        "`React.FormEvent<HTMLFormElement>` for form submit handlers, "
        "`React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>` for input handlers, "
        "`unknown` with type guards for catch blocks."
    )

    constraints = [
        "Project must be Next.js App Router + TypeScript.",
        "Keep it minimal but complete.",
        "Derive ALL entity fields, pages, and routes from the app_spec and architecture_contract — do not hardcode or omit fields.",
        "Sample/seed data must include every field defined in the entity type.",
        "In-memory store is fine; prefer server components + route handlers or server actions.",
        "Avoid external DB for MVP.",
    ]
    if not has_scaffold:
        constraints.insert(1, "Provide all files required for pnpm install and pnpm build.")

    # Include the manifest so the coder knows exactly what files to generate
    manifest = state.get("manifest", [])
    manifest_listing = [e.get("path", "") for e in manifest if e.get("path")]
    if manifest_listing:
        constraints.append(
            f"Generate EXACTLY these files (no more, no fewer): {', '.join(manifest_listing)}. "
            "Do NOT create files that are not in this list. "
            "Do NOT create components/ui/ wrapper files unless they appear in this list."
        )

    user_payload: dict = {
        "app_spec": _require_spec(state),
        "architecture_contract": state.get("architecture_contract", {}),
        "file_manifest": [
            {"path": e.get("path"), "description": e.get("description", "")}
            for e in manifest
        ],
        "constraints": constraints,
        "output_requirements": [
            "Output ONLY ===FILE: path=== ... ===END FILE=== blocks.",
            "Every file must contain full, complete code (no placeholders).",
            "Do not reference files that you did not include.",
            "Honor ui_strategy mode in architecture_contract.",
            "Do NOT wrap file contents in markdown code fences (```). Output raw code only.",
        ],
        "attempt": gen_attempt,
    }

    # On retry, feed failure context so the model doesn't repeat mistakes
    if gen_attempt > 1:
        fix_history = state.get("fix_history", [])
        error_patterns = []
        for entry in fix_history:
            summary = entry.get("error_summary", "")
            if summary:
                error_patterns.append(summary)
        seen = set()
        unique_errors = []
        for e in error_patterns:
            key = e[:100]
            if key not in seen:
                seen.add(key)
                unique_errors.append(e)

        reviewer_verdict = state.get("review_verdict", {})
        user_payload["previous_attempt_failures"] = {
            "error_patterns": unique_errors[:5],
            "reviewer_reasoning": reviewer_verdict.get("reasoning", ""),
            "instructions": [
                "The previous generation attempt failed to build after multiple fix attempts.",
                "Avoid the error patterns listed above.",
                "Ensure all component props interfaces are complete — include variant, size, asChild, and other common props.",
                "Include a root layout.tsx in the app directory.",
                "Every TypeScript lambda parameter must have an explicit type annotation.",
            ],
        }

        # Log what we're feeding back
        log_detail(f"Regeneration context: {len(unique_errors)} error patterns from previous attempt")
        for i, err in enumerate(unique_errors[:5], 1):
            log_detail(f"  error {i}: {err[:150]}")
        if reviewer_verdict.get("reasoning"):
            log_detail(f"  reviewer reasoning: {reviewer_verdict['reasoning'][:200]}")

    user = json.dumps(user_payload)

    raw, stats = dmr_chat_raw(
        model=CODER_MODEL, system=system, user=user,
        max_tokens=65536, temperature=0.2,
        label="generate-all",
    )
    elapsed = time.monotonic() - t0
    log_detail(f"Model response received in {elapsed:.1f}s")

    generated_files = parse_fenced_files(raw)
    if not generated_files:
        log_detail(f"No fenced files found. Response start: {raw[:500]!r}")
        raise RuntimeError("Generate step returned no files in fence format")

    log_detail(f"Generated {len(generated_files)} files")

    # Detect degenerate output: if we got far fewer files than the manifest
    # planned, the model likely went off the rails (e.g. CSS repetition loop)
    manifest = state.get("manifest", [])
    expected_count = len(manifest) if manifest else 3  # minimum sanity
    if stats and stats.finish_reason == "length" and len(generated_files) < expected_count // 2:
        log_detail(
            f"WARNING: degenerate generation — {len(generated_files)} files vs "
            f"{expected_count} expected, finish_reason=length. Discarding output."
        )
        # If this is a retry, we've exhausted our attempts
        if gen_attempt >= MAX_GENERATE_ATTEMPTS:
            raise RuntimeError(
                f"Generate step produced degenerate output on attempt {gen_attempt} "
                f"({len(generated_files)} files, finish_reason=length). Giving up."
            )
        # Otherwise, record the attempt and let the graph retry
        timings = _record_step(state, "generate", elapsed, stats,
                               files=len(generated_files), attempt=gen_attempt,
                               degenerate=True)
        return {
            "generate_attempt": gen_attempt, "fix_attempt": 0,
            "step_timings": timings,
            "last_build_ok": False,
            "last_build_log": f"Degenerate generation: {len(generated_files)} files, expected {expected_count}",
        }

    # Merge with scaffold: scaffold provides the base, generated files overlay
    if has_scaffold:
        merged = dict(scaffold_files)

        # Handle EXTRA_DEPS.json: merge into scaffold package.json
        extra_deps_raw = generated_files.pop("EXTRA_DEPS.json", None)
        if extra_deps_raw:
            try:
                extra = json.loads(extra_deps_raw)
                pkg = json.loads(merged.get("package.json", "{}"))
                for dep, ver in extra.get("dependencies", {}).items():
                    pkg.setdefault("dependencies", {})[dep] = ver
                for dep, ver in extra.get("devDependencies", {}).items():
                    pkg.setdefault("devDependencies", {})[dep] = ver
                merged["package.json"] = json.dumps(pkg, indent=2)
                log_detail(f"Merged EXTRA_DEPS: {list(extra.get('dependencies', {}).keys())}")
            except (json.JSONDecodeError, AttributeError) as e:
                log_detail(f"WARNING: could not parse EXTRA_DEPS.json: {e}")

        # Overlay generated files (skip config files the scaffold owns,
        # unless the coder explicitly replaced them)
        for path, content in generated_files.items():
            merged[path] = content

        files = merged
    else:
        files = generated_files

    # Reconcile imports: ensure every npm package used in source files
    # is listed in package.json dependencies
    files = _reconcile_imports(files)

    # Sanitize known issues before the first build
    files = _sanitize_generated_files(files)

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
            if rel not in expected and fname not in ("pnpm-lock.yaml", "run.log", "summary.txt"):
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
    has_modules = os.path.isdir(node_modules) and os.listdir(node_modules)
    if has_modules and pkg_hash and pkg_hash == last_hash:
        log_detail("deps already installed, skipping")
        elapsed = time.monotonic() - t0
        timings = _record_step(state, "install", elapsed, skipped=True)
        return {"step_timings": timings}

    if shutil.which("pnpm") is None:
        raise RuntimeError("pnpm not found in PATH")

    p = _run_cmd(["pnpm", "install"], cwd=project_dir)

    # If install failed due to 404 packages, remove them from package.json
    # AND strip their imports from source files, then retry
    files_updated = False
    if p.returncode != 0 and "ERR_PNPM_FETCH_404" in (p.stdout or ""):
        bad_pkgs = re.findall(
            r"ERR_PNPM_FETCH_404.*?registry\.npmjs\.org/([^\s:]+)",
            p.stdout or "",
        )
        if bad_pkgs:
            files = dict(state.get("files", {}))
            pkg_json_str = files.get("package.json", "")
            try:
                pkg = json.loads(pkg_json_str)
                deps = pkg.get("dependencies", {})
                removed = []
                for raw in bad_pkgs:
                    pkg_name = raw.replace("%2F", "/")
                    if pkg_name in deps:
                        del deps[pkg_name]
                        removed.append(pkg_name)
                if removed:
                    pkg["dependencies"] = deps
                    new_pkg_json = json.dumps(pkg, indent=2)
                    files["package.json"] = new_pkg_json
                    # Write fixed package.json to disk
                    pkg_path = os.path.join(project_dir, "package.json")
                    with open(pkg_path, "w", encoding="utf-8") as f:
                        f.write(new_pkg_json)
                    # Strip imports of removed packages from source files
                    for pkg_name in removed:
                        import_re = re.compile(
                            rf"^import\s+.*?from\s+['\"](?:{re.escape(pkg_name)})['\"];?\s*\n?",
                            re.MULTILINE,
                        )
                        for fpath in list(files.keys()):
                            if not fpath.endswith((".ts", ".tsx", ".js", ".jsx")):
                                continue
                            new_content = import_re.sub("", files[fpath])
                            if new_content != files[fpath]:
                                files[fpath] = new_content
                                # Write updated source to disk
                                abs_path = os.path.join(project_dir, fpath)
                                with open(abs_path, "w", encoding="utf-8") as f:
                                    f.write(new_content)
                                log_detail(f"Install fix: stripped import of {pkg_name} from {fpath}")
                    files_updated = True
                    log_detail(f"Install fix: removed non-existent packages: {', '.join(removed)}")
                    # Retry install
                    p = _run_cmd(["pnpm", "install"], cwd=project_dir)
            except (json.JSONDecodeError, KeyError):
                pass

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
    result: dict = {
        "last_installed_package_json_hash": pkg_hash or last_hash,
        "step_timings": timings,
    }
    # If we patched files during install (removed bad packages + their imports),
    # propagate the updated files dict back to state
    if files_updated:
        result["files"] = files
        result["package_json_hash"] = _hash_text(files["package.json"])
    return result


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


def _try_mechanical_fix(build_log: str, files: dict[str, str]) -> tuple[dict[str, str], list[str]] | None:
    """Try to fix build errors mechanically without calling the LLM.

    Returns (patches, removals) where patches is {path: new_content} and
    removals is a list of paths to delete from the file map. Returns None
    if no mechanical fix applies.
    """
    patches: dict[str, str] = {}
    # Track files to remove from the files dict (renames)
    removals: list[str] = []

    # --- Fix 0: next.config.ts → next.config.mjs ---
    if "next.config.ts is not supported" in build_log or "Configuring Next.js via 'next.config.ts'" in build_log:
        if "next.config.ts" in files and "next.config.mjs" not in files:
            content = files["next.config.ts"]
            content = re.sub(r":\s*NextConfig\b", "", content)
            content = re.sub(r"import\s+type\s*\{[^}]*\}\s*from\s*['\"]next['\"];?\n?", "", content)
            patches["next.config.mjs"] = content
            removals.append("next.config.ts")
            log_detail("Mechanical fix: renamed next.config.ts → next.config.mjs")

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

    # --- Fix 3: "Property 'X' does not exist on type '... & SomeProps'" ---
    # The model generates a component with an incomplete props interface, then
    # uses props that aren't defined.  Fix: add the missing prop as optional.
    missing_props = re.findall(
        r"Property '(\w+)' does not exist on type 'IntrinsicAttributes & (\w+)'",
        build_log,
    )
    if missing_props:
        # Build index: find which file defines each interface/type
        _INTERFACE_RE = re.compile(
            r"(?:export\s+)?(?:interface|type)\s+(\w+)\s*(?:extends\s+[^{]+)?\{",
        )
        type_to_file: dict[str, str] = {}
        for fpath, content in files.items():
            if not fpath.endswith((".ts", ".tsx")):
                continue
            for m in _INTERFACE_RE.finditer(content):
                type_to_file[m.group(1)] = fpath

        # Also try to infer prop types from the error's assignable-from type
        # e.g. "Type '{ asChild: true; variant: "secondary"; size: string; }'"
        prop_types: dict[str, str] = {}
        type_block = re.search(
            r"Type '\{([^}]+)\}' is not assignable", build_log,
        )
        if type_block:
            for prop_match in re.finditer(
                r"(\w+):\s*(?:\"[^\"]*\"|'[^']*'|true|false|\w+)",
                type_block.group(1),
            ):
                pname = prop_match.group(0)
                # Extract name and value to infer type
                parts = pname.split(":", 1)
                if len(parts) == 2:
                    val = parts[1].strip().strip("\"'")
                    if val in ("true", "false"):
                        prop_types[parts[0].strip()] = "boolean"
                    else:
                        prop_types[parts[0].strip()] = "string"

        # Group missing props by their target type
        props_by_type: dict[str, set[str]] = {}
        for prop_name, type_name in missing_props:
            props_by_type.setdefault(type_name, set()).add(prop_name)

        for type_name, prop_names in props_by_type.items():
            if type_name not in type_to_file:
                continue
            fpath = type_to_file[type_name]
            content = patches.get(fpath, files[fpath])

            # Find the interface/type block and add missing props before
            # the closing brace
            for prop_name in prop_names:
                # Skip if already defined
                if re.search(rf"\b{re.escape(prop_name)}\s*[?:]", content):
                    continue
                inferred_type = prop_types.get(prop_name, "string")
                new_prop = f"  {prop_name}?: {inferred_type};\n"

                # Insert before the closing brace of the interface/type
                # Find the interface definition, then its closing brace
                pattern = re.compile(
                    rf"((?:export\s+)?(?:interface|type)\s+{re.escape(type_name)}\s*"
                    rf"(?:extends\s+[^{{]+)?\{{[^}}]*?)(}})",
                    re.DOTALL,
                )
                match = pattern.search(content)
                if match:
                    content = content[:match.end(1)] + new_prop + content[match.start(2):]
                    log_detail(f"Mechanical fix: added {prop_name}?: {inferred_type} to {type_name} in {fpath}")

            patches[fpath] = content

    # --- Fix 4: unused catch variables ---
    # ESLint flags both `catch (error)` and `catch (_error)` as unused.
    # The correct fix is to omit the parameter: `catch {`
    unused_catch_vars = re.findall(
        r"^\./([^:]+):\d+:\d+\s+Error:\s+'(_?\w+)'\s+is defined but never used\.\s+@typescript-eslint/no-unused-vars",
        build_log, re.MULTILINE,
    )
    if unused_catch_vars:
        # Group by file
        catch_fixes_by_file: dict[str, set[str]] = {}
        for fpath, var_name in unused_catch_vars:
            catch_fixes_by_file.setdefault(fpath, set()).add(var_name)

        for fpath, var_names in catch_fixes_by_file.items():
            if fpath not in files:
                continue
            content = patches.get(fpath, files[fpath])
            changed = False
            for var_name in var_names:
                # Remove unused catch parameter: catch (error) { → catch {
                new_content = re.sub(
                    rf"\bcatch\s*\(\s*{re.escape(var_name)}\s*(?::\s*\w+)?\s*\)",
                    "catch",
                    content,
                )
                # Remove unused import: import { X } from "...";
                if new_content == content:
                    new_content = re.sub(
                        rf"^import\s+\{{[^}}]*\b{re.escape(var_name)}\b[^}}]*\}}\s+from\s+['\"][^'\"]+['\"];?\s*\n",
                        "",
                        content,
                        flags=re.MULTILINE,
                    )
                if new_content != content:
                    content = new_content
                    changed = True
            if changed:
                patches[fpath] = content
                log_detail(f"Mechanical fix: removed unused catch params/imports in {fpath}")

    # --- Fix 5: unused standalone imports ---
    # e.g. "'NextRequest' is defined but never used"
    unused_imports = re.findall(
        r"^\./([^:]+):\d+:\d+\s+Error:\s+'(\w+)'\s+is defined but never used\.\s+@typescript-eslint/no-unused-vars",
        build_log, re.MULTILINE,
    )
    if unused_imports:
        imports_by_file: dict[str, set[str]] = {}
        for fpath, var_name in unused_imports:
            imports_by_file.setdefault(fpath, set()).add(var_name)

        for fpath, var_names in imports_by_file.items():
            if fpath not in files:
                continue
            content = patches.get(fpath, files[fpath])
            changed = False
            for var_name in var_names:
                # Skip if already handled by catch fix
                if fpath in catch_fixes_by_file and var_name in catch_fixes_by_file.get(fpath, set()):
                    continue
                # Remove the name from a multi-import: import { A, B } from "..."
                # If it's the only import, remove the whole line
                # Single import: import { NextRequest } from "next/server";
                single_import = re.compile(
                    rf"^import\s+\{{\s*{re.escape(var_name)}\s*\}}\s+from\s+['\"][^'\"]+['\"];?\s*\n",
                    re.MULTILINE,
                )
                if single_import.search(content):
                    content = single_import.sub("", content)
                    changed = True
                else:
                    # Multi-import: remove just this name from { A, B, C }
                    # "var_name, " or ", var_name"
                    new_content = re.sub(
                        rf"\b{re.escape(var_name)}\s*,\s*", "", content,
                    )
                    if new_content == content:
                        new_content = re.sub(
                            rf",\s*{re.escape(var_name)}\b", "", content,
                        )
                    if new_content != content:
                        content = new_content
                        changed = True
            if changed:
                patches[fpath] = content
                log_detail(f"Mechanical fix: removed unused imports in {fpath}")

    if not patches and not removals:
        return None
    return patches, removals


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
        result = _try_mechanical_fix(build_log, files)
        if result:
            mechanical_patches, removals = result
            patched_files = sorted(mechanical_patches.keys())
            log_detail(f"Mechanical fix patched: {', '.join(patched_files)}")
            merged = dict(files)
            for r in removals:
                merged.pop(r, None)
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
        "IMPORTANT: Use next.config.mjs (NOT next.config.ts) — Next.js 14 does not support TypeScript config files.\n"
        "Do NOT import from @radix-ui/* or use shadcn patterns. Use plain HTML elements styled with Tailwind CSS.\n"
        "KNOWN FIX PATTERNS:\n"
        "- marked() returns string | Promise<string>. Cast it: `marked(markdown) as string`. "
        "Do NOT make wrapper functions async just to handle this — cast instead.\n"
        "- ESLint 'defined but never used': remove the catch parameter entirely — use `catch {` instead of `catch (error)`. "
        "For unused imports, delete the import line.\n"
        "- ESLint 'never reassigned, use const': change `let` to `const`.\n"
        "- 'X is possibly undefined': add a guard `if (!x) return notFound();` or `if (!x) throw ...` before usage."
    )

    # Send app-specific project files as context (skip scaffold boilerplate)
    # Scaffold config files don't need to be in the fix context — they're correct
    _SKIP_IN_FIX = _SCAFFOLD_ONLY_FILES | {
        "next-env.d.ts", "README.md", "pnpm-lock.yaml",
    }
    snapshot_text = ""
    for path in sorted(files.keys()):
        if path in _SKIP_IN_FIX:
            continue
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

    # Parse deletions (with guardrails — never delete critical files)
    _PROTECTED_FILES = {"package.json", "tsconfig.json", "next.config.mjs", "next.config.js"}
    deletion_pattern = re.compile(r"===DELETE:\s*(.+?)\s*===")
    raw_deletions = deletion_pattern.findall(raw)
    deletions = [d for d in raw_deletions if d not in _PROTECTED_FILES]
    blocked = [d for d in raw_deletions if d in _PROTECTED_FILES]
    if blocked:
        log_detail(f"Blocked deletion of protected files: {', '.join(blocked)}")

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
    tee_print("\n" + summary)

    project_dir = state.get("project_dir")
    if project_dir:
        summary_path = os.path.join(project_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        log_detail(f"Summary written to {summary_path}")


def done_node(state: FactoryState) -> dict:
    _emit_summary(state)
    close_log_file()
    return {}


def fail_node(state: FactoryState) -> dict:
    _emit_summary(state)
    close_log_file()
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
    result = _try_mechanical_fix(build_log, files)
    if result:
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

    # --- Spin detection: if the same file has been patched 2+ times,
    #     the fix loop isn't making progress ---
    patch_counts: dict[str, int] = {}
    for entry in fix_history:
        for p in entry.get("patches", []):
            if not p.startswith("DELETE:"):
                patch_counts[p] = patch_counts.get(p, 0) + 1
    repeat_patches = {f: n for f, n in patch_counts.items() if n >= 2}

    if repeat_patches and gen_attempt < MAX_GENERATE_ATTEMPTS:
        repeat_summary = ", ".join(f"{f} ({n}x)" for f, n in repeat_patches.items())
        log_detail(f"Spin detected: repeated patches to {repeat_summary} — forcing regenerate")
        elapsed = time.monotonic() - t0
        timings = _record_step(state, "review", elapsed, spin_detected=True)
        return {
            "review_verdict": {
                "action": "regenerate",
                "reasoning": f"Fix loop is spinning: {repeat_summary} patched repeatedly without progress.",
                "guidance": "",
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


def decide_after_generate(state: FactoryState) -> str:
    """Route after generate: write if files present, fail if degenerate."""
    files = state.get("files")
    if files:
        return "write"
    # No files means degenerate output — check if we can retry
    gen_attempt = state.get("generate_attempt", 0)
    if gen_attempt < MAX_GENERATE_ATTEMPTS:
        return "regenerate"
    return "fail"


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

    Pipeline: policy -> scaffold -> manifest -> review_manifest -> generate
              -> write -> install -> build
    with fix loop (mechanical + LLM) and regeneration fallback.
    """
    g = StateGraph(FactoryState)

    g.add_node("policy", policy_node)
    g.add_node("scaffold", scaffold_node)
    g.add_node("manifest", manifest_node)
    g.add_node("review_manifest", review_manifest_node)
    g.add_node("generate", generate_node)
    g.add_node("write", write_node)
    g.add_node("install", install_node)
    g.add_node("build", build_node)
    g.add_node("review", review_node)
    g.add_node("fix", fix_node)
    g.add_node("done", done_node)
    g.add_node("fail", fail_node)

    g.add_edge(START, "policy")
    g.add_edge("policy", "scaffold")
    g.add_edge("scaffold", "manifest")
    g.add_edge("manifest", "review_manifest")
    g.add_edge("review_manifest", "generate")

    g.add_conditional_edges(
        "generate",
        decide_after_generate,
        {
            "write": "write",
            "regenerate": "generate",
            "fail": "fail",
        },
    )
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
