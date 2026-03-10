"""MVP pipeline: plan manifest -> generate files -> verify -> write -> build check.

This is the simpler pipeline — a single-pass generation with retry on missing files.
See factory.py for the evolved version with build-fix loops.
"""

import json
import os
import subprocess
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from langgraph_factory.config import CODER_MODEL, FOREMAN_MODEL, OUTPUT_DIR, LINT_MAX_LOOPS
from langgraph_factory.llm import langchain_chat
from langgraph_factory.utils import extract_json, normalize_path, log_step


class Manifest(TypedDict):
    required_files: list[str]
    file_contexts: dict[str, str]
    global_context: str
    acceptance: list[str]
    notes: str


class MVPState(TypedDict, total=False):
    spec: str
    manifest: Manifest
    required_files: list[str]
    file_contexts: dict[str, str]
    files: dict[str, str]
    missing_files: list[str]
    attempt: int
    max_attempts: int
    notes: str
    output_dir: str


def _require_spec(state: MVPState) -> str:
    spec = state.get("spec")
    if not spec:
        raise RuntimeError("Missing spec in state")
    return spec


def _out_dir(state: MVPState) -> str:
    return state.get("output_dir", OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def plan_node(state: MVPState) -> dict:
    """Call the foreman model to produce a file manifest."""
    log_step("Plan manifest")

    system = (
        "You are a software tech lead. Output STRICT JSON only. "
        "No markdown, no commentary."
    )
    user = f"""
Return STRICT JSON ONLY with this exact shape:
{{
  "required_files": [ ... ],
  "file_contexts": {{"path": "purpose", ...}},
  "global_context": "...",
  "acceptance": ["pnpm install", "pnpm dev"],
  "notes": "..."
}}

Constraints:
- MUST use Next.js App Router (no src/pages).
- MUST include a minimal test setup and at least one test file.
- MUST NOT include any routes outside the spec.
- Include enough detail in global_context + file_contexts for a coder to write files in isolation.

App spec:
{_require_spec(state)}
""".strip()

    last_error = None
    for _ in range(3):
        try:
            text = langchain_chat(FOREMAN_MODEL, system, user, temperature=0.3)
            payload = extract_json(text)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Plan JSON parse failed: {last_error}") from last_error

    required_files = [normalize_path(p) for p in payload.get("required_files", [])]
    file_contexts = {
        normalize_path(k): v for k, v in payload.get("file_contexts", {}).items()
    }
    if not required_files:
        raise RuntimeError("Manifest required_files is empty")

    manifest: Manifest = {
        "required_files": required_files,
        "file_contexts": file_contexts,
        "global_context": payload.get("global_context", ""),
        "acceptance": payload.get("acceptance", ["pnpm install", "pnpm dev"]),
        "notes": payload.get("notes", ""),
    }
    return {
        "manifest": manifest,
        "required_files": required_files,
        "file_contexts": file_contexts,
        "attempt": 0,
        "missing_files": [],
    }


def validate_manifest_node(state: MVPState) -> dict:
    """Validate manifest structure before generation."""
    log_step("Validate manifest")
    required = state.get("required_files", [])
    if not required:
        return {"missing_files": ["__manifest__:required_files_empty"]}

    forbidden = [p for p in required if p.startswith("src/pages") or "/pages/" in p]
    if forbidden:
        return {"missing_files": ["__manifest__:forbidden_pages_routes"]}

    checks = [
        ("app/", "missing_app"),
        ("components/", "missing_components"),
        ("lib/", "missing_lib"),
    ]
    missing = []
    for prefix, label in checks:
        if not any(p.startswith(prefix) for p in required):
            missing.append(f"__manifest__:{label}")

    if not any("test" in p or "__tests__" in p for p in required):
        missing.append("__manifest__:missing_tests")

    return {"missing_files": missing}


def generate_node(state: MVPState) -> dict:
    """Generate all files as a JSON mapping."""
    attempt = state.get("attempt", 0)
    log_step(f"Generate files (attempt {attempt + 1})")

    system = (
        "You are a meticulous full-stack engineer. Output STRICT JSON only. "
        "No markdown, no commentary. "
        'Return {"files": {"path":"content", ...}} and include EVERY required file.'
    )
    user = json.dumps({
        "spec": _require_spec(state),
        "required_files": state.get("required_files", []),
        "missing_files": state.get("missing_files", []),
        "instructions": [
            "Return STRICT JSON only.",
            "Include full file contents for every path.",
            "If a file is listed as missing_files, prioritize it.",
            "Do not omit package.json / tsconfig / next config if needed.",
        ],
        "attempt": attempt,
    })

    last_error = None
    for _ in range(3):
        try:
            text = langchain_chat(CODER_MODEL, system, user)
            payload = extract_json(text)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Generate JSON parse failed: {last_error}") from last_error

    files = payload.get("files", {})
    merged = dict(state.get("files", {}))
    merged.update(files)
    return {"files": merged, "attempt": attempt + 1}


def verify_node(state: MVPState) -> dict:
    """Verify all required files are present."""
    log_step("Verify generated files")
    required = state.get("required_files", [])
    files = state.get("files", {})

    forbidden = [p for p in files if p.startswith("src/pages") or "/pages/" in p]
    if forbidden:
        raise RuntimeError(f"Generated forbidden pages routes: {forbidden}")

    missing = [p for p in required if p not in files or not str(files[p]).strip()]
    return {"missing_files": missing}


def write_node(state: MVPState) -> dict:
    """Write generated files to disk."""
    log_step("Write files to disk")
    out_dir = _out_dir(state)
    os.makedirs(out_dir, exist_ok=True)

    for rel_path, content in state.get("files", {}).items():
        target = os.path.join(out_dir, rel_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

    return {"notes": f"Wrote files to {out_dir}"}


def build_check_node(state: MVPState) -> dict:
    """Run pnpm install + build if available."""
    log_step("Build check")
    out_dir = _out_dir(state)
    try:
        subprocess.run(["pnpm", "-v"], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return {"notes": "pnpm not available; skipped build check"}

    subprocess.run(["pnpm", "install"], check=True, cwd=out_dir)
    subprocess.run(["pnpm", "build"], check=True, cwd=out_dir)
    return {"notes": "Build check passed"}


def fail_node(state: MVPState) -> dict:
    missing = state.get("missing_files", [])
    raise RuntimeError(f"Failed to generate required files after retries. Missing: {missing}")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def should_retry(state: MVPState) -> str:
    missing = state.get("missing_files", [])
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 6)

    if not missing:
        return "write"
    if attempt >= max_attempts:
        return "fail"
    return "generate"


def manifest_retry(state: MVPState) -> str:
    missing = state.get("missing_files", [])
    return "plan" if missing else "generate"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_mvp_graph():
    """Build the MVP pipeline graph.

    Flow: plan -> validate_manifest -> generate -> verify -> write -> build_check
    With retry loops on manifest validation and file verification.
    """
    g = StateGraph(MVPState)

    g.add_node("plan", plan_node)
    g.add_node("validate_manifest", validate_manifest_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)
    g.add_node("write", write_node)
    g.add_node("build_check", build_check_node)
    g.add_node("fail", fail_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "validate_manifest")
    g.add_conditional_edges(
        "validate_manifest",
        manifest_retry,
        {"plan": "plan", "generate": "generate"},
    )
    g.add_edge("generate", "verify")
    g.add_conditional_edges(
        "verify",
        should_retry,
        {"generate": "generate", "write": "write", "fail": "fail"},
    )
    g.add_edge("write", "build_check")
    g.add_edge("build_check", END)

    return g.compile()
