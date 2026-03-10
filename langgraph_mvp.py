"""Generate a Next.js CRUD scaffold via LLM planning and per-file coding."""

from __future__ import annotations

import json
import os
import subprocess
from typing import TypedDict

from pydantic import SecretStr

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:12434/engines/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "local-dummy")

FOREMAN_MODEL = "docker.io/ai/deepseek-r1-distill-llama:8B-Q4_K_M"
CODER_MODEL = "docker.io/ai/qwen3-coder-next:latest"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_DIR, "factory_out_langgraph")

LINT_MAX_LOOPS = 10
BUILD_MAX_LOOPS = 10


def log_step(message: str) -> None:
    print(f"[progress] {message}")


class Manifest(TypedDict):
    """Manifest produced by the planner."""

    required_files: list[str]
    file_contexts: dict[str, str]
    global_context: str
    acceptance: list[str]
    notes: str


class FactoryState(TypedDict, total=False):
    spec: str
    manifest: Manifest
    required_files: list[str]
    file_contexts: dict[str, str]
    files: dict[str, str]
    missing_files: list[str]
    attempt: int
    max_attempts: int
    notes: str


def _chat(llm: ChatOpenAI, system: str, user: str) -> str:
    """Send a minimal chat request and return content as a string."""
    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    content = resp.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            [item if isinstance(item, str) else json.dumps(item) for item in content]
        )
    return json.dumps(content)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from model output."""
    text = text.strip()
    if not text:
        raise ValueError("Empty response from model")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("No JSON object found in model output")


def _normalize_path(path: str) -> str:
    path = path.strip()
    if path.startswith("./"):
        path = path[2:]
    if path.startswith("/"):
        path = path.lstrip("/")
    return path


def _require_spec(state: FactoryState) -> str:
    spec = state.get("spec")
    if not spec:
        raise RuntimeError("Missing spec in state")
    return spec


def plan_manifest(spec: str) -> Manifest:
    """Call the foreman once to produce a complete manifest."""
    log_step("Plan manifest")
    llm = ChatOpenAI(
        model=FOREMAN_MODEL,
        base_url=BASE_URL,
        api_key=SecretStr(API_KEY),
        temperature=0.3,
    )

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
{spec}
""".strip()

    last_error: Exception | None = None
    for _ in range(3):
        try:
            text = _chat(llm, system, user)
            payload = _extract_json(text)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
    else:
        raise RuntimeError(f"Plan JSON parse failed: {last_error}") from last_error

    required_files = [_normalize_path(p) for p in payload.get("required_files", [])]
    file_contexts = {
        _normalize_path(k): v for k, v in payload.get("file_contexts", {}).items()
    }

    if not required_files:
        raise RuntimeError("Manifest required_files is empty")

    return {
        "required_files": required_files,
        "file_contexts": file_contexts,
        "global_context": payload.get("global_context", ""),
        "acceptance": payload.get("acceptance", ["pnpm install", "pnpm dev"]),
        "notes": payload.get("notes", ""),
    }


def generate_file_content(*, spec: str, manifest: Manifest, path: str) -> str:
    """Generate content for a single file."""
    log_step(f"Generate {path}")
    llm = ChatOpenAI(
        model=CODER_MODEL,
        base_url=BASE_URL,
        api_key=SecretStr(API_KEY),
        temperature=0.2,
    )

    system = (
        "You are a meticulous full-stack engineer. Output STRICT JSON only. "
        "No markdown, no commentary."
    )

    purpose = manifest["file_contexts"].get(path, "")
    user = {
        "spec": spec,
        "global_context": manifest.get("global_context", ""),
        "required_files": manifest.get("required_files", []),
        "file": {"path": path, "purpose": purpose},
        "instructions": [
            "Return STRICT JSON only.",
            "Return {\"path\": <path>, \"content\": <full file contents>}.",
            "Do not include any other keys.",
        ],
    }

    last_error: Exception | None = None
    for _ in range(3):
        try:
            text = _chat(llm, system, json.dumps(user))
            payload = _extract_json(text)
            content = payload.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty file content from model")
            return content
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            continue

    raise RuntimeError(f"File generation failed for {path}: {last_error}") from last_error


def write_file(path: str, content: str) -> None:
    """Write a single file to disk."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = _normalize_path(path)
    target = os.path.join(OUT_DIR, path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)


def verify_files_exist(required_files: list[str]) -> list[str]:
    """Check for missing required files on disk."""
    missing = []
    for rel_path in required_files:
        target = os.path.join(OUT_DIR, _normalize_path(rel_path))
        if not os.path.exists(target):
            missing.append(rel_path)
    return missing


def run_command(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    """Run a command and return (exit_code, output)."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, output.strip()


def apply_fix_from_coder(*, spec: str, manifest: Manifest, error_output: str) -> None:
    """Ask coder to fix errors and write changed files."""
    log_step("Apply fix from lint/build output")
    llm = ChatOpenAI(
        model=CODER_MODEL,
        base_url=BASE_URL,
        api_key=SecretStr(API_KEY),
        temperature=0.2,
    )

    system = (
        "You are a meticulous full-stack engineer. Output STRICT JSON only. "
        "No markdown, no commentary."
    )

    user = {
        "spec": spec,
        "global_context": manifest.get("global_context", ""),
        "required_files": manifest.get("required_files", []),
        "error_output": error_output,
        "instructions": [
            "Return STRICT JSON only.",
            "Return {\"files\": {\"path\": \"content\", ...}} with only changed files.",
            "Do not include any other keys.",
        ],
    }

    text = _chat(llm, system, json.dumps(user))
    payload = _extract_json(text)
    files = payload.get("files", {})
    if not isinstance(files, dict) or not files:
        raise RuntimeError("Coder did not return any fixes")
    for path, content in files.items():
        if isinstance(content, str):
            write_file(path, content)


def save_manifest(manifest: Manifest) -> None:
    """Persist manifest to disk."""
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def plan_node(state: FactoryState) -> dict:
    """Plan manifest and store required files in state."""
    log_step("Plan manifest (graph)")
    app_spec = _require_spec(state)
    manifest = plan_manifest(app_spec)
    return {
        "manifest": manifest,
        "required_files": manifest["required_files"],
        "file_contexts": manifest.get("file_contexts", {}),
        "attempt": 0,
        "missing_files": [],
    }


def validate_manifest_node(state: FactoryState) -> dict:
    """Validate manifest constraints before generation."""
    log_step("Validate manifest")
    required = state.get("required_files", [])
    if not required:
        return {"missing_files": ["__manifest__:required_files_empty"]}

    forbidden = [p for p in required if p.startswith("src/pages") or "/pages/" in p]
    if forbidden:
        return {"missing_files": ["__manifest__:forbidden_pages_routes"]}

    has_app = any(p.startswith("app/") for p in required)
    has_components = any(p.startswith("components/") for p in required)
    has_lib = any(p.startswith("lib/") for p in required)
    has_tests = any("test" in p or "__tests__" in p for p in required)

    missing_categories = []
    if not has_app:
        missing_categories.append("__manifest__:missing_app")
    if not has_components:
        missing_categories.append("__manifest__:missing_components")
    if not has_lib:
        missing_categories.append("__manifest__:missing_lib")
    if not has_tests:
        missing_categories.append("__manifest__:missing_tests")

    if missing_categories:
        return {"missing_files": missing_categories}

    return {"missing_files": []}


def generate_node(state: FactoryState) -> dict:
    """Generate files as JSON mapping from path to content."""
    attempt = state.get("attempt", 0)
    log_step(f"Generate files (attempt {attempt + 1})")
    llm = ChatOpenAI(
        model=CODER_MODEL,
        base_url=BASE_URL,
        api_key=SecretStr(API_KEY),
        temperature=0.2,
    )

    required = state.get("required_files", [])
    missing = state.get("missing_files", [])
    attempt = state.get("attempt", 0)

    system = (
        "You are a meticulous full-stack engineer. Output STRICT JSON only. "
        "No markdown, no commentary. "
        'Return {"files": {"path":"content", ...}} and include EVERY required file.'
    )

    user = {
        "spec": _require_spec(state),
        "required_files": required,
        "missing_files": missing,
        "instructions": [
            "Return STRICT JSON only.",
            "Include full file contents for every path.",
            "If a file is listed as missing_files, prioritize it.",
            "Do not omit package.json / tsconfig / next config if needed.",
        ],
        "attempt": attempt,
    }

    last_error = None
    for _ in range(3):
        try:
            text = _chat(llm, system, json.dumps(user))
            payload = _extract_json(text)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
    else:
        raise RuntimeError(f"Generate JSON parse failed: {last_error}") from last_error
    files = payload.get("files", {})

    # Merge for incremental completion (important!)
    merged = dict(state.get("files", {}))
    merged.update(files)
    return {"files": merged, "attempt": attempt + 1}


def verify_node(state: FactoryState) -> dict:
    """Verify all required files are present in generated output."""
    log_step("Verify generated files")
    required = state.get("required_files", [])
    files = state.get("files", {})
    forbidden = [p for p in files if p.startswith("src/pages") or "/pages/" in p]
    if forbidden:
        raise RuntimeError(f"Generated forbidden pages routes: {forbidden}")
    missing = [p for p in required if p not in files or not str(files[p]).strip()]
    return {"missing_files": missing}


def should_retry(state: FactoryState) -> str:
    """Decide whether to retry generation or proceed."""
    missing = state.get("missing_files", [])
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 6)

    if not missing:
        return "write"
    if attempt >= max_attempts:
        return "fail"
    return "generate"


def manifest_retry(state: FactoryState) -> str:
    """Decide whether to re-plan the manifest or proceed to generation."""
    missing = state.get("missing_files", [])
    return "plan" if missing else "generate"


def write_node(state: FactoryState) -> dict:
    """Write generated files to disk."""
    log_step("Write files to disk")
    os.makedirs(OUT_DIR, exist_ok=True)
    for rel_path, content in state.get("files", {}).items():
        target = os.path.join(OUT_DIR, rel_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
    return {"notes": f"Wrote files to {OUT_DIR}"}


def build_check_node(state: FactoryState) -> dict:  # pylint: disable=unused-argument
    """Run pnpm install/build if pnpm is available."""
    log_step("Run build check")
    # Optional: run a quick build check if pnpm is available
    try:
        subprocess.run(["pnpm", "-v"], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return {"notes": "pnpm not available; skipped build check"}

    try:
        subprocess.run(["pnpm", "install"], check=True, cwd=OUT_DIR)
        subprocess.run(["pnpm", "build"], check=True, cwd=OUT_DIR)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Build check failed") from exc
    return {"notes": "Build check passed"}


def fail_node(state: FactoryState) -> dict:
    """Fail fast when required files cannot be generated."""
    log_step("Fail: exceeded retry limit")
    missing = state.get("missing_files", [])
    raise RuntimeError(f"Failed to generate required files after retries. Missing: {missing}")


def build_graph():
    """Build and compile the LangGraph workflow."""
    g = StateGraph(FactoryState)
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
        {
            "plan": "plan",
            "generate": "generate",
        },
    )
    g.add_edge("generate", "verify")

    g.add_conditional_edges(
        "verify",
        should_retry,
        {
            "generate": "generate",
            "write": "write",
            "fail": "fail",
        },
    )

    g.add_edge("write", "build_check")
    g.add_edge("build_check", END)
    return g.compile()


if __name__ == "__main__":
    APP_SPEC = """
Build a minimal CRUD web app scaffold using:
- Next.js App Router (TypeScript)
- shadcn/ui components
- A single entity: Products (id, name, price, createdAt)
- In-memory storage is fine (no DB) for MVP
- Provide pages:
    - /products (list + create button)
    - /products/new (create form)
    - /products/[id] (view details)
- Use server actions or route handlers (your choice) but keep it simple and documented.
- Include a minimal test setup and at least one test file.

Output requirements:
1) Provide a file tree.
2) For each file, output:
     - Relative path
     - Full file contents
3) Keep it runnable with `pnpm dev`.
""".strip()

    graph = build_graph()
    result = graph.invoke({"spec": APP_SPEC})
    print(result.get("notes", "done"))
    print(f"Missing files at end: {result.get('missing_files', [])}")
    print(f"Output dir: {OUT_DIR}")
