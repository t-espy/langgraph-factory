import os
import json
import time
import shutil
import subprocess
import hashlib
import re
from datetime import datetime
from typing import TypedDict, Dict, Optional

import requests
from langgraph.graph import StateGraph, START, END

# ---- CONFIG ----
DMR_BASE = os.environ.get("DMR_BASE_URL", "http://localhost:12434/engines/v1")
# DMR ignores the key, but some clients require it. We'll send one anyway.
DMR_API_KEY = os.environ.get("DMR_API_KEY", "local-dummy")

FOREMAN_MODEL = "docker.io/ai/deepseek-r1-distill-llama:8B-Q4_K_M"
CODER_MODEL = "docker.io/ai/qwen3-coder-next:latest"

PROJECT_ROOT = os.path.expanduser("~/lg_factory_out")
PROJECT_NAME = f"crud-products-{int(time.time())}"
PROJECT_DIR = os.path.join(PROJECT_ROOT, PROJECT_NAME)

MAX_GENERATE_ATTEMPTS = 2   # full regen attempts
MAX_FIX_ATTEMPTS = 4        # patch attempts after build failures


def log_step(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [progress] {message}")


def log_detail(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [detail] {message}")


class FactoryState(TypedDict, total=False):
    spec: str
    architecture_contract: dict
    acceptance: list[str]
    files: Dict[str, str]
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


def _normalize_path(path: str, project_dir: str | None = None) -> str:
        path = path.strip().strip("'\"")
        if project_dir and path.startswith(project_dir):
                path = path[len(project_dir) :]
        path = path.replace("\\", "/")
        if path.startswith("./"):
                path = path[2:]
        return path.lstrip("/")


def _extract_referenced_paths(build_log: str, project_dir: str | None) -> list[str]:
        if not build_log:
                return []
        pattern = re.compile(r"([A-Za-z0-9_./\\-]+\.(?:ts|tsx|js|jsx|json|mjs|cjs|cts|mts|css|scss|mdx))")
        matches = pattern.findall(build_log)
        normalized = []
        for raw in matches:
                path = _normalize_path(raw, project_dir)
                if path and path not in normalized:
                        normalized.append(path)
        return normalized


def _log_build_failure(build_log: str, build_attempt: int, patched_files: list[str]) -> None:
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
        else:
                log_detail("Patched files: (none)")


def dmr_chat_json(model: str, system: str, user: str, max_tokens: int = 4000, temperature: float = 0.2) -> dict:
    """
    Calls Docker Model Runner OpenAI-compatible /chat/completions with JSON mode enabled.
    DMR supports response_format JSON mode. :contentReference[oaicite:2]{index=2}
    """
    url = f"{DMR_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DMR_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=1800)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    try:
        return _extract_json(content)
    except (ValueError, json.JSONDecodeError) as e:
        # Hard fail: JSON mode should prevent this, but if it happens we want the raw output.
        raise RuntimeError(f"Model returned non-JSON content:\n{content}") from e


def _extract_json(text: str) -> dict:
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


def warmup_node(state: FactoryState) -> dict:
    # Warm Qwen so you don't pay the ~60s cold start inside the generation step.
    # (You observed ~60s cold, ~1s warm.)
    log_step("Warmup model")
    try:
        _ = dmr_chat_json(
            model=CODER_MODEL,
            system="Return JSON only.",
            user='{"ok":"warmup"}',
            max_tokens=32,
            temperature=0.0,
        )
    except Exception:
        # Warmup is opportunistic; don’t fail the pipeline on it.
        pass
    return {}


def foreman_policy_node(state: FactoryState) -> dict:
    log_step("Generate policy")
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
            "Goal: pnpm build must pass."
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
                    "primitives": ["button", "input", "label", "card", "table", "textarea", "select"]
                },
                "entity": {
                    "name": "Product",
                    "fields": [
                        {"name": "id", "type": "string"},
                        {"name": "name", "type": "string"},
                        {"name": "price", "type": "number"},
                        {"name": "createdAt", "type": "string"}
                    ]
                },
                "routes": ["/products", "/products/new", "/products/[id]"],
                "server_client_rules": {
                    "prefer_server_components": "boolean",
                    "use_client_only_for_forms": "boolean"
                },
                "acceptance": ["pnpm build"]
            }
        }
    })
    out = dmr_chat_json(
        model=FOREMAN_MODEL,
        system=system,
        user=user,
        max_tokens=1200,
        temperature=0.4,
    )
    contract = out.get("architecture_contract", {})
    acceptance = contract.get("acceptance", []) if isinstance(contract, dict) else []
    return {"architecture_contract": contract, "acceptance": acceptance}


def generate_project_node(state: FactoryState) -> dict:
    log_step("Generate full project")
    gen_attempt = state.get("generate_attempt", 0) + 1
    log_detail(f"Generate attempt {gen_attempt} started")

    system = (
        "You are a meticulous senior engineer generating a complete runnable project. "
        "Return STRICT JSON only. "
        'Schema: {"files": {"relative/path": "full file contents", ...}, "notes": "string"} '
        "Do not omit required config files; ensure pnpm build will succeed. "
        "Obey the architecture_contract strictly. "
        "If ui_strategy.mode is no_ui_imports, do NOT import @/components/ui/* and use plain HTML + minimal CSS (or Tailwind only if included). "
        "If ui_strategy.mode is local_primitives, create the required primitive files and import them. "
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
            "Avoid external DB for MVP."
        ],
        "output_requirements": [
            "Return JSON only with a files map.",
            "Every value must be full file contents (no placeholders).",
            "Do not reference files that you did not include.",
            "Honor ui_strategy mode in architecture_contract."
        ],
        "attempt": gen_attempt
    })

    log_detail("Calling model (this may take a while)...")
    start = time.monotonic()
    out = dmr_chat_json(
        model=CODER_MODEL,
        system=system,
        user=user,
        max_tokens=6000,
        temperature=0.2,
    )
    elapsed = time.monotonic() - start
    log_detail(f"Model response received in {elapsed:.1f}s")

    files = out.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"Generate step returned no files. Output keys: {list(out.keys())}")

    log_detail(f"Generated {len(files)} files")

    return {"files": files, "generate_attempt": gen_attempt, "fix_attempt": 0}


def write_files_node(state: FactoryState) -> dict:
    log_step("Write files to disk")
    project_dir = state.get("project_dir") or PROJECT_DIR
    os.makedirs(project_dir, exist_ok=True)

    files = state.get("files", {})
    if not files:
        raise RuntimeError("No files to write.")

    # Write files
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


def run_cmd(cmd, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def install_deps_node(state: FactoryState) -> dict:
    log_step("Install dependencies")
    project_dir = _require_project_dir(state)
    package_json_hash = state.get("package_json_hash")
    last_installed_hash = state.get("last_installed_package_json_hash")

    node_modules_path = os.path.join(project_dir, "node_modules")
    node_modules_exists = os.path.isdir(node_modules_path)
    package_json_changed = bool(package_json_hash) and package_json_hash != last_installed_hash

    if node_modules_exists and not package_json_changed:
        log_detail("deps already installed, skipping")
        return {}

    # Prefer pnpm if present
    if shutil.which("pnpm") is None:
        raise RuntimeError("pnpm not found in PATH. Install pnpm, then rerun.")

    p = run_cmd(["pnpm", "install"], cwd=project_dir)
    if p.returncode != 0:
        return {"last_build_ok": False, "last_build_log": f"pnpm install failed:\n{p.stdout}"}
    return {"last_installed_package_json_hash": package_json_hash or last_installed_hash}


def build_node(state: FactoryState) -> dict:
    log_step("Run pnpm build")
    project_dir = _require_project_dir(state)
    build_attempt = state.get("build_attempt", 0) + 1

    if shutil.which("pnpm") is None:
        raise RuntimeError("pnpm not found in PATH. Install pnpm, then rerun.")

    p = run_cmd(["pnpm", "build"], cwd=project_dir)
    ok = (p.returncode == 0)
    if not ok:
        _log_build_failure(p.stdout or "", build_attempt, state.get("last_patched_files", []))
    return {"last_build_ok": ok, "last_build_log": p.stdout, "build_attempt": build_attempt}


def fix_from_build_log_node(state: FactoryState) -> dict:
    log_step("Apply fix from build log")
    fix_attempt = state.get("fix_attempt", 0) + 1
    log_detail(f"Fix attempt {fix_attempt}")

    system = (
        "You are a senior engineer fixing a broken Next.js project. "
        "Return STRICT JSON only. "
        "Do NOT refactor the project structure. "
        "Do NOT rewrite the entire project. "
        "Do NOT introduce new dependencies unless required by the build error. "
        "Prefer minimal edits to the smallest number of files. "
        'Return JSON with shape: {"patches": {"path": "full new file content"}, "explanation": "..."}'
    )

    files = state.get("files", {})
    file_index = {k: len(v.encode("utf-8")) for k, v in files.items()}
    project_dir = state.get("project_dir")
    build_log = state.get("last_build_log", "")

    referenced_paths = _extract_referenced_paths(build_log, project_dir)
    snapshots: Dict[str, str] = {}

    if "package.json" in files:
        snapshots["package.json"] = files["package.json"]
    if "tsconfig.json" in files:
        snapshots["tsconfig.json"] = files["tsconfig.json"]
    for path in list(files.keys()):
        if path.startswith("next.config."):
            snapshots[path] = files[path]

    referenced_added = 0
    for path in referenced_paths:
        if path in files and path not in snapshots:
            snapshots[path] = files[path]
            referenced_added += 1
            if referenced_added >= 5:
                break

    user = json.dumps({
        "architecture_contract": state.get("architecture_contract", {}),
        "build_log": build_log[-12000:],  # cap log
        "file_index_bytes": file_index,
        "file_snapshots": snapshots,
        "instructions": [
            "Propose patches as full file contents.",
            "If adding new files is required, include them in patches.",
            "Keep changes minimal and consistent with Next.js App Router + TS.",
            "Do not introduce new dependencies unless needed.",
            "Honor ui_strategy from architecture_contract.",
            "If missing @/components/ui/* imports are present and ui_strategy allows local_primitives, create the missing component files."
        ],
        "attempt": fix_attempt
    })

    out = dmr_chat_json(
        model=CODER_MODEL,
        system=system,
        user=user,
        max_tokens=4000,
        temperature=0.2,
    )

    patches = out.get("patches")
    if not isinstance(patches, dict) or not patches:
        raise RuntimeError(f"Fix step produced no patches. Keys: {list(out.keys())}")

    patched_files = sorted(patches.keys())
    log_detail(f"Patched files: {', '.join(patched_files)}")

    # Apply patches to in-memory files map
    merged = dict(files)
    merged.update(patches)
    return {"files": merged, "fix_attempt": fix_attempt, "last_patched_files": patched_files}


def decide_next(state: FactoryState) -> str:
    if state.get("last_build_ok"):
        return "done"

    # If pnpm install failed, try a full regenerate once (often missing package.json etc.)
    log = state.get("last_build_log", "")
    if "pnpm install failed" in log:
        if state.get("generate_attempt", 0) < MAX_GENERATE_ATTEMPTS:
            return "regenerate"
        return "fail"

    # Build failed: attempt fixes
    if state.get("fix_attempt", 0) < MAX_FIX_ATTEMPTS:
        return "fix"

    # As a fallback, try one regenerate if we haven't already
    if state.get("generate_attempt", 0) < MAX_GENERATE_ATTEMPTS:
        return "regenerate"

    return "fail"


def done_node(state: FactoryState) -> dict:
    return {}


def fail_node(state: FactoryState) -> dict:
    raise RuntimeError(
        "Failed to reach a successful pnpm build within retry limits.\n\n"
        f"Project dir: {state.get('project_dir')}\n\n"
        f"Last build log:\n{state.get('last_build_log','')}"
    )


def build_graph():
    g = StateGraph(FactoryState)

    g.add_node("warmup", warmup_node)
    g.add_node("policy", foreman_policy_node)
    g.add_node("generate", generate_project_node)
    g.add_node("write", write_files_node)
    g.add_node("install", install_deps_node)
    g.add_node("build", build_node)
    g.add_node("fix", fix_from_build_log_node)
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


if __name__ == "__main__":
    spec = """
Build a minimal CRUD web app using Next.js App Router (TypeScript) and shadcn/ui styling patterns.
Entity: Products (id, name, price, createdAt).
Pages:
- /products (list products; link to create)
- /products/new (create form)
- /products/[id] (details)
Use in-memory storage (no DB) for MVP.
Must build with pnpm build.
""".strip()

    os.makedirs(PROJECT_ROOT, exist_ok=True)

    graph = build_graph()
    result = graph.invoke({"spec": spec, "project_dir": PROJECT_DIR})

    print("\n✅ Build succeeded")
    print(f"Project: {result.get('project_dir')}")
    print("Next:")
    print(f"  cd {result.get('project_dir')}")
    print("  pnpm dev")