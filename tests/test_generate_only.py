"""Targeted test: run just warmup + policy + generate, check output for common issues."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph_factory.llm import dmr_chat_json
from langgraph_factory.config import CODER_MODEL, FOREMAN_MODEL

SPEC = """\
A Next.js App Router CRUD app for managing products.
Entities: Product (id, name, description, price, category, created_at).
Pages: /products list, /products/new create, /products/[id] view, /products/[id]/edit edit.
API: GET/POST /api/products, GET/PUT/DELETE /api/products/[id].
Use in-memory storage.
"""


def main():
    # Step 1: Policy
    print("=" * 60)
    print("STEP 1: Policy (foreman model)")
    print("=" * 60)
    policy_system = "You are a senior software architect. Return STRICT JSON only."
    policy_user = json.dumps({
        "task": "Produce an architecture_contract for a Next.js App Router CRUD app.",
        "app_spec": SPEC,
        "requirements": [
            "Do NOT output a file list. Output only the architecture contract.",
            "Prefer Next.js App Router + TypeScript.",
            "Package manager must be pnpm.",
            "In-memory storage acceptable for MVP.",
            "Goal: pnpm build must pass.",
        ],
        "output_schema": {
            "architecture_contract": {
                "project_layout": "app_router",
                "typescript": True,
                "package_manager": "pnpm",
                "styling": "tailwind",
                "acceptance": ["pnpm build"],
            },
        },
    })
    contract = dmr_chat_json(
        model=FOREMAN_MODEL, system=policy_system, user=policy_user,
        max_tokens=2000, temperature=0.4,
    )
    print(f"\nPolicy output keys: {list(contract.keys())}")
    print(json.dumps(contract, indent=2)[:1000])

    # Step 2: Generate
    print("\n" + "=" * 60)
    print("STEP 2: Generate (coder model)")
    print("=" * 60)
    arch_contract = contract.get("architecture_contract", contract)

    gen_system = (
        "You are a meticulous senior engineer generating a complete runnable project. "
        "Return STRICT JSON only. "
        '{"files": {"relative/path": "full file contents", ...}, "notes": "string"} '
        "Do not omit required config files; ensure pnpm build will succeed. "
        "Obey the architecture_contract strictly. "
        "Create proper reusable UI component files (e.g. Button, Input, Table) under @/components/ui/. "
        "Do not depend on shadcn CLI — write the component code directly. "
        "IMPORTANT: Use next.config.mjs (NOT next.config.ts) — Next.js 14 does not support TypeScript config files."
    )
    gen_user = json.dumps({
        "app_spec": SPEC,
        "architecture_contract": arch_contract,
        "constraints": [
            "Project must be Next.js App Router + TypeScript.",
            "Provide all files required for pnpm install and pnpm build.",
            "Keep it minimal but complete.",
            "In-memory store is fine; prefer server components + route handlers.",
            "Avoid external DB for MVP.",
            "Use next.config.mjs (NOT next.config.ts) — Next.js 14 does not support .ts config.",
        ],
        "output_requirements": [
            "Return JSON only with a files map.",
            "Every value must be full file contents (no placeholders).",
            "Do not reference files that you did not include.",
            "Honor ui_strategy mode in architecture_contract.",
        ],
        "attempt": 1,
    })

    out = dmr_chat_json(
        model=CODER_MODEL, system=gen_system, user=gen_user,
        max_tokens=16000, temperature=0.2,
    )

    # Step 3: Analyze output
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    files = out.get("files", {})
    print(f"Files generated: {len(files)}")
    for path in sorted(files.keys()):
        size = len(files[path])
        flag = ""
        if "next.config.ts" in path:
            flag = "  *** BAD: should be .mjs not .ts ***"
        print(f"  {path} ({size:,} chars){flag}")

    # Check for known issues
    issues = []
    config_files = [p for p in files if p.startswith("next.config")]
    if not config_files:
        issues.append("MISSING: no next.config file at all")
    for cf in config_files:
        if cf.endswith(".ts"):
            issues.append(f"BAD CONFIG: {cf} — Next.js 14 won't accept .ts config")
        elif cf.endswith(".mjs") or cf.endswith(".js"):
            print(f"\n  next.config OK: {cf}")

    if "package.json" not in files:
        issues.append("MISSING: package.json")
    if "tsconfig.json" not in files:
        issues.append("MISSING: tsconfig.json")

    has_layout = any("layout" in p for p in files)
    if not has_layout:
        issues.append("MISSING: no layout file found")

    if issues:
        print(f"\nISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo issues found — output looks good!")

    return len(issues) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
