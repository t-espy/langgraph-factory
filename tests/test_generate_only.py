"""Targeted test: run policy and optionally generate, check output for common issues.

Usage:
    python tests/test_generate_only.py              # run both policy + generate
    python tests/test_generate_only.py --policy-only  # run just policy
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph_factory.llm import dmr_chat_json, dmr_chat_raw
from langgraph_factory.config import CODER_MODEL, FOREMAN_MODEL
from langgraph_factory.utils import parse_fenced_files

SPEC = """\
A Next.js App Router CRUD app for managing products.
Entities: Product (id, name, description, price, category, created_at).
Pages: /products list, /products/new create, /products/[id] view, /products/[id]/edit edit.
API: GET/POST /api/products, GET/PUT/DELETE /api/products/[id].
Use in-memory storage.
"""


def run_policy():
    print("=" * 60)
    print("STEP 1: Policy (foreman model)")
    print("=" * 60)
    system = (
        "You are a senior software architect. "
        "Return STRICT JSON only."
    )
    user = json.dumps({
        "task": "Produce an architecture_contract for a Next.js App Router CRUD app generator.",
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
    contract, stats = dmr_chat_json(
        model=FOREMAN_MODEL, system=system, user=user,
        max_tokens=3000, temperature=0.4,
        label="policy",
    )
    print(f"\nPolicy output keys: {list(contract.keys())}")
    print(json.dumps(contract, indent=2)[:2000])
    return contract


def run_generate(contract):
    print("\n" + "=" * 60)
    print("STEP 2: Generate (coder model)")
    print("=" * 60)
    arch_contract = contract.get("architecture_contract", contract)

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
        "app_spec": SPEC,
        "architecture_contract": arch_contract,
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
        "attempt": 1,
    })

    raw, stats = dmr_chat_raw(
        model=CODER_MODEL, system=system, user=user,
        max_tokens=6000, temperature=0.2,
        label="generate",
    )

    files = parse_fenced_files(raw)

    # Analyze output
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-only", action="store_true",
                        help="Run only the policy step")
    args = parser.parse_args()

    contract = run_policy()

    if args.policy_only:
        return True

    return run_generate(contract)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
