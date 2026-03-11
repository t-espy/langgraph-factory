"""Tests for pure functions in langgraph_factory.factory.

These tests cover deterministic logic that doesn't require LLM calls
or filesystem access.
"""

import json

import pytest

from langgraph_factory.factory import (
    _extract_npm_packages,
    _reconcile_imports,
    _sanitize_generated_files,
    _try_mechanical_fix,
    _validate_manifest,
)


# ---------------------------------------------------------------------------
# _validate_manifest
# ---------------------------------------------------------------------------


class TestValidateManifest:
    def _make_manifest(self, paths: list[str]) -> list[dict]:
        return [{"path": p, "description": "test"} for p in paths]

    def test_valid_manifest_no_scaffold(self):
        manifest = self._make_manifest([
            "package.json", "tsconfig.json", "next.config.mjs",
            "src/app/layout.tsx", "src/app/page.tsx",
        ])
        issues = _validate_manifest(manifest, has_scaffold=False)
        assert issues == []

    def test_missing_package_json(self):
        manifest = self._make_manifest([
            "tsconfig.json", "next.config.mjs",
            "src/app/layout.tsx", "src/app/page.tsx",
        ])
        issues = _validate_manifest(manifest, has_scaffold=False)
        assert any("package.json" in i for i in issues)

    def test_missing_page_file(self):
        manifest = self._make_manifest([
            "package.json", "tsconfig.json", "next.config.mjs",
            "src/app/layout.tsx", "src/lib/data.ts",
        ])
        issues = _validate_manifest(manifest, has_scaffold=False)
        assert any("page" in i.lower() for i in issues)

    def test_next_config_ts_flagged(self):
        manifest = self._make_manifest([
            "package.json", "tsconfig.json", "next.config.ts",
            "src/app/layout.tsx", "src/app/page.tsx",
        ])
        issues = _validate_manifest(manifest, has_scaffold=False)
        assert any("next.config.ts" in i for i in issues)

    def test_scaffold_mode_skips_config_checks(self):
        """With scaffold, config files aren't required."""
        manifest = self._make_manifest([
            "src/app/page.tsx", "src/app/layout.tsx",
        ])
        issues = _validate_manifest(manifest, has_scaffold=True)
        assert issues == []

    def test_scaffold_mode_warns_config_in_manifest(self):
        """With scaffold, config files in manifest trigger a warning."""
        manifest = self._make_manifest([
            "src/app/page.tsx", "package.json",
        ])
        issues = _validate_manifest(manifest, has_scaffold=True)
        assert any("scaffold already provides" in i for i in issues)

    def test_scaffold_mode_still_requires_page(self):
        manifest = self._make_manifest(["src/lib/data.ts"])
        issues = _validate_manifest(manifest, has_scaffold=True)
        assert any("page" in i.lower() for i in issues)


# ---------------------------------------------------------------------------
# _sanitize_generated_files
# ---------------------------------------------------------------------------


class TestSanitizeGeneratedFiles:
    def test_renames_next_config_ts_to_mjs(self):
        files = {
            "next.config.ts": 'import type { NextConfig } from "next";\nconst config: NextConfig = {};\nexport default config;',
            "src/app/page.tsx": "export default function Page() {}",
            "src/app/layout.tsx": "export default function Layout({ children }) { return children; }",
        }
        result = _sanitize_generated_files(files)
        assert "next.config.ts" not in result
        assert "next.config.mjs" in result
        assert "NextConfig" not in result["next.config.mjs"]

    def test_no_rename_if_mjs_exists(self):
        files = {
            "next.config.ts": "const config = {};",
            "next.config.mjs": "const config = {};",
            "src/app/page.tsx": "export default function Page() {}",
            "src/app/layout.tsx": "export default function Layout({ children }) { return children; }",
        }
        result = _sanitize_generated_files(files)
        assert "next.config.ts" in result
        assert "next.config.mjs" in result

    def test_injects_missing_layout(self):
        files = {
            "src/app/page.tsx": "export default function Page() {}",
        }
        result = _sanitize_generated_files(files)
        assert "src/app/layout.tsx" in result
        assert "RootLayout" in result["src/app/layout.tsx"]

    def test_injects_globals_css_with_layout(self):
        files = {
            "src/app/page.tsx": "export default function Page() {}",
        }
        result = _sanitize_generated_files(files)
        assert "src/app/globals.css" in result

    def test_no_layout_injection_if_exists(self):
        files = {
            "src/app/layout.tsx": "custom layout",
            "src/app/page.tsx": "page content",
        }
        result = _sanitize_generated_files(files)
        assert result["src/app/layout.tsx"] == "custom layout"

    def test_detects_app_dir_without_src(self):
        files = {
            "app/page.tsx": "export default function Page() {}",
        }
        result = _sanitize_generated_files(files)
        assert "app/layout.tsx" in result
        assert "app/globals.css" in result

    def test_does_not_mutate_input(self):
        files = {"src/app/page.tsx": "content", "src/app/layout.tsx": "layout"}
        original = dict(files)
        _sanitize_generated_files(files)
        assert files == original


# ---------------------------------------------------------------------------
# _extract_npm_packages
# ---------------------------------------------------------------------------


class TestExtractNpmPackages:
    def test_basic_import(self):
        files = {"src/lib/md.ts": 'import { marked } from "marked";'}
        assert "marked" in _extract_npm_packages(files)

    def test_scoped_package(self):
        files = {"src/app/page.tsx": 'import { Slot } from "@radix-ui/react-slot";'}
        assert "@radix-ui/react-slot" in _extract_npm_packages(files)

    def test_skips_relative_import(self):
        files = {"src/app/page.tsx": 'import { data } from "./data";'}
        assert _extract_npm_packages(files) == set()

    def test_skips_alias_import(self):
        files = {"src/app/page.tsx": 'import { data } from "@/lib/data";'}
        assert _extract_npm_packages(files) == set()

    def test_skips_builtins(self):
        files = {"src/lib/utils.ts": 'import React from "react";\nimport next from "next";'}
        assert _extract_npm_packages(files) == set()

    def test_subpath_import(self):
        files = {"src/lib/utils.ts": 'import merge from "lodash/merge";'}
        assert "lodash" in _extract_npm_packages(files)

    def test_require_syntax(self):
        files = {"src/lib/utils.js": "const uuid = require('uuid');"}
        assert "uuid" in _extract_npm_packages(files)

    def test_skips_non_source_files(self):
        files = {"data.json": '{"import": "fake"}'}
        assert _extract_npm_packages(files) == set()


# ---------------------------------------------------------------------------
# _reconcile_imports
# ---------------------------------------------------------------------------


class TestReconcileImports:
    def _make_files(self, deps: dict, imports: str) -> dict[str, str]:
        pkg = {"name": "test", "dependencies": deps}
        return {
            "package.json": json.dumps(pkg),
            "src/lib/utils.ts": imports,
        }

    def test_adds_missing_package(self):
        files = self._make_files({}, 'import { marked } from "marked";')
        result = _reconcile_imports(files)
        pkg = json.loads(result["package.json"])
        assert "marked" in pkg["dependencies"]

    def test_no_change_when_already_declared(self):
        files = self._make_files(
            {"marked": "^5.0.0"},
            'import { marked } from "marked";',
        )
        result = _reconcile_imports(files)
        pkg = json.loads(result["package.json"])
        assert pkg["dependencies"]["marked"] == "^5.0.0"

    def test_no_package_json_returns_unchanged(self):
        files = {"src/lib/utils.ts": 'import { marked } from "marked";'}
        result = _reconcile_imports(files)
        assert result == files

    def test_invalid_package_json_returns_unchanged(self):
        files = {
            "package.json": "not json",
            "src/lib/utils.ts": 'import { marked } from "marked";',
        }
        result = _reconcile_imports(files)
        assert result == files


# ---------------------------------------------------------------------------
# _try_mechanical_fix
# ---------------------------------------------------------------------------


class TestTryMechanicalFix:
    def test_next_config_ts_rename(self):
        build_log = "Configuring Next.js via 'next.config.ts' is not supported"
        files = {
            "next.config.ts": 'import type { NextConfig } from "next";\nconst config: NextConfig = {};\nexport default config;',
        }
        result = _try_mechanical_fix(build_log, files)
        assert result is not None
        patches, removals = result
        assert "next.config.mjs" in patches
        assert "next.config.ts" in removals
        assert "NextConfig" not in patches["next.config.mjs"]

    def test_missing_npm_module(self):
        build_log = "Module not found: Can't resolve 'uuid'"
        files = {
            "package.json": json.dumps({"name": "test", "dependencies": {}}),
        }
        result = _try_mechanical_fix(build_log, files)
        assert result is not None
        patches, _ = result
        pkg = json.loads(patches["package.json"])
        assert "uuid" in pkg["dependencies"]

    def test_missing_scoped_npm_module(self):
        build_log = "Module not found: Can't resolve '@radix-ui/react-slot'"
        files = {
            "package.json": json.dumps({"name": "test", "dependencies": {}}),
        }
        result = _try_mechanical_fix(build_log, files)
        assert result is not None
        patches, _ = result
        pkg = json.loads(patches["package.json"])
        assert "@radix-ui/react-slot" in pkg["dependencies"]

    def test_missing_type_import(self):
        build_log = "./src/app/page.tsx:5:10 - error TS2304: Cannot find name 'Post'"
        files = {
            "src/app/page.tsx": 'export default function Page() { const p: Post = {}; }',
            "src/lib/types.ts": "export interface Post {\n  id: string;\n  title: string;\n}",
        }
        result = _try_mechanical_fix(build_log, files)
        assert result is not None
        patches, _ = result
        assert "src/app/page.tsx" in patches
        assert "import { Post }" in patches["src/app/page.tsx"]

    def test_no_fix_returns_none(self):
        build_log = "Some unknown error we can't handle"
        files = {"src/app/page.tsx": "content"}
        result = _try_mechanical_fix(build_log, files)
        assert result is None

    def test_unused_catch_variable(self):
        build_log = (
            "./src/app/api/posts/route.ts:8:12  Error: 'error' is defined but never used.  "
            "@typescript-eslint/no-unused-vars"
        )
        files = {
            "src/app/api/posts/route.ts": (
                "export async function GET() {\n"
                "  try {\n"
                "    return Response.json({ ok: true });\n"
                "  } catch (error) {\n"
                "    return Response.json({ error: 'fail' }, { status: 500 });\n"
                "  }\n"
                "}\n"
            ),
        }
        result = _try_mechanical_fix(build_log, files)
        assert result is not None
        patches, _ = result
        assert "src/app/api/posts/route.ts" in patches
        assert "catch {" in patches["src/app/api/posts/route.ts"]
        assert "catch (error)" not in patches["src/app/api/posts/route.ts"]

    def test_unused_underscore_catch_variable(self):
        build_log = (
            "./src/app/api/posts/route.ts:8:12  Error: '_error' is defined but never used.  "
            "@typescript-eslint/no-unused-vars"
        )
        files = {
            "src/app/api/posts/route.ts": (
                "export async function GET() {\n"
                "  try {\n"
                "    return Response.json({ ok: true });\n"
                "  } catch (_error) {\n"
                "    return Response.json({ error: 'fail' }, { status: 500 });\n"
                "  }\n"
                "}\n"
            ),
        }
        result = _try_mechanical_fix(build_log, files)
        assert result is not None
        patches, _ = result
        assert "catch {" in patches["src/app/api/posts/route.ts"]

    def test_unused_import_removal(self):
        build_log = (
            "./src/app/api/posts/route.ts:1:10  Error: 'NextRequest' is defined but never used.  "
            "@typescript-eslint/no-unused-vars"
        )
        files = {
            "src/app/api/posts/route.ts": (
                'import { NextRequest } from "next/server";\n'
                "export async function GET() {\n"
                "  return Response.json({ ok: true });\n"
                "}\n"
            ),
        }
        result = _try_mechanical_fix(build_log, files)
        assert result is not None
        patches, _ = result
        assert "NextRequest" not in patches["src/app/api/posts/route.ts"]

    def test_missing_prop_on_interface(self):
        build_log = (
            "Property 'variant' does not exist on type 'IntrinsicAttributes & ButtonProps'\n"
            "Type '{ variant: \"secondary\"; }' is not assignable"
        )
        files = {
            "src/components/button.tsx": (
                "export interface ButtonProps {\n"
                "  children: React.ReactNode;\n"
                "}\n"
            ),
        }
        result = _try_mechanical_fix(build_log, files)
        assert result is not None
        patches, _ = result
        assert "src/components/button.tsx" in patches
        assert "variant?" in patches["src/components/button.tsx"]
