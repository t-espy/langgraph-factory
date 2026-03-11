"""Tests for langgraph_factory.utils — pure utility functions."""

import pytest

from langgraph_factory.utils import (
    extract_json,
    extract_referenced_paths,
    normalize_path,
    parse_fenced_files,
)


# ---------------------------------------------------------------------------
# parse_fenced_files
# ---------------------------------------------------------------------------


class TestParseFencedFiles:
    def test_single_file(self):
        text = "===FILE: src/app/page.tsx===\nexport default function Page() {}\n===END FILE==="
        result = parse_fenced_files(text)
        assert result == {"src/app/page.tsx": "export default function Page() {}"}

    def test_multiple_files(self):
        text = (
            "===FILE: src/app/page.tsx===\npage content\n===END FILE===\n"
            "===FILE: src/lib/data.ts===\nlib content\n===END FILE==="
        )
        result = parse_fenced_files(text)
        assert len(result) == 2
        assert result["src/app/page.tsx"] == "page content"
        assert result["src/lib/data.ts"] == "lib content"

    def test_strips_markdown_tsx_fence(self):
        text = "===FILE: src/app/page.tsx===\n```tsx\nconst x = 1;\n```\n===END FILE==="
        result = parse_fenced_files(text)
        assert result["src/app/page.tsx"] == "const x = 1;"

    def test_strips_markdown_typescript_fence(self):
        text = "===FILE: src/lib/data.ts===\n```typescript\nconst x = 1;\n```\n===END FILE==="
        result = parse_fenced_files(text)
        assert result["src/lib/data.ts"] == "const x = 1;"

    def test_strips_markdown_bare_fence(self):
        text = "===FILE: src/app/globals.css===\n```\nbody { margin: 0; }\n```\n===END FILE==="
        result = parse_fenced_files(text)
        assert result["src/app/globals.css"] == "body { margin: 0; }"

    def test_strips_nested_delimiters(self):
        text = (
            "===FILE: src/app/page.tsx===\n"
            "===FILE: src/app/page.tsx===\n"
            "real content\n"
            "===END FILE==="
        )
        result = parse_fenced_files(text)
        assert "real content" in result["src/app/page.tsx"]

    def test_whitespace_in_path(self):
        text = "===FILE:  src/app/page.tsx ===\ncontent\n===END FILE==="
        result = parse_fenced_files(text)
        assert "src/app/page.tsx" in result

    def test_empty_input(self):
        assert parse_fenced_files("") == {}

    def test_no_fenced_blocks(self):
        assert parse_fenced_files("just some random text") == {}

    def test_preserves_internal_triple_backticks(self):
        """Triple backticks in the middle of content should be preserved."""
        text = (
            "===FILE: README.md===\n"
            "# Hello\n"
            "Some code:\n"
            "```js\n"
            "console.log('hi');\n"
            "```\n"
            "More text\n"
            "===END FILE==="
        )
        result = parse_fenced_files(text)
        # The opening ```js at the start gets stripped, but internal ones won't
        # since the regex is anchored to ^ and $
        assert "More text" in result["README.md"]


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_clean_json(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        result = extract_json('Here is the output:\n{"key": "value"}\nDone.')
        assert result == {"key": "value"}

    def test_json_with_thinking_block(self):
        result = extract_json('<think>reasoning here</think>\n{"key": "value"}')
        assert result == {"key": "value"}

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = extract_json(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="Empty response"):
            extract_json("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="Empty response"):
            extract_json("   \n  ")

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object"):
            extract_json("just some text without braces")

    def test_invalid_json_raises(self):
        with pytest.raises((ValueError, Exception)):
            extract_json("{invalid json content}")


# ---------------------------------------------------------------------------
# normalize_path
# ---------------------------------------------------------------------------


class TestNormalizePath:
    def test_strips_leading_dot_slash(self):
        assert normalize_path("./src/app/page.tsx") == "src/app/page.tsx"

    def test_strips_leading_slash(self):
        assert normalize_path("/src/app/page.tsx") == "src/app/page.tsx"

    def test_strips_quotes(self):
        assert normalize_path('"src/app/page.tsx"') == "src/app/page.tsx"
        assert normalize_path("'src/app/page.tsx'") == "src/app/page.tsx"

    def test_strips_project_dir_prefix(self):
        result = normalize_path(
            "/home/user/project/src/app/page.tsx",
            project_dir="/home/user/project/",
        )
        assert result == "src/app/page.tsx"

    def test_normalizes_backslashes(self):
        assert normalize_path("src\\app\\page.tsx") == "src/app/page.tsx"

    def test_clean_path_unchanged(self):
        assert normalize_path("src/app/page.tsx") == "src/app/page.tsx"


# ---------------------------------------------------------------------------
# extract_referenced_paths
# ---------------------------------------------------------------------------


class TestExtractReferencedPaths:
    def test_extracts_ts_paths(self):
        log = "./src/lib/posts.ts:14:5  Error: prefer-const"
        result = extract_referenced_paths(log)
        assert "src/lib/posts.ts" in result

    def test_extracts_tsx_paths(self):
        log = "./src/app/page.tsx:1:1  Error: missing export"
        result = extract_referenced_paths(log)
        assert "src/app/page.tsx" in result

    def test_deduplicates(self):
        log = (
            "./src/lib/data.ts:10 Error\n"
            "./src/lib/data.ts:20 Error\n"
        )
        result = extract_referenced_paths(log)
        assert result.count("src/lib/data.ts") == 1

    def test_empty_log(self):
        assert extract_referenced_paths("") == []

    def test_no_paths(self):
        assert extract_referenced_paths("Build succeeded!") == []

    def test_strips_project_dir(self):
        log = "/home/user/project/src/app/page.tsx:1:1 Error"
        result = extract_referenced_paths(log, project_dir="/home/user/project/")
        assert "src/app/page.tsx" in result
