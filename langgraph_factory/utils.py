"""Shared utilities for langgraph-factory pipelines."""

import json
import re
from datetime import datetime


def log_step(message: str) -> None:
    """Log a high-level pipeline step."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [progress] {message}")


def log_detail(message: str) -> None:
    """Log implementation details within a step."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [detail] {message}")


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that qwen3-coder may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _repair_file_objects(text: str) -> str:
    """Repair a common model error: file values as objects instead of strings.

    The model sometimes writes:
        "src/app/page.tsx": { "use client": false, "import ...code..." }
    instead of:
        "src/app/page.tsx": "\"use client\";\\nimport ...code..."

    This regex finds file-path keys followed by { and attempts to replace
    the malformed object with a concatenated string.
    """
    # Pattern: "path.tsx": { "key": value, "code..." }
    # We detect this by looking for file extensions followed by ": {"
    # then a "use client" or "use server" key with a boolean value
    pattern = re.compile(
        r'"([^"]+\.(?:tsx?|jsx?|mjs|css))"\s*:\s*\{\s*"(use (?:client|server))"\s*:\s*(?:true|false)\s*,\s*"',
    )

    repairs = 0
    while True:
        match = pattern.search(text)
        if not match:
            break

        # Find the opening { after the file path key
        obj_start = text.index("{", match.start() + len(match.group(1)) + 2)

        # Find the matching closing } by counting braces
        depth = 0
        obj_end = None
        for i in range(obj_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    obj_end = i
                    break

        if obj_end is None:
            break

        # Extract the directive and the code string
        directive = match.group(2)
        # The code is the last string value in the object
        inner = text[obj_start + 1 : obj_end]
        # Find the last quoted string value (the actual code)
        last_quote_end = inner.rfind('"')
        if last_quote_end == -1:
            break
        # Walk back to find the start of this string
        search_from = inner.rfind('",', 0, last_quote_end)
        if search_from == -1:
            search_from = inner.rfind('":', 0, last_quote_end)
        if search_from == -1:
            break
        code_start = inner.index('"', search_from + 1)
        code_content = inner[code_start + 1 : last_quote_end]

        # Reconstruct as a proper string value
        file_content = f'\\"{directive}\\";\\n{code_content}'
        replacement = f'"{match.group(1)}": "{file_content}"'
        text = text[:match.start()] + replacement + text[obj_end + 1:]
        repairs += 1

        if repairs > 50:  # safety valve
            break

    if repairs:
        log_detail(f"Repaired {repairs} file value(s) from object to string")
    return text


def extract_json(text: str) -> dict:
    """Extract a JSON object from model output, tolerating surrounding text."""
    text = text.strip()
    if not text:
        raise ValueError("Empty response from model")

    # Strip thinking blocks before parsing
    text = _strip_thinking(text)

    # Try parsing as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

        # Try repairing common model errors
        repaired = _repair_file_objects(snippet)
        if repaired != snippet:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        # Final attempt failed — dump context for debugging
        try:
            json.loads(snippet)
        except json.JSONDecodeError as exc:
            pos = exc.pos or 0
            context_start = max(0, pos - 200)
            context_end = min(len(snippet), pos + 200)
            log_detail(f"JSON parse error at char {pos}: {exc.msg}")
            log_detail(f"Context: ...{snippet[context_start:context_end]!r}...")
            raise

    raise ValueError("No JSON object found in model output")


def parse_fenced_files(text: str) -> dict[str, str]:
    """Parse fenced file output format into a {path: content} dict.

    Expected format:
        ===FILE: relative/path.tsx===
        file contents here
        (no escaping needed)
        ===END FILE===

    This format avoids JSON escaping issues when file contents contain
    quotes, backticks, or other characters that break JSON strings.
    """
    pattern = re.compile(
        r"===FILE:\s*(.+?)\s*===\n(.*?)===END FILE===",
        re.DOTALL,
    )
    files = {}
    for match in pattern.finditer(text):
        path = match.group(1).strip()
        content = match.group(2)
        # Remove trailing newline added before ===END FILE===
        if content.endswith("\n"):
            content = content[:-1]
        files[path] = content

    return files


def normalize_path(path: str, project_dir: str | None = None) -> str:
    """Normalize a file path: strip quotes, leading slashes, project dir prefix."""
    path = path.strip().strip("'\"")
    if project_dir and path.startswith(project_dir):
        path = path[len(project_dir):]
    path = path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


def extract_referenced_paths(build_log: str, project_dir: str | None = None) -> list[str]:
    """Extract file paths referenced in a build log."""
    if not build_log:
        return []
    pattern = re.compile(
        r"([A-Za-z0-9_./\\-]+\.(?:ts|tsx|js|jsx|json|mjs|cjs|cts|mts|css|scss|mdx))"
    )
    matches = pattern.findall(build_log)
    normalized = []
    for raw in matches:
        path = normalize_path(raw, project_dir)
        if path and path not in normalized:
            normalized.append(path)
    return normalized
