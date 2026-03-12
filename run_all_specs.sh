#!/usr/bin/env bash
# Run all example specs and collect results.
#
# Usage:
#   ./run_all_specs.sh                  # run all examples
#   ./run_all_specs.sh blog_markdown    # run only matching example(s)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/examples"
RUNS_DIR="${FACTORY_RUNS_DIR:-$SCRIPT_DIR/runs}"

passed=0
failed=0
skipped=0
results=()

for spec in "$EXAMPLES_DIR"/*.py; do
    name="$(basename "$spec" .py)"

    # If a filter was given, only run matching specs
    if [[ $# -gt 0 ]] && [[ "$name" != *"$1"* ]]; then
        skipped=$((skipped + 1))
        results+=("SKIP  $name")
        continue
    fi

    echo ""
    echo "================================================================"
    echo "  SPEC: $name"
    echo "================================================================"

    if python3 "$spec"; then
        passed=$((passed + 1))
        results+=("PASS  $name")
    else
        failed=$((failed + 1))
        results+=("FAIL  $name")
    fi
done

# Summary
echo ""
echo "================================================================"
echo "  ALL SPECS COMPLETE"
echo "================================================================"
for r in "${results[@]}"; do
    echo "  $r"
done
echo ""
echo "  Passed: $passed  Failed: $failed  Skipped: $skipped"
echo "================================================================"

exit $failed
