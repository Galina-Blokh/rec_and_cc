#!/usr/bin/env bash
# Rebuild the build report: collect the record, then render the page into .report/.
# Uses only the 0307 credentials from ~/.env (git_api_key_0307); never prints them.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="${REPORT_OUT:-$root/.report}"
mkdir -p "$out"
if [ -z "${GH_TOKEN:-}" ] && [ -f "$HOME/.env" ]; then
  GH_TOKEN="$(grep -E '^git_api_key_0307=' "$HOME/.env" | head -1 | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//')"
  export GH_TOKEN
fi
[ -f "$out/report-data.json" ] && cp "$out/report-data.json" "$out/report-data.prev.json"
uv run --no-sync python "$root/scripts/report/collect.py" --root "$root" --out "$out/report-data.json"
uv run --no-sync python "$root/scripts/report/render.py" --data "$out/report-data.json" --out "$out/report.html"
uv run --no-sync python "$root/scripts/report/render.py" --data "$out/report-data.json" --out "$out/artifact.html" --fragment
if [ -f "$out/report-data.prev.json" ]; then
  uv run --no-sync python "$root/scripts/report/diff.py" "$out/report-data.prev.json" "$out/report-data.json"
fi
