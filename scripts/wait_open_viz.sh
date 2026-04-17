#!/usr/bin/env bash
# assumes repo root cwd or invoked via: bash scripts/wait_open_viz.sh
set -euo pipefail

cd "$(dirname "$0")/.."

OPEN_BROWSER="${OPEN_BROWSER:-1}"
compose=(docker compose -f docker-compose.yml -f docker-compose.st.yml)

url=""
for _ in $(seq 1 60); do
  hp=$("${compose[@]}" port visualizer 8502 2>/dev/null | sed 's/^.*://' || true)
  if [[ -n "${hp}" ]] && curl -sf "http://127.0.0.1:${hp}/_stcore/health" >/dev/null 2>&1; then
    url="http://localhost:${hp}/"
    break
  fi
  sleep 0.5
done

if [[ -z "${url}" ]]; then
  echo "streamlit: timed out waiting for health (see: ${compose[*]} logs visualizer)" >&2
  exit 1
fi

echo "streamlit: ${url}"
if [[ "${OPEN_BROWSER}" == "1" ]]; then
  (command -v open >/dev/null 2>&1 && open "${url}") \
    || (command -v xdg-open >/dev/null 2>&1 && xdg-open "${url}") \
    || true
fi
