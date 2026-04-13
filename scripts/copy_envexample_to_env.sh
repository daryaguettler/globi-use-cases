#!/usr/bin/env bash
# seed .env from .env.example if missing
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
env="$root/.env"
example="$root/.env.example"

if [[ -f "$env" ]]; then
  exit 0
fi
if [[ -f "$example" ]]; then
  cp "$example" "$env"
  echo "copied .env.example to .env"
else
  touch "$env"
  echo "created empty .env (add .env.example to seed defaults)"
fi
