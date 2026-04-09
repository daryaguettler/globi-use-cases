#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$root/.env" ]]; then
  exit 0
fi
if [[ -f "$root/.env.example" ]]; then
  cp "$root/.env.example" "$root/.env"
  echo "copied .env.example to .env"
  exit 0
fi
touch "$root/.env"
echo "created empty .env (add .env.example to seed defaults)"
