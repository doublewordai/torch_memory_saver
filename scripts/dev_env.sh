#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Avoid leaking the stale user-site torch_memory_saver install into this checkout.
export PYTHONNOUSERSITE=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

. "${repo_root}/.venv/bin/activate"

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec "${SHELL:-/bin/bash}" -i
