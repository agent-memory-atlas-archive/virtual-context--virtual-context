#!/usr/bin/env bash
# Smoke-test a recommended preset from a clean install.
#
# Builds a fresh virtual environment, installs this checkout with the extras
# the README tells users to install, writes the preset with
# `virtual-context init $PRESET`, validates it, and drives the proxy with
# scripts/smoke_recommended.py. Tagging and summaries are real calls to the
# model the preset names: PRESET=recommended (default) needs
# OPENROUTER_API_KEY; PRESET=recommended-local needs that model on a local
# OpenAI-compatible server. The model provider is a local stub.
#
# Usage: [PRESET=recommended-local] scripts/smoke_recommended.sh [workdir]
set -euo pipefail

PRESET="${PRESET:-recommended}"
if [ "$PRESET" = "recommended" ]; then
  : "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set}"
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$(mktemp -d -t vc-smoke)}"
PYTHON="${PYTHON:-python3.11}"

echo "workdir: $WORK"
"$PYTHON" -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --quiet --upgrade pip
"$WORK/venv/bin/python" -m pip install --quiet -e "$REPO[proxy,embeddings]"

cd "$WORK"
"$WORK/venv/bin/virtual-context" init "$PRESET" --force
"$WORK/venv/bin/virtual-context" -c "$WORK/virtual-context.yaml" config validate

"$WORK/venv/bin/python" "$REPO/scripts/smoke_recommended.py" "$WORK" "$WORK/venv/bin/virtual-context"
