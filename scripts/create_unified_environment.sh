#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${1:-${ROOT}/.venv-conda}"
CONDA_BIN="${CONDA_BIN:-conda}"

"${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.9.12 pip
"${ENV_PREFIX}/bin/python" -m pip install --upgrade pip
"${ENV_PREFIX}/bin/python" -m pip install -r "${ROOT}/requirements-unified.txt"
"${ENV_PREFIX}/bin/python" -m pip check

printf 'Unified environment created at %s\n' "${ENV_PREFIX}"
printf 'Use: DOTSAFENET_PYTHON=%s/bin/python scripts/run_unified_profile.sh ...\n' "${ENV_PREFIX}"
