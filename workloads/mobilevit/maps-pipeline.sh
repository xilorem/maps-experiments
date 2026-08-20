#!/usr/bin/env bash
set -euo pipefail

artifact_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$artifact_root"
export MAPS_EXPERIMENTS_ROOT="$artifact_root"
python="$artifact_root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  python=python3
fi
"$python" -m maps_experiments.experiment
