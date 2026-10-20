#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON:-$root/.venv/bin/python}" -u "$root/workloads/resnet/experiment.py" "$@"
