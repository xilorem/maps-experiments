# MAPS experiments

This repository contains thin, independently runnable experiment drivers for the MAPS paper
results.

## Workloads

- [MobileViT](workloads/mobilevit/README.md): compare pipelined MAPS execution with the
  SDK's sequential full-mesh test on 4x4, 8x8, and 16x16 MAGIA-v3 meshes.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
```

The runner discovers sibling `MAPS` and `magia-sdk-v3` checkouts. `MAPS_ROOT`,
`MAGIA_SDK_ROOT`, and `PYTHON` override those defaults.
