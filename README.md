# MAPS experiments

This repository contains thin, independently runnable experiment drivers for the MAPS paper
results.

## Workloads

- [MobileViT](workloads/mobilevit/README.md): compare pipelined MAPS execution with the
  sequential full-mesh reference on a 32×32 MAGIA-v3 mesh.
- [MobileViT nodes 170–179](workloads/mobilevit_170-179/README.md): the smaller FP16
  slice experiment and its preserved allocation studies.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
```

The runner discovers sibling `MAPS` and `magia-sdk-v3` checkouts. `MAPS_ROOT`,
`MAGIA_SDK_ROOT`, and `PYTHON` override those defaults.
