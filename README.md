# MAPS experiments

This repository is the frozen reproducibility artifact for MAPS paper results. Each
Workload owns one ONNX graph, and every bash file is one complete Experiment. Support
tooling is Python so bash remains reserved for publication units.

## Workloads

- [MobileViT](workloads/mobilevit/README.md): FP16 nodes 170 through 179. The
  `maps-pipeline.sh` Experiment proves pipelined MAPS execution on a 4x4 MAGIA-v3 GVSoC
  Mesh with 32 Execution Tokens and two Token Slots.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
```

The Experiment discovers sibling `MAPS` and `magia-sdk-v3` checkouts. `MAPS_ROOT` and
`MAGIA_SDK_ROOT` override those locations. Both checkouts, the MAPS compiler tools, the
MAGIA-v3 GVSoC target, CMake, the RISC-V compiler, and a complete Spatz LLVM toolchain are
validated before a build starts. `LLVM_INSTALL_DIR` selects a nonstandard Spatz LLVM
installation; otherwise the SDK-local installation or the system Clang installation is
used.
