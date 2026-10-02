# MobileViT Workload

This workload freezes the FP16 MobileViT slice formed by original ONNX nodes 170 through
179. The graph and sixteen distinct deterministic FP16 inputs are stored together under
`model/`; a run consumes the requested prefix of those inputs.

## MAPS versus full-mesh comparison

The hypothesis is that ordinary MAPS placement improves token throughput over full-mesh
model parallelism as the physical MAGIA-v3 mesh scales. The full-mesh SDK test uses every
physical tile for every layer, synchronizes between layers, and executes tokens serially.
MAPS receives the physical mesh and the requested Token Slot count without an active-tile
constraint.

Run the complete matrix with two Execution Tokens and two Token Slots:

```bash
./run-experiment.sh
```

The two controls are independent:

```bash
./run-experiment.sh --tokens 4 --token-slots 2
```

The runner loops over 4x4, 8x8, and 16x16. For each mesh it calls MAPS' `make build`, then
uses the SDK's `make build` and `make run` targets for both the generated MAPS application
and the SDK-owned `onnx_mobilevit_slice` full-mesh test.

Each invocation writes to a fresh timestamped directory under `results/`. `results.csv`
contains one row per strategy, mesh, and token with Completion Cycle, Completion Interval,
active tiles, and any available numerical diagnostics. Build failures, simulator failures,
and incomplete timing results stop the experiment. Numerical mismatches, non-finite counts,
and missing numerical diagnostics never gate later configurations or CSV generation.

The stored input file contains sixteen tokens, so `--tokens` accepts values from 1 through
16.
