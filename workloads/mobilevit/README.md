# MobileViT Workload

This Workload freezes the FP16 MobileViT slice formed by original ONNX nodes 170 through
179, including its real Initializers. Its input and output shape is `[1,128,4,16]`.

## MAPS pipeline proof

Hypothesis: ordinary MAPS placement on a physical 4x4 MAGIA-v3 Mesh pipelines distinct
Execution Tokens across tiles instead of globally serializing them.

Run the complete Experiment from this directory:

```bash
./maps-pipeline.sh
```

The command uses seed `170179` to generate 32 distinct FP16 Runtime Inputs uniformly from
`[-1,1]`, evaluates every token with ONNX Runtime, asks MAPS to plan the physical Mesh with
two Token Slots and no active-tile constraint, builds the generated MAGIA Application, and
runs it in GVSoC.

The first invocation creates an artifact-owned incremental SDK build under `.build/`.
Later invocations reuse the compiled SDK libraries and boot ROM and rebuild only the MAPS
Application when its generated sources change. The external SDK checkout is never edited.

The measured window starts at one common global boundary after boot, initialization,
Runtime Input loading, and Initializer loading. A token completes only when the last tile
writing its graph output has finished. Trace printing and numerical validation happen
after the measured run. Every output must be finite and match its ONNX Runtime reference
with `atol=0.5` and `rtol=0.05`.

A successful run creates a fresh directory under `results/` containing `measurements.csv`,
`metadata.json`, build and simulation logs, the generated Application, and simulator work.
The run is rejected unless all 32 Completion Cycles are ordered, every output validates,
and runtime intervals prove that different tokens overlap on different tiles.
