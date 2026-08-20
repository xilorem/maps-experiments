# MobileViT Workload

This Workload freezes the FP16 MobileViT slice formed by original ONNX nodes 170 through
179, including its real Initializers. Its input and output shape is `[1,128,4,16]`.

## MAPS versus full-mesh comparison

Hypothesis: ordinary MAPS placement on a physical 4x4 MAGIA-v3 Mesh improves sustained
token throughput over full-mesh model parallelism, which synchronizes every layer and
executes tokens serially.

Run the complete Experiment from this directory:

```bash
./compare-4x4.sh
```

The command uses seed `170179` to generate two distinct FP16 Runtime Inputs uniformly from
`[-1,1]` and evaluates every token with ONNX Runtime. It then executes both strategies with
the same inputs: MAPS plans the physical Mesh with two Token Slots and no active-tile
constraint, while the artifact-owned reference uses all 16 tiles for every layer, globally
synchronizes between layers, and completes one token before starting the next.

The first invocation creates an artifact-owned incremental SDK build under `.build/`.
Later invocations reuse the compiled SDK libraries and boot ROM and rebuild only the MAPS
Application when its generated sources change. The external SDK checkout is never edited.

The measured window starts at one common global boundary after boot, initialization,
Runtime Input loading, and Initializer loading. A token completes only when the last tile
writing its graph output has finished. Trace printing and numerical validation happen
after the measured run. Every output must be finite and match its ONNX Runtime reference
with `atol=0.5` and `rtol=0.05`.

A successful run creates a fresh directory under `results/` containing a shared
`measurements.csv`, `metadata.json`, separate build and simulation logs, both Applications,
and simulator work. The CSV is published only after both strategies complete both tokens in
order and validate every output; the MAPS trace must also prove that different tokens
overlap on different tiles.
