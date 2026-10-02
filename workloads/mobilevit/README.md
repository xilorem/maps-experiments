# Full MobileViT workload

Compare pipelined MAPS execution with sequential full-mesh execution of the complete
MobileViT-v2 050 network at its native 256×256 input. The supplied ONNX model has 227
nodes; the full-mesh reference executes 205 layers, aliasing Reshape, Split, and Flatten
as views. Its layer dispatch and deterministic FP16 weight blob come from the SDK's
`onnx_mobilevit_v2` test, using the identical ONNX model.

From the repository root:

```bash
source ../maps-env.sh  # local toolchain environment, when available
./workloads/mobilevit/run-experiment.sh --tokens 2 --token-slots 2
```

The default mesh is **32×32**. With the current MAPS planner, this full graph forms 216
stages and needs 408 tiles just for the minimum L1-feasible allocation with two Token
Slots. The slice's 4×4 and 8×8 meshes cannot hold its stages; 16×16 fails the minimum
L1 allocation. `MESH_SIZES="32"` overrides the mesh list; unsupported configurations
stop with the planner's diagnostic. Feasibility can change with Token Slot count.

Execution Tokens (`--tokens`, 1–16) and Token Slots (`--token-slots`, positive integer)
are independent. Both strategies consume exactly the same prefix of sixteen distinct,
deterministic FP16 input images. MAPS chooses its active tiles freely. The reference
executes each layer on every physical tile with a barrier between layers and processes
tokens serially. An experiment-owned CMake target builds the reference alongside the
MAPS application; no SDK source changes are required.

Each run writes `results/<UTC timestamp>/results.csv`, with Completion Cycle,
Completion Interval, active tiles, and available numerical diagnostics per token and
strategy. Build or simulation failures and incomplete timing stop the run. Numerical
mismatches and non-finite values are reported without gating later runs. The reference
keeps printing outside the measured execution window and preserves each token's output
before reusing its activation arena.

The numerical references are **FP32 ONNX Runtime CPU outputs rounded to FP16**, recorded
in `model/manifest.json`. They are approximate diagnostics for the hardware FP16
arithmetic. MAPS reports bitwise checksum differences; full-mesh output comparisons use
`atol=0.002, rtol=0.02` and report non-finite elements separately. Missing references
produce empty validation CSV fields, following the slice's behavior.

The runner also supports `MAPS_TIMINGS=1`, `MAPS_IMISS_TRACE=1` (requires timings),
`COMMUNICATION_WEIGHT`, `MAPS_ROOT`, `MAGIA_SDK_ROOT`, `PYTHON`, and `LLVM_INSTALL_DIR`,
using the slice's shared timing, CSV, and calibration reporters. Detailed instruction
traces can be large on the full network.

Regenerate the frozen artifacts with the repository virtual environment:

```bash
./.venv/bin/python workloads/mobilevit/model/generator.py
./.venv/bin/python workloads/mobilevit/model/freeze.py
```

The generator retains the SDK's FP16 arithmetic replay to create the weight blob and
layer table; `freeze.py` creates the token corpus, ONNX diagnostics, and SHA-256 manifest.
Preparation verifies the model, blob, table, and input hashes to prevent mismatched
artifacts. The generated intermediate `model/layers/` tensors are ignored by Git.
