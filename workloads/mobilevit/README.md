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

Use `--token-progress` (or `MAPS_TOKEN_PROGRESS=1`) to print
`MAPS_TOKEN_DONE tile=N token=N cycle=N` immediately after each tile processes a token.
These messages do not wait for the final global barrier. Outgoing asynchronous
transfers may still be in flight at that point. Progress printing adds overhead,
so leave it disabled for timing measurements. The SDK build option is
`make build maps_token_progress=1` (CMake: `-DMAPS_TOKEN_PROGRESS=ON`).

The diagnostics below build and simulate like `run-experiment.sh`: the SDK comes
from `MAGIA_SDK_ROOT` (or `--sdk-root` where offered), the toolchains from the
environment (`source ../maps-env.sh`), the SDK is built with `PULP_CORES` (default
0) and GVSoC runs with `GVSOC_PULP_CORES` (default 8) idle PULP cores per tile.
Each one first runs `make gvsoc` for its mesh (incremental; it rebuilds only
after GVSoC source changes) and builds in its own output directory. `make build`
regenerates the SDK's `tile_config.h` for the mesh; the diagnostics restore it
afterwards, but do not build another mesh from the same SDK while a diagnostic
is building.

To isolate the eight first-stage shards (original tiles 416–423) from the current
32×32 allocation on a 4×4 mesh:

```bash
./.venv/bin/python workloads/mobilevit/diagnostics/isolate-first-stage.py \
  --mesh-root workloads/mobilevit/results/<run>/32x32
```

The diagnostic preserves the generated shard shapes, global origins, L1 offsets,
two token slots, actual inputs/weights, and Spatz task bundle. It reports L2-read
and operation boundaries and checks every im2col and matmul result against a host
reference with FP16 rounding after each fused multiply-add. Downstream FIFO sends
are removed, and the read/compute loop reproduces the normal runner's ordering.
Results go into `maps-gvsoc-diagnostic/first-stage-4x4/`, including `maps.log`,
`manifest.json`, and `statistics.json`. This diagnostic assumes this allocation's
eight 2048-column shards; it does not regenerate an allocation for a smaller mesh.

Use `diagnostics/isolate-first-two-stages.py` with the same `--mesh-root` to
include the five sigmoid shards (original tiles 228–230 and 261–262) and all
first-stage-to-second-stage FIFO transfers. This variant uses the actual SDK
asynchronous FIFO runner and prints dispatcher entry/return values. It removes
only sends to later stages, checks received inputs and sigmoid outputs exactly
against FP16 arithmetic replay, and writes `first-two-stages-4x4/` artifacts.
The sigmoid replay includes the SDK's fast approximation and saturated unsigned
conversion; its deviation from mathematical sigmoid is recorded separately in
`manifest.json`. A dispatcher
error stops the diagnostic after recording it; `--timeout` bounds simulation
wall time (default 180 seconds). Neither diagnostic changes an existing run.
The diagnostics bundle the dedicated sigmoid Spatz task and dispatch sigmoid
through its unary adapter. Inputs and outputs must be contiguous FP16 shards;
unsupported layouts return an error rather than silently running scalar code.

The two-stage test also checks each first-stage L2 input immediately before
im2col, which distinguishes input DMA corruption from compute errors. Received
sigmoid inputs are checked as the dispatcher sees them: the runtime reads them in
place from the FIFO slot, which it releases after the token. Use
`--dma-read-trace <test tile>` to record that tile's AXI-to-OBI iDMA read bursts
(issue and completion) in `gvsoc/dma-read.log`; this trace is separate from
`maps.log`. `diagnostics/analyze-dma-read-order.py gvsoc/dma-read.log` reports
bursts that complete ahead of earlier ones.

Run a full existing allocation with persistent stage and kernel monitoring:

```bash
./.venv/bin/python workloads/mobilevit/diagnostics/run-monitored.py \
  --mesh-root workloads/mobilevit/results/<run>/32x32
```

This copies the application and builds separately with `MAPS_TOKEN_PROGRESS=1`
and `MAPS_KERNEL_PROGRESS=1`, without allocation, placement, or instruction tracing.
The detached monitor writes `progress.json` every 15 seconds, `stage-events.jsonl`
for stage/token completion, and `maps.log` for raw operation starts and returns.
Kernel or runtime errors stop only this simulator and produce `kernel-errors.json`
and `failure.md` with the stage, tile, token, operation, and return code.
An active operation without a return remains visible in `progress.json`.
DMA im2col emits `MAPS_KERNEL_STEP` after each kernel plane; its latest step
and absolute cycle are recorded on the active operation.
The current diagnostic uses the existing two-token MobileViT input set.
Use `--stage-limit 10` to retain only the first ten stages at their original
physical tile positions on the same mesh. Transfers to excluded stages are
removed and excluded tiles wait at barriers. Stage input dependencies are
validated before building; final-model output checksums are replaced with
prefix completion markers.
A zero kernel return code is not a numerical comparison against reference outputs.

Run increasing MAPS stage prefixes sequentially on the original 32x32 mesh:

```bash
./.venv/bin/python workloads/mobilevit/diagnostics/run-prefix-series.py \
  --mesh-root workloads/mobilevit/results/<run>/32x32 --stall-minutes 30
```

The detached series starts with stage 0, then 0–1, then 0–2, and stops at the
first build/kernel/simulator failure or suspected stall. Only the MAPS application
is built; the separate full-mesh example is excluded. The final stage of each
prefix copies each output shard for both tokens into dedicated packed L2 buffers
and checks its copy checksum. `MAPS_PREFIX_OUTPUT_DONE` records buffer addresses,
byte counts and checksums; this proves export completion, not reference accuracy.
A prefix passes only when all tile/token processing markers, all output-buffer
markers, and simulator exit 0 are present. `summary.json` tracks the series and
`prefix-NNN/progress.json` tracks each run. Thirty minutes without any log or
progress marker stops that run and writes `stall-report.json`; it is a suspected
stall, not a proof of deadlock. CPU elementwise kernels emit progress every 4096
elements. The series does not restart or stop unrelated simulations.

Regenerate the frozen artifacts with the repository virtual environment:

```bash
./.venv/bin/python workloads/mobilevit/model/generator.py
./.venv/bin/python workloads/mobilevit/model/freeze.py
```

The generator retains the SDK's FP16 arithmetic replay to create the weight blob and
layer table; `freeze.py` creates the token corpus, ONNX diagnostics, and SHA-256 manifest.
Preparation verifies the model, blob, table, and input hashes to prevent mismatched
artifacts. The generated intermediate `model/layers/` tensors are ignored by Git.

The dedicated hardware adapters can be checked against NumPy references for
all distinct reshape, transpose, and depthwise layouts in a frozen allocation:

```sh
./.venv/bin/python workloads/mobilevit/diagnostics/test-hw-adapters.py \
  --mesh-root workloads/mobilevit/results/<run>/32x32 \
  --output workloads/mobilevit/results/<run>/32x32/maps-gvsoc-diagnostic/hw-adapters
```

This focused regression uses a 2×2 simulation and two independent inputs per
layout. It exercises the same operation dispatcher used by the model: DMA for
reshape/transpose and a dedicated Spatz task for depthwise convolution with
FP32 accumulation and FP16 output. `statistics.json` records the result;
`maps.log` contains each numerical check. `--timeout` bounds simulation wall
time (default 3600 seconds; the 151 layouts of the current allocation take
about 19 minutes). Production prefix runs retain 32×32.
