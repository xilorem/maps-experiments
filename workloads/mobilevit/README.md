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

FIFO event sleep and live stage/kernel monitoring are **enabled by default**.
The application and Spatz bundle are rebuilt from the SDK, including the new
MAPS depthwise im2col + FP16 GEMM path. Use `--maps-only` to run only the full
MAPS workload and still generate `results.csv`:

```bash
./workloads/mobilevit/run-experiment.sh --maps-only
```

If simulator configuration or compilation fails after MAPS generates the
application, resume without rerunning input preparation, allocation, or placement:

```bash
./workloads/mobilevit/run-experiment.sh --resume-run workloads/mobilevit/results/<run>
```

This reuses each saved mesh allocation and prepared input token count, completes
the incremental simulator/application builds, then runs and collects results.
The saved allocation fixes the token-slot count. A run with an existing
`results.csv` is rejected to avoid duplicating completed results.

To keep the whole experiment, including CSV collection, running after closing
the terminal:

```bash
nohup ./workloads/mobilevit/run-experiment.sh --maps-only \
  > mobilevit-experiment.log 2>&1 &
```

The runner prints the result directory at startup. For each mesh it writes
`maps-monitor.log` (live stage completions), `maps-monitor/progress.json`
(active operations and token progress), `maps-monitor/stage-events.jsonl`,
and `maps.log` (raw simulator output). The run-root `results.csv` is collected
after each strategy completes: MAPS rows are saved before full-mesh execution
starts, then full-mesh rows are appended after it finishes. `--maps-only`
contains only MAPS rows. A failed or incomplete strategy stops the experiment
and retains diagnostics and any rows from previously completed strategies.

The event-sleep model is built separately in `fifo-platform-tree/`. Use
`--platform-tree /absolute/path/to/libplatform_tree_fifo_sleep.so` to reuse a
matching model for the same mesh. Select polling with `--no-fifo-event-sleep`;
`MAPS_STAGE_PROGRESS=0` also disables stage monitoring in polling mode.
`--stage-progress` enables it explicitly. The environment equivalents are
`MAPS_FIFO_EVENT_SLEEP` (default 1), `MAPS_STAGE_PROGRESS` (default 1),
`MAPS_ONLY` (default 0), and `MAPS_PLATFORM_TREE`.

The default mesh is **32×32**. With the current MAPS planner, this full graph forms 216
stages and needs 408 tiles just for the minimum L1-feasible allocation with two Token
Slots. The slice's 4×4 and 8×8 meshes cannot hold its stages; 16×16 fails the minimum
L1 allocation. `MESH_SIZES="32"` overrides the mesh list; unsupported configurations
stop with the planner's diagnostic. Feasibility can change with Token Slot count.

Execution Tokens (`--tokens`, 1–16) and Token Slots (`--token-slots`, positive integer)
are independent. In comparison mode both strategies consume exactly the same prefix of sixteen distinct,
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
transfers may still be in flight at that point. Progress printing adds overhead;
it is enabled by default with stage monitoring. The SDK build option is
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

Use `--instruction-trace` (or `MAPS_INSTRUCTION_TRACE=1`) to record CV32 and
Spatz instructions during the MAPS simulation. Traces are disabled by default
and written to `<run>/<mesh>/maps-gvsoc/cv32-insn.log` and `spatz-insn.log`.
The separate sequential full-mesh reference is unaffected. Tracing every tile
can produce large files and slow simulation. To trace selected tiles instead,
use `--trace-tiles 0,416,417` (which also enables tracing), or set
`MAPS_INSTRUCTION_TRACE=1 MAPS_TRACE_TILES=0,416,417`. Tile IDs must belong to
the selected mesh; the default selection is `all`.

```bash
./workloads/mobilevit/run-experiment.sh --tokens 2 --trace-tiles 0,416
```

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
Event-driven FIFO receives are enabled by default in this diagnostic as well:

```bash
./.venv/bin/python workloads/mobilevit/diagnostics/run-monitored.py \
  --mesh-root workloads/mobilevit/results/<run>/32x32 --fifo-event-sleep
```

The diagnostic installs the updated Python generators and builds an isolated
compiled platform tree with `magia_v3/fifo_events=1` for the same mesh. The
normal installed platform-tree library is preserved. To reuse a matching
tree, pass `--platform-tree /absolute/path/to/libplatform_tree.so`.
The default installed polling tree does not match the experiment; the JSON
fallback can fail during model construction on 32×32. The diagnostic builds with
`MAPS_FIFO_EVENT_SLEEP=1` and enables the model attribute `magia_v3/fifo_events=1`.
The model adds an experimental remote software-event doorbell at
`0x80000000 + tile_id * 4`; this is a simulation-only extension with no
claimed RTL equivalent. Producers signal event 7 after publishing a FIFO tail.
Consumers sleep with only that event and outstanding outgoing-DMA completion
events enabled, rechecking FIFO counters and DMA tickets after clearing stale
notifications to avoid missed wake-ups. The original event mask is restored
after each wait. `--no-fifo-event-sleep` selects polling receives; producer
backpressure still polls in both modes. The SDK feature itself remains opt-in
through `MAPS_FIFO_EVENT_SLEEP=1`; these MobileViT runners enable it by default.
`diagnostics/isolate-first-two-stages.py` also accepts `--fifo-event-sleep` for
numerical validation of the stem and sigmoid transfers. The focused
`diagnostics/fifo-sleep-smoke.c` checks early arrivals and a forwarding dependency
that requires DMA completion to wake a tile waiting for its next input.

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
reshape/transpose and fused, tile-local grouped im2col + Spatz GEMM for
depthwise convolution with FP16 accumulation and bias added after reduction.
`statistics.json` records the result;
`maps.log` contains each numerical check. `--timeout` bounds simulation wall
time (default 3600 seconds). Production prefix runs retain 32×32.

MAPS transpose offloads valid FP16 permutations to a dedicated Spatz task when
the generated application bundles it. The general path uses strided vector
loads/stores, with raw scalar copies for rows that fail the VLSU word-alignment
requirement. It handles arbitrary permutations, ranks 1–6, strides and cropped
origins without floating-point arithmetic. Other types and applications without
the task retain the existing control-core/DMA implementation.
The vector task recognizes two contiguous input FP16 streams that become adjacent
output pairs, including stage 70's `[1,64,2,2,16,8]` to `[1,64,16,2,8,2]`
permutation. It widens/interleaves raw integer vectors and stores aligned
32-bit output words instead of
submitting and waiting for a tiny DMA transfer per pair. This is a bit-preserving
copy, with no floating-point arithmetic or additional tensor scratch. Normal
origin and non-overlap checks apply before dispatch. Without the Spatz task,
the scalar packed-pair path remains available. Set `MAPS_HAS_TRANSPOSE_SPATZ_TASK=0`
to compare against scalar packed copies, and also `MAPS_TRANSPOSE_PACKED_PAIRS=0`
for the generic DMA baseline.
The adapter diagnostic accepts `--stage70-only`, `--generic-transpose`, and
`--scalar-transpose`, `--transpose-edges`, and `--raw-transpose` for focused
cycle comparisons and arbitrary bit-pattern checks.
Dispatch cycles are recorded in `statistics.json` as `operation_cycles`.

MAPS depthwise convolution builds bounded im2col row bands with ticketed DMA
in the runtime's existing 64 KiB task scratch region, then runs each channel's
`[1,K] × [K,N]` GEMM on Spatz and copies the result to the planned output layout.
The expanded matrix stays local; existing generated graphs, FIFO interfaces,
and tile assignments remain valid. All positive horizontal and vertical
convolution strides use this path for contiguous FP16 tensor columns. Horizontal
stride 1 copies each valid rectangle with one 2-D DMA transfer; larger strides
gather the selected halfwords using one 2-D transfer per output row. This avoids
depending on the simulator's unsupported nested 3-D DMA repetitions.
Noncontiguous tensor columns or insufficient scratch use the direct FP16 task.
Set the control-core compile
definition `MAPS_DEPTHWISE_IM2COL_GEMM=0` to measure the direct implementation.
The separate full-mesh convolution implementation is unchanged. Rebuild both
the control application and Spatz bundle when using the new parameter layout.

Automatic MAPS depthwise dispatch is enabled by default. A calibrated shape
cost selects planned vector execution only with at least 20% estimated headroom;
large spatial shards with few channels retain im2col. The planner uses the same
shape cost. Set `MAPS_DEPTHWISE_VECTOR_AUTO=0` to disable selection.
The explicit control-core definition `MAPS_DEPTHWISE_VECTOR_DIRECT=1` forces
the older row vector implementation for benchmarking. For contiguous FP16
columns, even horizontal strides, even input widths and word-aligned input
rows/channels, one Spatz launch processes the entire shard. Aligned strided
32-bit vector loads select the required FP16 halfwords through a small vector
scratch buffer; masked loads handle padding without reading invalid pixels.
The task retains FP16 fused accumulation and adds bias afterward. Aligned
outputs use vector stores; other output rows use scalar stores. Unsupported
layouts retain the existing im2col or scalar-gather path. Full mesh is unchanged.

Compare the exact stage-153 shard on an isolated 2x2 mesh:

```sh
./.venv/bin/python workloads/mobilevit/diagnostics/benchmark-depthwise.py \
  --mesh-root workloads/mobilevit/results/<run>/32x32 \
  --output /tmp/stage153-vector --stage153 --vector-direct
```

Use `--stage75` instead of `--stage153` for the exact 64-channel stage-75
shard. `--stage 79` and `--stage 157` benchmark the exact stride-1
shards with their saved filters and biases. Omit `--vector-direct` (and
`--vector-plan`) for the im2col baseline. Without `--stage153`, the vector
benchmark also exercises padding, dilation, channel multipliers, odd output
widths and clipped origins, along with existing fallback cases. `--profile`
records phase cycles and submission counts. `--active-tiles 2` uses two workers
instead of four on the same 2x2 mesh. Activations are synthetic; stage-153
weights and biases come from the saved application.

`--vector-plan` additionally precomputes indexed input offsets and padding masks
once per block of up to 64 spatial output positions, then reuses them across
channels. One vector can span multiple output rows. It requires the same
input alignment and even-input-width contract, but supports every positive
horizontal/vertical stride. Odd horizontal strides use precomputed per-lane
shifts to select the low or high FP16 half; even strides retain a uniform shift.
It supports filters with up to 49 elements;
larger filters retain the row-based vector path for even strides or the
existing im2col/scalar fallback otherwise. The geometry tables use about
26 KiB of Spatz stack. Automatic dispatch avoids this path when large spatial
outputs with few channels make geometry preparation more expensive than im2col. When a complete output row of
32–64 values fits a vector, the task instead caches one horizontal template
and rebases the input for each row. This avoids repeated geometry preparation
on the stage-12 shape, including masks for top/bottom padding. The corresponding control-core
definitions are `MAPS_DEPTHWISE_VECTOR_DIRECT=1` and
`MAPS_DEPTHWISE_VECTOR_PLAN=1`. With `--profile`, `vector_profiles` in
`statistics.json` separates geometry, vector input loading/extraction, weight
loading/FMA, and bias/output stores; those measurements include profiling
overhead. Unprofiled dispatch cycles are used for speed comparisons.

The MAPS Spatz GEMM task vectorizes transposed weights when K is even and B
is word aligned. It gathers aligned pairs of weights across output columns,
uses each word for two successive FP16 FMAs in ascending K order, then applies
the existing bias operation. Odd output counts and unaligned output rows are
supported; odd K or unaligned B retain scalar arithmetic. This path is enabled
by default; `MATMUL_TRANSPOSED_B_VECTOR=0` in the Spatz compiler definitions
selects the previous scalar implementation. The adapter diagnostic's
`--matmul-only` checks the saved classifier with real weights/bias and synthetic
inputs plus batching, bias, alignment and tail cases. Add `--scalar-matmul` for
a scalar comparison using the same deterministic inputs and bitwise reference.

`benchmark-depthwise.py --stage 27` exercises all eleven saved stage-27 channel
shards, including both 11-channel and 12-channel layouts, on four workers of
the isolated 2x2 mesh. Add `--vector-plan` to compare the new kernel.

`test-hw-adapters.py --stage0 --profile-gemm --profile-im2col` profiles all eight
saved stem shards. It checks im2col against a host reference, then tests GEMM
using that reference matrix, the saved weights/biases and ascending-K FP16 FMA
rounding. `operation_cycles` records total dispatch time; `im2col_profiles`
records zero-fill, DMA submission/wait cycles, descriptor count and cleared
bytes. `gemm_profiles` separates matrix arithmetic from bias addition. Omit the
profile flags for a baseline. Instrumentation is disabled in ordinary builds;
diagnostic builds reserve L1 counters at offset `0xBFF00` and after the GEMM
parameter block. The probe excludes input loading and numerical checking from
kernel timing and does not include full-workload FIFO/communication traffic.

Compare the original FP32 kernel, direct FP16 kernel, and grouped im2col + GEMM
on a 2×2 mesh using the stage-6 shard shape (one 128×128 channel, 3×3 filter):

```bash
./.venv/bin/python workloads/mobilevit/diagnostics/benchmark-depthwise.py \
  --mesh-root workloads/mobilevit/results/<run>/32x32 \
  --output /tmp/depthwise-direct --direct
./.venv/bin/python workloads/mobilevit/diagnostics/benchmark-depthwise.py \
  --mesh-root workloads/mobilevit/results/<run>/32x32 \
  --output /tmp/depthwise-im2col
```

Use `--original-fp32` for the SDK HEAD kernel and its FP32 reference, or
`--edges-only` for vertical stride and clipped input-halo cases. `--stride-suite`
compares the stage-12 shard (three channels, 128×128 input, 64×64 output, 3×3,
stride 2) and stride-3, asymmetric-stride, dilation, and clipped-halo cases.
The benchmark uses deterministic synthetic activations and actual stage-6
MobileViT weights and biases (stage-12 weights with `--stride-suite`).
It checks each output against a host reference, verifies output
layout gaps, and measures dispatch cycles including im2col, task launches,
GEMM, and output copies, excluding input staging and numerical checks.

`--filter-suite` additionally checks 1×1, rectangular 3×5, 5×5 stride-2,
and dilated rectangular 2×4 filters against FP16 references, using synthetic
filter values. Depthwise dimensions come from operation descriptors; the
implementation is not restricted to MobileViT's 3×3 filters.

Padding-only im2col initialization and FP16 vector bias addition are enabled
by default. Numerical MAPS output validation saves `maps-output-tokenN.bin`
and `maps-validation.json` alongside each mesh log. CSV mismatch counts use
the reference tolerances (atol 0.002, rtol 0.02), with nonfinite values counted
separately; legacy logs without output bytes retain checksum-only validation.
Start a fresh experiment after these planner changes: resume keeps the old allocation.

Im2col per-transfer progress prints are disabled by default. Stage/kernel
begin/end monitoring and live elapsed-cycle summaries remain enabled. To debug
individual im2col rows, set the control application compile definition
`MAPS_IM2COL_STEP_PROGRESS=1` together with `MAPS_KERNEL_PROGRESS=1`.
These inner-loop prints substantially perturb cycle measurements for small
convolutions; leave them off for performance comparisons.

Build a standalone 32x32 full-mesh executable (two serial tokens) without MAPS
planning or starting a simulator:

```bash
.venv/bin/python workloads/mobilevit/full-mesh/build-standalone.py
./workloads/mobilevit/results/full-mesh-32x32/run.sh
```

The build packages `mobilevit_full_mesh_32x32.elf` and its Spatz boot ROM in that
result directory. After both tokens finish, tile 0 prints
`REFERENCE_COMPLETION token=0 cycle=...` and
`REFERENCE_COMPLETION token=1 cycle=...`. Cycles are cumulative from the common
execution-window start after initialization; each is the maximum recorded over
all 1024 tiles at the final layer. The token-1 value includes token-0 execution.
