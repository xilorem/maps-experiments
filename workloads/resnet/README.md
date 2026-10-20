# ResNet18 support status

The supplied model imports, rewrites and specializes for MAGIA-v3, with device
implementations for all resulting nodes. MaxPool lowers to a channel-sharded
FP16 Spatz task, bundled automatically by MAPS-IR. A full experiment runner is available below; a validated full-network result
is still pending.

MaxPool supports NCHW values, positive strides, rectangular kernels, explicit
asymmetric padding, SAME_UPPER/SAME_LOWER/VALID and ceil mode. Unit dilation is
required; optional index outputs are rejected. Input/output channel shards are
packed and retain complete spatial planes. Full-mesh pooling is unchanged.

The current adapter reuses the SDK kernel, including scalar handling of
misaligned windows. This is functional support, with no claim of a vector speedup.
The isolated ResNet spatial shape (2 channels, 112x112 -> 56x56, 3x3/stride 2)
takes about 1,016,191 cycles per token. Six layouts, two tokens each, checked
13,026 output values with zero mismatches, including negative padding edges,
odd widths, rectangular kernels, batches, ceil mode, and stride 3.

Reproduce the isolated tests from the experiments repository:

```bash
.venv/bin/python workloads/mobilevit/diagnostics/test-hw-adapters.py \
  --maxpool-only --mesh-root workloads/mobilevit/results/<saved-run>/32x32 \
  --output workloads/resnet/diagnostics/maxpool-support
```

Use `--sdk-root /path/to/isolated/sdk` when another experiment may be building;
the diagnostic builds a 2x2 application and temporarily updates SDK tile config.
Measurements and logs are in `diagnostics/maxpool-support/` (ignored by Git).

## Full experiment runner

From the experiments repository, run:

```bash
./workloads/resnet/run-experiment.sh
```

Defaults are a 32x32 mesh, two execution tokens, two token slots, MAPS followed
by full mesh. `--maps-only` omits the reference strategy. FIFO event sleep and
live kernel/stage monitoring are enabled. Each run snapshots SDK sources into
its result directory and uses separate build directories; it reuses the
installed simulator without rebuilding the shared GVSoC installation.

The runner prints its result directory immediately. During MAPS simulation,
`STAGE_COMPUTE_DONE` and `STAGE_PROCESSING_DONE` show `cycle` and
`elapsed_cycles`; elapsed cycles span the earliest kernel start through the
latest tile completion in that stage/token. They are simulated cycles, not
wall-clock seconds. Full mesh prints per-layer begin/end cycles and elapsed
cycles. CSV completion cycles use the measured execution window.

Files under `results/<run>/`:

- `results.csv`: appended immediately after each strategy completes.
- `32x32/maps.log`: raw MAPS simulator output.
- `32x32/maps-monitor/progress.json`: active kernels and stage/token progress.
- `32x32/maps-monitor/stage-events.jsonl`: stage events and cycle measurements.
- `32x32/full-mesh.log`: full-mesh live layer timing and validation.
- `32x32/maps-validation.json`: numerical MAPS output errors against the ONNX reference.

Inputs are deterministic FP16 random activations. Both strategies use the same
inputs and FP16 weights; validation compares their outputs to ONNX Runtime.
Numerical mismatches are reported in CSV rather than failing the run. The
SDK's approximate softmax can differ from ONNX softmax. A 4x4 smoke network
completed both strategies and produced all four rows: MAPS had zero mismatches;
full mesh had two of eight output values outside tolerance on token 0 and zero
on token 1. The complete ResNet network has not yet been run with this runner.

To prepare or build without launching simulation, use `--prepare-only` or
`--build-only`. Resume an incomplete build/run with:

```bash
./workloads/resnet/run-experiment.sh --resume-run workloads/resnet/results/<run>
```

Resume keeps the saved allocation and input data, refreshes the isolated SDK
sources, rebuilds, and restarts simulation at token zero. Existing CSV results
are protected against accidental duplication. Optional instruction tracing is
available with `--instruction-trace` or `--trace-tiles 0,1`.

Im2col per-transfer progress prints are opt-in (`MAPS_IM2COL_STEP_PROGRESS=1`);
normal stage/kernel live timings stay enabled. On the same 4x4 smoke allocation,
disabling only these messages reduced the slowest token-1 im2col from 195,200 to
32,562 cycles and the token completion interval from 224,454 to 139,506 cycles.
Outputs were bitwise identical. This is a smoke-network measurement, not a
full ResNet18 throughput result. Profiling showed about 130k–163k cycles spent
formatting/printing the 27 im2col progress records per tile. Diagnostic logs and
comparisons are under `diagnostics/im2col-investigation/` (ignored by Git).

The MAPS-IR memory check accounts for packed FP16 initializers in the SDK's
192 MiB bulk region separately from the 15 MiB application runtime-data arena.
ResNet's 23,369,424-byte initializer image exceeds the runtime arena but fits
the bulk region. Runtime buffers retain their independent arena limit. Compiler
failures now retain their execution plan, inputs to code generation, and detailed
backend error in a `maps-application.generation-failure-*` directory, so the
underlying compiler error is available even when the CLI summary is generic.
