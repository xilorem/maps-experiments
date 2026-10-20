#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [--resume-run DIR] [--tokens N] [--token-slots N] [--maps-only] [--stage-progress] [--token-progress] [--fifo-event-sleep | --no-fifo-event-sleep] [--platform-tree FILE] [--instruction-trace] [--trace-tiles all|N,N,...]"
  echo "FIFO event sleep and stage monitoring are enabled by default."
}

tokens=2
token_slots=2
resume_run=""
maps_timings="${MAPS_TIMINGS:-0}"
maps_imiss_trace="${MAPS_IMISS_TRACE:-0}"
maps_token_progress="${MAPS_TOKEN_PROGRESS:-0}"
maps_instruction_trace="${MAPS_INSTRUCTION_TRACE:-0}"
maps_trace_tiles="${MAPS_TRACE_TILES:-all}"
maps_fifo_event_sleep="${MAPS_FIFO_EVENT_SLEEP:-1}"
maps_stage_progress="${MAPS_STAGE_PROGRESS:-1}"
maps_only="${MAPS_ONLY:-0}"
maps_platform_tree="${MAPS_PLATFORM_TREE:-}"
communication_weight="${COMMUNICATION_WEIGHT:-1.0}"
if [[ "$maps_timings" != 0 && "$maps_timings" != 1 ]]; then
  echo "MAPS_TIMINGS must be 0 or 1" >&2
  exit 2
fi
if [[ "$maps_imiss_trace" != 0 && "$maps_imiss_trace" != 1 ]]; then
  echo "MAPS_IMISS_TRACE must be 0 or 1" >&2
  exit 2
fi
if [[ "$maps_token_progress" != 0 && "$maps_token_progress" != 1 ]]; then
  echo "MAPS_TOKEN_PROGRESS must be 0 or 1" >&2
  exit 2
fi
if [[ "$maps_imiss_trace" == 1 && "$maps_timings" != 1 ]]; then
  echo "MAPS_IMISS_TRACE requires MAPS_TIMINGS=1" >&2
  exit 2
fi
while (($#)); do
  case "$1" in
  --resume-run)
    [[ $# -ge 2 ]] || { usage >&2; exit 2; }
    resume_run="$2"
    shift 2
    ;;
  --tokens)
    [[ $# -ge 2 ]] || { usage >&2; exit 2; }
    tokens="$2"
    shift 2
    ;;
  --token-slots)
    [[ $# -ge 2 ]] || { usage >&2; exit 2; }
    token_slots="$2"
    shift 2
    ;;
  --token-progress)
    maps_token_progress=1
    shift
    ;;
  --stage-progress)
    maps_stage_progress=1
    shift
    ;;
  --fifo-event-sleep)
    maps_fifo_event_sleep=1
    shift
    ;;
  --no-fifo-event-sleep)
    maps_fifo_event_sleep=0
    shift
    ;;
  --maps-only)
    maps_only=1
    shift
    ;;
  --platform-tree)
    [[ $# -ge 2 ]] || { usage >&2; exit 2; }
    maps_platform_tree="$2"
    shift 2
    ;;
  --instruction-trace)
    maps_instruction_trace=1
    shift
    ;;
  --trace-tiles)
    [[ $# -ge 2 ]] || { usage >&2; exit 2; }
    maps_trace_tiles="$2"
    maps_instruction_trace=1
    shift 2
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
  esac
done

for setting in maps_fifo_event_sleep maps_stage_progress maps_only; do
  if [[ "${!setting}" != 0 && "${!setting}" != 1 ]]; then
    echo "$setting must be 0 or 1" >&2
    exit 2
  fi
done
if ((maps_fifo_event_sleep)); then
  maps_stage_progress=1
fi
if ((maps_stage_progress)); then
  maps_token_progress=1
fi
if [[ -n "$maps_platform_tree" && ! -f "$maps_platform_tree" ]]; then
  echo "platform-tree library not found: $maps_platform_tree" >&2
  exit 2
fi
if [[ -n "$maps_platform_tree" ]]; then
  maps_platform_tree="$(cd "$(dirname "$maps_platform_tree")" && pwd)/$(basename "$maps_platform_tree")"
fi

if [[ "$maps_instruction_trace" != 0 && "$maps_instruction_trace" != 1 ]]; then
  echo "MAPS_INSTRUCTION_TRACE must be 0 or 1" >&2
  exit 2
fi
if [[ "$maps_trace_tiles" != all && ! "$maps_trace_tiles" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
  echo "--trace-tiles must be all or comma-separated non-negative tile IDs" >&2
  exit 2
fi

if [[ ! "$tokens" =~ ^[1-9][0-9]*$ || "$tokens" -gt 16 || ! "$token_slots" =~ ^[1-9][0-9]*$ ]]; then
  echo "--tokens must be 1..16; --token-slots must be positive" >&2
  exit 2
fi

printf 'MobileViT run: execution tokens=%s, token slots=%s, FIFO sleep=%s, stage progress=%s, MAPS only=%s\n' "$tokens" "$token_slots" "$maps_fifo_event_sleep" "$maps_stage_progress" "$maps_only"

experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$experiment_root/../.." && pwd)"
checkout_root="$(dirname "$repository_root")"
maps_root="${MAPS_ROOT:-$checkout_root/MAPS}"
sdk_root="${MAGIA_SDK_ROOT:-$checkout_root/magia-sdk-v3}"
# MAPS does not use the PULP cluster, so the SDK is built without it (pulp_cores=0,
# which also selects the spatz_on_magia tests, where the full-mesh reference
# lives). The GVSoC MAGIA v3 model needs at least one PULP core per tile; they
# stay idle.
pulp_cores="${PULP_CORES:-0}"
gvsoc_pulp_cores="${GVSOC_PULP_CORES:-8}"
python="${PYTHON:-$repository_root/.venv/bin/python}"
if [[ ! -x "$python" ]]; then
  python=python3
fi
# GVSoC's launcher invokes python3 from PATH. Use the same environment as
# preparation/collection instead of accidentally selecting system Python.
python_directory="$("$python" -c 'import pathlib, sys; print(pathlib.Path(sys.executable).parent)')"
export PATH="$python_directory:$PATH"
if [[ -n "${COREV_ROOT:-}" ]]; then
  export PATH="$COREV_ROOT/bin:$PATH"
  if ! command -v riscv32-corev-elf-gcc >/dev/null 2>&1; then
    echo "CORE-V compiler not found; check COREV_ROOT" >&2
    exit 2
  fi
else
  if ! command -v riscv32-unknown-elf-gcc >/dev/null 2>&1; then
    for candidate in "${PULP_ROOT:-$checkout_root/toolchains/pulp-toolchain}" "$sdk_root/pulp-toolchain"; do
      if [[ -x "$candidate/bin/riscv32-unknown-elf-gcc" ]]; then
        export PATH="$candidate/bin:$PATH"
        break
      fi
    done
  fi
  if ! command -v riscv32-unknown-elf-gcc >/dev/null 2>&1; then
    echo "RISC-V compiler not found; set PULP_ROOT to the toolchain directory" >&2
    exit 2
  fi
fi
if [[ -z "${LLVM_INSTALL_DIR:-}" ]]; then
  for candidate in "$sdk_root/llvm/install" "$checkout_root/toolchains/llvm20-root/usr/lib/llvm-20" /usr/lib/llvm-20 /usr/lib/llvm-18; do
    if [[ -x "$candidate/bin/clang" && -x "$candidate/bin/llvm-objcopy" && -x "$candidate/bin/llvm-objdump" ]]; then
      LLVM_INSTALL_DIR="$candidate"
      break
    fi
  done
fi
: "${LLVM_INSTALL_DIR:?set LLVM_INSTALL_DIR to a Spatz-capable LLVM installation}"
export LLVM_INSTALL_DIR

# The checkout-local LLVM package needs its accompanying shared libraries.
llvm_runtime_libs="$(dirname "$LLVM_INSTALL_DIR")/x86_64-linux-gnu"
if [[ -d "$llvm_runtime_libs" ]]; then
  export LD_LIBRARY_PATH="$llvm_runtime_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

model="$experiment_root/model/mobilevitv2_050.cvnets_in1k.bs1.simplified.wfold.noexpand.gn.convfold.single_output.onnx"
stored_inputs="$experiment_root/model/inputs.bin"
stored_references="$experiment_root/model/references.bin"
run_id="$(date -u +%Y%m%dT%H%M%S.%NZ)"
run_root="$experiment_root/results/$run_id"
if [[ -n "$resume_run" ]]; then
  [[ -d "$resume_run/prepared" ]] || { echo "resume run has no prepared data: $resume_run" >&2; exit 2; }
  run_root="$(cd "$resume_run" && pwd)"
  [[ ! -e "$run_root/results.csv" ]] || { echo "resume run already has CSV results; refusing to append duplicate results" >&2; exit 2; }
fi
prepared="$run_root/prepared"
csv="$run_root/results.csv"

mkdir -p "$prepared"
echo "Run directory: $run_root"
echo "Results CSV: $csv"
if [[ -z "$resume_run" ]]; then
  "$python" "$experiment_root/results.py" prepare \
  --model "$model" \
  --inputs "$stored_inputs" \
  --references "$stored_references" \
  --tokens "$tokens" \
  --output "$prepared"
else
  # Prepared input data fixes the token count; keep the generated allocation.
  if [[ -f "$run_root/run-config.json" ]]; then
    tokens="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens"])' "$run_root/run-config.json")"
  else
    tokens="$("$python" -c 'import json,math,pathlib,sys; m=json.load(open(sys.argv[2])); print(pathlib.Path(sys.argv[1]).stat().st_size // (2*math.prod(m["input_shape"])))' "$prepared/inputs.bin" "$experiment_root/model/manifest.json")"
  fi
fi
input_name="$(<"$prepared/input-name.txt")"

# Override MESH_SIZES to choose the mesh sizes for this experiment.
mesh_sizes="${MESH_SIZES:-32}"
if [[ -z "$resume_run" ]]; then
  "$python" -c 'import json,sys; json.dump({"tokens":int(sys.argv[2])},open(sys.argv[1],"w"))' "$run_root/run-config.json" "$tokens"
else
  mesh_sizes=""
  for saved_mesh in "$run_root"/*x*; do
    [[ -d "$saved_mesh/maps-application" && -f "$saved_mesh/execution-plan.json" ]] || continue
    mesh_name="$(basename "$saved_mesh")"
    [[ "$mesh_name" =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]] || continue
    [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]] || continue
    mesh_sizes+=" ${mesh_name%x*}"
  done
  [[ -n "$mesh_sizes" ]] || { echo "resume run has no saved square-mesh allocations" >&2; exit 2; }
  echo "Reusing generated MAPS allocation; execution tokens=$tokens"
fi
for tiles in $mesh_sizes; do
  [[ "$tiles" =~ ^[1-9][0-9]*$ ]] || { echo "invalid mesh size: $tiles" >&2; exit 2; }
  mesh="${tiles}x${tiles}"
  mesh_root="$run_root/$mesh"
  application="$mesh_root/maps-application"
  sdk_build="$mesh_root/sdk-build"
  maps_log="$mesh_root/maps.log"
  full_mesh_log="$mesh_root/full-mesh.log"
  build_log="$mesh_root/build.log"
  execution_plan="$mesh_root/execution-plan.json"

  maps_run_args=()
  if ((maps_instruction_trace)); then
    trace_args='$(GVRUN_COMMON_ARGS)'
    if [[ "$maps_trace_tiles" == all ]]; then
      trace_args+=" --trace=tile-[0-9]+-cv32-core/insn:cv32-insn.log --trace=tile-[0-9]+-snitch-spatz/insn:spatz-insn.log"
    else
      IFS=, read -r -a trace_tiles <<< "$maps_trace_tiles"
      for trace_tile in "${trace_tiles[@]}"; do
        if [[ ${#trace_tile} -gt 9 ]] || ((trace_tile >= tiles * tiles)); then
          echo "trace tile $trace_tile is outside the $mesh mesh" >&2
          exit 2
        fi
        trace_args+=" --trace=tile-$trace_tile-cv32-core/insn:cv32-insn.log --trace=tile-$trace_tile-snitch-spatz/insn:spatz-insn.log"
      done
    fi
    maps_run_args+=("GVRUN_ARGS=$trace_args run")
    echo "MAPS instruction traces: $mesh_root/maps-gvsoc/{cv32-insn,spatz-insn}.log (tiles=$maps_trace_tiles)"
  fi

  mkdir -p "$mesh_root"

  if [[ -z "$resume_run" ]]; then
    make -C "$maps_root" build \
    MODEL="$model" \
    TARGET=magia-v3 \
    MESH="$mesh" \
    TOKEN_SLOTS="$token_slots" \
    COMMUNICATION_WEIGHT="$communication_weight" \
    NAME=maps_mobilevit \
    INPUT="$input_name=$prepared/inputs.bin" \
    EXECUTION_PLAN="$execution_plan" \
    APPLICATION="$application" 2>&1 | tee -a "$build_log"

  # CSV collection needs start/end cycles even without detailed timing traces.
  cat >>"$application/CMakeLists.txt" <<'EOF'

target_compile_definitions(maps_mobilevit PRIVATE MAPS_RESULT_TRACE=1)
EOF
  if ((maps_fifo_event_sleep)); then
    echo 'target_compile_definitions(maps_mobilevit PRIVATE MAPS_FIFO_EVENT_SLEEP=1)' >>"$application/CMakeLists.txt"
  fi
  if ((maps_stage_progress)); then
    "$python" "$experiment_root/diagnostics/run-monitored.py" \
      --mesh-root "$mesh_root" --prepare-application "$application"
  fi

  # Build the experiment-owned full-mesh reference unless MAPS-only was requested.
  if ((!maps_only)); then
    cat >>"$application/CMakeLists.txt" <<EOF

set(MVIT_DATA "$prepared/full-mesh-data.bin")
set(MVIT_PREPARED "$prepared")
set(MVIT_TOKENS "$tokens")
add_subdirectory("$experiment_root/full-mesh" full_mesh)
EOF
  fi

  if ((maps_timings)); then
    cat >>"$application/CMakeLists.txt" <<'EOF'

# Experiment-only detailed timing trace for this generated MAPS application.
target_compile_definitions(maps_mobilevit PRIVATE
  MAPS_EXPERIMENT_TRACE=1
)
EOF
  fi
  else
    [[ -f "$application/CMakeLists.txt" ]] || { echo "missing generated application: $application" >&2; exit 2; }
    "$python" - "$application/CMakeLists.txt" "$maps_fifo_event_sleep" <<'PY'
from pathlib import Path
import re, sys
p = Path(sys.argv[1])
source = re.sub(r'MAPS_FIFO_EVENT_SLEEP=[01]', 'MAPS_FIFO_EVENT_SLEEP=' + sys.argv[2], p.read_text())
if 'MAPS_FIFO_EVENT_SLEEP=' not in source:
    source += '\ntarget_compile_definitions(maps_mobilevit PRIVATE MAPS_FIFO_EVENT_SLEEP=' + sys.argv[2] + ')\n'
p.write_text(source)
PY
    if ! rg -q 'MAPS_RESULT_TRACE=1' "$application/CMakeLists.txt"; then
      echo 'target_compile_definitions(maps_mobilevit PRIVATE MAPS_RESULT_TRACE=1)' >>"$application/CMakeLists.txt"
    fi
    if ((maps_stage_progress)); then
      "$python" "$experiment_root/diagnostics/run-monitored.py" --mesh-root "$mesh_root" --prepare-application "$application"
    fi
    if ! rg -q 'add_subdirectory\(.*full_mesh\)' "$application/CMakeLists.txt"; then
      maps_only=1
    fi
  fi

  make -C "$sdk_root" gvsoc tiles="$tiles" pulp_cores="$gvsoc_pulp_cores" 2>&1 | tee -a "$build_log"

  platform_tree="$maps_platform_tree"
  if ((maps_fifo_event_sleep)) && [[ -z "$platform_tree" ]]; then
    "$python" "$experiment_root/diagnostics/fifo_sleep_tree.py" \
      --sdk-root "$sdk_root" --mesh "$tiles" --output "$mesh_root/fifo-platform-tree" \
      2>&1 | tee -a "$build_log"
    platform_tree="$mesh_root/fifo-platform-tree/libplatform_tree_fifo_sleep.so"
  fi

  sdk_build_args=(
    tiles="$tiles"
    pulp_cores="$pulp_cores"
    CMAKE_BUILDDIR="$sdk_build"
    maps_application_dir="$application"
    maps_token_progress="$maps_token_progress"
    maps_kernel_progress="$maps_stage_progress"
  )
  if ((maps_timings)); then
    sdk_build_args+=(maps_experiment_trace=0)
  fi
  make -C "$sdk_root" build "${sdk_build_args[@]}" test=spatz_bootrom 2>&1 | tee -a "$build_log"
  make -C "$sdk_root" build "${sdk_build_args[@]}" test=maps_mobilevit 2>&1 | tee -a "$build_log"
  if ((!maps_only)); then
    make -C "$sdk_root" build "${sdk_build_args[@]}" test=onnx_mobilevit_full_mesh 2>&1 | tee -a "$build_log"
  fi

  if ((maps_stage_progress)); then
    monitor_args=(--foreground --mesh-root "$mesh_root" --sdk-root "$sdk_root"
      --sdk-build "$sdk_build" --tokens "$tokens" --output "$mesh_root/maps-monitor"
      --maps-log "$maps_log" --work-dir "$mesh_root/maps-gvsoc")
    if ((maps_fifo_event_sleep)); then
      monitor_args+=(--fifo-event-sleep)
    else
      monitor_args+=(--no-fifo-event-sleep)
    fi
    if [[ -n "$platform_tree" ]]; then
      monitor_args+=(--platform-tree "$platform_tree")
    fi
    if ((maps_instruction_trace)); then
      monitor_args+=(--instruction-trace --trace-tiles "$maps_trace_tiles")
    fi
    "$python" "$experiment_root/diagnostics/run-monitored.py" "${monitor_args[@]}" \
      2>&1 | tee "$mesh_root/maps-monitor.log"
  else
    make -C "$sdk_root" run \
      tiles="$tiles" \
      pulp_cores="$gvsoc_pulp_cores" \
      CMAKE_BUILDDIR="$sdk_build" \
      GVSOC_WORK_DIR="$mesh_root/maps-gvsoc" \
      test=maps_mobilevit \
      "${maps_run_args[@]}" \
      platform=gvsoc | tee "$maps_log"
  fi

  "$python" "$experiment_root/results.py" collect-maps \
    --mesh "$mesh" --tokens "$tokens" --maps-log "$maps_log" \
    --reference-checksums "$prepared/reference-checksums.csv" --csv "$csv"

  if ((!maps_only)); then
    make -C "$sdk_root" run \
      tiles="$tiles" \
      pulp_cores="$gvsoc_pulp_cores" \
      CMAKE_BUILDDIR="$sdk_build" \
      GVSOC_WORK_DIR="$mesh_root/full-mesh-gvsoc" \
      test=onnx_mobilevit_full_mesh \
      platform=gvsoc | tee "$full_mesh_log"
    "$python" "$experiment_root/results.py" collect-full-mesh \
      --mesh "$mesh" --tokens "$tokens" --full-mesh-log "$full_mesh_log" \
      --reference-available "$prepared/reference-available.txt" --csv "$csv"
  fi
  if ((maps_timings)); then
    imiss_args=()
    if ((maps_imiss_trace)); then
      imiss_work="$mesh_root/imiss-gvsoc"
      mkdir -p "$imiss_work"
      imiss_model_args=()
      if ((maps_fifo_event_sleep)); then
        imiss_model_args+=(--attr magia_v3/fifo_events=1)
      fi
      if [[ -n "$platform_tree" ]]; then
        imiss_model_args+=(--gv-opt "platform_tree=$platform_tree")
      fi
      "$sdk_root/gvsoc/install/bin/gvrun" \
        --target "magia_v3:n_tiles_x=$tiles,n_tiles_y=$tiles,nb_pulp_cores=$gvsoc_pulp_cores" \
        --param binary="$sdk_build/bin/maps_mobilevit" \
        --work-dir "$imiss_work" \
        --attr "magia_v3/spatz_romfile=$sdk_build/bin/bootrom/spatz_init.bin" \
        "${imiss_model_args[@]}" \
        --trace-level=trace \
        --trace=kill-module \
        --vcd \
        '--event=.*snitch-spatz/event_instr' \
        '--event=.*snitch-spatz/event_imiss' \
        run | tee "$mesh_root/imiss.log"
      imiss_args=(--trace-vcd "$imiss_work/all.vcd")
    fi
    "$python" "$experiment_root/results.py" timings \
      --maps-log "$maps_log" \
      --application "$application" \
      --execution-plan "$execution_plan" \
      --output "$mesh_root/maps-timings"
    "$python" "$experiment_root/calibration.py" \
      --mesh "$mesh" \
      --maps-log "$maps_log" \
      --application "$application" \
      "${imiss_args[@]}" \
      --output "$mesh_root/calibration.csv"
  fi
done

echo "Results: $csv"
