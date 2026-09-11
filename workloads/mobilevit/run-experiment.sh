#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [--tokens N] [--token-slots N]"
}

tokens=2
token_slots=2
maps_timings="${MAPS_TIMINGS:-0}"
maps_imiss_trace="${MAPS_IMISS_TRACE:-0}"
if [[ "$maps_timings" != 0 && "$maps_timings" != 1 ]]; then
  echo "MAPS_TIMINGS must be 0 or 1" >&2
  exit 2
fi
if [[ "$maps_imiss_trace" != 0 && "$maps_imiss_trace" != 1 ]]; then
  echo "MAPS_IMISS_TRACE must be 0 or 1" >&2
  exit 2
fi
if [[ "$maps_imiss_trace" == 1 && "$maps_timings" != 1 ]]; then
  echo "MAPS_IMISS_TRACE requires MAPS_TIMINGS=1" >&2
  exit 2
fi
while (($#)); do
  case "$1" in
  --tokens)
    tokens="$2"
    shift 2
    ;;
  --token-slots)
    token_slots="$2"
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

experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$experiment_root/../.." && pwd)"
checkout_root="$(dirname "$repository_root")"
maps_root="${MAPS_ROOT:-$checkout_root/MAPS}"
sdk_root="${MAGIA_SDK_ROOT:-$checkout_root/magia-sdk-v3}"
python="${PYTHON:-$repository_root/.venv/bin/python}"
if [[ ! -x "$python" ]]; then
  python=python3
fi
if [[ -z "${LLVM_INSTALL_DIR:-}" ]]; then
  for candidate in "$sdk_root/llvm/install" /usr/lib/llvm-20 /usr/lib/llvm-18; do
    if [[ -x "$candidate/bin/clang" && -x "$candidate/bin/llvm-objcopy" && -x "$candidate/bin/llvm-objdump" ]]; then
      LLVM_INSTALL_DIR="$candidate"
      break
    fi
  done
fi
: "${LLVM_INSTALL_DIR:?set LLVM_INSTALL_DIR to a Spatz-capable LLVM installation}"
export LLVM_INSTALL_DIR

model="$experiment_root/model/mobilevit-nodes-170-179.onnx"
stored_inputs="$experiment_root/model/inputs.bin"
stored_references="$experiment_root/model/references.bin"
run_id="$(date -u +%Y%m%dT%H%M%S.%NZ)"
run_root="$experiment_root/results/$run_id"
prepared="$run_root/prepared"
csv="$run_root/results.csv"

mkdir -p "$prepared"
"$python" "$experiment_root/results.py" prepare \
  --model "$model" \
  --inputs "$stored_inputs" \
  --references "$stored_references" \
  --tokens "$tokens" \
  --output "$prepared"
input_name="$(<"$prepared/input-name.txt")"

# MESH_SIZES is used for focused development runs; an ordinary experiment runs all three.
mesh_sizes="${MESH_SIZES:-4 8}"
for tiles in $mesh_sizes; do
  mesh="${tiles}x${tiles}"
  mesh_root="$run_root/$mesh"
  application="$mesh_root/maps-application"
  sdk_build="$mesh_root/sdk-build"
  maps_log="$mesh_root/maps.log"
  full_mesh_log="$mesh_root/full-mesh.log"
  build_log="$mesh_root/build.log"
  execution_plan="$mesh_root/execution-plan.json"

  mkdir -p "$mesh_root"

  make -C "$maps_root" build \
    MODEL="$model" \
    TARGET=magia-v3 \
    MESH="$mesh" \
    TOKEN_SLOTS="$token_slots" \
    NAME=maps_mobilevit_slice \
    INPUT="$input_name=$prepared/inputs.bin" \
    APPLICATION="$application" 2>&1 | tee -a "$build_log"

  make -C "$maps_root" plan \
    MODEL="$model" \
    TARGET=magia-v3 \
    MESH="$mesh" \
    TOKEN_SLOTS="$token_slots" \
    EXECUTION_PLAN="$execution_plan" 2>&1 | tee -a "$build_log"

  if ((maps_timings)); then
    cat >>"$application/CMakeLists.txt" <<'EOF'

# Experiment-only detailed timing trace for this generated MAPS application.
target_compile_definitions(maps_mobilevit_slice PRIVATE
  MAPS_EXPERIMENT_TRACE=1
)
EOF
  fi

  make -C "$sdk_root" gvsoc tiles="$tiles" 2>&1 | tee -a "$build_log"

  sdk_build_args=(
    tiles="$tiles"
    CMAKE_BUILDDIR="$sdk_build"
    maps_application_dir="$application"
    mobilevit_slice_data="$prepared/slice-data.bin"
    mobilevit_slice_tokens="$tokens"
  )
  if ((maps_timings)); then
    sdk_build_args+=(maps_experiment_trace=0)
  fi
  make -C "$sdk_root" build "${sdk_build_args[@]}" test=spatz_bootrom 2>&1 | tee -a "$build_log"
  make -C "$sdk_root" build "${sdk_build_args[@]}" test=maps_mobilevit_slice 2>&1 | tee -a "$build_log"
  make -C "$sdk_root" build "${sdk_build_args[@]}" test=onnx_mobilevit_slice 2>&1 | tee -a "$build_log"

  make -C "$sdk_root" run \
    tiles="$tiles" \
    CMAKE_BUILDDIR="$sdk_build" \
    GVSOC_WORK_DIR="$mesh_root/maps-gvsoc" \
    test=maps_mobilevit_slice \
    platform=gvsoc | tee "$maps_log"

  make -C "$sdk_root" run \
    tiles="$tiles" \
    CMAKE_BUILDDIR="$sdk_build" \
    GVSOC_WORK_DIR="$mesh_root/full-mesh-gvsoc" \
    test=onnx_mobilevit_slice \
    platform=gvsoc | tee "$full_mesh_log"

  "$python" "$experiment_root/results.py" collect \
    --mesh "$mesh" \
    --tokens "$tokens" \
    --maps-log "$maps_log" \
    --full-mesh-log "$full_mesh_log" \
    --reference-checksums "$prepared/reference-checksums.csv" \
    --reference-available "$prepared/reference-available.txt" \
    --csv "$csv"
  if ((maps_timings)); then
    imiss_args=()
    if ((maps_imiss_trace)); then
      imiss_work="$mesh_root/imiss-gvsoc"
      mkdir -p "$imiss_work"
      "$sdk_root/gvsoc/install/bin/gvrun" \
        --target magia_v3 \
        --param binary="$sdk_build/bin/maps_mobilevit_slice" \
        --work-dir "$imiss_work" \
        --attr "magia_v3/n_tiles_x=$tiles" \
        --attr "magia_v3/n_tiles_y=$tiles" \
        --attr "magia_v3/spatz_romfile=$sdk_build/bin/bootrom/spatz_init.bin" \
        --trace-level=trace \
        --trace=kill-module \
        --vcd \
        '--event=.*snitch-spatz/event_imiss' \
        run | tee "$mesh_root/imiss.log"
      imiss_args=(--imiss-vcd "$imiss_work/all.vcd")
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
