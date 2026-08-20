"""Run the 4x4 pipelined MAPS MobileViT proof."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import onnx

from .dependencies import Dependencies, discover_dependencies
from .mobilevit import (
    INPUT_SEED,
    REFERENCE_ATOL,
    REFERENCE_RTOL,
    TOKEN_COUNT,
    TOKEN_SLOTS,
    extract_mobilevit_slice,
    generate_onnx_references,
    generate_runtime_inputs,
)
from .runtime import instrument_application, parse_runtime_log, write_measurements


APPLICATION_NAME = "maps_mobilevit_pipeline"


def _run(
    command: list[str],
    log: Path,
    *,
    cwd: Path | None = None,
    environ: dict[str, str] | None = None,
) -> None:
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        status = process.wait()
    if status != 0:
        raise subprocess.CalledProcessError(status, command)


def _revision(repository: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {"path": str(repository), "commit": commit, "dirty": dirty}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _toolchain_provenance(root: Path) -> dict[str, object]:
    tools = ("clang", "llvm-objcopy", "llvm-objdump", "ld.lld")
    binaries = {tool: root / "bin" / tool for tool in tools}
    clang_version = subprocess.run(
        [str(binaries["clang"]), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    return {
        "path": str(root),
        "clang_version": clang_version,
        "tool_sha256": {
            tool: _sha256(binary) for tool, binary in binaries.items()
        },
    }


def _build_application(
    dependencies: Dependencies,
    model: Path,
    inputs: Path,
    application: Path,
    log: Path,
) -> None:
    input_name = onnx.load(model).graph.input[0].name
    python_environment = os.environ.copy()
    python_environment["PYTHONPATH"] = str(dependencies.maps_root)
    _run(
        [
            sys.executable,
            "-m",
            "maps.cli",
            "build",
            str(model),
            "--target",
            "magia-v3",
            "--mesh",
            "4x4",
            "--token-slots",
            str(TOKEN_SLOTS),
            "--name",
            APPLICATION_NAME,
            "--input",
            f"{input_name}={inputs}",
            "--output",
            str(application),
        ],
        log,
        cwd=dependencies.maps_root,
        environ=python_environment,
    )


def _build_in_sdk(
    dependencies: Dependencies,
    application: Path,
    build: Path,
    log: Path,
) -> Path:
    llvm = dependencies.spatz_llvm_root / "bin"
    _run(
        [
            "cmake",
            "-S",
            str(dependencies.magia_sdk_root),
            "-B",
            str(build),
            "-DTARGET_PLATFORM=magia_v3",
            "-DTILES=4",
            "-DPULP_CORE_COUNT=8",
            "-DCOMPILER=GCC_PULP",
            "-DUSE_CCACHE=OFF",
            "-DSPATZ_TESTS=ON",
            f"-DSPATZ_LLVM_PATH={dependencies.spatz_llvm_root}",
            f"-DSPATZ_CLANG={llvm / 'clang'}",
            f"-DSPATZ_OBJCOPY={llvm / 'llvm-objcopy'}",
            f"-DSPATZ_OBJDUMP={llvm / 'llvm-objdump'}",
            f"-DMAPS_APPLICATION_DIR={application}",
        ],
        log,
    )
    _run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "spatz_bootrom",
            APPLICATION_NAME,
            "-j",
            "8",
        ],
        log,
    )
    return build / f"bin/{APPLICATION_NAME}"


def _simulate(
    dependencies: Dependencies,
    executable: Path,
    bootrom: Path,
    work: Path,
    log: Path,
) -> str:
    simulation_log = work.parent / "simulation.log"
    _run(
        [
            str(dependencies.magia_sdk_root / "gvsoc/install/bin/gvrun"),
            "--target",
            "magia_v3",
            "--param",
            f"binary={executable}",
            "--work-dir",
            str(work),
            "--attr",
            "magia_v3/n_tiles_x=4",
            "--attr",
            "magia_v3/n_tiles_y=4",
            "--attr",
            "magia_v3/nb_pulp_cores=8",
            "--attr",
            f"magia_v3/spatz_romfile={bootrom}",
            "run",
        ],
        simulation_log,
    )
    log.write_text(
        log.read_text(encoding="utf-8")
        + "\n"
        + simulation_log.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return simulation_log.read_text(encoding="utf-8")


def run() -> Path:
    artifact_root = Path(
        os.environ.get("MAPS_EXPERIMENTS_ROOT", Path.cwd())
    ).resolve()
    dependencies = discover_dependencies(artifact_root, os.environ)
    model = artifact_root / "workloads/mobilevit/model/mobilevit-nodes-170-179.onnx"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    result_dir = artifact_root / f"workloads/mobilevit/results/{timestamp}-maps-4x4"
    work = result_dir / "work"
    work.mkdir(parents=True)
    log = result_dir / "run.log"
    log.write_text("", encoding="utf-8")

    reproduced_model = work / "mobilevit-nodes-170-179.onnx"
    extract_mobilevit_slice(
        dependencies.maps_root / "examples/mobilenet.onnx",
        reproduced_model,
    )
    if reproduced_model.read_bytes() != model.read_bytes():
        raise RuntimeError("frozen MobileViT slice differs from reproducible extraction")

    inputs = generate_runtime_inputs(work / "runtime-inputs.bin")
    references = generate_onnx_references(
        model,
        inputs,
        work / "onnx-runtime-references.bin",
    )
    application = work / "application"
    _build_application(dependencies, model, inputs, application, log)
    application_facts = instrument_application(application, references)
    build = artifact_root / "workloads/mobilevit/.build/maps-4x4"
    executable = _build_in_sdk(dependencies, application, build, log)
    simulation_text = _simulate(
        dependencies,
        executable,
        build / "bin/bootrom/spatz_init.bin",
        work / "gvsoc-work",
        log,
    )
    runtime_result = parse_runtime_log(
        simulation_text,
        TOKEN_COUNT,
        application_facts["active_tile_ids"],
    )
    write_measurements(
        result_dir / "measurements.csv",
        runtime_result,
        application_facts["active_tiles"],
        TOKEN_SLOTS,
    )
    metadata = {
        "schema": 1,
        "experiment": "maps-mobilevit-pipeline",
        "strategy": "maps",
        "mesh": {"width": 4, "height": 4},
        "execution_tokens": TOKEN_COUNT,
        "token_slots": TOKEN_SLOTS,
        "active_tiles": application_facts["active_tiles"],
        "output_writers": application_facts["output_writers"],
        "input_seed": INPUT_SEED,
        "validation": {"atol": REFERENCE_ATOL, "rtol": REFERENCE_RTOL},
        "measurement_window": {
            "start": "common post-input, post-initializer global boundary",
            "completion": "last globally required output writer for each token",
            "excluded": [
                "build",
                "boot",
                "initialization",
                "Runtime Input loading",
                "trace printing",
                "validation",
            ],
        },
        "artifacts": {
            "model_sha256": _sha256(model),
            "runtime_inputs_sha256": _sha256(inputs),
            "onnx_runtime_references_sha256": _sha256(references),
        },
        "dependencies": {
            "artifact": _revision(artifact_root),
            "maps": _revision(dependencies.maps_root),
            "magia_sdk": _revision(dependencies.magia_sdk_root),
            "spatz_llvm": _toolchain_provenance(dependencies.spatz_llvm_root),
        },
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Validated MAPS MobileViT pipeline result: {result_dir}")
    return result_dir


def main() -> int:
    try:
        run()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"experiment failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
