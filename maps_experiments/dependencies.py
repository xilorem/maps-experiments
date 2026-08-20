"""Dependency discovery for reproducible MAPS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Mapping


@dataclass(frozen=True)
class Dependencies:
    maps_root: Path
    magia_sdk_root: Path
    spatz_llvm_root: Path


def _validate_gvsoc(sdk: Path) -> None:
    gvrun = sdk / "gvsoc/install/bin/gvrun"
    model_dir = sdk / "gvsoc/install/models"
    with tempfile.TemporaryDirectory(prefix="maps-gvsoc-preflight-") as directory:
        temporary = Path(directory)
        component_file = temporary / "components.config"
        command = [
            str(gvrun),
            "--target",
            "magia_v3",
            "--work-dir",
            str(temporary / "work"),
            "--attr",
            "magia_v3/n_tiles_x=4",
            "--attr",
            "magia_v3/n_tiles_y=4",
            "--attr",
            "magia_v3/nb_pulp_cores=8",
            f"--builddir={temporary / 'build'}",
            f"--installdir={model_dir}",
            f"--component-file={component_file}",
            "components",
            "--py-stack",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not component_file.is_file():
            detail = (result.stderr or result.stdout).strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise ValueError(f"MAGIA-v3 GVSoC target generation failed{suffix}")
        match = re.search(
            r"^CONFIG_COMPONENTS=(.+)$",
            component_file.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if match is None:
            raise ValueError("MAGIA-v3 GVSoC target has no component manifest")
        missing = [
            component
            for component in match.group(1).split()
            if not (model_dir / f"{component}.so").is_file()
        ]
        if missing:
            raise ValueError(
                "MAGIA-v3 GVSoC installation is stale; missing models: "
                + ", ".join(missing)
            )


def _spatz_llvm(environ: Mapping[str, str], sdk: Path) -> Path:
    configured = environ.get("LLVM_INSTALL_DIR")
    candidates = []
    if configured is not None:
        candidates.append(Path(configured).expanduser())
    candidates.append(sdk / "llvm/install")
    clang = shutil.which("clang")
    if clang is not None:
        candidates.append(Path(clang).resolve().parent.parent)
    required = ("clang", "llvm-objcopy", "llvm-objdump", "ld.lld")
    for candidate in candidates:
        root = candidate.resolve()
        if all((root / "bin" / tool).is_file() for tool in required):
            return root
    raise ValueError(
        "LLVM_INSTALL_DIR does not identify a complete Spatz LLVM toolchain "
        f"(required tools: {', '.join(required)})"
    )


def _checkout(
    artifact_root: Path,
    environ: Mapping[str, str],
    variable: str,
    conventional_name: str,
    marker: str,
) -> Path:
    configured = environ.get(variable)
    path = (
        Path(configured).expanduser()
        if configured is not None
        else artifact_root.resolve().parent / conventional_name
    ).resolve()
    if not path.is_dir() or not (path / marker).is_file():
        raise ValueError(
            f"{variable} does not identify a valid checkout: {path} "
            f"(missing {marker})"
        )
    return path


def discover_dependencies(
    artifact_root: Path,
    environ: Mapping[str, str],
) -> Dependencies:
    """Resolve sibling checkouts and reject invalid locations immediately."""

    maps_root = _checkout(
        artifact_root, environ, "MAPS_ROOT", "MAPS", "pyproject.toml"
    )
    sdk = _checkout(
        artifact_root,
        environ,
        "MAGIA_SDK_ROOT",
        "magia-sdk-v3",
        "CMakeLists.txt",
    )
    dependencies = Dependencies(
        maps_root=maps_root,
        magia_sdk_root=sdk,
        spatz_llvm_root=_spatz_llvm(environ, sdk),
    )
    required_files = (
        dependencies.maps_root
        / "maps-ir/build/tools/maps-translate/maps-plan-import",
        dependencies.maps_root
        / "maps-ir/build/tools/maps-translate/maps-codegen",
        dependencies.magia_sdk_root / "gvsoc/install/bin/gvrun",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(
        command
        for command in ("cmake", "riscv32-unknown-elf-gcc")
        if shutil.which(command) is None
    )
    if missing:
        raise ValueError(
            "required build or simulation dependencies are missing: "
            + ", ".join(missing)
        )
    _validate_gvsoc(sdk)
    return dependencies
