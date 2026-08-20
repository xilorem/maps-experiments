"""Prepare and parse the artifact-owned full-mesh MobileViT reference."""

from __future__ import annotations

import csv
from pathlib import Path
import re
import shutil

import onnx
from onnx import numpy_helper

from .mobilevit import REFERENCE_ATOL, REFERENCE_RTOL, TOKEN_COUNT
from .runtime import RuntimeResult


APPLICATION_NAME = "full_mesh_mobilevit_slice"
MESH_TILES = 16
_COMPLETION = re.compile(
    r"^REFERENCE_COMPLETION token=(\d+) cycle=(\d+)$", re.MULTILINE
)
_VALIDATION = re.compile(
    r"^REFERENCE_VALIDATION token=(\d+) mismatches=(\d+) nonfinite=(\d+)$",
    re.MULTILINE,
)
_TILES = re.compile(r"^REFERENCE_TILES count=(\d+)$", re.MULTILINE)


def _initializer_bytes(model: onnx.ModelProto, name: str) -> bytes:
    initializer = next(value for value in model.graph.initializer if value.name == name)
    return numpy_helper.to_array(initializer).astype("<f2", copy=False).tobytes()


def prepare_reference_application(
    source: Path,
    application: Path,
    model_path: Path,
    inputs: Path,
    references: Path,
) -> None:
    """Copy the frozen reference sources and supply this run's data payload."""

    shutil.copytree(source, application)
    model = onnx.load(model_path)
    payloads = (
        ("INPUTS", inputs.read_bytes()),
        ("REFERENCES", references.read_bytes()),
        (
            "GN_SCALE",
            _initializer_bytes(model, "_gn_fused_scale_15"),
        ),
        (
            "GN_BIAS",
            _initializer_bytes(model, "_gn_fused_bias_15"),
        ),
        (
            "QKV_WEIGHT",
            _initializer_bytes(
                model,
                "base_module.stages.4.1.transformer.0.attn.qkv_proj.weight",
            ),
        ),
        (
            "QKV_BIAS",
            _initializer_bytes(
                model,
                "base_module.stages.4.1.transformer.0.attn.qkv_proj.bias",
            ),
        ),
        (
            "OUT_WEIGHT",
            _initializer_bytes(
                model,
                "base_module.stages.4.1.transformer.0.attn.out_proj.weight",
            ),
        ),
        (
            "OUT_BIAS",
            _initializer_bytes(
                model,
                "base_module.stages.4.1.transformer.0.attn.out_proj.bias",
            ),
        ),
    )
    data = bytearray()
    offsets: dict[str, int] = {}
    for name, value in payloads:
        offsets[name] = len(data)
        data.extend(value)
    data_dir = application / "data"
    data_dir.mkdir()
    (data_dir / "reference-data.bin").write_bytes(data)
    definitions = [
        "#ifndef MOBILEVIT_REFERENCE_CONFIG_H_",
        "#define MOBILEVIT_REFERENCE_CONFIG_H_",
        f"#define REFERENCE_TOKEN_COUNT ({TOKEN_COUNT}u)",
        f"#define REFERENCE_ATOL ({REFERENCE_ATOL}f)",
        f"#define REFERENCE_RTOL ({REFERENCE_RTOL}f)",
    ]
    definitions.extend(
        f"#define REFERENCE_{name}_OFFSET ({offset}u)"
        for name, offset in offsets.items()
    )
    definitions.extend(("#endif", ""))
    (application / "include/reference_config.h").write_text(
        "\n".join(definitions), encoding="utf-8"
    )


def parse_reference_log(text: str, token_count: int = TOKEN_COUNT) -> RuntimeResult:
    """Reject an incomplete, unordered, partial-mesh, or invalid reference run."""

    completion_matches = _COMPLETION.findall(text)
    completions = {int(token): int(cycle) for token, cycle in completion_matches}
    if len(completion_matches) != token_count or set(completions) != set(range(token_count)):
        raise ValueError("reference completion events are missing or duplicated")
    cycles = tuple(completions[token] for token in range(token_count))
    if any(right <= left for left, right in zip(cycles, cycles[1:])):
        raise ValueError("reference completion events are out of order")

    validation_matches = _VALIDATION.findall(text)
    validations = {
        int(token): (int(mismatches), int(nonfinite))
        for token, mismatches, nonfinite in validation_matches
    }
    if len(validation_matches) != token_count or set(validations) != set(range(token_count)):
        raise ValueError("reference validation results are missing or duplicated")
    bad = {token: result for token, result in validations.items() if result != (0, 0)}
    if bad:
        raise ValueError(f"reference numerical validation failed: {bad}")
    tile_counts = [int(value) for value in _TILES.findall(text)]
    if tile_counts != [MESH_TILES]:
        raise ValueError("reference did not report the complete 4x4 Mesh")

    intervals = (None,) + tuple(
        right - left for left, right in zip(cycles, cycles[1:])
    )
    return RuntimeResult(cycles, intervals, ())


def write_comparison(
    output: Path,
    maps_result: RuntimeResult,
    reference_result: RuntimeResult,
    maps_active_tiles: int,
    token_slots: int,
) -> None:
    """Write one shared result set only after both strategies succeeded."""

    fields = (
        "strategy",
        "mesh_width",
        "mesh_height",
        "token_index",
        "completion_cycle",
        "completion_interval",
        "execution_tokens",
        "token_slots",
        "active_tiles",
    )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for strategy, result, slots, active_tiles in (
            ("maps", maps_result, token_slots, maps_active_tiles),
            ("full-mesh", reference_result, None, MESH_TILES),
        ):
            for token, (cycle, interval) in enumerate(
                zip(result.completion_cycles, result.completion_intervals)
            ):
                writer.writerow(
                    {
                        "strategy": strategy,
                        "mesh_width": 4,
                        "mesh_height": 4,
                        "token_index": token,
                        "completion_cycle": cycle,
                        "completion_interval": "" if interval is None else interval,
                        "execution_tokens": len(result.completion_cycles),
                        "token_slots": "" if slots is None else slots,
                        "active_tiles": active_tiles,
                    }
                )


def comparison_summary(
    maps_result: RuntimeResult, reference_result: RuntimeResult
) -> dict[str, dict[str, int]]:
    """Return directly inspectable latency, interval, and total-cycle values."""

    summary: dict[str, dict[str, int]] = {}
    for strategy, result in (
        ("maps", maps_result),
        ("full-mesh", reference_result),
    ):
        interval = result.completion_intervals[-1]
        if interval is None:
            raise ValueError("comparison needs at least two Execution Tokens")
        summary[strategy] = {
            "first_token_latency": result.completion_cycles[0],
            "completion_interval": interval,
            "total_cycles": result.completion_cycles[-1],
        }
    return summary
