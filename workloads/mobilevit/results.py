#!/usr/bin/env python3
"""Prepare MobileViT inputs and turn simulator logs into experiment CSV rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np
import onnx
from onnx import numpy_helper


ELEMENTS = 8192
SHAPE = (1, 128, 4, 16)
FIELDS = (
    "mesh",
    "strategy",
    "token",
    "completion_cycle",
    "completion_interval",
    "active_tiles",
    "validation_mismatches",
    "validation_nonfinite",
)
MAPS_TOKEN = re.compile(
    r"^MAPS_TOKEN tile=(\d+) token=(\d+) start=(\d+) end=(\d+) output=(\d+)$",
    re.MULTILINE,
)
REFERENCE_COMPLETION = re.compile(
    r"^REFERENCE_COMPLETION token=(\d+) cycle=(\d+)$", re.MULTILINE
)
REFERENCE_VALIDATION = re.compile(
    r"^REFERENCE_VALIDATION token=(\d+) mismatches=(\d+) nonfinite=(\d+)$",
    re.MULTILINE,
)
REFERENCE_TILES = re.compile(r"^REFERENCE_TILES count=(\d+)$", re.MULTILINE)
MAPS_CHECKSUM = re.compile(
    r"^maps_mobilevit_slice token (\d+) output .* checksum: ([0-9a-fA-F]{8})$",
    re.MULTILINE,
)


def initializer_bytes(model: onnx.ModelProto, name: str) -> bytes:
    value = next(item for item in model.graph.initializer if item.name == name)
    return numpy_helper.to_array(value).astype("<f2", copy=False).tobytes()


def prepare(args: argparse.Namespace) -> None:
    stored = np.fromfile(args.inputs, dtype="<f2")
    if stored.size % ELEMENTS:
        raise ValueError("stored input file does not contain complete tokens")
    available = stored.size // ELEMENTS
    if not 0 < args.tokens <= available:
        raise ValueError(f"requested {args.tokens} tokens; stored input has {available}")

    values = stored[: args.tokens * ELEMENTS].reshape((args.tokens, *SHAPE))
    args.output.mkdir(parents=True, exist_ok=True)
    runtime_inputs = args.output / "inputs.bin"
    values.tofile(runtime_inputs)

    model = onnx.load(args.model)
    input_name = model.graph.input[0].name
    references = np.zeros_like(values)
    references_available = False
    if args.references.is_file():
        try:
            stored_references = np.fromfile(args.references, dtype="<f2")
        except OSError:
            stored_references = np.empty(0, dtype="<f2")
        required = args.tokens * ELEMENTS
        if stored_references.size >= required:
            references = stored_references[:required].reshape(values.shape)
            references_available = True

    payload = b"".join(
        (
            runtime_inputs.read_bytes(),
            references.tobytes(),
            initializer_bytes(model, "_gn_fused_scale_15"),
            initializer_bytes(model, "_gn_fused_bias_15"),
            initializer_bytes(
                model,
                "base_module.stages.4.1.transformer.0.attn.qkv_proj.weight",
            ),
            initializer_bytes(
                model,
                "base_module.stages.4.1.transformer.0.attn.qkv_proj.bias",
            ),
            initializer_bytes(
                model,
                "base_module.stages.4.1.transformer.0.attn.out_proj.weight",
            ),
            initializer_bytes(
                model,
                "base_module.stages.4.1.transformer.0.attn.out_proj.bias",
            ),
        )
    )
    (args.output / "slice-data.bin").write_bytes(payload)
    (args.output / "input-name.txt").write_text(input_name, encoding="utf-8")
    checksum_lines = []
    if references_available:
        for token, reference in enumerate(references):
            checksum = 2166136261
            for byte in reference.tobytes():
                checksum = ((checksum ^ byte) * 16777619) & 0xFFFFFFFF
            checksum_lines.append(f"{token},{checksum:08x}")
    (args.output / "reference-checksums.csv").write_text(
        ("\n".join(checksum_lines) + "\n") if checksum_lines else "",
        encoding="utf-8",
    )
    (args.output / "reference-available.txt").write_text(
        "1\n" if references_available else "0\n", encoding="utf-8"
    )


def ordered(values: dict[int, int], tokens: int, label: str) -> list[int]:
    if set(values) != set(range(tokens)):
        raise ValueError(f"{label} token results are missing or duplicated")
    result = [values[token] for token in range(tokens)]
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"{label} completion cycles are not increasing")
    return result


def maps_rows(
    text: str, mesh: str, tokens: int, reference_checksums: Path
) -> list[dict[str, object]]:
    matches = [tuple(map(int, match)) for match in MAPS_TOKEN.findall(text)]
    tiles = {tile for tile, _, _, _, _ in matches}
    if not tiles:
        raise ValueError("MAPS log contains no token timing")
    expected = {(tile, token) for tile in tiles for token in range(tokens)}
    observed = {(tile, token) for tile, token, _, _, _ in matches}
    if len(matches) != len(expected) or observed != expected:
        raise ValueError("MAPS tile/token timing is incomplete or duplicated")
    if any(end <= start for _, _, start, end, _ in matches):
        raise ValueError("MAPS log contains an invalid token interval")

    baseline = min(start for _, _, start, _, _ in matches)
    completions = {
        token: max(
            end - baseline
            for _, item_token, _, end, output in matches
            if item_token == token and output
        )
        for token in range(tokens)
    }
    cycles = ordered(completions, tokens, "MAPS")

    overlap = any(
        left_tile != right_tile
        and left_token != right_token
        and left_start < right_end
        and right_start < left_end
        for left_tile, left_token, left_start, left_end, _ in matches
        for right_tile, right_token, right_start, right_end, _ in matches
    )
    if tokens > 1 and not overlap:
        raise ValueError("MAPS trace does not show cross-tile token overlap")

    expected = {
        int(token): checksum
        for token, checksum in csv.reader(
            reference_checksums.read_text(encoding="utf-8").splitlines()
        )
    }
    actual = {int(token): checksum.lower() for token, checksum in MAPS_CHECKSUM.findall(text)}
    validations = {
        token: (int(actual[token] != expected[token]), "")
        for token in range(tokens)
        if token in actual and token in expected
    }
    return rows(mesh, "maps", cycles, len(tiles), validations)


def reference_rows(
    text: str, mesh: str, tokens: int, references_available: bool
) -> list[dict[str, object]]:
    completion_matches = REFERENCE_COMPLETION.findall(text)
    if len(completion_matches) != tokens:
        raise ValueError("full-mesh completion results are missing or duplicated")
    cycles = ordered(
        {int(token): int(cycle) for token, cycle in completion_matches},
        tokens,
        "full-mesh",
    )

    validation_matches = REFERENCE_VALIDATION.findall(text)
    validations = {
        int(token): (int(mismatches), int(nonfinite))
        for token, mismatches, nonfinite in validation_matches
        if int(token) < tokens
    } if references_available else {}
    tile_counts = REFERENCE_TILES.findall(text)
    expected_tiles = int(mesh.split("x", maxsplit=1)[0]) ** 2
    if tile_counts != [str(expected_tiles)]:
        raise ValueError("full-mesh test did not use every physical tile")
    return rows(mesh, "full-mesh", cycles, expected_tiles, validations)


def rows(
    mesh: str,
    strategy: str,
    cycles: list[int],
    active_tiles: int,
    validations: dict[int, tuple[int | str, int | str]] | None,
) -> list[dict[str, object]]:
    result = []
    for token, cycle in enumerate(cycles):
        mismatch, nonfinite = (
            validations.get(token, ("", "")) if validations is not None else ("", "")
        )
        result.append(
            {
                "mesh": mesh,
                "strategy": strategy,
                "token": token,
                "completion_cycle": cycle,
                "completion_interval": "" if token == 0 else cycle - cycles[token - 1],
                "active_tiles": active_tiles,
                "validation_mismatches": mismatch,
                "validation_nonfinite": nonfinite,
            }
        )
    return result


def collect(args: argparse.Namespace) -> None:
    references_available = (
        args.reference_available.read_text(encoding="utf-8").strip() == "1"
    )
    result = maps_rows(
        args.maps_log.read_text(encoding="utf-8"),
        args.mesh,
        args.tokens,
        args.reference_checksums,
    )
    result.extend(
        reference_rows(
            args.full_mesh_log.read_text(encoding="utf-8"),
            args.mesh,
            args.tokens,
            references_available,
        )
    )
    write_header = not args.csv.exists()
    with args.csv.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(result)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--model", type=Path, required=True)
    prepare_parser.add_argument("--inputs", type=Path, required=True)
    prepare_parser.add_argument("--references", type=Path, required=True)
    prepare_parser.add_argument("--tokens", type=int, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.set_defaults(function=prepare)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--mesh", required=True)
    collect_parser.add_argument("--tokens", type=int, required=True)
    collect_parser.add_argument("--maps-log", type=Path, required=True)
    collect_parser.add_argument("--full-mesh-log", type=Path, required=True)
    collect_parser.add_argument("--reference-checksums", type=Path, required=True)
    collect_parser.add_argument("--reference-available", type=Path, required=True)
    collect_parser.add_argument("--csv", type=Path, required=True)
    collect_parser.set_defaults(function=collect)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
