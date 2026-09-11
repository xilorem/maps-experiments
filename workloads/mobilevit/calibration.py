#!/usr/bin/env python3
"""Compare MAGIA-v3 kernel timings with the MAPS calibration formulas."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import ceil, prod
from pathlib import Path
import re


FIELDS = (
    "mesh",
    "tile",
    "token",
    "operation_index",
    "operation_kind",
    "input_elements",
    "output_elements",
    "operation_count",
    "dominance",
    "raw_measured_cycles",
    "instruction_fetch_cycles",
    "measured_cycles",
    "predicted_cycles",
    "absolute_error",
    "relative_error_percent",
)
MAPS_OPERATION = re.compile(
    r"^maps t(?P<tile>\d+) tok (?P<token>\d+) slot \d+ op "
    r"(?P<index>\d+)(?: start (?P<start>\d+) end (?P<end>\d+))? "
    r"cycles (?P<cycles>\d+)$",
    re.MULTILINE,
)
LINEAR_MODELS = {
    "OP_MATMUL": (3_500, 7.45),
    "OP_ADD": (2_400, 2.25),
    "OP_MUL": (2_200, 3.0),
    "OP_SUB": (2_818, 0.196),
    "OP_DIV": (3_156, 0.1135),
    "OP_RELU": (2_200, 5.10),
    "OP_SOFTMAX_EXP": (2_500, 0.107),
    "OP_GROUP_REDUCE": (2_100, 4.0),
    "OP_GROUP_CENTERED_REDUCE": (2_100, 1.484),
    "OP_GROUP_NORMALIZE": (2_500, 0.583),
    "OP_REDUCE_SUM": (2_060, 1 / 7.4),
    "OP_REDUCE_MAX": (2_200, 0.0922),
    "OP_IM2COL": (1_800, 4.0),
}
GEMM_LOOP_CYCLES = 5.8
SPATZ_FP16_VECTOR_ELEMENTS = 128
MUL_VECTOR_BLOCK_CYCLES = (37.6, 20.3)
BINARY_VECTOR_BLOCK_CYCLES = {"OP_SUB": 77.33, "OP_DIV": 52.0}
BROADCAST_SCALAR_ELEMENT_CYCLES = {
    "OP_MUL": 17.38,
    "OP_SUB": 575.3,
    "OP_DIV": 581.6,
}
REDUCTION_OUTPUT_CYCLES = 33.5
MOVEMENT_KINDS = frozenset(
    {"OP_IM2COL", "OP_ALL_REDUCE_SUM", "OP_ALL_REDUCE_MAX"}
)


@dataclass(frozen=True)
class Operation:
    kind: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    params: tuple[int, ...]
    participants: int

    @property
    def input_elements(self) -> tuple[int, ...]:
        return tuple(prod(shape) for shape in self.input_shapes)

    @property
    def output_elements(self) -> tuple[int, ...]:
        return tuple(prod(shape) for shape in self.output_shapes)

    @property
    def operation_count(self) -> int:
        if self.kind == "OP_MATMUL":
            batch = self.params[3] if len(self.params) > 3 else 1
            return prod(self.params[:3]) * batch
        if self.kind in {
            "OP_GROUP_REDUCE",
            "OP_GROUP_CENTERED_REDUCE",
            "OP_REDUCE_SUM",
            "OP_REDUCE_MAX",
        }:
            return self.input_elements[0]
        return self.output_elements[0]


def array_entries(source: str, declaration: str) -> list[str]:
    match = re.search(declaration, source)
    if not match:
        return []
    start = source.find("{", match.end())
    depth = 0
    entries = []
    entry_start = None
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
            if depth == 2:
                entry_start = index
        elif source[index] == "}":
            if depth == 2 and entry_start is not None:
                entries.append(source[entry_start : index + 1])
                entry_start = None
            depth -= 1
            if depth == 0:
                return entries
    raise ValueError(f"unterminated generated {declaration} array")


def _shape(payload: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in re.findall(r"(\d+)u?", payload))
    return tuple(value for value in values if value != 0)


def generated_operations(application: Path) -> dict[tuple[int, int], Operation]:
    operations = {}
    for path in sorted((application / "src" / "tiles").glob("tile_*.c")):
        source = path.read_text(encoding="utf-8")
        hartids = re.findall(r"\.hartid = (\d+)u", source)
        if not hartids:
            continue
        tile = int(hartids[-1])
        entries = array_entries(source, r"static const op_desc_t .*_ops\[\] =")
        for index, entry in enumerate(entries):
            kind = re.search(r"\.kind = (OP_[A-Z0-9_]+)", entry)
            num_inputs = re.search(r"\.num_inputs = (\d+)u", entry)
            num_outputs = re.search(r"\.num_outputs = (\d+)u", entry)
            params = re.search(r"\.params = \{([^}]*)\}", entry)
            participants = re.search(r"\.num_participants = (\d+)u", entry)
            if not kind or not num_inputs or not num_outputs:
                raise ValueError(f"incomplete operation descriptor in {path}")
            shapes = tuple(
                _shape(match.group(1))
                for match in re.finditer(r"\.shape = \{([^}]*)\}", entry)
            )
            inputs = int(num_inputs.group(1))
            outputs = int(num_outputs.group(1))
            if len(shapes) != inputs + outputs:
                raise ValueError(f"operation shape count disagrees in {path}")
            operations[tile, index] = Operation(
                kind.group(1),
                shapes[:inputs],
                shapes[inputs:],
                tuple(int(value) for value in re.findall(r"(\d+)u?", params.group(1)))
                if params
                else (),
                int(participants.group(1)) if participants else 0,
            )
    return operations


def predicted_cycles(operation: Operation) -> int | None:
    if operation.kind in {"OP_ALL_REDUCE_SUM", "OP_ALL_REDUCE_MAX"}:
        if operation.participants <= 1:
            return 650
        elements = operation.output_elements[0]
        return 650 + ceil(elements * 2 * (operation.participants - 1))
    model = LINEAR_MODELS.get(operation.kind)
    if model is None:
        return None
    startup, throughput = model
    if operation.kind == "OP_MATMUL":
        m_size, k_size, n_size = operation.params[:3]
        batch_size = operation.params[3] if len(operation.params) > 3 else 1
        vector_blocks = ceil(n_size / SPATZ_FP16_VECTOR_ELEMENTS)
        loop_work = batch_size * m_size * k_size * vector_blocks
        return startup + ceil(
            GEMM_LOOP_CYCLES * loop_work
            + operation.operation_count / throughput
        )
    elif operation.kind in {"OP_MUL", "OP_SUB", "OP_DIV"}:
        rows, row_len, scalar_broadcast = broadcast_geometry(operation)
        if row_len % 2 != 0:
            return startup + ceil(
                BROADCAST_SCALAR_ELEMENT_CYCLES[operation.kind]
                * operation.operation_count
            )
        vector_blocks = rows * ceil(row_len / SPATZ_FP16_VECTOR_ELEMENTS)
        block_cycles = (
            MUL_VECTOR_BLOCK_CYCLES[scalar_broadcast]
            if operation.kind == "OP_MUL"
            else BINARY_VECTOR_BLOCK_CYCLES[operation.kind]
        )
        return startup + ceil(block_cycles * vector_blocks)
    elif operation.kind == "OP_REDUCE_SUM":
        return startup + ceil(
            operation.operation_count / throughput
            + REDUCTION_OUTPUT_CYCLES * operation.output_elements[0]
        )
    return startup + ceil(operation.operation_count / throughput)


def broadcast_geometry(operation: Operation) -> tuple[int, int, bool]:
    output_shape = operation.output_shapes[0]
    output_elements = operation.output_elements[0]
    broadcasts = tuple(
        (shape, elements)
        for shape, elements in zip(operation.input_shapes, operation.input_elements)
        if elements < output_elements
    )
    if not broadcasts:
        return 1, output_elements, False

    broadcast_shape, broadcast_elements = broadcasts[0]
    if broadcast_shape[-1] == 1 and broadcast_shape[:-1] == output_shape[:-1]:
        return broadcast_elements, output_shape[-1], True

    row_len = 1
    for broadcast_dim, output_dim in zip(
        reversed(broadcast_shape), reversed(output_shape)
    ):
        if broadcast_dim != output_dim:
            break
        row_len *= output_dim
    return output_elements // row_len, row_len, False


def instruction_fetch_stalls(
    vcd_path: Path, measurements: list[dict[str, int | None]]
) -> dict[tuple[int, int, int, int, int], int]:
    """Return Spatz instruction-fetch stall cycles overlapping each operation.

    GVSoC's Spatz ``event_imiss`` is high while the scalar core waits for an
    instruction refill. Operation timestamps and the VCD share the tile clock,
    so overlap can be subtracted without assigning NoC delay to kernel compute.
    """
    windows: dict[int, list[dict[str, int | None]]] = {}
    for measurement in measurements:
        if measurement["start"] is None or measurement["end"] is None:
            continue
        windows.setdefault(int(measurement["tile"]), []).append(measurement)

    signal_tiles: dict[str, int] = {}
    scope: list[str] = []
    period_signal: str | None = None
    period_ps: int | None = None
    now_ps = 0
    high_since: dict[str, int] = {}
    overlap_ps: dict[tuple[int, int, int, int, int], int] = {}

    def close_interval(signal: str, end_ps: int) -> None:
        begin_ps = high_since.pop(signal)
        tile = signal_tiles[signal]
        if period_ps is None:
            raise ValueError("VCD does not define the tile clock period")
        for measurement in windows.get(tile, ()):
            start = int(measurement["start"]) * period_ps
            end = int(measurement["end"]) * period_ps
            overlap = max(0, min(end_ps, end) - max(begin_ps, start))
            if overlap:
                key = (
                    tile,
                    int(measurement["token"]),
                    int(measurement["index"]),
                    int(measurement["start"]),
                    int(measurement["end"]),
                )
                overlap_ps[key] = overlap_ps.get(key, 0) + overlap

    with vcd_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("$scope module "):
                scope.append(line.split()[2])
            elif line.startswith("$upscope"):
                scope.pop()
            elif line.startswith("$var"):
                fields = line.split()
                if fields[4] == "period" and "tile-clock" in scope:
                    period_signal = fields[3]
                elif fields[4] == "event_imiss":
                    match = re.search(r"tile-(\d+)-snitch-spatz", "/".join(scope))
                    if match:
                        signal_tiles[fields[3]] = int(match.group(1))
            elif line.startswith("#"):
                now_ps = int(line[1:])
            elif line.startswith("b"):
                value, signal = line.split()
                if signal == period_signal and "z" not in value.lower():
                    period_ps = int(value[1:], 2)
            elif line[:1] in {"0", "1"}:
                value, signal = line[0], line[1:].strip()
                if signal not in signal_tiles:
                    continue
                if value == "1" and signal not in high_since:
                    high_since[signal] = now_ps
                elif value == "0" and signal in high_since:
                    close_interval(signal, now_ps)

    for signal in tuple(high_since):
        close_interval(signal, now_ps)
    if period_ps is None or not signal_tiles:
        raise ValueError("VCD contains no Spatz instruction-fetch stall signals")
    return {key: value // period_ps for key, value in overlap_ps.items()}


def calibration_rows(
    mesh: str,
    application: Path,
    maps_log: Path,
    imiss_vcd: Path | None = None,
) -> list[dict[str, object]]:
    dimensions = tuple(int(value) for value in mesh.lower().split("x"))
    if len(dimensions) != 2 or any(value <= 0 for value in dimensions):
        raise ValueError("mesh must have WIDTHxHEIGHT form")
    operations = generated_operations(application)
    measurements = [
        {
            key: int(value) if value is not None else None
            for key, value in match.groupdict().items()
        }
        for match in MAPS_OPERATION.finditer(maps_log.read_text(encoding="utf-8"))
    ]
    if not measurements:
        raise ValueError("MAPS log contains no operation timings")
    fetch_stalls = instruction_fetch_stalls(imiss_vcd, measurements) if imiss_vcd else {}
    warm_tokens = {int(row["token"]) for row in measurements if int(row["token"]) > 0}
    selected_tokens = warm_tokens or {0}
    slowest: dict[tuple[int, Operation], tuple[int, int, int, int, int]] = {}
    for measurement in measurements:
        token = int(measurement["token"])
        if token not in selected_tokens:
            continue
        if imiss_vcd and (measurement["start"] is None or measurement["end"] is None):
            raise ValueError("instruction-fetch adjustment requires operation start/end cycles")
        tile = int(measurement["tile"])
        index = int(measurement["index"])
        operation = operations[tile, index]
        raw_measured = int(measurement["cycles"])
        timing_key = (
            tile,
            token,
            index,
            int(measurement["start"] or 0),
            int(measurement["end"] or 0),
        )
        fetch_cycles = fetch_stalls.get(timing_key, 0)
        measured = raw_measured - fetch_cycles
        key = (token, operation)
        if key not in slowest or measured > slowest[key][2]:
            slowest[key] = (tile, index, measured, raw_measured, fetch_cycles)
    rows = []
    for (token, operation), measurement in slowest.items():
        tile, index, measured, raw_measured, fetch_cycles = measurement
        predicted = predicted_cycles(operation)
        if predicted is None:
            continue
        error = predicted - measured
        rows.append(
            {
                "mesh": mesh,
                "tile": tile,
                "token": token,
                "operation_index": index,
                "operation_kind": operation.kind,
                "input_elements": ";".join(map(str, operation.input_elements)),
                "output_elements": ";".join(map(str, operation.output_elements)),
                "operation_count": operation.operation_count,
                "dominance": "movement" if operation.kind in MOVEMENT_KINDS else "compute",
                "raw_measured_cycles": raw_measured,
                "instruction_fetch_cycles": fetch_cycles,
                "measured_cycles": measured,
                "predicted_cycles": predicted,
                "absolute_error": abs(error),
                "relative_error_percent": f"{100.0 * abs(error) / measured:.2f}",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--maps-log", type=Path, required=True)
    parser.add_argument("--imiss-vcd", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = calibration_rows(args.mesh, args.application, args.maps_log, args.imiss_vcd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
