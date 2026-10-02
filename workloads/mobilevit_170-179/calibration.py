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
    "launch_cycles",
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
    "OP_MATMUL": (581, 7.45),
    "OP_ADD": (69, 2.472),
    "OP_MUL": (56, 3.0),
    "OP_SUB": (47, 1.0),
    "OP_DIV": (47, 1.0),
    "OP_RELU": (805, 5.10),
    "OP_SOFTMAX_EXP": (69, 0.039),
    "OP_GROUP_REDUCE": (1_692, 6.63),
    "OP_GROUP_CENTERED_REDUCE": (1_685, 1 / 0.5234375),
    "OP_GROUP_NORMALIZE": (7_794, 4 / 3),
    "OP_REDUCE_SUM": (92, 0.125),
    "OP_REDUCE_MAX": (45, 0.05),
    "OP_IM2COL": (1_800, 4.0),
}
GEMM_LOOP_CYCLES = 5.658
SPATZ_FP16_VECTOR_ELEMENTS = 128
MUL_VECTOR_BLOCK_CYCLES = (18.5, 20.0)
MUL_VECTOR_ELEMENT_CYCLES = 0.256
BINARY_VECTOR_BLOCK_CYCLES = {"OP_SUB": 63.5, "OP_DIV": 63.5}
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
        element_cycles = (
            0.0
            if operation.kind != "OP_MUL" or scalar_broadcast
            else MUL_VECTOR_ELEMENT_CYCLES * operation.operation_count
        )
        return startup + ceil(block_cycles * vector_blocks + element_cycles)
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


def spatz_trace_timings(
    vcd_path: Path, measurements: list[dict[str, int | None]]
) -> dict[tuple[int, int, int, int, int], tuple[int, int | None]]:
    """Return instruction stalls and the Spatz task span for each operation.

    The outer CV32 operation window contains a placement-dependent launch path.
    The task span starts at Spatz's first retired instruction and ends at its IRQ
    exit. ``event_imiss`` overlap is then removed from that span.
    """
    windows: dict[int, list[dict[str, int | None]]] = {}
    for measurement in measurements:
        if measurement["start"] is None or measurement["end"] is None:
            continue
        windows.setdefault(int(measurement["tile"]), []).append(measurement)

    signals: dict[str, tuple[int, str]] = {}
    scope: list[str] = []
    period_signal: str | None = None
    period_ps: int | None = None
    now_ps = 0
    high_since: dict[str, int] = {}
    imiss_intervals: list[tuple[int, int, int]] = []
    first_instruction: dict[tuple[int, int, int, int, int], int] = {}
    irq_exit: dict[tuple[int, int, int, int, int], int] = {}

    def close_interval(signal: str, end_ps: int) -> None:
        begin_ps = high_since.pop(signal)
        tile, _ = signals[signal]
        imiss_intervals.append((tile, begin_ps, end_ps))

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
                elif fields[4] in {"event_imiss", "event_instr", "irq_exit"}:
                    match = re.search(r"tile-(\d+)-snitch-spatz", "/".join(scope))
                    if match:
                        signals[fields[3]] = (int(match.group(1)), fields[4])
            elif line.startswith("#"):
                now_ps = int(line[1:])
            elif line.startswith("b"):
                value, signal = line.split()
                if signal == period_signal and "z" not in value.lower():
                    period_ps = int(value[1:], 2)
            elif line[:1] in {"0", "1"}:
                value, signal = line[0], line[1:].strip()
                if signal not in signals:
                    continue
                tile, kind = signals[signal]
                if kind == "event_imiss":
                    if value == "1" and signal not in high_since:
                        high_since[signal] = now_ps
                    elif value == "0" and signal in high_since:
                        close_interval(signal, now_ps)
                elif value == "1" and period_ps is not None:
                    cycle = now_ps // period_ps
                    for measurement in windows.get(tile, ()):
                        start = int(measurement["start"])
                        end = int(measurement["end"])
                        if not start <= cycle <= end:
                            continue
                        key = (
                            tile,
                            int(measurement["token"]),
                            int(measurement["index"]),
                            start,
                            end,
                        )
                        if kind == "event_instr":
                            first_instruction.setdefault(key, now_ps)
                        else:
                            irq_exit[key] = now_ps

    for signal in tuple(high_since):
        close_interval(signal, now_ps)
    if period_ps is None or not signals:
        raise ValueError("VCD contains no Spatz trace signals")

    timings = {}
    for tile, tile_windows in windows.items():
        for measurement in tile_windows:
            key = (
                tile,
                int(measurement["token"]),
                int(measurement["index"]),
                int(measurement["start"]),
                int(measurement["end"]),
            )
            task_start = first_instruction.get(key)
            task_end = irq_exit.get(key)
            task_cycles = None
            if task_start is not None and task_end is not None and task_end >= task_start:
                task_cycles = (task_end - task_start) // period_ps
            overlap_start = task_start or int(measurement["start"]) * period_ps
            overlap_end = task_end or int(measurement["end"]) * period_ps
            fetch_ps = sum(
                max(0, min(end, overlap_end) - max(begin, overlap_start))
                for interval_tile, begin, end in imiss_intervals
                if interval_tile == tile
            )
            timings[key] = (fetch_ps // period_ps, task_cycles)
    return timings


def instruction_fetch_stalls(
    vcd_path: Path, measurements: list[dict[str, int | None]]
) -> dict[tuple[int, int, int, int, int], int]:
    return {
        key: fetch_cycles
        for key, (fetch_cycles, _) in spatz_trace_timings(
            vcd_path, measurements
        ).items()
    }


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
    trace_timings = spatz_trace_timings(imiss_vcd, measurements) if imiss_vcd else {}
    warm_tokens = {int(row["token"]) for row in measurements if int(row["token"]) > 0}
    selected_tokens = warm_tokens or {0}
    selected: dict[tuple[int, Operation], tuple[int, int, int, int, int, int]] = {}
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
        fetch_cycles, task_cycles = trace_timings.get(timing_key, (0, None))
        compute = operation.kind not in MOVEMENT_KINDS
        launch_cycles = 0
        if compute and task_cycles is not None:
            launch_cycles = raw_measured - task_cycles
            measured = task_cycles - fetch_cycles
        else:
            measured = raw_measured - fetch_cycles
        key = (token, operation)
        candidate = (tile, index, measured, raw_measured, launch_cycles, fetch_cycles)
        if key not in selected or (
            compute and measured < selected[key][2]
        ) or (
            not compute and measured > selected[key][2]
        ):
            selected[key] = candidate
    rows = []
    for (token, operation), measurement in selected.items():
        tile, index, measured, raw_measured, launch_cycles, fetch_cycles = measurement
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
                "launch_cycles": launch_cycles,
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
    parser.add_argument("--trace-vcd", "--imiss-vcd", dest="trace_vcd", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = calibration_rows(args.mesh, args.application, args.maps_log, args.trace_vcd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
