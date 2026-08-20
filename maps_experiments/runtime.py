"""Instrument and parse one measured MAPS pipeline execution."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .mobilevit import REFERENCE_ATOL, REFERENCE_RTOL


_TRACE = re.compile(
    r"^MAPS_TRACE tile=(\d+) token=(\d+) start=(\d+) end=(\d+)$",
    re.MULTILINE,
)
_COMPLETION = re.compile(
    r"^MAPS_COMPLETION token=(\d+) cycle=(\d+)$",
    re.MULTILINE,
)
_VALIDATION = re.compile(
    r"^MAPS_VALIDATION token=(\d+) mismatches=(\d+) nonfinite=(\d+)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class TileTokenInterval:
    tile: int
    token: int
    start: int
    end: int


@dataclass(frozen=True)
class RuntimeResult:
    completion_cycles: tuple[int, ...]
    completion_intervals: tuple[int | None, ...]
    tile_token_intervals: tuple[TileTokenInterval, ...]


def _c_source(
    application_name: str,
    input_name: str,
    output_name: str,
    tensor_bytes: int,
    tensor_elements: int,
    token_count: int,
    mesh_tiles: int,
) -> str:
    return f'''/* MAPS MobileViT paper-experiment instrumentation. */
#include "{application_name}.h"

#include <stddef.h>
#include "utils/maps_operations.h"
#include "utils/printf.h"

extern const uint8_t {application_name}_input_{input_name}_start[];
extern const uint16_t maps_experiment_reference_start[];

volatile uint32_t maps_experiment_window_start __attribute__((section(".l2_bulk.maps_experiment")));
volatile uint32_t maps_experiment_completion_cycles[{token_count}] __attribute__((section(".l2_bulk.maps_experiment")));
volatile uint32_t maps_experiment_trace_start[{mesh_tiles}][{token_count}] __attribute__((section(".l2_bulk.maps_experiment")));
volatile uint32_t maps_experiment_trace_end[{mesh_tiles}][{token_count}] __attribute__((section(".l2_bulk.maps_experiment")));
volatile uint32_t maps_experiment_trace_valid[{mesh_tiles}] __attribute__((section(".l2_bulk.maps_experiment")));
volatile uint32_t maps_experiment_output_cycle[{mesh_tiles}][{token_count}] __attribute__((section(".l2_bulk.maps_experiment")));
volatile uint32_t maps_experiment_output_writer_valid[{mesh_tiles}] __attribute__((section(".l2_bulk.maps_experiment")));
volatile uint32_t maps_experiment_validation_failures __attribute__((section(".l2_bulk.maps_experiment")));

void {application_name}_handle_input(uint32_t token) {{
  uint8_t *input = {application_name}_input_{input_name}(token);
  const uint8_t *data = {application_name}_input_{input_name}_start;
  for (size_t index = 0; index < {tensor_bytes}u; ++index)
    input[index] = data[index + token * {tensor_bytes}u];
}}

void {application_name}_handle_output(uint32_t token) {{
  const uint16_t *actual = (const uint16_t *){application_name}_output_{output_name}(token);
  const uint16_t *reference = maps_experiment_reference_start + token * {tensor_elements}u;
  uint32_t mismatches = 0u;
  uint32_t nonfinite = 0u;
  for (size_t index = 0; index < {tensor_elements}u; ++index) {{
    const uint16_t actual_bits = actual[index];
    const float actual_value = maps_operation_f16_to_f32(actual_bits);
    const float expected = maps_operation_f16_to_f32(reference[index]);
    const float difference = actual_value > expected
        ? actual_value - expected : expected - actual_value;
    const float expected_absolute = expected < 0.0f ? -expected : expected;
    if ((actual_bits & 0x7c00u) == 0x7c00u)
      ++nonfinite;
    if (difference > {REFERENCE_ATOL}f + {REFERENCE_RTOL}f * expected_absolute)
      ++mismatches;
  }}
  if (mismatches != 0u || nonfinite != 0u)
    ++maps_experiment_validation_failures;
  printf("MAPS_VALIDATION token=%u mismatches=%u nonfinite=%u\\n",
         token, mismatches, nonfinite);
}}
'''


def _instrument_runner(
    source: str,
    application_name: str,
    token_count: int,
    mesh_tiles: int,
    output_writers: int,
) -> str:
    upper_name = application_name.upper()
    declarations = f'''
static inline uint32_t maps_experiment_read_cycle(void) {{
  uint32_t cycle;
  __asm__ volatile("rdcycle %0" : "=r"(cycle));
  return cycle;
}}

extern volatile uint32_t maps_experiment_window_start;
extern volatile uint32_t maps_experiment_completion_cycles[{token_count}];
extern volatile uint32_t maps_experiment_trace_start[{mesh_tiles}][{token_count}];
extern volatile uint32_t maps_experiment_trace_end[{mesh_tiles}][{token_count}];
extern volatile uint32_t maps_experiment_trace_valid[{mesh_tiles}];
extern volatile uint32_t maps_experiment_output_cycle[{mesh_tiles}][{token_count}];
extern volatile uint32_t maps_experiment_output_writer_valid[{mesh_tiles}];
extern volatile uint32_t maps_experiment_validation_failures;
'''
    source = source.replace(
        '#include "utils/maps_utils_v2.h"\n',
        '#include "utils/maps_utils_v2.h"\n' + declarations,
        1,
    )
    original = re.compile(
        r"  if \(active\)\n"
        r"    maps_fifo_run_tile_tokens\(\n"
        r"        &plan, [A-Z0-9_]+_EXECUTION_TOKENS, &idma, &event_unit\);"
    )
    replacement = f'''  if (hartid == 0u)
    maps_experiment_window_start = maps_experiment_read_cycle();
  fsync_sync_global(&fsync);
  eu_fsync_wait(&event_unit, {upper_name}_WAIT_MODE);
  if (active) {{
    maps_experiment_trace_valid[hartid] = 1u;
    for (uint32_t token = 0u; token < {token_count}u; ++token) {{
      maps_experiment_trace_start[hartid][token] =
          maps_experiment_read_cycle() - maps_experiment_window_start;
      maps_fifo_run_tile_token(&plan, token, &idma, &event_unit);
      maps_experiment_trace_end[hartid][token] =
          maps_experiment_read_cycle() - maps_experiment_window_start;
      if (plan.num_l2_writes != 0u) {{
        maps_experiment_output_writer_valid[hartid] = 1u;
        maps_experiment_output_cycle[hartid][token] =
            maps_experiment_trace_end[hartid][token];
      }}
    }}
  }}'''
    source, replacements = original.subn(replacement, source, count=1)
    if replacements != 1:
        raise ValueError("generated runner does not contain the expected token loop")

    output_loop = re.compile(
        rf"  if \(hartid == 0u\) \{{\n"
        rf"    for \(uint32_t token = 0u; token < [A-Z0-9_]+_EXECUTION_TOKENS; \+\+token\)\n"
        rf"      {re.escape(application_name)}_handle_output\(token\);\n"
        rf"  \}}\n  return 0;"
    )
    reporting = f'''  if (hartid == 0u) {{
    printf("MAPS_WINDOW_START cycle=%u\\n", maps_experiment_window_start);
    for (uint32_t token = 0u; token < {token_count}u; ++token) {{
      uint32_t completion = 0u;
      uint32_t writers = 0u;
      for (uint32_t tile = 0u; tile < {mesh_tiles}u; ++tile) {{
        if (maps_experiment_output_writer_valid[tile] == 0u)
          continue;
        ++writers;
        const uint32_t cycle = maps_experiment_output_cycle[tile][token];
        if (cycle > completion)
          completion = cycle;
      }}
      if (writers != {output_writers}u)
        return -1;
      maps_experiment_completion_cycles[token] = completion;
      printf("MAPS_COMPLETION token=%u cycle=%u\\n",
             token, completion);
    }}
    for (uint32_t tile = 0u; tile < {mesh_tiles}u; ++tile)
      if (maps_experiment_trace_valid[tile] != 0u)
        for (uint32_t token = 0u; token < {token_count}u; ++token)
          printf("MAPS_TRACE tile=%u token=%u start=%u end=%u\\n",
                 tile, token, maps_experiment_trace_start[tile][token],
                 maps_experiment_trace_end[tile][token]);
    for (uint32_t token = 0u; token < {token_count}u; ++token)
      {application_name}_handle_output(token);
    if (maps_experiment_validation_failures != 0u)
      return -1;
  }}
  return 0;'''
    source, replacements = output_loop.subn(lambda _: reporting, source, count=1)
    if replacements != 1:
        raise ValueError("generated runner does not contain the expected output loop")
    return source


def instrument_application(application: Path, reference: Path) -> dict[str, Any]:
    """Add cycle/overlap recording and target-side numerical validation."""

    manifest = json.loads((application / "manifest.json").read_text(encoding="utf-8"))
    name = manifest["application"]["name"]
    token_count = manifest["execution"]["tokens"]
    mesh = manifest["planned_mesh"]
    mesh_tiles = mesh["width"] * mesh["height"]
    input_tensor = manifest["tensors"]["inputs"][0]
    output_tensor = manifest["tensors"]["outputs"][0]
    output_writers = sum(
        "GLOBAL_OUTPUT" in path.read_text(encoding="utf-8")
        for path in (application / "src/tiles").glob("*.c")
    )
    if output_writers <= 0:
        raise ValueError("generated application has no global-output writers")

    (application / "src/application.c").write_text(
        _c_source(
            name,
            input_tensor["normalized_name"],
            output_tensor["normalized_name"],
            input_tensor["byte_size"],
            output_tensor["byte_size"] // 2,
            token_count,
            mesh_tiles,
        ),
        encoding="utf-8",
    )
    runner = application / f"src/{name}_runner.c"
    runner.write_text(
        _instrument_runner(
            runner.read_text(encoding="utf-8"),
            name,
            token_count,
            mesh_tiles,
            output_writers,
        ),
        encoding="utf-8",
    )
    reference_target = application / f"data/{name}.reference.bin"
    reference_target.write_bytes(reference.read_bytes())
    (application / "src/maps_experiment_reference.S.in").write_text(
        f'''.section .l2_bulk.maps_reference, "a", @progbits
.balign 16
.global maps_experiment_reference_start
maps_experiment_reference_start:
.incbin "@CMAKE_CURRENT_SOURCE_DIR@/data/{name}.reference.bin"
''',
        encoding="utf-8",
    )
    cmake = application / "CMakeLists.txt"
    cmake.write_text(
        cmake.read_text(encoding="utf-8")
        + f'''
configure_file(
  src/maps_experiment_reference.S.in
  ${{CMAKE_CURRENT_BINARY_DIR}}/maps_experiment_reference.S
  @ONLY
)
target_sources({name} PRIVATE ${{CMAKE_CURRENT_BINARY_DIR}}/maps_experiment_reference.S)
''',
        encoding="utf-8",
    )
    return {
        "output_writers": output_writers,
        "active_tiles": len(manifest["active_physical_tiles"]),
        "active_tile_ids": tuple(manifest["active_physical_tiles"]),
    }


def parse_runtime_log(
    text: str,
    token_count: int,
    active_tiles: tuple[int, ...],
) -> RuntimeResult:
    """Reject incomplete, serialized, out-of-order, or incorrect execution."""

    completion_matches = _COMPLETION.findall(text)
    completions = {int(token): int(cycle) for token, cycle in completion_matches}
    if len(completion_matches) != token_count or set(completions) != set(range(token_count)):
        raise ValueError("completion events are missing or duplicated")
    cycles = tuple(completions[token] for token in range(token_count))
    if any(right <= left for left, right in zip(cycles, cycles[1:])):
        raise ValueError("token completion events are out of order")

    validation_matches = _VALIDATION.findall(text)
    validations = {
        int(token): (int(mismatches), int(nonfinite))
        for token, mismatches, nonfinite in validation_matches
    }
    if len(validation_matches) != token_count or set(validations) != set(range(token_count)):
        raise ValueError("validation results are missing or duplicated")
    bad = {token: result for token, result in validations.items() if result != (0, 0)}
    if bad:
        raise ValueError(f"numerical validation failed: {bad}")

    intervals = tuple(
        TileTokenInterval(*(int(value) for value in match))
        for match in _TRACE.findall(text)
    )
    expected_pairs = {
        (tile, token) for tile in active_tiles for token in range(token_count)
    }
    observed_pairs = [(interval.tile, interval.token) for interval in intervals]
    if len(observed_pairs) != len(expected_pairs) or set(observed_pairs) != expected_pairs:
        raise ValueError("runtime trace is incomplete or duplicated")
    if any(interval.start >= interval.end for interval in intervals):
        raise ValueError("runtime trace contains an invalid interval")
    overlap = any(
        left.tile != right.tile
        and left.token != right.token
        and max(left.start, right.start) < min(left.end, right.end)
        for index, left in enumerate(intervals)
        for right in intervals[index + 1 :]
    )
    if not overlap:
        raise ValueError("runtime trace is globally serialized")
    completion_intervals = (None,) + tuple(
        right - left for left, right in zip(cycles, cycles[1:])
    )
    return RuntimeResult(cycles, completion_intervals, intervals)


def write_measurements(
    output: Path,
    result: RuntimeResult,
    active_tiles: int,
    token_slots: int,
) -> None:
    """Write one inspectable result row per Execution Token."""

    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "strategy",
                "mesh_width",
                "mesh_height",
                "token_index",
                "completion_cycle",
                "completion_interval",
                "execution_tokens",
                "token_slots",
                "active_tiles",
            ),
        )
        writer.writeheader()
        for token, (cycle, interval) in enumerate(
            zip(result.completion_cycles, result.completion_intervals)
        ):
            writer.writerow(
                {
                    "strategy": "maps",
                    "mesh_width": 4,
                    "mesh_height": 4,
                    "token_index": token,
                    "completion_cycle": cycle,
                    "completion_interval": "" if interval is None else interval,
                    "execution_tokens": len(result.completion_cycles),
                    "token_slots": token_slots,
                    "active_tiles": active_tiles,
                }
            )
