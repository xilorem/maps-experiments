#!/usr/bin/env python3
"""Build an existing full MobileViT allocation and monitor stages and errors.

No allocation or placement is repeated. Copies the generated application into a
separate run directory, enables operation and token progress, and runs GVSoC's
optimized launcher with optional instruction tracing. The monitor persists after its
launching terminal exits. A kernel error stops only this simulator process group.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

import magia_sdk


def tile_source(app, tile):
    for digits in (4, 3, 2, 1):
        path = app / f'src/tiles/tile_{tile:0{digits}d}.c'
        if path.exists():
            return path
    raise FileNotFoundError(f'No generated source for tile {tile} in {app}')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def add_prefix_outputs(app, output, execution):
    # The exporters go in the runner: the slice alias table the runtime fills is a
    # static in a header inline, so a tile source would get its own, empty copy.
    manifest, exporters, cases = [], [], []
    for tile in execution['stages'][-1]['submesh']['tile_ids']:
        path = tile_source(app, tile)
        source = path.read_text()
        final = source.rsplit('.outputs = ', 1)[1].split('\n  },', 1)[0]
        shapes = re.findall(r'\.shape = \{([^}]+)\}', final)
        widths = re.findall(r'\.elem_bytes = (\d+)u?', final)
        ranks = re.findall(r'\.rank = (\d+)u?', final)
        sizes = []
        for shape, width, rank in zip(shapes, widths, ranks):
            n = 1
            for dim in shape.split(',')[:int(rank)]:
                n *= int(dim.strip().rstrip('u'))
            sizes.append(n * int(width))
        if not sizes or not all(sizes):
            raise ValueError(f'No terminal outputs on tile {tile}')
        total = sum(sizes)
        name = f'maps_prefix_export_{tile}'
        manifest.append({'tile': tile, 'outputs': sizes, 'bytes_per_token': total})
        cases.append(f'    case {tile}u: {name}(&plan); break;')
        code = 'static volatile uint8_t @NAME@_buffer[2][@SIZE@]\n    __attribute__((aligned(16), section(".l2_arena.maps_application")));\nvoid @NAME@(const fifo_tile_plan_t *plan) {\n    const op_desc_t *op = &plan->ops[plan->num_ops - 1u];\n    for (uint32_t token = 0; token < 2u; ++token) {\n        uint32_t cursor = 0;\n        for (uint32_t which = 0; which < op->num_outputs; ++which) {\n            const subslice_desc_t *sub = &op->outputs[which];\n            uint32_t count = maps_shape_elems(sub->rank, sub->shape);\n            const volatile uint8_t *src = (const volatile uint8_t *)local_subslice_addr(\n                (const tile_plan_t *)plan, sub, maps_token_slot((const tile_plan_t *)plan, token));\n            uint32_t checksum = 2166136261u, start = cursor;\n            for (uint32_t index = 0; index < count; ++index) {\n                uint32_t remaining = index, offset = 0;\n                for (uint32_t dim = sub->rank; dim-- > 0u;) {\n                    offset += (remaining % sub->shape[dim]) * sub->strides_bytes[dim];\n                    remaining /= sub->shape[dim];\n                }\n                for (uint32_t byte = 0; byte < sub->elem_bytes; ++byte) {\n                    uint8_t value = src[offset + byte];\n                    @NAME@_buffer[token][cursor++] = value;\n                    checksum = (checksum ^ value) * 16777619u;\n                }\n            }\n            asm volatile("fence rw, rw" ::: "memory");\n            uint32_t copied = 2166136261u;\n            for (uint32_t index = start; index < cursor; ++index) {\n                uint32_t value = @NAME@_buffer[token][index];\n                /* Explicitly zero upper bits: the current indexed-byte-load\n                 * checksum path disagrees with byte comparison in GVSoC. */\n                asm volatile("andi %0, %0, 255" : "+r"(value));\n                copied = (copied ^ value) * 16777619u;\n            }\n            if (copied != checksum) {\n                uint32_t mismatches = 0u;\n                for (uint32_t index = 0; index < count; ++index) {\n                    uint32_t remaining = index, offset = 0;\n                    for (uint32_t dim = sub->rank; dim-- > 0u;) {\n                        offset += (remaining % sub->shape[dim]) * sub->strides_bytes[dim];\n                        remaining /= sub->shape[dim];\n                    }\n                    for (uint32_t byte = 0; byte < sub->elem_bytes; ++byte) {\n                        uint32_t packed = start + index * sub->elem_bytes + byte;\n                        uint8_t expected = src[offset + byte];\n                        uint8_t actual = @NAME@_buffer[token][packed];\n                        if (actual != expected) {\n                            if (mismatches < 8u)\n                                printf("MAPS_PREFIX_BYTE_MISMATCH tile=%u token=%u output=%u byte=%u expected=%u actual=%u\\n",\n                                       plan->hartid, token, which, packed - start, expected, actual);\n                            ++mismatches;\n                        }\n                    }\n                }\n                printf("MAPS_PREFIX_OUTPUT_ERROR tile=%u token=%u op=%u output=%u code=-1 expected_checksum=%u actual_checksum=%u mismatches=%u bytes=%u address=%u cycle=%u\\n",\n                       plan->hartid, token, plan->num_ops - 1u, which, checksum, copied,\n                       mismatches, cursor - start, (uint32_t)(uintptr_t)&@NAME@_buffer[token][start], maps_read_cycle());\n                maps_trap();\n            }\n            printf("MAPS_PREFIX_OUTPUT_DONE tile=%u token=%u op=%u output=%u bytes=%u checksum=%u address=%u cycle=%u\\n",\n                   plan->hartid, token, plan->num_ops - 1u, which, cursor - start,\n                   copied, (uint32_t)(uintptr_t)&@NAME@_buffer[token][start], maps_read_cycle());\n        }\n    }\n}\n'.replace('@NAME@', name).replace('@SIZE@', str(total))
        exporters.append(code)
    atomic_json(output / 'output-buffers.json', manifest)
    runner = app / 'src/maps_mobilevit_runner.c'
    source = runner.read_text().replace('int maps_mobilevit_run(void)', ''.join(exporters) + '\nint maps_mobilevit_run(void)')
    anchor = '  fsync_sync_global();\n  eu_fsync_wait(MAPS_MOBILEVIT_WAIT_MODE);\n  if (active)\n    maps_fifo_flush_experiment_trace('
    replacement = '  if (active) {\n    switch (hartid) {\n' + '\n'.join(cases) + '\n    default: break;\n    }\n  }\n' + anchor
    assert anchor in source
    runner.write_text(source.replace(anchor, replacement, 1))


def limit_stages(mesh_root, app, output, count):
    execution = json.loads((mesh_root / 'execution-plan.json').read_text())
    if not 1 <= count <= len(execution['stages']):
        raise ValueError('Invalid stage limit')
    execution['stages'] = execution['stages'][:count]
    retained = {tile for stage in execution['stages'] for tile in stage['submesh']['tile_ids']}
    atomic_json(output / 'diagnostic-execution-plan.json', execution)
    for tile in retained:
        path = tile_source(app, tile)
        source = path.read_text()
        for producer in re.findall(r'\.src_hartid = (\d+)', source):
            if int(producer) not in retained:
                raise ValueError(f'Tile {tile} depends on excluded tile {producer}')
        source, _ = magia_sdk.filter_sends(source, lambda destination: destination in retained)
        path.write_text(source)
    cmake = app / 'CMakeLists.txt'
    source = cmake.read_text()
    source = re.sub(r'^  src/tiles/tile_(\d+)\.c\n', lambda m: m[0] if int(m[1]) in retained else '', source, flags=re.M)
    cmake.write_text(source)
    plans = app / 'src/maps_mobilevit.c'
    source = plans.read_text()
    source = re.sub(r'^extern const fifo_tile_plan_t maps_mobilevit_tile_(\d+)_plan;\n', lambda m: m[0] if int(m[1]) in retained else '', source, flags=re.M)
    source = re.sub(r'^  \[(\d+)\] = &maps_mobilevit_tile_\d+_plan,\n', lambda m: m[0] if int(m[1]) in retained else '', source, flags=re.M)
    plans.write_text(source)
    runner = app / 'src/maps_mobilevit_runner.c'
    source = runner.read_text().replace('maps_operation_runtime_init(&runtime)', '(active ? maps_operation_runtime_init(&runtime) : 0)')
    source = source.replace('maps_mobilevit_handle_output(token);', 'printf("MAPS_PREFIX_DONE stages=' + str(count) + ' token=%u\\n", token);')
    runner.write_text(source)
    add_prefix_outputs(app, output, execution)
    return execution


def monitor(mesh_root, output, sdk, command, env, *, tokens=2, application=None, maps_log=None):
    log_path = maps_log or output / 'maps.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'diagnostic-execution-plan.json'
    execution = json.loads((plan_path if plan_path.exists() else mesh_root / 'execution-plan.json').read_text())
    stage_for_tile, operations, stages = {}, {}, []
    for stage_id, stage in enumerate(execution['stages']):
        tiles = stage['submesh']['tile_ids']
        stages.append({'stage': stage_id, 'name': stage['name'], 'tiles': tiles})
        for tile in tiles:
            if tile in stage_for_tile:
                raise ValueError(f'Tile {tile} assigned to multiple stages')
            stage_for_tile[tile] = stage_id
            source = tile_source(application or output / 'application', tile).read_text()
            operations[tile] = re.findall(r'\.kind = (OP_[A-Z0-9_]+),', source)
    atomic_json(output / 'stage-map.json', stages)
    buffers = {}
    output_manifest = output / "output-buffers.json"
    expected_buffers = sum(len(v["outputs"]) * tokens for v in json.loads(output_manifest.read_text())) if output_manifest.exists() else 0
    initialized, ready = set(), set()
    completed, computed = defaultdict(set), defaultdict(set)
    active, successful, errors = {}, set(), []
    reported_compute, reported_completion = set(), set()
    stage_begins, stage_compute_ends, stage_processing_ends = {}, {}, {}
    first_start = time.monotonic()
    last_event = None
    last_cycle = 0
    state = 'running'
    lines_seen = 0
    process = subprocess.Popen(command, cwd=sdk, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)
    (output / 'simulator.pid').write_text(str(process.pid) + '\n')
    atomic_json(output / 'run.json', {'started_at': now(), 'mesh_root': str(mesh_root),
                                    'monitor_pid': os.getpid(), 'simulator_pid': process.pid,
                                    'mesh': execution['mesh']['width'], 'stages': len(stages),
                                    'tiles': len(stage_for_tile), 'tokens': tokens,
                                    'command': command, 'maps_log': str(log_path),
                                    'model_directory': env.get('GVRUN_MODEL_PATH', str(sdk / 'gvsoc/install/models'))})
    print(f'RUN_STARTED simulator_pid={process.pid} stages={len(stages)} tiles={len(stage_for_tile)} tokens={tokens}', flush=True)
    messages = queue.Queue()

    def read_output():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=read_output, daemon=True).start()

    def snapshot():
        rows = []
        for stage in stages:
            row = dict(stage)
            row['tokens'] = []
            for token in range(tokens):
                done = completed[(stage['stage'], token)]
                compute = computed[(stage['stage'], token)]
                row['tokens'].append({'token': token, 'computed_tiles': len(compute),
                                      'processed_tiles': len(done), 'total_tiles': len(stage['tiles']),
                                      'compute_complete': len(compute) == len(stage['tiles']),
                                      'processing_complete': len(done) == len(stage['tiles']),
                                      'pending_tiles': sorted(set(stage['tiles']) - done)})
            rows.append(row)
        atomic_json(output / 'progress.json', {'checked_at': now(), 'state': state,
                                             'wall_seconds': time.monotonic() - first_start,
                                             'simulator_pid': process.pid,
                                             'initialized_tiles': len(initialized), 'ready_tiles': len(ready),
                                             'total_tiles': len(stage_for_tile), 'tokens': tokens,
                                             'completed_tile_tokens': sum(len(v) for v in completed.values()),
                                             'successful_kernel_calls': len(successful),
                                             'output_buffers_completed': len(buffers),
                                             'expected_output_buffers': expected_buffers,
                                             'output_buffers': list(buffers.values()),
                                             'last_observed_cycle': last_cycle, 'last_event_at': last_event,
                                             'log_lines': lines_seen, 'errors': errors,
                                             'active_kernels': sorted(active.values(), key=lambda r: (r['stage'], r['tile'])),
                                             'stages': rows,
                                             'note': 'Kernel success means return code zero, not numerical verification. Tile/token processing completion may precede outgoing DMA drain.'})
        atomic_json(output / 'kernel-errors.json', errors)

    marker = re.compile(r'(MAPS_PREFIX_OUTPUT_DONE|MAPS_PREFIX_OUTPUT_ERROR|MAPS_KERNEL_STEP|MAPS_KERNEL_BEGIN|MAPS_KERNEL_DONE|MAPS_KERNEL_ERROR|MAPS_RUNTIME_ERROR|MAPS_INIT_DONE|MAPS_INIT_ERROR|MAPS_TILE_READY|MAPS_TOKEN_DONE)\b(.*)')
    values = re.compile(r'(\w+)=([^\s]+)')
    next_snapshot = 0
    snapshot()
    with log_path.open('w', buffering=1) as log, (output / 'stage-events.jsonl').open('w', buffering=1) as events:
        while True:
            try:
                line = messages.get(timeout=1)
            except queue.Empty:
                line = ''
            if line is None:
                break
            if line:
                log.write(line)
                lines_seen += 1
                match = marker.search(line)
                if match:
                    kind, rest = match.groups()
                    data = dict(values.findall(rest))
                    tile = int(data['tile'])
                    stage_id = stage_for_tile.get(tile, -1)
                    token, op = int(data.get('token', -1)), int(data.get('op', -1))
                    last_event = now()
                    if 'cycle' in data:
                        last_cycle = max(last_cycle, int(data['cycle']))
                    record = {'observed_at': last_event, 'stage': stage_id, 'stage_name': stages[stage_id]['name'] if stage_id >= 0 else None,
                              'tile': tile, 'token': token, 'op': op,
                              'operation': operations[tile][op] if tile in operations and 0 <= op < len(operations[tile]) else None,
                              'marker': kind, 'details': data}
                    if kind == 'MAPS_INIT_DONE':
                        if tile in stage_for_tile:
                            initialized.add(tile)
                    elif kind == 'MAPS_TILE_READY':
                        if tile in stage_for_tile:
                            ready.add(tile)
                    elif kind == 'MAPS_PREFIX_OUTPUT_DONE':
                        buffers[(tile, token, int(data['output']))] = record
                        events.write(json.dumps(record) + '\n')
                    elif kind == 'MAPS_KERNEL_BEGIN':
                        key = (stage_id, token)
                        cycle = int(data['cycle'])
                        stage_begins[key] = min(stage_begins.get(key, cycle), cycle)
                        active[tile] = record
                    elif kind == 'MAPS_KERNEL_STEP':
                        if tile in active:
                            active[tile]['latest_step'] = data
                            active[tile]['progress_at'] = last_event
                    elif kind == 'MAPS_KERNEL_DONE':
                        active.pop(tile, None)
                        successful.add((tile, token, op))
                        if op == len(operations[tile]) - 1:
                            key = (stage_id, token)
                            computed[key].add(tile)
                            stage_compute_ends[key] = max(stage_compute_ends.get(key, 0), int(data['cycle']))
                    elif kind == 'MAPS_TOKEN_DONE':
                        key = (stage_id, token)
                        completed[key].add(tile)
                        stage_processing_ends[key] = max(stage_processing_ends.get(key, 0), int(data['cycle']))
                    else:
                        errors.append(record)
                        state = 'kernel_failed' if kind == 'MAPS_KERNEL_ERROR' else 'runtime_failed'
                        print('FAILURE ' + json.dumps(record), flush=True)
                        events.write(json.dumps(record) + '\n')
                        snapshot()
                        magia_sdk.stop_process_group(process)
                    for bucket, reported, event in [(computed, reported_compute, 'STAGE_COMPUTE_DONE'),
                                                      (completed, reported_completion, 'STAGE_PROCESSING_DONE')]:
                        key = (stage_id, token)
                        if stage_id >= 0 and token >= 0 and key not in reported and len(bucket[key]) == len(stages[stage_id]['tiles']):
                            reported.add(key)
                            result = {'event': event, 'stage': stage_id, 'token': token,
                                      'name': stages[stage_id]['name'], 'tiles': len(bucket[key]), 'observed_at': now()}
                            ends = stage_compute_ends if event == 'STAGE_COMPUTE_DONE' else stage_processing_ends
                            end = ends.get(key)
                            begin = stage_begins.get(key)
                            result.update(cycle=end, elapsed_cycles=end-begin if begin is not None and end is not None else None)
                            events.write(json.dumps(result) + '\n')
                            print(f'{event} stage={stage_id} token={token} tiles={len(bucket[key])} cycle={end} elapsed_cycles={result["elapsed_cycles"]} name={stages[stage_id]["name"]}', flush=True)
            if time.monotonic() >= next_snapshot:
                snapshot()
                next_snapshot = time.monotonic() + 15
        result = process.wait()
    expected = len(stage_for_tile) * tokens
    complete = sum(len(v) for v in completed.values()) == expected and len(buffers) == expected_buffers
    if not errors:
        state = 'completed' if result == 0 and complete else 'simulator_failed'
    if state == 'simulator_failed':
        errors.append({'marker': 'SIMULATOR_EXIT', 'code': result, 'observed_at': now(),
                       'completed_tile_tokens': sum(len(v) for v in completed.values()), 'expected_tile_tokens': expected})
    snapshot()
    (output / 'exit-status').write_text(str(result) + '\n')
    if errors:
        (output / 'failure.md').write_text('# MobileViT monitored run failure\n\n```json\n' + json.dumps(errors, indent=2) + '\n```\n\nSee progress.json for active kernels and pending tiles, and ' + str(log_path) + ' for raw output.\n')
    print(f'RUN_FINISHED state={state} exit_status={result} completed_tile_tokens={sum(len(v) for v in completed.values())}/{expected}', flush=True)
    return 0 if state == 'completed' else 1


def prepare_progress(app):
    cmake = app / 'CMakeLists.txt'
    bundle = cmake.read_text()
    transpose = 'add_spatz_task(' in bundle
    if transpose and 'transpose/fp16/spatz/spatz_task/transpose_strided_fp16_spatz_task.c' not in bundle:
        bundle = bundle.replace('  TASK_SOURCES\n', '  TASK_SOURCES\n    ${MAPS_SPATZ_KERNELS_DIR}/transpose/fp16/spatz/spatz_task/transpose_strided_fp16_spatz_task.c\n', 1)
        bundle = bundle.replace('  INCLUDE_DIRS\n', '  INCLUDE_DIRS\n    ${MAPS_SPATZ_KERNELS_DIR}/transpose/fp16/spatz/include\n', 1)
        bundle += '\ntarget_compile_definitions(maps_mobilevit PRIVATE MAPS_HAS_TRANSPOSE_SPATZ_TASK=1)\n'
        cmake.write_text(bundle)
    # Preserve raw output values for tolerance-based validation after the run.
    hooks=app/'src/application.c'
    if hooks.exists():
        hook_source=hooks.read_text()
        if 'MAPS_OUTPUT_BYTES' not in hook_source:
            handler=hook_source.find('void maps_mobilevit_handle_output(')
            tail=hook_source[handler:] if handler>=0 else ''
            pointer=re.search(r'const uint8_t \*(\w+) = maps_mobilevit_output_\w+\(token\);',tail)
            size=re.search(r'index < (\d+)u;',tail)
            if pointer and size:
                name=pointer[1]; count=int(size[1]); end=hook_source.rfind('\n}')
                dump=(f'\n  for (uint32_t offset=0; offset<{count}u; offset+=16u) {{\n'
                      '    printf("MAPS_OUTPUT_BYTES token=%u offset=%u data=", token, offset);\n'
                      f'    for (uint32_t i=offset; i<offset+16u && i<{count}u; ++i) printf("%02x", {name}[i]);\n'
                      '    printf("\\n");\n  }\n')
                hooks.write_text(hook_source[:end]+dump+hook_source[end:])
    runner = app / 'src/maps_mobilevit_runner.c'
    source = runner.read_text()
    if transpose and 'runtime.transpose_fp16_task =' not in source:
        source = source.replace('  runtime.relu_fp16_task =', '  runtime.transpose_fp16_task = TRANSPOSE_STRIDED_FP16_SPATZ_TASK;\n  runtime.relu_fp16_task =', 1)
    for field, symbol, task in (
        ('depthwise_conv_fp16_task', 'DEPTHWISE_CONV_FP16_SPATZ_TASK', 'depthwise_conv_fp16_spatz_task.c'),
        ('sigmoid_fp16_task', 'SIGMOID_FP16_SPATZ_TASK', 'sigmoid_fp16_spatz_task.c'),
    ):
        binding = f'  runtime.{field} = {symbol};\n'
        if task not in bundle:
            source = source.replace(binding, '')
        elif f'runtime.{field} =' not in source:
            source = source.replace('  runtime.relu_fp16_task =', binding + '  runtime.relu_fp16_task =', 1)
    source = source.replace('if (maps_operation_runtime_init(&runtime) != 0)\n    return -1;', '''int init_result = maps_operation_runtime_init(&runtime);
  if (init_result != 0) {
    printf("MAPS_INIT_ERROR tile=%u code=%d\\n", hartid, init_result);
    return -1;
  }
  printf("MAPS_INIT_DONE tile=%u cycle=%u\\n", hartid, maps_read_cycle());''')
    if 'printf("MAPS_TILE_READY ' not in source:
        source = source.replace('maps_fifo_init_tile(&plan);', '''maps_fifo_init_tile(&plan);
  printf("MAPS_TILE_READY tile=%u cycle=%u\\n", hartid, maps_read_cycle());''')
    runner.write_text(source)


def simulation_environment():
    env = os.environ.copy()
    toolchain = Path('/home/zaino/repos/toolchains')
    llvm = Path(env.get('LLVM_INSTALL_DIR', toolchain / 'llvm20-root/usr/lib/llvm-20'))
    env['LLVM_INSTALL_DIR'] = str(llvm)
    env['LD_LIBRARY_PATH'] = str(llvm.parent / 'x86_64-linux-gnu') + ':' + env.get('LD_LIBRARY_PATH', '')
    env['PATH'] = str(Path(sys.executable).parent) + ':' + str(toolchain / 'pulp-toolchain/bin') + ':' + env['PATH']
    env['CMAKE_BUILD_PARALLEL_LEVEL'] = env.get('CMAKE_BUILD_PARALLEL_LEVEL', '8')
    return env


def run_simulation(args, mesh_root, sdk, output, app, build, env):
    execution = json.loads((mesh_root / 'execution-plan.json').read_text())
    mesh_size = execution['mesh']['width']
    work = (args.work_dir or output / 'gvsoc').resolve()
    work.mkdir(parents=True, exist_ok=True)
    platform_tree = args.platform_tree
    if args.fifo_event_sleep and platform_tree is None:
        from fifo_sleep_tree import build_platform_tree
        atomic_json(output / 'progress.json', {'state': 'building_platform_tree', 'checked_at': now()})
        try:
            platform_tree = build_platform_tree(sdk, mesh_size, output / 'platform-tree', env)
        except subprocess.CalledProcessError as error:
            atomic_json(output / 'progress.json', {'state': 'platform_tree_build_failed',
                                                 'checked_at': now(), 'code': error.returncode})
            raise
    command = magia_sdk.gvrun_command(sdk, build, 'maps_mobilevit', work, mesh_size)
    if args.fifo_event_sleep:
        command[-1:-1] = ['--attr', 'magia_v3/fifo_events=1']
    if platform_tree:
        command[-1:-1] = ['--gv-opt', f'platform_tree={platform_tree.resolve()}']
    if args.instruction_trace:
        trace_tiles = ['[0-9]+'] if args.trace_tiles == 'all' else args.trace_tiles.split(',')
        for tile in trace_tiles:
            command[-1:-1] = [f'--trace=tile-{tile}-cv32-core/insn:cv32-insn.log',
                              f'--trace=tile-{tile}-snitch-spatz/insn:spatz-insn.log']
        command[-1:-1] = ['--trace-level=trace', '--trace=kill-module']
    return monitor(mesh_root, output, sdk, command, env, tokens=args.tokens,
                   application=app, maps_log=args.maps_log.resolve() if args.maps_log else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh-root', required=True, type=Path)
    parser.add_argument('--sdk-root', type=Path, default=magia_sdk.default_sdk_root())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage-limit', type=int, help='run only the first N stages on the original mesh')
    sleep = parser.add_mutually_exclusive_group()
    sleep.add_argument('--fifo-event-sleep', dest='fifo_event_sleep', action='store_true',
                       help='GVSoC FIFO doorbells and event-driven receive sleep (default)')
    sleep.add_argument('--no-fifo-event-sleep', dest='fifo_event_sleep', action='store_false',
                       help='use polling FIFO receives')
    parser.set_defaults(fifo_event_sleep=True)
    parser.add_argument('--prepare-application', type=Path, help='only add live progress markers to this generated application')
    parser.add_argument('--sdk-build', type=Path, help='run an already-built application without copying or rebuilding it')
    parser.add_argument('--tokens', type=int, default=2, help='number of execution tokens in the generated application')
    parser.add_argument('--maps-log', type=Path, help='raw simulator log path (default: output/maps.log)')
    parser.add_argument('--work-dir', type=Path, help='GVSoC work directory (default: output/gvsoc)')
    parser.add_argument('--instruction-trace', action='store_true', help='enable CV32 and Spatz instruction traces')
    parser.add_argument('--trace-tiles', default='all', help='instruction trace tile IDs: all or N,N,...')
    parser.add_argument('--platform-tree', type=Path,
                        help='matching compiled GVSoC platform-tree library for an experimental model')
    parser.add_argument('--foreground', action='store_true', help='run the monitor in this process')
    args = parser.parse_args()
    if not 1 <= args.tokens <= 16:
        parser.error('--tokens must be 1..16')
    if args.sdk_build and args.stage_limit is not None:
        parser.error('--sdk-build cannot be combined with --stage-limit')
    if args.trace_tiles != 'all':
        if not re.fullmatch(r'(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*', args.trace_tiles):
            parser.error('--trace-tiles must be all or comma-separated non-negative tile IDs')
        args.instruction_trace = True
    mesh_root, sdk = args.mesh_root.resolve(), args.sdk_root.resolve()
    output = (args.output or mesh_root / 'maps-gvsoc-diagnostic' / ('full-monitored-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    execution = json.loads((mesh_root / 'execution-plan.json').read_text())
    tile_count = execution['mesh']['width'] * execution['mesh']['height']
    if args.trace_tiles != 'all' and any(int(t) >= tile_count for t in args.trace_tiles.split(',')):
        parser.error('--trace-tiles includes a tile outside the mesh')
    if args.prepare_application:
        prepare_progress(args.prepare_application.resolve())
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if not args.foreground:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--mesh-root', str(mesh_root),
                   '--sdk-root', str(sdk), '--output', str(output), '--foreground']
        if args.stage_limit is not None:
            command.extend(['--stage-limit', str(args.stage_limit)])
        command.append('--fifo-event-sleep' if args.fifo_event_sleep else '--no-fifo-event-sleep')
        command.extend(['--tokens', str(args.tokens)])
        for option, value in [('--sdk-build', args.sdk_build), ('--maps-log', args.maps_log), ('--work-dir', args.work_dir)]:
            if value:
                command.extend([option, str(value.resolve())])
        if args.instruction_trace:
            command.extend(['--instruction-trace', '--trace-tiles', args.trace_tiles])
        if args.platform_tree:
            command.extend(['--platform-tree', str(args.platform_tree.resolve())])
        with (output / 'monitor.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (output / 'monitor.pid').write_text(str(process.pid) + '\n')
        print(json.dumps({'monitor_pid': process.pid, 'output': str(output)}, indent=2))
        return 0
    (output / 'monitor.pid').write_text(str(os.getpid()) + '\n')
    if args.sdk_build:
        app = mesh_root / 'maps-application'
        build = args.sdk_build.resolve()
        if not (build / 'bin/maps_mobilevit').is_file():
            parser.error(f'Missing built binary: {build / "bin/maps_mobilevit"}')
        return run_simulation(args, mesh_root, sdk, output, app, build, simulation_environment())
    if args.tokens != 2:
        parser.error('standalone rebuild expects the existing two-token input set; --sdk-build supports other token counts')
    atomic_json(output / 'progress.json', {'state': 'building', 'checked_at': now()})
    app = output / 'application'
    shutil.copytree(mesh_root / 'maps-application', app, dirs_exist_ok=True)
    cmake = app / 'CMakeLists.txt'
    source = cmake.read_text()
    if args.fifo_event_sleep:
        source += '\ntarget_compile_definitions(maps_mobilevit PRIVATE MAPS_FIFO_EVENT_SLEEP=1)\n'
    if 'depthwise_conv/fp16/spatz/spatz_task/depthwise_conv_fp16_spatz_task.c' not in source:
        source = source.replace('  TASK_SOURCES\n', '  TASK_SOURCES\n    ${MAPS_SPATZ_KERNELS_DIR}/depthwise_conv/fp16/spatz/spatz_task/depthwise_conv_fp16_spatz_task.c\n')
        source = source.replace('  INCLUDE_DIRS\n', '  INCLUDE_DIRS\n    ${MAPS_SPATZ_KERNELS_DIR}/depthwise_conv/fp16/spatz/include\n')
    if 'MAPS_HAS_DEPTHWISE_CONV_SPATZ_TASK=1' not in source:
        source += '\ntarget_compile_definitions(maps_mobilevit PRIVATE MAPS_HAS_DEPTHWISE_CONV_SPATZ_TASK=1)\n'
    if 'sigmoid/fp16/spatz/spatz_task/sigmoid_fp16_spatz_task.c' not in source:
        source = source.replace('  TASK_SOURCES\n', '  TASK_SOURCES\n    ${MAPS_SPATZ_KERNELS_DIR}/sigmoid/fp16/spatz/spatz_task/sigmoid_fp16_spatz_task.c\n')
        source = source.replace('  INCLUDE_DIRS\n', '  INCLUDE_DIRS\n    ${MAPS_SPATZ_KERNELS_DIR}/sigmoid/fp16/spatz/include\n')
    if 'MAPS_HAS_SIGMOID_SPATZ_TASK=1' not in source:
        source += '\ntarget_compile_definitions(maps_mobilevit PRIVATE MAPS_HAS_SIGMOID_SPATZ_TASK=1)\n'
    source = re.sub(r'^add_subdirectory\([^\n]*full_mesh\)\n', '', source, flags=re.M)
    source = re.sub(r'target_compile_definitions\(maps_mobilevit PRIVATE\s+MAPS_EXPERIMENT_TRACE=1\s*\)', '', source)
    cmake.write_text(source)
    prepare_progress(app)
    if args.stage_limit is not None:
        limit_stages(mesh_root, app, output, args.stage_limit)
    env = magia_sdk.build_env(sdk)
    execution = json.loads((mesh_root / 'execution-plan.json').read_text())
    mesh_size = execution['mesh']['width']
    assert mesh_size == execution['mesh']['height']
    build = output / 'sdk-build'
    try:
        with (output / 'build.log').open('w') as log:
            magia_sdk.build(sdk, env, log, mesh_size, build, app, ['spatz_bootrom', 'maps_mobilevit'],
                            ['maps_token_progress=1', 'maps_kernel_progress=1', 'maps_experiment_trace=0'])
    except subprocess.CalledProcessError as error:
        atomic_json(output / 'progress.json', {'state': 'build_failed', 'checked_at': now(), 'code': error.returncode})
        raise
    return run_simulation(args, mesh_root, sdk, output, app, build, env)


if __name__ == '__main__':
    raise SystemExit(main())
