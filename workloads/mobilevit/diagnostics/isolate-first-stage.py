#!/usr/bin/env python3
"""Extract the eight stem-convolution shards from an existing MobileViT run.

Keeps generated slice/operation/L2-read descriptors, two slots and real data.
Remaps tiles 416..423 to 0..7 on a 4x4 mesh; downstream sends are removed.
The runner reports operation boundaries and checks every intermediate/output
element against a host reference, including FP16 rounding after each FMA.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

import magia_sdk


def generate(mesh_root, output):
    original = mesh_root / "maps-application"
    app = output / "application"
    app.mkdir(parents=True, exist_ok=True)
    asm = []

    def blob(symbol, path):
        asm.extend([f'.section .l2_bulk.{symbol},"a",@progbits', '.balign 16',
                    f'.global {symbol}', f'{symbol}:', f'.incbin "{path}"'])

    blob("maps_mobilevit_initializers_start", original / "data/maps_mobilevit.initializers.bin")
    asm.extend(['.global maps_mobilevit_initializers_end', 'maps_mobilevit_initializers_end:',
                '.global maps_mobilevit_initializers_size',
                '.set maps_mobilevit_initializers_size, maps_mobilevit_initializers_end - maps_mobilevit_initializers_start'])
    blob("maps_mobilevit_l2_global_163", original / "data/maps_mobilevit.input_input.bin")
    x = np.fromfile(original / "data/maps_mobilevit.input_input.bin", dtype='<f2').reshape(2, 3, 256, 256)
    weights = np.fromfile(original / "data/maps_mobilevit.initializers.bin", dtype='<f2',
                          count=432, offset=1964384).reshape(16, 27)
    bias = np.fromfile(original / "data/maps_mobilevit.initializers.bin", dtype='<f2',
                       count=16, offset=1965248)
    mesh = json.loads((mesh_root / 'execution-plan.json').read_text())['mesh']
    declarations, plans, refs, manifest = [], [], [], []
    for tile, old in enumerate(range(416, 424)):
        source = magia_sdk.tile_source(original, mesh, old).read_text()
        slices_text = source.split(f"static const slice_desc_t maps_mobilevit_tile_{old}_slices[] = {{", 1)[1].split("\n};", 1)[0]
        shapes = [list(map(int, s.split(','))) for s in re.findall(r'\.shape = \{([^}]+)\}', slices_text)]
        origins = [list(map(int, s.split(','))) for s in re.findall(r'\.full_offset = \{([^}]+)\}', slices_text)]
        assert shapes[3][:2] == [27, 2048] and shapes[4][:4] == [1, 16, 16, 128]
        assert source.count('.kind = OP_') == 2
        start = origins[3][1]
        columns = np.zeros((2, 27, 2048), dtype=np.float16)
        positions = np.arange(start, start + 2048)
        for k in range(27):
            c, rem = divmod(k, 9)
            ky, kx = divmod(rem, 3)
            iy = (positions // 128) * 2 + ky - 1
            ix = (positions % 128) * 2 + kx - 1
            valid = (iy >= 0) & (iy < 256) & (ix >= 0) & (ix < 256)
            columns[:, k, valid] = x[:, c, iy[valid], ix[valid]]
        # Product of two binary16 values and their binary16 accumulator is
        # evaluated in float64, then rounded once, matching fmadd.h/vfmacc.
        y = np.zeros((2, 16, 2048), dtype=np.float16)
        for k in range(27):
            y = (y.astype(np.float64) + weights[None, :, k, None].astype(np.float64)
                 * columns[:, None, k, :].astype(np.float64)).astype(np.float16)
        y = (y.astype(np.float64) + bias[None, :, None].astype(np.float64)).astype(np.float16)
        for name, data in [('columns', columns), ('output', y)]:
            path = app / f"golden_{tile}_{name}.bin"
            data.astype('<f2').tofile(path)
            symbol = f"golden_{tile}_{name}"
            blob(symbol, path)
            declarations.append(f"extern const uint16_t {symbol}[];")
        # Keep the original descriptors, including absolute shard origins and
        # every slot offset. Only physical owners and the stage boundary change.
        text, _ = magia_sdk.filter_sends(source, lambda destination: False)
        text = text.replace(f'.owner_hartid = {old},', f'.owner_hartid = {tile},')
        text = text.replace(f'.hartid = {old}u,', f'.hartid = {tile}u,')
        (app / f"tile_{tile}.c").write_text(text)
        name = f"maps_mobilevit_tile_{magia_sdk.padded_tile_id(mesh, old)}_plan"
        declarations.append(f"extern const fifo_tile_plan_t {name};")
        plans.append(f"&{name}")
        refs.append(f"{{golden_{tile}_columns, golden_{tile}_output}}")
        manifest.append({'original_tile': old, 'test_tile': tile, 'im2col_global_column_start': start,
                         'shapes': shapes, 'origins': origins})
    (app / 'data.S').write_text('\n'.join(asm) + '\n')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    bundle = (original / 'CMakeLists.txt').read_text().split('add_executable(', 1)[0]
    if 'sigmoid/fp16/spatz/spatz_task/sigmoid_fp16_spatz_task.c' not in bundle:
        bundle = bundle.replace('  TASK_SOURCES\n', '  TASK_SOURCES\n    ${MAPS_SPATZ_KERNELS_DIR}/sigmoid/fp16/spatz/spatz_task/sigmoid_fp16_spatz_task.c\n')
        bundle = bundle.replace('  INCLUDE_DIRS\n', '  INCLUDE_DIRS\n    ${MAPS_SPATZ_KERNELS_DIR}/sigmoid/fp16/spatz/include\n')
    cmake = bundle.replace('set(MAPS_APPLICATION maps_mobilevit)', 'set(MAPS_APPLICATION maps_mobilevit_first_stage)')
    cmake += '\nadd_executable(${MAPS_APPLICATION} main.c data.S\n' + ''.join(f'  tile_{i}.c\n' for i in range(8)) + ')\n'
    cmake += '''target_compile_options(${MAPS_APPLICATION} PRIVATE -O2)
target_link_libraries(${MAPS_APPLICATION} PUBLIC runtime hal idma_lib redmule_lib fsync_lib eu_lib)
target_embed_spatz_task(TARGET ${MAPS_APPLICATION} TASK_NAME ${MAPS_APPLICATION})
target_compile_definitions(${MAPS_APPLICATION} PRIVATE
  MAPS_HAS_ADD_SPATZ_TASK=1 MAPS_HAS_MUL_SPATZ_TASK=1 MAPS_HAS_MATMUL_SPATZ_TASK=1
  MAPS_HAS_SIGMOID_SPATZ_TASK=1 MAPS_HAS_RELU_SPATZ_TASK=1 MAPS_HAS_SOFTMAX_EXP_SPATZ_TASK=1
  MAPS_HAS_GROUP_REDUCE_SPATZ_TASK=1 MAPS_HAS_GROUP_CENTERED_REDUCE_SPATZ_TASK=1
  MAPS_HAS_GROUP_NORMALIZE_SPATZ_TASK=1 MAPS_HAS_REDUCE_SUM_SPATZ_TASK=1
  MAPS_HAS_REDUCE_MAX_SPATZ_TASK=1 MAPS_HAS_BINARY_BCAST_SPATZ_TASK=1)
'''
    (app / 'CMakeLists.txt').write_text(cmake)
    main = r'''
#include "maps_mobilevit_first_stage_task_bin.h"
#include "fsync.h"
#include "redmule.h"
#include "tile.h"
#include "utils/maps_operations.h"
#include "utils/maps_utils_v2.h"
#include "utils/printf.h"
@DECLARATIONS@
static const fifo_tile_plan_t *const plans[8] = {@PLANS@};
static const uint16_t *const golden[8][2] = {@REFS@};
static volatile uint32_t errors[16] = {1};

static uint32_t check(const uint16_t *actual, const uint16_t *expected,
                      uint32_t count, uint32_t tile, uint32_t token, uint32_t op)
{
    uint32_t bad = 0;
    for (uint32_t i = 0; i < count; ++i) {
        /* Signed zero is equivalent; all other values must match bit for bit. */
        if (actual[i] != expected[i] && ((actual[i] | expected[i]) & 0x7fff)) {
            if (bad < 3)
                printf("FIRST_STAGE_MISMATCH tile=%u token=%u op=%u index=%u actual=%04x expected=%04x\n",
                       tile, token, op, i, actual[i], expected[i]);
            ++bad;
        }
    }
    printf("FIRST_STAGE_CHECK tile=%u token=%u op=%u elements=%u errors=%u cycle=%u\n",
           tile, token, op, count, bad, maps_read_cycle());
    return bad;
}

int main(void)
{
    const uint32_t hartid = get_hartid();
    idma_init(); redmule_init(); fsync_init(); eu_init();
    eu_clear_events(0xffffffffu);
    eu_fsync_init(0u); eu_idma_init(0u); eu_redmule_init(0u);
    /* Field by field, like the generated runner: CORE-V GCC ICEs (maybe_postinc)
     * on a zero-filled designated initialiser. */
    maps_operation_runtime_t runtime;
    runtime.kernel_abi_version = MAPS_KERNEL_ABI_VERSION;
    runtime.task_bundle_abi_version = MAPS_TASK_BUNDLE_ABI_VERSION;
    runtime.spatz_binary_start = SPATZ_BINARY_START;
    runtime.spatz_params = (void *)(get_l1_base(hartid) + MAPS_OPERATION_TASK_SCRATCH_OFFSET);
    runtime.spatz_params_bytes = MAPS_OPERATION_TASK_SCRATCH_BYTES;
    runtime.spatz_initialized = 0u;
    runtime.background = 0;
    runtime.background_ctx = 0;
    runtime.add_fp16_task = ADD_FP16_SPATZ_TASK;
    runtime.mul_bcast_fp16_task = MUL_BCAST_FP16_SPATZ_TASK;
    runtime.matmul_fp16_task = MATMUL_ROWS_FP16_SPATZ_TASK;
    runtime.relu_fp16_task = RELU_FP16_SPATZ_TASK;
    runtime.sigmoid_fp16_task = SIGMOID_FP16_SPATZ_TASK;
    runtime.depthwise_conv_fp16_task = 0u;
    runtime.softmax_exp_fp16_task = SOFTMAX_EXP_FP16_SPATZ_TASK;
    runtime.group_reduce_fp16_task = GROUP_REDUCE_FP16_SPATZ_TASK;
    runtime.group_centered_reduce_fp16_task = GROUP_CENTERED_REDUCE_FP16_SPATZ_TASK;
    runtime.group_normalize_fp16_task = GROUP_NORMALIZE_FP16_SPATZ_TASK;
    runtime.reducesum_fp16_task = REDUCESUM_FP16_SPATZ_TASK;
    runtime.reducemax_fp16_task = REDUCEMAX_FP16_SPATZ_TASK;
    runtime.binary_bcast_fp16_task = BINARY_BCAST_FP16_SPATZ_TASK;
    printf("FIRST_STAGE_INIT tile=%u cycle=%u\n", hartid, maps_read_cycle());
    if (maps_operation_runtime_init(&runtime) != 0) return -1;
    errors[hartid] = 0;
    fifo_tile_plan_t plan;
    if (hartid < 8) {
        plan = *plans[hartid];
        plan.l1_data_base += get_l1_base(hartid);
        plan.operation_runtime = &runtime;
        maps_fifo_init(&plan);
    }
    fsync_sync_global(); eu_fsync_wait(WFE);
    /* Like the generated runner: descriptors to L1, then the initializer reads. */
    if (hartid < 8) { maps_fifo_plan_to_l1(&plan); maps_fifo_init_tile(&plan); }
    fsync_sync_global(); eu_fsync_wait(WFE);
    if (hartid < 8) {
        /* Same read/compute ordering as maps_fifo_run_tile_tokens; stage
         * outputs terminate here so there are no downstream FIFO sends. */
        for (uint32_t token = 0; token < 2; ++token) {
            uint32_t slot = maps_token_slot((const tile_plan_t *)&plan, token);
            printf("FIRST_STAGE_READ_BEGIN tile=%u token=%u cycle=%u\n", hartid, token, maps_read_cycle());
            for (uint32_t i = 0; i < plan.num_l2_reads; ++i)
                issue_l2_read_token((const tile_plan_t *)&plan, &plan.l2_reads[i], token, slot);
            printf("FIRST_STAGE_READ_DONE tile=%u token=%u cycle=%u\n", hartid, token, maps_read_cycle());
            for (uint32_t op = 0; op < plan.num_ops; ++op) {
                uint32_t start = maps_read_cycle();
                printf("FIRST_STAGE_OP_BEGIN tile=%u token=%u op=%u cycle=%u\n", hartid, token, op, start);
                int result = maps_execute_operation((const tile_plan_t *)&plan, &plan.ops[op], slot, &runtime);
                printf("FIRST_STAGE_OP_DONE tile=%u token=%u op=%u result=%d cycles=%u\n",
                       hartid, token, op, result, maps_read_cycle() - start);
                if (result) return -1;
                uint32_t count = op == 0 ? 27u * 2048u : 16u * 2048u;
                const uint16_t *data = (const uint16_t *)local_subslice_addr(
                    (const tile_plan_t *)&plan, &plan.ops[op].outputs[0], slot);
                errors[hartid] += check(data, golden[hartid][op] + token * count, count, hartid, token, op);
            }
            printf("MAPS_TOKEN_DONE tile=%u token=%u cycle=%u\n", hartid, token, maps_read_cycle());
        }
    }
    fsync_sync_global(); eu_fsync_wait(WFE);
    uint32_t total = 0;
    for (uint32_t i = 0; i < 16; ++i) total += errors[i];
    if (hartid == 0) printf("FIRST_STAGE_RESULT errors=%u active_tiles=8 tokens=2\n", total);
    spatz_clk_dis();
    return total ? -1 : 0;
}
'''
    main = main.replace('@DECLARATIONS@', '\n'.join(declarations)).replace('@PLANS@', ', '.join(plans)).replace('@REFS@', ', '.join(refs))
    (app / 'main.c').write_text(main)
    return app


def summarize(output):
    log = (output / 'maps.log').read_text()
    checks = [dict(zip(('tile', 'token', 'op', 'elements', 'errors', 'cycle'), map(int, m)))
              for m in re.findall(r'FIRST_STAGE_CHECK tile=(\d+) token=(\d+) op=(\d+) elements=(\d+) errors=(\d+) cycle=(\d+)', log)]
    completed = {(int(tile), int(token)) for tile, token in
                 re.findall(r'MAPS_TOKEN_DONE tile=(\d+) token=(\d+)', log)}
    expected_checks = {(tile, token, op) for tile in range(8) for token in range(2) for op in range(2)}
    actual_checks = {(c['tile'], c['token'], c['op']) for c in checks}
    operation_cycles = {name: [] for name in ('im2col', 'matmul')}
    results = []
    for tile, token, op, result, cycles in re.findall(
            r'FIRST_STAGE_OP_DONE tile=(\d+) token=(\d+) op=(\d+) result=(-?\d+) cycles=(\d+)', log):
        operation_cycles[('im2col', 'matmul')[int(op)]].append(int(cycles))
        results.append(int(result))
    passed = (len(checks) == 32 and actual_checks == expected_checks
              and completed == {(tile, token) for tile in range(8) for token in range(2)}
              and sum(c['errors'] for c in checks) == 0
              and len(results) == 32 and all(r == 0 for r in results)
              and 'FIRST_STAGE_RESULT errors=0 active_tiles=8 tokens=2' in log)
    summary = {'passed': passed, 'mesh': '4x4', 'active_tiles': 8,
               'tokens': 2, 'completed_tile_tokens': len(completed),
               'checked_elements': sum(c['elements'] for c in checks),
               'mismatches': sum(c['errors'] for c in checks),
               'operation_cycles': {name: {'min': min(cycles), 'max': max(cycles),
                                          'mean': sum(cycles) / len(cycles)}
                                    for name, cycles in operation_cycles.items() if cycles},
               'checks': checks,
               'scope': 'Original shard descriptors, real inputs/weights, L2 reads, IDMA im2col and Spatz matmul. Downstream FIFO sends removed; read/compute ordering reproduced with operation-boundary prints.'}
    (output / 'statistics.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main(target='maps_mobilevit_first_stage', output_name='first-stage-4x4', mesh_size=4):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh-root', required=True, type=Path)
    parser.add_argument('--sdk-root', type=Path, default=magia_sdk.default_sdk_root())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--fifo-event-sleep', action='store_true',
                        help='experimental GVSoC FIFO doorbells and receive sleep')
    parser.add_argument('--timeout', type=float, default=180,
                        help='maximum simulator wall time in seconds (default: 180)')
    parser.add_argument('--blocking-fifo', action='store_true',
                        help='two-stage diagnostic: use blocking FIFO sends for comparison')
    parser.add_argument('--dma-read-trace', type=int,
                        help='trace the selected tile\'s input DMA AXI reads')
    args = parser.parse_args()
    args.sdk_root = args.sdk_root.resolve()
    output = (args.output or args.mesh_root / 'maps-gvsoc-diagnostic' / output_name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    app = generate(args.mesh_root.resolve(), output)
    if args.fifo_event_sleep:
        with (app / 'CMakeLists.txt').open('a') as cmake:
            cmake.write(f'\ntarget_compile_definitions({target} PRIVATE MAPS_FIFO_EVENT_SLEEP=1)\n')
    if args.blocking_fifo:
        source = (app / 'main.c').read_text()
        call = 'maps_fifo_run_tile_tokens(&plan, 2u);'
        if call not in source:
            parser.error('--blocking-fifo requires the two-stage diagnostic')
        source = source.replace(call, '''printf("TWO_STAGE_RUN_MODE blocking\\n");
        for (uint32_t token = 0; token < 2; ++token)
            maps_fifo_run_tile_token(&plan, token);''')
        (app / 'main.c').write_text(source)
    print(f'Generated isolated stage in {app}', flush=True)
    if args.generate_only:
        return
    env = magia_sdk.build_env(args.sdk_root)
    build = output / 'sdk-build'
    start = time.time()
    with (output / 'build.log').open('w') as log:
        magia_sdk.build(args.sdk_root, env, log, mesh_size, build, app,
                        ('spatz_bootrom', target), ['maps_token_progress=1'])
    print(f'Build completed in {time.time() - start:.1f}s.', flush=True)
    work = output / 'gvsoc'
    work.mkdir(exist_ok=True)
    (output / 'exit-status').unlink(missing_ok=True)
    (output / 'statistics.json').unlink(missing_ok=True)
    options = ['--trace-level=trace', '--trace=kill-module']
    if args.dma_read_trace is not None:
        options.append(f'--trace=tile-{args.dma_read_trace}-idma-axi2obi/axi_read/trace:dma-read.log')
    command = magia_sdk.gvrun_command(args.sdk_root, build, target, work, mesh_size, options)
    if args.fifo_event_sleep:
        command[-1:-1] = ['--attr', 'magia_v3/fifo_events=1']
    start = time.time()
    with (output / 'maps.log').open('w') as log:
        process = subprocess.Popen(command, cwd=args.sdk_root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        (output / 'pid').write_text(str(process.pid) + '\n')
        print(f'Started simulator PID {process.pid}; log {output / "maps.log"}', flush=True)
        stopped_reason = None
        while process.poll() is None:
            current_log = (output / 'maps.log').read_text()
            if 'ISOLATED_OPERATION_FAILURE' in current_log:
                stopped_reason = 'Operation dispatcher returned an error; stopped diagnostic process group.'
            elif 'ISOLATED_INPUT_FAILURE' in current_log:
                stopped_reason = 'L2 input validation failed before im2col; stopped diagnostic process group.'
            elif time.time() - start > args.timeout:
                stopped_reason = f'Diagnostic timed out after {args.timeout}s.'
            if stopped_reason:
                magia_sdk.stop_process_group(process)
                break
            time.sleep(1)
        result = process.returncode
    (output / 'exit-status').write_text(str(result) + '\n')
    (output / 'run.json').write_text(json.dumps({'command': command, 'exit_status': result,
                                               'wall_seconds': time.time() - start,
                                               'blocking_fifo': args.blocking_fifo,
                                               'model_directory': env.get('GVRUN_MODEL_PATH', str(args.sdk_root / 'gvsoc/install/models')),
                                               'stopped_reason': stopped_reason}, indent=2) + '\n')
    summary = summarize(output)
    print(f'Simulator exited {result} after {time.time() - start:.1f}s', flush=True)
    print(f"Validation passed={summary['passed']}; checked {summary['checked_elements']} elements; mismatches={summary['mismatches']}", flush=True)
    raise SystemExit(result if result else (0 if summary['passed'] else 1))


if __name__ == '__main__':
    main()
