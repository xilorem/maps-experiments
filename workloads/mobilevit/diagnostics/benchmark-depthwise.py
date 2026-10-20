#!/usr/bin/env python3
"""Compare MAPS direct FP16 depthwise with fused local im2col + GEMM.

Runs the stage-6 shard shape and edge cases on a 2x2 mesh. Uses real
MobileViT filter/bias values with deterministic synthetic activations.
Checks every output against a host FP16 FMA reference; times only dispatch.
Full-mesh convolution sources and the ongoing full simulation are untouched.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np

import magia_sdk


def generate(root, output, direct, fp32=False, edges_only=False, sdk=None, stride_suite=False, filter_suite=False, stage153=False, profile=False, active_tiles=4, vector_direct=False, vector_plan=False, stage75=False, stage=None):
    exact_stage=stage or (75 if stage75 else 153 if stage153 else None)
    app = output / 'application'
    app.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1706)
    cases = [
        dict(name='stage6', ih=128, iw=128, oh=128, ow=128, c=1, multiplier=1, stride=1, dilation=1, pad=1, tokens=2),
        dict(name='odd_halo_multiplier', ih=13, iw=19, oh=11, ow=17, c=2, multiplier=2, stride=1, dilation=2, pad=2, tokens=1, origin_h=7, origin_w=5, input_c=3, weight_c=6, row_padding=3),
        dict(name='stride2', ih=25, iw=35, oh=13, ow=18, c=1, multiplier=1, stride=2, dilation=1, pad=1, tokens=1),
        dict(name='strided_columns_fallback', ih=11, iw=17, oh=11, ow=17, c=1, multiplier=1, stride=1, dilation=1, pad=1, tokens=1, column_stride=4),
        dict(name='small_scratch_fallback', ih=11, iw=17, oh=11, ow=17, c=1, multiplier=1, stride=1, dilation=1, pad=1, tokens=1, scratch=512),
        dict(name='no_bias', ih=9, iw=15, oh=9, ow=15, c=2, multiplier=1, stride=1, dilation=1, pad=1, tokens=1, bias=False),
    ]
    if edges_only:
        cases = [
            dict(name='vertical_stride2_dma', ih=25, iw=17, oh=13, ow=17, c=1, multiplier=1, stride=2, stride_w=1, dilation=1, pad=1, tokens=2),
            dict(name='clipped_halo_empty_blocks', ih=5, iw=7, oh=11, ow=13, c=1, multiplier=1, stride=1, dilation=1, pad=1, tokens=2, origin_h=3, origin_w=3, output_origin_h=0, output_origin_w=0, scratch=1024),
        ]
    if stride_suite:
        cases = [
            dict(name='stage12', ih=128, iw=128, oh=64, ow=64, c=3, multiplier=1, stride=2, dilation=1, pad=1, tokens=2),
            dict(name='stride3_odd_width', ih=25, iw=31, oh=9, ow=11, c=2, multiplier=1, stride=3, dilation=1, pad=1, tokens=2, row_padding=3),
            dict(name='asymmetric_dilated_multiplier', ih=17, iw=23, oh=9, ow=9, c=3, multiplier=3, stride=2, stride_w=3, dilation=2, pad=2, tokens=2, origin_h=6, origin_w=9, output_origin_h=3, output_origin_w=3, input_c=2, weight_c=6),
            dict(name='large_stride_clipped', ih=5, iw=7, oh=7, ow=9, c=1, multiplier=1, stride=7, stride_w=11, dilation=1, pad=1, tokens=2, origin_h=3, origin_w=3, output_origin_h=0, output_origin_w=0, scratch=1024, bias=False),
        ]
    if filter_suite:
        cases = [
            dict(name='filter1x1', ih=9, iw=13, oh=9, ow=13, c=2, multiplier=1, stride=1, dilation=1, pad=0, tokens=2, kernel_h=1, kernel_w=1),
            dict(name='filter3x5', ih=13, iw=17, oh=13, ow=17, c=2, multiplier=1, stride=1, dilation=1, pad=1, pad_w=2, tokens=2, kernel_h=3, kernel_w=5),
            dict(name='filter5x5_stride2', ih=21, iw=25, oh=11, ow=13, c=2, multiplier=1, stride=2, dilation=1, pad=2, tokens=2, kernel_h=5, kernel_w=5),
            dict(name='filter2x4_dilated', ih=15, iw=19, oh=13, ow=7, c=2, multiplier=1, stride=1, stride_w=2, dilation=2, pad=0, tokens=2, kernel_h=2, kernel_w=4, bias=False),
        ]
    if vector_direct and not exact_stage:
        cases += [
            dict(name='vector_even_stride2', ih=18, iw=20, oh=9, ow=10, c=2, multiplier=1, stride=2, dilation=1, pad=1, tokens=2),
            dict(name='vector_odd_output', ih=18, iw=18, oh=9, ow=9, c=2, multiplier=1, stride=2, dilation=1, pad=1, tokens=2),
            dict(name='vector_stride4_multiplier', ih=18, iw=22, oh=9, ow=6, c=3, multiplier=3, stride=2, stride_w=4, dilation=1, pad=1, tokens=2),
            dict(name='vector_dilation2', ih=16, iw=20, oh=8, ow=10, c=2, multiplier=1, stride=2, dilation=2, pad=2, tokens=2),
            dict(name='vector_clipped_origins', ih=6, iw=8, oh=9, ow=11, c=2, multiplier=1, stride=2, dilation=1, pad=1, tokens=2, origin_h=6, origin_w=8, output_origin_h=0, output_origin_w=0),
        ]
    if vector_plan and not exact_stage:
        cases += [
            dict(name='row_template_odd_width', ih=18, iw=66, oh=9, ow=33, c=2, multiplier=1, stride=2, dilation=1, pad=1, tokens=2, row_padding=3),
            dict(name='row_template_clipped', ih=8, iw=82, oh=12, ow=40, c=2, multiplier=1, stride=2, dilation=1, pad=1, tokens=2, origin_h=6, origin_w=3, output_origin_h=0, output_origin_w=0),
            dict(name='row_template_dilated', ih=18, iw=128, oh=9, ow=64, c=2, multiplier=1, stride=2, dilation=2, pad=2, kernel_h=3, kernel_w=5, pad_w=4, tokens=2),
            dict(name='plan_multiple_blocks', ih=18, iw=34, oh=9, ow=17, c=2, multiplier=1, stride=2, dilation=1, pad=1, tokens=2),
            dict(name='plan_filter3x5', ih=18, iw=20, oh=9, ow=10, c=2, multiplier=1, stride=2, dilation=1, pad=1, pad_w=2, kernel_h=3, kernel_w=5, tokens=2),
            dict(name='plan_filter5x5_no_bias', ih=18, iw=20, oh=9, ow=10, c=2, multiplier=1, stride=2, dilation=1, pad=2, kernel_h=5, kernel_w=5, bias=False, tokens=2),
            dict(name='plan_large_filter_fallback', ih=18, iw=20, oh=8, ow=9, c=2, multiplier=1, stride=2, dilation=1, pad=2, kernel_h=7, kernel_w=8, tokens=2),
        ]
    if vector_plan and not exact_stage:
        for stride in (1,3,5,7,11):
            cases.append(dict(name=f'odd_stride{stride}', ih=20, iw=22, oh=(20+stride-1)//stride, ow=(22+stride-1)//stride,
                c=3, multiplier=3 if stride==5 else 1, stride=stride, dilation=1, pad=1, tokens=2, bias=stride!=7, origin_h=3 if stride==7 else 0, origin_w=5 if stride==7 else 0, output_origin_h=0, output_origin_w=0))
        cases += [
            dict(name='large_positive_stride', ih=6, iw=8, oh=3, ow=9, c=2, multiplier=1, stride=1073741825, dilation=1, pad=1, tokens=2),
            dict(name='odd_stride3_wide_blocks', ih=20, iw=198, oh=7, ow=66, c=2, multiplier=1, stride=3, dilation=1, pad=1, tokens=2),
            dict(name='odd_stride5_row_template', ih=20, iw=196, oh=4, ow=40, c=2, multiplier=1, stride=5, dilation=1, pad=1, tokens=2),
            dict(name='odd_asymmetric_dilated', ih=20, iw=22, oh=10, ow=8, c=3, multiplier=3, stride=2, stride_w=3, dilation=2, pad=2, tokens=2),
        ]
    if stage153:
        cases = [dict(name='stage153', ih=16, iw=16, oh=8, ow=8, c=192, multiplier=1, stride=2, dilation=1, pad=1, tokens=2)]
    if stage75:
        cases = [dict(name='stage75', ih=32, iw=32, oh=16, ow=16, c=64, multiplier=1, stride=2, dilation=1, pad=1, tokens=2)]
    if exact_stage and exact_stage!=27:
        c,ih,oh,stride={75:(64,32,16,2),79:(64,16,16,1),153:(192,16,8,2),157:(256,8,8,1)}[exact_stage]
        cases=[dict(name=f'stage{exact_stage}',ih=ih,iw=ih,oh=oh,ow=oh,c=c,multiplier=1,stride=stride,dilation=1,pad=1,tokens=2)]
    if exact_stage==27:
        cases=[]
        for source_tile in (791,822,823,824,853,854,855,884,885,886,917):
            text=(root/f'maps-application/src/tiles/tile_{source_tile:04d}.c').read_text()
            c=int(re.search(r'\.shape = \{(\d+), 1, 3, 3',text)[1])
            first=int(re.search(r'\.full_offset = \{(\d+),',text)[1])
            cases.append(dict(name=f'stage27_tile{source_tile}',ih=64,iw=64,oh=32,ow=32,c=c,multiplier=1,stride=2,dilation=1,pad=1,tokens=2,input_c=first,weight_c=first))
    initializers = root / 'maps-application/data/maps_mobilevit.initializers.bin'
    source_tile = {27:791,75:120,79:982,153:735,157:897}[exact_stage] if exact_stage else 714 if stride_suite else 320
    source = (root / f'maps-application/src/tiles/tile_{source_tile:04d}.c').read_text()
    offsets = [int(v) for v in re.findall(r'\.src_l2_addr = \(uint32_t\)maps_mobilevit_initializers_start \+ (\d+)u', source)]
    channel_origin = int(re.search(r'\.full_offset = \{(\d+),', source)[1])
    source_channels = {27:128,75:256,79:192,153:384,157:256}[exact_stage] if exact_stage else 32
    weights = np.fromfile(initializers, dtype='<f2', count=source_channels*9, offset=offsets[0]-channel_origin*18).reshape(source_channels, 3, 3)
    biases = np.fromfile(initializers, dtype='<f2', count=source_channels, offset=offsets[1]-channel_origin*2)
    asm, declarations, descriptors, manifest = [], [], [], []
    def blob(name, values):
        path = app / (name + '.bin')
        values.tofile(path)
        asm.extend([f'.section .l2_bulk.{name},"a",@progbits', '.balign 16', f'.global {name}', f'{name}:', f'.incbin "{path}"'])
        declarations.append(f'extern const uint16_t {name}[];')
    for tile in range(active_tiles):
        for ci, case in enumerate(cases):
            ih, iw, oh, ow, c, mult = (case[k] for k in ['ih', 'iw', 'oh', 'ow', 'c', 'multiplier'])
            ic = (c + mult - 1) // mult
            x = rng.uniform(-1, 1, (case['tokens'], ic, ih, iw)).astype(np.float16)
            w = weights[31-tile-c+1:32-tile].copy()
            b = biases[31-tile-c+1:32-tile].copy()
            if exact_stage:
                first=case['weight_c'] if exact_stage==27 else (tile % (source_channels//c))*c
                w=weights[first:first+c].copy(); b=biases[first:first+c].copy()
            kh,kw=case.get('kernel_h',3),case.get('kernel_w',3)
            if filter_suite or (kh,kw)!=(3,3):
                w=rng.uniform(-.5,.5,(c,kh,kw)).astype(np.float16)
            y = np.zeros((case['tokens'], c, oh, ow), dtype=np.float32 if fp32 else np.float16)
            if fp32 and case.get('bias', True):
                y += b[None, :, None, None].astype(np.float32)
            for ky in range(kh):
                for kx in range(kw):
                    yy = (np.arange(oh)+case.get('output_origin_h',case.get('origin_h',0)))*case['stride']-case.get('origin_h',0)-case['pad']+ky*case['dilation']
                    xx = (np.arange(ow)+case.get('output_origin_w',case.get('origin_w',0)))*case.get('stride_w',case['stride'])-case.get('origin_w',0)-case.get('pad_w',case['pad'])+kx*case['dilation']
                    samples = np.zeros_like(y)
                    valid_y = np.flatnonzero((yy >= 0) & (yy < ih))
                    valid_x = np.flatnonzero((xx >= 0) & (xx < iw))
                    for channel in range(c):
                        samples[:, channel, valid_y[:, None], valid_x] = x[:, channel//mult, yy[valid_y][:, None], xx[valid_x]]
                    y = (y.astype(np.float64) + samples.astype(np.float64)*w[None, :, ky, kx, None, None].astype(np.float64)).astype(np.float32 if fp32 else np.float16)
            if not fp32 and case.get('bias', True):
                y = (y.astype(np.float64) + b[None, :, None, None].astype(np.float64)).astype(np.float16)
            col = case.get('column_stride', 2)
            in_row = iw*col
            out_row = (ow + case.get('row_padding', 0))*col
            packed_x = np.zeros((case['tokens'], ic, ih, in_row//2), dtype='<u2')
            packed_x[..., :iw*(col//2):col//2] = x.view(np.uint16)
            prefix = f'case_{tile}_{ci}'
            for suffix, data in [('input', packed_x), ('weight', w.astype('<f2')), ('bias', b.astype('<f2')), ('golden', y.astype('<f2'))]:
                blob(prefix+'_'+suffix, data)
            descriptors.append('{' + ','.join([
                f'"{case["name"]}"', f'{prefix}_input', f'{prefix}_weight', f'{prefix}_bias', f'{prefix}_golden',
                *map(str, [ih, iw, oh, ow, ic, c, mult, case['stride'], case.get('stride_w',case['stride']), case['dilation'], case['pad'], case['tokens'], col, in_row, out_row, case.get('origin_h',0), case.get('origin_w',0), case.get('output_origin_h',case.get('origin_h',0)), case.get('output_origin_w',case.get('origin_w',0)), case.get('input_c',0), case.get('weight_c',0), int(case.get('bias',True)), case.get('scratch',65536),kh,kw,case.get('pad_w',case['pad'])])
            ]) + '}')
            manifest.append(dict(tile=tile, case=ci, **case))
    (app / 'data.S').write_text('\n'.join(asm)+'\n')
    cmake = '''set(MAPS_APPLICATION maps_depthwise_bench)
add_spatz_task(TEST_NAME ${MAPS_APPLICATION}
  TASK_SOURCES ${CMAKE_SOURCE_DIR}/kernels/depthwise_conv/fp16/spatz/spatz_task/depthwise_conv_fp16_spatz_task.c
  FIRST_TASK_NAME depthwise_conv_fp16_spatz_task
  INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/kernels/depthwise_conv/fp16/spatz/include)
add_executable(${MAPS_APPLICATION} main.c data.S)
target_compile_options(${MAPS_APPLICATION} PRIVATE -O2)
target_link_libraries(${MAPS_APPLICATION} PUBLIC runtime hal idma_lib redmule_lib fsync_lib eu_lib)
target_embed_spatz_task(TARGET ${MAPS_APPLICATION} TASK_NAME ${MAPS_APPLICATION})
target_compile_definitions(${MAPS_APPLICATION} PRIVATE MAPS_DEPTHWISE_VECTOR_AUTO=0 MAPS_HAS_DEPTHWISE_CONV_SPATZ_TASK=1 MAPS_FIFO_INPLACE_RECV=0 MAPS_TOKEN_PROGRESS=1)
'''
    if vector_plan:
        cmake += 'target_compile_definitions(${MAPS_APPLICATION} PRIVATE MAPS_DEPTHWISE_VECTOR_PLAN=1)\n'
    if vector_direct:
        cmake += 'target_compile_definitions(${MAPS_APPLICATION} PRIVATE MAPS_DEPTHWISE_VECTOR_DIRECT=1)\n'
    if profile:
        cmake += 'target_compile_definitions(${MAPS_APPLICATION} PRIVATE MAPS_DEPTHWISE_PROFILE=1)\n'
    cmake += f'target_compile_definitions(${{MAPS_APPLICATION}} PRIVATE MAPS_DEPTHWISE_IM2COL_GEMM={int(not direct)})\n'
    if fp32:
        original = subprocess.check_output(['git','show','HEAD:kernels/depthwise_conv/fp16/spatz/spatz_task/depthwise_conv_fp16_spatz_task.c'],cwd=sdk,text=True)
        (app/'original-depthwise.c').write_text(original)
        cmake = cmake.replace('${CMAKE_SOURCE_DIR}/kernels/depthwise_conv/fp16/spatz/spatz_task/depthwise_conv_fp16_spatz_task.c', '${CMAKE_CURRENT_SOURCE_DIR}/original-depthwise.c')
    (app/'CMakeLists.txt').write_text(cmake)
    main = (Path(__file__).with_name('benchmark-depthwise.c')).read_text()
    main = main.replace('@DECLARATIONS@', '\n'.join(declarations)).replace('@CASES@', ',\n'.join(descriptors)).replace('@CASE_COUNT@', str(len(cases))).replace('@ACTIVE_TILES@', str(active_tiles))
    (app/'main.c').write_text(main)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh-root', required=True, type=Path)
    parser.add_argument('--sdk-root', type=Path, default=Path('/home/zaino/repos/magia-sdk-v3'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--original-fp32', action='store_true', help='measure the original HEAD kernel with its FP32 reference; implies --direct')
    parser.add_argument('--edges-only', action='store_true', help='test vertical stride and wholly clipped im2col blocks')
    parser.add_argument('--stride-suite', action='store_true', help='benchmark stage 12 and arbitrary/asymmetric stride cases')
    parser.add_argument('--filter-suite', action='store_true', help='test 1x1, 3x5, 5x5 and 2x4 filters with strides and dilation')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--stage153', action='store_true', help='benchmark exact 192-channel stage-153 shard with real filters and biases')
    parser.add_argument('--profile', action='store_true', help='measure preparation, Spatz launch/wait and output-copy phases')
    parser.add_argument('--active-tiles', type=int, choices=(1,2,3,4), default=4, help='active workers on the isolated 2x2 mesh')
    parser.add_argument('--vector-direct', action='store_true', help='try aligned-word vector gathers on supported even-stride layouts')
    parser.add_argument('--vector-plan', action='store_true', help='reuse indexed-load geometry across channels and vectorize across output rows')
    parser.add_argument('--stage75', action='store_true', help='benchmark exact 64-channel stage-75 shard with real filters/biases')
    parser.add_argument('--stage', type=int, choices=(27,75,79,153,157), help='benchmark an exact saved depthwise shard')
    args = parser.parse_args()
    if args.vector_plan: args.vector_direct = True
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.original_fp32: args.direct = True
    app = generate(args.mesh_root.resolve(), output, args.direct, args.original_fp32, args.edges_only, args.sdk_root, args.stride_suite, args.filter_suite, args.stage153, args.profile, args.active_tiles, args.vector_direct, args.vector_plan, args.stage75, args.stage)
    args.sdk_root = args.sdk_root.resolve()
    env = magia_sdk.build_env(args.sdk_root)
    build = output/'sdk-build'
    with (output/'build.log').open('w') as log:
        magia_sdk.build(args.sdk_root, env, log, 2, build, app,
                        ['spatz_bootrom', 'maps_depthwise_bench'])
    print('Build complete', flush=True)
    work = output/'gvsoc'; work.mkdir(exist_ok=True)
    command = magia_sdk.gvrun_command(args.sdk_root, build, 'maps_depthwise_bench', work, 2)
    start = time.monotonic()
    with (output/'maps.log').open('w') as log:
        process = subprocess.Popen(command, cwd=args.sdk_root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (output/'pid').write_text(str(process.pid)+'\n')
        print(f'Simulator PID {process.pid}', flush=True)
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
    log = (output/'maps.log').read_text()
    checks = [dict(zip(['tile','case','token','result','cycles','elements','errors','im2col'],map(int,m))) for m in re.findall(r'DW_CHECK tile=(\d+) case=(\d+) token=(\d+) result=(-?\d+) cycles=(\d+) elements=(\d+) errors=(\d+) im2col=(\d+)',log)]
    profiles = [dict(zip(['tile','case','token','prepare','spatz','copy','gathers','launches','stores'],map(int,m))) for m in re.findall(r'DW_PROFILE tile=(\d+) case=(\d+) token=(\d+) prepare=(\d+) spatz=(\d+) copy=(\d+) gathers=(\d+) launches=(\d+) stores=(\d+)',log)]
    vector_profiles=[dict(zip(['tile','case','token','geometry','input','fma','output'],map(int,m))) for m in re.findall(r'DW_VECTOR_PROFILE tile=(\d+) case=(\d+) token=(\d+) geometry=(\d+) input=(\d+) fma=(\d+) output=(\d+)',log)]
    manifest = json.loads((output/'manifest.json').read_text())
    expected = {(c['tile'],c['case'],t) for c in manifest for t in range(c['tokens'])}
    actual = {(c['tile'],c['case'],c['token']) for c in checks}
    expected_modes = {(c['tile'],c['case']): int(not args.direct and c.get('column_stride',2)==2 and c.get('scratch',65536)!=512) for c in manifest}
    if args.vector_direct:
        for c in manifest:
            if c.get('column_stride',2)==2 and (c.get('stride_w',c['stride'])%2==0 or (args.vector_plan and c.get('kernel_h',3)*c.get('kernel_w',3)<=49)) and c['iw']%2==0 and c['iw']*2%4==0:
                expected_modes[c['tile'],c['case']]=0
    passed = (process.returncode==0 and len(checks)==len(expected) and actual==expected
              and all(c['result']==0 and c['errors']==0 and c['cycles']>0
                      and c['im2col']==expected_modes[c['tile'],c['case']] for c in checks)
              and 'DW_RESULT errors=0' in log)
    if args.profile:
        measured={(c['tile'],c['case'],c['token']):c for c in checks}
        passed = passed and len(profiles)==len(checks) and all(
            (p['tile'],p['case'],p['token']) in measured and
            p['prepare']+p['spatz']+p['copy'] <= measured[p['tile'],p['case'],p['token']]['cycles']
            for p in profiles)
        if args.stage153 and not args.direct and not args.vector_direct:
            passed = passed and all(p['gathers']==13248 and p['launches']==192 and p['stores']==192 for p in profiles)
    input_description = ('Deterministic synthetic activations and filters; MobileViT biases.' if args.filter_suite
                         else f'Deterministic synthetic activations; original MobileViT stage-{args.stage or (75 if args.stage75 else 153 if args.stage153 else 12 if args.stride_suite else 6)} weights and biases.')
    summary = dict(vector_profiles=vector_profiles, vector_plan=args.vector_plan, vector_direct=args.vector_direct, profile=args.profile, profiles=profiles, passed=passed, direct=args.direct, original_fp32=args.original_fp32, mesh='2x2', active_tiles=args.active_tiles, wall_seconds=time.monotonic()-start, exit_status=process.returncode, checked_elements=sum(c['elements'] for c in checks), mismatches=sum(c['errors'] for c in checks), checks=checks, command=command, input=input_description)
    (output/'statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['checks','command','profiles','vector_profiles']}), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__=='__main__': main()
