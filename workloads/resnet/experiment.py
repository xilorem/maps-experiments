#!/usr/bin/env python3
"""Run complete ResNet MAPS and sequential full-mesh experiments."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
REPOS = ROOT.parents[2]
COMMON = ROOT.parent / 'mobilevit'
sys.path.insert(0, str(COMMON / 'diagnostics'))
import magia_sdk

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(model_path, sdk, output, tokens):
    """Pack the same FP16 parameters/inputs for both execution strategies."""
    output.mkdir(parents=True, exist_ok=True)
    model = onnx.shape_inference.infer_shapes(onnx.load(model_path))
    shapes = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
              for v in list(model.graph.input)+list(model.graph.value_info)+list(model.graph.output)}
    values = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    input_name = next(v.name for v in model.graph.input if v.name not in values)
    shape = shapes[input_name]
    if shape[0] != 1:
        raise ValueError('ResNet experiment expects batch one')
    inputs = np.random.default_rng(20260728).normal(0, .25, (tokens,)+shape).astype('<f2')
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    references = np.stack([session.run(None, {input_name: x.astype(np.float32)})[0] for x in inputs]).astype('<f2')
    (output/'inputs.bin').write_bytes(inputs.tobytes())
    (output/'references.bin').write_bytes(references.tobytes())
    (output/'input-name.txt').write_text(input_name)
    (output/'reference-available.txt').write_text('1\n')
    lines = []
    for token, ref in enumerate(references):
        checksum = 2166136261
        for byte in ref.tobytes(): checksum = ((checksum ^ byte)*16777619)&0xffffffff
        lines.append(f'{token},{checksum:08x}\n')
    (output/'reference-checksums.csv').write_text(''.join(lines))
    generator = load('resnet_blob_generator', sdk/'tests/spatz_on_magia/onnx_resnet18/test_data/generator.py')
    blob, arena = generator.Blob(), generator.Arena()
    inputs_off = blob.add(inputs)
    places = {input_name: (1, inputs_off)}
    entries = []
    kinds = {'Conv':'CONV','Relu':'RELU','MaxPool':'MAXPOOL','Add':'ADD',
             'GlobalAveragePool':'GLOBALAVERAGEPOOL','Gemm':'GEMM','Softmax':'SOFTMAX'}
    for node in model.graph.node:
        if node.op_type == 'Flatten':
            places[node.output[0]] = places[node.input[0]]
            continue
        if node.op_type not in kinds:
            raise ValueError(f'Unsupported full-mesh reference operation: {node.op_type}')
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == 'Conv' and (attrs.get('group',1)!=1 or tuple(attrs.get('dilations',[1,1]))!=(1,1)):
            raise ValueError('ResNet reference requires dense, unit-dilation convolution')
        if node.op_type == 'Gemm' and (attrs.get('transB',0)!=1 or attrs.get('transA',0)!=0 or attrs.get('alpha',1)!=1 or attrs.get('beta',1)!=1):
            raise ValueError('ResNet reference requires Gemm transB=1, alpha=beta=1')
        dims = shapes[node.output[0]]
        dest = arena.alloc(int(np.prod(dims)))
        sources = [places[n] for n in node.input if n in places]
        first = sources[0]; second = sources[1] if len(sources)>1 else (0,0)
        woff = boff = 0; bias = 0
        if node.op_type in ('Conv','Gemm'):
            woff = blob.add(values[node.input[1]])
            if len(node.input)>2:
                bias = 1; boff = blob.add(values[node.input[2]])
            elif node.op_type == 'Gemm':
                bias = 1; boff = blob.add(np.zeros(dims[-1],np.float16))
        ishape = shapes[node.input[0]]; ishape = ishape+(1,)*(4-len(ishape))
        oshape = dims+(1,)*(4-len(dims))
        kernel = attrs.get('kernel_shape',[0,0]); stride = attrs.get('strides',[0,0]); pad = attrs.get('pads',[0,0,0,0])
        fields = dict(op='RN18_'+kinds[node.op_type],has_bias=bias,
            src0_space=first[0],src0_off=first[1],src1_space=second[0],src1_off=second[1],dst_off=dest,
            w_off=woff,b_off=boff,in_shape='{'+','.join(map(str,ishape))+'}',out_shape='{'+','.join(map(str,oshape))+'}',
            kernel_h=kernel[0],kernel_w=kernel[1],stride_h=stride[0],stride_w=stride[1],pad_h=pad[0],pad_w=pad[1],
            numel=int(np.prod(dims)),name=json.dumps(f'{len(entries):02d}_{node.op_type}'))
        entries.append('{'+','.join(f'.{k}={v}' for k,v in fields.items())+'}')
        places[node.output[0]] = (0,dest)
    refs_off = blob.add(references); blob.write(output/'full-mesh-data.bin')
    header = (sdk/'tests/spatz_on_magia/onnx_resnet18/include/resnet18_graph.h').read_text().split('#define RN18_NUM_LAYERS')[0]
    macros = dict(NUM_LAYERS=len(entries),ARENA_BYTES=arena.size,INPUT_OFF=inputs_off,INPUTS_OFF=inputs_off,
        INPUT_ELEMENTS=int(np.prod(shape)),OUTPUT_OFF=places[model.graph.output[0].name][1],OUTPUT_LEN=int(np.prod(shapes[model.graph.output[0].name])),REFERENCES_OFF=refs_off)
    header += '\n'.join(f'#define RN18_{key} ({value})' for key,value in macros.items())
    header += '\nstatic const rn18_layer_t resnet18_layers[] = {\n'+',\n'.join(entries)+'\n};\n#endif\n'
    (output/'resnet18_graph.h').write_text(header)
    (output/'manifest.json').write_text(json.dumps({'input_shape':shape,'output_shape':shapes[model.graph.output[0].name],'tokens':tokens},indent=2)+'\n')


def reference_sources(sdk, output):
    output.mkdir(exist_ok=True)
    source = (sdk/'tests/spatz_on_magia/onnx_resnet18/main.c').read_text().split('int main(void)')[0]
    source = source.replace('#include "resnet18_graph.h"','#include "resnet18_graph.h"\n#include "utils/maps_operation_indexing.h"')
    source = source.replace('run_layer(const rn18_layer_t *L)', 'run_layer(const rn18_layer_t *L, uint32_t token)')
    source = source.replace('operand(L->src0_space, L->src0_off);', '(L->src0_space == RN18_SPACE_BLOB ? blob_at(RN18_INPUTS_OFF) + token * RN18_INPUT_ELEMENTS : operand(L->src0_space, L->src0_off));')
    tail = (COMMON/'full-mesh/main.c').read_text().split('static volatile uint32_t window_start',1)[1]
    tail = 'static volatile uint32_t window_start'+tail.replace('MVIT_','RN18_').replace('mobilevit_layers','resnet18_layers')
    tail = tail.replace('            run_layer(&resnet18_layers[i], token);', '''            uint32_t begin = read_cycle();
            if (HID == 0) printf("FULL_MESH_LAYER_BEGIN token=%u layer=%u cycle=%u name=%s\\n", token,i,begin,resnet18_layers[i].name);
            run_layer(&resnet18_layers[i], token);''')
    tail = tail.replace('            barrier();\n        }','''            barrier();
            if (HID == 0) printf("FULL_MESH_LAYER_DONE token=%u layer=%u cycle=%u elapsed_cycles=%u name=%s\\n",token,i,read_cycle(),read_cycle()-begin,resnet18_layers[i].name);
        }''',1)
    (output/'main.c').write_text(source+tail)
    (output/'resnet18_data.S.in').write_text('    .section .l2_bulk, "a", @progbits\n    .globl resnet18_data\n    .balign 4\nresnet18_data:\n    .incbin "@RN18_DATA@"\n')
    shim = output/'shim'; shim.mkdir(exist_ok=True)
    kernels = ['conv2dgemm','relu','add','maxpool2d','globalaveragepool','gemm','softmax']
    for name in kernels:
        (shim/f'{name}_fp16_spatz_task_bin.h').write_text('#include "onnx_resnet_experiment_task_bin.h"\n')
    cmake = '''set(TEST_NAME onnx_resnet_experiment)
set(KERNELS_DIR ${CMAKE_SOURCE_DIR}/kernels)
set(KERNELS '''+' '.join(kernels)+''')
foreach(K ${KERNELS})
 list(APPEND TASKS ${KERNELS_DIR}/${K}/fp16/spatz/spatz_task/${K}_fp16_spatz_task.c)
 list(APPEND SOURCES ${KERNELS_DIR}/${K}/fp16/spatz/src/${K}_fp16_spatz.c)
 list(APPEND INCLUDES ${KERNELS_DIR}/${K}/fp16/spatz/include)
endforeach()
list(APPEND TASKS ${KERNELS_DIR}/add/fp16/spatz/spatz_task/add_bcast_fp16_spatz_task.c)
add_spatz_task(TEST_NAME ${TEST_NAME} TASK_SOURCES ${TASKS} FIRST_TASK_NAME conv2dgemm_fp16_spatz_task INCLUDE_DIRS ${INCLUDES})
configure_file(resnet18_data.S.in ${CMAKE_CURRENT_BINARY_DIR}/resnet18_data.S @ONLY)
set_property(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/resnet18_data.S APPEND PROPERTY OBJECT_DEPENDS ${RN18_DATA})
add_cv32_executable_with_spatz(TARGET_NAME ${TEST_NAME}
 SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c ${CMAKE_CURRENT_BINARY_DIR}/resnet18_data.S ${SOURCES}
 INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/shim ${RN18_PREPARED} ${INCLUDES})
target_compile_definitions(${TEST_NAME} PRIVATE RN18_TOKENS=${RN18_TOKENS})
target_compile_options(${TEST_NAME} PRIVATE -O2)
'''
    (output/'CMakeLists.txt').write_text(cmake)


def snapshot_sdk(source, dest):
    dest.mkdir(exist_ok=True)
    for name in ['CMakeLists.txt','Makefile','cmake','devices','drivers','hal','kernels','scripts','targets','tests']:
        src, dst = source/name, dest/name
        if src.is_dir(): shutil.copytree(src,dst,dirs_exist_ok=True)
        else: shutil.copy2(src,dst)
    if not (dest/'gvsoc').exists(): (dest/'gvsoc').symlink_to(source/'gvsoc',target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',type=int,default=2)
    parser.add_argument('--token-slots',type=int,default=2)
    parser.add_argument('--mesh',type=int,default=32)
    parser.add_argument('--maps-only',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--resume-run',type=Path)
    parser.add_argument('--model',type=Path,default=next((ROOT/'model').glob('*.onnx')))
    parser.add_argument('--instruction-trace',action='store_true')
    parser.add_argument('--trace-tiles',default='all')
    args = parser.parse_args()
    if not 1<=args.tokens<=16 or args.token_slots<1 or args.mesh<1: parser.error('tokens must be 1..16, slots and mesh positive')
    if args.trace_tiles != 'all':
        if not re.fullmatch(r'(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*', args.trace_tiles):
            parser.error('--trace-tiles must be all or comma-separated tile IDs')
        if any(int(tile)>=args.mesh*args.mesh for tile in args.trace_tiles.split(',')):
            parser.error('--trace-tiles contains a tile outside the mesh')
        args.instruction_trace = True
    root = (args.resume_run or ROOT/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
    root.mkdir(parents=True,exist_ok=True)
    print(f'Run directory: {root}',flush=True)
    print(f'Results CSV: {root / "results.csv"}',flush=True)
    config_path = root/'run-config.json'
    if args.resume_run:
        saved = json.loads(config_path.read_text())
        for key in ('tokens','token_slots','mesh','maps_only','model'): setattr(args,key,saved[key])
        if (root/'results.csv').exists(): parser.error('Resume refused: results.csv already exists')
    else:
        config_path.write_text(json.dumps({k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2)+'\n')
    sdk_source = Path(os.environ.get('MAGIA_SDK_ROOT',REPOS/'magia-sdk-v3')).resolve()
    prepared = root/'prepared'
    if not args.resume_run: prepare(Path(args.model),sdk_source,prepared,args.tokens)
    if args.prepare_only: return 0
    sdk = root/'sdk-source'
    snapshot_sdk(sdk_source,sdk)
    mesh = root/f'{args.mesh}x{args.mesh}'; mesh.mkdir(exist_ok=True)
    app = mesh/'maps-application'; build = mesh/'sdk-build'
    monitor = load('resnet_monitor',COMMON/'diagnostics/run-monitored.py')
    env = monitor.simulation_environment(); env['CMAKE_BUILD_PARALLEL_LEVEL'] = os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL','4')
    def run(command,cwd,log):
        print('Running: '+' '.join(map(str,command)),flush=True)
        with log.open('a',buffering=1) as stream:
            proc = subprocess.Popen(list(map(str,command)),cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout: stream.write(line); print(line,end='',flush=True)
            if proc.wait(): raise subprocess.CalledProcessError(proc.returncode,command)
    maps_root = Path(os.environ.get('MAPS_ROOT',REPOS/'MAPS'))
    if not (app/'CMakeLists.txt').exists():
        run(['make','build',f'MODEL={Path(args.model).resolve()}','TARGET=magia-v3',f'MESH={args.mesh}x{args.mesh}',f'TOKEN_SLOTS={args.token_slots}',
             'NAME=maps_mobilevit',f'INPUT={(prepared/"input-name.txt").read_text()}={prepared}/inputs.bin',
             f'EXECUTION_PLAN={mesh}/execution-plan.json',f'APPLICATION={app}'],maps_root,mesh/'build.log')
    monitor.prepare_progress(app)
    reference_sources(sdk,root/'full-mesh')
    cmake = app/'CMakeLists.txt'; text = cmake.read_text()
    if 'RESNET_EXPERIMENT_OPTIONS' not in text:
        text += '\n# RESNET_EXPERIMENT_OPTIONS\ntarget_compile_definitions(maps_mobilevit PRIVATE MAPS_FIFO_EVENT_SLEEP=1 MAPS_RESULT_TRACE=1)\n'
        if not args.maps_only:
            text += f'\nset(RN18_DATA "{prepared}/full-mesh-data.bin")\nset(RN18_PREPARED "{prepared}")\nset(RN18_TOKENS {args.tokens})\nadd_subdirectory("{root}/full-mesh" full_mesh)\n'
        cmake.write_text(text)
    targets = ['spatz_bootrom','maps_mobilevit']+([] if args.maps_only else ['onnx_resnet_experiment'])
    for target in targets:
        run(['make','build',f'tiles={args.mesh}','pulp_cores=0',f'CMAKE_BUILDDIR={build}',f'maps_application_dir={app}',
             'maps_token_progress=1','maps_kernel_progress=1',f'test={target}'],sdk,mesh/'build.log')
    if args.build_only:
        print('Build complete; no simulator started.',flush=True); return 0
    from fifo_sleep_tree import build_platform_tree
    tree = build_platform_tree(sdk,args.mesh,mesh/'fifo-platform-tree',env)
    (mesh/'maps-monitor').mkdir(exist_ok=True)
    ns = argparse.Namespace(work_dir=mesh/'maps-gvsoc',platform_tree=tree,fifo_event_sleep=True,
        instruction_trace=args.instruction_trace,trace_tiles=args.trace_tiles,maps_log=mesh/'maps.log',tokens=args.tokens)
    if monitor.run_simulation(ns,mesh,sdk,mesh/'maps-monitor',app,build,env): return 1
    reporting = load('resnet_reporting',COMMON/'results.py')
    reporting.collect_maps(argparse.Namespace(maps_log=mesh/'maps.log',mesh=f'{args.mesh}x{args.mesh}',tokens=args.tokens,
        reference_checksums=prepared/'reference-checksums.csv',csv=root/'results.csv'))
    if not args.maps_only:
        work = mesh/'full-mesh-gvsoc'; work.mkdir(exist_ok=True)
        command = magia_sdk.gvrun_command(sdk, build, 'onnx_resnet_experiment', work, args.mesh,
            ['--attr', 'magia_v3/fifo_events=1', '--gv-opt', f'platform_tree={tree}'])
        run(command,sdk,mesh/'full-mesh.log')
        reporting.collect_full_mesh(argparse.Namespace(full_mesh_log=mesh/'full-mesh.log',mesh=f'{args.mesh}x{args.mesh}',tokens=args.tokens,
            reference_available=prepared/'reference-available.txt',csv=root/'results.csv'))
    print(f'Results: {root / "results.csv"}',flush=True)
    return 0

if __name__ == '__main__': raise SystemExit(main())
