#!/usr/bin/env python3
"""Numerically exercise dedicated MAPS DMA reshape/transpose and Spatz depthwise.
Uses every distinct layout/origin/parameter combination in the frozen allocation.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np

import magia_sdk


blocks = magia_sdk.descriptor_blocks


def field(text, name):
    m = re.search(r'\.' + name + r' = (\{[^}]*\}|\w+)', text)
    if not m: raise ValueError(name)
    if m[1].startswith('{'): return [int(v.strip().rstrip('u')) for v in m[1][1:-1].split(',')]
    return int(m[1].rstrip('u')) if m[1][0].isdigit() else m[1]


def c_array(values): return '{' + ','.join(str(int(x)) + 'u' for x in values) + '}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--sdk-root', type=Path, default=magia_sdk.default_sdk_root())
    p.add_argument('--timeout', type=float, default=3600, help='maximum simulator wall time in seconds (default: 3600)')
    args = p.parse_args()
    root, output = args.mesh_root.resolve(), args.output.resolve()
    app = output / 'application'; app.mkdir(parents=True, exist_ok=True)
    cases, seen = [], set()
    for tile_path in sorted((root/'maps-application/src/tiles').glob('*.c')):
        text = tile_path.read_text()
        if not any('.kind = ' + k in text for k in ('OP_DEPTHWISE_CONV', 'OP_RESHAPE', 'OP_TRANSPOSE')): continue
        slices = blocks(re.search(r'static const slice_desc_t \w+\[\] = \{(.*?)\n\};', text, re.S)[1])
        ops = blocks(re.search(r'static const op_desc_t \w+\[\] = \{(.*?)\n\};', text, re.S)[1])
        for op in ops:
            kind = field(op, 'kind')
            if kind not in ('OP_DEPTHWISE_CONV','OP_RESHAPE','OP_TRANSPOSE'): continue
            inputs = blocks(op.split('.inputs = {',1)[1].split('},\n    .num_outputs',1)[0])[:field(op,'num_inputs')]
            outputs = blocks(op.split('.outputs = {',1)[1].split('},\n    .params',1)[0])[:field(op,'num_outputs')]
            subs = []
            for desc in inputs + outputs:
                rank = field(desc,'rank'); sid = field(desc,'slice_id')
                origin = [a+b for a,b in zip(field(slices[sid],'full_offset'),field(desc,'offset'))][:rank]
                subs.append({'rank':rank,'shape':field(desc,'shape')[:rank],'strides':field(desc,'strides_bytes')[:rank], 'origin':origin,'width':field(desc,'elem_bytes')})
            params = field(op,'params')
            key=json.dumps([kind,subs,params],sort_keys=True)
            if key in seen: continue
            seen.add(key); cases.append({'kind':kind,'subs':subs,'params':params,'example':tile_path.name})
    rng=np.random.default_rng(9121)
    assembly, entries, manifest = [], [], []
    offsets=[0x1000,0x50000,0x52000,0x60000]
    for i,case in enumerate(cases):
        kind,subs,params=case['kind'],case['subs'],case['params']; inputs=subs[:-1]; out=subs[-1]
        data, references = [], []
        for token in range(2):
            arrays=[]; buffers=[]
            for sub in inputs:
                span=sub['width']+sum((n-1)*stride for n,stride in zip(sub['shape'],sub['strides']))
                buf=np.zeros(span,dtype=np.uint8)
                arr=np.ndarray(tuple(sub['shape']),dtype='<f2',buffer=buf,strides=tuple(sub['strides']))
                arr[:]=rng.normal(0,.4,size=tuple(sub['shape'])).astype(np.float16)
                arrays.append(arr);buffers.append(buf)
            span=out['width']+sum((n-1)*stride for n,stride in zip(out['shape'],out['strides']))
            goldbuf=np.zeros(span,dtype=np.uint8);gold=np.ndarray(tuple(out['shape']),dtype='<f2',buffer=goldbuf,strides=tuple(out['strides']))
            if kind=='OP_RESHAPE': gold[:]=arrays[0].reshape(tuple(out['shape']))
            elif kind=='OP_TRANSPOSE':
                arr=arrays[0]; perm=params[:out['rank']]; selection=[]
                for axis in range(inputs[0]['rank']):
                    d=perm.index(axis);delta=out['origin'][d]-inputs[0]['origin'][axis]
                    selection.append(slice(delta,delta+out['shape'][d]))
                gold[:]=arr[tuple(selection)].transpose(tuple(perm))
            else:
                x,w=arrays[:2];bias=arrays[2] if len(arrays)>2 else None
                stride_h,stride_w,dh,dw,pad_h,pad_w,mult=params[:7]
                for n in range(out['shape'][0]):
                    for c in range(out['shape'][1]):
                        ic=(inputs[1]['origin'][0]+c)//mult-inputs[0]['origin'][1]
                        acc=np.full(tuple(out['shape'][2:]),np.float32(bias[c]) if bias is not None else np.float32(0),dtype=np.float32)
                        yy=np.arange(out['shape'][2])[:,None];xx=np.arange(out['shape'][3])[None,:]
                        for ky in range(w.shape[2]):
                            for kx in range(w.shape[3]):
                                iy=(out['origin'][2]+yy)*stride_h-pad_h+ky*dh-inputs[0]['origin'][2]
                                ix=(out['origin'][3]+xx)*stride_w-pad_w+kx*dw-inputs[0]['origin'][3]
                                valid=(iy>=0)&(iy<x.shape[2])&(ix>=0)&(ix<x.shape[3])
                                value=x[n,ic,np.clip(iy,0,x.shape[2]-1),np.clip(ix,0,x.shape[3]-1)].astype(np.float32)
                                acc=np.where(valid,acc+value*np.float32(w[c,0,ky,kx]),acc).astype(np.float32)
                        gold[n,c]=acc.astype(np.float16)
            for j,buf in enumerate(buffers):
                name=f'case_{i}_token_{token}_input_{j}';path=app/f'{name}.bin';buf.tofile(path)
                assembly.append(f'.section .l2_bulk.{name},"a",@progbits\n.balign 16\n.global {name}\n{name}:\n.incbin "{path}"\n')
                data.append((token,j,name,len(buf)))
            name=f'case_{i}_token_{token}_gold';path=app/f'{name}.bin';goldbuf.tofile(path)
            assembly.append(f'.section .l2_bulk.{name},"a",@progbits\n.balign 16\n.global {name}\n{name}:\n.incbin "{path}"\n');references.append(name)
        descs=[];slice_descs=[]
        for j,sub in enumerate(subs):
            sid=j if j<len(inputs) else 3
            descs.append('{.slice_id='+str(sid)+',.rank='+str(sub['rank'])+',.shape='+c_array(sub['shape'])+',.elem_type=ELEM_F16,.elem_bytes=2,.strides_bytes='+c_array(sub['strides'])+'}')
            slice_descs.append('['+str(sid)+']={.slice_id='+str(sid)+',.full_offset='+c_array(sub['origin'])+',.l1_offset_bytes='+str(offsets[sid])+',.num_slots=1}')
        blobs=[];sizes=[]
        for token in range(2):
            rows=[d for d in data if d[0]==token];blobs.append(c_array([]) if not rows else '{'+','.join(d[2] for d in rows)+'}');sizes.append('{'+','.join(str(d[3]) for d in rows)+'}')
        entry='{.kind_name="'+kind+'",.slices={'+','.join(slice_descs)+'},.op={.kind='+kind+',.num_inputs='+str(len(inputs))+',.inputs={'+','.join(descs[:-1])+'},.num_outputs=1,.outputs={'+descs[-1]+'},.params='+c_array(params)+'},.inputs={'+','.join(blobs)+'},.sizes={'+','.join(sizes)+'},.gold={'+','.join(references)+'},.output_bytes='+str(len(goldbuf))+'}'
        entries.append(entry);manifest.append({'id':i,**case,'output_bytes':len(goldbuf)})
    (app/'data.S').write_text('\n'.join(assembly))
    declarations='\n'.join('extern const uint8_t '+name+'[];' for a in assembly for name in re.findall(r'\.global (\w+)',a))
    template=Path(__file__).with_name('test-hw-adapters-main.c').read_text()
    (app/'main.c').write_text(template.replace('@DECLARATIONS@',declarations).replace('@CASES@',',\n'.join(entries)))
    (app/'CMakeLists.txt').write_text('''set(MAPS_APPLICATION test_maps_hw_adapters)
set(KERNEL_DIR ${CMAKE_SOURCE_DIR}/kernels/depthwise_conv/fp16/spatz)
add_spatz_task(TEST_NAME ${MAPS_APPLICATION} TASK_SOURCES ${KERNEL_DIR}/spatz_task/depthwise_conv_fp16_spatz_task.c FIRST_TASK_NAME depthwise_conv_fp16_spatz_task INCLUDE_DIRS ${KERNEL_DIR}/include)
add_executable(${MAPS_APPLICATION} main.c data.S)
target_compile_options(${MAPS_APPLICATION} PRIVATE -O2)
target_compile_definitions(${MAPS_APPLICATION} PRIVATE MAPS_HAS_DEPTHWISE_CONV_SPATZ_TASK=1)
target_link_libraries(${MAPS_APPLICATION} PUBLIC runtime hal idma_lib redmule_lib fsync_lib eu_lib)
target_embed_spatz_task(TARGET ${MAPS_APPLICATION} TASK_NAME ${MAPS_APPLICATION})
''')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Generated',len(cases),'distinct layouts',flush=True)
    sdk=args.sdk_root.resolve();build=output/'sdk-build';env=magia_sdk.build_env(sdk)
    with (output/'build.log').open('w') as log:
        magia_sdk.build(sdk,env,log,2,build,app,('spatz_bootrom','test_maps_hw_adapters'))
    work=output/'gvsoc';work.mkdir(exist_ok=True)
    start=time.monotonic()
    with (output/'maps.log').open('w') as log:
        proc=subprocess.Popen(magia_sdk.gvrun_command(sdk,build,'test_maps_hw_adapters',work,2),cwd=sdk,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            exit_status=proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            magia_sdk.stop_process_group(proc);exit_status='timeout'
    text=(output/'maps.log').read_text();checks=re.findall(r'HW_CHECK case=(\d+) token=(\d+) kind=(\w+) values=(\d+) errors=(\d+) code=(-?\d+)',text)
    passed=exit_status==0 and len(checks)==len(cases)*2 and all(int(c[4])==0 and int(c[5])==0 for c in checks)
    stats={'passed':passed,'exit_status':exit_status,'wall_seconds':time.monotonic()-start,'cases':len(cases),'checks':len(checks),'values':sum(int(c[3]) for c in checks),'errors':sum(int(c[4]) for c in checks)}
    (output/'statistics.json').write_text(json.dumps(stats,indent=2)+'\n');print(stats,flush=True)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
