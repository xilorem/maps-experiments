#!/usr/bin/env python3
"""Numerically exercise dedicated MAPS DMA reshape/transpose and Spatz depthwise.
Uses every distinct layout/origin/parameter combination in the frozen allocation.
"""
import argparse
import json
from pathlib import Path
import re
import struct
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
    p.add_argument('--maxpool-only', action='store_true')
    p.add_argument('--im2col-sweep', action='store_true', help='compare tiny and full-workload im2col geometry')
    p.add_argument('--mesh-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--sdk-root', type=Path, default=magia_sdk.default_sdk_root())
    p.add_argument('--timeout', type=float, default=3600, help='maximum simulator wall time in seconds (default: 3600)')
    p.add_argument('--generic-transpose', action='store_true', help='disable packed-pair transpose for cycle comparison')
    p.add_argument('--scalar-transpose', action='store_true', help='disable Spatz transpose for packed-scalar comparison')
    p.add_argument('--stage70-only', action='store_true', help='test only the two exact stage-70 tile layouts')
    p.add_argument('--raw-transpose', action='store_true', help='check arbitrary 16-bit patterns, including NaNs and signed zero')
    p.add_argument('--transpose-edges', action='store_true', help='test additional ranks, odd lengths, permutations and cropped origins')
    p.add_argument('--reduce-only', action='store_true', help='test frozen sum reductions and additional axes/layouts')
    p.add_argument('--mul-only', action='store_true', help='test frozen multiplication descriptors and scalar constants')
    p.add_argument('--matmul-only', action='store_true', help='test classifier and transposed/batched GEMM edge cases')
    p.add_argument('--scalar-matmul', action='store_true', help='disable transposed-weight vectors for comparison')
    p.add_argument('--stage0', action='store_true', help='profile and validate exact stage-0 im2col and GEMM descriptors')
    p.add_argument('--profile-gemm', action='store_true', help='split Spatz GEMM arithmetic and bias cycle counts')
    p.add_argument('--profile-im2col', action='store_true', help='split control-core im2col zeroing and DMA timing')
    p.add_argument('--scalar-bias', action='store_true', help='disable vector bias addition for comparison')
    args = p.parse_args()
    root, output = args.mesh_root.resolve(), args.output.resolve()
    app = output / 'application'; app.mkdir(parents=True, exist_ok=True)
    cases, seen = [], set()
    for tile_path in sorted((root/'maps-application/src/tiles').glob('*.c')):
        if args.maxpool_only or args.im2col_sweep: continue
        if args.stage0 and tile_path.name not in tuple(f'tile_{i:04d}.c' for i in range(416,424)): continue
        if args.matmul_only and tile_path.name != 'tile_1021.c': continue
        if args.stage70_only and tile_path.name not in ('tile_0058.c','tile_0059.c'): continue
        text = tile_path.read_text()
        if not any('.kind = ' + k in text for k in (('OP_IM2COL','OP_MATMUL') if args.stage0 else ('OP_MATMUL',) if args.matmul_only else ('OP_MUL',) if args.mul_only else ('OP_REDUCE_SUM',) if args.reduce_only else ('OP_DEPTHWISE_CONV', 'OP_RESHAPE', 'OP_TRANSPOSE'))): continue
        slices = blocks(re.search(r'static const slice_desc_t \w+\[\] = \{(.*?)\n\};', text, re.S)[1])
        ops = blocks(re.search(r'static const op_desc_t \w+\[\] = \{(.*?)\n\};', text, re.S)[1])
        for op in ops:
            kind = field(op, 'kind')
            if args.mul_only and field(op, 'num_inputs') != 1: continue
            if kind not in (('OP_IM2COL','OP_MATMUL') if args.stage0 else ('OP_MATMUL',) if args.matmul_only else ('OP_MUL',) if args.mul_only else ('OP_REDUCE_SUM',) if args.reduce_only else ('OP_DEPTHWISE_CONV','OP_RESHAPE','OP_TRANSPOSE')): continue
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
    if args.maxpool_only:
        for name,n,c,ih,iw,oh,ow,kh,kw,sh,sw,ph,pw in [
            ('resnet_pool_shard',1,2,112,112,56,56,3,3,2,2,1,1),
            ('negative_padding',1,2,5,8,3,4,3,3,2,2,1,1),
            ('rectangular_ceil_batch',2,3,5,7,3,4,2,3,2,2,0,1),
            ('odd_width_stride1',1,2,5,7,5,7,3,3,1,1,1,1),
            ('stride3_asymmetric',1,3,7,8,3,3,3,2,3,3,1,0),
            ('one_by_one',1,2,3,8,3,8,1,1,1,1,0,0),
        ]:
            def sub(shape):
                strides=[2]*4
                for d in range(2,-1,-1): strides[d]=strides[d+1]*shape[d+1]
                return dict(rank=4,shape=shape,strides=strides,origin=[0]*4,width=2)
            cases.append(dict(kind='OP_MAX_POOL',subs=[sub([n,c,ih,iw]),sub([n,c,oh,ow])],params=[kh,kw,sh,sw,ph,pw,0,0],example=name))
    if args.transpose_edges:
        cases=[]
        for shape,perm,lengths,starts in [
            ([257],[0],None,None),
            ([5,17],[1,0],None,None),
            ([3,5,9],[2,0,1],None,None),
            ([2,3,5,7],[2,0,3,1],[3,2,4,2],[1,0,1,1]),
            ([2,3,2,3,5,7],[5,3,1,4,0,2],None,None),
            ([3,2,17],[0,2,1],None,None),
        ]:
            def dense_strides(dims):
                result=[2]*len(dims)
                for i in range(len(dims)-2,-1,-1):result[i]=result[i+1]*dims[i+1]
                return result
            origin=[4+i*3 for i in range(len(shape))]
            dims=lengths or [shape[i] for i in perm]
            starts=starts or [0]*len(shape)
            subs=[{'rank':len(shape),'shape':shape,'strides':dense_strides(shape),'origin':origin,'width':2},
                  {'rank':len(shape),'shape':dims,'strides':dense_strides(dims),'origin':[origin[axis]+starts[i] for i,axis in enumerate(perm)],'width':2}]
            cases.append({'kind':'OP_TRANSPOSE','subs':subs,'params':perm+[0]*(8-len(perm)),'example':'synthetic-transpose-edge'})
    if args.reduce_only:
        for shape in ([2,3,4], [2,3,5], [1,64,8,1]):
            for axis in range(len(shape)):
                for padded in (False, True):
                    dims=list(shape); dims[axis]=1
                    def strides(dims):
                        result=[2]*len(dims)
                        for d in range(len(dims)-2,-1,-1):result[d]=result[d+1]*dims[d+1]
                        if padded: result=[v*2 for v in result]
                        return result
                    subs=[{'rank':len(shape),'shape':list(shape),'strides':strides(shape),'origin':[0]*len(shape),'width':2},
                          {'rank':len(dims),'shape':dims,'strides':strides(dims),'origin':[0]*len(dims),'width':2}]
                    cases.append({'kind':'OP_REDUCE_SUM','subs':subs,'params':[axis]+[0]*7,'example':'synthetic-reduce-axis'})
    if args.mul_only:
        for length in (1, 3, 64, 257, 1024):
            for constant in (0., 1/64, -2., .1):
                sub={'rank':1,'shape':[length],'strides':[2],'origin':[0],'width':2}
                bits=struct.unpack('<I',struct.pack('<f',constant))[0]
                cases.append({'kind':'OP_MUL','subs':[dict(sub),dict(sub)],'params':[bits]+[0]*7,'example':'synthetic-constant-mul'})
    if args.mul_only:
        for shape,bshape in (([2,8],[1,8]),([2,8],[2,1]),([2,7],[1,7]),([2,7],[2,1])):
            subs=[{'rank':2,'shape':v,'strides':[v[1]*2,2],'origin':[0,0],'width':2} for v in (shape,bshape,shape)]
            cases.append({'kind':'OP_MUL','subs':subs,'params':[0]*8,'example':'synthetic-broadcast-mul'})
    if args.matmul_only:
        for m,k,o,batches,ab,bb,bias,trans,unaligned in [
            (1,2,1,1,0,0,'none',1,False),
            (3,16,65,1,0,0,'column',1,False),
            (2,8,9,1,0,0,'row',1,False),
            (3,16,65,2,1,1,'matrix',1,False),
            (2,16,65,2,0,1,'scalar',1,False),
            (3,16,65,2,1,1,'matrix_shared',1,False),
            (2,16,65,2,1,0,'column',1,False),
            (2,7,9,1,0,0,'column',1,False),
            (2,8,9,1,0,0,'column',1,True),
            (2,8,10,1,0,0,'column',0,False),
            (1,256,1001,1,0,0,'column',1,False),
        ]:
            def dense(dims,base_offset=0):
                strides=[2]*len(dims)
                for d in range(len(dims)-2,-1,-1):strides[d]=strides[d+1]*dims[d+1]
                return {'rank':len(dims),'shape':dims,'strides':strides,'origin':[0]*len(dims),'width':2,'base_offset':base_offset}
            subs=[dense(([batches] if ab else [])+[m,k]),dense(([batches] if bb else [])+([o,k] if trans else [k,o]),2 if unaligned else 0)]
            if bias!='none':
                dims={'column':[o],'row':[m,1],'matrix':[batches,m,o],'matrix_shared':[m,o],'scalar':[1]}[bias]
                subs.append(dense(dims))
            subs.append(dense(([batches] if batches>1 else [])+[m,o],2 if unaligned else 0))
            cases.append({'kind':'OP_MATMUL','subs':subs,'params':[m,k,o,batches,ab,bb,0,trans],'example':'synthetic-matmul-'+bias})
    if args.stage0:
        for name,ih,iw,c,kh,kw,sh,sw,dh,dw,ph,pw,oh,ow,local_h in [
            ('one_by_one',7,8,2,1,1,1,1,1,1,0,0,7,8,7),
            ('rectangular_dilated',7,8,2,3,5,1,1,2,2,2,4,7,8,7),
            ('stride_three',7,8,1,3,3,3,3,1,1,1,1,3,3,3),
            ('wholly_padded',2,2,1,3,3,1,1,1,1,5,5,10,10,2),
        ]:
            def sub(dims):
                strides=[2]*len(dims)
                for d in range(len(dims)-2,-1,-1):strides[d]=strides[d+1]*dims[d+1]
                return {'rank':len(dims),'shape':dims,'strides':strides,'origin':[0]*len(dims),'width':2}
            cases.append({'kind':'OP_IM2COL','subs':[sub([1,c,ih,iw]),sub([c*kh*kw,local_h*ow])],
                'params':[kh|(kw<<16),sh|(sw<<16),dh|(dw<<16),ph|(pw<<16),ph|(pw<<16),ih|(iw<<16),oh|(ow<<16),1],
                'example':'synthetic-im2col-'+name})
    if args.im2col_sweep:
        saved=root/'maps-gvsoc-diagnostic/stage0-padding-vector-bias-profile/manifest.json'
        if saved.exists():
            cases.append(json.loads(saved.read_text())[0])
        for h,w,stride in [(8,8,1),(16,16,1),(32,32,1),(8,8,2),(16,16,2),(32,32,2)]:
            def sub(dims):
                strides=[2]*len(dims)
                for d in range(len(dims)-2,-1,-1):strides[d]=strides[d+1]*dims[d+1]
                return dict(rank=len(dims),shape=dims,strides=strides,origin=[0]*len(dims),width=2)
            oh,ow=(h+stride-1)//stride,(w+stride-1)//stride
            cases.append(dict(kind='OP_IM2COL',subs=[sub([1,3,h,w]),sub([27,oh*ow])],
                params=[3|(3<<16),stride|(stride<<16),1|(1<<16),1|(1<<16),1|(1<<16),h|(w<<16),oh|(ow<<16),1],
                example=f'im2col-{h}x{w}-stride{stride}'))
    im2col_references={}
    rng=np.random.default_rng(9121)
    assembly, entries, manifest = [], [], []
    offsets=[0x1000,0x10000,0x90000,0xA0000] if args.matmul_only or args.stage0 else [0x1000,0x50000,0x52000,0x60000]
    for i,case in enumerate(cases):
        kind,subs,params=case['kind'],case['subs'],case['params']; inputs=subs[:-1]; out=subs[-1]
        data, references = [], []
        for token in range(2):
            arrays=[]; buffers=[]
            for input_index,sub in enumerate(inputs):
                span=sub['width']+sum((n-1)*stride for n,stride in zip(sub['shape'],sub['strides']))
                buf=np.zeros(span,dtype=np.uint8)
                arr=np.ndarray(tuple(sub['shape']),dtype='<f2',buffer=buf,strides=tuple(sub['strides']))
                arr[:]=rng.normal(0,.4,size=tuple(sub['shape'])).astype(np.float16)
                if kind=='OP_MAX_POOL' and case['example']=='negative_padding': arr[:]=-np.abs(arr)-np.float16(.25)
                if args.raw_transpose and kind=='OP_TRANSPOSE':
                    arr.view(np.uint16)[:]=rng.integers(0,65536,size=tuple(sub['shape']),dtype=np.uint16)
                    arr.view(np.uint16).flat[:8]=[0x0000,0x8000,0x7c00,0xfc00,0x7e01,0xfe55,0x0001,0xffff]
                if args.stage0 and kind=='OP_MATMUL':
                    if input_index==1:
                        arr[:]=im2col_references[case['example'],token]
                    elif input_index in (0,2):
                        tile_text=(root/'maps-application/src/tiles'/case['example']).read_text()
                        sources=[int(x) for x in re.findall(r'\.src_l2_addr = \(uint32_t\)maps_mobilevit_initializers_start \+ (\d+)u',tile_text)]
                        arr[:]=np.fromfile(root/'maps-application/data/maps_mobilevit.initializers.bin',dtype='<f2',offset=sources[0 if input_index==0 else 1],count=arr.size).reshape(arr.shape)
                if args.matmul_only and case['example']=='tile_1021.c' and input_index in (1,2):
                    tile_text=(root/'maps-application/src/tiles/tile_1021.c').read_text()
                    sources=[int(x) for x in re.findall(r'\.src_l2_addr = \(uint32_t\)maps_mobilevit_initializers_start \+ (\d+)u',tile_text)]
                    arr[:]=np.fromfile(root/'maps-application/data/maps_mobilevit.initializers.bin',dtype='<f2',offset=sources[input_index-1],count=arr.size).reshape(arr.shape)
                arrays.append(arr);buffers.append(buf)
            span=out['width']+sum((n-1)*stride for n,stride in zip(out['shape'],out['strides']))
            goldbuf=np.zeros(span,dtype=np.uint8);gold=np.ndarray(tuple(out['shape']),dtype='<f2',buffer=goldbuf,strides=tuple(out['strides']))
            if kind=='OP_MAX_POOL':
                kh,kw,sh,sw,ph,pw,_,_=params
                x=arrays[0]
                for yy in range(out['shape'][2]):
                    for xx in range(out['shape'][3]):
                        top,left=yy*sh-ph,xx*sw-pw
                        window=x[:,:,max(0,top):min(x.shape[2],top+kh),max(0,left):min(x.shape[3],left+kw)]
                        gold[:,:,yy,xx]=window.max(axis=(-2,-1))
            elif kind=='OP_IM2COL':
                kh,kw=params[0]&65535,params[0]>>16
                sh,sw=params[1]&65535,params[1]>>16
                dh,dw=params[2]&65535,params[2]>>16
                ph,pw=params[3]&65535,params[3]>>16
                ih,iw=params[5]&65535,params[5]>>16
                ow=params[6]>>16
                x=arrays[0]; origin=inputs[0]['origin']
                positions=out['origin'][1]+np.arange(out['shape'][1])
                for row in range(out['shape'][0]):
                    kernel=out['origin'][0]+row
                    channel=kernel//(kh*kw)-origin[1]
                    yy=(positions//ow)*sh-ph+(kernel%(kh*kw)//kw)*dh
                    xx=(positions%ow)*sw-pw+(kernel%kw)*dw
                    valid=(yy>=0)&(yy<ih)&(xx>=0)&(xx<iw)&(yy>=origin[2])&(yy<origin[2]+x.shape[2])&(xx>=origin[3])&(xx<origin[3]+x.shape[3])
                    gold[row,valid]=x[0,channel,yy[valid]-origin[2],xx[valid]-origin[3]]
                im2col_references[case['example'],token]=gold.copy()
            elif kind=='OP_MATMUL':
                m,k,o,batches,ab,bb,row_bias,trans=params
                result=np.zeros((batches,m,o),dtype=np.float16)
                for batch in range(batches):
                    a=arrays[0][batch] if ab else arrays[0]
                    b=arrays[1][batch] if bb else arrays[1]
                    for depth in range(k):
                        weights=b[:,depth] if trans else b[depth,:]
                        result[batch]=(result[batch].astype(np.float64)+a[:,depth,None].astype(np.float64)*weights[None,:].astype(np.float64)).astype(np.float16)
                if len(arrays)>2:
                    bias=arrays[2]
                    if row_bias: bias=bias.reshape((batches,m,1) if batches>1 and bias.size==batches*m else (m,1))
                    result=(result.astype(np.float64)+bias.astype(np.float64)).astype(np.float16)
                gold[:]=result.reshape(gold.shape)
            elif kind=='OP_MUL':
                rhs=arrays[1].astype(np.float32) if len(arrays)>1 else struct.unpack('<f',struct.pack('<I',params[0]))[0]
                gold[:]=(arrays[0].astype(np.float32)*rhs).astype(np.float16)
            elif kind=='OP_REDUCE_SUM':
                # Exact small integers keep both FP16 and FP32 accumulation exact.
                arrays[0][:]=rng.integers(-3,4,size=arrays[0].shape).astype(np.float16)
                gold[:]=arrays[0].astype(np.float32).sum(axis=params[0],keepdims=True).reshape(gold.shape).astype(np.float16)
            elif kind=='OP_RESHAPE': gold[:]=arrays[0].reshape(tuple(out['shape']))
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
                        acc=np.zeros(tuple(out['shape'][2:]),dtype=np.float16)
                        yy=np.arange(out['shape'][2])[:,None];xx=np.arange(out['shape'][3])[None,:]
                        for ky in range(w.shape[2]):
                            for kx in range(w.shape[3]):
                                iy=(out['origin'][2]+yy)*stride_h-pad_h+ky*dh-inputs[0]['origin'][2]
                                ix=(out['origin'][3]+xx)*stride_w-pad_w+kx*dw-inputs[0]['origin'][3]
                                valid=(iy>=0)&(iy<x.shape[2])&(ix>=0)&(ix<x.shape[3])
                                value=x[n,ic,np.clip(iy,0,x.shape[2]-1),np.clip(ix,0,x.shape[3]-1)].astype(np.float64)
                                # Evaluate the exact binary16 product and sum,
                                # then round once per FP16 fused multiply-add.
                                acc=np.where(valid,acc.astype(np.float64)+value*np.float64(w[c,0,ky,kx]),acc).astype(np.float16)
                        if bias is not None:
                            acc=(acc.astype(np.float64)+np.float64(bias[c])).astype(np.float16)
                        gold[n,c]=acc
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
            slice_descs.append('['+str(sid)+']={.slice_id='+str(sid)+',.full_offset='+c_array(sub['origin'])+',.l1_offset_bytes='+str(offsets[sid]+sub.get('base_offset',0))+',.num_slots=1}')
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
add_spatz_task(TEST_NAME ${MAPS_APPLICATION}
  TASK_SOURCES ${KERNEL_DIR}/spatz_task/depthwise_conv_fp16_spatz_task.c
    ${CMAKE_SOURCE_DIR}/kernels/transpose/fp16/spatz/spatz_task/transpose_strided_fp16_spatz_task.c
    ${CMAKE_SOURCE_DIR}/kernels/reducesum/fp16/spatz/spatz_task/reducesum_fp16_spatz_task.c
    ${CMAKE_SOURCE_DIR}/kernels/mul/fp16/spatz/spatz_task/mul_bcast_fp16_spatz_task.c
    ${CMAKE_SOURCE_DIR}/kernels/matmul/fp16/spatz/spatz_task/matmul_rows_fp16_spatz_task.c
    ${CMAKE_SOURCE_DIR}/kernels/maxpool2d/fp16/spatz/spatz_task/maxpool2d_fp16_spatz_task.c
  FIRST_TASK_NAME depthwise_conv_fp16_spatz_task
  INCLUDE_DIRS ${KERNEL_DIR}/include ${CMAKE_SOURCE_DIR}/kernels/transpose/fp16/spatz/include ${CMAKE_SOURCE_DIR}/kernels/reducesum/fp16/spatz/include ${CMAKE_SOURCE_DIR}/kernels/mul/fp16/spatz/include ${CMAKE_SOURCE_DIR}/kernels/matmul/fp16/spatz/include ${CMAKE_SOURCE_DIR}/kernels/maxpool2d/fp16/spatz/include)
add_executable(${MAPS_APPLICATION} main.c data.S)
target_compile_options(${MAPS_APPLICATION} PRIVATE -O2)
target_compile_definitions(${MAPS_APPLICATION} PRIVATE MAPS_HAS_MAXPOOL_SPATZ_TASK=1 MAPS_HAS_MATMUL_SPATZ_TASK=1 MAPS_HAS_MUL_SPATZ_TASK=1 MAPS_HAS_REDUCE_SUM_SPATZ_TASK=1 MAPS_HAS_DEPTHWISE_CONV_SPATZ_TASK=1 MAPS_TOKEN_PROGRESS=1)
target_link_libraries(${MAPS_APPLICATION} PUBLIC runtime hal idma_lib redmule_lib fsync_lib eu_lib)
target_embed_spatz_task(TARGET ${MAPS_APPLICATION} TASK_NAME ${MAPS_APPLICATION})
''')
    if args.profile_im2col:
        with (app/'CMakeLists.txt').open('a') as cmake: cmake.write('\ntarget_compile_definitions(test_maps_hw_adapters PRIVATE MAPS_IM2COL_PROFILE=1)\n')
    if args.profile_gemm:
        cmake=app/'CMakeLists.txt'
        cmake.write_text('list(APPEND SPATZ_CFLAGS_DEFINES -DMAPS_MATMUL_PROFILE=1)\n'+cmake.read_text()+'\ntarget_compile_definitions(test_maps_hw_adapters PRIVATE MAPS_MATMUL_PROFILE=1)\n')
    if args.scalar_bias:
        cmake=app/'CMakeLists.txt'
        cmake.write_text('list(APPEND SPATZ_CFLAGS_DEFINES -DMATMUL_VECTOR_BIAS=0)\n'+cmake.read_text())
    if args.scalar_matmul:
        cmake=app/'CMakeLists.txt'
        cmake.write_text('list(APPEND SPATZ_CFLAGS_DEFINES -DMATMUL_TRANSPOSED_B_VECTOR=0)\n'+cmake.read_text())
    if args.generic_transpose:
        with (app/'CMakeLists.txt').open('a') as cmake:
            cmake.write('target_compile_definitions(test_maps_hw_adapters PRIVATE MAPS_TRANSPOSE_PACKED_PAIRS=0)\n')
    if not args.generic_transpose and not args.scalar_transpose:
        with (app/'CMakeLists.txt').open('a') as cmake:
            cmake.write('target_compile_definitions(test_maps_hw_adapters PRIVATE MAPS_HAS_TRANSPOSE_SPATZ_TASK=1)\n')
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
    cycles=[{'case':int(c),'token':int(t),'cycles':int(v)} for c,t,v in re.findall(r'HW_CYCLES case=(\d+) token=(\d+) cycles=(\d+)',text)]
    gemm_profiles=[dict(zip(['case','token','compute','bias'],map(int,c))) for c in re.findall(r'HW_GEMM_PROFILE case=(\d+) token=(\d+) compute=(\d+) bias=(\d+)',text)]
    im2col_profiles=[dict(zip(['case','token','zero','dma','calls','bytes'],map(int,c))) for c in re.findall(r'HW_IM2COL_PROFILE case=(\d+) token=(\d+) zero=(\d+) dma=(\d+) calls=(\d+) bytes=(\d+)',text)]
    stats={'im2col_profiles':im2col_profiles,'gemm_profiles':gemm_profiles,'passed':passed,'exit_status':exit_status,'wall_seconds':time.monotonic()-start,'cases':len(cases),'checks':len(checks),'values':sum(int(c[3]) for c in checks),'errors':sum(int(c[4]) for c in checks),'operation_cycles':cycles}
    (output/'statistics.json').write_text(json.dumps(stats,indent=2)+'\n');print({k:v for k,v in stats.items() if k not in ('operation_cycles','gemm_profiles','im2col_profiles')},flush=True)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
