"""Check reference packing and runner generation without a full network simulation."""
import importlib.util
from pathlib import Path
import re
import sys
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('resnet_experiment',ROOT/'experiment.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_prepare_packs_distinct_tokens_and_references(tmp_path):
    graph=h.make_graph([h.make_node('MaxPool',['x'],['y'],kernel_shape=[3,3],strides=[2,2],pads=[1,1,1,1])],
        'pool',[h.make_tensor_value_info('x',T.FLOAT,[1,2,5,7])],
        [h.make_tensor_value_info('y',T.FLOAT,[1,2,3,4])])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',20)]);model.ir_version=10
    path=tmp_path/'pool.onnx';onnx.save(model,path)
    output=tmp_path/'prepared';sdk=runner.REPOS/'magia-sdk-v3'
    runner.prepare(path,sdk,output,2)
    inputs=np.fromfile(output/'inputs.bin',dtype='<f2').reshape(2,1,2,5,7)
    references=np.fromfile(output/'references.bin',dtype='<f2').reshape(2,1,2,3,4)
    assert not np.array_equal(inputs[0],inputs[1])
    for token in range(2):
        for y in range(3):
            for x in range(4):
                np.testing.assert_array_equal(references[token,:,:,y,x],inputs[token,:,:,max(0,y*2-1):min(5,y*2+2),max(0,x*2-1):min(7,x*2+2)].max(axis=(-2,-1)))
    header=(output/'resnet18_graph.h').read_text()
    offset=int(re.search(r'#define RN18_REFERENCES_OFF \((\d+)\)',header)[1])
    blob=(output/'full-mesh-data.bin').read_bytes()
    assert blob[:inputs.nbytes]==inputs.tobytes()
    assert blob[offset:]==references.tobytes()
    assert len((output/'reference-checksums.csv').read_text().splitlines())==2
    runner.reference_sources(sdk,tmp_path/'full-mesh')
    source=(tmp_path/'full-mesh/main.c').read_text()
    assert 'token * RN18_INPUT_ELEMENTS' in source
    assert 'FULL_MESH_LAYER_DONE' in source
    assert 'elapsed_cycles=' in source
    assert 'REFERENCE_COMPLETION' in source


def test_monitor_accepts_small_mesh_tile_names_and_absent_tasks(tmp_path):
    monitor=runner.load('test_resnet_monitor',runner.COMMON/'diagnostics/run-monitored.py')
    app=tmp_path/'app';(app/'src/tiles').mkdir(parents=True)
    (app/'src/tiles/tile_02.c').write_text('')
    assert monitor.tile_source(app,2).name=='tile_02.c'
    (app/'CMakeLists.txt').write_text('set(APPLICATION maps_mobilevit)\n')
    (app/'src/maps_mobilevit_runner.c').write_text('  runtime.relu_fp16_task = RELU_FP16_SPATZ_TASK;\n')
    monitor.prepare_progress(app)
    source=(app/'src/maps_mobilevit_runner.c').read_text()
    assert 'DEPTHWISE_CONV_FP16_SPATZ_TASK' not in source
    assert 'SIGMOID_FP16_SPATZ_TASK' not in source
