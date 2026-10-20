"""Exercise runner/monitor/CSV wiring without allocating or simulating MobileViT."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / 'experiments/workloads/mobilevit'
    root.mkdir(parents=True)
    shutil.copy2(ROOT / 'run-experiment.sh', root)
    shutil.copy2(ROOT / 'results.py', root)
    shared = root.parent / 'mobilevit_170-179'
    shared.mkdir()
    shutil.copy2(ROOT.parent / 'mobilevit_170-179/results.py', shared)
    diagnostics = root / 'diagnostics'
    diagnostics.mkdir()
    for name in ['run-monitored.py', 'fifo_sleep_tree.py', 'magia_sdk.py']:
        shutil.copy2(ROOT / 'diagnostics' / name, diagnostics)
    tools = tmp_path / 'tools'
    tools.mkdir()
    sdk = tmp_path / 'sdk'
    (sdk / 'gvsoc/install/bin').mkdir(parents=True)
    calls = tmp_path / 'calls.jsonl'
    common = '''import json, os, pathlib, sys
with open(os.environ['TEST_CALLS'], 'a') as log:
    log.write(json.dumps({'program': pathlib.Path(sys.argv[0]).name, 'args': sys.argv[1:]})+'\\n')
'''
    def executable(path, source):
        path.write_text(f'#!{sys.executable}\n' + common + source)
        path.chmod(0o755)
    executable(tools / 'python-wrapper', '''
args = sys.argv[1:]
if args[0].endswith('results.py') and args[1]=='prepare':
    output=pathlib.Path(args[args.index('--output')+1]); output.mkdir(parents=True,exist_ok=True)
    for name,value in [('input-name.txt','input'),('inputs.bin','input'),('full-mesh-data.bin','blob'),('reference-available.txt','0\\n'),('reference-checksums.csv','')]:
        (output/name).write_text(value)
elif args[0].endswith('fifo_sleep_tree.py'):
    output=pathlib.Path(args[args.index('--output')+1]); output.mkdir(parents=True,exist_ok=True)
    (output/'libplatform_tree_fifo_sleep.so').write_text('test model')
else:
    os.execv(sys.executable,[sys.executable,*args])
''')
    executable(tools / 'make', '''
args=sys.argv[1:]
values=dict(a.split('=',1) for a in args if '=' in a)
if 'MODEL' in values:
    app=pathlib.Path(values['APPLICATION']); (app/'src/tiles').mkdir(parents=True)
    (app/'CMakeLists.txt').write_text('set(MAPS_APPLICATION maps_mobilevit)\\n')
    (app/'src/maps_mobilevit_runner.c').write_text('  runtime.relu_fp16_task = RELU_FP16_SPATZ_TASK;\\n  if (maps_operation_runtime_init(&runtime) != 0)\\n    return -1;\\n  maps_fifo_init_tile(&plan);\\n')
    for tile in [0,1]: (app/f'src/tiles/tile_{tile:04d}.c').write_text('.kind = OP_RESHAPE,\\n')
    pathlib.Path(values['EXECUTION_PLAN']).write_text(json.dumps({'mesh':{'width':2,'height':2},'stages':[{'name':'first','submesh':{'tile_ids':[0]}},{'name':'second','submesh':{'tile_ids':[1]}}]}))
elif 'gvsoc' in args and os.environ.get('TEST_CONFIG_FAILURE'):
    sys.exit(1)
elif 'build' in args:
    build=pathlib.Path(values['CMAKE_BUILDDIR']); (build/'bin/bootrom').mkdir(parents=True,exist_ok=True)
    (build/'bin'/values['test']).touch()
elif 'run' in args:
    sdk=pathlib.Path(args[args.index('-C')+1])
    os.execv(str(sdk/'gvsoc/install/bin/gvrun'),['gvrun','--param','binary='+values['test']])
''')
    executable(tools / 'riscv32-unknown-elf-gcc', 'print("test compiler")\n')
    executable(sdk / 'gvsoc/install/bin/gvrun', '''
args=sys.argv[1:]; tokens=int(os.environ['TEST_TOKENS'])
if any('onnx_mobilevit_full_mesh' in a for a in args):
    if os.environ.get('TEST_REFERENCE_FAILURE'):
        sys.exit(1)
    print('REFERENCE_TILES count=4')
    for token in range(tokens): print(f'REFERENCE_COMPLETION token={token} cycle={100+100*token}')
else:
    if os.environ.get('TEST_SIM_FAILURE'):
        print('MAPS_KERNEL_ERROR tile=0 token=0 op=0 code=-9 cycle=100')
        sys.exit(1)
    for tile in [0,1]:
        print(f'MAPS_INIT_DONE tile={tile} cycle=10')
        print(f'MAPS_TILE_READY tile={tile} cycle=20')
    for token in range(tokens):
        for tile in [0,1]:
            start=100+token*100+tile*50; end=start+100
            print(f'MAPS_KERNEL_BEGIN tile={tile} token={token} op=0 cycle={start}')
            print(f'MAPS_KERNEL_DONE tile={tile} token={token} op=0 cycle={end}')
            print(f'MAPS_TOKEN_DONE tile={tile} token={token} cycle={end}')
            print(f'MAPS_TOKEN tile={tile} token={token} start={start} end={end} output={tile}')
''')
    env = os.environ.copy()
    for name in list(env):
        if name.startswith('MAPS_') or name.startswith('MAGIA_'):
            env.pop(name)
    env.update(PATH=str(tools)+':'+env['PATH'], PYTHON=str(tools/'python-wrapper'),
               MAPS_ROOT=str(tmp_path/'maps'), MAGIA_SDK_ROOT=str(sdk),
               LLVM_INSTALL_DIR=str(tmp_path/'llvm'), MESH_SIZES='2',
               TEST_CALLS=str(calls), TEST_TOKENS='2')
    return root, calls, env


@pytest.mark.parametrize('arguments,tokens,strategies,sleep,monitor', [
    ([], 2, ['maps','full-mesh'], True, True),
    (['--maps-only','--tokens','3','--trace-tiles','1'], 3, ['maps'], True, True),
    (['--maps-only','--no-fifo-event-sleep'], 2, ['maps'], False, True),
    (['--maps-only','--no-fifo-event-sleep'], 2, ['maps'], False, False),
])
def test_experiment_csv_and_stage_monitoring(experiment, arguments, tokens, strategies, sleep, monitor):
    root, calls, env = experiment
    env['TEST_TOKENS'] = str(tokens)
    if not monitor: env['MAPS_STAGE_PROGRESS']='0'
    run = subprocess.run(['bash',str(root/'run-experiment.sh'),*arguments],env=env,
                         capture_output=True,text=True,timeout=30)
    assert run.returncode==0, run.stdout+'\n'+run.stderr
    output=next((root/'results').iterdir()); mesh=output/'2x2'
    rows=list(csv.DictReader((output/'results.csv').open()))
    assert len(rows)==tokens*len(strategies)
    assert set(r['strategy'] for r in rows)==set(strategies)
    assert [int(r['completion_cycle']) for r in rows if r['strategy']=='maps']==[150+100*t for t in range(tokens)]
    cmake=(mesh/'maps-application/CMakeLists.txt').read_text()
    assert 'MAPS_RESULT_TRACE=1' in cmake
    assert ('MAPS_FIFO_EVENT_SLEEP=1' in cmake)==sleep
    assert ('full_mesh)' in cmake)==('full-mesh' in strategies)
    commands=[json.loads(l) for l in calls.read_text().splitlines()]
    assert any(c['args'][0].endswith('fifo_sleep_tree.py') for c in commands if c['program']=='python-wrapper')==sleep
    if monitor:
        progress=json.loads((mesh/'maps-monitor/progress.json').read_text())
        assert progress['state']=='completed'
        assert progress['tokens']==tokens
        assert progress['completed_tile_tokens']==2*tokens
        events=[json.loads(l) for l in (mesh/'maps-monitor/stage-events.jsonl').read_text().splitlines()]
        assert len([e for e in events if e['event']=='STAGE_PROCESSING_DONE'])==2*tokens
        assert 'MAPS_INIT_DONE' in (mesh/'maps-application/src/maps_mobilevit_runner.c').read_text()
        command=json.loads((mesh/'maps-monitor/run.json').read_text())['command']
        assert ('magia_v3/fifo_events=1' in command)==sleep
        assert any(a.startswith('platform_tree=') for a in command)==sleep
        if '--trace-tiles' in arguments:
            assert '--trace=tile-1-cv32-core/insn:cv32-insn.log' in command
            assert '--trace=tile-1-snitch-spatz/insn:spatz-insn.log' in command
    else:
        assert not (mesh/'maps-monitor').exists()


def test_fifo_sleep_validation(experiment):
    root, calls, env=experiment
    env['MAPS_FIFO_EVENT_SLEEP']='invalid'
    run=subprocess.run(['bash',str(root/'run-experiment.sh')],env=env,capture_output=True,text=True)
    assert run.returncode==2
    assert not calls.exists()


def test_kernel_failure_retains_diagnostics_without_csv(experiment):
    root, calls, env=experiment
    env['TEST_SIM_FAILURE']='1'
    run=subprocess.run(['bash',str(root/'run-experiment.sh'),'--maps-only'],env=env,
                       capture_output=True,text=True,timeout=30)
    assert run.returncode!=0
    output=next((root/'results').iterdir())
    assert not (output/'results.csv').exists()
    monitor=output/'2x2/maps-monitor'
    progress=json.loads((monitor/'progress.json').read_text())
    assert progress['state']=='kernel_failed'
    assert progress['errors'][0]['details']['code']=='-9'
    assert (monitor/'failure.md').is_file()


def test_maps_rows_survive_full_mesh_failure(experiment):
    root, calls, env=experiment
    env['TEST_REFERENCE_FAILURE']='1'
    run=subprocess.run(['bash',str(root/'run-experiment.sh')],env=env,
                       capture_output=True,text=True,timeout=30)
    assert run.returncode!=0
    output=next((root/'results').iterdir())
    rows=list(csv.DictReader((output/'results.csv').open()))
    assert len(rows)==2
    assert all(row['strategy']=='maps' for row in rows)
    assert [row['token'] for row in rows]==['0','1']


def test_resume_reuses_preparation_and_allocation(experiment):
    root, calls, env=experiment
    env.update(TEST_CONFIG_FAILURE='1', TEST_TOKENS='3')
    failed=subprocess.run(['bash',str(root/'run-experiment.sh'),'--tokens','3'],env=env,
                          capture_output=True,text=True,timeout=30)
    assert failed.returncode!=0
    output=next((root/'results').iterdir())
    env.pop('TEST_CONFIG_FAILURE')
    resumed=subprocess.run(['bash',str(root/'run-experiment.sh'),'--resume-run',str(output)],env=env,
                           capture_output=True,text=True,timeout=30)
    assert resumed.returncode==0, resumed.stdout+'\n'+resumed.stderr
    commands=[json.loads(line) for line in calls.read_text().splitlines()]
    assert sum(any(a.startswith('MODEL=') for a in c['args']) for c in commands if c['program']=='make')==1
    assert sum('prepare' in c['args'] for c in commands if c['program']=='python-wrapper')==1
    rows=list(csv.DictReader((output/'results.csv').open()))
    assert len(rows)==6
    assert (output/'2x2/maps-application/src/maps_mobilevit_runner.c').read_text().count('printf("MAPS_TILE_READY ')==1
