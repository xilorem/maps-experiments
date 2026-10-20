#!/usr/bin/env python3
"""Build the complete 32x32 MAGIA-v3 MobileViT reference, two serial tokens."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
WORKLOAD = SOURCE.parent
REPOS = WORKLOAD.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORKLOAD/'results/full-mesh-32x32')
    parser.add_argument('--sdk-root', type=Path, default=REPOS/'magia-sdk-v3')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1: parser.error('--jobs must be positive')
    output, sdk_source = args.output.resolve(), args.sdk_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared, app, sdk, build = (output/name for name in ('prepared','application','sdk-source','sdk-build'))
    subprocess.run([sys.executable,str(WORKLOAD/'results.py'),'prepare',
        '--model',str(WORKLOAD/'model/mobilevitv2_050.cvnets_in1k.bs1.simplified.wfold.noexpand.gn.convfold.single_output.onnx'),
        '--inputs',str(WORKLOAD/'model/inputs.bin'),'--references',str(WORKLOAD/'model/references.bin'),
        '--tokens','2','--output',str(prepared)],check=True)
    # make build rewrites tile_config.h; a private source tree avoids races with
    # another experiment's build. The installed simulator is read through a link.
    sdk.mkdir(exist_ok=True)
    for name in ('CMakeLists.txt','Makefile','cmake','devices','drivers','hal','kernels','scripts','targets','tests'):
        source, dest = sdk_source/name, sdk/name
        if source.is_dir(): shutil.copytree(source,dest,dirs_exist_ok=True)
        else: shutil.copy2(source,dest)
    if not (sdk/'gvsoc').exists(): (sdk/'gvsoc').symlink_to(sdk_source/'gvsoc',target_is_directory=True)
    app.mkdir(exist_ok=True)
    (app/'CMakeLists.txt').write_text(f'''set(MVIT_DATA "{prepared}/full-mesh-data.bin")
set(MVIT_PREPARED "{prepared}")
set(MVIT_TOKENS 2)
add_subdirectory("{SOURCE}" full_mesh)
''')
    llvm = Path(os.environ.get('LLVM_INSTALL_DIR',REPOS/'toolchains/llvm20-root/usr/lib/llvm-20'))
    env = os.environ.copy()
    env['LLVM_INSTALL_DIR'] = str(llvm)
    env['PATH'] = str(Path(sys.executable).parent)+':'+str(REPOS/'toolchains/pulp-toolchain/bin')+':'+env['PATH']
    env['LD_LIBRARY_PATH'] = str(llvm.parent/'x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
    env['CMAKE_BUILD_PARALLEL_LEVEL'] = str(args.jobs)
    with (output/'build.log').open('w') as log:
        for target in ('spatz_bootrom','onnx_mobilevit_full_mesh'):
            print(f'Building {target} for 32x32...',flush=True)
            subprocess.run(['make','build','tiles=32','pulp_cores=0',f'CMAKE_BUILDDIR={build}',
                f'maps_application_dir={app}',f'test={target}'],cwd=sdk,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    executable = output/'mobilevit_full_mesh_32x32.elf'
    rom = output/'spatz_init.bin'
    shutil.copy2(build/'bin/onnx_mobilevit_full_mesh',executable)
    shutil.copy2(build/'bin/bootrom/spatz_init.bin',rom)
    run = output/'run.sh'
    run.write_text('''#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/gvsoc"
'''+f'export PATH={shlex.quote(str(Path(sys.executable).parent))}:"$PATH"\n'+
        f'export LD_LIBRARY_PATH={shlex.quote(str(llvm.parent/"x86_64-linux-gnu"))}:"${{LD_LIBRARY_PATH:-}}"\n'+
        f'exec stdbuf -oL -eL {shlex.quote(str(sdk_source/"gvsoc/install/bin/gvrun"))} \\\n'+'''  --target magia_v3:n_tiles_x=32,n_tiles_y=32,nb_pulp_cores=8 \\
  --param "binary=$here/mobilevit_full_mesh_32x32.elf" \\
  --work-dir "$here/gvsoc" \\
  --attr "magia_v3/spatz_romfile=$here/spatz_init.bin" run "$@"
''')
    run.chmod(0o755)
    (output/'artifact.json').write_text(json.dumps({
        'workload':'full MobileViT-v2 050, 256x256 input','target':'magia-v3',
        'mesh':[32,32],'execution_tokens':2,'strategy':'sequential full mesh',
        'completion_cycles':'Cumulative simulated cycles from the common post-initialization execution-window start; max across all tiles at the final layer. Both records print after both tokens finish.',
        'elf':executable.name,'elf_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
        'boot_rom':rom.name,'source':str(SOURCE/'main.c'),
        'expected_records':['REFERENCE_COMPLETION token=0 cycle=<cycles>','REFERENCE_COMPLETION token=1 cycle=<cycles>']
    },indent=2)+'\n')
    print(f'Executable: {executable}\nRun: {run}',flush=True)


if __name__ == '__main__': main()
