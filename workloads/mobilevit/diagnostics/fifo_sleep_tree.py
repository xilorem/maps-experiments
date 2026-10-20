#!/usr/bin/env python3
"""Build an isolated compiled GVSoC tree matching the FIFO doorbell model."""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import magia_sdk


def build_platform_tree(sdk, mesh, output, env=None):
    sdk, output = Path(sdk).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    install = sdk / 'gvsoc/install'
    # Install only the experimental Python generators; preserve the normal
    # installed platform-tree library. FIFO events remain off by default.
    for name in ('arch', 'tile', 'soc'):
        source = sdk / f'gvsoc/pulp/pulp/chips/magia_v3/{name}.py'
        destination = install / f'generators/pulp/chips/magia_v3/{name}.py'
        if source.read_bytes() != destination.read_bytes():
            shutil.copy2(source, destination)
    generated = output / 'generated'
    components = output / 'magia_v3.cfg'
    command = [str(install / 'bin/gvrun'), '--target', f'magia_v3:n_tiles_x={mesh},n_tiles_y={mesh},nb_pulp_cores={magia_sdk.GVSOC_PULP_CORES}',
               '--attr', 'magia_v3/fifo_events=1',
               '--component-file', str(components),
               '--builddir', str(generated), '--installdir', str(install),
               'components']
    environment = os.environ.copy() if env is None else env
    with (output / 'generate.log').open('w') as log:
        subprocess.run(command, cwd=sdk, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    library = output / 'libplatform_tree_fifo_sleep.so'
    # This translation unit contains the static tree, not simulated kernels.
    # O0 keeps its large tables cheap to compile without affecting simulation.
    command = shlex.split(environment.get('CXX', 'c++')) + [
        '-O0', '-DNDEBUG', '-std=gnu++17', '-fPIC', '-shared',
        '-I' + str(generated),
        '-I' + str(sdk / 'gvsoc/engine/engine/include'),
        str(output / 'magia_v3.tree.cpp'), '-o', str(library)]
    with (output / 'compile.log').open('w') as log:
        subprocess.run(command, cwd=sdk, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy2(output / 'magia_v3.tree.cpp.sig', Path(str(library) + '.sig'))
    return library


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-root', type=Path, required=True)
    parser.add_argument('--mesh', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build_platform_tree(args.sdk_root, args.mesh, args.output), flush=True)
