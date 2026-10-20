"""Build and launch helpers shared by the MobileViT diagnostics.

Mirror run-experiment.sh: the SDK comes from MAGIA_SDK_ROOT (default
<checkout>/magia-sdk-v3) and the toolchains from the environment (source
maps-env.sh first). The SDK is built without the PULP cluster (PULP_CORES,
default 0), while GVSoC's MAGIA v3 target needs at least one PULP core per
tile (GVSOC_PULP_CORES, default 8); they stay idle.

Builds go to the caller's directory. `make build` regenerates the SDK's
tile_config.h for its mesh, so build() restores it afterwards. `make gvsoc` is
incremental (a no-op takes seconds) and must not be skipped: a platform library
built before a GVSoC source change crashes the simulator.
"""

import os
from pathlib import Path
import re
import signal
import subprocess
import sys

CHECKOUT_ROOT = Path(__file__).resolve().parents[4]
PULP_CORES = os.environ.get('PULP_CORES', '0')
GVSOC_PULP_CORES = os.environ.get('GVSOC_PULP_CORES', '8')


def default_sdk_root():
    return Path(os.environ.get('MAGIA_SDK_ROOT', CHECKOUT_ROOT / 'magia-sdk-v3'))


def build_env(sdk):
    env = os.environ.copy()
    if 'LLVM_INSTALL_DIR' not in env:
        for candidate in (sdk / 'llvm/install',
                          CHECKOUT_ROOT / 'toolchains/llvm20-root/usr/lib/llvm-20',
                          Path('/usr/lib/llvm-20'), Path('/usr/lib/llvm-18')):
            if all((candidate / 'bin' / tool).exists()
                   for tool in ('clang', 'llvm-objcopy', 'llvm-objdump')):
                env['LLVM_INSTALL_DIR'] = str(candidate)
                break
        else:
            raise SystemExit('set LLVM_INSTALL_DIR to a Spatz-capable LLVM installation')
    runtime_libs = Path(env['LLVM_INSTALL_DIR']).parent / 'x86_64-linux-gnu'
    if runtime_libs.is_dir():
        env['LD_LIBRARY_PATH'] = str(runtime_libs) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['PATH'] = str(Path(sys.executable).parent) + ':' + env['PATH']
    env.setdefault('CMAKE_BUILD_PARALLEL_LEVEL', '8')
    return env


def build(sdk, env, log, tiles, build_dir, application, targets, options=()):
    """Build GVSoC for this mesh, then each SDK target of the application."""
    subprocess.run(['make', 'gvsoc', f'tiles={tiles}', f'pulp_cores={GVSOC_PULP_CORES}'],
                   cwd=sdk, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    config = sdk / 'targets/magia/include/addr_map/tile_config.h'
    saved = config.read_bytes()
    try:
        for target in targets:
            subprocess.run(['make', 'build', f'tiles={tiles}', f'pulp_cores={PULP_CORES}',
                            f'CMAKE_BUILDDIR={build_dir}', f'maps_application_dir={application}',
                            *options, f'test={target}'],
                           cwd=sdk, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        # Rewrite only on change, so other build trees do not see a newer header.
        if config.read_bytes() != saved:
            config.write_bytes(saved)


def stop_process_group(process):
    """Stop a simulator started with start_new_session=True, children included."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def gvrun_command(sdk, build_dir, binary, work, tiles, options=()):
    return ['stdbuf', '-oL', '-eL', str(sdk / 'gvsoc/install/bin/gvrun'),
            '--target', f'magia_v3:n_tiles_x={tiles},n_tiles_y={tiles},nb_pulp_cores={GVSOC_PULP_CORES}',
            '--param', f'binary={build_dir}/bin/{binary}', '--work-dir', str(work),
            '--attr', f'magia_v3/spatz_romfile={build_dir}/bin/bootrom/spatz_init.bin',
            *options, 'run']


def tile_source(app, mesh, tile):
    """Generated tile source, padded like the emitter's getPaddedPhysicalTileId."""
    return app / f'src/tiles/tile_{padded_tile_id(mesh, tile)}.c'


def padded_tile_id(mesh, tile):
    width = max(2, len(str(mesh['width'] * mesh['height'] - 1)))
    return f'{tile:0{width}d}'


def descriptor_blocks(text):
    """Split generated array entries, without changing nested descriptors."""
    blocks, depth, begin = [], 0, None
    for index, character in enumerate(text):
        if character == '{':
            if depth == 0:
                begin = index
            depth += 1
        elif character == '}':
            depth -= 1
            if depth == 0:
                blocks.append(text[begin:index + 1])
    assert depth == 0
    return blocks


def filter_sends(source, keep):
    """Keep the FIFO sends whose destination tile satisfies keep.

    Returns the tile source and the number of sends kept. Without any, the
    array is removed and the plan points at no sends.
    """
    match = re.search(r'static const fifo_send_desc_t (\w+)\[\] = \{(.*?)\n\};', source, re.S)
    if not match:
        return source, 0
    sends = [block for block in descriptor_blocks(match[2])
             if keep(int(re.search(r'\.dst_hartid = (\d+)', block)[1]))]
    replacement = ('static const fifo_send_desc_t ' + match[1] + '[] = {\n'
                   + ',\n'.join(sends) + ',\n};') if sends else ''
    source = source[:match.start()] + replacement + source[match.end():]
    source = re.sub(r'\.num_sends = \d+u,', f'.num_sends = {len(sends)}u,', source)
    if not sends:
        source = re.sub(r'\.sends = [^,]+,', '.sends = 0,', source)
        # The L1 descriptor-copy assertion sizes the removed array.
        source = source.replace(f'sizeof({match[1]})', '0u')
    return source, len(sends)
