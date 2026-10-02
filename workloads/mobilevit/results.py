#!/usr/bin/env python3
"""Full MobileViT preparation; timing and CSV reporting shared with the slice."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('mobilevit_slice_results', ROOT.parent / 'mobilevit_170-179/results.py')
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)
shared.MAPS_CHECKSUM = re.compile(
    r'^maps_mobilevit token (\d+) output .* checksum: ([0-9a-fA-F]{8})$', re.MULTILINE)


def prepare(args):
    model_dir = args.model.parent
    manifest = json.loads((model_dir / 'manifest.json').read_text())
    for path in (args.model, model_dir / 'mobilevit_data.bin', model_dir / 'mobilevit_graph.h', args.inputs):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256'][path.name]:
            raise ValueError(f'frozen artifact changed: {path}; regenerate model artifacts')
    elements = int(np.prod(manifest['input_shape']))
    output_elements = int(np.prod(manifest['output_shape']))
    stored = np.fromfile(args.inputs, dtype='<f2')
    if stored.size % elements or not 1 <= args.tokens <= stored.size // elements:
        raise ValueError('requested tokens must be between 1 and the stored token count')
    inputs = stored[:args.tokens * elements].tobytes()
    references = np.zeros((args.tokens, output_elements), dtype='<f2')
    available = False
    if args.references.is_file():
        stored_refs = np.fromfile(args.references, dtype='<f2')
        if stored_refs.size >= references.size:
            references[:] = stored_refs[:references.size].reshape(references.shape)
            available = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'inputs.bin').write_bytes(inputs)
    blob = (model_dir / 'mobilevit_data.bin').read_bytes()
    (args.output / 'full-mesh-data.bin').write_bytes(blob + inputs + references.tobytes())
    header = (model_dir / 'mobilevit_graph.h').read_text()
    header += (f'\n#define MVIT_INPUTS_OFF ({len(blob)})\n'
               f'#define MVIT_REFERENCES_OFF ({len(blob) + len(inputs)})\n'
               f'#define MVIT_INPUT_ELEMENTS ({elements})\n')
    (args.output / 'mobilevit_graph.h').write_text(header)
    import onnx
    (args.output / 'input-name.txt').write_text(onnx.load(args.model).graph.input[0].name)
    checksums = []
    if available:
        for token, reference in enumerate(references):
            checksum = 2166136261
            for byte in reference.tobytes():
                checksum = ((checksum ^ byte) * 16777619) & 0xffffffff
            checksums.append(f'{token},{checksum:08x}\n')
    (args.output / 'reference-checksums.csv').write_text(''.join(checksums))
    (args.output / 'reference-available.txt').write_text('1\n' if available else '0\n')


def main():
    # The shared parser resolves prepare through its module globals.
    shared.prepare = prepare
    args = shared.parser().parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
