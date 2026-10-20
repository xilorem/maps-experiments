#!/usr/bin/env python3
"""Full MobileViT preparation; timing and CSV reporting shared with the slice."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

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
    if available: (args.output / 'references.bin').write_bytes(references.tobytes())
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


def collect_maps(args):
    text=args.maps_log.read_text(encoding='utf-8')
    result = shared.maps_rows(text, args.mesh, args.tokens, args.reference_checksums)
    chunks=re.findall(r'^MAPS_OUTPUT_BYTES token=(\d+) offset=(\d+) data=([0-9a-fA-F]+)$',text,re.M)
    if chunks:
        prepared=args.reference_checksums.parent
        available=(prepared/'reference-available.txt').read_text().strip()=='1'
        references=None
        if available:
            path=prepared/'references.bin'
            if path.exists(): references=np.fromfile(path,dtype='<f2').reshape(args.tokens,-1)
            else:
                header=(prepared/'mobilevit_graph.h').read_text()
                offset=int(re.search(r'#define MVIT_REFERENCES_OFF \((\d+)\)',header)[1])
                references=np.frombuffer((prepared/'full-mesh-data.bin').read_bytes()[offset:],dtype='<f2').reshape(args.tokens,-1)
        report=[]
        for token,row in enumerate(result):
            pieces=sorted((int(offset),bytes.fromhex(data)) for t,offset,data in chunks if int(t)==token)
            output=bytearray()
            for offset,data in pieces:
                if offset!=len(output): raise ValueError('incomplete or overlapping MAPS output dump')
                output.extend(data)
            if not output or len(output)%2: raise ValueError('missing or invalid MAPS FP16 output dump')
            actual=np.frombuffer(output,dtype='<f2').astype(np.float64)
            if references is not None and len(actual)!=references.shape[1]:
                raise ValueError('MAPS output size does not match reference')
            args.maps_log.with_name(f'maps-output-token{token}.bin').write_bytes(output)
            finite=np.isfinite(actual)
            entry={'token':token,'checksum_mismatch':row['validation_mismatches'],
                   'nonfinite':int((~finite).sum()),'reference_available':available}
            row['validation_nonfinite']=entry['nonfinite']
            if references is not None:
                expected=references[token].astype(np.float64)
                valid=finite & np.isfinite(expected)
                difference=np.abs(actual-expected)
                mismatch=int((valid & (difference>0.002+0.02*np.abs(expected))).sum())
                row['validation_mismatches']=mismatch
                entry.update(mismatches=mismatch,atol=0.002,rtol=0.02,
                             max_abs_error=float(difference[valid].max(initial=0)),
                             max_rel_error=float((difference[valid]/np.maximum(np.abs(expected[valid]),1e-12)).max(initial=0)))
            report.append(entry)
        args.maps_log.with_name('maps-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    append_rows(args.csv, result)


def collect_full_mesh(args):
    result = shared.reference_rows(args.full_mesh_log.read_text(encoding='utf-8'),
                                   args.mesh, args.tokens,
                                   args.reference_available.read_text().strip() == '1')
    append_rows(args.csv, result)


def append_rows(path, result):
    write_header = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=shared.FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(result)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('collect-maps', 'collect-full-mesh'):
        maps = sys.argv[1] == 'collect-maps'
        parser = argparse.ArgumentParser(description='Append one completed strategy to the experiment CSV.')
        parser.add_argument('--mesh', required=True)
        parser.add_argument('--tokens', type=int, required=True)
        if maps:
            parser.add_argument('--maps-log', type=Path, required=True)
            parser.add_argument('--reference-checksums', type=Path, required=True)
        else:
            parser.add_argument('--full-mesh-log', type=Path, required=True)
            parser.add_argument('--reference-available', type=Path, required=True)
        parser.add_argument('--csv', type=Path, required=True)
        (collect_maps if maps else collect_full_mesh)(parser.parse_args(sys.argv[2:]))
        return
    # The shared parser resolves prepare through its module globals.
    shared.prepare = prepare
    args = shared.parser().parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
