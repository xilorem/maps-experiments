#!/usr/bin/env python3
"""Regenerate the deterministic token corpus and its FP32 ONNX diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'mobilevitv2_050.cvnets_in1k.bs1.simplified.wfold.noexpand.gn.convfold.single_output.onnx'


def main():
    session = ort.InferenceSession(str(MODEL), providers=['CPUExecutionProvider'])
    shape = session.get_inputs()[0].shape
    values = np.random.default_rng(20260729).standard_normal((16, *shape)).astype('<f2')
    references = np.stack([session.run(None, {'input': value.astype(np.float32)})[0]
                           for value in values]).astype('<f2')
    values.tofile(ROOT / 'inputs.bin')
    references.tofile(ROOT / 'references.bin')
    manifest = {'tokens': 16, 'seed': 20260729, 'input_shape': shape,
                'output_shape': list(references.shape[1:]),
                'references': 'FP32 ONNX Runtime CPU outputs rounded to FP16',
                'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [MODEL, ROOT / 'mobilevit_data.bin', ROOT / 'mobilevit_graph.h',
                            ROOT / 'inputs.bin', ROOT / 'references.bin']}}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
