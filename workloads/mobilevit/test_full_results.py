from argparse import Namespace
import importlib.util
from pathlib import Path
import re

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('full_mobilevit_results', ROOT / 'results.py')
results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(results)


def arguments(tmp_path, tokens=2):
    return Namespace(model=next((ROOT / 'model').glob('*.onnx')),
                     inputs=ROOT / 'model/inputs.bin', references=ROOT / 'model/references.bin',
                     tokens=tokens, output=tmp_path / 'prepared')


def test_full_mesh_and_maps_consume_same_distinct_tokens(tmp_path):
    args = arguments(tmp_path)
    results.prepare(args)
    header = (args.output / 'mobilevit_graph.h').read_text()
    offset = int(re.search(r'#define MVIT_INPUTS_OFF \((\d+)\)', header)[1])
    ref_offset = int(re.search(r'#define MVIT_REFERENCES_OFF \((\d+)\)', header)[1])
    blob = (args.output / 'full-mesh-data.bin').read_bytes()
    inputs = (args.output / 'inputs.bin').read_bytes()
    assert blob[offset:ref_offset] == inputs
    assert inputs == args.inputs.read_bytes()[:2 * 3 * 256 * 256 * 2]
    values = np.frombuffer(inputs, dtype='<f2').reshape(2, -1)
    assert not np.array_equal(values[0], values[1])
    assert blob[ref_offset:] == args.references.read_bytes()[:2 * 1000 * 2]
    assert (args.output / 'reference-available.txt').read_text() == '1\n'
    assert len((args.output / 'reference-checksums.csv').read_text().splitlines()) == 2


@pytest.mark.parametrize('tokens', [0, 17])
def test_invalid_token_count(tmp_path, tokens):
    with pytest.raises(ValueError, match='requested tokens'):
        results.prepare(arguments(tmp_path, tokens))


def test_missing_reference_does_not_stop_preparation(tmp_path):
    args = arguments(tmp_path)
    args.references = tmp_path / 'missing.bin'
    results.prepare(args)
    assert (args.output / 'reference-available.txt').read_text() == '0\n'
    assert (args.output / 'reference-checksums.csv').read_text() == ''


def test_collector_recognizes_full_model_application_name(tmp_path):
    checksums = tmp_path / 'checksums.csv'
    checksums.write_text('0,12345678\n1,abcdef01\n')
    log = ('MAPS_TOKEN tile=0 token=0 start=100 end=200 output=0\n'
           'MAPS_TOKEN tile=1 token=0 start=200 end=400 output=1\n'
           'MAPS_TOKEN tile=0 token=1 start=210 end=310 output=0\n'
           'MAPS_TOKEN tile=1 token=1 start=400 end=600 output=1\n'
           'maps_mobilevit token 0 output output checksum: 12345678\n'
           'maps_mobilevit token 1 output output checksum: abcdef02\n')
    rows = results.shared.maps_rows(log, '32x32', 2, checksums)
    assert [r['completion_cycle'] for r in rows] == [300, 500]
    assert [r['validation_mismatches'] for r in rows] == [0, 1]
