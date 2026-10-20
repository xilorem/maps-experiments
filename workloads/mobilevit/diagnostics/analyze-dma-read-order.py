#!/usr/bin/env python3
"""Report DMA AXI read bursts that complete ahead of earlier ones.

Reads the trace of one tile's AXI to OBI iDMA read manager (idma_v3), e.g.
--trace=tile-N-idma-axi2obi/axi_read/trace:dma-read.log. Bursts are identified
by their outstanding-read slot, which is reused once the burst is done.
"""

import argparse
import json
from pathlib import Path
import re


def analyze(path):
    pending, reordered = [], []
    sent, received = 0, 0
    for line in path.read_text().splitlines():
        submission = re.search(r'Issuing read burst \(slot: (\d+), addr: (0x[0-9a-f]+)', line)
        if submission:
            pending.append({'request': submission[1], 'source_address': submission[2]})
            sent += 1
        response = re.search(r'Read burst done \(slot: (\d+)\)', line)
        if response:
            received += 1
            index = next((i for i, request in enumerate(pending)
                          if request['request'] == response[1]), None)
            if index is None:
                raise ValueError(f'Response without a matching outstanding request: {line}')
            if index:
                reordered.append({'cycle': int(line.split(':')[1]),
                                  'expected': pending[0], 'actual': pending[index]})
            pending.pop(index)
    return {'sent': sent, 'received': received, 'outstanding': len(pending),
            'out_of_order_count': len(reordered), 'out_of_order_responses': reordered}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path, help='one tile DMA AXI read trace')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.trace)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'out_of_order_responses'}, indent=2))
