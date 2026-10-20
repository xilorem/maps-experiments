#!/usr/bin/env python3
"""Run MAPS MobileViT stage prefixes sequentially, stopping at the first failure.

Retains the original 32x32 placement. Both token outputs must be exported to L2
and checked before continuing. No-progress timeout is a suspected stall, not a
proof of deadlock. Every run uses the MAPS application only.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def terminate(pid):
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stall-minutes', type=float, default=30)
    parser.add_argument('--prior-series', type=Path, help='retain successful prefixes and rerun its current prefix')
    parser.add_argument('--start-stage-count', type=int, help='explicit prefix stage count to start at')
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args()
    mesh = args.mesh_root.resolve()
    root = (args.output or mesh / 'maps-gvsoc-diagnostic' / ('prefix-series-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.foreground:
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--mesh-root', str(mesh), '--output', str(root), '--stall-minutes', str(args.stall_minutes), '--foreground']
        if args.prior_series:
            cmd.extend(['--prior-series', str(args.prior_series.resolve())])
        if args.start_stage_count is not None:
            cmd.extend(['--start-stage-count', str(args.start_stage_count)])
        with (root / 'series.log').open('w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (root / 'series.pid').write_text(str(proc.pid) + '\n')
        print(json.dumps({'pid': proc.pid, 'output': str(root)}, indent=2))
        return
    execution = json.loads((mesh / 'execution-plan.json').read_text())
    stages = execution['stages']
    summary = {'state': 'running', 'mesh': f"{execution['mesh']['width']}x{execution['mesh']['height']}", 'tokens': 2, 'stall_minutes': args.stall_minutes, 'results': []}
    runner = Path(__file__).with_name('run-monitored.py')
    start_count = 1
    if args.prior_series:
        prior = read(args.prior_series / 'summary.json')
        start_count = args.start_stage_count or prior['current_stage_count']
        summary['results'] = [r for r in prior['results'] if r['passed'] and r['stage_count'] < start_count]
        summary['prior_series'] = str(args.prior_series.resolve())
    if args.start_stage_count is not None:
        start_count = args.start_stage_count
    if not 1 <= start_count <= len(stages):
        raise ValueError('Invalid starting stage count')
    for count in range(start_count, len(stages) + 1):
        out = root / f'prefix-{count:03d}'
        out.mkdir(exist_ok=True)
        summary.update(current_stage_count=count, current_last_stage=count - 1, current_output=str(out))
        write(root / 'summary.json', summary)
        print(f'START prefix=0..{count - 1} stages={count}', flush=True)
        cmd = [sys.executable, '-u', str(runner), '--mesh-root', str(mesh), '--output', str(out), '--stage-limit', str(count), '--foreground']
        with (out / 'monitor.log').open('w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        last_signature, changed_at, stalled = None, time.monotonic(), False
        while proc.poll() is None:
            progress = read(out / 'progress.json')
            signature = (progress.get('state'), progress.get('log_lines'), progress.get('last_event_at'), progress.get('output_buffers_completed'))
            if signature != last_signature:
                changed_at, last_signature = time.monotonic(), signature
            if time.monotonic() - changed_at > args.stall_minutes * 60:
                stalled = True
                failure = {'state': 'suspected_stall', 'reason': f'No log/progress marker for {args.stall_minutes} minutes', 'progress': progress}
                write(out / 'stall-report.json', failure)
                pid_path = out / 'simulator.pid'
                if pid_path.exists():
                    terminate(int(pid_path.read_text()))
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    terminate(proc.pid)
                break
            time.sleep(5)
        if proc.poll() is None:
            proc.wait(timeout=10)
        progress = read(out / 'progress.json')
        passed = not stalled and proc.returncode == 0 and progress.get('state') == 'completed' and progress.get('expected_output_buffers', 0) > 0 and progress.get('output_buffers_completed') == progress.get('expected_output_buffers')
        row = {'stage_count': count, 'last_stage': count - 1, 'passed': passed, 'state': 'suspected_stall' if stalled else progress.get('state', 'monitor_failed'), 'exit_status': proc.returncode, 'output': str(out), 'output_buffers_completed': progress.get('output_buffers_completed'), 'expected_output_buffers': progress.get('expected_output_buffers'), 'errors': progress.get('errors', [])}
        summary['results'].append(row)
        print('RESULT ' + json.dumps(row), flush=True)
        if not passed:
            summary['state'] = row['state']
            write(root / 'summary.json', summary)
            return 1
        write(root / 'summary.json', summary)
    summary['state'] = 'all_prefixes_completed'
    write(root / 'summary.json', summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
