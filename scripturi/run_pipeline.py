"""Run the selected experiment stage with the current Python interpreter."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['audit', 'main', 'sensitivity', 'transfer', 'long-window', 'all'], default='audit')
    args = parser.parse_args()
    environment = os.environ.copy()
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        environment[name] = '4'
    environment['PYTHONUTF8'] = '1'
    steps = [('benchmark.py', ['--stage', args.stage])]
    if args.stage == 'all':
        steps.extend([('summarize_results.py', []), ('validate_experiments.py', [])])
    for filename, arguments in steps:
        subprocess.run([sys.executable, str(ROOT / 'scripturi' / filename), *arguments], cwd=ROOT, env=environment, check=True)
    if args.stage != 'all':
        print('Stage completed. Full tables and final validations require all 33 experiment/seed configurations.')


if __name__ == '__main__':
    main()
