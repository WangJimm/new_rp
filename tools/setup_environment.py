"""Create a local Windows/Linux venv; installs only when the user runs this tool."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description='Install the portable RP training environment.')
    parser.add_argument('--torch-build',choices=('cpu','cu118','cu121'),default='cpu')
    parser.add_argument('--dry-run',action='store_true',help='Print commands without creating a venv or downloading packages.')
    args=parser.parse_args()
    if sys.version_info[:2] not in ((3,11),(3,12)):
        raise SystemExit('Use Python 3.11 or 3.12; the source environment uses Python 3.12.3.')
    venv=ROOT/'.venv'
    python=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    commands=[]
    if not python.is_file(): commands.append([sys.executable,'-m','venv',str(venv)])
    commands.extend([
        [str(python),'-m','pip','install','--upgrade','pip'],
        [str(python),'-m','pip','install','numpy==1.26.4'],
        [str(python),'-m','pip','install',f'torch==2.3.1+{args.torch_build}','--index-url',f'https://download.pytorch.org/whl/{args.torch_build}'],
        [str(python),'-m','pip','install','-r',str(ROOT/'requirements.txt'),'-c',str(ROOT/'constraints.txt')],
        [str(python),'-m','pip','check'],
    ])
    for command in commands:
        print(subprocess.list2cmdline(command),flush=True)
        if not args.dry_run: subprocess.run(command,cwd=ROOT,check=True)
    print('Next: use .venv Python to run -m scripts.check_installation, then -m scripts.run_experiment --env new_env --rp_variant proposed')

if __name__=='__main__': main()
