"""Check dependencies, GPU status and scenario inputs without starting SUMO."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
import shutil
import sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--env',choices=('env','new_env'),default='new_env')
    parser.add_argument('--require-cuda',action='store_true')
    args=parser.parse_args()
    import numpy as np
    import torch
    import torch_geometric
    import matplotlib
    import traci
    import sumolib
    from rp.experiment import PROJECT_DIR, build_selected_env_config, validate_environment_inputs
    from scripts.run_experiment import bootstrap_sumo_environment
    torch.from_numpy(np.asarray([1.0],dtype=np.float32))
    config=build_selected_env_config(PROJECT_DIR,env_name=args.env)
    bootstrap_sumo_environment(config)
    validate_environment_inputs(config)
    cuda=torch.cuda.is_available()
    if args.require_cuda and not cuda:
        raise SystemExit('CUDA is unavailable: verify NVIDIA driver and the installed PyTorch CUDA build.')
    report={'python':sys.version.split()[0], 'environment':args.env,
        'packages':{p:metadata.version(p) for p in ('torch','torch-geometric','numpy','matplotlib','eclipse-sumo','sumolib','traci')},
        'cuda_available':cuda,'cuda_runtime':torch.version.cuda,
        'gpu_name':torch.cuda.get_device_name(0) if cuda else None,
        'sumo_home':str(config.paths.sumo_home),'sumo_binary':shutil.which('sumo'),
        'data_root':str(config.paths.run_dir),
        'scenarios':{split:sum(r.split==split for r in config.environment.scenario_catalog)
                     for split in ('training','validation','testing')},
        'numpy_tensor_bridge':'passed','data_inputs':'passed','sumo_started':False}
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
