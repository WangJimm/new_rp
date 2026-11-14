"""Explicit training RNG controls and a portable audit trail for fresh runs."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import warnings

import numpy as np
import torch


def add_reproducibility_args(parser):
    parser.add_argument('--training-seed', type=int, default=None,
                        help='Seed Python, NumPy and PyTorch; independent of scenario-schedule seed.')
    parser.add_argument('--deterministic', action='store_true',
                        help='Require deterministic PyTorch algorithms; set PYTHONHASHSEED before Python starts.')


def apply_reproducibility_args(config, args):
    seed = getattr(args, 'training_seed', None)
    deterministic = getattr(args, 'deterministic', False)
    if seed is not None and not 0 <= seed < 2**32:
        raise ValueError('--training-seed must be in [0, 2**32).')
    if deterministic and seed is None:
        raise ValueError('--deterministic requires --training-seed.')
    return replace(config, training_seed=seed, deterministic=deterministic)


def seed_everything(seed, *, deterministic=False):
    if seed is None:
        return
    if deterministic:
        # Must precede the first CUDA operation. PYTHONHASHSEED cannot be set retroactively.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if os.environ.get('PYTHONHASHSEED') != str(seed):
            warnings.warn('For reproducible hash iteration, set PYTHONHASHSEED before starting Python.')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(bundle, args, output_dir, session_info):
    """Hash actual data and source, not just the dataset's claimed checksum list."""
    config = bundle.config
    root = config.repo_root
    data_root = config.paths.run_dir
    paths = {config.paths.net_file, config.paths.config_file}
    paths.update(data_root.glob('*.xml'))
    for directory in ('training_data', 'validation_data', 'testing_data',
                      'training_data_goal', 'validation_data_goal', 'testing_data_goal'):
        p = data_root / directory
        if p.exists():
            paths.update(f for f in p.rglob('*') if f.is_file())
    for filename in ('scenario_manifest.csv', 'split_protocol.json', 'dataset_files_manifest.csv'):
        if (data_root / filename).exists():
            paths.add(data_root / filename)
    data_hashes = {p.relative_to(data_root).as_posix(): sha256_file(p) for p in sorted(paths)}
    source_hashes = {}
    for directory in ('DQN', 'rp', 'scripts'):
        for p in sorted((root / directory).rglob('*.py')):
            source_hashes[p.relative_to(root).as_posix()] = sha256_file(p)
    libraries = {}
    for name in ('numpy', 'torch', 'torch-geometric', 'eclipse-sumo', 'traci', 'sumolib'):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    agent = bundle.agent
    optimizer_groups = getattr(getattr(agent, 'optimizer', None), 'param_groups', [])
    payload = {
        'environment': config.environment.name, 'variant': config.rp_variant,
        'arguments': vars(args), 'runtime_config': asdict(config),
        'training_seed': config.training_seed, 'deterministic_requested': config.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'python': sys.version, 'platform': platform.platform(), 'libraries': libraries,
        'device': str(agent.device),
        'gpu': torch.cuda.get_device_name(agent.device) if agent.device.type == 'cuda' else None,
        'torch_num_threads': torch.get_num_threads(),
        'sumo_command': session_info.get('cmd'),
        'data_sha256': data_hashes, 'source_sha256': source_hashes,
        'data_digest': hashlib.sha256(json.dumps(data_hashes, sort_keys=True).encode()).hexdigest(),
        'model_shape': asdict(agent.shape) if hasattr(agent, 'shape') and hasattr(agent.shape, '__dataclass_fields__') else None,
        'hyperparameters': {key: getattr(agent, key, None) for key in
                            ('gamma', 'gae_lambda', 'train_batch_size', 'sgd_minibatch_size',
                             'num_sgd_iter', 'clip_param', 'entropy_coeff', 'vf_loss_coeff',
                             'vf_clip_param', 'grad_clip', 'target_kl')},
        'optimizer': [{'lr': group['lr'], 'eps': group.get('eps')} for group in optimizer_groups],
        'scope': 'Fresh-run reproducibility within matching software/device; cross-platform bit equality is not guaranteed.',
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'run_configuration.json'
    # Never replace the original audit trail on a continuation.
    if path.exists():
        suffix = 'resume' if getattr(args, 'resume_state', None) else 'rerun'
        index = 1
        while (output_dir / f'run_configuration_{suffix}_{index}.json').exists():
            index += 1
        path = output_dir / f'run_configuration_{suffix}_{index}.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    print(f'run configuration: {path}')
    return path
