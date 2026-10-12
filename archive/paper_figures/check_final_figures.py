"""Read-only prerequisite checks before scheduling publication figures."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
from src.config import Settings
from src.data_protocol import sha256_file
from src.ml_dataset import load_bundle
from scripts.paper_plots import load_final_records, read_json

OPERATORS = ('fno', 'deeponet', 'pod_deeponet')


def check_inputs(cfg, seeds):
    data = load_bundle(cfg)
    records = load_final_records(cfg, OPERATORS, seeds)
    dataset_hash = sha256_file(cfg.dataset_path)
    selected = read_json(cfg._root(cfg.results_root)/cfg.case_tag/'selection/pod_deeponet_selected.json')
    if selected['dataset_sha256'] != dataset_hash or selected.get('test_used', False):
        raise ValueError('Selection dataset mismatch or test used for selection')
    basis = cfg._root(selected.get('basis_file', cfg.pod_basis_path))
    if sha256_file(basis) != selected['basis_sha256']:
        raise ValueError('Selected POD basis changed')
    for op in OPERATORS:
        for run in records[op]:
            current = replace(cfg, operator=op, run_name=run['run'], train_seed=run['seed'])
            scored = run['summary']
            if scored['dataset_sha256'] != dataset_hash or scored['grid'] != [cfg.Ny, cfg.Nx]:
                raise ValueError('Scored dataset/grid mismatch: '+run['run'])
            if [int(r['case_id']) for r in run['rows']] != np.sort(data['test_idx']).tolist():
                raise ValueError('Scored test IDs mismatch: '+run['run'])
            if sha256_file(current.model_path) != scored['checkpoint_sha256']:
                raise ValueError('Scored checkpoint changed: '+run['run'])
            log = Path(current.log_outdir)
            summary = read_json(log/'training_summary.json')
            if summary['dataset_sha256'] != dataset_hash or summary['seed'] != run['seed']:
                raise ValueError('Training summary mismatch: '+run['run'])
            if summary['architecture'] != scored['architecture']:
                raise ValueError('Training/evaluation architecture mismatch: '+run['run'])
            if not (log/'loss_history.csv').is_file():
                raise FileNotFoundError(log/'loss_history.csv')
            if op == 'pod_deeponet':
                if scored['pod_metadata']['basis_sha256'] != selected['basis_sha256'] or \
                   scored['architecture']['latent'] != selected['winner']['pod_modes']:
                    raise ValueError('POD final run does not match selected basis/rank')
    diagnostic = cfg._root(max(selected['trials'], key=lambda r:r['pod_modes']).get('basis_file', str(basis))).with_suffix('.json')
    info = read_json(diagnostic)
    if info['test_used'] or info['dataset_sha256'] != dataset_hash:
        raise ValueError('Oracle diagnostic dataset/split mismatch')
    print(f"Inputs verified: {len(seeds)*3} checkpoints, {len(data['test_idx'])} paired test cases; selected POD rank {selected['winner']['pod_modes']}.", flush=True)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-tag', default='combined_0500')
    p.add_argument('--seeds', default='321,777,2026,31415,27182')
    a = p.parse_args()
    family, size = a.case_tag.rsplit('_', 1)
    cfg = Settings(dataset_family=family, selected_dataset_size=int(size), n_cases=int(size), device='cpu')
    check_inputs(cfg, [int(x) for x in a.seeds.split(',')])

if __name__ == '__main__':
    main()
