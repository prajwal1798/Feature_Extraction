"""Re-export saved learning/case statistics; no training, inference or benchmark."""
import argparse
import json
from pathlib import Path
from src.config import Settings
from scripts.paper_plots import load_final_records, read_json, plot_learning
from scripts.final_figures import plot_case_rmse, OPERATORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-tag', default='combined_0500')
    parser.add_argument('--seeds', default='321,777,2026,31415,27182')
    parser.add_argument('--primary-seed', type=int, default=321)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or args.primary_seed not in seeds:
        raise ValueError('Use at least two distinct seeds, including the declared primary seed')
    seeds = [args.primary_seed] + [s for s in seeds if s != args.primary_seed]
    family, size = args.case_tag.rsplit('_', 1)
    cfg = Settings(dataset_family=family, selected_dataset_size=int(size), device='cpu')
    records = load_final_records(cfg, OPERATORS, seeds)
    selected = read_json(cfg._root(cfg.results_root)/cfg.case_tag/'selection/pod_deeponet_selected.json')
    pod_modes = int(selected['winner']['pod_modes'])
    if selected.get('test_used', False):
        raise ValueError('POD selection record reports use of the test set')
    for run in records['pod_deeponet']:
        scored = run['summary']
        if (scored['dataset_sha256'] != selected['dataset_sha256'] or
            scored['architecture']['latent'] != pod_modes or
            scored['pod_metadata']['basis_sha256'] != selected['basis_sha256']):
            raise ValueError('Final POD scores do not match the selected rank/basis')
    provenance = []
    for op, runs in records.items():
        for run in runs:
            folder = cfg._root(cfg.logs_root)/cfg.case_tag/run['run']
            summary = read_json(folder/'training_summary.json')
            scored = run['summary']
            if (summary['dataset_sha256'] != scored['dataset_sha256'] or
                summary['architecture'] != scored['architecture'] or
                summary['operator'] != op or summary['seed'] != run['seed']):
                raise ValueError('Training/evaluation provenance mismatch: '+run['run'])
            provenance.append(dict(operator=op, seed=run['seed'],
                history=str(folder/'loss_history.csv'), checkpoint_sha256=scored['checkpoint_sha256']))
    output = Path(args.output) if args.output else cfg._root(cfg.figures_root)/cfg.case_tag/'paper_final'
    output.mkdir(parents=True, exist_ok=True)
    plot_learning(cfg, records, output)
    plot_case_rmse(records, output)
    notes = dict(primary_seed=args.primary_seed, seeds=seeds, selected_pod_modes=pod_modes,
        test_cases=len(records[OPERATORS[0]][0]['rows']), provenance=provenance,
        learning='04: declared primary seed only, fully opaque; 04b: all seeds in separate panels. '
                 'RMSE_K = sqrt(recorded point-weighted masked MSE)*(saved T_max - saved T_min). '
                 'No smoothing, clipping, curve averaging or selection of a favourable seed. '
                 'Training is the minibatch aggregate during optimization, not a reevaluated end-of-epoch score.',
        case_rmse='11: each operator has its own linear-y panel, with sorted numeric test case IDs. '
                  'The line is the arithmetic mean of that case RMSE across seeds; shading is mean ± one sample SD (ddof=1). '
                  'This is seed variability, not a confidence interval or ensemble prediction error. '
                  'Models have separate y ranges. Lines connect identifiers as visual guides, not a physical progression. '
                  'Negative lower band limits are retained and are statistical limits, not negative RMSE observations.')
    (output/'REVISED_CURVES_NOTES.json').write_text(json.dumps(notes, indent=2))
    print(f'Exported 04, 04b, 11 as PDF/SVG/600-dpi PNG, plus CSV data and notes: {output}', flush=True)
    print(f'Primary seed: {args.primary_seed}; seed band: {seeds}; test cases: {notes["test_cases"]}; selected POD modes: {pod_modes}', flush=True)


if __name__ == '__main__':
    main()
