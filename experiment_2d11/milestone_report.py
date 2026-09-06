"""Publish local historical comparisons without interrupting GPU work."""
import argparse
import csv
from pathlib import Path
import time
from .common import FROZEN, TARGETS, INTERPRETATION, atomic_json, atomic_bytes, read_json, sha256
from .analysis import collect, interim_plot

MILESTONES = {5000: '2_62144B', 10000: '5_24288B', 15000: '7_86432B'}


def publish(run):
    run = Path(run)
    _, evaluations, _ = collect(run)
    historical = {int(r['step']): r for r in csv.DictReader((FROZEN/'baseline_learning_curve.csv').open())}
    published = []
    for u, label in MILESTONES.items():
        output = run/f'INTERIM_{label}.json'
        if output.exists() or not all(f'{kind}:{u}' in evaluations for kind in ('ce', 'hella')):
            continue
        ce, hs = (evaluations[f'{kind}:{u}'] for kind in ('ce', 'hella'))
        assert ce['completed_updates'] == hs['completed_updates'] == u
        assert ce['ownership_verified'] and hs['ownership_verified']
        assert ce['example_count'] == 1280 and hs['example_count'] == 10042
        g = historical[u]
        result = dict(completed_updates=u, logical_targets=u*TARGETS,
            h_ce=ce['score'], g_ce=float(g['val_loss']),
            h_hellaswag=hs['score'], g_hellaswag=float(g['hellaswag_accuracy']),
            comparison='matched exposure, historical G aggregate scores',
            historical_comparison_has_paired_ci=False,
            original_stopping_rules_changed=False,
            endpoint_policy='See current user-authorized run controls; this reporter does not stop or restart training.',
            h_identities=ce['identities'],
            historical_csv_sha256=sha256(FROZEN/'baseline_learning_curve.csv'),
            evaluated_at=max(ce['completed_at'], hs['completed_at']), published_at=time.time(),
            interpretation=INTERPRETATION)
        result['ce_gap_h_minus_g'] = result['h_ce'] - result['g_ce']
        result['hellaswag_advantage_percentage_points'] = 100*(result['h_hellaswag'] - result['g_hellaswag'])
        text = (f"# H versus historical GPT-2 at {u*TARGETS:,} targets\n\n"
            f"Completed updates: {u:,}. Both scores use matching training exposure.\n\n"
            "| Metric | Fresh H | Historical GPT-2 |\n|---|---:|---:|\n"
            f"| Validation CE (lower is better) | {result['h_ce']:.8f} | {result['g_ce']:.8f} |\n"
            f"| HellaSwag accuracy | {100*result['h_hellaswag']:.4f}% | {100*result['g_hellaswag']:.4f}% |\n\n"
            f"CE gap H minus G: {result['ce_gap_h_minus_g']:.8f} nats. "
            f"HellaSwag difference H minus G: {result['hellaswag_advantage_percentage_points']:.4f} percentage points.\n\n"
            "This is a historical aggregate comparison of single training trajectories. "
            "No fresh paired evaluation or paired confidence interval is available for the intermediate GPT-2 weights. "
            "It is not a completed matched 10B comparison.\n\n" + INTERPRETATION + "\n")
        atomic_bytes(run/f'INTERIM_{label}.md', text.encode())
        atomic_json(output, result)
        published.append(str(output))
    if published:
        interim_plot(run)
    return published


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        for path in publish(args.run):
            print('MATCHED_COMPARISON_AVAILABLE ' + path, flush=True)
        if (not args.watch or (args.run/'STOP_VERIFICATION.json').exists()
                or all((args.run/f'INTERIM_{label}.json').exists() for label in MILESTONES.values())):
            break
        time.sleep(30)
