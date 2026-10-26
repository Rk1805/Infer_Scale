"""Build assignment figures and numerical summaries from real held-out runs."""
import csv
import json
import os
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[1]
ARTIFACTS = ROOT / 'artifacts' / 'rl_study'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'build' / 'report-evidence' / 'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

result = json.loads((ARTIFACTS / 'test_evaluation.json').read_text())
manifest = json.loads((ARTIFACTS / 'training_manifest.json').read_text())
assert result['checkpoint_sha256'] == manifest['checkpoint_sha256']
assert result['split'] == 'test'
assert not set(manifest['training_seeds']) & set(result['seeds'])
assert not set(manifest['validation_seeds']) & set(result['seeds'])
profiles = result['profiles']
policies = ['fixed', 'threshold', 'queue', 'ppo']
names = dict(fixed='Fixed', threshold='Threshold', queue='Queue', ppo='PPO')
colors = dict(fixed='#9ba6a3', threshold='#c69861', queue='#66859b', ppo='#22725d')


def stats(values):
    values = np.array(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return dict(mean=mean, ci95_half=half, count=len(values))


summary = dict(training_steps=manifest['training_steps'], selected_step=manifest['selected_step'],
               checkpoint_sha256=result['checkpoint_sha256'], rows=[])
for profile in profiles:
    for policy in policies:
        runs = [row for row in result['runs'] if row['profile'] == profile and row['policy'] == policy]
        assert len(runs) == len(result['seeds'])
        row = dict(profile=profile, policy=policy)
        for metric in ['total_cost', 'p95_latency_ms', 'slo_attainment_lower_bound',
                       'slo_attainment_upper_bound', 'overdue_requests', 'pending_requests',
                       'scaling_events']:
            row[metric] = stats([run['metrics'][metric] for run in runs])
        row['slo_met_runs'] = sum(run['metrics']['slo_attainment_lower_bound'] >= .95 for run in runs)
        savings = []
        for run in runs:
            baseline = next(b for b in result['runs'] if b['profile'] == profile
                            and b['policy'] == 'fixed' and b['seed'] == run['seed'])
            assert baseline['trace_hash'] == run['trace_hash']
            savings.append(100 * (baseline['metrics']['total_cost'] - run['metrics']['total_cost'])
                           / baseline['metrics']['total_cost'])
        row['paired_cost_savings_percent'] = stats(savings)
        summary['rows'].append(row)
(ARTIFACTS / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
definitions = []
for row in summary['rows']:
    prefix = row['profile'].title() + row['policy'].title()
    for suffix, value, digits in [
        ('Cost', row['total_cost']['mean'], 3),
        ('Attainment', 100 * row['slo_attainment_lower_bound']['mean'], 2),
        ('Latency', row['p95_latency_ms']['mean'] / 1000, 3),
    ]:
        definitions.append('\\newcommand{\\' + prefix + suffix + '}{' + f'{value:.{digits}f}' + '}')
(REPORT / 'study_numbers.tex').write_text('\n'.join(definitions) + '\n', encoding='utf-8')

plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.axisbelow': True, 'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 2, figsize=(7, 3.65), layout='constrained')
with (ARTIFACTS / 'training.csv').open() as handle:
    training = list(csv.DictReader(handle))
with (ARTIFACTS / 'validation.csv').open() as handle:
    validation = list(csv.DictReader(handle))
steps = np.array([int(r['timesteps']) for r in training])
returns = np.array([float(r['episode_return']) for r in training])
rolling = np.array([returns[max(0, i-19):i+1].mean() for i in range(len(returns))])
ax = axes[0, 0]
ax.plot(steps, rolling, color='#9ba6a3', lw=1, label='Train: rolling 20 episodes')
ax.plot([int(r['timesteps']) for r in validation],
        [float(r['mean_validation_return']) for r in validation],
        color=colors['ppo'], marker='o', ms=2.5, lw=1, label='Validation: deterministic')
ax.axvline(manifest['selected_step'], color='#c69861', linestyle=':', lw=1)
ax.set(title='(a) Measured learning history', xlabel='Environment steps', ylabel='Episode return')
ax.legend(fontsize=6.7, loc='best', frameon=False)
x = np.arange(len(profiles))
width = .19
for ax, metric, factor, title, ylabel in [
    (axes[0, 1], 'total_cost', 1, '(b) Held-out resource cost', 'USD per 180 s'),
    (axes[1, 0], 'slo_attainment_lower_bound', 100, '(c) Held-out deadline attainment', 'Confirmed on-time (%)'),
    (axes[1, 1], 'p95_latency_ms', .001, '(d) Held-out response latency', 'Mean run p95 (s)'),
]:
    for index, policy in enumerate(policies):
        rows = [next(r for r in summary['rows'] if r['profile'] == profile and r['policy'] == policy)
                for profile in profiles]
        values = [r[metric]['mean'] * factor for r in rows]
        errors = [r[metric]['ci95_half'] * factor for r in rows]
        ax.bar(x + (index-1.5)*width, values, width, color=colors[policy], label=names[policy],
               yerr=errors, capsize=2, error_kw={'linewidth': .65})
    ax.set(xticks=x, xticklabels=[p.title() for p in profiles], title=title, ylabel=ylabel)
    ax.tick_params(labelsize=8)
axes[0, 1].legend(fontsize=6.6, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.02),
                  frameon=False, columnspacing=.65, handlelength=.9)
axes[0, 1].set_ylim(0, max(r['total_cost']['mean'] for r in summary['rows']) * 1.3)
axes[1, 0].axhline(95, color='#a45749', linestyle='--', lw=.8)
axes[1, 0].set_ylim(0, 108)
axes[1, 1].axhline(2, color='#a45749', linestyle='--', lw=.8)
for ax in axes.flat:
    ax.grid(axis='y', alpha=.18)
for ext in ('pdf', 'png'):
    fig.savefig(REPORT / 'figures' / ('assignment_results.' + ext), dpi=240)
plt.close(fig)
for row in summary['rows']:
    print(row['profile'], row['policy'],
          f"cost={row['total_cost']['mean']:.3f}",
          f"on_time={100 * row['slo_attainment_lower_bound']['mean']:.2f}%",
          f"p95={row['p95_latency_ms']['mean'] / 1000:.3f}s",
          f"met={row['slo_met_runs']}/5")
