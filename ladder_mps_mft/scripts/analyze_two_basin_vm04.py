"""Analyze synced 95/5 chi=200 anchors at (t0,V)=(1.4,-0.4), without DMRG.

Preserves source artifacts and acceptance flags. Energies are trajectory
diagnostics, not an accepted-solution ranking. Times are solver-time estimates
unless allocation accounting has been synchronized separately.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit_scf_numerics import _julia_array
from audit_spatial_phase_defects import field_profiles
from plot_mf_energy_histories import read_history

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "output/phase1_gpu/20260908_square_two_basin_95_5_80_anchors"
OUT = PROJECT / "docs/reports/two_basin_vm04_20260910"
BULK = slice(5, 59)  # Rungs 6--59, matching the seed/basin reports.
COLORS = {"stripe": "#245A9C", "pairing": "#B65F16"}
LABELS = {"stripe": "95% stripe seed", "pairing": "95% pairing seed"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_csv(name, values):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return str(value)
    return value


def channel_vectors(fields):
    # Input arrays have Julia-native axes, with iteration last.
    n = fields["alpha"].shape[-1]
    alpha, beta, mu = (fields[k] for k in ("alpha", "beta", "mu_cdw"))
    charge = mu.mean(axis=0).T
    return dict(pairing=alpha.reshape(-1, n).T,
                exchange_charge=((beta[0] + beta[1]) / 2).reshape(-1, n).T,
                exchange_spin=((beta[1] - beta[0]) / 2).reshape(-1, n).T,
                charge_uniform=charge.mean(axis=1, keepdims=True),
                charge_modulation=charge - charge.mean(axis=1, keepdims=True),
                spin=((mu[1] - mu[0]) / 2).T)


def load(family, manifest, jobs):
    row = next(r for r in manifest if r['family'] == family and float(r['V']) == -.4 and float(r['t0']) == 1.4)
    branch_dir = RUN / 'results' / row['label']
    paths = list(branch_dir.rglob('state.h5'))
    assert len(paths) == 1, paths
    path = paths[0]
    compact_row = next(r for r in rows(branch_dir / 'stateless_manifest.tsv') if r['relative_path'] == path.relative_to(branch_dir).as_posix())
    assert sha(path) == compact_row['compact_sha256']
    config_path = RUN / 'configs' / (row['label'] + '.segment-001.toml')
    config = tomllib.loads(config_path.read_text())
    assert sha(config_path) == row['config_sha256']
    seed_path = RUN / 'seeds' / (row['label'] + '.h5')
    assert sha(seed_path) == row['seed_sha256'] == config['run']['inherit_sha256']
    cfg = config['convergence']
    job = next(r for r in jobs if r['label'] == row['label'])
    with h5py.File(path) as f:
        text = lambda k: f[k][()].decode()
        assert (f['model/L'][()], f['model/t0'][()], f['model/V'][()]) == (64, 1.4, -.4)
        assert text('provenance/slurm_job_id') == job['job_id']
        for key in ('model_fingerprint', 'numerical_fingerprint', 'implementation_sha256', 'ep_source_sha256'):
            assert text('provenance/' + key) == row[key]
        assert text('provenance/config_sha256') == row['config_sha256']
        assert text('provenance/inherit_sha256') == row['seed_sha256']
        h = f['history']; n = len(h['iteration'])
        np.testing.assert_array_equal(h['iteration'][()], np.arange(1, 81))
        modes = [x.decode() for x in h['update_mode'][()]]
        assert modes == ['initial'] + ['unmixed_probe'] * 79
        fields = {s: {k: _julia_array(h[f'fields/{s}/{k}']) for k in ('alpha', 'beta', 'mu_cdw')}
                  for s in ('applied', 'measured')}
        with h5py.File(seed_path) as seed:
            assert seed['seed_provenance/epsilon'][()] == .05
            for key in fields['applied']:
                np.testing.assert_array_equal(fields['applied'][key][..., 0], _julia_array(seed['fields/restart/' + key]))
        for key in fields['applied']:
            np.testing.assert_array_equal(fields['applied'][key][..., 1:], fields['measured'][key][..., :-1])
        profiles = field_profiles(fields['measured']['alpha'], fields['measured']['mu_cdw'])
        initial_profiles = field_profiles(fields['applied']['alpha'][..., :1], fields['applied']['mu_cdw'][..., :1])
        vectors = {s: channel_vectors(fields[s]) for s in fields}
        channel_data = {name: {key: value[()] for key, value in group.items()} for name, group in h['channels'].items()}
        gate_rows = []
        for name, data in channel_data.items():
            x, y = vectors['applied'][name], vectors['measured'][name]
            absolute = np.max(np.abs(y - x), axis=1)
            relative = np.linalg.norm(y - x, axis=1) / np.maximum.reduce([np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1), np.full(n, np.finfo(float).eps)])
            np.testing.assert_allclose(absolute, data['absolute'], rtol=1e-7, atol=1e-16)
            np.testing.assert_allclose(relative, data['relative'], rtol=1e-7, atol=1e-12)
            for i in range(n):
                gate_rows.append(dict(family=family, iteration=i+1, channel=name,
                    **{k: clean(v[i]) for k, v in data.items()}))
        history = {k: h[k][()] for k in ('iteration', 'wall_seconds', 'field_abs_residual', 'field_rel_residual',
            'density', 'chemical_potential', 'dmrg_sweep_gate_pass', 'target_density_corrected_variational_energy')}
        final_dmrg = h['dmrg']['0080']
        final_sweeps = final_dmrg['sweep_energy'][()]
        correlations = {key: _julia_array(value) for key, value in f['correlations'].items()}
        pair = correlations['pair']
        rung = .5 * (pair[0::2, 1::2].diagonal() + pair[1::2, 0::2].diagonal())
        left = np.arange(0, 126, 2)
        leg = .25 * (pair[left, left+2] + pair[left+2, left] + pair[left+1, left+3] + pair[left+3, left+1])
        charge = (correlations['density_down'] + correlations['density_up']).reshape(64, 2).mean(axis=1)
        energy = history['target_density_corrected_variational_energy'] / 128
        spin_rms = np.sqrt(np.mean(profiles['spin_odd'][:, BULK]**2, axis=1))
        pairing_rms = np.sqrt(np.mean(profiles['pair_leg_even'][:, BULK]**2, axis=1))
        template = initial_profiles['spin_odd'][0, BULK]
        projection = profiles['spin_odd'][:, BULK] @ template / (template @ template)
        tail = slice(-5, None)
        gates = dict(
            minimum_iterations=n >= cfg['minimum_iterations'],
            global_field_window=bool(np.all((history['field_abs_residual'][tail] <= cfg['field_abs_tol']) | (history['field_rel_residual'][tail] <= cfg['field_rel_tol']))),
            global_slow_mode=bool(f['fixed_point_extrapolated_abs_residual'][()] <= cfg['field_abs_tol'] or f['fixed_point_extrapolated_rel_residual'][()] <= cfg['field_rel_tol']),
            density_window=bool(np.max(np.abs(history['density'][tail]-.9375)) <= cfg['density_tol']),
            inner_dmrg_window=bool(np.all(history['dmrg_sweep_gate_pass'][tail])),
            corrected_energy_window=bool(np.ptp(energy[tail]) <= cfg['variational_energy_tol']),
            identity=bool(f['hamiltonian_identity_error_per_site'][()] <= cfg['hamiltonian_identity_tol']),
            effective_energy=bool(f['effective_eigenvalue_error_per_site'][()] <= cfg['effective_energy_consistency_tol']),
            **{name: bool(np.all(data['passes'][tail])) for name, data in channel_data.items()})
        summary = dict(family=family, job_id=job['job_id'], source=str(path.relative_to(PROJECT)),
            source_sha256=sha(path), full_source_sha256=compact_row['full_sha256'], config_sha256=sha(config_path),
            fingerprints={key: row[key] for key in ('model_fingerprint','numerical_fingerprint','implementation_sha256','ep_source_sha256')},
            iterations=n, status=text('status'), accepted=bool(f['accepted'][()]), gates_last5=gates,
            corrected_energy_per_site=energy[-1], corrected_energy_span_last5=np.ptp(energy[-5:]),
            corrected_energy_span_last20=np.ptp(energy[-20:]), corrected_energy_span_last40=np.ptp(energy[-40:]),
            global_relative_residual=history['field_rel_residual'][-1],
            density_error=abs(history['density'][-1]-.9375),
            identity_error_per_site=f['hamiltonian_identity_error_per_site'][()],
            last_dmrg_sweep_delta_total=abs(final_sweeps[-1]-final_sweeps[-2]),
            last_dmrg_sweep_discarded_weight=final_dmrg['sweep_max_discarded_weight'][-1],
            solver_seconds=history['wall_seconds'].sum(), solver_node_hours=history['wall_seconds'].sum()/14400,
            reserved_node_hours=float(job['reserved_node_hours']),
            pairing_field_rms_final=pairing_rms[-1], spin_field_rms_initial=float(np.sqrt(np.mean(template**2))),
            spin_field_rms_first=spin_rms[0], spin_field_rms_final=spin_rms[-1],
            spin_field_rms_last20_range=[spin_rms[-20:].min(), spin_rms[-20:].max()],
            stripe_template_projection_last20_range=[projection[-20:].min(), projection[-20:].max()],
            first_spin_rms_below_1e7=int(np.flatnonzero(spin_rms<1e-7)[0]+1),
            final_rung_pair_mean=rung[BULK].mean(), final_leg_pair_mean=leg[5:58].mean(),
            channel_final={name: {key: clean(value[-1]) for key, value in data.items()} for name,data in channel_data.items()},
            channel_last20={name: dict(max_absolute=data['absolute'][-20:].max(),
                max_relative=data['relative'][-20:].max(), median_cosine=np.median(data['cosine'][-20:]),
                passes=int(data['passes'][-20:].sum())) for name,data in channel_data.items()})
        assert np.isfinite(energy).all()
    _, energy_rows = read_history(path)
    log = (RUN / 'logs' / f"{row['label']}.s1-{job['job_id']}.out").read_text()
    log_iterations = re.findall(r'^MF\s+(\d+)\s', log, flags=re.M)
    assert [int(i) for i in log_iterations] == list(range(1,81))
    return dict(summary=summary, history=history, profiles=profiles, correlations=correlations,
        initial=initial_profiles, channel_data=channel_data, gate_rows=gate_rows, energy_rows=energy_rows,
        rung=rung, leg=leg, charge=charge, spin_rms=spin_rms, pairing_rms=pairing_rms, projection=projection)


def plot_energy_history(data, reference):
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), layout='constrained')
    for family, d in data.items():
        iteration = d['history']['iteration']
        energy = d['history']['target_density_corrected_variational_energy'] / 128
        style = dict(color=COLORS[family], label=LABELS[family],
                     linestyle='-' if family == 'stripe' else '--', linewidth=1.7)
        axes[0].plot(iteration, energy, **style)
        axes[1].plot(iteration[:15], energy[:15], marker='o', markersize=3, **style)
        axes[2].plot(iteration[19:], (energy[19:] - reference) / 1e-9, **style)
    for ax in axes[:2]:
        ax.set_ylabel('Corrected canonical energy per site [t]')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    axes[1].set_ylim(axes[0].get_ylim())
    axes[2].set_ylabel(r'$(E/N - E_{\mathrm{ref}})\,/\,(10^{-9}\,t)$')
    axes[2].axhline(0, color='0.5', linewidth=.7, zorder=0)
    for ax, title, limits in zip(axes,
            ('Full history', 'Beginning: first 15 evaluations', 'Late fluctuations: expanded energy scale'),
            ((1, 80), (1, 15), (20, 80))):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('MF iteration')
        ax.set_xlim(*limits)
        ax.grid(alpha=.2)
    axes[1].set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    axes[0].legend(fontsize=9, loc='center right')
    fig.suptitle('Square (t0,V)=(1.4,-0.4), L=64, chi=200: energy from both 95%/5% seeds', fontsize=13)
    fig.supxlabel(
        f'Eref = {reference:.12f} t/site (mean of both final 20 records). '
        'Iteration 1 is the first MF evaluation, not a separate seed energy.\n'
        'Both endpoints remain unaccepted; these curves compare trajectories.', fontsize=9)
    fig.savefig(OUT/'energy_convergence.png', dpi=180)
    fig.savefig(OUT/'energy_convergence.pdf')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, jobs = rows(RUN/'manifest.tsv'), rows(RUN/'jobs.tsv')
    data = {family: load(family, manifest, jobs) for family in ('stripe','pairing')}
    a,b=data.values()
    assert a['summary']['fingerprints'] == b['summary']['fingerprints']
    agreement = {}
    for key in a['correlations']:
        x,y=a['correlations'][key],b['correlations'][key]
        agreement[key] = dict(max_absolute_difference=np.max(np.abs(x-y)), relative_l2_difference=np.linalg.norm(x-y)/max(np.linalg.norm(x),np.linalg.norm(y)))
    agreement['corrected_energy_per_site_absolute_difference']=abs(a['summary']['corrected_energy_per_site']-b['summary']['corrected_energy_per_site'])
    with (OUT/'sacct_allocations_supplied.txt').open() as stream:
        allocations = list(csv.DictReader(stream, delimiter='|'))
    assert {r['JobIDRaw'] for r in allocations} == {d['summary']['job_id'] for d in data.values()}
    for row in allocations:
        assert row['State'] == 'COMPLETED'
        row['elapsed_seconds'] = int(row['ElapsedRaw'])
        row['node_hours'] = row['elapsed_seconds'] / 14400
    for d in data.values():
        allocation = next(r for r in allocations if r['JobIDRaw'] == d['summary']['job_id'])
        d['summary']['allocation_elapsed_seconds'] = allocation['elapsed_seconds']
        d['summary']['allocation_node_hours'] = allocation['node_hours']
    summary=dict(point=dict(L=64,t0=1.4,V=-.4,chi=200), runs=[d['summary'] for d in data.values()],
        terminal_agreement=agreement, total_solver_seconds=sum(d['summary']['solver_seconds'] for d in data.values()),
        total_solver_node_hours=sum(d['summary']['solver_node_hours'] for d in data.values()),
        timing_basis='Solver: sum of saved MF wall_seconds / 14400, excluding allocation overhead. Allocation: user-supplied sacct ElapsedRaw / 14400, one GPU = quarter node.',
        allocations=allocations, total_allocation_node_hours=sum(r['node_hours'] for r in allocations),
        allocation_source='sacct_allocations_supplied.txt; output pasted by user on 2026-09-10; local accounting ledger is not modified',
        reservation_node_hours_completed_pair=6, reservation_node_hours_all_four_anchors=12)
    (OUT/'analysis.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_csv('energy_history.csv',[r for d in data.values() for r in d['energy_rows']])
    write_csv('channel_history.csv',[r for d in data.values() for r in d['gate_rows']])
    write_csv('basin_history.csv',[dict(family=family,iteration=i+1,leg_pairing_field_bulk_rms=d['pairing_rms'][i],
        spin_odd_field_bulk_rms=d['spin_rms'][i],signed_projection_onto_initial_stripe=d['projection'][i])
        for family,d in data.items() for i in range(80)])
    fig,axes=plt.subplots(2,3,figsize=(13.8,8),layout='constrained')
    reference=np.mean(np.concatenate([d['history']['target_density_corrected_variational_energy'][-20:]/128 for d in data.values()]))
    for family,d in data.items():
        t=d['history']['iteration']; color=COLORS[family]; label=LABELS[family]
        style='-' if family=='stripe' else '--'
        axes[0,0].plot(t,d['pairing_rms'],style,color=color,label=label)
        axes[0,1].semilogy(t,d['spin_rms'],style,color=color,label=label)
        axes[0,2].plot(t[19:],d['history']['target_density_corrected_variational_energy'][19:]/128-reference,style,color=color,label=label)
        axes[1,0].plot(np.arange(1,65),d['rung'],style,color=color,label=label+'; rung')
        axes[1,0].plot(np.arange(1,64)+.5,d['leg'],style,color=color,label=label+'; leg')
        axes[1,1].plot(np.arange(1,65),d['charge'],style,color=color,label=label)
        axes[1,2].plot(t[39:],d['channel_data']['spin']['absolute'][39:],style,color=color,label=label)
    titles=['Pairing grows from the stripe seed','Stripe spin order decays to a small plateau','Corrected energy fluctuates near a common value',
            'Final physical pairs: opposite leg/rung signs','Final charge profiles overlap','Small spin residuals exceed the absolute gate']
    ylabels=['Bulk leg-pairing MF field RMS [t]','Bulk leg-odd spin MF field RMS [t]',
             'Energy/site minus common tail mean [t]','Symmetrized anomalous pair amplitude','Electrons per site, averaged over legs','Maximum spin-channel residual [t]']
    for i,ax in enumerate(axes.flat):
        ax.set_title(titles[i],fontsize=10)
        ax.set_ylabel(ylabels[i]); ax.set_xlabel('MF iteration' if i<3 or i==5 else 'Rung')
        ax.grid(alpha=.2)
        if i!=1: ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useOffset=False)
        if i in (0,1): ax.set_xlim(1,80)
    axes[0,0].set_ylim(bottom=0); axes[1,0].axhline(0,color='0.45',lw=.7)
    axes[1,2].axhline(1e-7,color='0.25',ls=':',label='Absolute tolerance = 1e-7')
    axes[0,0].legend(fontsize=9); axes[1,2].legend(fontsize=8)
    axes[1,0].text(34,.027,'Leg pairs',ha='center',fontsize=9)
    axes[1,0].text(34,-.042,'Rung pairs',ha='center',fontsize=9)
    fig.suptitle('Square (t0,V)=(1.4,-0.4), L=64, chi=200: two unmixed 80-iteration trajectories\n'
                 'Bulk RMS uses rungs 6–59; both stored endpoints remain unaccepted',fontsize=13)
    fig.savefig(OUT/'basin_convergence.png',dpi=160)
    fig.savefig(OUT/'basin_convergence.pdf')
    plt.close(fig)
    plot_energy_history(data, reference)
    print(json.dumps(clean(summary),indent=2,allow_nan=False))


if __name__=='__main__': main()
