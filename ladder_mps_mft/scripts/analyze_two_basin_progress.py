"""September 16 synced progress: read-only HDF5 analysis, no solver runs.

Reuse the existing field/profile conventions. Stitch square parent histories
only after verifying compact/full hashes and the exact raw-map handoff.
Stored acceptance is authoritative; endpoint energies are diagnostics only.
"""
import csv
import json
import re
import tomllib

import h5py
import numpy as np
import matplotlib.pyplot as plt

import analyze_two_basin_vm04 as common

PROJECT = common.PROJECT
ROOT = PROJECT / 'output/phase1_gpu'
OUT = PROJECT / 'docs/reports/two_basin_progress_20260916'
SQUARE = '20260915_square_t014_v000_two_basin_finish20'
CUBIC = '20260915_cubic_unfrustrated_two_basin_95_5_60'
FAMILIES = ('stripe', 'pairing')


def text(f, key):
    return f[key][()].decode()


def nodes(spin):
    """Linear zero crossings of the staggered leg-odd spin, in rung units."""
    envelope = spin * (-1.)**np.arange(64)
    ix = np.flatnonzero(envelope[:-1]*envelope[1:] < 0)
    return ix + 1 - envelope[ix]/(envelope[ix+1]-envelope[ix])


def compact_record(path):
    branch = path.parents[2]
    record = next(r for r in common.rows(branch/'stateless_manifest.tsv')
                  if r['relative_path'] == path.relative_to(branch).as_posix())
    assert common.sha(path) == record['compact_sha256']
    return record


def read_arrays(path):
    with h5py.File(path) as f:
        h = f['history']
        fields = {s: {k: common._julia_array(h[f'fields/{s}/{k}'])
                     for k in ('alpha', 'beta', 'mu_cdw')}
                  for s in ('applied', 'measured')}
        n = len(h['iteration'])
        np.testing.assert_array_equal(h['iteration'][()], np.arange(1, n+1))
        assert [x.decode() for x in h['update_mode'][()]] == ['initial']+['unmixed_probe']*(n-1)
        for k in fields['applied']:
            np.testing.assert_array_equal(fields['applied'][k][..., 1:], fields['measured'][k][..., :-1])
        geom = text(f, 'model/transverse_geometry')
        g = float(f['model/effective_mf_coupling_tp2_over_ep'][()])
        factor = (2 if geom == 'square' else 6)*g
        assert geom in ('square', 'cubic_unfrustrated')
        prof = common.field_profiles(fields['measured']['alpha'], fields['measured']['mu_cdw'])
        # In both geometries K = z*g*leg_swap. Spin is odd under that swap;
        # charge-even = z*g*(n_rung-1)/2. Same-leg alpha = z*g*C_pair.
        spin = -prof['spin_odd']/factor
        charge = 1+2*prof['charge_even']/factor
        a = fields['measured']['alpha']
        ix = np.arange(63)
        leg = .25*(a[ix, ix+1, 0, 0, :]+a[ix+1, ix, 0, 0, :]
                   +a[ix, ix+1, 1, 1, :]+a[ix+1, ix, 1, 1, :]).T/factor
        corr = {k: common._julia_array(v) for k, v in f['correlations'].items()}
        physical_spin = (corr['density_up']-corr['density_down'])/2
        np.testing.assert_allclose(spin[-1], (physical_spin[::2]-physical_spin[1::2])/2, atol=1e-13)
        np.testing.assert_allclose(charge[-1], (corr['density_up']+corr['density_down']).reshape(64, 2).mean(axis=1), atol=1e-13)
        p = corr['pair']; left = 2*ix
        physical_leg = .25*(p[left, left+2]+p[left+2, left]+p[left+1, left+3]+p[left+3, left+1])
        np.testing.assert_allclose(leg[-1], physical_leg, atol=1e-13)
        rung = .5*(p[::2, 1::2].diagonal()+p[1::2, ::2].diagonal())
        energy = h['target_density_corrected_variational_energy'][()]/128
        np.testing.assert_allclose(h['variational_energy'][()]/128+
            h['chemical_potential'][()]*(.9375-h['density'][()]), energy, atol=5e-13, rtol=0)
        assert np.isfinite(energy).all()
        return dict(fields=fields, spin=spin, charge=charge, leg=leg, rung=rung,
                    spin_rms=np.sqrt(np.mean(spin[:, 5:59]**2, axis=1)),
                    pair_rms=np.sqrt(np.mean(leg[:, 5:58]**2, axis=1)), energy=energy)


def load(run, row, jobs, accounting):
    paths = list((run/'results'/row['label']).rglob('state.h5'))
    assert len(paths) == 1
    path = paths[0]; compact = compact_record(path)
    cfgpath = run/'configs'/(row['label']+'.segment-001.toml')
    assert common.sha(cfgpath) == row['config_sha256']
    config = tomllib.loads(cfgpath.read_text()); cfg = config['convergence']
    assert config['dmrg']['maxdim'] == 200
    assert config['mixing']['method'] == 'linear' and config['mixing']['damping'] == 1
    job = next(j for j in jobs if j['label'] == row['label'])
    d = read_arrays(path); n = len(d['energy']); w = int(cfg['stable_iterations'])
    tail = slice(-w, None)
    with h5py.File(path) as f:
        h = f['history']
        assert (f['model/t0'][()], f['model/V'][()], f['model/L'][()]) == (float(row['t0']), float(row['V']), 64)
        fingerprints = {k: text(f, 'provenance/'+k) for k in
            ('model_fingerprint', 'numerical_fingerprint', 'implementation_sha256', 'ep_source_sha256')}
        assert all(v == row[k] for k, v in fingerprints.items())
        assert text(f, 'provenance/config_sha256') == row['config_sha256']
        assert text(f, 'provenance/slurm_job_id') == job['job_id']
        parent_count = 0
        if 'parent_sha256' in row:
            parent = PROJECT/row['source_compact_state'].split('/ladder_mps_mft/')[-1]
            pc = compact_record(parent)
            assert pc['full_sha256'] == row['parent_sha256'] == text(f, 'provenance/parent_sha256')
            assert pc['compact_sha256'] == row['source_compact_sha256']
            pd = read_arrays(parent); parent_count = len(pd['energy'])
            assert parent_count == int(row['parent_iterations'])
            with h5py.File(parent) as pf:
                assert text(pf, 'provenance/model_fingerprint') == fingerprints['model_fingerprint']
                for k in d['fields']['applied']:
                    np.testing.assert_array_equal(d['fields']['applied'][k][..., 0], common._julia_array(pf['fields/restart/'+k]))
            d['parent'] = {k: pd[k] for k in ('energy', 'spin_rms', 'pair_rms', 'spin', 'charge', 'leg')}
        else:
            seed = run/'seeds'/(row['label']+'.h5')
            assert common.sha(seed) == row['seed_sha256'] == text(f, 'provenance/inherit_sha256')
            with h5py.File(seed) as sf:
                for k in d['fields']['applied']:
                    np.testing.assert_array_equal(d['fields']['applied'][k][..., 0], common._julia_array(sf['fields/restart/'+k]))
        channels = {name: {k: v[()] for k, v in group.items()} for name, group in h['channels'].items()}
        vectors = {s: common.channel_vectors(d['fields'][s]) for s in d['fields']}
        for name, c in channels.items():
            x, y = (vectors[s][name] for s in ('applied', 'measured'))
            np.testing.assert_allclose(np.max(np.abs(y-x), axis=1), c['absolute'], atol=1e-15)
            both = np.concatenate((x[tail], y[tail])); span = np.ptp(both, axis=0)
            absolute = np.max(np.abs(span)); relative = np.linalg.norm(span)/max(np.max(np.linalg.norm(both, axis=1)), np.finfo(float).eps)
            np.testing.assert_allclose(absolute, c['window_absolute'][-1], atol=1e-15)
            np.testing.assert_allclose(relative, c['window_relative'][-1], atol=1e-12)
            assert bool(c['window_passes'][-1]) == bool(absolute <= cfg['channel_noise_floor'] or relative <= cfg['field_rel_tol'])
        gates = dict(minimum=n >= cfg['minimum_iterations'],
            global_field=bool(np.all((h['field_abs_residual'][tail] <= cfg['field_abs_tol']) |
                                    (h['field_rel_residual'][tail] <= cfg['field_rel_tol']))),
            global_slow=bool(f['fixed_point_extrapolated_abs_residual'][()] <= cfg['field_abs_tol'] or f['fixed_point_extrapolated_rel_residual'][()] <= cfg['field_rel_tol']),
            density=bool(np.all(np.abs(h['density'][tail]-.9375) <= cfg['density_tol'])),
            inner_dmrg=bool(np.all(h['dmrg_sweep_gate_pass'][tail])),
            energy=bool(np.ptp(d['energy'][tail]) <= cfg['variational_energy_tol']),
            identity=bool(f['hamiltonian_identity_error_per_site'][()] <= cfg['hamiltonian_identity_tol']),
            effective=bool(f['effective_eigenvalue_error_per_site'][()] <= cfg['effective_energy_consistency_tol']))
        for name, c in channels.items():
            gates[name+'_steps'] = bool(np.all(c['passes'][tail]))
            gates[name+'_span'] = bool(c['window_passes'][-1])
        summary = dict(campaign=run.name, label=row['label'], geometry=row['geometry'],
            t0=float(row['t0']), V=float(row['V']), family=row['family'], job_id=job['job_id'],
            source=path.relative_to(PROJECT).as_posix(), compact_sha256=compact['compact_sha256'],
            full_sha256=compact['full_sha256'], config_sha256=row['config_sha256'], fingerprints=fingerprints,
            parent_sha256=row.get('parent_sha256'), status=text(f, 'status'), accepted=bool(f['accepted'][()]),
            period=int(f['fundamental_period'][()]), iterations=n, parent_iterations=parent_count,
            cumulative_iterations=n+parent_count, controls=cfg,
            energy_final=d['energy'][-1], energy_span_last10=np.ptp(d['energy'][tail]),
            energy_change_segment=d['energy'][-1]-d['energy'][0],
            spin_rms_final=d['spin_rms'][-1], pair_rms_final=d['pair_rms'][-1],
            pair_rms_first=d['pair_rms'][0], rung_pair_rms=np.sqrt(np.mean(d['rung'][5:59]**2)),
            charge_std_final=np.std(d['charge'][-1, 5:59]),
            spin_fractional_change_last10=d['spin_rms'][-1]/d['spin_rms'][-10]-1,
            spin_profile_max_change_last10=np.max(np.abs(d['spin'][-1]-d['spin'][-10])),
            node_positions_first=nodes(d['spin'][0]), node_positions_final=nodes(d['spin'][-1]),
            nodes_last10_start=nodes(d['spin'][-10]),
            global_relative=h['field_rel_residual'][-1], slow_relative=f['fixed_point_extrapolated_rel_residual'][()],
            slow_lambda=f['fixed_point_contraction_estimate'][()], slow_cosine=f['fixed_point_residual_cosine'][()],
            gates=gates, failed_gates=[k for k, v in gates.items() if not v],
            channels_final={name: {k: v[-1] for k, v in c.items()} for name, c in channels.items()},
            solver_seconds=h['wall_seconds'][()].sum(), solver_node_hours=h['wall_seconds'][()].sum()/14400,
            reserved_node_hours=float(job['reserved_node_hours']), allocation=None)
        for key in ('spin', 'charge'):
            v = d[key][-1]; amp = np.abs(np.fft.rfft(v-v.mean()))/64
            mode = int(np.argmax(amp[1:])+1)
            summary[key+'_dft'] = dict(mode=mode, q_over_pi=mode/32, amplitude=amp[mode])
        records = [a for a in accounting if a['job_id'] == job['job_id']]
        if records:
            ac = records[-1]
            assert ac['campaign'] == run.name and ac['label'] == row['label'] and ac['sacct_state'] == 'COMPLETED'
            cost = int(ac['elapsed_raw_seconds'])/3600*float(ac['effective_node_fraction'])
            np.testing.assert_allclose(cost, float(ac['measured_node_hours']), atol=1e-9)
            summary['allocation'] = dict(seconds=int(ac['elapsed_raw_seconds']), node_hours=cost,
                reconciled_utc=ac['reconciled_utc'], source='output/project_budget/additional_node_hours_reconciliations.tsv')
    log = (run/'logs'/f"{row['label']}.s1-{job['job_id']}.out").read_text()
    assert [int(i) for i in re.findall(r'^MF\s+(\d+)\s', log, flags=re.M)] == list(range(1, n+1))
    del d['fields']; d['summary'] = summary
    return d


def style(family):
    return dict(color=common.COLORS[family], ls='-' if family == 'stripe' else '--', lw=1.7,
                label=common.LABELS[family])


def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=170)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)


def figures(data):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    square = [d for d in data if d['summary']['geometry'] == 'square']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for d in square:
        s = d['summary']; family = s['family']; offset = s['parent_iterations']
        full = {k: np.concatenate((d['parent'][k], d[k])) for k in ('energy', 'spin_rms', 'pair_rms')}
        for ax, key in zip(axes[0], full):
            ax.plot(np.arange(1, len(full[key])+1), full[key], **style(family))
            ax.axvline(offset+.5, color=common.COLORS[family], alpha=.4, ls=':', lw=1)
            ax.set_xlabel('Cumulative MF evaluation')
        it = np.arange(1, len(d['energy'])+1)+offset
        axes[1, 0].plot(it, d['energy'], **style(family))
        for ax, key in zip(axes[1, 1:], ('charge', 'spin')):
            mult = (-1.)**np.arange(64) if key == 'spin' else 1
            ax.plot(np.arange(1, 65), d[key][-1]*mult, **style(family))
            ax.plot(np.arange(1, 65), d['parent'][key][-1]*mult, color=common.COLORS[family], alpha=.3, lw=1)
            ax.set_xlabel('Rung')
    titles = ('Energy: full histories', 'Spin: full histories', 'Pairing: full histories',
              'Energy: continuation only', 'Charge profiles', 'Staggered spin profiles')
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title); ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('Corrected energy per site [t]')
    axes[1, 0].set_ylabel('Corrected energy per site [t]')
    axes[1, 0].set_xlabel('Cumulative MF evaluation')
    for ax in (axes[0, 0], axes[1, 0]): ax.ticklabel_format(axis='y', useOffset=False, style='plain')
    axes[0, 1].set_ylabel('Bulk leg-odd spin RMS')
    axes[0, 2].set_ylabel('Bulk leg-pair RMS'); axes[0, 2].set_yscale('log')
    axes[1, 1].set_ylabel('Electrons per site')
    axes[1, 2].set_ylabel(r'$(-1)^{r-1}(S^z_{r,1}-S^z_{r,2})/2$')
    axes[0, 1].legend(fontsize=9)
    fig.suptitle('Square (t0,V)=(1.4,0): pairing collapses during the 20-step continuations', fontsize=15)
    fig.supxlabel('L=64, chi=200. Dotted lines: continuation begins. Faint spatial curves: parent endpoints.\n'
                  'Both endpoints remain unaccepted; energies describe evolving trajectories.', fontsize=10)
    save(fig, 'square_continuation')
    cubic = [d for d in data if d['summary']['geometry'] == 'cubic_unfrustrated']
    points = sorted({(d['summary']['t0'], d['summary']['V']) for d in cubic})
    fig, axes = plt.subplots(3, len(points), figsize=(15, 9), layout='constrained')
    for ix, point in enumerate(points):
        for d in cubic:
            s = d['summary']
            if (s['t0'], s['V']) != point: continue
            for iy, key in enumerate(('energy', 'spin_rms', 'pair_rms')):
                axes[iy, ix].plot(np.arange(1, len(d[key])+1), d[key], **style(s['family']))
        axes[0, ix].set_title(f't0={point[0]:g}, V={point[1]:g}')
        axes[0, ix].ticklabel_format(axis='y', useOffset=False, style='plain')
        axes[1, ix].set_ylim(0, .39)
        axes[2, ix].set(yscale='log', ylim=(1e-13, .1), xlabel='MF evaluation')
    for ax in axes.flat: ax.grid(alpha=.2)
    for ax, label in zip(axes[:, 0], ('Corrected energy per site [t]', 'Bulk leg-odd spin RMS', 'Bulk leg-pair RMS')): ax.set_ylabel(label)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('Cubic unfrustrated: both seeds reach stripe states at all four synced points', fontsize=15)
    fig.supxlabel('L=64, chi=200; physical observables (geometry coupling divided out). All eight reach 60 evaluations; none accepted.\n'
                  'Energy panels use individual y scales. The other five coordinates have no synced results.', fontsize=10)
    save(fig, 'cubic_histories')
    fig, axes = plt.subplots(2, len(points), figsize=(15, 6.6), layout='constrained')
    for ix, point in enumerate(points):
        for d in cubic:
            s = d['summary']
            if (s['t0'], s['V']) != point: continue
            for iy, key in enumerate(('charge', 'spin')):
                mult = (-1.)**np.arange(64) if key == 'spin' else 1
                axes[iy, ix].plot(np.arange(1, 65), d[key][-1]*mult, **style(s['family']))
                axes[iy, ix].plot(np.arange(1, 65), d[key][-10]*mult, color=common.COLORS[s['family']], alpha=.25, lw=1)
        axes[0, ix].set_title(f't0={point[0]:g}, V={point[1]:g}')
        axes[1, ix].set_xlabel('Rung')
    for ax in axes.flat: ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('Electrons per site')
    axes[1, 0].set_ylabel('Staggered leg-odd spin')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Cubic endpoint profiles: agreement in stripe texture, with small residual motion', fontsize=15)
    fig.supxlabel('Dark curves: evaluation 60. Faint curves: evaluation 51. Physical profiles; no accepted endpoints.', fontsize=10)
    save(fig, 'cubic_profiles')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    accounting = common.rows(PROJECT/'output/project_budget/additional_node_hours_reconciliations.tsv')
    data = []; inventory = []
    for run in sorted(ROOT.glob('20260915*')):
        manifest, jobs = common.rows(run/'manifest.tsv'), common.rows(run/'jobs.tsv')
        for row in manifest:
            count = len(list((run/'results'/row['label']).rglob('state.h5')))
            inventory.append(dict(campaign=run.name, label=row['label'], state_count=count,
                                  jobs=[j['job_id'] for j in jobs if j['label'] == row['label']]))
            if count and run.name in (SQUARE, CUBIC): data.append(load(run, row, jobs, accounting))
    comparisons = []
    for geom, t0, v in sorted({(d['summary']['geometry'], d['summary']['t0'], d['summary']['V']) for d in data}):
        pair = {d['summary']['family']: d for d in data if
                (d['summary']['geometry'], d['summary']['t0'], d['summary']['V']) == (geom, t0, v)}
        a, b = (pair[f] for f in FAMILIES)
        assert a['summary']['fingerprints'] == b['summary']['fingerprints']
        comparisons.append(dict(geometry=geom, t0=t0, V=v,
            absolute_endpoint_energy_difference=abs(a['energy'][-1]-b['energy'][-1]),
            max_charge_difference=np.max(np.abs(a['charge'][-1]-b['charge'][-1])),
            max_spin_difference=np.max(np.abs(a['spin'][-1]-b['spin'][-1])),
            max_pair_difference=np.max(np.abs(a['leg'][-1]-b['leg'][-1]))))
    summaries = [d['summary'] for d in data]
    report = dict(date='2026-09-16', source='user-synchronized output; no live scheduler access',
        inventory=inventory, runs=summaries, comparisons=comparisons,
        new_mf_evaluations=sum(s['iterations'] for s in summaries),
        square_actual_node_hours=sum(s['allocation']['node_hours'] for s in summaries if s['allocation']),
        cubic_solver_node_hours=sum(s['solver_node_hours'] for s in summaries if s['geometry'] == 'cubic_unfrustrated'))
    (OUT/'analysis.json').write_text(json.dumps(common.clean(report), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    rows = []
    for d in data:
        s = d['summary']
        for i in range(s['iterations']):
            rows.append(dict(campaign=s['campaign'], family=s['family'], t0=s['t0'], V=s['V'],
                segment_iteration=i+1, cumulative_iteration=i+1+s['parent_iterations'],
                corrected_energy_per_site=d['energy'][i], spin_rms=d['spin_rms'][i], pair_rms=d['pair_rms'][i]))
    with (OUT/'histories.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figures(data)
    print(json.dumps(common.clean({k: report[k] for k in ('comparisons', 'new_mf_evaluations', 'square_actual_node_hours', 'cubic_solver_node_hours')}), indent=2))
    for s in summaries:
        print(s['geometry'], s['t0'], s['V'], s['family'], 'failed:', ', '.join(s['failed_gates']),
              'lambda=', s['slow_lambda'], 'spin-node shift=', np.asarray(s['node_positions_final'])-np.asarray(s['nodes_last10_start']))


if __name__ == '__main__':
    main()
