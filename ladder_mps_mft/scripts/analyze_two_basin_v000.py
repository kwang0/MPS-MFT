"""Analyze the synced V=0 raw anchors; reuse the validated V=-0.4 loader."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

import analyze_two_basin_vm04 as common

OUT = common.PROJECT / 'docs/reports/two_basin_v000_20260912'
BULK = common.BULK


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style(family):
    return dict(color=common.COLORS[family], label=common.LABELS[family],
                linestyle='-' if family == 'stripe' else '--', linewidth=1.7)


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=170)
    fig.savefig(OUT / (name + '.pdf'))
    plt.close(fig)


def residual_geometry(d):
    """Separate rescaling from profile motion in the final raw update."""
    channels = {}
    for name in ('spin', 'charge_modulation', 'exchange_charge', 'exchange_spin'):
        x = d['vectors']['applied'][name][-1]
        y = d['vectors']['measured'][name][-1]
        r = y - x
        xx, rr = x @ x, r @ r
        parallel = ((r @ x) / xx) * x
        perpendicular = r - parallel
        np.testing.assert_allclose(parallel @ parallel + perpendicular @ perpendicular,
                                   rr, rtol=1e-12, atol=1e-25)
        record = dict(
            amplitude_fractional_change=np.linalg.norm(y) / np.linalg.norm(x) - 1,
            residual_parallel_power_fraction=(parallel @ parallel) / rr,
            shape_relative_l2=np.linalg.norm(perpendicular) / np.linalg.norm(y))
        if len(x) == 128:
            rung_power = (r * r).reshape(64, 2).sum(axis=1)
            record['residual_power_fraction_rungs_6_to_59'] = rung_power[BULK].sum() / rr
            record['largest_residual_rungs'] = (np.argsort(rung_power)[-8:][::-1] + 1).tolist()
        channels[name] = record
    spin_envelope = d['profiles']['spin_odd'] * (-1.)**np.arange(1, 65)
    positions = {}
    for iteration in (20, 40, 60, 79, 80):
        if iteration > len(spin_envelope):
            continue
        p = spin_envelope[iteration - 1]
        indices = np.flatnonzero(p[:-1] * p[1:] < 0)
        positions[str(iteration)] = [i + 1 - p[i] / (p[i + 1] - p[i]) for i in indices]
    return dict(channels=channels, staggered_spin_zero_crossing_rungs=positions,
                zero_crossing_method='Linear interpolation between adjacent measured MF rungs; not a fitted physical displacement.')


def plots(data):
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.3), layout='constrained')
    for family, d in data.items():
        t = d['history']['iteration']
        axes[0,0].semilogy(t, d['pairing_rms'], **style(family))
        axes[0,1].plot(t, d['spin_rms'], **style(family))
        axes[0,2].semilogy(t, d['history']['field_rel_residual'], **style(family))
        axes[1,0].plot(np.arange(1,65), d['charge'], **style(family))
        axes[1,1].plot(np.arange(1,65), d['physical_spin_odd'] * (-1.)**np.arange(1,65), **style(family))
        axes[1,2].plot(np.arange(1,64)+.5, d['leg'], **style(family))
        axes[1,2].plot(np.arange(1,65), d['rung'], **style(family))
    titles = ['Pairing weakens in both trajectories', 'Spin order grows from the pairing seed',
              'Global residuals remain above tolerance', 'Final charge profiles',
              'Final staggered spin envelopes', 'Final physical leg and rung pairs']
    labels = ['Bulk leg-pairing MF field RMS [t]', 'Bulk leg-odd spin MF field RMS [t]',
              'Global relative field residual', 'Electrons per site, averaged over legs',
              r'$(-1)^x(S^z_0-S^z_1)/2$', 'Symmetrized anomalous pair amplitude']
    for i, ax in enumerate(axes.flat):
        ax.set_title(titles[i], fontsize=10)
        ax.set_ylabel(labels[i]); ax.set_xlabel('MF iteration' if i < 3 else 'Rung')
        ax.grid(alpha=.2)
        if i < 3: ax.set_xlim(1,80)
    axes[0,0].legend(fontsize=9)
    axes[0,1].set_ylim(bottom=0)
    axes[0,2].axhline(1e-4, color='.25', ls=':', label='Relative tolerance = 1e-4')
    axes[0,2].legend(fontsize=8)
    axes[1,1].axhline(0, color='.5', lw=.7)
    axes[1,2].axhline(0, color='.5', lw=.7)
    axes[1,2].text(33, .003, 'Leg pairs', ha='center', fontsize=9)
    axes[1,2].text(33, -.0045, 'Rung pairs', ha='center', fontsize=9)
    fig.suptitle('Square (t0,V)=(1.4,0.0), L=64, chi=200: two raw MF trajectories\n'
                 'Stripe seed: 80 evaluations; pairing seed: wall-time stop at 62; neither endpoint accepted', fontsize=13)
    save(fig, 'basin_convergence')

    fig, axes = plt.subplots(1,3,figsize=(14.4,4.6),layout='constrained')
    for family, d in data.items():
        t = d['history']['iteration']; energy = d['history']['target_density_corrected_variational_energy']/128
        axes[0].plot(t, energy, **style(family))
        axes[1].plot(t[:15], energy[:15], marker='o', markersize=3, **style(family))
        axes[2].plot(t[19:], energy[19:], **style(family))
    for ax, title, limits in zip(axes, ['Full history', 'Beginning: first 15 evaluations', 'Later evolution: expanded energy scale'],
                                 [(1,80),(1,15),(20,80)]):
        ax.set_title(title, fontsize=10); ax.set_xlim(*limits)
        ax.set_xlabel('MF iteration'); ax.set_ylabel('Corrected canonical energy per site [t]')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False); ax.grid(alpha=.2)
    axes[0].legend(fontsize=9); axes[1].set_xticks([1,3,5,7,9,11,13,15])
    axes[1].set_ylim(axes[0].get_ylim())
    end = data['pairing']['summary']
    axes[2].plot(end['iterations'],end['corrected_energy_per_site'], 'x', color=common.COLORS['pairing'], ms=9)
    fig.suptitle('Square (t0,V)=(1.4,0.0): energy from both 95%/5% seeds', fontsize=13)
    fig.supxlabel('Iteration 1 is the first MF evaluation. Cross: pairing-seeded deadline record, density tolerance missed.\n'
                  'These are unfinished trajectory energies; no accepted-state ranking.', fontsize=9)
    save(fig, 'energy_convergence')

    fig, axes = plt.subplots(2,2,figsize=(11.5,7.8),layout='constrained')
    time_colors = ['#8193A6','#497BA2','#1C576F','#8B6C35','#B65F16','#6E252B']
    for col, (family,d) in enumerate(data.items()):
        n = d['summary']['iterations']; chosen = sorted(set([1,10,20,40,min(60,n),n]))
        for color, iteration in zip(time_colors, chosen):
            axes[0,col].plot(np.arange(1,65),d['profiles']['spin_odd'][iteration-1]*(-1.)**np.arange(1,65),
                             color=color,label=str(iteration))
            axes[1,col].plot(np.arange(1,65),d['profiles']['pair_leg_even'][iteration-1],color=color,label=str(iteration))
        axes[0,col].set_title(common.LABELS[family]); axes[0,col].legend(title='MF iteration',ncol=3,fontsize=8)
        axes[0,col].set_ylabel('Staggered leg-odd spin MF field [t]')
        axes[1,col].set_ylabel('Leg-even pairing MF field [t]')
    for ax in axes.flat:
        ax.set_xlabel('Rung'); ax.grid(alpha=.2); ax.axhline(0,color='.5',lw=.6)
    fig.suptitle('Spatial profiles at selected raw evaluations\nSquare (t0,V)=(1.4,0.0), L=64, chi=200',fontsize=13)
    save(fig,'profile_evolution')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data = {family:common.load(family,common.rows(common.RUN/'manifest.tsv'),common.rows(common.RUN/'jobs.tsv'),V=0.)
            for family in ('stripe','pairing')}
    assert data['stripe']['summary']['fingerprints'] == data['pairing']['summary']['fingerprints']
    snapshots, profiles, sources = [], [], []
    for family,d in data.items():
        summary = d['summary']; n = summary['iterations']
        summary['residual_geometry'] = residual_geometry(d)
        spin = .5*(d['correlations']['density_up']-d['correlations']['density_down'])
        d['physical_spin_odd'] = .5*(spin[0::2]-spin[1::2])
        summary['physical_spin_odd_bulk_rms'] = np.sqrt(np.mean(d['physical_spin_odd'][BULK]**2))
        summary['physical_charge_bulk_rms_about_bulk_mean'] = np.std(d['charge'][BULK])
        peak = np.argmax(d['pairing_rms'])
        summary['pairing_peak'] = dict(iteration=peak+1,bulk_rms=d['pairing_rms'][peak],
            terminal_fraction=d['pairing_rms'][-1]/d['pairing_rms'][peak])
        for w in (5,10,20):
            summary[f'last{w}_changes'] = dict(
                spin_rms_fraction=d['spin_rms'][-1]/d['spin_rms'][-w]-1,
                pairing_rms_fraction=d['pairing_rms'][-1]/d['pairing_rms'][-w]-1,
                energy_per_site=(d['history']['target_density_corrected_variational_energy'][-1]-
                                 d['history']['target_density_corrected_variational_energy'][-w])/128)
        summary['inner_dmrg_passes_last5'] = int(d['history']['dmrg_sweep_gate_pass'][-5:].sum())
        summary['spin_change_over_published_floor'] = summary['channel_final']['spin']['absolute']/5e-7
        summary['energy_at_40'] = d['history']['target_density_corrected_variational_energy'][39]/128
        field_pass = (d['history']['field_abs_residual'] <= 1e-7) | (d['history']['field_rel_residual'] <= 1e-4)
        summary['global_field_gate_passing_iterations'] = (np.flatnonzero(field_pass)+1).tolist()
        summary['spin_rms_growth_from_first_measurement'] = d['spin_rms'][-1]/d['spin_rms'][0]
        for name, profile in [('charge',d['charge']),('spin_odd',d['physical_spin_odd'])]:
            amplitude = np.abs(np.fft.rfft(profile-profile.mean()))/64
            mode = np.argmax(amplitude[1:])+1
            summary[f'{name}_full_length_dft'] = dict(mode=int(mode),q_over_pi=mode/32,amplitude=amplitude[mode])
        for i in range(n):
            snapshots.append(dict(family=family,iteration=i+1,
                pairing_field_bulk_rms=d['pairing_rms'][i],spin_field_bulk_rms=d['spin_rms'][i],
                stripe_template_projection=d['projection'][i],
                corrected_energy_per_site=d['history']['target_density_corrected_variational_energy'][i]/128,
                global_relative_residual=d['history']['field_rel_residual'][i],
                density=d['history']['density'][i],inner_dmrg_pass=bool(d['history']['dmrg_sweep_gate_pass'][i])))
            for x in range(64):
                profiles.append(dict(family=family,iteration=i+1,rung=x+1,
                    spin_odd_mf=d['profiles']['spin_odd'][i,x],pair_leg_even_mf=d['profiles']['pair_leg_even'][i,x]))
        path = common.PROJECT / summary['source']
        with h5py.File(path) as f:
            reconstructed = f['history/variational_energy'][()] + 128*f['history/chemical_potential'][()]*(.9375-f['history/density'][()])
            np.testing.assert_allclose(reconstructed,d['history']['target_density_corrected_variational_energy'],rtol=0,atol=5e-13)
            summary['final_mu_search_status'] = f['history/mu_search_status'][-1].decode()
            summary['final_mu_density_converged'] = bool(f['history/mu_density_converged'][-1])
            summary['final_mu_evaluations'] = int(f['history/mu_evaluations'][-1])
            summary['global_slow_mode_extrapolated_relative'] = float(f['fixed_point_extrapolated_rel_residual'][()])
            summary['global_slow_mode_lambda'] = float(f['fixed_point_contraction_estimate'][()])
        sources.append(dict(family=family,job_id=summary['job_id'],path=path.relative_to(common.PROJECT).as_posix(),sha256=summary['source_sha256']))
    comparison = dict(terminal_corrected_energy_difference_pairing_minus_stripe=
        data['pairing']['summary']['corrected_energy_per_site']-data['stripe']['summary']['corrected_energy_per_site'])
    for name,a,b in [('physical_charge',data['stripe']['charge'],data['pairing']['charge']),
                     ('physical_spin_odd',data['stripe']['physical_spin_odd'],data['pairing']['physical_spin_odd'])]:
        a,b=a[BULK],b[BULK]
        comparison[name] = dict(cosine=np.dot(a-a.mean(),b-b.mean())/(np.linalg.norm(a-a.mean())*np.linalg.norm(b-b.mean())),
                                relative_l2_difference=np.linalg.norm(a-b)/np.linalg.norm(a),maximum_difference=np.max(np.abs(a-b)))
    output = dict(point=dict(t0=1.4,V=0.,L=64,chi=200),runs=[d['summary'] for d in data.values()],comparison=comparison,
                  solver_seconds_total=sum(d['summary']['solver_seconds'] for d in data.values()),
                  solver_node_hours_total=sum(d['summary']['solver_node_hours'] for d in data.values()),
                  accounting_basis='Saved MF wall_seconds / 14400; one GPU is quarter node. Allocation overhead excluded; awaiting sacct.')
    receipt=OUT/'sacct_allocations_supplied.txt'
    if receipt.exists():
        with receipt.open() as stream: allocations=list(csv.DictReader(stream,delimiter='|'))
        assert {r['JobIDRaw'] for r in allocations} == {d['summary']['job_id'] for d in data.values()}
        for row in allocations:
            row['node_hours']=int(row['ElapsedRaw'])/14400
        output['allocations']=allocations
        output['allocation_node_hours_total']=sum(r['node_hours'] for r in allocations)
    (OUT/'analysis.json').write_text(json.dumps(common.clean(output),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_csv('energy_history.csv',[r for d in data.values() for r in d['energy_rows']])
    write_csv('channel_history.csv',[r for d in data.values() for r in d['gate_rows']])
    write_csv('basin_history.csv',snapshots); write_csv('spatial_history.csv',profiles); write_csv('sources.csv',sources)
    plots(data)
    for source in sources:
        assert common.sha(common.PROJECT / source['path']) == source['sha256']
    print(json.dumps(common.clean(output),indent=2,allow_nan=False))


if __name__ == '__main__': main()
