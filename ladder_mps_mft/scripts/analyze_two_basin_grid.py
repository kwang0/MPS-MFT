"""Preliminary square phase assignment from all 18 synced raw 95/5 histories.

No DMRG, source mutation, acceptance relabeling, or accepted-energy ranking.
Run from any directory with the existing Python scientific environment.
"""
import csv
import json
import tomllib
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import analyze_two_basin_vm04 as common

PROJECT = common.PROJECT
OUT = PROJECT / 'docs/reports/two_basin_grid_20260915'
CAMPAIGNS = ('20260908_square_two_basin_95_5_80_anchors',
             '20260910_square_two_basin_95_5_40_remainder')
T0 = (1., 1.2, 1.4)
VS = (0., -.2, -.4)
PHASE_COLORS = {'S': '#245A9C', 'D': '#A24700', 'S*': '#245A9C', '?': '#60656C'}


def write_csv(name, values):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(common.clean(values))


def classify(spin, pair, opposite_signs):
    # Descriptive separators inside the observed amplitude gaps, not critical
    # exponents, thermodynamic thresholds, or convergence criteria.
    if spin > 1e-3 and pair < 1e-4:
        return 'S'
    if spin < 1e-4 and pair > 1e-3 and opposite_signs:
        return 'D'
    return 'S+P' if spin > 1e-3 and pair > 1e-4 else '?'


def analyze_run(row, run_directory, manifest, jobs, accounting):
    t0, v, family = float(row['t0']), float(row['V']), row['family']
    d = common.load(family, manifest, jobs, t0=t0, V=v, run_directory=run_directory)
    config_path = run_directory/'configs'/(row['label']+'.segment-001.toml')
    assert tomllib.loads(config_path.read_text())['dmrg']['maxdim'] == 200
    s, h, cfg = d['summary'], d['history'], d['controls']
    s.update(t0=t0, V=v, campaign=run_directory.name, label=row['label'])
    n, w = s['iterations'], int(cfg['stable_iterations'])
    tail = slice(-w, None)
    energy = h['target_density_corrected_variational_energy'] / 128
    spin = (d['correlations']['density_up'] - d['correlations']['density_down']) / 2
    spin_odd = (spin[::2] - spin[1::2]) / 2
    spin_rms = np.sqrt(np.mean(spin_odd[common.BULK]**2))
    pair_rms = np.sqrt(np.mean(d['leg'][5:58]**2))
    opposite = bool(s['final_leg_pair_mean'] * s['final_rung_pair_mean'] < 0)
    s.update(physical_spin_odd_bulk_rms=spin_rms, physical_leg_pair_bulk_rms=pair_rms,
             physical_charge_bulk_std=np.std(d['charge'][common.BULK]),
             physical_leg_pair_bulk_std=np.std(d['leg'][5:58]),
             phase_observation=classify(spin_rms, pair_rms, opposite),
             energy_span_last10=np.ptp(energy[-10:]), stable_window=w,
             spin_fractional_change_last10=d['spin_rms'][-1]/d['spin_rms'][-10]-1,
             pairing_fractional_change_last10=d['pairing_rms'][-1]/d['pairing_rms'][-10]-1,
             spin_fractional_change_last5=d['spin_rms'][-1]/d['spin_rms'][-5]-1,
             pairing_fractional_change_last5=d['pairing_rms'][-1]/d['pairing_rms'][-5]-1)
    # Recompute configured-window gates; the anchor helper's gates_last5 is
    # intentionally retained only as a historical five-record diagnostic.
    gates = dict(minimum_iterations=n >= cfg['minimum_iterations'],
        global_field=bool(np.all((h['field_abs_residual'][tail] <= cfg['field_abs_tol']) |
                                (h['field_rel_residual'][tail] <= cfg['field_rel_tol']))),
        global_slow_mode=s['gates_last5']['global_slow_mode'],
        density=bool(np.max(np.abs(h['density'][tail]-.9375)) <= cfg['density_tol']),
        inner_dmrg=bool(np.all(h['dmrg_sweep_gate_pass'][tail])),
        energy=bool(np.ptp(energy[tail]) <= cfg['variational_energy_tol']),
        identity=s['gates_last5']['identity'], effective_energy=s['gates_last5']['effective_energy'])
    for channel, values in d['channel_data'].items():
        gates[channel+'_steps'] = bool(np.all(values['passes'][tail]))
        if cfg.get('channel_noise_floor', 0) > 0:
            x, y = (d['vectors'][source][channel][tail] for source in ('applied', 'measured'))
            all_vectors = np.concatenate((x, y))
            span = np.ptp(all_vectors, axis=0)
            absolute = np.max(np.abs(span))
            relative = np.linalg.norm(span)/max(np.max(np.linalg.norm(all_vectors, axis=1)), np.finfo(float).eps)
            np.testing.assert_allclose(absolute, values['window_absolute'][-1], rtol=1e-7, atol=1e-15)
            np.testing.assert_allclose(relative, values['window_relative'][-1], rtol=1e-7, atol=1e-12)
            passes = bool(absolute <= cfg['channel_noise_floor'] or relative <= cfg['field_rel_tol'])
            assert passes == bool(values['window_passes'][-1])
            gates[channel+'_span'] = passes
    s['gates_configured_window'] = gates
    s['failed_gates'] = [key for key, value in gates.items() if not value]
    s['controls'] = cfg
    with h5py.File(PROJECT/s['source']) as f:
        s['period'] = int(f['fundamental_period'][()])
        s['global_slow_mode_lambda'] = float(f['fixed_point_contraction_estimate'][()])
        s['global_slow_mode_relative'] = float(f['fixed_point_extrapolated_rel_residual'][()])
        s['last_density_search_status'] = f['history/mu_search_status'][-1].decode()
        s['last_density_search_converged'] = bool(f['history/mu_density_converged'][-1])
        assert f['model/transverse_geometry'][()].decode() == 'square'
        coupling = float(f['provenance/effective_mf_coupling_tp2_over_ep'][()])
        # Square density kernel swaps the two legs: the leg-odd spin
        # combination therefore acquires a minus sign.
        np.testing.assert_allclose(d['profiles']['spin_odd'][-1], -2*coupling*spin_odd, rtol=1e-7, atol=1e-13)
        np.testing.assert_allclose(f['history/variational_energy'][()] / 128 +
            h['chemical_potential'] * (.9375-h['density']), energy, rtol=0, atol=5e-13)
    # Independently reconcile the physical spin expectation with measured MF.
    nonzero = np.abs(spin_odd) > 1e-8
    ratios = d['profiles']['spin_odd'][-1][nonzero] / spin_odd[nonzero]
    s['spin_field_to_physical_ratio'] = float(np.median(ratios)) if len(ratios) else None
    for name, profile in (('charge', d['charge']), ('spin', spin_odd)):
        amplitudes = np.abs(np.fft.rfft(profile-profile.mean())) / 64
        mode = int(np.argmax(amplitudes[1:])+1)
        s[name+'_dft'] = dict(mode=mode, q_over_pi=mode/32, amplitude=amplitudes[mode])
    records = [r for r in accounting if r['job_id'] == s['job_id']]
    if records:
        record = records[-1]
        assert record['campaign'] == run_directory.name
        assert record['label'] == row['label'] and record['sacct_state'] == 'COMPLETED'
        cost = int(record['elapsed_raw_seconds'])/3600*float(record['effective_node_fraction'])
        np.testing.assert_allclose(cost, float(record['measured_node_hours']), atol=1e-9, rtol=0)
        s['allocation'] = dict(source='output/project_budget/additional_node_hours_reconciliations.tsv',
            reconciled_utc=record['reconciled_utc'], seconds=int(record['elapsed_raw_seconds']), node_hours=cost)
    # Retain small derived arrays, not all full vector histories, across runs.
    return dict(summary=s, history=h, energy=energy, spin_rms=d['spin_rms'],
                pairing_rms=d['pairing_rms'], charge=d['charge'], spin=spin_odd,
                pair_leg=d['leg'], pair_rung=d['rung'])


def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=180)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)


def figures(data, points):
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9, 7.4))
    fig.subplots_adjust(left=.12, right=.97, top=.79, bottom=.23)
    ax.set(xlim=(.90,1.50), ylim=(-.48,.075), xticks=T0, yticks=VS,
           xlabel=r'$t_0/t$', ylabel=r'$V/t$')
    ax.grid(alpha=.2, zorder=0)
    for p in points:
        phase=p['phase']
        ax.scatter(p['t0'],p['V'],s=2100,c=PHASE_COLORS[phase],
                   marker='s' if phase=='D' else 'o',edgecolors='white',linewidths=2,zorder=3)
        ax.text(p['t0'],p['V'],phase,color='white',fontsize=23,fontweight='bold',ha='center',va='center',zorder=4)
        ax.text(p['t0'],p['V']-.057,p['seed_summary'],ha='center',va='top',fontsize=10,color='#333333')
    fig.suptitle('Preliminary square-grid phase diagram',fontsize=19,y=.97)
    fig.text(.5,.92,r'$L=64$, $\chi=200$, $n=0.9375$, $t_\perp/t=0.1$ | raw MF updates; reciprocal 95% / 5% seeds',ha='center',fontsize=11)
    legend=[Line2D([],[],marker='o',color='none',markerfacecolor=PHASE_COLORS['S'],markeredgecolor='none',markersize=11,label='S: stripe CDW / SDW'),
            Line2D([],[],marker='s',color='none',markerfacecolor=PHASE_COLORS['D'],markeredgecolor='none',markersize=11,label='D: d-wave-like paired basin')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.895),ncol=2,frameon=False,fontsize=11)
    fig.text(.12,.135,'S*: stripe start has negligible pairing; pairing start is still evolving toward stripes.',fontsize=10)
    fig.text(.12,.093,'Labels describe the two observed trajectories. All 18 endpoints remain formally unaccepted.',fontsize=10,fontweight='bold')
    fig.text(.12,.052,'No interpolated boundary or converged ground-state energy ranking.\n14 starts: 40 updates each; anchors: 80 / 80 at V=-0.4 and 80 / 62 at V=0.',fontsize=10)
    save(fig,'preliminary_phase_diagram')
    # Same coordinate order and axes throughout the supporting 3 x 3 grids.
    for key, title, ylabel, limits in (
        ('spin_rms','Spin-order histories from both seeds','Bulk leg-odd spin MF RMS [t]',(1e-9,.1)),
        ('pairing_rms','Pairing histories from both seeds','Bulk leg-pair MF RMS [t]',(1e-13,.01)),
        ('energy','Corrected canonical energy: full histories','Energy per site [t]',None),
        ('energy_late','Corrected canonical energy: final 15 evaluations','Energy per site [t]',None)):
        fig, axes = plt.subplots(3,3,figsize=(13,10),layout='constrained')
        for iy,v in enumerate(VS):
            for ix,t0 in enumerate(T0):
                ax=axes[iy,ix]
                for family in ('stripe','pairing'):
                    d=data[(t0,v,family)]
                    sl=slice(-15,None) if key=='energy_late' else slice(None)
                    y=d['energy'] if key.startswith('energy') else d[key]
                    ax.plot(d['history']['iteration'][sl],y[sl],color=common.COLORS[family],
                            ls='-' if family=='stripe' else '--',label=common.LABELS[family],lw=1.6)
                ax.set_title(f't0={t0:g}, V={v:g}',fontsize=11)
                if limits: ax.set_yscale('log'); ax.set_ylim(*limits)
                else: ax.ticklabel_format(axis='y',style='plain',useOffset=False)
                ax.grid(alpha=.2)
                if iy==2: ax.set_xlabel('MF evaluation')
                if ix==0: ax.set_ylabel(ylabel)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(title+'\nSquare grid; solid = stripe seed, dashed = pairing seed; no accepted endpoints',fontsize=14)
        save(fig,key+'_grid')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    accounting=common.rows(PROJECT/'output/project_budget/additional_node_hours_reconciliations.tsv')
    data={}
    for campaign in CAMPAIGNS:
        run=PROJECT/'output/phase1_gpu'/campaign
        manifest,jobs=common.rows(run/'manifest.tsv'),common.rows(run/'jobs.tsv')
        assert len(manifest)==(4 if 'anchors' in campaign else 14)
        for row in manifest:
            key=(float(row['t0']),float(row['V']),row['family'])
            assert key not in data
            data[key]=analyze_run(row,run,manifest,jobs,accounting)
            s=data[key]['summary']
            print(f"Loaded {key}: {s['iterations']} {s['phase_observation']} accepted={s['accepted']}",flush=True)
    assert set(data)=={(t,v,f) for t in T0 for v in VS for f in ('stripe','pairing')}
    points=[]
    for v in VS:
        for t0 in T0:
            a,b=(data[(t0,v,f)] for f in ('stripe','pairing'))
            sa,sb=a['summary'],b['summary']
            assert sa['fingerprints']==sb['fingerprints']
            observed={sa['phase_observation'],sb['phase_observation']}
            phase=next(iter(observed)) if len(observed)==1 else '?'
            if observed=={'S','S+P'} and sb['pairing_fractional_change_last5'] < -.2 and sb['spin_fractional_change_last5'] > .05:
                phase='S*'
            assert phase in ('S','D','S*','?')
            point=dict(t0=t0,V=v,phase=phase,seed_summary=('both seeds → '+phase) if phase!='S*' else 'pairing seed → stripe (ongoing)',
                stripe_iterations=sa['iterations'],pairing_iterations=sb['iterations'],
                accepted_count=int(sa['accepted'])+int(sb['accepted']),
                endpoint_energy_pairing_minus_stripe=sb['corrected_energy_per_site']-sa['corrected_energy_per_site'],
                physical_spin_rms_range=sorted([sa['physical_spin_odd_bulk_rms'],sb['physical_spin_odd_bulk_rms']]),
                physical_pair_rms_range=sorted([sa['physical_leg_pair_bulk_rms'],sb['physical_leg_pair_bulk_rms']]),
                physical_charge_max_between_seeds=np.max(np.abs(a['charge']-b['charge'])),
                maximum_global_relative_residual=max(sa['global_relative_residual'],sb['global_relative_residual']),
                energy_ranking_eligible=bool(sa['accepted'] and sb['accepted']))
            points.append(point)
    summaries=[d['summary'] for d in data.values()]
    payload=dict(date='2026-09-15',L=64,chi=200,points=points,runs=summaries,
        classification='Observed basins and direction of raw MF evolution, not accepted-energy selection. S/D thresholds are descriptive separators; see script.',
        total_iterations=sum(s['iterations'] for s in summaries),
        accepted_count=sum(s['accepted'] for s in summaries),
        total_solver_node_hours=sum(s['solver_node_hours'] for s in summaries),
        allocation_jobs_available=sum('allocation' in s for s in summaries),
        total_allocation_node_hours=sum(s.get('allocation',{}).get('node_hours',0) for s in summaries))
    (OUT/'analysis.json').write_text(json.dumps(common.clean(payload),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fields=('t0','V','family','job_id','phase_observation','iterations','status','accepted',
            'physical_spin_odd_bulk_rms','physical_leg_pair_bulk_rms','final_rung_pair_mean',
            'physical_charge_bulk_std','global_relative_residual','corrected_energy_per_site','energy_span_last10',
            'density_error','spin_fractional_change_last10','pairing_fractional_change_last10')
    write_csv('run_summary.csv',[{**{k:s[k] for k in fields},'failed_gates':'; '.join(s['failed_gates'])} for s in summaries])
    write_csv('sources.csv',[{k:s[k] for k in ('t0','V','family','job_id','campaign','source','source_sha256','full_source_sha256','config_sha256')} for s in summaries])
    write_csv('iteration_history.csv',[dict(t0=t,V=v,family=f,iteration=int(it),
        energy_per_site=d['energy'][i],spin_field_rms=d['spin_rms'][i],pairing_field_rms=d['pairing_rms'][i],
        global_relative_residual=d['history']['field_rel_residual'][i],density=d['history']['density'][i])
        for (t,v,f),d in data.items() for i,it in enumerate(d['history']['iteration'])])
    write_csv('terminal_profiles.csv',[dict(t0=t,V=v,family=f,rung=x+1,
        physical_charge=d['charge'][x],physical_spin_odd=d['spin'][x],
        physical_rung_pair=d['pair_rung'][x])
        for (t,v,f),d in data.items() for x in range(64)])
    figures(data,points)
    for s in summaries:
        assert common.sha(PROJECT/s['source'])==s['source_sha256']
    print(json.dumps(common.clean({k:v for k,v in payload.items() if k!='runs'}),indent=2))


if __name__=='__main__':
    main()
