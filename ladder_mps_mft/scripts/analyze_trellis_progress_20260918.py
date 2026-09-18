"""Read-only audit of the September 18 trellis sync, including partial stdout.

Trellis Hartree fields mix densities and normal bonds. Physical profiles must
come from stored correlations, never from the square/cubic field inversion.
"""
import json
import re
import tomllib

import h5py
import numpy as np
import matplotlib.pyplot as plt

import analyze_two_basin_campaigns_20260918 as report

base = report.base
PROJECT = base.PROJECT
RUN = base.ROOT/'20260916_trellis_two_basin_comparison_60'
OUT = PROJECT/'docs/reports/trellis_progress_20260918'


def load(row, job, accounting):
    paths = list((RUN/'results'/row['label']).rglob('state.h5'))
    assert len(paths) == 1
    path = paths[0]; compact = base.compact_record(path)
    cfgpath = RUN/'configs'/(row['label']+'.segment-001.toml')
    assert base.common.sha(cfgpath) == row['config_sha256']
    config = tomllib.loads(cfgpath.read_text()); cfg = config['convergence']
    assert config['dmrg']['maxdim'] == 200
    assert config['mixing']['damping'] == 1 and config['mixing']['method'] == 'linear'
    seed = RUN/'seeds'/(row['label']+'.h5')
    assert base.common.sha(seed) == row['seed_sha256']
    with h5py.File(path) as f, h5py.File(seed) as sf:
        assert base.text(f,'artifact_kind') == 'trellis_mps_mft_state'
        assert base.text(f,'model/transverse_geometry') == 'trellis'
        assert base.text(f,'model/trellis_cell') == row['trellis_cell']
        for key in ('U','t0','V','L','density','tau0','tau1'):
            assert f['model/'+key][()] == float(row[key])
        fingerprints = {k:base.text(f,'provenance/'+k) for k in
            ('model_fingerprint','numerical_fingerprint','implementation_sha256','ep_source_sha256')}
        assert all(v == row[k] for k,v in fingerprints.items())
        assert base.text(f,'provenance/config_sha256') == row['config_sha256']
        assert base.text(f,'provenance/inherit_sha256') == row['seed_sha256']
        assert base.text(f,'provenance/slurm_job_id') == job['job_id']
        assert base.text(f,'provenance/ep_mode') == 'exact'
        assert f['provenance/ep_signed'][()] == float(row['ep_signed'])
        n = len(f['history/cell_sweep']); sites = int(f['physical_sites'][()])
        np.testing.assert_array_equal(f['history/cell_sweep'][()],np.arange(1,n+1))
        assert sites == 128*int(row['spatial_ladders'])
        energy = f['history/target_density_corrected_variational_energy_per_site'][()]
        assert np.isfinite(energy).all()
        canonical = f['history/canonical_variational_energy'][()]
        corrected = f['history/target_density_corrected_variational_energy'][()]
        np.testing.assert_allclose(energy,corrected/sites,rtol=0,atol=1e-14)
        ladders = []; energies = []; corrected_energies = []
        for name,g in f['ladders'].items():
            h = g['history']; diag = g['convergence']; e = h['energy']; tail = slice(-cfg['stable_iterations'],None)
            np.testing.assert_array_equal(h['iteration'][()],np.arange(1,n+1))
            assert set(h['update_mode'][()]) == {b'unmixed_probe'}
            fields = {s:{k:base.common._julia_array(h[f'fields/{s}/{k}'])
                for k in ('alpha','beta','mu_cdw')} for s in ('applied','measured')}
            for k in fields['applied']:
                np.testing.assert_array_equal(fields['applied'][k][...,1:],fields['measured'][k][...,:-1])
                np.testing.assert_array_equal(fields['applied'][k][...,0],base.common._julia_array(sf[f'ladders/{name}/fields/{k}']))
            vectors = {s:base.common.channel_vectors(v) for s,v in fields.items()}
            channels = {k:{q:v[()] for q,v in c.items()} for k,c in h['channels'].items()}
            for channel,c in channels.items():
                x,y = (vectors[s][channel] for s in ('applied','measured'))
                np.testing.assert_allclose(np.max(np.abs(y-x),axis=1),c['absolute'],atol=1e-15)
                both = np.concatenate((x[tail],y[tail])); span = np.ptp(both,axis=0)
                absolute = np.max(np.abs(span)); relative = np.linalg.norm(span)/max(np.max(np.linalg.norm(both,axis=1)),np.finfo(float).eps)
                np.testing.assert_allclose(absolute,c['window_absolute'][-1],atol=1e-15)
                np.testing.assert_allclose(relative,c['window_relative'][-1],atol=1e-12)
                assert bool(c['window_passes'][-1]) == bool(absolute <= cfg['channel_noise_floor'] or relative <= cfg['field_rel_tol'])
            c = h['correlations']; up,down = c['density_up'][()],c['density_down'][()]
            pair = c['pair'][()]  # HDF5 is time-first; symmetric averages are transpose-invariant.
            charge = (up+down).reshape(n,64,2).mean(axis=2)
            spin = ((up-down)[:,::2]-(up-down)[:,1::2])/4
            ix = np.arange(63)*2; j = np.arange(64)*2
            leg = (pair[:,ix,ix+2]+pair[:,ix+2,ix]+pair[:,ix+1,ix+3]+pair[:,ix+3,ix+1])/4
            rung = (pair[:,j,j+1]+pair[:,j+1,j])/2
            np.testing.assert_allclose(charge.mean(axis=1),h['density'][()],atol=1e-14,rtol=0)
            for k in c:
                np.testing.assert_array_equal(c[k][-1],g['correlations/'+k][()])
            can = e['canonical_variational_energy'][()]; target = e['target_density_corrected_variational_energy'][()]
            np.testing.assert_allclose(can+h['chemical_potential'][()]*(.9375-h['density'][()])*128,target,atol=1e-11,rtol=0)
            direct = sum(e[k][()] for k in ('bare_ladder_energy','pair_transverse_energy','exchange_transverse_energy','density_transverse_energy'))
            np.testing.assert_allclose(direct,can,atol=1e-10,rtol=0)
            energies.append(can); corrected_energies.append(target)
            gaps = np.array([abs(np.diff(h[f'dmrg/{i:04d}/sweep_energy'][-2:])[0]) for i in range(1,n+1)])
            sr = np.sqrt(np.mean(spin[:,5:59]**2,axis=1)); pr = np.sqrt(np.mean(leg[:,5:58]**2,axis=1))
            gates = dict(minimum=n>=cfg['minimum_iterations'],
                global_field=np.all((h['field_abs_residual'][tail]<=cfg['field_abs_tol']) | (h['field_rel_residual'][tail]<=cfg['field_rel_tol'])),
                global_slow=diag['fixed_point_extrapolated_abs_residual'][()]<=cfg['field_abs_tol'] or diag['fixed_point_extrapolated_rel_residual'][()]<=cfg['field_rel_tol'],
                density=np.all(np.abs(h['density'][tail]-.9375)<=cfg['density_tol']),
                inner_dmrg=np.all(gaps[tail]<=cfg['dmrg_sweep_energy_tol']),
                energy=np.ptp(target[tail]/128)<=cfg['variational_energy_tol'],
                identity=diag['hamiltonian_identity_error_per_site'][()]<=cfg['hamiltonian_identity_tol'],
                effective=diag['effective_eigenvalue_error_per_site'][()]<=cfg['effective_energy_consistency_tol'])
            for k,c in channels.items():
                gates[k+'_steps'] = np.all(c['passes'][tail]); gates[k+'_span'] = bool(c['window_passes'][-1])
            summary = dict(ladder=name,spin_rms_first=sr[0],spin_rms_final=sr[-1],pair_rms_first=pr[0],pair_rms_final=pr[-1],
                leg_pair_mean=np.mean(leg[-1,5:58]),rung_pair_mean=np.mean(rung[-1,5:59]),
                charge_std_final=np.std(charge[-1,5:59]),central_charge_std=np.std(charge[-1,16:48]),
                central_spin_rms=np.sqrt(np.mean(spin[-1,16:48]**2)),
                spin_fractional_change_last10=sr[-1]/sr[-10]-1,pair_fractional_change_last10=pr[-1]/pr[-10]-1,
                profile_max_change_last10={k:np.max(np.abs(v[-1]-v[-10])) for k,v in [('spin',spin),('charge',charge),('leg_pair',leg)]},
                global_relative=h['field_rel_residual'][-1],last10_max_global_relative=np.max(h['field_rel_residual'][tail]),
                final_density_error=abs(h['density'][-1]-.9375),last10_max_density_error=np.max(np.abs(h['density'][tail]-.9375)),
                energy_span_last10=np.ptp(target[tail]/128),
                dmrg_last10_failing_iterations=[i+1 for i in range(n-10,n) if gaps[i]>cfg['dmrg_sweep_energy_tol']],
                final_dmrg_sweep_delta=gaps[-1],final_dmrg_discarded_weight=h[f'dmrg/{n:04d}/sweep_max_discarded_weight'][-1],
                gates=gates,failed_gates=[k for k,v in gates.items() if not v],
                channels_final={k:{q:v[-1] for q,v in c.items()} for k,c in channels.items()})
            ladders.append(dict(summary=summary,spin=spin,charge=charge,leg=leg,rung=rung,spin_rms=sr,pair_rms=pr,
                density=h['density'][()],dmrg_gap=gaps,global_relative=h['field_rel_residual'][()]))
        np.testing.assert_allclose(np.sum(energies,axis=0),canonical,atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.sum(corrected_energies,axis=0),corrected,atol=1e-12,rtol=0)
        allocation = [a for a in accounting if a['job_id']==job['job_id']]
        assert not allocation  # No reconciled cost in this dated local snapshot.
        summary = dict(campaign=RUN.name,label=row['label'],job_id=job['job_id'],cell=row['trellis_cell'],family=row['family'],
            source=path.relative_to(PROJECT).as_posix(),compact_sha256=compact['compact_sha256'],full_sha256=compact['full_sha256'],
            config_sha256=row['config_sha256'],seed_sha256=row['seed_sha256'],fingerprints=fingerprints,controls=cfg,
            status=base.text(f,'status'),accepted=bool(f['accepted'][()]),cell_sweeps=n,ladder_solves=n*len(ladders),
            physical_sites=sites,energy_per_site_final=energy[-1],energy_span_last10=np.ptp(energy[-10:]),
            solver_seconds=np.sum(f['history/wall_seconds'][()]),solver_node_hours=np.sum(f['history/wall_seconds'][()])/14400,
            allocation=None,reserved_node_hours=float(job['reserved_node_hours']),energy_ranking_eligible=False,
            ladders=[l['summary'] for l in ladders])
    logpath = RUN/'logs'/f"{row['label']}.s1-{job['job_id']}.out"
    text = logpath.read_text(); steps = re.findall(r'^TRELLIS cell_sweep=(\d+)',text,re.M)
    assert list(map(int,steps)) == list(range(1,n+1))
    summary.update(log_source=logpath.relative_to(PROJECT).as_posix(),log_sha256=base.common.sha(logpath))
    return dict(summary=summary,energy=energy,ladders=ladders)


def partial(row,job):
    p = RUN/'logs'/f"{row['label']}.s1-{job['job_id']}.out"
    text = p.read_text() if p.exists() else ''
    rows = re.findall(r'^TRELLIS cell_sweep=(\d+) ladders=(\d+) Evar_target/site=([\d.eE+-]+) status=(\w+)',text,re.M)
    records = [dict(cell_sweep=int(i),spatial_ladders=int(n),energy_per_site=float(e),status=s) for i,n,e,s in rows]
    return dict(label=row['label'],cell=row['trellis_cell'],family=row['family'],job_id=job['job_id'],
        source=p.relative_to(PROJECT).as_posix(),sha256=base.common.sha(p) if p.exists() else None,
        state_available=False,records=records,limitation='Synced stdout only; no physical profiles or terminal acceptance/cost evidence.')


def figures(d):
    x = np.arange(1,len(d['energy'])+1); l = d['ladders'][0]
    fig,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axes[0,0].plot(x,d['energy'],color='#6D4E9B');axes[0,0].set(title='Energy from the first evaluation',ylabel='Corrected energy per site [t]')
    axes[0,1].plot(x[-20:],d['energy'][-20:],color='#6D4E9B');axes[0,1].set(title='Late energy plateau',ylabel='Corrected energy per site [t]')
    axes[1,0].plot(x,l['spin_rms'],label='Spin RMS',color='#245A9C')
    axes[1,0].plot(x,l['pair_rms'],label='Leg-pair RMS',color='#B65F16');axes[1,0].set(yscale='log',ylabel='Bulk physical order',title='Stripe seed evolves toward pairing');axes[1,0].legend()
    axes[1,1].plot(x,l['global_relative'],color='#6D4E9B',label='Global relative residual');axes[1,1].axhline(1e-4,color='0.5',ls=':',label='Relative tolerance')
    axes[1,1].set(yscale='log',ylabel='Raw-map relative residual',title='Global residual passes late; other gates fail');axes[1,1].legend(fontsize=9)
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlabel('Cell sweep / MF evaluation')
    for ax in axes[0]:ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.suptitle('Trellis one-ladder stripe seed: paired endpoint after 60 raw sweeps',fontsize=15)
    fig.supxlabel('t0=1, V=0, tau0=tau1=0.1, L64, chi200. Stored status: maximum_iterations; accepted=false.',fontsize=10)
    report.save(fig,OUT,'histories')
    fig,axes = plt.subplots(3,1,figsize=(9,10),layout='constrained')
    for i in (0,9,59):
        alpha=1 if i==59 else .45
        axes[0].plot(np.arange(1,65),l['charge'][i],label=f'Evaluation {i+1}',alpha=alpha)
        axes[1].plot(np.arange(1,65),l['spin'][i]*(-1.)**np.arange(64),alpha=alpha,label=f'Evaluation {i+1}')
    axes[2].plot(np.arange(1,64)+.5,l['leg'][-1],color='#B65F16',label='Final leg bonds')
    axes[2].plot(np.arange(1,65),l['rung'][-1],color='#245A9C',label='Final rung bonds')
    axes[0].set_ylabel('Electrons per site');axes[1].set_ylabel('Staggered leg-odd spin');axes[2].set_ylabel('Physical pairing')
    axes[0].legend(ncol=3);axes[2].legend();axes[2].set_xlabel('Rung / bond midpoint')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Trellis: spin texture disappears, opposite leg/rung pairing survives',fontsize=14)
    fig.supxlabel('Physical correlations read directly from HDF5. Residual end-dependent charge oscillations do not establish bulk CDW order.\nFinal spin RMS is 1.21e-5, compared with 6.27e-2 at evaluation 1. No accepted endpoint.',fontsize=9)
    report.save(fig,OUT,'profiles')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest = base.common.rows(RUN/'manifest.tsv'); jobs = base.common.rows(RUN/'jobs.tsv')
    accounting = base.common.rows(PROJECT/'output/project_budget/additional_node_hours_reconciliations.tsv')
    data = []; missing = []
    for row in manifest:
        job = next(j for j in jobs if j['label']==row['label'])
        if list((RUN/'results'/row['label']).rglob('state.h5')):data.append(load(row,job,accounting))
        else:missing.append(partial(row,job))
    assert len(data)==1 and data[0]['summary']['cell']=='one_ladder' and data[0]['summary']['family']=='stripe'
    payload = dict(date='2026-09-18',runs=[d['summary'] for d in data],missing=missing,
        total_complete_cell_sweeps=sum(d['summary']['cell_sweeps'] for d in data),
        total_complete_ladder_solves=sum(d['summary']['ladder_solves'] for d in data),
        total_solver_node_hours=sum(d['summary']['solver_node_hours'] for d in data),accepted_count=0,
        interpretation='One completed pairing-dominated trajectory. No accepted within-cell or between-cell energetic comparison.')
    (OUT/'analysis.json').write_text(json.dumps(base.common.clean(payload),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report.write_csv(OUT/'sources.csv',[{k:d['summary'][k] for k in ('label','job_id','source','compact_sha256','full_sha256','config_sha256','seed_sha256')} for d in data])
    report.write_csv(OUT/'iteration_history.csv',[dict(ladder=l['summary']['ladder'],cell_sweep=i+1,energy_per_site=d['energy'][i],
        spin_rms=l['spin_rms'][i],pair_rms=l['pair_rms'][i],charge_std=np.std(l['charge'][i,5:59]),density=l['density'][i],
        global_relative=l['global_relative'][i],dmrg_last_sweep_delta=l['dmrg_gap'][i])
        for d in data for l in d['ladders'] for i in range(d['summary']['cell_sweeps'])])
    report.write_csv(OUT/'terminal_profiles.csv',[dict(ladder=l['summary']['ladder'],rung=i+1,charge=l['charge'][-1,i],spin_odd=l['spin'][-1,i],
        rung_pair=l['rung'][-1,i],leg_pair_right=l['leg'][-1,i] if i<63 else None) for d in data for l in d['ladders'] for i in range(64)])
    report.write_csv(OUT/'partial_log_histories.csv',[dict(label=m['label'],job_id=m['job_id'],**r) for m in missing for r in m['records']])
    for d in data:
        figures(d)
        assert base.common.sha(PROJECT/d['summary']['source'])==d['summary']['compact_sha256']
    print(json.dumps(base.common.clean(payload['runs'][0]),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
