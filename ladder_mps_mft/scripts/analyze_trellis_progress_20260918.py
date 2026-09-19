"""Read-only audit of all four trellis endpoints synced September 19.

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
        records = [a for a in accounting if a['job_id']==job['job_id']]
        allocation = None
        if records:
            ac = records[-1]
            assert ac['campaign'] == RUN.name and ac['label'] == row['label'] and ac['sacct_state'] == 'COMPLETED'
            cost = int(ac['elapsed_raw_seconds'])/3600*float(ac['effective_node_fraction'])
            np.testing.assert_allclose(cost,float(ac['measured_node_hours']),atol=1e-9)
            allocation = dict(seconds=int(ac['elapsed_raw_seconds']),node_hours=cost,
                reconciled_utc=ac['reconciled_utc'],source='output/project_budget/additional_node_hours_reconciliations.tsv')
        summary = dict(campaign=RUN.name,label=row['label'],job_id=job['job_id'],cell=row['trellis_cell'],family=row['family'],
            source=path.relative_to(PROJECT).as_posix(),compact_sha256=compact['compact_sha256'],full_sha256=compact['full_sha256'],
            config_sha256=row['config_sha256'],seed_sha256=row['seed_sha256'],fingerprints=fingerprints,controls=cfg,
            status=base.text(f,'status'),accepted=bool(f['accepted'][()]),cell_sweeps=n,ladder_solves=n*len(ladders),
            physical_sites=sites,energy_per_site_final=energy[-1],energy_span_last10=np.ptp(energy[-10:]),
            solver_seconds=np.sum(f['history/wall_seconds'][()]),solver_node_hours=np.sum(f['history/wall_seconds'][()])/14400,
            allocation=allocation,reserved_node_hours=float(job['reserved_node_hours']),energy_ranking_eligible=False,
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




def comparisons(data):
    result = []
    for cell in ('one_ladder','two_ladder'):
        a,b = [next(d for d in data if d['summary']['cell']==cell and d['summary']['family']==fam)
               for fam in ('stripe','pairing')]
        assert a['summary']['fingerprints'] == b['summary']['fingerprints']
        assert a['summary']['controls'] == b['summary']['controls']
        # Only one global spin reversal and pairing gauge per entire spatial cell.
        spin_sign = np.sign(sum(np.dot(x['spin'][-1],y['spin'][-1]) for x,y in zip(a['ladders'],b['ladders']))) or 1
        pair_sign = np.sign(sum(np.dot(x['leg'][-1],y['leg'][-1]) for x,y in zip(a['ladders'],b['ladders']))) or 1
        result.append(dict(cell=cell,endpoint_pairing_minus_stripe=b['energy'][-1]-a['energy'][-1],
            sum_last10_energy_ranges=a['summary']['energy_span_last10']+b['summary']['energy_span_last10'],
            spin_alignment_sign=spin_sign,pair_alignment_sign=pair_sign,
            profile_max_seed_difference={key:max(np.max(np.abs(x[key][-1]-sign*y[key][-1]))
                for x,y in zip(a['ladders'],b['ladders']))
                for key,sign in [('charge',1),('spin',spin_sign),('leg',pair_sign),('rung',pair_sign)]},
            energy_ranking_eligible=bool(a['summary']['accepted'] and b['summary']['accepted'])))
    return result


def all_figures(data):
    fig,axes = plt.subplots(3,2,figsize=(12,9),layout='constrained')
    for col,cell in enumerate(('one_ladder','two_ladder')):
        for d in data:
            s=d['summary']
            if s['cell']!=cell:continue
            x=np.arange(1,s['cell_sweeps']+1)
            axes[0,col].plot(x,d['energy'],**report.style(s['family']))
            for l in d['ladders']:
                opts=report.style(s['family']);opts['label']+=f", {l['summary']['ladder']}"
                if l['summary']['ladder']=='B':opts.update(ls=':',lw=2.2)
                axes[1,col].plot(x,l['spin_rms'],**opts)
                axes[2,col].plot(x,l['pair_rms'],**opts)
        axes[0,col].set_title('Skew one-ladder cell' if col==0 else 'Rectangular two-ladder cell (A/B)')
        axes[0,col].ticklabel_format(axis='y',style='plain',useOffset=False)
        axes[0,col].legend(fontsize=8)
        for row in (1,2):axes[row,col].set_yscale('log');axes[row,col].legend(fontsize=8)
        axes[2,col].set_xlabel('Cell sweep')
    for ax in axes.flat:ax.grid(alpha=.2)
    for ax,label in zip(axes[:,0],('Corrected energy per site [t]','Physical spin RMS','Physical leg-pair RMS')):ax.set_ylabel(label)
    fig.suptitle('Trellis (t0,V)=(1,0): all four completed raw-update histories',fontsize=14)
    fig.supxlabel('L=64 per ladder, chi=200, tau0=tau1=0.1. A/B are spatial ladders; energy is normalized by 128 or 256 sites.',fontsize=9)
    report.save(fig,OUT,'histories')
    ordered=sorted(data,key=lambda d:(d['summary']['cell'],d['summary']['family']!='stripe'))
    fig,axes=plt.subplots(3,4,figsize=(16,8),layout='constrained')
    for col,d in enumerate(ordered):
        s=d['summary']; n=s['cell_sweeps']
        for l in d['ladders']:
            name=l['summary']['ladder'];offset=0 if name=='A' else -.5
            x=np.arange(1,65)+offset;color='#245A9C' if name=='A' else '#B65F16'
            for row,key in enumerate(('charge','spin','leg')):
                xx=x[:-1]+.5 if key=='leg' else x
                multiplier=(-1.)**np.arange(64) if key=='spin' else 1
                axes[row,col].plot(xx,l[key][-1]*multiplier,color=color,label=f'{name}, sweep {n}')
                axes[row,col].plot(xx,l[key][-10]*multiplier,color=color,alpha=.3,lw=1)
            axes[2,col].plot(x,l['rung'][-1],color=color,ls=':',label=f'{name}, rung')
        axes[0,col].set_title(('One ladder' if s['cell']=='one_ladder' else 'Two ladders')+' / '+s['family']+' seed')
        axes[0,col].legend(fontsize=8);axes[2,col].legend(fontsize=8,ncol=2)
        axes[2,col].set_xlabel('Physical rung / bond position')
    for ax in axes.flat:ax.grid(alpha=.2)
    for ax,label in zip(axes[:,0],('Electrons per site','Staggered leg-odd spin','Anomalous pairing')):ax.set_ylabel(label)
    fig.suptitle('Trellis endpoint profiles from stored physical correlations',fontsize=14)
    fig.supxlabel('Solid: final; faint: nine sweeps earlier; dotted: final rung pairing. B is shifted by -1/2 rung. Raw signs retained.',fontsize=10)
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
    assert len(data)==4 and not missing
    accounted=[d['summary']['allocation'] for d in data if d['summary']['allocation']]
    payload = dict(date='2026-09-19',runs=[d['summary'] for d in data],missing=missing,within_cell_comparisons=comparisons(data),
        total_complete_cell_sweeps=sum(d['summary']['cell_sweeps'] for d in data),
        total_complete_ladder_solves=sum(d['summary']['ladder_solves'] for d in data),
        total_solver_node_hours=sum(d['summary']['solver_node_hours'] for d in data),accepted_count=sum(d['summary']['accepted'] for d in data),
        allocation_jobs_available=len(accounted),known_allocation_node_hours=sum(a['node_hours'] for a in accounted),
        interpretation='Completed endpoint observations; within-cell seed differences remain diagnostic unless both endpoints pass. Different spatial cells have different finite-OBC embeddings.')
    (OUT/'analysis.json').write_text(json.dumps(base.common.clean(payload),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report.write_csv(OUT/'sources.csv',[{k:d['summary'][k] for k in ('label','job_id','source','compact_sha256','full_sha256','config_sha256','seed_sha256')} for d in data])
    report.write_csv(OUT/'iteration_history.csv',[dict(label=d['summary']['label'],cell=d['summary']['cell'],family=d['summary']['family'],ladder=l['summary']['ladder'],cell_sweep=i+1,energy_per_site=d['energy'][i],
        spin_rms=l['spin_rms'][i],pair_rms=l['pair_rms'][i],charge_std=np.std(l['charge'][i,5:59]),density=l['density'][i],
        global_relative=l['global_relative'][i],dmrg_last_sweep_delta=l['dmrg_gap'][i])
        for d in data for l in d['ladders'] for i in range(d['summary']['cell_sweeps'])])
    report.write_csv(OUT/'terminal_profiles.csv',[dict(label=d['summary']['label'],cell=d['summary']['cell'],family=d['summary']['family'],ladder=l['summary']['ladder'],rung=i+1,physical_position=i+1-(.5 if l['summary']['ladder']=='B' else 0),charge=l['charge'][-1,i],spin_odd=l['spin'][-1,i],
        rung_pair=l['rung'][-1,i],leg_pair_right=l['leg'][-1,i] if i<63 else None) for d in data for l in d['ladders'] for i in range(64)])
    # Preserve the old log-only CSV as a dated historical artifact; JSON missing=[] is authoritative.
    all_figures(data)
    for d in data:
        assert base.common.sha(PROJECT/d['summary']['source'])==d['summary']['compact_sha256']
    print(json.dumps(base.common.clean({k:v for k,v in payload.items() if k not in ('runs','missing')}),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
