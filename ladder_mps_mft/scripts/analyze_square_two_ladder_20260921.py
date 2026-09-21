"""Read-only first square A/B endpoint comparison; no DMRG or source mutation."""
import json
import re
import tomllib

import h5py
import numpy as np
import matplotlib.pyplot as plt

import analyze_two_basin_vm04 as old

PROJECT = old.PROJECT
RUN = PROJECT / 'output/phase1_gpu/20260920_square_two_ladder_two_basin_60'
OUT = PROJECT / 'docs/reports/square_two_ladder_20260920'
BULK, BONDS = slice(5, 59), slice(5, 58)


def text(f, key):
    return f[key][()].decode()


def profiles(c):
    up, down, pair = (c[k][()] for k in ('density_up', 'density_down', 'pair'))
    charge = (up + down).reshape(-1, 64, 2).mean(axis=2)
    spin = ((up - down)[..., ::2] - (up - down)[..., 1::2]) / 4
    i, j = np.arange(63)*2, np.arange(64)*2
    leg = (pair[..., i, i+2]+pair[..., i+2, i]+pair[..., i+1, i+3]+pair[..., i+3, i+1])/4
    rung = (pair[..., j, j+1]+pair[..., j+1, j])/2
    return dict(charge=charge, spin=spin, leg=leg, rung=rung)


def profile_metrics(p):
    return dict(spin_rms=np.sqrt(np.mean(p['spin'][..., BULK]**2, axis=-1)),
        pair_rms=np.sqrt(np.mean(p['leg'][..., BONDS]**2, axis=-1)),
        charge_std=np.std(p['charge'][..., BULK], axis=-1))


def main():
    manifest = old.rows(RUN/'manifest.tsv')
    row = next(r for r in manifest if r['family']=='pairing' and float(r['V'])==-.4)
    branch = RUN/'results'/row['label']
    paths = list(branch.rglob('state.h5'))
    assert len(paths)==1
    path = paths[0]
    receipts = {r['relative_path']:r for r in old.rows(branch/'stateless_manifest.tsv')}
    receipt = receipts[path.relative_to(branch).as_posix()]
    assert old.sha(path)==receipt['compact_sha256']
    config_path = RUN/'configs'/(row['label']+'.segment-001.toml')
    seed_path = RUN/'seeds'/(row['label']+'.h5')
    assert old.sha(config_path)==row['config_sha256']
    assert old.sha(seed_path)==row['seed_sha256']
    config = tomllib.loads(config_path.read_text()); cfg=config['convergence']
    assert config['mixing']['damping']==1 and config['dmrg']['maxdim']==200
    data, summaries, diagnostics = {}, {}, {}
    with h5py.File(path) as f, h5py.File(seed_path) as sf:
        assert text(f,'artifact_kind')=='spatial_cell_mps_mft_state'
        assert text(f,'model/transverse_geometry')=='square'
        assert int(f['physical_sites'][()])==256 and int(f['spatial_ladders'][()])==2
        for k in ('L','t','U','V','t0','tp','density','r_range'):
            assert f['model/'+k][()]==config['model'][k]
        for k in ('model_fingerprint','numerical_fingerprint','implementation_sha256','ep_source_sha256','config_sha256'):
            assert text(f,'provenance/'+k)==row[k]
        assert text(f,'provenance/inherit_sha256')==row['seed_sha256']
        assert text(f,'provenance/slurm_job_id')=='58654275'
        n=len(f['history/cell_sweep']); assert n==40
        energy=f['history/target_density_corrected_variational_energy_per_site'][()]
        canonical=f['history/canonical_variational_energy'][()]/256
        np.testing.assert_allclose(canonical,f['history/canonical_variational_energy_per_site'][()],atol=1e-15)
        for name in ('A','B'):
            g=f['ladders/'+name]; h=g['history']; e=h['energy']; diag=g['convergence']
            np.testing.assert_array_equal(h['iteration'][()],np.arange(1,n+1))
            assert set(h['update_mode'][()])=={b'unmixed_probe'}
            fields={s:{k:old._julia_array(h[f'fields/{s}/{k}']) for k in ('alpha','beta','mu_cdw')} for s in ('applied','measured')}
            for k in fields['applied']:
                np.testing.assert_array_equal(fields['applied'][k][...,1:],fields['measured'][k][...,:-1])
                np.testing.assert_array_equal(fields['applied'][k][...,0],old._julia_array(sf[f'ladders/{name}/fields/{k}']))
            vectors={s:old.channel_vectors(v) for s,v in fields.items()}
            channels={}
            for k,c in h['channels'].items():
                x,y=(vectors[s][k] for s in ('applied','measured'))
                np.testing.assert_allclose(np.max(np.abs(y-x),axis=1),c['absolute'][()],atol=1e-15)
                both=np.concatenate((x[-10:],y[-10:])); span=np.ptp(both,axis=0)
                a=np.max(np.abs(span)); r=np.linalg.norm(span)/max(np.max(np.linalg.norm(both,axis=1)),np.finfo(float).eps)
                np.testing.assert_allclose(a,c['window_absolute'][-1],atol=1e-15)
                channels[k]=dict(step_max_abs_last10=np.max(c['absolute'][-10:]),window_abs=a,window_rel=r,
                    steps_pass=bool(np.all(c['passes'][-10:])),window_pass=bool(a<=cfg['channel_noise_floor'] or r<=cfg['field_rel_tol']))
                assert channels[k]['window_pass']==bool(c['window_passes'][-1])
            p=profiles(h['correlations']); metrics=profile_metrics(p)
            for k in h['correlations']:
                np.testing.assert_array_equal(h['correlations/'+k][-1],g['correlations/'+k][()])
            np.testing.assert_allclose(p['charge'].mean(axis=1),h['density'][()],atol=1e-14,rtol=0)
            direct=sum(e[k][()] for k in ('bare_ladder_energy','pair_transverse_energy','exchange_transverse_energy','density_transverse_energy'))
            np.testing.assert_allclose(direct,e['canonical_variational_energy'][()],atol=1e-11,rtol=0)
            np.testing.assert_allclose(e['canonical_variational_energy'][()]+h['chemical_potential'][()]*(.9375-h['density'][()])*128,
                e['target_density_corrected_variational_energy'][()],atol=1e-11,rtol=0)
            dmrg=np.array([abs(np.diff(h[f'dmrg/{i:04d}/sweep_energy'][-2:])[0]) for i in range(1,n+1)])
            gates=dict(minimum=n>=cfg['minimum_iterations'],
                field=np.all((h['field_abs_residual'][-10:]<=cfg['field_abs_tol']) | (h['field_rel_residual'][-10:]<=cfg['field_rel_tol'])),
                slow=diag['fixed_point_extrapolated_abs_residual'][()]<=cfg['field_abs_tol'] or diag['fixed_point_extrapolated_rel_residual'][()]<=cfg['field_rel_tol'],
                density=np.max(np.abs(h['density'][-10:]-.9375))<=cfg['density_tol'],inner_dmrg=np.all(dmrg[-10:]<=cfg['dmrg_sweep_energy_tol']),
                energy=np.ptp(e['target_density_corrected_variational_energy'][-10:]/128)<=cfg['variational_energy_tol'],
                identity=diag['hamiltonian_identity_error_per_site'][()]<=cfg['hamiltonian_identity_tol'],
                effective=diag['effective_eigenvalue_error_per_site'][()]<=cfg['effective_energy_consistency_tol'])
            assert all(gates.values()) and all(c['window_pass'] and c['steps_pass'] for c in channels.values())
            s=dict(**{k:v[-1] for k,v in metrics.items()},leg_mean=p['leg'][-1,BONDS].mean(),rung_mean=p['rung'][-1,BULK].mean(),
                density=h['density'][-1],relative_residual=h['field_rel_residual'][-1],absolute_residual=h['field_abs_residual'][-1],
                spin_rms_first=metrics['spin_rms'][0],spin_rms_max=metrics['spin_rms'].max(),spin_rms_last10_max=metrics['spin_rms'][-10:].max(),
                energy_span_last10=np.ptp(e['target_density_corrected_variational_energy'][-10:]/128),
                canonical=e['canonical_variational_energy'][-1]/128,corrected=e['target_density_corrected_variational_energy'][-1]/128,
                gates=gates,channels=channels)
            data[name]=dict(p=p,metrics=metrics); summaries[name]=s
        np.testing.assert_allclose(energy,sum(f[f'ladders/{name}/history/energy/target_density_corrected_variational_energy'][()] for name in ('A','B'))/256,atol=1e-14)
        np.testing.assert_allclose(canonical,sum(f[f'ladders/{name}/history/energy/canonical_variational_energy'][()] for name in ('A','B'))/256,atol=1e-14)
        summary=dict(source=path.relative_to(PROJECT).as_posix(),compact_sha256=receipt['compact_sha256'],full_sha256=receipt['full_sha256'],
            job_id='58654275',status=text(f,'status'),accepted=bool(f['accepted'][()]),iterations=n,
            corrected=energy[-1],canonical=canonical[-1],energy_span_last10=np.ptp(energy[-10:]),solver_seconds=np.sum(f['history/wall_seconds'][()]),
            model_fingerprint=row['model_fingerprint'],config_sha256=row['config_sha256'],seed_sha256=row['seed_sha256'],ladders=summaries)
        assert summary['accepted'] and summary['status']=='fixed_point'
    for name in ('A','B'):
        dp=path.parent/f'diagnostics_ladder_{name}.h5'
        dr=receipts[dp.relative_to(branch).as_posix()]
        assert old.sha(dp)==dr['compact_sha256']
        with h5py.File(dp) as f:
            assert text(f,'state_sha256')==receipt['full_sha256'] and bool(f['measurement_complete'][()])
            assert bool(f['full_pair_correlations'][()]) and bool(f['accepted'][()])
            assert text(f,'spatial_ladder')==name and f['iteration'][()]==40
            np.testing.assert_allclose(f['density'][()].reshape(64,2).mean(axis=1),data[name]['p']['charge'][-1],atol=1e-12)
            for k,ex in (('charge',f['density'][()]),('spin',f['spin'][()])):
                np.testing.assert_allclose(f[k+'_correlation'][()]-np.outer(ex,ex),f[k+'_connected'][()],atol=1e-14)
            pc=f['pair_correlations']; anomalous=pc['expectation'][()]
            np.testing.assert_allclose(pc['removal'][()]-np.outer(anomalous,anomalous),pc['removal_connected'][()],atol=1e-14)
            diagnostics[name]=dict(source=dp.relative_to(PROJECT).as_posix(),sha256=dr['compact_sha256'],complete=True,
                seconds=f['measurement_wall_seconds'][()],pair_basis=len(anomalous),pair_channels=[x.decode() for x in np.unique(pc['basis_class'][()])])
    reference={fam:old.load(fam,old.rows(old.RUN/'manifest.tsv'),old.rows(old.RUN/'jobs.tsv')) for fam in ('pairing','stripe')}
    comparisons={}; old_plots={}
    for fam,d in reference.items():
        with h5py.File(PROJECT/d['summary']['source']) as f:
            p={k:np.asarray(v).reshape(-1) for k,v in profiles(f['correlations']).items()}
            assert text(f,'model/transverse_geometry')=='square'
            assert float(f['model/E_p'][()])==float(row['ep_denominator'])
            assert float(f['model/E_p_signed'][()])==float(row['ep_signed'])
            g=2*.1**2/float(f['model/E_p'][()])
            for k in ('L','t','U','V','t0','tp','density','r_range'):
                assert f['model/'+k][()]==config['model'][k]
            a=old._julia_array(f['history/fields/measured/alpha'])
            i=np.arange(63)
            leg_history=((a[i,i+1,0,0]+a[i+1,i,0,0]+a[i,i+1,1,1]+a[i+1,i,1,1])/4/g).T
            np.testing.assert_allclose(leg_history[-1],p['leg'],atol=1e-12)
            can=f['history/variational_energy'][()]/128
        met=profile_metrics(p)
        comparisons[fam]=dict(source=d['summary']['source'],sha256=d['summary']['source_sha256'],accepted=d['summary']['accepted'],
            iterations=d['summary']['iterations'],**met,leg_mean=np.mean(p['leg'][BONDS]),rung_mean=np.mean(p['rung'][BULK]),
            canonical=can[-1],corrected=d['summary']['corrected_energy_per_site'],
            new_minus_old_corrected=summary['corrected']-d['summary']['corrected_energy_per_site'],
            new_minus_old_canonical=summary['canonical']-can[-1],
            max_profile_difference={name:{k:np.max(np.abs(data[name]['p'][k][-1]-p[k])) for k in p} for name in ('A','B')})
        old_plots[fam]=dict(p=p,spin=d['spin_rms']/g,pair=np.sqrt(np.mean(leg_history[:,BONDS]**2,axis=1)),energy=d['history']['target_density_corrected_variational_energy']/128)
    ab={k:dict(max_final=np.max(np.abs(data['A']['p'][k][-1]-data['B']['p'][k][-1])),
        rms_first=np.sqrt(np.mean((data['A']['p'][k][0]-data['B']['p'][k][0])**2)),
        rms_final=np.sqrt(np.mean((data['A']['p'][k][-1]-data['B']['p'][k][-1])**2))) for k in data['A']['p']}
    stripe_log=RUN/'logs/square__two_ladder__stripe_eps005_t014_vm04_chi200_raw60.s1-58654274.out'
    logged=re.findall(r'^SPATIAL_CELL cell_sweep=(\d+) ladders=2 Evar_target/site=([\d.eE+-]+) status=(\w+)',stripe_log.read_text(),re.M)
    stripe=dict(source=stripe_log.relative_to(PROJECT).as_posix(),sha256=old.sha(stripe_log),iterations=int(logged[-1][0]),
        corrected=float(logged[-1][1]),status=logged[-1][2],state_available=False,
        limitation='Synced stdout reaches fixed_point and begins measurements; no synced terminal state or completed measurement evidence.')
    assert old.sha(path)==receipt['compact_sha256']
    result=dict(completed=summary,diagnostics=diagnostics,single_ladder=comparisons,AB_difference=ab,stripe_log_only=stripe,
        methods='Physical spin: leg-odd Sz, rungs 6-59; pair: symmetrized nearest-neighbor leg anomalous expectation, bonds 6-58. Cell energy /256; single ladder /128. Same model at A=B, distinct acceptance controls; no accepted-branch energy ranking.')
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'first_result_20260921.json').write_text(json.dumps(old.clean(result),indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(14,7.6),layout='constrained')
    styles={'A':('#16679e','-'),'B':('#db7b20','--')}
    for name,d in data.items():
        color,ls=styles[name]; label='Two-ladder '+name
        for ax,key in zip(axes[0,:2],('pair_rms','spin_rms')):
            ax.plot(np.arange(1,n+1),d['metrics'][key],color=color,ls=ls,label=label)
        axes[1,0].plot(np.arange(1,65),d['p']['charge'][-1],color=color,ls=ls,label=label)
        axes[1,1].plot(np.arange(1,64)+.5,d['p']['leg'][-1],color=color,ls=ls,label=label)
        axes[1,2].plot(np.arange(1,65),d['p']['spin'][-1],color=color,ls=ls,label=label)
    axes[0,2].plot(np.arange(1,n+1),energy-summary['corrected'],color='#16679e',label='Two-ladder cell')
    for fam,d in old_plots.items():
        color,ls=('#343434',':') if fam=='pairing' else ('#929292','-.')
        label='One-ladder '+fam+' seed'
        axes[0,0].plot(np.arange(1,len(d['pair'])+1),d['pair'],color=color,ls=ls,label=label)
        axes[0,1].plot(np.arange(1,len(d['spin'])+1),d['spin'],color=color,ls=ls,label=label)
        axes[0,2].plot(np.arange(1,len(d['energy'])+1),d['energy']-summary['corrected'],color=color,ls=ls,label=label)
        for ax,k in zip(axes[1],('charge','leg','spin')):
            ax.plot(np.arange(1,len(d['p'][k])+1)+(0.5 if k=='leg' else 0),d['p'][k],color=color,ls=ls,label=label)
    axes[0,1].set_yscale('log'); axes[0,2].set_yscale('symlog',linthresh=1e-7)
    for ax,title,ylabel in zip(axes.flat,
        ('Pairing survives','Seeded spin decays','Full energy histories','Endpoint density','Endpoint leg pairing','Endpoint residual spin'),
        ('Physical leg-pair RMS','Physical spin RMS','Corrected E/site − final A/B [t]','Electrons/site','Anomalous pair expectation','Leg-odd Sz')):
        ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(alpha=.2)
    for ax in axes[0]:ax.set_xlabel('MF evaluation / cell sweep')
    for ax in axes[1]:ax.set_xlabel('Rung position')
    axes[0,0].legend(fontsize=8);axes[0,2].legend(fontsize=8)
    axes[1,2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.suptitle('Square (t₀,V)=(1.4,−0.4), χ=200: first A/B result\nPairing seed accepted at 40 sweeps; single-ladder references stop at 80, unaccepted',fontsize=13)
    fig.savefig(OUT/'first_result_comparison_20260921.png',dpi=170);plt.close(fig)
    print(json.dumps(old.clean(result),indent=2))


if __name__=='__main__':
    main()
