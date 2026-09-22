"""Read-only correlation survey and completed square A/B analysis; no new DMRG.

Run with Python + numpy/scipy/h5py/matplotlib. Source outputs remain immutable.
All two-dimensional HDF5 arrays are transposed back to Julia index order.
"""
from pathlib import Path
import csv
import json
import tomllib
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import h5py
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_two_basin_vm04 as old
from analyze_square_two_ladder_20260921 import profiles, profile_metrics

P = old.PROJECT
OUT = P/'docs/reports/pair_correlations_20260922'
AB_RUN = P/'output/phase1_gpu/20260920_square_two_ladder_two_basin_60'
RETRY = P/'output/phase1_diagnostics'
REFS = (16, 24, 32)  # One-based rung coordinates; both directions are retained.
WINDOWS = ((4, 12), (8, 20), (12, 28))
COLORS = {'bare':'#343434', 'paired':'#17699b', 'stripe':'#be6427', 'cubic':'#8875a9', 'trellis':'#548348'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'savefig.facecolor':'white','pdf.fonttype':42})


def scalar(g, k):
    value = g[k][()]
    return value.decode() if isinstance(value, bytes) else value.item() if isinstance(value, np.generic) else value


def matrix(g, k):
    return g[k][()].T


def local(path):
    return P / str(path).replace('\\','/').split('/ladder_mps_mft/')[-1]


def dump(name, value):
    (OUT/name).write_text(json.dumps(old.clean(value), indent=2)+'\n', encoding='utf-8')


def csvout(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(old.clean(rows))


def read_diagnostic(path, state_hash, fingerprint, expected_status, accepted):
    with h5py.File(path) as f:
        assert scalar(f,'state_sha256') == state_hash
        assert scalar(f,'model_fingerprint') == fingerprint
        assert scalar(f,'status') == expected_status and bool(scalar(f,'accepted')) == accepted
        assert scalar(f,'measurement_complete') and scalar(f,'full_pair_correlations')
        assert scalar(f,'measurement_version') == 'equal_time_v2'
        pc=f['pair_correlations']; fpair=pc['expectation'][()]
        add, rem, conn= [matrix(pc,k) for k in ('addition','removal','removal_connected')]
        ac=matrix(pc,'addition_connected')
        err=float(np.max(np.abs(rem-np.outer(fpair.conj(),fpair)-conn)))
        np.testing.assert_allclose(rem-np.outer(fpair.conj(),fpair),conn,atol=2e-13,rtol=0)
        np.testing.assert_allclose(add-np.outer(fpair,fpair.conj()),ac,atol=2e-13,rtol=0)
        minimum=[]
        for a in (rem,conn,add,ac):
            assert np.isfinite(a).all()
            np.testing.assert_allclose(a,a.T.conj(),atol=2e-12,rtol=0)
            minimum.append(float(np.linalg.eigvalsh(a)[0]))
        assert min(minimum)>-1e-9, (path,minimum)
        labels=np.array([v.decode() if isinstance(v,bytes) else v for v in pc['basis_class'][()]])
        site1,site2=pc['basis_site1'][()],pc['basis_site2'][()]
        assert len(labels)==318 and scalar(f,'L')==64
        indices={k:np.where(labels==k)[0] for k in ('rung','leg0','leg1','onsite0','onsite1')}
        for k,ind in indices.items():
            np.testing.assert_array_equal((site1[ind]+1)//2,np.arange(1,len(ind)+1))
        for k in ('rung','leg0','leg1'):
            np.testing.assert_allclose(rem[np.ix_(indices[k],indices[k])].real,matrix(pc,k+'_removal'),atol=2e-12)
        density,spin=f['density'][()],f['spin'][()]
        np.testing.assert_allclose(matrix(f,'charge_correlation')-np.outer(density,density),matrix(f,'charge_connected'),atol=2e-12)
        np.testing.assert_allclose(matrix(f,'spin_correlation')-np.outer(spin,spin),matrix(f,'spin_connected'),atol=2e-12)
        return dict(path=path, sha256=old.sha(path),source_sha256=state_hash,
            geometry=scalar(f,'geometry'),t0=scalar(f,'t0'),V=scalar(f,'V'),
            status=expected_status,accepted=accepted,iteration=scalar(f,'iteration'),
            ladder=scalar(f,'spatial_ladder'),measurement_seconds=scalar(f,'measurement_wall_seconds'),
            implementation=scalar(f,'measurement_implementation_sha256'),chi=200,
            density=density.reshape(64,2).mean(axis=1),
            spin=(spin[::2]-spin[1::2])/2,
            raw=rem,connected=conn,expectation=fpair,indices=indices,
            max_imaginary=float(max(np.max(np.abs(a.imag)) for a in (rem,conn,add))),
            subtraction_error=err,minimum_gram_eigenvalue=min(minimum))


def backfill():
    parent=RETRY/'20260921_latest_correlations_retry1'
    retry=RETRY/'20260921_latest_correlations_retry2'
    original=old.rows(parent/'manifest.tsv'); subset=old.rows(retry/'manifest.tsv')
    mapping=old.rows(retry/'retry_rows.tsv')
    assert len(original)==56 and len(subset)==7
    for row, lineage in zip(subset,mapping):
        expected=dict(original[int(lineage['parent_index'])-1],index=row['index'])
        assert row==expected
    for run in (parent,retry):
        assert old.sha(run/'manifest.tsv')==(run/'manifest.sha256').read_text().split()[0]
    samples=[]; inventory=[]
    for row in original:
        sources=[run/'results'/row['campaign']/row['label'] for run in (parent,retry)]
        ready=[d for d in sources if (d/'measurement_receipt.toml').is_file()]
        assert len(ready)==1, ready
        directory=ready[0]; receipt=tomllib.loads((directory/'measurement_receipt.toml').read_text())
        assert receipt['source_sha256']==row['source_sha256'] and receipt['source_path']==row['source_path']
        assert len(receipt['files'])==int(row['samples'])==len(receipt['sha256'])
        state=local(row['compact_path']); config=local(row['config_path'])
        assert old.sha(state)==row['compact_sha256'] and old.sha(config)==row['config_sha256']
        with h5py.File(state) as f:
            assert scalar(f,'provenance/model_fingerprint')==row['model_fingerprint']
            assert scalar(f,'analysis_storage/full_artifact_sha256')==row['source_sha256']
            assert scalar(f,'status')==row['status']
            for name,hashvalue in zip(receipt['files'],receipt['sha256']):
                path=directory/name; assert old.sha(path)==hashvalue
                s=read_diagnostic(path,row['source_sha256'],row['model_fingerprint'],row['status'],row['accepted']=='true')
                s.update(id=f"backfill_{row['index']}{s['ladder']}", campaign=row['campaign'],label=row['label'],
                    parent_index=int(row['index']),source=state,category='backfill')
                g=f['ladders/'+s['ladder']] if s['ladder'] else f
                c=g['correlations']; up,down=c['density_up'][()],c['density_down'][()]
                np.testing.assert_allclose((up+down).reshape(64,2).mean(axis=1),s['density'],atol=2e-11,rtol=0)
                np.testing.assert_allclose(((up-down)[::2]-(up-down)[1::2])/4,s['spin'],atol=2e-11,rtol=0)
                pair=matrix(c,'pair'); ii=s['indices']['rung']
                np.testing.assert_allclose(s['expectation'][ii],np.diag(pair,1)[::2]+np.diag(pair,-1)[::2],atol=2e-11,rtol=0)
                s['residual']=float(g['history/field_abs_residual'][-1])
                samples.append(s)
        inventory.append(dict(parent_index=row['index'],campaign=row['campaign'],label=row['label'],
            result_run=directory.parts[-4],mps=int(row['samples']),source_sha256=row['source_sha256'],
            compact_sha256=row['compact_sha256'],config_sha256=row['config_sha256']))
    assert len(samples)==58
    csvout('coverage.csv',inventory)
    return samples


def square_ab():
    samples=[]; runs=[]; histories={}
    for row in old.rows(AB_RUN/'manifest.tsv'):
        branch=AB_RUN/'results'/row['label']; paths=list(branch.rglob('state.h5')); assert len(paths)==1
        state=paths[0]; receipts={r['relative_path']:r for r in old.rows(branch/'stateless_manifest.tsv')}
        record=receipts[state.relative_to(branch).as_posix()]; assert old.sha(state)==record['compact_sha256']
        cfgpath=AB_RUN/'configs'/(row['label']+'.segment-001.toml')
        seed=AB_RUN/'seeds'/(row['label']+'.h5')
        assert old.sha(cfgpath)==row['config_sha256'] and old.sha(seed)==row['seed_sha256']
        config=tomllib.loads(cfgpath.read_text()); cfg=config['convergence']
        assert config['dmrg']['maxdim']==200 and config['mixing']['damping']==1
        with h5py.File(state) as f, h5py.File(seed) as sf:
            assert scalar(f,'artifact_kind')=='spatial_cell_mps_mft_state'
            for key in ('model_fingerprint','numerical_fingerprint','implementation_sha256','config_sha256','ep_source_sha256'):
                assert scalar(f,'provenance/'+key)==row[key]
            assert scalar(f,'provenance/inherit_sha256')==row['seed_sha256']
            assert scalar(f,'accepted') and scalar(f,'status')=='fixed_point'
            for k in ('L','t','U','V','t0','tp','density','r_range'):
                assert scalar(f,'model/'+k)==config['model'][k]
            n=len(f['history/cell_sweep']); summary={}
            for name in ('A','B'):
                g=f['ladders/'+name]; h=g['history']; e=h['energy']; dg=g['convergence']
                assert scalar(dg,'accepted') and scalar(dg,'status')=='fixed_point'
                assert set(h['update_mode'][()])=={b'unmixed_probe'}
                fields={s:{k:old._julia_array(h[f'fields/{s}/{k}']) for k in ('alpha','beta','mu_cdw')} for s in ('applied','measured')}
                for key in fields['applied']:
                    np.testing.assert_array_equal(fields['applied'][key][...,1:],fields['measured'][key][...,:-1])
                    np.testing.assert_array_equal(fields['applied'][key][...,0],old._julia_array(sf[f'ladders/{name}/fields/{key}']))
                vectors={s:old.channel_vectors(v) for s,v in fields.items()}
                channel_pass=True
                for key,c in h['channels'].items():
                    x,y=(vectors[s][key] for s in ('applied','measured'))
                    np.testing.assert_allclose(np.max(np.abs(y-x),axis=1),c['absolute'][()],atol=1e-15)
                    both=np.concatenate((x[-10:],y[-10:])); span=np.ptp(both,axis=0)
                    a=np.max(abs(span)); r=np.linalg.norm(span)/max(np.max(np.linalg.norm(both,axis=1)),np.finfo(float).eps)
                    np.testing.assert_allclose(a,c['window_absolute'][-1],atol=1e-15)
                    channel_pass &= bool(np.all(c['passes'][-10:]) and (a<=cfg['channel_noise_floor'] or r<=cfg['field_rel_tol']))
                dmrg=np.array([abs(np.diff(h[f'dmrg/{i:04d}/sweep_energy'][-2:])[0]) for i in range(1,n+1)])
                gates=dict(minimum=n>=cfg['minimum_iterations'],channels=channel_pass,
                    field=np.all((h['field_abs_residual'][-10:]<=cfg['field_abs_tol'])|(h['field_rel_residual'][-10:]<=cfg['field_rel_tol'])),
                    slow=scalar(dg,'fixed_point_extrapolated_abs_residual')<=cfg['field_abs_tol'] or scalar(dg,'fixed_point_extrapolated_rel_residual')<=cfg['field_rel_tol'],
                    density=np.max(abs(h['density'][-10:]-.9375))<=cfg['density_tol'],dmrg=np.all(dmrg[-10:]<=cfg['dmrg_sweep_energy_tol']),
                    energy=np.ptp(e['target_density_corrected_variational_energy'][-10:]/128)<=cfg['variational_energy_tol'],
                    identity=scalar(dg,'hamiltonian_identity_error_per_site')<=cfg['hamiltonian_identity_tol'],
                    effective=scalar(dg,'effective_eigenvalue_error_per_site')<=cfg['effective_energy_consistency_tol'])
                assert all(gates.values()),(row['label'],name,gates)
                direct=sum(e[k][()] for k in ('bare_ladder_energy','pair_transverse_energy','exchange_transverse_energy','density_transverse_energy'))
                np.testing.assert_allclose(direct,e['canonical_variational_energy'][()],atol=1e-11,rtol=0)
                np.testing.assert_allclose(e['canonical_variational_energy'][()]+h['chemical_potential'][()]*(.9375-h['density'][()])*128,e['target_density_corrected_variational_energy'][()],atol=1e-11,rtol=0)
                p=profiles(h['correlations']); met=profile_metrics(p)
                diagnostic=state.parent/f'diagnostics_ladder_{name}.h5'
                dr=receipts[diagnostic.relative_to(branch).as_posix()]; assert old.sha(diagnostic)==dr['compact_sha256']
                s=read_diagnostic(diagnostic,record['full_sha256'],row['model_fingerprint'],'fixed_point',True)
                np.testing.assert_allclose(s['density'],p['charge'][-1],atol=2e-11,rtol=0)
                s.update(id=f"square_AB_{row['family']}_{row['V']}_{name}",category='square_AB',campaign=AB_RUN.name,
                    label=row['label'],source=state,parent_index=0,residual=float(h['field_abs_residual'][-1]))
                samples.append(s)
                summary[name]=dict(**{k:v[-1] for k,v in met.items()},density=h['density'][-1],residual=s['residual'],gates=gates)
                histories[row['label']+name]=dict(p=p,metrics=met)
            energy=f['history/target_density_corrected_variational_energy_per_site'][()]
            canonical=f['history/canonical_variational_energy_per_site'][()]
            np.testing.assert_allclose(energy,sum(f[f'ladders/{a}/history/energy/target_density_corrected_variational_energy'][()] for a in ('A','B'))/256,atol=1e-14)
            np.testing.assert_allclose(canonical,sum(f[f'ladders/{a}/history/energy/canonical_variational_energy'][()] for a in ('A','B'))/256,atol=1e-14)
            pa,pb=(histories[row['label']+a]['p'] for a in ('A','B'))
            runs.append(dict(label=row['label'],family=row['family'],V=float(row['V']),t0=float(row['t0']),
                iterations=n,accepted=True,status='fixed_point',job_id=scalar(f,'provenance/slurm_job_id'),
                canonical=canonical[-1],corrected=energy[-1],solver_seconds=np.sum(f['history/wall_seconds'][()]),
                energy_span_last10=np.ptp(energy[-10:]),ladders=summary,
                ab_difference={k:float(np.max(abs(pa[k][-1]-pb[k][-1]))) for k in pa},
                source=state.relative_to(P).as_posix(),compact_sha256=record['compact_sha256'],full_sha256=record['full_sha256']))
    assert len(samples)==8
    dump('square_AB_results.json',runs)
    return samples,runs,histories


def bare():
    directory=P/'output/bare_stage1/20260901_bare_t014_v0_stage1/stateless_results'
    path=directory/'stage1.h5'
    rec=next(r for r in old.rows(directory/'stateless_manifest.tsv') if r['relative_path']=='stage1.h5')
    assert old.sha(path)==rec['compact_sha256']
    with h5py.File(path) as f:
        assert scalar(f,'complete') and scalar(f,'covariance_psd_pass')
        assert 'sqrt(2)' in scalar(f,'pairing/field_metric')
        assert np.max(abs(f['zero_field_raw_map/pair'][()]))<1e-12
        # For disjoint bonds on this real number-conserving state,
        # covariance = (addition+removal)/2 = Re(removal). Contact entries
        # do not obey this identity and are explicitly masked out.
        matrices={k:matrix(f,f'pairing/{k}/covariance') for k in ('rung','leg0','leg1')}
        for k,a in matrices.items():
            a[np.abs(np.subtract.outer(np.arange(len(a)),np.arange(len(a)))) < (1 if k=='rung' else 2)] = np.nan
        return dict(id='bare',category='isolated',geometry='isolated',t0=1.4,V=0.,chi=1200,accepted=True,
            status='backbone_gates_passed',iteration=0,ladder='',path=path,source=path,
            sha256=old.sha(path),source_sha256=rec['full_sha256'],label='isolated t0=1.4 V=0 chi=1200',
            density=f['diagnostics/density'][()].reshape(64,2).mean(axis=1),
            spin=(f['diagnostics/spin'][::2]-f['diagnostics/spin'][1::2])/2,
            bare_matrices=matrices,residual=np.nan)


def channel(s,kind,left='rung',right=None):
    right=left if right is None else right
    if s['id']=='bare':
        assert left==right
        return s['bare_matrices'][left]
    return s[kind][np.ix_(s['indices'][left],s['indices'][right])].real


def distance_values(s,kind='connected',left='rung',right=None,edge=8,max_r=32):
    a=channel(s,kind,left,right); n,m=a.shape
    r=np.arange(1,max_r+1); means=[]; spreads=[]; magnitude=[]
    for d in r:
        vals=[a[i,j] for i in range(edge,n-edge) for j in (i-d,i+d) if edge<=j<m-edge]
        vals=np.asarray(vals); vals=vals[np.isfinite(vals)]
        means.append(np.mean(vals) if len(vals) else np.nan)
        magnitude.append(np.mean(abs(vals)) if len(vals) else np.nan)
        spreads.append(np.percentile(vals,[10,90]) if len(vals) else [np.nan,np.nan])
    return r,np.asarray(means),np.asarray(magnitude),np.asarray(spreads)


def measure_summary(s,edge=8):
    raw=channel(s,'raw'); conn=channel(s,'connected'); n=len(raw)
    x=np.arange(n); mask=(x[:,None]>=edge)&(x[:,None]<n-edge)&(x[None,:]>=edge)&(x[None,:]<n-edge)
    sep=abs(x[:,None]-x[None,:]); regions={ 'short':mask&(sep>=2)&(sep<=4), 'long':mask&(sep>=16)&(sep<=24)}
    values=dict(id=s['id'],geometry=s['geometry'],t0=s['t0'],V=s['V'],chi=s['chi'],
        accepted=s['accepted'],status=s['status'],iteration=s['iteration'],ladder=s['ladder'],
        label=s['label'],residual=s['residual'],spin_rms=float(np.sqrt(np.mean(s['spin'][5:59]**2))))
    for name,m in regions.items():
        for label,a in (('raw',raw),('connected',conn)):
            values[name+'_'+label+'_mean']=float(np.nanmean(a[m]))
            values[name+'_'+label+'_abs']=float(np.nanmean(abs(a[m])))
    if s['id']!='bare':
        values['rung_anomalous_rms']=float(np.sqrt(np.mean(abs(s['expectation'][s['indices']['rung'][8:56]])**2)))
        for left,right,label in [('leg0','leg0','leg0'),('leg1','leg1','leg1'),('rung','leg0','rung_leg0'),('rung','leg1','rung_leg1')]:
            a=channel(s,'connected',left,right)
            ii,jj=np.indices(a.shape);m=(ii>=8)&(ii<56)&(jj>=8)&(jj<55)&(abs(ii-jj)>=2)&(abs(ii-jj)<=8)
            values[label+'_mean_2_8']=float(np.mean(a[m])); values[label+'_negative_fraction_2_8']=float(np.mean(a[m]<-1e-12))
    return values


def fits(samples):
    result=[]
    for s in samples:
        a=channel(s,'connected')
        for ref in REFS:
            for direction in (-1,1):
                for lo,hi in WINDOWS:
                    r=np.arange(lo,hi+1);j=ref-1+direction*r
                    valid=(j>=8)&(j<56);r=r[valid];j=j[valid]
                    if len(r)<6: continue
                    y=a[ref-1,j];keep=np.isfinite(y)&(abs(y)>1e-12)
                    if np.count_nonzero(keep)<6: continue
                    r=r[keep];y=y[keep];ly=np.log(abs(y))
                    for mode,x in (('power',np.log(r)),('exponential',r)):
                        slope,intercept=np.polyfit(x,ly,1);res=ly-(intercept+slope*x)
                        r2=1-np.sum(res**2)/np.sum((ly-ly.mean())**2)
                        result.append(dict(id=s['id'],reference=ref,direction=direction,window_low=lo,window_high=hi,
                            actual_low=int(r.min()),actual_high=int(r.max()),points=len(r),fit=mode,
                            exponent_or_inverse_length=-slope,r2=r2,log_rmse=np.sqrt(np.mean(res**2)),
                            sign_changes=int(np.sum(y[1:]*y[:-1]<0)),minimum_magnitude=np.min(abs(y))))
    csvout('reference_window_fits.csv',result)
    return result


def walls(s):
    hole=1-s['density']; staggered=s['spin']*(-1.)**np.arange(64)
    peaks,_=find_peaks(hole,distance=8,prominence=.1*np.ptp(hole[8:56]))
    peaks=peaks[(peaks>=12)&(peaks<52)]
    roots=np.where(staggered[:-1]*staggered[1:]<0)[0]+.5
    roots=roots[(roots>=8)&(roots<55)]
    a=channel(s,'connected'); strength=np.full(64,np.nan)
    for i in range(12,52):
        js=[j for d in (2,3,4) for j in (i-d,i+d) if 8<=j<56]
        strength[i]=np.mean(abs(a[i,js]))
    rho=float(spearmanr(hole[12:52],strength[12:52]).statistic)
    ordering=np.argsort(hole[12:52]); low=ordering[:10]+12; high=ordering[-10:]+12
    return dict(id=s['id'],hole_peaks_rungs=(peaks+1).tolist(),spin_nodes_rungs=(roots+1).tolist(),
        wall_nearest_node_distance=[float(min(abs(roots-i))) for i in peaks] if len(roots) else [],
        spearman_hole_local_pair=rho,high_hole_pair=np.mean(strength[high]),low_hole_pair=np.mean(strength[low]),
        high_low_ratio=np.mean(strength[high])/np.mean(strength[low]),strength=strength)


def savefig(fig,name):
    fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=180);plt.close(fig)


def representatives(samples):
    def row(n,ladder=''):
        return next(s for s in samples if s.get('parent_index')==n and s['ladder']==ladder)
    return [next(s for s in samples if s['id']=='bare'),row(2),row(18),row(32),row(56,'A')]


def figures(samples,summaries,fitrows,runs,histories):
    reps=representatives(samples)
    titles=['Isolated, V=0, chi=1200','Square paired, V=-0.4','Square stripe, V=0','Cubic stripe, V=-0.4','Trellis A stripe, V=0']
    colors=list(COLORS.values())
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for s,label,color in zip(reps,titles,colors):
        for ax,kind in zip(axes[:2],('raw','connected')):
            r,y,mag,sp=distance_values(s,kind)
            ax.loglog(r,mag,color=color,label=label)
        r,y,mag,sp=distance_values(s,'connected');axes[2].semilogy(r,mag/mag[1],color=color,label=label)
    for ax,title in zip(axes,('Full rung-pair correlations','Connected rung-pair correlations','Connected decay, normalized at r=2')):
        ax.set_title(title,fontsize=11);ax.set_xlabel('Separation r (rungs)');ax.grid(alpha=.18,which='both')
    axes[0].set_ylabel('Bulk mean |P(r)|');axes[1].set_ylabel('Bulk mean |P^c(r)|');axes[2].set_ylabel('|P^c(r)| / |P^c(2)|')
    axes[0].legend(fontsize=8,loc='lower left');fig.suptitle('Pair correlations: same operator convention; rungs 9–56\nCoupled endpoints: chi=200, unaccepted snapshots; isolated reference: chi=1200',fontsize=11)
    savefig(fig,'pair_decay_comparison')

    fig,axes=plt.subplots(2,3,figsize=(12.6,7),layout='constrained')
    for col,s in enumerate(reps[1:4]):
        for kind,ls in [('raw','-'),('connected','--')]:
            for (left,right,label),color in zip([('rung','rung','rung–rung'),('leg0','leg0','leg–leg'),('rung','leg0','rung–leg')],['#17699b','#548348','#be6427']):
                r,y,mag,_=distance_values(s,kind,left,right,max_r=28)
                axes[0,col].plot(r,y,ls=ls,color=color,label=label+(' full' if kind=='raw' else ' connected'))
        axes[0,col].set_yscale('symlog',linthresh=1e-7);axes[0,col].axhline(0,color='.6',lw=.6)
        axes[0,col].set_title(titles[col+1]);axes[0,col].set_xlabel('Bond-start separation (rungs)')
        mat=channel(s,'connected','rung','leg0')[8:56,8:55]
        image=axes[1,col].imshow(mat,cmap='RdBu_r',norm=matplotlib.colors.SymLogNorm(linthresh=1e-5,vmin=-.02,vmax=.02),origin='lower',extent=(9,56,9,57),aspect='auto')
        axes[1,col].set_xlabel('Leg-bond start rung');axes[1,col].set_ylabel('Rung-pair position')
    axes[0,0].set_ylabel('Signed bulk mean P or P^c');axes[0,0].legend(fontsize=7,ncol=2)
    fig.colorbar(image,ax=axes[1,:],shrink=.8,label='Connected rung–leg correlation')
    fig.suptitle('Signs survive the connected subtraction; positive/negative values are retained',fontsize=12)
    savefig(fig,'pair_channel_signs')

    chosen=[reps[2],reps[3],reps[4]];fig,axes=plt.subplots(3,3,figsize=(12.6,8.5),layout='constrained')
    for col,s in enumerate(chosen):
        w=walls(s);x=np.arange(1,65)
        axes[0,col].plot(x,1-s['density'],color='#17699b');axes[0,col].set_title(titles[col+2])
        axes[1,col].plot(x,s['spin']*(-1.)**np.arange(64),color='#8875a9');axes[1,col].axhline(0,color='.6',lw=.6)
        axes[2,col].plot(x,w['strength'],color='#be6427')
        for ax in axes[:,col]:
            for p in w['hole_peaks_rungs']:ax.axvline(p,color='.65',lw=.7,ls=':')
            ax.axvspan(1,8,color='.9');ax.axvspan(57,64,color='.9');ax.set_xlim(1,64);ax.grid(alpha=.12)
        axes[2,col].set_xlabel('Rung');axes[2,col].text(.03,.93,f"Hole-rich / hole-poor = {w['high_low_ratio']:.2f}",transform=axes[2,col].transAxes,fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    for ax,label in zip(axes[:,0],('Holes per site','Staggered leg-odd spin','Mean |P^c(i,i±r)|, r=2–4')):ax.set_ylabel(label)
    fig.suptitle('Short-distance pair correlations and stripe texture\nDotted lines: interior hole-density maxima; shading: excluded edge rungs',fontsize=12)
    savefig(fig,'stripe_pairing_profiles')

    fig,axes=plt.subplots(1,2,figsize=(11.6,4.5),layout='constrained')
    for i,(s,label,color) in enumerate(zip(reps,titles,colors)):
        for row in fitrows:
            if row['id']==s['id'] and row['fit']=='power':
                shift=(row['reference']-24)/100 + (.03 if row['direction']>0 else -.03)
                marker={4:'o',8:'s',12:'^'}[row['window_low']]
                axes[0].scatter(i+shift,row['exponent_or_inverse_length'],color=color,marker=marker,s=23,alpha=.65)
        power=[v for v in fitrows if v['id']==s['id'] and v['fit']=='power']
        exp=[v for v in fitrows if v['id']==s['id'] and v['fit']=='exponential']
        axes[1].scatter([v['log_rmse'] for v in power],[v['log_rmse'] for v in exp],color=color,label=label,s=22)
    axes[0].set_xticks(range(5),['Isolated','Sq. paired','Sq. stripe','Cubic stripe','Trellis stripe']);axes[0].set_ylabel('Effective power exponent from |P^c|');axes[0].grid(alpha=.2)
    axes[1].plot([0,3],[0,3],color='.4',ls=':');axes[1].set_xlabel('Power-law log residual RMS');axes[1].set_ylabel('Exponential log residual RMS');axes[1].legend(fontsize=8)
    fig.suptitle('Reference/window sensitivity: rungs 16, 24, 32; both directions\nCircles: r=4–12; squares: 8–20; triangles: 12–28 (trimmed to bulk)',fontsize=11)
    savefig(fig,'reference_window_sensitivity')

    fig,axes=plt.subplots(2,3,figsize=(12.8,7.6),layout='constrained')
    for v,rowaxes in zip((-.4,-.2),axes):
        for run in [r for r in runs if r['V']==v]:
            color='#17699b' if run['family']=='pairing' else '#be6427'
            for name,ls in [('A','-'),('B','--')]:
                d=histories[run['label']+name];x=np.arange(1,run['iterations']+1);label=run['family']+' seed, '+name
                rowaxes[0].plot(x,d['metrics']['pair_rms'],color=color,ls=ls,label=label)
                rowaxes[1].semilogy(x,d['metrics']['spin_rms'],color=color,ls=ls,label=label)
                rowaxes[2].plot(np.arange(1,64)+.5,d['p']['leg'][-1],color=color,ls=ls,label=label)
        for ax,title in zip(rowaxes,('Leg-pair amplitude','Magnetic relaxation','Terminal leg-pair profile')):ax.set_title(f'V={v:g}: '+title,fontsize=11);ax.grid(alpha=.18)
        rowaxes[0].set_ylabel('Physical leg-pair RMS');rowaxes[1].set_ylabel('Leg-odd spin RMS');rowaxes[2].set_ylabel('Anomalous leg-pair expectation')
        for ax in rowaxes[:2]:ax.set_xlabel('Cell sweep')
        rowaxes[2].set_xlabel('Bond midpoint (rungs)')
    axes[0,0].legend(fontsize=8);fig.suptitle('Square A/B, t0=1.4, chi=200: all four runs pass fixed-point gates',fontsize=13)
    savefig(fig,'square_AB_complete')

    fig,axes=plt.subplots(1,2,figsize=(11.8,4.3),layout='constrained')
    for v,ax in zip((-.4,-.2),axes):
        for s in [s for s in samples if s['category']=='square_AB' and s['V']==v]:
            color='#17699b' if 'pairing' in s['label'] else '#be6427';ls='-' if s['ladder']=='A' else '--'
            r,y,mag,_=distance_values(s,'connected');ax.loglog(r,mag,color=color,ls=ls,label=('pairing' if 'pairing' in s['label'] else 'stripe')+' seed '+s['ladder'])
        one=[s for s in samples if s['category']=='backfill' and s['geometry']=='square' and s['t0']==1.4 and s['V']==v and 'fine' not in s['campaign']]
        for i,s in enumerate(one):
            r,y,mag,_=distance_values(s,'connected');ax.loglog(r,mag,color='.25',ls=':' if i==0 else '-.',label='one-ladder '+('pairing' if 'pairing' in s['label'] else 'stripe'))
        ax.set_title(f'Square t0=1.4, V={v:g}');ax.set_xlabel('Separation r (rungs)');ax.set_ylabel('Bulk mean |P^c_rung(r)|');ax.grid(alpha=.18,which='both')
    axes[0].legend(fontsize=8);fig.suptitle('Accepted square A/B pair correlations compared with one-ladder snapshots',fontsize=12)
    savefig(fig,'square_AB_pair_correlations')

    fig,axes=plt.subplots(1,2,figsize=(11.8,4.5),layout='constrained')
    selections=[([1,2],[31,32]),([53,54],[55,56])]
    for ax,(paired,striped),title in zip(axes,selections,('Square / cubic: t0=1.4, V=-0.4','Trellis one / two ladders: t0=1, V=0')):
        for s in samples:
            idx=s.get('parent_index',0)
            if idx not in paired+striped:continue
            color='#17699b' if idx in paired else '#be6427'
            for kind,ls in [('raw','-'),('connected','--')]:
                r,mean,mag,_=distance_values(s,kind)
                ax.loglog(r,mag,color=color,ls=ls,alpha=.8)
        ax.set_title(title,fontsize=11);ax.set_xlabel('Separation r (rungs)');ax.set_ylabel('Bulk mean pair-correlation magnitude');ax.grid(alpha=.18,which='both')
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([],[],color='#17699b',label='Paired endpoint(s)'),Line2D([],[],color='#be6427',label='Striped endpoint(s)'),Line2D([],[],color='.3',label='Full'),Line2D([],[],color='.3',ls='--',label='Connected')],fontsize=8)
    fig.suptitle('Matched intraladder parameters, chi=200: both seeds and each spatial ladder\nAll displayed states are unaccepted terminal snapshots',fontsize=11)
    savefig(fig,'matched_pair_comparisons')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    samples=backfill(); print('Verified backfill: 56 branches / 58 MPSs',flush=True)
    ab,runs,histories=square_ab();samples+=ab; samples.append(bare())
    print('Verified square A/B: 4 accepted branches / 8 MPSs; isolated reference loaded',flush=True)
    summaries=[measure_summary(s) for s in samples]
    # CSV includes absent cross-channel fields for the older isolated reference.
    keys=list(dict.fromkeys(k for s in summaries for k in s))
    csvout('correlation_summary.csv',[{k:s.get(k,'') for k in keys} for s in summaries])
    fitrows=fits(samples)
    representative=representatives(samples)
    wall_results=[walls(s) for s in samples if s.get('parent_index') in (17,18,31,32,55,56)]
    dump('stripe_wall_comparison.json',wall_results)
    cuts=[]
    for s in samples:
        for edge in (4,8,12):
            summary=measure_summary(s,edge)
            cuts.append(dict(id=s['id'],excluded_edge_rungs=edge,**{k:summary[k] for k in ('short_connected_abs','long_connected_abs','long_raw_abs')}))
    csvout('bulk_cut_sensitivity.csv',cuts)
    provenance=[dict(id=s['id'],path=s['path'].relative_to(P).as_posix(),sha256=s['sha256'],
        source_sha256=s['source_sha256'],status=s['status'],accepted=s['accepted'],
        **{k:s[k] for k in ('implementation','subtraction_error','minimum_gram_eigenvalue','max_imaginary') if k in s}) for s in samples]
    dump('source_validation.json',dict(backfill_branches=56,backfill_mps=58,square_AB_branches=4,square_AB_mps=8,sources=provenance))
    curves=[]
    for s in samples:
        for kind in ('raw','connected'):
            for channelname in ('rung','leg0','leg1'):
                r,mean,magnitude,spread=distance_values(s,kind,channelname)
                curves.extend(dict(id=s['id'],kind=kind,channel=channelname,r=int(d),mean=x,mean_absolute=y,p10=lo,p90=hi) for d,x,y,(lo,hi) in zip(r,mean,magnitude,spread))
    csvout('distance_curves.csv',curves)
    figures(samples,summaries,fitrows,runs,histories)
    print(json.dumps(old.clean({'representatives':[s for s in summaries if s['id'] in {r['id'] for r in representative}],
        'square_AB':runs,'walls':[{k:v for k,v in w.items() if k!='strength'} for w in wall_results]}),indent=2))


if __name__=='__main__':
    main()
