"""Focused final-sync supplements: cut order parameters and endpoint diagnostics.

Reads immutable state/correlation histories and previously audited inventories.
Does not run DMRG, change acceptance, or rank unaccepted solutions.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
import analyze_two_basin_campaigns_20260918 as report
import analyze_trellis_progress_20260918 as trellis

base = report.base
PROJECT = base.PROJECT
OUT = PROJECT/'docs/reports/campaign_review_20260918'


def dump(path, value):
    path.write_text(json.dumps(base.common.clean(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def order_cuts():
    out=PROJECT/'docs/reports/square_fine_cuts_20260918'
    sources=json.loads((out/'variational_energy_analysis.json').read_text())['endpoints']
    assert len(sources)==20 and len({s['source'] for s in sources})==20
    rows=[]
    for s in sources:
        path=PROJECT/s['source'];assert base.common.sha(path)==s['source_sha256']
        d=base.read_arrays(path)
        rows.append(dict(cut=s['cut'],coordinate=s['coordinate'],family=s['family'],sampling=s['sampling'],
            accepted=s['accepted'],iterations=s['cumulative_iterations'],
            spin_rms=d['spin_rms'][-1],central_spin_rms=np.sqrt(np.mean(d['spin'][-1,16:48]**2)),
            pair_rms=d['pair_rms'][-1],central_pair_rms=np.sqrt(np.mean(d['leg'][-1,16:47]**2)),
            spin_range_last10=np.ptp(d['spin_rms'][-10:]),pair_range_last10=np.ptp(d['pair_rms'][-10:]),
            source=s['source'],source_sha256=s['source_sha256']))
        assert base.common.sha(path)==s['source_sha256']
    report.write_csv(out/'order_parameter_cuts.csv',rows)
    dump(out/'order_parameter_analysis.json',dict(date='2026-09-19',
        spin_definition='Leg-odd physical Sz: (Sz_leg0-Sz_leg1)/2; RMS over rungs 6..59.',
        pair_definition='Real symmetric nearest-leg anomalous amplitude, averaged over both legs before RMS; bonds with left rung 6..58.',
        central_windows='Spin rungs 17..48, pair bonds with left rung 17..47.',
        interpretation='Independent finite-iteration starts, not equilibrium branches. No new convergence or order-of-transition claim.',endpoints=rows))
    for logarithmic in (False,True):
        fig,axes=plt.subplots(2,2,figsize=(11.5,7.8),layout='constrained')
        for col,cut in enumerate(('V','t0')):
            for family,marker in [('stripe','o'),('pairing','s')]:
                rr=sorted([r for r in rows if r['cut']==cut and r['family']==family],key=lambda r:r['coordinate'])
                x=[r['coordinate'] for r in rr]
                for row,key in enumerate(('spin_rms','pair_rms')):
                    ax=axes[row,col];y=[r[key] for r in rr]
                    ax.plot(x,y,**report.style(family))
                    for r,xx,yy in zip(rr,x,y):
                        ax.plot(xx,yy,marker=marker,ms=7,color=report.COLORS[family],
                                mfc=report.COLORS[family] if r['sampling']=='fine' else 'white')
                    if row==0:
                        ax.plot(x,[r['central_spin_rms'] for r in rr],color=report.COLORS[family],ls=':',lw=1.1,
                                marker='+',ms=6,label=report.LABELS[family]+' (central 32 rungs)')
                    if logarithmic:ax.set_yscale('log')
                    else:ax.set_ylim(bottom=0)
                    ax.grid(alpha=.2);ax.set_xticks(x)
            axes[0,col].set_title('Square, t0=1.4' if cut=='V' else 'Square, V=-0.4')
            axes[1,col].set_xlabel('V / t' if cut=='V' else 't0 / t')
        axes[0,0].set_ylabel('Physical spin RMS');axes[1,0].set_ylabel('Anomalous leg-pair RMS')
        axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=9)
        fig.suptitle('Spin and pairing across the finer square transition cuts',fontsize=14)
        fig.supxlabel('Filled: fine points; open: coarse anchors with latest continuations. Lines join independent starts.\nDotted spin curves use the central window to expose end-weighted remnants. All endpoints are unaccepted.',fontsize=9)
        report.save(fig,out,'order_parameter_cuts_log' if logarithmic else 'order_parameter_cuts')
    # Keep the earlier combined RMS/energy-gap figure consistent with these windows.
    fine=json.loads((out/'analysis.json').read_text())
    report.cut_summary(out,[dict(summary=s) for s in fine['runs']],fine)
    return rows


def endpoint_supplements():
    posout=PROJECT/'docs/reports/square_positive_v_20260918'
    positive=json.loads((posout/'analysis.json').read_text())
    pref=next(s for s in positive['runs'] if s['family']=='stripe_weak_other')
    pos=[]
    for s in positive['runs']:
        path=PROJECT/s['source'];assert base.common.sha(path)==s['compact_sha256']
        d=base.read_arrays(path)
        pos.append(dict(family=s['family'],job_id=s['job_id'],spin_rms=s['spin_rms_final'],pair_rms=s['pair_rms_final'],
            energy=s['energy_final'],energy_minus_stripe=s['energy_final']-pref['energy_final'],
            energy_range=s['energy_span_last10'],charge_mode=s['charge_dft']['mode'],spin_mode=s['spin_dft']['mode'],
            spin_nodes_final=base.nodes(d['spin'][-1]),spin_nodes_first=base.nodes(d['spin'][0]),
            final_spin_profile_change=np.max(np.abs(d['spin'][-1]-d['spin'][-10])),
            final_charge_profile_change=np.max(np.abs(d['charge'][-1]-d['charge'][-10])),
            source=s['source'],source_sha256=s['compact_sha256']))
        assert base.common.sha(path)==s['compact_sha256']
    dump(posout/'endpoint_comparison.json',dict(date='2026-09-19',reference_family=pref['family'],runs=pos))
    tpay=json.loads((trellis.OUT/'analysis.json').read_text())
    diagnostics=[];plotdata=[]
    for s in tpay['runs']:
        path=PROJECT/s['source'];assert base.common.sha(path)==s['compact_sha256']
        with h5py.File(path) as f:
            fields=[];ladders=[]
            for name,g in f['ladders'].items():
                h=g['history'];n=len(h['iteration'])
                fields.extend([v[()].reshape(n,-1) for v in h['fields/measured'].values()])
                spin=((h['correlations/density_up'][()]-h['correlations/density_down'][()])[:,::2]
                     -(h['correlations/density_up'][()]-h['correlations/density_down'][()])[:,1::2])/4
                charge=(h['correlations/density_up'][()]+h['correlations/density_down'][()]).reshape(n,64,2).mean(axis=2)
                ladders.append(dict(name=name,spin=spin,charge=charge))
            vectors=np.concatenate(fields,axis=1)
            lag1=np.max(np.abs(vectors[1:]-vectors[:-1]),axis=1)
            lag2=np.max(np.abs(vectors[2:]-vectors[:-2]),axis=1)
            delta=vectors[-1]-vectors[-2];prior=vectors[-2]-vectors[-3]
            r=dict(cell=s['cell'],family=s['family'],source=s['source'],source_sha256=s['compact_sha256'],
                final_field_lag1_max=lag1[-1],final_field_lag2_max=lag2[-1],lag2_over_lag1=lag2[-1]/lag1[-1],
                last_field_increment_cosine=np.dot(delta,prior)/(np.linalg.norm(delta)*np.linalg.norm(prior)),
                final_increment_norm_ratio=np.linalg.norm(delta)/np.linalg.norm(prior),
                lag1_at_sweep51=lag1[49],lag1_at_sweep60=lag1[-1],
                ladders=[dict(ladder=l['name'],spin_nodes=base.nodes(l['spin'][-1]),
                    charge_mode=int(np.argmax(np.abs(np.fft.rfft(l['charge'][-1]-l['charge'][-1].mean()))[1:])+1),
                    spin_mode=int(np.argmax(np.abs(np.fft.rfft(l['spin'][-1]-l['spin'][-1].mean()))[1:])+1),
                    spin_profile_step=np.max(np.abs(l['spin'][-1]-l['spin'][-2]))) for l in ladders])
            diagnostics.append(r)
            if s['cell']=='two_ladder':plotdata.append((s,lag1,lag2,ladders))
        assert base.common.sha(path)==s['compact_sha256']
    fig,axes=plt.subplots(2,2,figsize=(12,7.8),layout='constrained')
    for col,(s,lag1,lag2,ladders) in enumerate(plotdata):
        axes[0,col].semilogy(np.arange(2,len(lag1)+2),lag1,label='Successive measured fields',color='#245A9C')
        axes[0,col].semilogy(np.arange(3,len(lag2)+3),lag2,label='Two-sweep separation',color='#B65F16',ls='--')
        axes[0,col].set_title('Two-ladder cell / '+s['family']+' seed');axes[0,col].legend(fontsize=9)
        axes[0,col].set_xlabel('Cell sweep')
        l=ladders[0]
        for i,color,ls in [(57,'#27816B',':'),(58,'#B65F16','--'),(59,'#245A9C','-')]:
            axes[1,col].plot(np.arange(1,65),l['spin'][i]*(-1.)**np.arange(64),color=color,ls=ls,label=f'A, sweep {i+1}')
        axes[1,col].legend(ncol=3,fontsize=8);axes[1,col].set_xlabel('Rung on spatial ladder A')
    axes[0,0].set_ylabel('Maximum absolute field difference')
    axes[1,0].set_ylabel('Staggered leg-odd spin')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Two-ladder trellis: large alternating relaxation remains at sweep 60',fontsize=14)
    fig.supxlabel('All spatial ladders and field channels enter the upper panels. Near recurrence after two sweeps is not an accepted orbit.',fontsize=9)
    report.save(fig,trellis.OUT,'two_ladder_relaxation')
    dump(trellis.OUT/'relaxation_diagnostics.json',dict(date='2026-09-19',runs=diagnostics))
    return pos,diagnostics


def accounting():
    ledger=PROJECT/'output/project_budget/additional_node_hours_reconciliations.tsv'
    records=base.common.rows(ledger); campaigns=[]
    for name in ('cubic_two_basin_grid_20260918','square_fine_cuts_20260918','square_positive_v_20260918','trellis_progress_20260918'):
        p=json.loads((PROJECT/'docs/reports'/name/'analysis.json').read_text());matched=[]
        for s in p['runs']:
            rr=[r for r in records if r['job_id']==s['job_id']]
            if not rr:continue
            r=rr[-1];assert r['label']==s['label'] and r['sacct_state']=='COMPLETED'
            expected_campaign=s.get('campaign') or (PROJECT/s['source']).relative_to(base.ROOT).parts[0]
            assert r['campaign']==expected_campaign
            cost=int(r['elapsed_raw_seconds'])/3600*float(r['effective_node_fraction'])
            np.testing.assert_allclose(cost,float(r['measured_node_hours']),atol=1e-9)
            matched.append(dict(job_id=s['job_id'],label=s['label'],node_hours=cost,seconds=int(r['elapsed_raw_seconds'])))
        sweeps=sum(s.get('iterations',s.get('cell_sweeps')) for s in p['runs'])
        campaigns.append(dict(report=name,runs=len(p['runs']),cell_updates=sweeps,
            ladder_solves=sum(s.get('ladder_solves',s.get('iterations')) for s in p['runs']),
            solver_node_hours=p['total_solver_node_hours'],accounted_jobs=len(matched),
            actual_node_hours=sum(r['node_hours'] for r in matched),jobs=matched))
    result=dict(date='2026-09-19',accounting_source=str(ledger.relative_to(PROJECT)),accounting_sha256=base.common.sha(ledger),campaigns=campaigns,
        total_runs=sum(r['runs'] for r in campaigns),total_cell_updates=sum(r['cell_updates'] for r in campaigns),
        total_ladder_solves=sum(r['ladder_solves'] for r in campaigns),
        total_solver_node_hours=sum(r['solver_node_hours'] for r in campaigns),
        total_actual_node_hours=sum(r['actual_node_hours'] for r in campaigns),
        total_accounted_jobs=sum(r['accounted_jobs'] for r in campaigns))
    dump(OUT/'completion_accounting.json',result)
    return result


if __name__=='__main__':
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cuts=order_cuts();pos,tre=endpoint_supplements();cost=accounting()
    print(json.dumps(base.common.clean(dict(positive=pos,trellis=tre,cost={k:v for k,v in cost.items() if k!='campaigns'})),indent=2))
