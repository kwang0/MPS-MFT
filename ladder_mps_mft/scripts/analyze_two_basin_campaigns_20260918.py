"""Analyze synced cubic grid, finer square cuts and positive-V seeds.

Read-only simulation analysis. Reuses the verified September 16 loader;
does not modify acceptance, choose energetic winners or run DMRG.
"""
import csv
import json
import re
import textwrap

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import analyze_two_basin_progress as base

PROJECT = base.PROJECT
CAMPAIGNS = {
    'cubic_two_basin_grid_20260918': base.CUBIC,
    'square_fine_cuts_20260918': '20260915_square_two_basin_fine_cuts_95_5_60',
    'square_positive_v_20260918': '20260915_square_t012_vp02_four_seeds_60',
}
COLORS = {'stripe': '#245A9C', 'pairing': '#B65F16',
          'stripe_weak_other': '#245A9C', 'pairing_weak_other': '#B65F16',
          'intertwined_lambda08': '#76569D', 'intertwined_lambda16': '#27816B'}
LABELS = {'stripe': '95% stripe seed', 'pairing': '95% pairing seed',
          'stripe_weak_other': '95% stripe seed', 'pairing_weak_other': '95% pairing seed',
          'intertwined_lambda08': 'Intertwined period 8', 'intertwined_lambda16': 'Intertwined period 16'}


def style(family):
    return dict(color=COLORS[family], ls='--' if family.startswith('pairing') else
                ('-.' if family.startswith('intertwined') else '-'), lw=1.6, label=LABELS[family])


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(base.common.clean(rows))


def save(fig, out, name):
    fig.savefig(out/(name+'.png'), dpi=170)
    fig.savefig(out/(name+'.pdf'))
    plt.close(fig)


def enrich(d, row):
    s = d['summary']; n = s['iterations']
    with h5py.File(PROJECT/s['source']) as f:
        assert (f['model/U'][()], f['model/tp'][()], f['model/density'][()]) == (8., .1, .9375)
        assert base.text(f, 'model/transverse_geometry') == row['geometry']
        h = f['history']
        d['history'] = {k: h[k][()] for k in ('iteration', 'density', 'field_abs_residual',
            'field_rel_residual', 'dmrg_sweep_gate_pass', 'wall_seconds')}
        dmrg = h['dmrg'][f'{n:04d}']
        s.update(ep_mode=base.text(f, 'model/E_p_mode'), ep_signed=float(f['model/E_p_signed'][()]),
                 coupling=float(f['model/effective_mf_coupling_tp2_over_ep'][()]),
                 density_error=abs(h['density'][-1]-.9375),
                 final_dmrg_sweep_delta=abs(np.diff(dmrg['sweep_energy'][-2:])[0]),
                 final_dmrg_discarded_weight=dmrg['sweep_max_discarded_weight'][-1],
                 final_dmrg_maxdim=dmrg['sweep_maxlinkdim'][-1])
        if 'ep_signed' in row:
            np.testing.assert_allclose(s['ep_signed'], float(row['ep_signed']), rtol=0, atol=1e-14)
            assert s['ep_mode'] == row['ep_mode']
        if 'ep_interpolation_weight' in row:
            w = float(row['ep_interpolation_weight'])
            expected = (1-w)*float(row['ep_lower_signed'])+w*float(row['ep_upper_signed'])
            np.testing.assert_allclose(s['ep_signed'], expected, atol=1e-14, rtol=0)
            s['ep_interpolation'] = {k: row[k] for k in row if k.startswith('ep_')}
    s.update(pair_fractional_change_last10=d['pair_rms'][-1]/d['pair_rms'][-10]-1,
             leg_pair_mean=float(np.mean(d['leg'][-1, 5:58])),
             rung_pair_mean=float(np.mean(d['rung'][5:59])),
             pair_profile_max_change_last10=np.max(np.abs(d['leg'][-1]-d['leg'][-10])),
             spin_rms_at10=d['spin_rms'][9], spin_rms_at30=d['spin_rms'][29],
             pair_rms_at10=d['pair_rms'][9], pair_rms_at30=d['pair_rms'][29],
             charge_std_change_last10=np.std(d['charge'][-1, 5:59])-np.std(d['charge'][-10, 5:59]))
    d['central_spin_rms'] = np.sqrt(np.mean(d['spin'][:,16:48]**2,axis=1))
    s.update(central_spin_rms_final=d['central_spin_rms'][-1],
             central_spin_fractional_change_last10=d['central_spin_rms'][-1]/d['central_spin_rms'][-10]-1,
             spin_outer14_each_end_weight=(np.sum(d['spin'][-1,:14]**2)+np.sum(d['spin'][-1,-14:]**2))/max(np.sum(d['spin'][-1]**2),1e-60))
    x = np.arange(20); y = np.log(np.maximum(d['spin_rms'][-20:], 1e-30))
    fit = np.polyfit(x, y, 1)
    s['spin_log_gain_last20'] = np.exp(fit[0])
    s['spin_log_fit_r2'] = 1-np.sum((y-np.polyval(fit, x))**2)/max(np.sum((y-y.mean())**2), 1e-30)
    s['first_persistent_pair_below_1e4'] = next((i+1 for i in range(n) if np.all(d['pair_rms'][i:]<1e-4)), None)
    spin, pair = s['spin_rms_final'], s['pair_rms_final']
    opposite = s['leg_pair_mean']*s['rung_pair_mean'] < 0
    if spin>1e-3 and pair<1e-4:
        phase = 'S'
    elif pair>1e-3 and opposite and spin<1e-4:
        phase = 'D'
    elif pair>1e-3 and opposite and spin<1e-3 and s['spin_fractional_change_last10']<-.2:
        phase = 'D*'  # Clear paired basin, residual spin still decaying.
    elif pair>1e-3 and spin>1e-3:
        phase = 'P+SDW'  # Observed simultaneous orders; stationarity not implied.
    else:
        phase = '?'
    s['phase_observation'] = phase
    for key, values in (('charge', d['charge'][-1]), ('spin', d['spin'][-1])):
        amp = np.abs(np.fft.rfft(values-values.mean()))/64
        s[key+'_dft']['spectrum'] = amp
    return d


def log_only(run, row, jobs):
    job = next(j for j in jobs if j['label']==row['label'])
    path = run/'logs'/f"{row['label']}.s1-{job['job_id']}.out"
    text = path.read_text() if path.exists() else ''
    matches = re.findall(r'^MF\s+(\d+)\s+n=([\d.eE+-]+)\s+mu=\s*([\d.eE+-]+)\s+r_abs=([\d.eE+-]+)\s+r_rel=([\d.eE+-]+)\s+Evar_target/site=([\d.eE+-]+).*?status=(\w+)', text, re.M)
    records = [dict(iteration=int(a), density=float(b), chemical_potential=float(c),
                    field_absolute=float(d), field_relative=float(e), energy_per_site=float(g), status=h)
               for a,b,c,d,e,g,h in matches]
    return dict(family=row['family'], label=row['label'], job_id=job['job_id'],
                source=str(path.relative_to(PROJECT)), sha256=base.common.sha(path) if path.exists() else None,
                state_available=False, records=records,
                last_completed_evaluation=records[-1]['iteration'] if records else None,
                limitation='stdout only; no synced spatial profiles, acceptance or terminal allocation evidence')


def export(out, data, missing):
    summaries = [d['summary'] for d in data]
    points = []
    for t0, v in sorted({(s['t0'], s['V']) for s in summaries}):
        group = [d for d in data if (d['summary']['t0'], d['summary']['V'])==(t0,v)]
        assert all(d['summary']['fingerprints']==group[0]['summary']['fingerprints'] for d in group)
        byfamily = {d['summary']['family']:d for d in group}
        record = dict(t0=t0, V=v, phases={k:d['summary']['phase_observation'] for k,d in byfamily.items()},
                      accepted_count=sum(d['summary']['accepted'] for d in group), energy_ranking_eligible=False)
        if set(byfamily)=={'stripe','pairing'}:
            a,b=byfamily['stripe'],byfamily['pairing']
            record.update(endpoint_energy_pairing_minus_stripe=b['energy'][-1]-a['energy'][-1],
                sum_last10_energy_ranges=a['summary']['energy_span_last10']+b['summary']['energy_span_last10'],
                max_charge_difference=np.max(np.abs(a['charge'][-1]-b['charge'][-1])),
                max_spin_difference=np.max(np.abs(a['spin'][-1]-b['spin'][-1])))
        points.append(record)
    accounted=[s for s in summaries if s['allocation'] is not None]
    payload=dict(date='2026-09-18', geometry=summaries[0]['geometry'], L=64, chi=200,
        runs=summaries, points=points, missing=missing, total_iterations=sum(s['iterations'] for s in summaries),
        accepted_count=sum(s['accepted'] for s in summaries),
        total_solver_node_hours=sum(s['solver_node_hours'] for s in summaries),
        allocation_jobs_available=len(accounted), allocation_jobs_total=len(summaries),
        known_allocation_node_hours=sum(s['allocation']['node_hours'] for s in accounted),
        unaccounted_solver_node_hours=sum(s['solver_node_hours'] for s in summaries if s['allocation'] is None),
        interpretation='Observed trajectories only. No accepted energetic selection; D* means residual spin decay; P+SDW means simultaneous orders, not established stationary coexistence.')
    (out/'analysis.json').write_text(json.dumps(base.common.clean(payload), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    fields=('t0','V','family','job_id','phase_observation','status','accepted','iterations',
            'spin_rms_final','pair_rms_final','rung_pair_mean','charge_std_final','global_relative',
            'energy_final','energy_span_last10','density_error','spin_fractional_change_last10',
            'pair_fractional_change_last10','spin_log_gain_last20','ep_mode','ep_signed','coupling',
            'final_dmrg_sweep_delta','final_dmrg_discarded_weight','solver_node_hours')
    write_csv(out/'run_summary.csv', [{**{k:s[k] for k in fields},
        'allocation_node_hours':s['allocation']['node_hours'] if s['allocation'] else None,
        'failed_gates':'; '.join(s['failed_gates'])} for s in summaries])
    write_csv(out/'sources.csv', [{k:s[k] for k in ('t0','V','family','job_id','campaign','source',
        'compact_sha256','full_sha256','config_sha256')} for s in summaries])
    write_csv(out/'iteration_history.csv', [dict(t0=s['t0'],V=s['V'],family=s['family'],job_id=s['job_id'],
        iteration=i+1, energy_per_site=d['energy'][i], spin_rms=d['spin_rms'][i], pair_rms=d['pair_rms'][i],
        central_spin_rms=d['central_spin_rms'][i],
        charge_std=np.std(d['charge'][i,5:59]), density=d['history']['density'][i],
        global_relative_residual=d['history']['field_rel_residual'][i])
        for d in data for s in [d['summary']] for i in range(s['iterations'])])
    write_csv(out/'terminal_profiles.csv', [dict(t0=s['t0'],V=s['V'],family=s['family'],rung=i+1,
        charge=d['charge'][-1,i],spin_odd=d['spin'][-1,i],rung_pair=d['rung'][i],
        leg_pair_right=d['leg'][-1,i] if i<63 else None)
        for d in data for s in [d['summary']] for i in range(64)])
    return payload


def history_grids(out, data, coordinates, geometry):
    nr = len(coordinates)//3
    for key,title,ylabel,log in (
        ('energy','Corrected canonical energy: full histories','Energy per site [t]',False),
        ('energy_late','Corrected canonical energy: final 15 evaluations','Energy per site [t]',False),
        ('spin_rms','Spin-order histories','Bulk physical leg-odd spin RMS',True),
        ('pair_rms','Pairing histories','Bulk physical leg-pair RMS',True)):
        fig, axes=plt.subplots(nr,3,figsize=(13,3.1*nr+1),layout='constrained',squeeze=False)
        for ax,(t0,v) in zip(axes.flat,coordinates):
            for d in data:
                s=d['summary']
                if (s['t0'],s['V'])!=(t0,v):continue
                y=d['energy'] if key.startswith('energy') else d[key]
                sl=slice(-15,None) if key=='energy_late' else slice(None)
                ax.plot(np.arange(1,len(y)+1)[sl],y[sl],**style(s['family']))
            ax.set_title(f't0={t0:g}, V={v:g}',fontsize=11);ax.grid(alpha=.2)
            if log:ax.set(yscale='log',ylim=(1e-8,.5) if key=='spin_rms' else (1e-13,.1))
            else:ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        for ax in axes[-1]:ax.set_xlabel('MF evaluation')
        for ax in axes[:,0]:ax.set_ylabel(ylabel)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(title+'\n'+geometry+'; solid = stripe seed, dashed = pairing seed; no accepted endpoints',fontsize=14)
        save(fig,out,('pairing_rms' if key=='pair_rms' else key)+'_grid')


def cubic_phase(out, data):
    fig,ax=plt.subplots(figsize=(9,7.4));fig.subplots_adjust(left=.12,right=.97,top=.79,bottom=.23)
    ax.set(xlim=(.9,1.5),ylim=(-.48,.075),xticks=[1,1.2,1.4],yticks=[0,-.2,-.4],xlabel=r'$t_0/t$',ylabel=r'$V/t$')
    ax.grid(alpha=.2)
    for t0 in (1.,1.2,1.4):
        for v in (0.,-.2,-.4):
            group=[d for d in data if (d['summary']['t0'],d['summary']['V'])==(t0,v)]
            assert len(group)==2 and all(d['summary']['phase_observation']=='S' for d in group)
            ax.scatter(t0,v,s=2100,color=COLORS['stripe'],edgecolors='white',linewidths=2,zorder=3)
            ax.text(t0,v,'S',ha='center',va='center',color='white',fontsize=23,fontweight='bold')
            ax.text(t0,v-.057,'both seeds → S',ha='center',va='top',fontsize=10)
    fig.suptitle('Preliminary cubic-unfrustrated phase diagram',fontsize=18,y=.97)
    fig.text(.5,.92,r'$L=64$, $\chi=200$, $n=0.9375$, $t_\perp/t=0.1$ | reciprocal 95% / 5% seeds',ha='center',fontsize=11)
    fig.text(.5,.86,'S: stripe CDW / SDW; negligible pairing',ha='center',fontsize=13,color=COLORS['stripe'])
    fig.text(.12,.135,'Both square paired coordinates also reach stripes here: (1.4,-0.4) and (1.4,-0.2).',fontsize=10)
    fig.text(.12,.093,'All 18 endpoints remain formally unaccepted; these are observed basin assignments.',fontsize=10,fontweight='bold')
    fig.text(.12,.052,'60 raw evaluations per start. No Anderson, interpolated boundary or energetic winner.',fontsize=10)
    save(fig,out,'preliminary_phase_diagram')


def cut_summary(out, data, payload):
    old=json.loads((PROJECT/'docs/reports/two_basin_grid_20260915/analysis.json').read_text())
    anchors=[dict(t0=s['t0'],V=s['V'],family=s['family'],spin=s['physical_spin_odd_bulk_rms'],pair=s['physical_leg_pair_bulk_rms']) for s in old['runs']]
    fig,axes=plt.subplots(2,3,figsize=(13.5,8),layout='constrained')
    for iy,(fixed,key,lo,hi,label) in enumerate(((1.4,'V',-.2,0.,'V/t at t0=1.4'),(-.4,'t0',1.2,1.4,'t0/t at V=-0.4'))):
        match=lambda s: s['t0']==fixed if key=='V' else s['V']==fixed
        new=[d['summary'] for d in data if match(d['summary'])]
        for family in ('stripe','pairing'):
            series=[dict(x=s[key],spin=s['spin_rms_final'],pair=s['pair_rms_final'],anchor=False) for s in new if s['family']==family]
            series += [dict(x=s[key],spin=s['spin'],pair=s['pair'],anchor=True) for s in anchors if match(s) and s['family']==family and s[key] in (lo,hi)]
            series.sort(key=lambda s:s['x'])
            for ix,obs in enumerate(('spin','pair')):
                ax=axes[iy,ix];ax.plot([s['x'] for s in series],[s[obs] for s in series],**style(family))
                for s in series:ax.scatter(s['x'],s[obs],s=48,facecolors='white' if s['anchor'] else COLORS[family],edgecolors=COLORS[family],zorder=4)
                ax.set_yscale('log');ax.set_ylim((1e-8,.4) if obs=='spin' else (1e-11,.1))
        points=sorted([p for p in payload['points'] if match(p)],key=lambda p:p[key])
        axes[iy,2].errorbar([p[key] for p in points],[1e6*p['endpoint_energy_pairing_minus_stripe'] for p in points],
            yerr=[1e6*p['sum_last10_energy_ranges'] for p in points],fmt='o',capsize=4,color='#4B4B4B')
        axes[iy,2].axhline(0,color='0.5',lw=.8)
        for ix,ax in enumerate(axes[iy]):
            ticks=([lo]+[p[key] for p in points]+[hi]) if ix<2 else [p[key] for p in points]
            ax.set_xticks(ticks, [f'{v:g}' for v in ticks])
            ax.set_xlabel(label);ax.grid(alpha=.2)
        axes[iy,0].set_ylabel('Bulk physical spin RMS');axes[iy,1].set_ylabel('Bulk physical leg-pair RMS')
        axes[iy,2].set_ylabel(r'$(E_{pair\ seed}-E_{stripe\ seed})/N$ [$10^{-6}t$]')
    for ax,title in zip(axes[0],('Spin across the cut','Pairing across the cut','Endpoint energy gap: new points only')):ax.set_title(title,fontsize=11)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Finer square cuts expose seed-dependent trajectories near the boundary',fontsize=15)
    fig.supxlabel('Filled: 60-step finer runs. Open: earlier coarse endpoints with different stopping controls; lines guide the eye.\n'
                  'Energy bars = ±sum of both last-ten energy ranges, not statistical errors or certified bounds. No accepted endpoints.',fontsize=9)
    save(fig,out,'cut_summary')


def profiles(out, data, coordinates, name, title):
    fig,axes=plt.subplots(3,len(coordinates),figsize=(6.2*len(coordinates),9),layout='constrained',squeeze=False)
    for ix,coord in enumerate(coordinates):
        for d in data:
            s=d['summary']
            if (s['t0'],s['V'])!=coord:continue
            reference=next(x for x in data if (x['summary']['t0'],x['summary']['V'])==coord)
            spin_sign=-1 if np.dot(reference['spin'][-1],d['spin'][-1])<0 else 1
            for iy,key in enumerate(('charge','spin','leg')):
                profile=d[key][-1]
                if key=='spin':profile=profile*(-1.)**np.arange(64)*spin_sign
                x=np.arange(1,len(profile)+1)+(.5 if key=='leg' else 0)
                axes[iy,ix].plot(x,profile,**style(s['family']))
                earlier=d[key][-10]*((-1.)**np.arange(64)*spin_sign if key=='spin' else 1)
                axes[iy,ix].plot(x,earlier,color=COLORS[s['family']],alpha=.25,lw=1)
            axes[2,ix].plot(np.arange(1,65),d['rung'],color=COLORS[s['family']],ls=':',lw=1)
        axes[0,ix].set_title(f't0={coord[0]:g}, V={coord[1]:g}')
        axes[2,ix].set_xlabel('Rung / bond midpoint')
    for ax in axes.flat:ax.grid(alpha=.2)
    for ax,lab in zip(axes[:,0],('Electrons per site','Staggered leg-odd spin','Physical pairing')):ax.set_ylabel(lab)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(textwrap.fill(title,48) if len(coordinates)==1 else title,fontsize=15)
    note='Strong curves: evaluation 60; faint: evaluation 51. Pairing dotted curves: final rung pairing. Spin signs aligned globally for comparison; raw data retained. No accepted endpoints.'
    fig.supxlabel(textwrap.fill(note,80) if len(coordinates)==1 else textwrap.fill(note,155),fontsize=9)
    save(fig,out,name)


def positive_figures(out,data,missing):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for d in data:
        family=d['summary']['family']; it=np.arange(1,len(d['energy'])+1)
        for ax,key in zip((axes[0,0],axes[0,1],axes[1,0]),('energy','spin_rms','pair_rms')):
            ax.plot(it,d[key],**style(family))
        axes[1,1].plot(it[-15:],d['energy'][-15:],**style(family))
    for m in missing:
        if m['records']:
            axes[0,0].plot([r['iteration'] for r in m['records']],[r['energy_per_site'] for r in m['records']],**style(m['family']))
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlabel('MF evaluation')
    axes[0,0].set_title('Energy: full histories; period 8 is log-only')
    axes[0,0].set_ylabel('Corrected energy per site [t]')
    axes[1,1].set_title('Energy: last 15 records of completed starts')
    axes[1,1].set_ylabel('Corrected energy per site [t]')
    for ax in (axes[0,0],axes[1,1]):ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[0,1].set(title='Spin: completed starts',ylabel='Bulk physical spin RMS')
    axes[1,0].set(title='Pairing: completed starts',ylabel='Bulk physical leg-pair RMS',yscale='log')
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Square (1.2,+0.2): three completed seeds lose pairing',fontsize=15)
    fig.supxlabel('Three 60-step states remain unaccepted. Period-8 seed: 46 complete MF records in synced stdout; no spatial state available.',fontsize=10)
    save(fig,out,'histories')
    profiles(out,data,[(1.2,.2)],'terminal_profiles','Positive V: stripe profiles after 60 evaluations')


def boundary_spin(out,data):
    d=next(d for d in data if (d['summary']['t0'],d['summary']['V'],d['summary']['family'])==(1.4,-.05,'pairing'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    x=np.arange(1,len(d['spin_rms'])+1)
    axes[0].plot(x,d['spin_rms'],label='Usual window: rungs 6–59',color='#245A9C')
    axes[0].plot(x,d['central_spin_rms'],label='Central window: rungs 17–48',color='#B65F16')
    axes[0].set(xlabel='MF evaluation',ylabel='Physical spin RMS',yscale='log',title='Usual RMS rises late; central RMS still falls')
    axes[0].legend(fontsize=9)
    for i,alpha in ((50,.4),(59,1)):
        axes[1].plot(np.arange(1,65),d['spin'][i]*(-1.)**np.arange(64),label=f'Evaluation {i+1}',alpha=alpha,color='#245A9C')
    axes[1].axvspan(17,48,color='#B65F16',alpha=.1,label='Central window')
    axes[1].set(xlabel='Rung',ylabel='Staggered leg-odd spin',title='Weak spin is concentrated near the open ends')
    axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Square (1.4, −0.05), pairing seed: an edge-weighted spin remainder',fontsize=14)
    fig.supxlabel('96.7% of final spin² lies in the outer 14 rungs at each end. This is not established bulk coexistence.\nAll profiles are physical correlations; endpoint remains unaccepted.',fontsize=10)
    save(fig,out,'boundary_spin')


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    accounting=base.common.rows(PROJECT/'output/project_budget/additional_node_hours_reconciliations.tsv')
    for name,campaign in CAMPAIGNS.items():
        run=base.ROOT/campaign;out=PROJECT/'docs/reports'/name;out.mkdir(parents=True,exist_ok=True)
        manifest,jobs=base.common.rows(run/'manifest.tsv'),base.common.rows(run/'jobs.tsv')
        data=[];missing=[]
        for row in manifest:
            if list((run/'results'/row['label']).rglob('state.h5')):
                d=enrich(base.load(run,row,jobs,accounting),row);data.append(d)
                s=d['summary'];print(name,s['t0'],s['V'],s['family'],s['phase_observation'],flush=True)
            else:missing.append(log_only(run,row,jobs))
        expected={'cubic_two_basin_grid_20260918':18,'square_fine_cuts_20260918':12,'square_positive_v_20260918':3}
        assert len(data)==expected[name]
        payload=export(out,data,missing)
        if name.startswith('cubic'):
            coords=[(t,v) for v in (0.,-.2,-.4) for t in (1.,1.2,1.4)]
            history_grids(out,data,coords,'Cubic unfrustrated');cubic_phase(out,data)
        elif 'fine_cuts' in name:
            coords=[(1.4,v) for v in (-.15,-.1,-.05)]+[(t,-.4) for t in (1.25,1.3,1.35)]
            history_grids(out,data,coords,'Square finer cuts');cut_summary(out,data,payload)
            profiles(out,data,[(1.4,-.05),(1.25,-.4)],'split_seed_profiles','Distinct endpoint textures at the two seed-dependent points')
            boundary_spin(out,data)
        else:
            positive_figures(out,data,missing)
            for m in missing:
                if m['records']:write_csv(out/(m['family']+'_log_history.csv'),m['records'])
        print(json.dumps({k:v for k,v in payload.items() if k not in ('runs','missing','points')},indent=2),flush=True)
        for d in data:assert base.common.sha(PROJECT/d['summary']['source'])==d['summary']['compact_sha256']


if __name__=='__main__':
    main()
