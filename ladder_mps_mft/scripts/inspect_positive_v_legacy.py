"""Extract a smooth seed recipe from an incomplete legacy profile, read-only."""
import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent / 'stateless_data/results_L_64_U_8.0_V_0.2_t0_1.2_t_p_0.1_chi_200_density_0.9375_gpu_nodamping.h5'
SOURCE_SHA = '8a8f5b917d11259d34ce773cb2860fb20d809f5dccfd252342f69a4596b9fec6'
REPORT = PROJECT / 'docs/reports/square_positive_v_seeds_20260915'
RECIPE = PROJECT / 'data/positive_v_intertwined_recipe.toml'


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    with h5py.File(SOURCE) as f:
        pair = f['C_pair_list'][-1].T
        up, down = (f[k][-1].T for k in ('C_exc_up_list', 'C_exc_dn_list'))
        density = (np.diag(up) + np.diag(down)).reshape(64, 2)
        sz = (np.diag(up) - np.diag(down)).reshape(64, 2) / 2
        n = density.mean(axis=1)
        spin = (sz[:, 0] - sz[:, 1]) / 2 * (-1.)**np.arange(64)
        rung = np.array([(pair[2*i, 2*i+1]+pair[2*i+1, 2*i])/2 for i in range(64)])
        leg = np.array([sum(pair[2*i+l, 2*(i+1)+l]+pair[2*(i+1)+l, 2*i+l] for l in (0, 1))/4 for i in range(63)])
        completed, records, mu = bool(f['completed'][()]), f['C_pair_list'].shape[0], float(f['mu'][()])
        # The file lacks a geometry label; compare its stored Hartree kernel.
        ep_rows = list(csv.DictReader((PROJECT/'data/E_p_values.csv').open()))
        ep_row = min((r for r in ep_rows if all(float(r[k]) == v for k,v in
            dict(L=64,U=8,V=.2,t0=1.2,density=.9375).items())), key=lambda r:(-int(r['chi']),float(r['rel_diff'])))
        g = .01/abs(float(ep_row['E_p']))
        discrepancies = {}
        for name, kernel in [('square',2*g*np.array([[0,1],[1,0]])),
            ('cubic_unfrustrated',6*g*np.array([[0,1],[1,0]])),
            ('cubic_frustrated',2*g*np.array([[2,1],[1,2]]))]:
            expected = np.array([((np.diag(a).reshape(64,2)-.5)@kernel.T).reshape(-1) for a in (down,up)]).T
            discrepancies[name] = float(np.max(abs(expected-f['mu_cdw'][()])))
    assert not completed and records == 60 and np.all(rung > 0) and np.all(leg < 0)
    bulk = slice(8,56)  # Rungs 9 through 56, both ends excluded symmetrically.
    nlo,nhi = np.quantile(n[bulk], [.05,.95])
    plo,phi = np.quantile(rung[bulk], [.05,.95])
    params = dict(charge_amplitude=float((nhi-nlo)/2),
                  spin_sz_amplitude=float(np.quantile(abs(spin[bulk]),.95)),
                  pair_rung_mean=float((phi+plo)/2), pair_rung_modulation=float((phi-plo)/2))
    coeffs = {k:[] for k in ('pair_same_leg','pair_cross_leg','exchange_same_leg','exchange_cross_leg')}
    for d in range(5):
        for cross in (False,True):
            ps, ns, envelope = [], [], []
            for i in range(8,56-d):
                for l in (0,1):
                    a,b = 2*i+l,2*(i+d)+(1-l if cross else l)
                    ps.append((pair[a,b]+pair[b,a])/2)
                    ns.append((up[a,b]+up[b,a]+down[a,b]+down[b,a])/4)
                    envelope.append((rung[i]+rung[i+d])/2)
            name = 'cross_leg' if cross else 'same_leg'
            coeffs['pair_'+name].append(float(np.dot(ps,envelope)/np.dot(envelope,envelope)))
            coeffs['exchange_'+name].append(float(np.mean(ns)))
    summary = dict(source=SOURCE.relative_to(PROJECT.parent).as_posix(), source_sha256=SOURCE_SHA,
        completed=completed, stored_records=records, mean_density=float(n.mean()), target_density=.9375,
        chemical_potential=mu, geometry='cubic_frustrated (inferred from stored Hartree kernel)',
        geometry_max_field_discrepancies=discrepancies,
        bulk_pair_hole_correlation=float(np.corrcoef(abs(rung[bulk]),1-n[bulk])[0,1]),
        bulk_pair_abs_spin_correlation=float(np.corrcoef(abs(rung[bulk]),abs(spin[bulk]))[0,1]),
        hole_peak_rungs=[i+1 for i in range(1,63) if n[i]<n[i-1] and n[i]<n[i+1]], **params)
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'legacy_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (REPORT/'legacy_profiles.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['rung','density','holes','staggered_sz_leg_odd','pair_rung','pair_leg_to_next'])
        writer.writerows((i+1,n[i],1-n[i],spin[i],rung[i],leg[i] if i<63 else '') for i in range(64))
    lines = ['# Legacy shape input only; not an accepted state or an MPS.',
        'schema_version = 1', f'source = "{summary["source"]}"', f'source_sha256 = "{SOURCE_SHA}"',
        'source_geometry = "cubic_frustrated"', 'source_geometry_evidence = "inferred from stored Hartree kernel"',
        'source_completed = false', f'source_records = {records}', 'L = 64', 'source_chi = 200',
        't0 = 1.2', 'V = 0.2', f'source_mean_density = {float(n.mean())!r}', 'target_density = 0.9375',
        'bulk_first_rung = 9', 'bulk_last_rung = 56',
        'definition = "5th/95th percentile amplitudes; translation-averaged relative bond coefficients; positions discarded"']
    lines += [f'{k} = {v!r}' for k,v in params.items()]
    lines += [f'{k} = {json.dumps(v)}' for k,v in coeffs.items()]
    RECIPE.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
    x=np.arange(1,65)
    axes[0].plot(x,1-n,color='#b45309',label='Hole density per site');axes[0].set_ylabel('1 - n')
    axes[1].plot(x,spin,color='#6b46a1');axes[1].axhline(0,color='.6',lw=.6);axes[1].set_ylabel('Staggered leg-odd Sz')
    axes[2].plot(x,rung,color='#2563eb',label='Rung singlet')
    axes[2].plot(x[:-1]+.5,leg,color='#2563eb',ls='--',label='Leg singlet');axes[2].set_ylabel('Pair amplitude');axes[2].legend()
    for ax in axes:
        for peak in summary['hole_peak_rungs']: ax.axvline(peak,color='.7',lw=.7,ls=':')
        ax.grid(alpha=.18)
    axes[-1].set_xlabel('Rung (leg pairing shown at bond midpoint)')
    fig.suptitle('Legacy (t0,V)=(1.2,+0.2), cubic frustrated; final stored profile\nIncomplete after 60 records; mean n=0.938803 versus target 0.9375',fontsize=12)
    fig.subplots_adjust(left=.13,right=.98,bottom=.09,top=.88,hspace=.15)
    fig.savefig(REPORT/'legacy_intertwined_profile.png',dpi=170);fig.savefig(REPORT/'legacy_intertwined_profile.pdf');plt.close(fig)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    print(json.dumps(summary,indent=2))
    print('Recipe SHA256:',hashlib.sha256(RECIPE.read_bytes()).hexdigest())


if __name__=='__main__':
    main()
