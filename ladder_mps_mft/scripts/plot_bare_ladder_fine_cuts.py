"""Bare-ladder measurements and explicitly labeled interpolation diagnostics."""
import csv
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reports/two_basin_fine_cuts_20260915'
CUTS = (
    dict(name='t0_1p4',axis='V',fixed='t0',value=1.4,bounds=(-.2,0.),targets=(-.15,-.1,-.05),title=r'$t_0/t=1.4$',xlabel=r'$V/t$'),
    dict(name='V_m0p4',axis='t0',fixed='V',value=-.4,bounds=(1.2,1.4),targets=(1.25,1.3,1.35),title=r'$V/t=-0.4$',xlabel=r'$t_0/t$'),
)


def prepare():
    source = ROOT / 'data/E_p_values.csv'
    original = source.read_bytes()
    with source.open(newline='') as stream:
        rows = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    data, measurements, estimates = {}, [], []
    for cut in CUTS:
        candidates = [r for r in rows if r['L']==64 and r['U']==8 and r['density']==.9375
                      and r[cut['fixed']]==cut['value']]
        x = np.array(sorted({r[cut['axis']] for r in candidates}))
        selected = [min([r for r in candidates if r[cut['axis']]==v],key=lambda r:(-r['chi'],r['rel_diff'])) for v in x]
        assert all(r['chi']==1000 for r in selected)
        en = np.array([r['E_N'] for r in selected]); ep = np.array([r['E_p'] for r in selected])
        lo,hi = cut['bounds']; lower,upper = np.flatnonzero(x==lo).item(),np.flatnonzero(x==hi).item()
        assert upper == lower+1 and lower>0 and upper<len(x)-1
        # Four neighboring measurements; diagnostic only, never used by MF runs.
        local = slice(lower-1,upper+2)
        cubic = np.polynomial.Polynomial.fit(x[local],ep[local],3)
        data[cut['name']] = dict(x=x,en=en,ep=ep,cubic=cubic)
        for row in selected:
            measurements.append(dict(cut=cut['name'],**row))
        for target in cut['targets']:
            weight=(target-lo)/(hi-lo)
            signed=(1-weight)*ep[lower]+weight*ep[upper]
            alternative=float(cubic(target))
            estimates.append(dict(cut=cut['name'],t0=target if cut['axis']=='t0' else cut['value'],
                V=target if cut['axis']=='V' else cut['value'],axis=cut['axis'],lower=lo,upper=hi,
                lower_ep=ep[lower],upper_ep=ep[upper],weight=weight,ep_signed=signed,
                ep_denominator=abs(signed),tp2_over_ep=.01/abs(signed),
                cubic_diagnostic_ep=alternative,cubic_vs_linear_coupling_fraction=abs(signed/alternative)-1))
        print(cut['name'],'E_N secant slopes:',np.diff(en)/np.diff(x))
    assert source.read_bytes()==original
    OUT.mkdir(parents=True,exist_ok=True)
    for filename,values in [('bare_ladder_measurements.csv',measurements),('interpolated_ep.csv',estimates)]:
        with (OUT/filename).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    (OUT/'bare_ladder_source_sha256.txt').write_text(hashlib.sha256(original).hexdigest()+'  data/E_p_values.csv\n')
    return data,estimates


def panel(ax,cut,d,metric,estimates):
    lo,hi=cut['bounds']
    ax.axvspan(lo,hi,color='#2563eb',alpha=.07)
    ax.plot(d['x'],d[metric],color='#999999',lw=.8,zorder=1)
    ax.scatter(d['x'],d[metric],color='#d97706',s=45,label='Bare-ladder measurements',zorder=4)
    if metric=='ep':
        selected=[r for r in estimates if r['cut']==cut['name']]
        tx=np.array([r[cut['axis']] for r in selected]); ty=np.array([r['ep_signed'] for r in selected])
        ax.plot([lo,hi],np.interp([lo,hi],d['x'],d['ep']),color='#2563eb',lw=1.4)
        ax.scatter(tx,ty,marker='D',facecolors='white',edgecolors='#2563eb',s=44,label='Proposed linear estimates',zorder=5)
        xx=np.linspace(lo,hi,100)
        ax.plot(xx,d['cubic'](xx),ls='--',color='#555555',lw=1.2,label='Cubic sensitivity guide')
    for target in cut['targets']:
        ax.axvline(target,color='#2563eb',lw=.6,alpha=.2)
    ax.set(xlabel=cut['xlabel'],ylabel=r'$E_0/t$ (whole ladder)' if metric=='en' else r'$E_p/t=-\Delta E_p/t$ (signed)')
    ax.set_title(('Bare ground-state energy; ' if metric=='en' else 'Bare pair binding; ')+cut['title'])
    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.grid(alpha=.18)
    if metric=='ep': ax.legend(fontsize=8,loc='best')


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)


def main():
    data,estimates=prepare()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8.8))
    for col,cut in enumerate(CUTS):
        for row,metric in enumerate(('en','ep')):
            panel(axes[row,col],cut,data[cut['name']],metric,estimates)
            single,ax=plt.subplots(figsize=(7,5))
            panel(ax,cut,data[cut['name']],metric,estimates)
            single.suptitle(r'$U/t=8$, $L=64$, $n=0.9375$, bare-ladder $\chi=1000$',fontsize=10)
            single.subplots_adjust(left=.16,right=.97,bottom=.14,top=.82)
            save(single,('ladder_E0s_U_8_' if metric=='en' else 'ladder_Eps_U_8_')+cut['name'])
    fig.suptitle(r'Bare-ladder cuts near the square-array boundary | $U/t=8$, $L=64$, $n=0.9375$, $\chi=1000$',fontsize=14)
    fig.subplots_adjust(left=.09,right=.98,top=.9,bottom=.11,wspace=.26,hspace=.32)
    fig.text(.09,.025,'Shading marks each finer-cut interval. No bare-ladder measurements exist at the six new points.\nThe cubic guide tests interpolation shape; it is not measured data or an uncertainty bound.',fontsize=10)
    save(fig,'bare_ladder_cuts')
    print('Saved bare-ladder overview, four separate plots, and source/interpolation tables.')


if __name__=='__main__':
    main()
