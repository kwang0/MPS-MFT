"""Show the four locally prepared templates; these are initial fields, not results."""
import csv
import hashlib
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT/'output/seed_previews/20260915_square_positive_v/control'
OUT = ROOT/'docs/reports/square_positive_v_seeds_20260915'


def main():
    with (CONTROL/'manifest.tsv').open(newline='') as stream:
        rows=list(csv.DictReader(stream,delimiter='\t'))
    assert len(rows)==4
    receipt_keys=('label','family','charge_wavelength','spin_envelope_wavelength','ep_mode','ep_signed',
        'model_fingerprint','numerical_fingerprint','implementation_sha256','ep_source_sha256','seed_sha256')
    with (OUT/'prepared_branches.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=receipt_keys);writer.writeheader()
        writer.writerows({k:row[k] for k in receipt_keys} for row in rows)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,3,figsize=(13,10),sharex=True)
    labels=['95% stripe + 5% pairing','95% pairing + 5% stripe',
            'Intertwined: charge/pair period 8','Intertwined: charge/pair period 16']
    profiles=[]
    for j,row in enumerate(rows):
        source=Path(row['seed'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row['seed_sha256']
        with h5py.File(source) as f:
            c=f['template_correlations']
            n=(c['density_down'][()]+c['density_up'][()]).reshape(64,2).mean(axis=1)
            sz=(c['density_up'][()]-c['density_down'][()]).reshape(64,2)/2
            spin=(sz[:,0]-sz[:,1])/2*(-1.)**np.arange(64)
            pair=c['pair'][()].T
            rung=np.array([(pair[2*i,2*i+1]+pair[2*i+1,2*i])/2 for i in range(64)])
            leg=np.array([sum(pair[2*i+l,2*(i+1)+l]+pair[2*(i+1)+l,2*i+l] for l in (0,1))/4 for i in range(63)])
            alpha=f['fields/restart/alpha'][()].transpose(3,2,1,0)
            assert np.all(alpha[:,:,0,1]==0) and np.all(alpha[:,:,1,0]==0)
        x=np.arange(1,65)
        axes[j,0].plot(x,1-n,color='#b45309');axes[j,0].axhline(.0625,color='.6',lw=.6,ls='--')
        axes[j,1].plot(x,spin,color='#6b46a1');axes[j,1].axhline(0,color='.6',lw=.6)
        axes[j,2].plot(x,rung,color='#2563eb',label='Rung')
        axes[j,2].plot(x[:-1]+.5,leg,color='#2563eb',ls='--',label='Leg')
        axes[j,2].axhline(0,color='.6',lw=.6)
        for i,title in enumerate(('Hole density 1 - n','Staggered leg-odd Sz','Singlet pair amplitude')):
            axes[j,i].set_title(labels[j]+'\n'+title,fontsize=9)
            axes[j,i].grid(alpha=.18)
            axes[j,i].ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
            if j==3: axes[j,i].set_xlabel('Rung')
        for i in range(64):
            profiles.append(dict(family=row['family'],rung=i+1,holes=1-n[i],staggered_sz=spin[i],
                pair_rung=rung[i],pair_leg_to_next=leg[i] if i<63 else ''))
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row['seed_sha256']
    axes[0,2].legend(loc='best',fontsize=8)
    fig.suptitle('Four prepared seeds for square (t0,V)=(1.2,+0.2), L=64, chi=200\nTemplate correlations; fields rebuilt with square couplings; no seed MPS or persistent pinning',fontsize=12)
    fig.subplots_adjust(left=.07,right=.98,bottom=.06,top=.89,wspace=.28,hspace=.60)
    fig.savefig(OUT/'four_seed_profiles.png',dpi=170);fig.savefig(OUT/'four_seed_profiles.pdf');plt.close(fig)
    with (OUT/'seed_profiles.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(profiles[0]));writer.writeheader();writer.writerows(profiles)
    print('Verified and plotted all four prepared seeds.')


if __name__=='__main__':
    main()
