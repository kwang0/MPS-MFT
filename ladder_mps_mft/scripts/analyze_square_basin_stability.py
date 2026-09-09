"""Read-only follow-up of weak SDW growth and the selected V=0 jump.

Replays the existing Anderson algebra on stored applied/measured fields; no
DMRG or new map evaluation. Scientific static plots compare the same field
trajectories across six seeds. A channel gain is diagnostic, not a Jacobian
eigenvalue or an energy-Hessian test.
"""
import csv
import json
import tomllib

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compile_square_grid import REPO, REPORT, js, sha
from audit_scf_numerics import _julia_array
from audit_spatial_phase_defects import field_profiles


def load(row):
    path = REPO / row["source"]
    assert sha(path) == row["source_sha256"]
    with h5py.File(path) as f:
        h=f["history"]
        n=len(h["iteration"])
        fields={s:{k:_julia_array(h[f"fields/{s}/{k}"]) for k in ("alpha","beta","mu_cdw")}
                for s in ("applied","measured")}
        vectors={s:np.concatenate([np.asarray(h[f"fields/{s}/{k}"]).reshape(n,-1)
                                  for k in ("alpha","beta","mu_cdw")],axis=1) for s in fields}
        profiles={s:field_profiles(fields[s]["alpha"],fields[s]["mu_cdw"]) for s in fields}
        spin={s:((fields[s]["mu_cdw"][1]-fields[s]["mu_cdw"][0])/2).T for s in fields}
        rms=lambda a:np.sqrt(np.mean(a[:,5:59]**2,axis=1))
        rows=[]
        for i in range(n):
            x,y=spin["applied"][i],spin["measured"][i]
            sx2=float(x@x); sy2=float(y@y)
            rows.append(dict(stored_iteration=i+1,plotted_iteration=i+2,
                update_mode=h["update_mode"][i].decode(),
                spin_applied_bulk_rms=rms(profiles["applied"]["spin_odd"])[i],
                spin_measured_bulk_rms=rms(profiles["measured"]["spin_odd"])[i],
                pairing_measured_bulk_rms=rms(profiles["measured"]["pair_d"])[i],
                spin_projection_gain=float(x@y/sx2) if sx2>1e-28 else np.nan,
                spin_input_output_cosine=float(x@y/np.sqrt(sx2*sy2)) if sx2*sy2>1e-56 else np.nan,
                spin_channel_relative_residual=float(np.linalg.norm(y-x)/max(np.linalg.norm(x),np.linalg.norm(y),1e-30)),
                middle_sdw_measured=float(fields["measured"]["mu_cdw"][1,62,i]-fields["measured"]["mu_cdw"][0,62,i]),
                density_error=float(abs(h["density"][i]-.9375)),
                corrected_energy_per_site=float(h["target_density_corrected_variational_energy"][i]/128) if "target_density_corrected_variational_energy" in h else np.nan,
                global_relative_residual=float(h["field_rel_residual"][i])))
        for i in range(1,n):
            if rows[i]["update_mode"]=="unmixed_probe":
                np.testing.assert_allclose(vectors["applied"][i],vectors["measured"][i-1],rtol=0,atol=1e-12)
        final_dmrg={k:np.asarray(v).tolist() for k,v in h[f"dmrg/{n:04d}"].items()} if "dmrg" in h else {}
    return dict(row=row,fields=fields,vectors=vectors,profiles=profiles,records=rows,final_dmrg=final_dmrg)


def write_csv(name, rows):
    with (REPORT/name).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def replay(data):
    cfg=tomllib.loads((REPO/data["row"]["config_source"]).read_text())["mixing"]
    x,y=data["vectors"]["applied"],data["vectors"]["measured"]
    rows=data["records"]
    # Initial raw probe has no mixer history; stored 22 is the first linear input.
    first=next(i for i,r in enumerate(rows) if r["update_mode"]=="linear")-1
    damping=cfg["damping"]; previous=None; history=[]; evidence=[]
    for i in range(first,len(rows)-1):
        residual_norm=np.linalg.norm(y[i]-x[i])
        if previous is not None and cfg["adaptive"]:
            if residual_norm>1.25*previous:damping=max(cfg["minimum_damping"],.5*damping)
            elif residual_norm<.8*previous:damping=min(cfg["maximum_damping"],1.1*damping)
        previous=residual_norm
        history=(history+[i])[-(cfg["memory"]+1):]
        r=(y[history]-x[history]).T
        if len(history)==1:
            coeff=np.ones(1)
        else:
            gram=r.T@r+cfg["regularization"]*np.eye(len(history))
            kkt=np.block([[gram,np.ones((len(history),1))],[np.ones((1,len(history))),np.zeros((1,1))]])
            coeff=np.linalg.solve(kkt,np.r_[np.zeros(len(history)),1.])[:-1]
        prediction=coeff@((1-damping)*x[history]+damping*y[history])
        error=float(np.max(np.abs(prediction-x[i+1])))
        assert error<1e-10,(i,error)
        evidence.append(dict(from_stored_iteration=i+1,to_stored_iteration=i+2,
            to_plotted_iteration=i+3,source_stored_iterations=[j+1 for j in history],
            damping=damping,coefficients=coeff.tolist(),max_abs_replay_error=error))
    return evidence


def main():
    selection=json.loads((REPORT/"selection.json").read_text())
    data=[load(row) for row in selection["candidates"]]
    summaries=[]
    for d in data:
        row=d["row"];records=d["records"]
        raw=[r for r in records if r["update_mode"]=="unmixed_probe"]
        tail=raw[-min(5,len(raw)):]
        summaries.append(dict(t0=row["t0"],V=row["V"],branch=row["branch"],status=row["status"],
            selected=row["selected"],records=len(records),last_mode=records[-1]["update_mode"],
            last_raw_stored_iteration=raw[-1]["stored_iteration"] if raw else 0,
            raw_tail_spin_growth_ratio=tail[-1]["spin_measured_bulk_rms"]/tail[0]["spin_measured_bulk_rms"] if tail and tail[0]["spin_measured_bulk_rms"]>0 else np.nan,
            raw_tail_median_spin_gain=float(np.median([r["spin_projection_gain"] for r in tail])) if tail else np.nan,
            raw_tail_median_spin_cosine=float(np.median([r["spin_input_output_cosine"] for r in tail])) if tail else np.nan,
            final_spin_bulk_rms=records[-1]["spin_measured_bulk_rms"],
            final_pair_bulk_rms=records[-1]["pairing_measured_bulk_rms"],
            final_spin_gain=records[-1]["spin_projection_gain"],
            final_spin_relative_residual=records[-1]["spin_channel_relative_residual"],
            final_global_relative_residual=records[-1]["global_relative_residual"],source=row["source"],source_sha256=row["source_sha256"]))
    write_csv("basin_seed_summary.csv",summaries)
    chosen=next(d for d in data if d["row"]["t0"]==1.4 and d["row"]["V"]==0 and d["row"]["selected"])
    evidence=replay(chosen)
    write_csv("sdw_jump_history.csv",chosen["records"])
    spin=(chosen["fields"]["measured"]["mu_cdw"][1]-chosen["fields"]["measured"]["mu_cdw"][0])/2
    before,after=spin[:,21],spin[:,22]
    result=dict(source=chosen["row"]["source"],source_sha256=chosen["row"]["source_sha256"],
                iteration_note="Plotted iteration = stored iteration + 1 because the seed is displayed at 1.",
                jump_spatial_spin_cosine=float(before@after/(np.linalg.norm(before)*np.linalg.norm(after))),
                jump_bulk_spin_rms_ratio=chosen["records"][22]["spin_measured_bulk_rms"]/chosen["records"][21]["spin_measured_bulk_rms"],
                final_stored_history_screen=chosen["row"]["audit"],
                replay=evidence,final_dmrg=chosen["final_dmrg"])
    (REPORT/"anderson_replay.json").write_text(js(result)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout="constrained",sharey=True,sharex=True)
    v0=[d for d in data if d["row"]["V"]==0 and d["row"]["t0"]==1.4]
    for ax,d in zip(axes.flat,v0):
        rr=d["records"]; t=[r["plotted_iteration"] for r in rr]
        ax.semilogy(t,[r["spin_measured_bulk_rms"] for r in rr],"o-",color="#245A9C",ms=3,label="Measured spin Hartree")
        ax.semilogy(t,[r["spin_applied_bulk_rms"] for r in rr],"--",color="#B65F16",label="Applied spin Hartree")
        mixed=[r["plotted_iteration"] for r in rr if r["update_mode"] in ("linear","anderson")]
        if mixed:ax.axvline(mixed[0]-.5,color="0.6",ls=":")
        ax.set_title(d["row"]["branch"].removeprefix("square__").removesuffix("_chi200_loose"),fontsize=10)
        ax.grid(alpha=.2);ax.set_xlim(1,30)
    axes[0,0].legend(fontsize=8)
    for ax in axes[:,0]:ax.set_ylabel("Leg-odd spin Hartree RMS [t]")
    for ax in axes[1]:ax.set_xlabel("Plotted iteration (seed is 1)")
    fig.suptitle("Six small seeds at square (t0,V)=(1.4,0), L=64, chi=200\nRungs 6–59; dotted line marks first mixed input; stored accepted endpoints",fontsize=12)
    fig.savefig(REPORT/"six_seed_sdw_growth.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    rr=chosen["records"];t=np.array([r["plotted_iteration"] for r in rr])
    for key,label,color,style in (("spin_measured_bulk_rms","Measured","#245A9C","o-"),("spin_applied_bulk_rms","Applied","#B65F16","s--")):
        axes[0].plot(t,[r[key] for r in rr],style,color=color,ms=4,label=label)
    axes[0].set_ylabel("Leg-odd spin Hartree RMS [t]");axes[0].legend(fontsize=9)
    axes[1].plot(t,[r["spin_projection_gain"] for r in rr],"o-",color="#245A9C",ms=4)
    axes[1].axhline(1.,color="0.4",ls="--")
    axes[1].set_ylabel("Spin output/input projection gain")
    for ax in axes:
        ax.set_xlim(18,26);ax.set_xticks(range(18,27));ax.grid(alpha=.2);ax.axvline(23.5,color="0.5",ls=":")
        ax.set_xlabel("Plotted iteration (seed is 1)")
    fig.suptitle("Selected stripe_pairing_m004: SDW drop at plotted 23 → 24\nSquare (t0,V)=(1.4,0), L=64, chi=200; first Anderson input at 24",fontsize=12)
    fig.savefig(REPORT/"sdw_jump.png",dpi=170)
    plt.close(fig)
    print(js(dict(verified_sources=len(data),verified_mixed_transitions=len(evidence),
                  max_abs_replay_error=max(r["max_abs_replay_error"] for r in evidence),
                  jump_coefficients=evidence[1]["coefficients"],
                  jump_bulk_spin_rms_ratio=result["jump_bulk_spin_rms_ratio"],
                  output_directory=str(REPORT))))


if __name__=="__main__":main()
