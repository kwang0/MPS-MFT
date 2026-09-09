"""Analyze the flagged endpoint from the compiled grid; never changes a source.

Chart contract: scientific static PNG/PDF, 30 MF updates, common iteration
axis. Residual/channel plots use log scales; canonical-energy change is a
clearly labeled offset per 128 sites. Spatial profiles use signed values and
explicit iteration labels. Palette and lines distinguish channels/stages.
"""
import csv
import tomllib

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit_scf_numerics import _scalar, _julia_array
from audit_spatial_phase_defects import field_profiles
from compile_square_grid import OUT, REPORT, js


def main():
    bundle = OUT / "square_grid_chi200.h5"
    with h5py.File(bundle) as f:
        point = f["points/t010_vm04"]
        source = point["source_snapshot"]
        h = source["history"]
        cfg = tomllib.loads(_scalar(point, "config_toml"))
        it = np.asarray(h["iteration"])
        rel = np.asarray(h["field_rel_residual"])
        absolute = np.asarray(h["field_abs_residual"])
        energy = np.asarray(h["target_density_corrected_variational_energy"]) / 128
        density_error = np.abs(np.asarray(h["density"]) - .9375)
        modes = [x.decode() for x in h["update_mode"]]
        accelerated = np.array([m in ("linear", "anderson", "linear_fallback") for m in modes])
        start = int(np.flatnonzero(accelerated)[0])
        best = start + int(np.argmin(rel[start:]))
        threshold = cfg["convergence"]["divergence_factor"] * rel[best]
        assert rel[-1] > threshold and np.all(rel[start:-1] <= cfg["convergence"]["divergence_factor"] * np.minimum.accumulate(rel[start:-1]))
        measured = {k: _julia_array(h["fields/measured/"+k]) for k in ("alpha", "beta", "mu_cdw")}
        applied = {k: _julia_array(h["fields/applied/"+k]) for k in measured}
        squared = {k: np.sum((measured[k]-applied[k])**2, axis=tuple(range(measured[k].ndim-1))) for k in measured}
        total_sq = sum(squared.values())
        residuals = np.concatenate([(measured[k]-applied[k]).reshape(-1,len(it)) for k in measured],axis=0)
        slow = []
        for idx in (4,20,21,28,29):
            previous, current = residuals[:,idx-1], residuals[:,idx]
            cosine = np.dot(current,previous)/(np.linalg.norm(current)*np.linalg.norm(previous))
            contraction = np.dot(current,previous)/np.dot(previous,previous)
            factor = max(1.,1./max(1.-contraction,np.finfo(float).eps)) if cosine>=.9 else 1.
            if cosine>=.9 and contraction>=1.:factor=float("inf")
            slow.append(dict(iteration=int(it[idx]),cosine=float(cosine),contraction=float(contraction),
                             extrapolated_relative_residual=float(rel[idx]*factor),
                             energy_change_per_site=float(abs(energy[idx]-energy[idx-1]))))
        profiles = field_profiles(measured["alpha"], measured["mu_cdw"])
        bulk = slice(5, 59)
        rms = lambda a: np.sqrt(np.mean(a[:, bulk]**2, axis=1))
        channel_rms = {"pair_d": rms(profiles["pair_d"]), "spin_odd": rms(profiles["spin_odd"]),
                       "charge_modulation": rms(profiles["charge_even"] - np.mean(profiles["charge_even"][:, bulk], axis=1, keepdims=True))}
        field_norms = {k: np.sqrt(np.sum(v**2, axis=tuple(range(v.ndim-1)))) / 128 for k,v in measured.items()}
        dmrg = {k: np.asarray(v).tolist() if isinstance(v, h5py.Dataset) else list(v) for k,v in h["dmrg/0030"].items()}
        summary = dict(iterations=len(it), status=_scalar(source,"status"),
                       stop_reason=_scalar(source,"convergence_reason"),
                       accelerated_start_iteration=int(it[start]), accelerated_best_iteration=int(it[best]),
                       accelerated_best_residual=float(rel[best]), factor=cfg["convergence"]["divergence_factor"],
                       stop_threshold=float(threshold), final_relative_residual=float(rel[-1]),
                       final_to_accelerated_best_ratio=float(rel[-1]/rel[best]),
                       all_history_best_iteration=int(it[np.argmin(rel)]),
                       final_absolute_residual=float(absolute[-1]), final_density_error=float(density_error[-1]),
                       final_energy_per_site=float(energy[-1]), last_energy_change_per_site=float(energy[-1]-energy[-2]),
                       final_identity_error_per_site=_scalar(source,"hamiltonian_identity_error_per_site"),
                       final_effective_consistency_error_per_site=_scalar(source,"effective_eigenvalue_error_per_site"),
                       final_residual_squared_fractions={k:float(v[-1]/total_sq[-1]) for k,v in squared.items()},
                       final_field_norm_per_site={k:float(v[-1]) for k,v in field_norms.items()},
                       maximum_field_norm_per_site={k:float(np.max(v)) for k,v in field_norms.items()},
                       final_bulk_profile_rms={k:float(v[-1]) for k,v in channel_rms.items()},
                       final_dmrg=dmrg, final_maxlinkdim=int(h["dmrg_maxlinkdim"][-1]),
                       final_solve_max_discarded_weight=float(h["dmrg_max_discarded_weight"][-1]),
                       selected_slow_mode_diagnostics=slow,
                       final_dmrg_last_sweep_energy_change=float(abs(dmrg["sweep_energy"][-1]-dmrg["sweep_energy"][-2])),
                       periodic_solution=False)
        rows=[]
        for idx,n in enumerate(it):
            row=dict(iteration=int(n),update_mode=modes[idx],relative_residual=rel[idx],absolute_residual=absolute[idx],
                     energy_per_site=energy[idx],density_error=density_error[idx])
            row.update({k+"_rms":v[idx] for k,v in channel_rms.items()})
            row.update({k+"_residual_l2":np.sqrt(v[idx]) for k,v in squared.items()})
            rows.append(row)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/"divergence.json").write_text(js(summary)+"\n",encoding="utf-8")
    with (REPORT/"divergence_history.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, axes=plt.subplots(2,2,figsize=(11,7),layout="constrained",sharex=True)
    blue,orange,purple="#245A9C","#B65F16","#72569D"
    ax=axes[0,0]
    ax.semilogy(it,rel,"o-",color=blue,ms=3,label="Raw-map relative residual")
    ax.axhline(.005,color="0.25",ls=":",label="Relative field tolerance: 0.005")
    ax.hlines(threshold,it[start],it[-1],color=orange,ls="--",label=f"Stop threshold: {threshold:.5f}")
    ax.set_ylabel("Relative field residual"); ax.legend(fontsize=8)
    ax=axes[0,1]
    ax.plot(it,(energy-energy[-1])*1e3,"o-",color=blue,ms=3)
    ax.set_ylabel(r"$(E_{corr}-E_{corr,30})/N$ [$10^{-3}t$/site]")
    ax.set_title("Energy relative to terminal record",fontsize=10)
    ax=axes[1,0]
    for k,label,color,style in (("pair_d","d-wave proxy",blue,"-"),("spin_odd","Leg-odd spin Hartree",orange,"--"),("charge_modulation","Charge Hartree modulation",purple,":")):
        ax.semilogy(it,np.maximum(channel_rms[k],1e-16),style,color=color,label=label)
    ax.set_ylabel("Bulk profile RMS [t]"); ax.legend(fontsize=8)
    ax=axes[1,1]
    for k,label,color,style in (("alpha","Pairing alpha",blue,"-"),("beta","Exchange beta",purple,":"),("mu_cdw","Hartree",orange,"--")):
        ax.semilogy(it,np.maximum(np.sqrt(squared[k]),1e-16),style,color=color,label=label)
    ax.set_ylabel("Raw residual channel L2 norm [t]"); ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.axvline(21.5,color="0.6",ls="--",lw=1)
        ax.grid(alpha=.18); ax.set_xlim(1,30)
    for ax in axes[1]: ax.set_xlabel("MF iteration; acceleration starts at 22")
    fig.suptitle("Square (t0,V)=(1.0,-0.4): divergence diagnostic\nL=64, chi=200; 30 updates; profile RMS uses rungs 6–59",fontsize=12)
    for ext in ("png","pdf"):fig.savefig(REPORT/f"divergence_history.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),layout="constrained",sharex=True)
    for idx,color,style in ((4,"0.5",":"),(20,blue,"--"),(28,purple,"-."),(29,orange,"-")):
        axes[0].plot(np.arange(1,65),profiles["pair_d"][idx],color=color,ls=style,label=f"Iteration {idx+1}")
        axes[1].plot(np.arange(1,65),profiles["spin_odd"][idx]*(-1.)**np.arange(64),color=color,ls=style)
        axes[2].plot(np.arange(1,65),profiles["charge_even"][idx],color=color,ls=style)
    axes[0].set_ylabel("d-wave proxy [t]"); axes[0].legend(ncol=4,fontsize=8)
    axes[1].set_ylabel("Staggered leg-odd\nspin Hartree [t]")
    axes[2].set_ylabel("Leg-even charge\nHartree [t]");axes[2].set_xlabel("Rung")
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle("Spatial evolution of the unconverged endpoint\nMeasured fields; square (t0,V)=(1.0,-0.4), L=64, chi=200",fontsize=12)
    for ext in ("png","pdf"):fig.savefig(REPORT/f"divergence_profiles.{ext}",dpi=180)
    plt.close(fig)
    print(js(summary))
    print("Selected trajectory records:")
    for i in (4,10,11,16,20,21,24,25,28,29): print(js(rows[i]))


if __name__=="__main__":main()
