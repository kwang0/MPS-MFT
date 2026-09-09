"""Export stored MF energies per site; each Hamiltonian receives its own panel.

python -B scripts/plot_mf_energy_histories.py --out REPORT_DIR STATE_OR_RUN [...]
Directories contribute terminal state.h5 files. Pass a checkpoint explicitly
for an unfinished run. No acceptance or phase ranking is inferred from traces.
"""
import argparse
import csv
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_history(path):
    with h5py.File(path) as f:
        text = lambda key: f[key][()].decode()
        model = tuple(float(f["model/"+k][()]) for k in ("L", "U", "V", "t0", "tp", "density"))
        geometry = text("model/transverse_geometry")
        sites = 2 * model[0]
        history = f["history"]
        canonical = np.asarray(history["variational_energy"]) / sites
        if "target_density_corrected_variational_energy" in history:
            corrected = np.asarray(history["target_density_corrected_variational_energy"]) / sites
            correction_kind = "stored"
        else:
            corrected = canonical + np.asarray(history["chemical_potential"]) * (model[5] - np.asarray(history["density"]))
            correction_kind = "reconstructed_from_mu_and_density"
        iterations = np.asarray(history["iteration"])
        assert len(iterations) == len(canonical) == len(corrected)
        assert np.isfinite(canonical).all() and np.isfinite(corrected).all()
        branch = text("provenance/branch_label")
        status = text("status")
        rows = [dict(source=str(path), geometry=geometry, L=int(model[0]), U=model[1], V=model[2],
                     t0=model[3], tp=model[4], density_target=model[5], branch=branch, status=status,
                     stored_iteration=int(i), canonical_energy_per_site=float(e),
                     corrected_canonical_energy_per_site=float(ec),
                     correction_kind=correction_kind,
                     density=float(history["density"][j]),
                     global_relative_residual=float(history["field_rel_residual"][j]))
                for j, (i,e,ec) in enumerate(zip(iterations, canonical, corrected))]
        return (geometry, *model), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = set()
    for value in args.paths:
        path = Path(value).resolve()
        if not path.exists(): raise FileNotFoundError(path)
        sources.update(path.rglob("state.h5") if path.is_dir() else [path])
    if not sources: raise ValueError("No terminal states found; pass an existing checkpoint to plot an unfinished run.")
    groups = {}; rows = []
    for path in sorted(sources):
        key, history = read_history(path)
        groups.setdefault(key, []).append(history); rows.extend(history)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "energy_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    columns = min(3, len(groups)); height = (len(groups) + columns - 1) // columns
    fig, axes = plt.subplots(height, columns, figsize=(5*columns, 3.8*height), squeeze=False, layout="constrained")
    for ax, (key, histories) in zip(axes.flat, sorted(groups.items())):
        for history in histories:
            row = history[0]
            ax.plot([r["stored_iteration"] for r in history],
                    [r["corrected_canonical_energy_per_site"] for r in history],
                    marker=".", lw=1.3, label=f"{row['branch']} ({row['status']})")
        geometry,L,U,V,t0,tp,density = key
        ax.set_title(f"{geometry}: t0={t0:g}, V={V:g}, L={int(L)}", fontsize=11)
        ax.set_xlabel("Stored MF iteration (first solve is 1)")
        ax.set_ylabel("Corrected canonical energy / site [t]")
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(groups):]: ax.set_visible(False)
    fig.suptitle("Stored MF energy trajectories — iteration is not physical time", fontsize=13)
    fig.savefig(args.out / "energy_history.png", dpi=160)
    fig.savefig(args.out / "energy_history.pdf")
    plt.close(fig)
    print(f"Exported {len(rows)} records from {len(sources)} states to {args.out}")


if __name__ == "__main__": main()
