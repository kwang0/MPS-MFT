"""Render the locally prepared two-family seeds and export their amplitudes."""
import csv
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_scf_numerics import _julia_array
from audit_spatial_phase_defects import field_profiles
from compile_square_grid import sha

PROJECT = Path(__file__).resolve().parents[1]
PREVIEW = PROJECT / "output/seed_previews/20260908_square_two_basin/eps005_iter80/grid"
REPORT = PROJECT / "docs/reports/two_basin_raw_20260908"


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    with (PREVIEW / "manifest.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 18
    selected = {}
    summary = []
    for row in rows:
        path = Path(row["seed"])
        assert sha(path) == row["seed_sha256"]
        with h5py.File(path) as f:
            profiles = field_profiles(_julia_array(f["fields/restart/alpha"])[..., None],
                                      _julia_array(f["fields/restart/mu_cdw"])[..., None])
        profiles["charge_even_modulation"] = profiles["charge_even"] - profiles["charge_even"][:,5:59].mean(axis=1, keepdims=True)
        values = {key: np.asarray(profiles[key])[0] for key in ("pair_d", "charge_even_modulation", "spin_odd")}
        summary.append(dict(t0=row["t0"], V=row["V"], family=row["family"],
                            **{key + "_bulk_rms": np.sqrt(np.mean(a[5:59]**2)) for key,a in values.items()},
                            seed_sha256=row["seed_sha256"]))
        if float(row["t0"]) == 1.4 and float(row["V"]) == 0:
            selected[row["family"]] = values
    with (REPORT / "seed_amplitudes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0]); writer.writeheader(); writer.writerows(summary)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained", sharex=True)
    for r, family in enumerate(("stripe", "pairing")):
        for c, (key, label, color) in enumerate((("pair_d", "d-wave pairing proxy", "#245A9C"),
                ("charge_even_modulation", "Charge Hartree modulation", "#6A4C93"),
                ("spin_odd", "Leg-odd spin Hartree", "#B65F16"))):
            ax = axes[r,c]; values = selected[family][key]
            ax.plot(np.arange(1, len(values)+1), values, color=color, lw=1.3)
            ax.axhline(0, color="0.7", lw=.6); ax.grid(alpha=.2)
            ax.set_title(label + (" (weak)" if (r==0 and c==0) or (r==1 and c==2) else ""), fontsize=10)
            ax.set_ylabel(("Stripe + weak pairing" if r==0 else "Pairing + weak stripe") + "\nField [t]" if c==0 else "Field [t]")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2,2))
            if r==1: ax.set_xlabel("Rung")
    fig.suptitle("Prepared seeds at square (t0,V)=(1.4,0), L=64, chi=200\n95% primary + 5% competing correlations; separate y-scales expose weak channels", fontsize=12)
    fig.savefig(REPORT / "seed_profiles.png", dpi=160)
    fig.savefig(REPORT / "seed_profiles.pdf")
    plt.close(fig)
    print(f"Verified and summarized {len(rows)} seeds; figure: {REPORT / 'seed_profiles.png'}")


if __name__ == "__main__": main()
