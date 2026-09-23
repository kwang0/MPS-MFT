"""Plot stripe-amplitude/pair-correlation associations in saved diagnostics.

Reuse the September 22 correlation definitions. Each mark is one spatial MPS;
seeds and A/B ladders are retained separately. No source state is modified.
Run from any directory with Python + numpy/h5py/scipy/matplotlib installed.
"""
from pathlib import Path
import csv
import hashlib
import json

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, NullLocator

from analyze_pair_correlations_20260922 import measure_summary

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "docs/reports/pair_correlations_20260922"
OUT = PROJECT / "docs/reports/stripe_pairing_grid_20260923"
GEOMETRIES = ("square", "cubic_unfrustrated", "trellis")
NAMES = {"square": "Square", "cubic_unfrustrated": "Cubic", "trellis": "Trellis"}
MARKERS = {1.0: "o", 1.2: "s", 1.25: "^", 1.3: "D", 1.35: "v", 1.4: "P"}
METRICS = ("short_raw_abs", "short_connected_abs", "long_raw_abs", "long_connected_abs")
TITLES = ("Short distance · full\n$r=2$–$4$", "Short distance · connected\n$r=2$–$4$",
          "Long distance · full\n$r=16$–$24$", "Long distance · connected\n$r=16$–$24$")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_points():
    old_rows = list(csv.DictReader((SOURCE / "correlation_summary.csv").open()))
    sources = {s["id"]: s for s in json.loads((SOURCE / "source_validation.json").read_text())["sources"]}
    points, receipts = [], []
    for old in old_rows:
        if old["geometry"] == "isolated":
            continue  # Its chi=1200 reference is not part of the coupled-state grid.
        rid, geometry = old["id"], old["geometry"]
        src = sources[rid]
        path = PROJECT / src["path"]
        assert sha(path) == src["sha256"], rid
        accepted = old["accepted"] == "True"
        with h5py.File(path) as f:
            assert bool(f["accepted"][()]) == accepted == src["accepted"]
            assert f["status"][()].decode() == old["status"] == src["status"]
            assert f["state_sha256"][()].decode() == src["source_sha256"]
            assert f["measurement_complete"][()] and f["full_pair_correlations"][()]
            assert f["measurement_version"][()].decode() == "equal_time_v2"
            assert f["L"][()] == 64 and f["U"][()] == 8
            assert abs(f["target_density"][()] - 15/16) < 1e-12
            assert f["geometry"][()].decode() == geometry
            for key in ("t0", "V"):
                assert float(f[key][()]) == float(old[key])
            pc = f["pair_correlations"]
            raw, connected = pc["removal"][()].T, pc["removal_connected"][()].T
            anomalous = pc["expectation"][()]
            np.testing.assert_allclose(raw, raw.T.conj(), atol=2e-12, rtol=0)
            np.testing.assert_allclose(raw - np.outer(anomalous.conj(), anomalous), connected, atol=2e-13, rtol=0)
            assert np.isfinite(raw).all() and np.isfinite(connected).all()
            labels = np.array([v.decode() for v in pc["basis_class"][()]])
            density = f["density"][()].reshape(64, 2).mean(axis=1)
            spin = (f["spin"][()][::2] - f["spin"][()][1::2]) / 2
            sample = dict(id=rid, geometry=geometry, t0=float(old["t0"]), V=float(old["V"]),
                chi=int(old["chi"]), accepted=accepted, status=old["status"], iteration=int(old["iteration"]),
                ladder=old["ladder"], label=old["label"], residual=float(old["residual"]),
                raw=raw, connected=connected, expectation=anomalous, spin=spin, density=density,
                indices={k: np.where(labels == k)[0] for k in ("rung", "leg0", "leg1")})
            summary = measure_summary(sample)
            for key in METRICS + ("rung_anomalous_rms",):
                np.testing.assert_allclose(summary[key], float(old[key]), atol=1e-14, rtol=1e-10)
            cell = "two_ladder" if rid.startswith("square_AB") or "two_ladder" in old["label"] else "one_ladder"
            bulk = slice(8, 56)
            point = {k: summary[k] for k in ("id", "geometry", "t0", "V", "chi", "accepted", "status", "iteration", "ladder", "label", "residual")}
            point.update(spatial_cell=cell,
                charge_rms=float(np.std(density[bulk])),
                spin_rms=float(np.sqrt(np.mean(spin[bulk]**2))),
                bulk_density=float(np.mean(density[bulk])),
                rung_anomalous_rms=summary["rung_anomalous_rms"])
            point.update({key: summary[key] for key in METRICS})
            point.update(diagnostic=src["path"], diagnostic_sha256=src["sha256"], source_sha256=src["source_sha256"])
            points.append(point)
            receipts.append(dict(id=rid, diagnostic=src["path"], sha256=src["sha256"], source_sha256=src["source_sha256"],
                                 pair_subtraction_max_error=float(np.max(abs(raw - np.outer(anomalous.conj(), anomalous) - connected)))))
    assert len(points) == 66 and len({p["id"] for p in points}) == 66
    assert {g: sum(p["geometry"] == g for p in points) for g in GEOMETRIES} == dict(square=42, cubic_unfrustrated=18, trellis=6)
    assert sum(p["accepted"] for p in points) == 8
    assert all(p["chi"] == 200 and all(p[k] > 0 for k in METRICS) for p in points)
    return points, receipts


def plot_grid(points, xkey, xlabel, name):
    fig, axes = plt.subplots(3, 4, figsize=(14.8, 10.1), sharex=True, sharey="col")
    fig.subplots_adjust(left=.105, right=.91, top=.835, bottom=.155, hspace=.30, wspace=.22)
    norm = Normalize(vmin=-.4, vmax=.2)
    cmap = plt.get_cmap("viridis")
    xmax = max(p[xkey] for p in points) * 1.09
    long_min = min(p[k] for p in points for k in METRICS[2:]) * .6
    long_max = max(p[k] for p in points for k in METRICS[2:]) * 1.8
    short_max = max(p[k] for p in points for k in METRICS[:2]) * 1.12
    for row, geometry in enumerate(GEOMETRIES):
        data = [p for p in points if p["geometry"] == geometry]
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            for point in sorted(data, key=lambda p: p["accepted"]):
                ax.scatter(point[xkey], point[metric], c=[cmap(norm(point["V"]))],
                    marker=MARKERS[point["t0"]], s=43, alpha=.84,
                    edgecolors="#363636", linewidths=.45, zorder=3)
                if point["accepted"]:
                    ax.scatter(point[xkey], point[metric], s=126, marker="o", facecolors="none",
                        edgecolors="#101010", linewidths=1.0, zorder=4)
            ax.set_xlim(-xmax*.025, xmax)
            if col < 2:
                ax.set_ylim(0, short_max)
                ax.set_yticks([0, .01, .02])
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            else:
                ax.set_yscale("log")
                ax.set_ylim(long_min, long_max)
                ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3])
                ax.yaxis.set_minor_locator(NullLocator())
            ax.grid(alpha=.17, linewidth=.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10, length=3)
            if row == 0:
                ax.set_title(TITLES[col], fontsize=12, pad=13)
            if col in (1, 3):
                ax.tick_params(labelleft=False)
            if geometry == "trellis":
                for cell, label in (("one_ladder", "1L"), ("two_ladder", "2L")):
                    subset = [p for p in data if p["spatial_cell"] == cell]
                    x = np.mean([p[xkey] for p in subset])
                    y = np.exp(np.mean(np.log([p[metric] for p in subset])))
                    ax.annotate(label, (x, y), xytext=(5, 7), textcoords="offset points", fontsize=9)
        fig.text(.025, (axes[row, 0].get_position().y0+axes[row, 0].get_position().y1)/2,
                 f"{NAMES[geometry]}\n{len(data)} MPSs", ha="center", va="center", fontsize=12, rotation=90)
    for col in (0,):
        for row in range(3):
            axes[row, col].set_ylabel("Mean pair magnitude", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=10)
    cax = fig.add_axes([.94, .24, .013, .49])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cax.set_title("$V$", fontsize=12, pad=9)
    colorbar.set_ticks(sorted({p["V"] for p in points}))
    handles = [Line2D([], [], linestyle="none", marker=m, color="#555555", markersize=6,
                      label=f"$t_0={t:g}$") for t, m in MARKERS.items()]
    handles.append(Line2D([], [], linestyle="none", marker="o", markerfacecolor="none", color="#111111", markersize=10,
                          label="Outer ring: accepted"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.50, .935), ncol=7,
               frameon=False, fontsize=10, handletextpad=.4, columnspacing=1.5)
    fig.suptitle("Stripe amplitude versus pair correlations across saved parameters", y=.99, fontsize=17)
    fig.text(.5, .954, "Each mark is one spatial MPS; seeds and A/B ladders remain separate (overlapping marks are retained).",
             ha="center", fontsize=10)
    fig.text(.5, .090, r"Full: $|\langle D_i^\dagger D_j\rangle|$     Connected: $|\langle D_i^\dagger D_j\rangle-\langle D_i\rangle^*\langle D_j\rangle|$     $D$: unnormalized rung singlet",
             ha="center", fontsize=10, fontfamily="DejaVu Sans")
    fig.text(.5, .058, "Common bulk: rungs 9–56; L=64, χ=200, U=8, target n=15/16. Long-distance axes are logarithmic.",
             ha="center", fontsize=10)
    fig.text(.5, .026, "58 unaccepted endpoints + 8 accepted square A/B MPSs. Parameter associations; tiny tails have no independent error bound.",
             ha="center", fontsize=10, color="#555555")
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    points, receipts = load_points()
    with (OUT / "points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    validation = dict(source_summary_sha256=sha(SOURCE / "correlation_summary.csv"),
        source_manifest_sha256=sha(SOURCE / "source_validation.json"),
        coupled_mps=66, source_branches=60, unaccepted_mps=58, accepted_mps=8,
        geometry_counts={g: sum(p["geometry"] == g for p in points) for g in GEOMETRIES},
        excluded="Isolated chi=1200 reference; prospective runs without diagnostics are not included.",
        checks="Diagnostic SHA-256, source-state lineage, status, model labels, pair Hermiticity, connected subtraction and four recomputed pair measures against the existing summary.",
        sources=receipts)
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2)+"\n", encoding="utf-8")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "pdf.fonttype":42})
    plot_grid(points, "charge_rms", "Charge-modulation RMS", "charge_pairing_grid")
    plot_grid(points, "spin_rms", "Spin-order RMS", "spin_pairing_grid")
    print(json.dumps({k:v for k,v in validation.items() if k != "sources"}, indent=2))
    print(f"Wrote two PNG/PDF grids and a {len(points)}-row table to {OUT}")


if __name__ == "__main__":
    main()
