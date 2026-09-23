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
from matplotlib.ticker import FormatStrFormatter, NullLocator, MaxNLocator

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
PAIRING_RMS_THRESHOLD = .01
UNIFORM_WEIGHT_THRESHOLD = .99
STRIPE_Q = {"charge": np.pi/8, "spin": 15*np.pi/16}


def stripe_correlations(f, density, spin):
    """Intraladder charge-even / longitudinal-spin-odd connected measures."""
    result = {}
    bulk = slice(8, 56)
    separation = abs(np.arange(48)[:, None] - np.arange(48)[None, :])
    for name, eta, expectation in (("charge", 1, density), ("spin", -1, spin)):
        site_expectation = f["density" if name == "charge" else "spin"][()]
        raw_site = f[name + "_correlation"][()].T
        conn_site = f[name + "_connected"][()].T
        np.testing.assert_allclose(raw_site - np.outer(site_expectation, site_expectation), conn_site,
                                   atol=2e-12, rtol=0)
        np.testing.assert_allclose(conn_site, conn_site.T.conj(), atol=2e-12, rtol=0)
        assert np.isfinite(conn_site).all()
        # n+ = (n0+n1)/2; m- = (Sz0-Sz1)/2. Keep this 1/4 normalization.
        rung = (conn_site[::2, ::2] + eta*conn_site[::2, 1::2]
                + eta*conn_site[1::2, ::2] + conn_site[1::2, 1::2])/4
        projected_raw = (raw_site[::2, ::2] + eta*raw_site[::2, 1::2]
                         + eta*raw_site[1::2, ::2] + raw_site[1::2, 1::2])/4
        np.testing.assert_allclose(projected_raw - np.outer(expectation, expectation), rung,
                                   atol=2e-12, rtol=0)
        matrix = rung[bulk, bulk].real
        for window, lo, hi in (("short", 2, 4), ("long", 16, 24)):
            mask = (separation >= lo) & (separation <= hi)
            result[f"{name}_{window}_connected_abs"] = float(np.mean(abs(matrix[mask])))
            result[f"{name}_{window}_connected_signed"] = float(np.mean(matrix[mask]))
        q = STRIPE_Q[name]
        weights = np.exp(1j*q*np.arange(8, 56))
        sf = float(np.real(weights.conj() @ matrix @ weights)/48)
        # Independent explicit sum checks Fourier sign and 1/N normalization.
        explicit = float(np.sum(matrix*np.cos(q*(np.arange(48)[:, None]-np.arange(48)[None, :])))/48)
        np.testing.assert_allclose(sf, explicit, atol=1e-13, rtol=1e-12)
        assert sf > 0
        result[f"{name}_stripe_q_connected"] = sf
        result[f"{name}_stripe_q_contact"] = float(np.trace(matrix)/48)
        result[f"{name}_stripe_q_offsite"] = sf - result[f"{name}_stripe_q_contact"]
    return result


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
            rung_pair = anomalous[sample["indices"]["rung"]][bulk]
            pair_uniform = float(abs(np.mean(rung_pair)))
            pair_uniform_fraction = float(pair_uniform**2/np.mean(abs(rung_pair)**2))
            high_pairing = point["rung_anomalous_rms"] > PAIRING_RMS_THRESHOLD
            if high_pairing:
                assert pair_uniform_fraction >= UNIFORM_WEIGHT_THRESHOLD, rid
            point.update(pair_uniform=pair_uniform,
                         pair_uniform_weight_fraction=pair_uniform_fraction,
                         state_group="uniform_pairing" if high_pairing else "stripe")
            point.update({key: summary[key] for key in METRICS})
            point.update(stripe_correlations(f, density, spin))
            point.update(diagnostic=src["path"], diagnostic_sha256=src["sha256"], source_sha256=src["source_sha256"])
            points.append(point)
            receipts.append(dict(id=rid, diagnostic=src["path"], sha256=src["sha256"], source_sha256=src["source_sha256"],
                                 pair_subtraction_max_error=float(np.max(abs(raw - np.outer(anomalous.conj(), anomalous) - connected)))))
    assert len(points) == 66 and len({p["id"] for p in points}) == 66
    assert {g: sum(p["geometry"] == g for p in points) for g in GEOMETRIES} == dict(square=42, cubic_unfrustrated=18, trellis=6)
    assert sum(p["accepted"] for p in points) == 8
    assert all(p["chi"] == 200 and all(p[k] > 0 for k in METRICS) for p in points)
    assert sum(p["state_group"] == "uniform_pairing" for p in points) == 24
    assert all(p["state_group"] == "uniform_pairing" for p in points if p["accepted"])
    return points, receipts


def plot_grid(points, xkey, xlabel, name):
    fig, axes = plt.subplots(3, 4, figsize=(14.8, 10.1), sharex=True, sharey="col")
    fig.subplots_adjust(left=.105, right=.91, top=.835, bottom=.155, hspace=.30, wspace=.22)
    norm = Normalize(vmin=-.4, vmax=.2)
    cmap = plt.get_cmap("viridis")
    xlo, xhi = min(p[xkey] for p in points), max(p[xkey] for p in points)
    xpad = (xhi-xlo)*.09
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
            ax.set_xlim(xlo-xpad, xhi+xpad)
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
                    if not subset:
                        continue
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
    handles = marker_handles(points)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.50, .935), ncol=7,
               frameon=False, fontsize=10, handletextpad=.4, columnspacing=1.5)
    fig.suptitle("Stripe states: stripe amplitude versus pair correlations", y=.99, fontsize=17)
    fig.text(.5, .954, "Uniformly paired states removed. Each mark is one spatial MPS; overlapping seeds/A/B ladders remain separate.",
             ha="center", fontsize=10)
    fig.text(.5, .090, r"Full: $|\langle D_i^\dagger D_j\rangle|$     Connected: $|\langle D_i^\dagger D_j\rangle-\langle D_i\rangle^*\langle D_j\rangle|$     $D$: unnormalized rung singlet",
             ha="center", fontsize=10, fontfamily="DejaVu Sans")
    fig.text(.5, .058, "Common bulk: rungs 9–56; L=64, χ=200, U=8, target n=15/16. Long-distance axes are logarithmic.",
             ha="center", fontsize=10)
    fig.text(.5, .026, "42 stripe-state endpoints, all unaccepted. Parameter associations; tiny tails have no independent error bound.",
             ha="center", fontsize=10, color="#555555")
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=170, facecolor="white")
    plt.close(fig)


def marker_handles(points):
    used = sorted({p["t0"] for p in points})
    handles = [Line2D([], [], linestyle="none", marker=MARKERS[t], color="#555555", markersize=6,
                      label=f"$t_0={t:g}$") for t in used]
    if any(p["accepted"] for p in points):
        handles.append(Line2D([], [], linestyle="none", marker="o", markerfacecolor="none", color="#111111", markersize=10,
                              label="Outer ring: accepted"))
    return handles


def paired_grid(points, *, wavevectors):
    if wavevectors:
        metrics = ("charge_stripe_q_connected", "spin_stripe_q_connected")
        titles = (r"Charge: $S_{n_+}^{\rm conn}(q_c=\pi/8)$",
                  r"Longitudinal spin: $S_{m_-}^{\rm conn}(q_s=15\pi/16)$")
        name = "paired_stripe_weights"
        size = (11.4, 8.0)
    else:
        metrics = ("charge_short_connected_abs", "charge_long_connected_abs",
                   "spin_short_connected_abs", "spin_long_connected_abs")
        titles = ("Charge · short\n$r=2$–$4$", "Charge · long\n$r=16$–$24$",
                  "Longitudinal spin · short\n$r=2$–$4$", "Longitudinal spin · long\n$r=16$–$24$")
        name = "paired_stripe_distance_correlations"
        size = (14.8, 8.0)
    geometries = [g for g in GEOMETRIES if any(p["geometry"] == g for p in points)]
    fig, axes = plt.subplots(len(geometries), len(metrics), figsize=size, squeeze=False)
    fig.subplots_adjust(left=.12, right=.9, top=.78, bottom=.24, hspace=.58, wspace=.4)
    norm, cmap = Normalize(vmin=-.4, vmax=.2), plt.get_cmap("viridis")
    xspan = max(p["pair_uniform"] for p in points if p["geometry"] == "square") - min(p["pair_uniform"] for p in points if p["geometry"] == "square")
    for row, geometry in enumerate(geometries):
        data = [p for p in points if p["geometry"] == geometry]
        xlo, xhi = min(p["pair_uniform"] for p in data), max(p["pair_uniform"] for p in data)
        # The trellis pair consists of almost identical seeds at ONE parameter;
        # keep a meaningful span rather than magnifying seed-level differences.
        xmin, xmax = (xlo+xhi)/2-xspan*.62, (xlo+xhi)/2+xspan*.62
        for col, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[row, col]
            for p in sorted(data, key=lambda p:p["accepted"]):
                ax.scatter(p["pair_uniform"], p[metric], c=[cmap(norm(p["V"]))],
                           marker=MARKERS[p["t0"]], s=55, alpha=.86, edgecolors="#333333", linewidths=.5, zorder=3)
                if p["accepted"]:
                    ax.scatter(p["pair_uniform"], p[metric], s=150, marker="o", facecolors="none", edgecolors="#111111", linewidths=1, zorder=4)
            ax.set_xlim(xmin, xmax)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            ylo, yhi = min(p[metric] for p in data), max(p[metric] for p in data)
            if not wavevectors and col % 2:
                ax.set_yscale("log")
                pad = max(.12, np.log10(yhi/ylo)*.13)
                ax.set_ylim(ylo/10**pad, yhi*10**pad)
                ax.set_yticks(np.geomspace(ylo/10**pad, yhi*10**pad, 3))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
                ax.yaxis.set_minor_locator(NullLocator())
            else:
                pad = max((yhi-ylo)*.16, (ylo+yhi)*.0125)
                ax.set_ylim(ylo-pad, yhi+pad)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10)
            ax.set_xlabel(r"Uniform pair amplitude $|\overline{\langle D_i\rangle}|$", fontsize=10)
            if row == 0:
                ax.set_title(title, fontsize=12, pad=11)
            if col == 0:
                ax.set_ylabel("Connected weight" if wavevectors else "Mean connected magnitude", fontsize=10)
            if geometry == "trellis":
                ax.text(.5, .94, "One coordinate\n(two overlapping seeds)", transform=ax.transAxes,
                        ha="center", va="top", fontsize=9, color="#555555", wrap=True)
        fig.text(.02, (axes[row,0].get_position().y0+axes[row,0].get_position().y1)/2,
                 f"{NAMES[geometry]}\n{len(data)} MPSs", rotation=90, ha="center", va="center", fontsize=12)
    cax = fig.add_axes([.935, .30, .014, .35])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap), cax=cax)
    cax.set_title("$V$", pad=8)
    bar.set_ticks(sorted({p["V"] for p in points}))
    fig.legend(handles=marker_handles(points), loc="upper center", bbox_to_anchor=(.5,.895), ncol=6,
               frameon=False, fontsize=10, columnspacing=1.5, handletextpad=.4)
    fig.suptitle("Uniformly paired states: stripe correlations versus pairing strength", y=.99, fontsize=16)
    fig.text(.5,.948,"24 paired MPSs: 22 square + 2 trellis. Cubic has no paired endpoints in this data set.",ha="center",fontsize=10)
    if wavevectors:
        definition = r"$S_O^{\rm conn}(q)=N_b^{-1}\sum_{ij}e^{iq(i-j)}[\langle O_iO_j\rangle-\langle O_i\rangle\langle O_j\rangle]$; fixed stripe wavevectors, contact terms included."
    else:
        definition = r"Mean $|\langle O_iO_j\rangle-\langle O_i\rangle\langle O_j\rangle|$ over the indicated separations; long-distance axes are logarithmic."
    fig.text(.5,.119,definition,ha="center",fontsize=10)
    fig.text(.5,.078,r"$n_+=(n_0+n_1)/2$, $m_-=(S^z_0-S^z_1)/2$; bulk rungs 9–56. Axis ranges differ by panel to show within-group variation.",ha="center",fontsize=10)
    fig.text(.5,.037,"16 unaccepted + 8 accepted MPSs. Equal-time intraladder correlations; parameter associations, not causal response functions.",ha="center",fontsize=10,color="#555555")
    for ext in ("png","pdf"):
        fig.savefig(OUT/f"{name}.{ext}",dpi=170,facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    points, receipts = load_points()
    stripe = [p for p in points if p["state_group"] == "stripe"]
    paired = [p for p in points if p["state_group"] == "uniform_pairing"]
    stripe_ids, paired_ids = {p["id"] for p in stripe}, {p["id"] for p in paired}
    assert not stripe_ids & paired_ids and stripe_ids | paired_ids == {p["id"] for p in points}
    assert {g:sum(p["geometry"] == g for p in paired) for g in GEOMETRIES} == dict(square=22,cubic_unfrustrated=0,trellis=2)
    for threshold in (.001, .01, .05):
        assert {p["id"] for p in points if p["rung_anomalous_rms"] > threshold} == paired_ids
    with (OUT / "points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    for name, subset in (("stripe_points.csv", stripe), ("paired_points.csv", paired)):
        with (OUT/name).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(points[0]));writer.writeheader();writer.writerows(subset)
    validation = dict(source_summary_sha256=sha(SOURCE / "correlation_summary.csv"),
        source_manifest_sha256=sha(SOURCE / "source_validation.json"),
        coupled_mps=66, source_branches=60, unaccepted_mps=58, accepted_mps=8,
        geometry_counts={g: sum(p["geometry"] == g for p in points) for g in GEOMETRIES},
        classification=dict(pairing_rms_threshold=PAIRING_RMS_THRESHOLD,
            minimum_uniform_weight=UNIFORM_WEIGHT_THRESHOLD, paired_mps=len(paired), stripe_mps=len(stripe),
            maximum_stripe_pair_rms=max(p["rung_anomalous_rms"] for p in stripe),
            minimum_paired_pair_rms=min(p["rung_anomalous_rms"] for p in paired),
            minimum_observed_paired_uniform_weight=min(p["pair_uniform_weight_fraction"] for p in paired),
            partition_unchanged_at_pair_rms_thresholds=[.001,.01,.05]),
        stripe_wavevectors={k:float(q) for k,q in STRIPE_Q.items()},
        excluded="Isolated chi=1200 reference; prospective runs without diagnostics are not included.",
        checks="Diagnostic SHA-256, source-state lineage, status, model labels, pair Hermiticity/subtraction and four recomputed pair measures; charge/spin connected subtraction, rung-channel projection and independent Fourier sum; complete disjoint 42/24 partition.",
        sources=receipts)
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2)+"\n", encoding="utf-8")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "pdf.fonttype":42})
    plot_grid(stripe, "charge_rms", "Charge-modulation RMS", "charge_pairing_grid")
    plot_grid(stripe, "spin_rms", "Spin-order RMS", "spin_pairing_grid")
    paired_grid(paired,wavevectors=True)
    paired_grid(paired,wavevectors=False)
    print(json.dumps({k:v for k,v in validation.items() if k != "sources"}, indent=2))
    print(f"Wrote four PNG/PDF grids and tables for {len(stripe)} stripe / {len(paired)} paired MPSs to {OUT}")


if __name__ == "__main__":
    main()
