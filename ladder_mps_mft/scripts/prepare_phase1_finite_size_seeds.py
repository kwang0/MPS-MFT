"""Extend the L=64 square terminal fields for a chi=200 length comparison.

Local preparation only. No MPS, source HDF5, scheduler or budget is modified.
Arrays retain Julia axis order internally. Every bond keeps its relative rung
offset; only its center coordinate is extended. Existing seed files may be
verified on rerun, but never overwritten with different fields.

Scientific figure contract: static profiles versus rung, identical physical
x/y scales across lengths; blue pairing and orange stripe lineages, with
distinct markers. Shading identifies the inserted bulk. These are applied
seed fields, not measurements from a longer-ladder calculation.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_RUN = PROJECT / "output/phase1_gpu/20260903_phase1_square_t014_v000_pairing_legacy_chi400_tight"
OUTPUT = PROJECT / "output/seed_previews/20260906_square_t014_v0_L96_L128_chi200"
FIGURES = PROJECT / "docs/reports/finite_size_seeds_20260906"
CONFIGS = PROJECT / "configs/phase1_gpu_square_size_compare_chi200"
SOURCE_HASHES = {
    "pairing": "bfcb03c7b3948a8b0f552fb45860b2fa352ea5a58feecc52ba4e215b7dfb52f1",
    "stripe": "229c118bec2f491997db0bf2be2da039863fca7695fdd0c44fdeadef1f2c76ab",
}
LABELS = {"pairing": "square__pairing_dwave_m000_chi400_tight",
          "stripe": "square__legacy_like_continuation_chi400_tight"}
METHOD = "fixed_edges_center_insert_v1"
PERIOD = 32
SOURCE_L = 64
R_RANGE = 4

sys.path.insert(0, str(PROJECT / "scripts"))
from audit_scf_numerics import _julia_array, _scalar
from audit_spatial_phase_defects import field_profiles


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def smoothstep(t):
    t = np.clip(t, 0., 1.)
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def extend_profile(values, centers, new_centers, target_L, lineage):
    """Extend all components of one fixed-separation bond profile together."""
    if target_L == SOURCE_L:
        np.testing.assert_array_equal(new_centers, centers)
        return values.copy()
    added = target_L - SOURCE_L
    if added <= 0 or added % PERIOD:
        raise ValueError("this construction requires a positive multiple of 32 added rungs")
    result = np.empty((len(new_centers),) + values.shape[1:])

    def at(c):
        index = int(round(c - centers[0]))
        if index < 0 or index >= len(centers) or abs(centers[index] - c) > 1e-12:
            raise ValueError(f"invalid source center {c}")
        return values[index]

    bulk_means = {
        parity: values[(centers >= 24.5) & (centers <= 40.5) &
                       (np.floor(centers).astype(int) % 2 == parity)].mean(axis=0)
        for parity in (0, 1)
    }
    for index, c in enumerate(new_centers):
        # Every field centered in the original left/right half is unchanged.
        if c <= 32.5:
            result[index] = at(c)
        elif c >= 32.5 + added:
            result[index] = at(c - added)
        elif lineage == "pairing":
            # Keep the source's two-sublattice structure. Taper only inside
            # the inserted segment, reaching an exactly constant bulk plateau.
            wl = 1 - smoothstep((c - 32.5) / 8)
            wr = 1 - smoothstep((32.5 + added - c) / 8)
            result[index] = (1 - wl - wr) * bulk_means[int(np.floor(c)) % 2]
            if wl > 0:
                result[index] += wl * at(c)
            if wr > 0:
                result[index] += wr * at(c - added)
        elif lineage == "stripe":
            # Periodize the central waveform by overlapping integer-shifted
            # copies. The seam blends source centers 12.5--20.5 with 44.5--52.5.
            # Positive quintic weights preserve amplitudes without overshoot.
            result[index] = 0
            total_weight = 0.
            for shift in range(-1, target_L // PERIOD + 1):
                s = c - PERIOD * shift
                if not 12.5 < s < 52.5:
                    continue
                weight = smoothstep((s - 12.5) / 8) * (1 - smoothstep((s - 44.5) / 8))
                result[index] += weight * at(s)
                total_weight += weight
            assert abs(total_weight - 1.) < 1e-12
        else:
            raise ValueError(lineage)
    return result


def extend_fields(source, target_L, lineage):
    alpha = np.zeros((target_L, target_L, 2, 2))
    beta = np.zeros((2, target_L, target_L, 2, 2))
    for offset in range(-R_RANGE, R_RANGE + 1):
        si = np.arange(max(0, -offset), min(SOURCE_L, SOURCE_L - offset))
        ti = np.arange(max(0, -offset), min(target_L, target_L - offset))
        sc, tc = si + 1 + offset / 2, ti + 1 + offset / 2
        alpha[ti, ti + offset] = extend_profile(source["alpha"][si, si + offset], sc, tc, target_L, lineage)
        for spin in (0, 1):
            beta[spin, ti, ti + offset] = extend_profile(source["beta"][spin, si, si + offset], sc, tc, target_L, lineage)
    hartree = source["mu_cdw"].T.reshape(SOURCE_L, 2, 2)
    mu = extend_profile(hartree, np.arange(1, 65), np.arange(1, target_L + 1), target_L, lineage)
    return dict(alpha=alpha, beta=beta, mu_cdw=mu.reshape(2 * target_L, 2).T)


def verify_fields(source, fields, L, lineage):
    added = L - 64
    for name, values in fields.items():
        assert np.isfinite(values).all(), name
        # Convex construction cannot manufacture a larger field component.
        assert np.max(np.abs(values)) <= np.max(np.abs(source[name])) + 1e-14, name
    np.testing.assert_array_equal(fields["mu_cdw"][:, :64], source["mu_cdw"][:, :64])
    np.testing.assert_array_equal(fields["mu_cdw"][:, -64:], source["mu_cdw"][:, -64:])
    for name, axes in (("alpha", (0, 1)), ("beta", (1, 2))):
        v, s = fields[name], source[name]
        if name == "alpha":
            np.testing.assert_array_equal(v[:32, :32], s[:32, :32])
            np.testing.assert_array_equal(v[-32:, -32:], s[-32:, -32:])
        else:
            np.testing.assert_array_equal(v[:, :32, :32], s[:, :32, :32])
            np.testing.assert_array_equal(v[:, -32:, -32:], s[:, -32:, -32:])
        far = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) > R_RANGE
        assert not np.any(v[far] if name == "alpha" else v[:, far]), name
    for spin in (0, 1):
        np.testing.assert_allclose(fields["beta"][spin], fields["beta"][spin].transpose(1, 0, 3, 2), atol=1e-14, rtol=0)
        for leg in (0, 1):
            assert not np.any(fields["beta"][spin, np.arange(L), np.arange(L), leg, leg])
    # Preserve the square-geometry zero cross-leg alpha/beta blocks.
    assert not np.any(fields["alpha"][:, :, 0, 1])
    assert not np.any(fields["alpha"][:, :, 1, 0])
    if lineage == "stripe" and added == 64:
        np.testing.assert_allclose(fields["mu_cdw"][:, 64:128], fields["mu_cdw"][:, 128:192], atol=1e-14, rtol=0)


def read_source(lineage):
    paths = list((SOURCE_RUN / "results" / LABELS[lineage]).rglob("state.h5"))
    assert len(paths) == 1
    path = paths[0]
    compact_hash = sha(path)
    manifest = path.parents[2] / "stateless_manifest.tsv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream, delimiter="\t") if r["relative_path"].endswith("/state.h5")]
    assert len(rows) == 1 and rows[0]["compact_sha256"] == compact_hash
    if SOURCE_HASHES[lineage]:
        assert compact_hash == SOURCE_HASHES[lineage]
    with h5py.File(path, "r") as f:
        assert _scalar(f, "model/L") == 64
        assert _scalar(f, "model/U") == 8 and _scalar(f, "model/V") == 0
        assert _scalar(f, "model/t0") == 1.4 and _scalar(f, "model/transverse_geometry") == "square"
        assert _scalar(f, "model/r_range") == R_RANGE
        fields = {name: _julia_array(f[f"fields/measured/{name}"]) for name in ("alpha", "beta", "mu_cdw")}
        metadata = dict(source_path=str(path.relative_to(PROJECT)).replace("\\", "/"),
                        source_compact_sha256=compact_hash,
                        source_full_sha256=_scalar(f, "analysis_storage/full_artifact_sha256"),
                        source_group="fields/measured", source_chi=400,
                        source_stored_status=_scalar(f, "status"),
                        source_stored_accepted=bool(_scalar(f, "accepted")),
                        chemical_potential=float(_scalar(f, "chemical_potential")),
                        signed_E_p=float(_scalar(f, "model/E_p_signed")))
    return fields, metadata


def save_seed(path, fields, metadata, L, lineage):
    if path.exists():
        with h5py.File(path, "r") as f:
            for name, value in fields.items():
                np.testing.assert_array_equal(_julia_array(f[f"fields/restart/{name}"]), value)
            assert _scalar(f, "seed_provenance/source_compact_sha256") == metadata["source_compact_sha256"]
            assert _scalar(f, "seed_provenance/method") == METHOD
        return
    with h5py.File(path, "x") as f:
        f["artifact_kind"] = np.bytes_("derived_field_seed")
        f["chemical_potential"] = metadata["chemical_potential"]
        f["model/L"] = L
        f["model/transverse_geometry"] = np.bytes_("square")
        for name, value in fields.items():
            f.create_dataset(f"fields/restart/{name}", data=value.transpose(tuple(reversed(range(value.ndim)))), compression="gzip", compression_opts=4, track_times=False)
        for name, value in dict(metadata, source_L=64, target_L=L, target_chi=200,
                                lineage=lineage, method=METHOD, added_spin_periods=(L-64)//32,
                                periodic_spin_envelope_rungs=32, charge_period_rungs=16,
                                target_density=0.9375, target_particle_number=int(2*L*0.9375),
                                exact_edge_half_rungs=32, relative_bond_range=R_RANGE,
                                fresh_mps=True, is_scf_solution=False).items():
            f["seed_provenance/" + name] = np.bytes_(value) if isinstance(value, str) else value


def dump_toml(raw):
    def value(v):
        return json.dumps(v, ensure_ascii=True)
    return "\n\n".join("[" + block + "]\n" + "\n".join(k + " = " + value(v) for k, v in settings.items()) for block, settings in raw.items()) + "\n"


def make_config(seed_path, seed_sha, L, lineage):
    with (PROJECT / "configs/phase1_gpu_square_v0_chi400_tight_compare.toml").open("rb") as stream:
        raw = tomllib.load(stream)
    label = f"square__{lineage}_L{L}_t014_v000_chi200"
    raw["model"]["L"] = L
    raw["pair_binding"]["reference_L"] = 64
    raw["dmrg"]["maxdim"] = 200
    raw["convergence"]["hamiltonian_identity_tol"] = 1e-8
    run = raw["run"]
    run.update(output_directory=f"output/phase1_gpu/UNPREPARED_SIZE_COMPARE/results/{label}",
               branch_label=lineage + "_length_control",
               preparation="square_t014_v0_fixed_coupling_length_comparison",
               direction="L64_to_L" + str(L), seed_label=METHOD + "_" + lineage,
               inherit_from=str(seed_path.relative_to(PROJECT)).replace("\\", "/"),
               inherit_sha256=seed_sha)
    path = CONFIGS / (label + ".toml")
    path.write_text("# Seed-review configuration; scratch output is assigned at campaign preparation.\n" + dump_toml(raw), encoding="utf-8")
    return path


def plot_and_export(all_fields):
    profiles = {key: {name: value[0] for name, value in field_profiles(fields["alpha"][..., None], fields["mu_cdw"][..., None]).items()}
                for key, fields in all_fields.items()}
    rows = []
    for (lineage, L), data in profiles.items():
        for i in range(L):
            rows.append(dict(lineage=lineage, L=L, rung=i+1,
                             inserted=32.5 < i+1 < L-31.5 if L > 64 else False,
                             pair_d=data["pair_d"][i], charge_even=data["charge_even"][i],
                             staggered_spin_odd=data["spin_odd"][i] * (-1.) ** i))
    with (FIGURES / "seed_profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(14, 8.6), sharex=True, sharey="col", layout="constrained")
    titles = ("Pairing field: leg-even minus rung", "Charge field: spin/leg average", "Spin field: staggered, leg-odd")
    for row, L in enumerate((64, 96, 128)):
        for lineage, color, marker in (("pairing", "#2563EB", "o"), ("stripe", "#D97706", "s")):
            p = profiles[lineage, L]
            for col, values in enumerate((p["pair_d"], p["charge_even"], p["spin_odd"] * (-1.) ** np.arange(L))):
                axes[row, col].plot(np.arange(1, L+1), values, color=color, marker=marker,
                                    markersize=2.2, markevery=4, label=lineage.capitalize())
        for col, ax in enumerate(axes[row]):
            if L > 64:
                ax.axvspan(32.5, L-31.5, color="#E5E7EB", alpha=0.65)
                ax.axvline(32.5, color="#9CA3AF", lw=0.6)
                ax.axvline(L-31.5, color="#9CA3AF", lw=0.6)
            ax.set_xlim(1, 128)
            ax.set_xticks((1, 32, 64, 96, 128))
            ax.grid(alpha=0.18)
            ax.set_ylabel(f"L={L}\nField [t]")
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            if row == 2:
                ax.set_xlabel("Rung (same horizontal scale in every panel)")
    axes[0, 0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Pairing and stripe seeds for the length comparison\nL=64 source fields; L=96/128 extensions at chi=200; shaded region is the inserted bulk", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / ("seed_snapshot." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    args = parser.parse_args()
    if args.source_run.resolve() != SOURCE_RUN.resolve():
        parser.error("this reviewed campaign is pinned to the declared L64 source run")
    for directory in (OUTPUT, FIGURES, CONFIGS):
        directory.mkdir(parents=True, exist_ok=True)
    all_fields, rows = {}, []
    for lineage in LABELS:
        source, metadata = read_source(lineage)
        all_fields[lineage, 64] = source
        for L in (96, 128):
            fields = extend_fields(source, L, lineage)
            verify_fields(source, fields, L, lineage)
            path = OUTPUT / f"{lineage}_L{L}_fields.h5"
            save_seed(path, fields, metadata, L, lineage)
            seed_sha = sha(path)
            config = make_config(path, seed_sha, L, lineage)
            all_fields[lineage, L] = fields
            rows.append(dict(lineage=lineage, L=L, chi=200, target_particles=int(2*L*.9375),
                             seed=str(path.relative_to(PROJECT)).replace("\\", "/"), sha256=seed_sha,
                             config=str(config.relative_to(PROJECT)).replace("\\", "/"),
                             source_compact_sha256=metadata["source_compact_sha256"],
                             source_full_sha256=metadata["source_full_sha256"],
                             signed_E_p=metadata["signed_E_p"], ep_reference_L=64,
                             max_abs_alpha=float(np.max(np.abs(fields["alpha"]))),
                             max_abs_beta=float(np.max(np.abs(fields["beta"]))),
                             max_abs_hartree=float(np.max(np.abs(fields["mu_cdw"]))),
                             edge_half_fields_exact=True, relative_range_preserved=True,
                             stage="seed_review_no_submission"))
    profiles = plot_and_export(all_fields)
    metrics = []
    for L in (64, 96, 128):
        p = profiles["stripe", L]
        charge = p["charge_even"]
        spin = p["spin_odd"] * (-1.) ** np.arange(L)
        troughs = np.flatnonzero((charge[1:-1] < charge[:-2]) & (charge[1:-1] < charge[2:])) + 2
        metrics.append(dict(L=L, charge_trough_rungs=troughs.tolist(),
                            charge_trough_count=len(troughs),
                            staggered_spin_zero_crossings=int(np.sum(spin[1:]*spin[:-1] < 0)),
                            max_charge_neighbor_step=float(np.max(np.abs(np.diff(charge)))),
                            max_spin_envelope_neighbor_step=float(np.max(np.abs(np.diff(spin))))))
    (FIGURES / "stripe_profile_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    (OUTPUT / "seed_manifest.json").write_text(json.dumps(dict(method=METHOD, branches=rows), indent=2)+"\n", encoding="utf-8")
    (FIGURES / "seed_manifest.json").write_text(json.dumps(dict(method=METHOD, branches=rows), indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(method=METHOD, branches=rows), indent=2))


if __name__ == "__main__":
    main()
