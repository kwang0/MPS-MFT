"""Read-only analysis of the synchronized square chi=400 two-lineage run.

Run locally: python -B -X utf8 ladder_mps_mft/docs/reports/chi400_comparison_20260905/analyze.py
Reuses the existing SCF audit and spatial-profile definitions. Never invokes
the accepted-only branch ranker for these unaccepted endpoints.

Chart contract: static scientific line figures, saved PNG and PDF. Energy and
residual histories have 32/40 records; spatial profiles have 64 rungs. Blue/
orange identify the two lineages; markers and dashed parent curves provide
non-color distinctions. All energy values are per 128 physical sites. The
energy chart uses a labeled offset, not a zero-baseline magnitude comparison.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tomllib

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = PROJECT / "output/phase1_gpu"
RUN = ROOT / "20260903_phase1_square_t014_v000_pairing_legacy_chi400_tight"
sys.path.insert(0, str(PROJECT / "scripts"))
from audit_scf_numerics import _audit_state, _history_fields, _scalar, _julia_array
from audit_spatial_phase_defects import field_profiles

LABELS = {
    "pairing": "square__pairing_dwave_m000_chi400_tight",
    "legacy": "square__legacy_like_continuation_chi400_tight",
}
COLORS = {"pairing": "#2563EB", "legacy": "#D97706"}
NAMES = {"pairing": "Pairing lineage", "legacy": "Legacy-like lineage"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (HERE / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def verify_manifest(branch):
    checks = []
    with (branch / "stateless_manifest.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            path = branch / row["relative_path"]
            checked = dict(branch=branch.name, path=str(path.relative_to(PROJECT)),
                           compact_sha256=sha(path), full_sha256=row["full_sha256"],
                           size_match=path.stat().st_size == int(row["compact_bytes"]))
            checked["hash_match"] = checked["compact_sha256"] == row["compact_sha256"]
            if row["kind"] == "stateless_hdf5":
                with h5py.File(path, "r") as f:
                    names = []
                    f.visit(names.append)
                    checked["no_mps"] = not any(re.fullmatch(r"psi(?:_N_[0-9]+)?", n.split("/")[-1]) for n in names)
                    checked["stateless"] = bool(_scalar(f, "analysis_storage/is_stateless_copy"))
                    checked["full_hash_metadata_match"] = _scalar(f, "analysis_storage/full_artifact_sha256") == row["full_sha256"]
            assert all(v for k, v in checked.items() if isinstance(v, bool)), checked
            checks.append(checked)
    return checks


def profiles(f):
    alpha = _julia_array(f["fields/measured/alpha"])[..., None]
    hartree = _julia_array(f["fields/measured/mu_cdw"])[..., None]
    result = {k: v[0] for k, v in field_profiles(alpha, hartree).items()}
    down = np.asarray(f["correlations/density_down"])
    up = np.asarray(f["correlations/density_up"])
    result["density"] = ((down + up)[0::2] + (down + up)[1::2]) / 2
    sz = (up - down) / 2
    result["staggered_spin_odd"] = (sz[0::2] - sz[1::2]) / 2 * (-1.) ** np.arange(len(down) // 2)
    return result


def profile_metrics(f, p):
    bulk = slice(16, 48)  # Central half: one-based rungs 17--48.
    down, up = np.asarray(f["correlations/density_down"]), np.asarray(f["correlations/density_up"])
    return {
        "max_abs_alpha": float(np.max(np.abs(f["fields/measured/alpha"]))),
        "max_abs_beta": float(np.max(np.abs(f["fields/measured/beta"]))),
        "max_abs_hartree": float(np.max(np.abs(f["fields/measured/mu_cdw"]))),
        "bulk_pair_d_mean": float(np.mean(p["pair_d"][bulk])),
        "bulk_pair_d_rms": float(np.sqrt(np.mean(p["pair_d"][bulk] ** 2))),
        "bulk_pair_rung_mean": float(np.mean(p["pair_rung"][bulk])),
        "bulk_pair_leg_mean": float(np.mean(p["pair_leg_even"][bulk])),
        "bulk_density_mean": float(np.mean(p["density"][bulk])),
        "bulk_density_peak_to_peak": float(np.ptp(p["density"][bulk])),
        "bulk_density_std": float(np.std(p["density"][bulk])),
        "bulk_spin_odd_rms": float(np.sqrt(np.mean(p["staggered_spin_odd"][bulk] ** 2))),
        "bulk_max_abs_site_Sz": float(np.max(np.abs((up - down)[32:96] / 2))),
    }


def load_branch(key):
    label = LABELS[key]
    paths = list((RUN / "results" / label).rglob("state.h5"))
    assert len(paths) == 1, paths
    path = paths[0]
    config_path = RUN / "configs" / f"{label}.segment-001.toml"
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    gate = config["convergence"]
    audit = _audit_state(RUN, path)
    with h5py.File(path, "r") as f:
        sites = int(_scalar(f, "model/L")) * 2
        h = f["history"]
        iterations = np.asarray(h["iteration"])
        energy = np.asarray(h["target_density_corrected_variational_energy"])
        density = np.asarray(h["density"])
        target = float(_scalar(f, "model/density"))
        mu = float(_scalar(f, "chemical_potential"))
        particle_number = float(np.sum(f["correlations/density_down"]) + np.sum(f["correlations/density_up"]))
        energies = {k: float(v[()]) for k, v in f["energy"].items()}
        hartree = _julia_array(f["fields/applied/mu_cdw"])
        down = np.asarray(f["correlations/density_down"])
        up = np.asarray(f["correlations/density_up"])
        energies["density_charge_component"] = float(np.sum((hartree[0] + hartree[1]) * (down + up - 1)) / 4)
        energies["density_spin_component"] = float(np.sum((hartree[1] - hartree[0]) * (up - down)) / 4)
        assert abs(energies["density_charge_component"] + energies["density_spin_component"] - energies["density_transverse_energy"]) < 1e-12
        recomputed = energies["canonical_variational_energy"] + mu * (sites * target - particle_number)
        assert abs(recomputed - energies["target_density_corrected_variational_energy"]) < 1e-11
        assert abs(recomputed - energy[-1]) < 1e-11
        assert abs(sum(energies[k] for k in ("bare_ladder_energy", "pair_transverse_energy", "exchange_transverse_energy", "density_transverse_energy")) - energies["canonical_variational_energy"]) < 1e-11
        assert abs(particle_number / sites - density[-1]) < 1e-12
        for k, value in [("fixed_point_rel_residual", audit["raw_rel_residual"]), ("fixed_point_extrapolated_rel_residual", audit["extrapolated_rel_residual"])]:
            assert np.isclose(_scalar(f, k), value, rtol=1e-9, atol=1e-13)
        row = dict(lineage=key, state_path=str(path.relative_to(PROJECT)),
                   status=_scalar(f, "status"), accepted=bool(_scalar(f, "accepted")),
                   solution_kind=_scalar(f, "solution_kind"), period=int(_scalar(f, "fundamental_period")),
                   job_id=_scalar(f, "provenance/slurm_job_id"), iterations=len(iterations), sites=sites,
                   generated_utc=_scalar(f, "provenance/generated_utc"),
                   density=density[-1], density_error=abs(density[-1] - target), mu=mu,
                   canonical_energy=energies["canonical_variational_energy"], corrected_energy=recomputed,
                   canonical_per_site=energies["canonical_variational_energy"] / sites,
                   corrected_per_site=recomputed / sites,
                   density_correction_per_site=energies["target_density_correction"] / sites,
                   last_energy_step_per_site=abs(energy[-1] - energy[-2]) / sites,
                   last5_energy_range_per_site=float(np.ptp(energy[-5:])) / sites,
                   last10_energy_range_per_site=float(np.ptp(energy[-10:])) / sites,
                   last10_max_energy_step_per_site=float(np.max(np.abs(np.diff(energy[-10:])))) / sites,
                   identity_error_per_site=float(_scalar(f, "hamiltonian_identity_error_per_site")),
                   effective_eigenvalue_error_per_site=float(_scalar(f, "effective_eigenvalue_error_per_site")),
                   raw_abs_residual=audit["raw_abs_residual"], raw_rel_residual=audit["raw_rel_residual"],
                   extrapolated_rel_residual=audit["extrapolated_rel_residual"],
                   slow_mode_lambda=audit["slow_mode_lambda"], slow_mode_cosine=audit["residual_cosine"],
                   wall_hours=float(np.sum(h["wall_seconds"])) / 3600.,
                   mu_search_status=_scalar(h, "mu_search_status")[-1].decode(),
                   full_artifact_sha256=_scalar(f, "analysis_storage/full_artifact_sha256"))
        row["raw_last_two_pass"] = bool(np.all((np.asarray(h["field_abs_residual"])[-2:] <= gate["field_abs_tol"]) | (np.asarray(h["field_rel_residual"])[-2:] <= gate["field_rel_tol"])))
        row["slow_mode_pass"] = bool(audit["extrapolated_gate_pass"])
        row["density_last_two_pass"] = bool(np.all(np.abs(density[-2:] - target) <= gate["density_tol"]))
        row["energy_stability_pass"] = row["last_energy_step_per_site"] <= gate["variational_energy_tol"]
        row["identity_pass"] = row["identity_error_per_site"] <= gate["hamiltonian_identity_tol"]
        row["effective_consistency_pass"] = row["effective_eigenvalue_error_per_site"] <= gate["effective_energy_consistency_tol"]
        fingerprints = {k: _scalar(f, "provenance/" + k) for k in ("model_fingerprint", "numerical_fingerprint", "implementation_sha256", "tree_sha256", "gpu_manifest_sha256", "ep_source_sha256", "tensor_scalar_type")}
        assert sha(config_path) == _scalar(f, "provenance/config_sha256")
        assert sha(RUN / "gpu-Manifest.toml") == fingerprints["gpu_manifest_sha256"]
        row.update(fingerprints)
        p = profiles(f)
        row.update(profile_metrics(f, p))
        applied = _history_fields(f, "applied")
        measured = _history_fields(f, "measured")
        vectors = lambda data: np.concatenate([v.reshape(-1, len(iterations)) for v in data.values()], axis=0)
        residuals = vectors(measured) - vectors(applied)
        raw_rel = np.asarray(h["field_rel_residual"])
        extrapolated = raw_rel.copy()
        for i in range(1, len(iterations)):
            prev, cur = residuals[:, i-1], residuals[:, i]
            cos = np.dot(prev, cur) / (np.linalg.norm(prev) * np.linalg.norm(cur))
            lam = np.dot(prev, cur) / np.dot(prev, prev)
            if cos >= gate["slow_mode_cosine_min"]:
                extrapolated[i] *= math.inf if lam >= 1 else max(1., 1 / (1 - lam))
        hist_profiles = field_profiles(measured["alpha"], measured["mu_cdw"])
        history = []
        for i, iteration in enumerate(iterations):
            dmrg = h[f"dmrg/{iteration:04d}"]
            sweeps = np.asarray(dmrg["sweep_energy"])
            item = dict(lineage=key, iteration=int(iteration),
                        mode=h["update_mode"][i].decode(), density=density[i], mu=h["chemical_potential"][i],
                        canonical_per_site=h["variational_energy"][i] / sites,
                        corrected_per_site=energy[i] / sites, raw_rel=raw_rel[i], raw_abs=h["field_abs_residual"][i],
                        extrapolated_rel=extrapolated[i], dmrg_sweeps=len(sweeps),
                        last_sweep_change=abs(sweeps[-1] - sweeps[-2]) if len(sweeps) > 1 else math.nan,
                        last_sweep_discarded_weight=np.asarray(dmrg["sweep_max_discarded_weight"])[-1],
                        dmrg_max_discarded_weight=h["dmrg_max_discarded_weight"][i],
                        maxlinkdim=int(h["dmrg_maxlinkdim"][i]),
                        cumulative_hours=float(np.sum(h["wall_seconds"][:i+1])) / 3600,
                        pair_d_bulk_rms=float(np.sqrt(np.mean(hist_profiles["pair_d"][i, 16:48] ** 2))),
                        spin_odd_field_bulk_rms=float(np.sqrt(np.mean(hist_profiles["spin_odd"][i, 16:48] ** 2))))
            history.append(item)
        for k in ("dmrg_sweeps", "last_sweep_change", "last_sweep_discarded_weight", "dmrg_max_discarded_weight", "maxlinkdim"):
            row[k] = history[-1][k]
        row["last_sweep_change_pass"] = row["last_sweep_change"] <= config["dmrg"]["energy_tol"]
        parent_path = ROOT / _scalar(f, "provenance/parent_checkpoint").split("/phase1_gpu/", 1)[1]
        parent_hash = _scalar(f, "provenance/parent_sha256")
    with h5py.File(parent_path, "r") as f:
        assert _scalar(f, "analysis_storage/full_artifact_sha256") == parent_hash
        pp = profiles(f)
        parent = dict(lineage=key, role="chi200_parent", state_path=str(parent_path.relative_to(PROJECT)),
                      status=_scalar(f, "status"), accepted=bool(_scalar(f, "accepted")),
                      recorded_full_sha256=parent_hash, compact_sha256=sha(parent_path),
                      corrected_per_site=float(_scalar(f, "energy/target_density_corrected_variational_energy")) / sites)
        parent.update(profile_metrics(f, pp))
    return row, history, energies, p, parent, pp, audit


def make_plots(results):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    fig.suptitle("Square chi=400: energy and SCF convergence\nL=64, t0=1.4, V=0, target n=0.9375; both endpoints unaccepted", fontsize=13)
    reference = results["pairing"][0]["corrected_per_site"]
    for key, (_, history, _, _, _, _, _) in results.items():
        x = [r["iteration"] for r in history]
        marker = "o" if key == "pairing" else "s"
        axes[0].plot(x, [(r["corrected_per_site"] - reference) * 1e4 for r in history],
                     color=COLORS[key], marker=marker, markersize=3, label=NAMES[key])
        axes[1].semilogy(x, [r["raw_rel"] for r in history], color=COLORS[key],
                         marker=marker, markersize=3, label=NAMES[key] + ": raw")
        finite = [r["extrapolated_rel"] if math.isfinite(r["extrapolated_rel"]) else math.nan for r in history]
        axes[1].semilogy(x, finite, color=COLORS[key], linestyle="--", alpha=0.65)
    axes[0].set(ylabel=r"$(E_{target}/N_s-e_{pair,final})\;[10^{-4}t]$", xlabel="SCF record", title="Target-density-corrected canonical energy")
    axes[0].axhline(0, color="#6B7280", lw=0.8)
    axes[0].legend(fontsize=9)
    axes[1].axhline(1e-4, color="#20242A", linestyle=":", label="Relative field gate")
    axes[1].set(ylabel="Relative field residual", xlabel="SCF record", title="Raw and slow-mode residuals (dashed)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axvline(21.5, color="#9CA3AF", linestyle=":", lw=0.8)
        ax.grid(alpha=0.2)
    fig.text(0.01, -0.015, "Vertical guide: end of initial raw-map probe. Infinite slow-mode estimates are gaps; these are diagnostics, not accepted-solution energies.", fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(HERE / f"energy_convergence.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.5), sharex=True, layout="constrained")
    fig.suptitle("Square two-lineage spatial profiles\nSolid: chi=400 terminal; dashed: chi=200 parent; shaded edges excluded from bulk metrics", fontsize=13)
    for key, (_, _, _, p, _, pp, _) in results.items():
        for ax, quantity in zip(axes, ("density", "staggered_spin_odd", "pair_d")):
            x = np.arange(1, 65)
            ax.plot(x, p[quantity], color=COLORS[key], label=NAMES[key],
                    marker="o" if key == "pairing" else "s", markersize=2.5, markevery=4)
            ax.plot(x, pp[quantity], color=COLORS[key], linestyle="--", alpha=0.55)
    for ax, label in zip(axes, ("Rung-averaged density", r"$(-1)^{i-1}(S^z_{i,1}-S^z_{i,2})/2$", r"Measured $\alpha_{leg,even}-\alpha_{rung}$ [t]")):
        ax.set_ylabel(label)
        ax.axvspan(1, 16.5, color="#E5E7EB", alpha=0.5)
        ax.axvspan(48.5, 64, color="#E5E7EB", alpha=0.5)
        ax.set_xlim(1, 64)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="lower right")
    axes[-1].set_xlabel("Rung")
    for extension in ("png", "pdf"):
        fig.savefig(HERE / f"spatial_profiles.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    checks = [c for label in LABELS.values() for c in verify_manifest(RUN / "results" / label)]
    results = {key: load_branch(key) for key in LABELS}
    endpoints = [x[0] for x in results.values()]
    for k in ("model_fingerprint", "numerical_fingerprint", "implementation_sha256", "tree_sha256", "gpu_manifest_sha256", "ep_source_sha256", "tensor_scalar_type"):
        assert len({r[k] for r in endpoints}) == 1, k
    write_csv("compact_checks.csv", checks)
    write_csv("endpoints.csv", endpoints)
    write_csv("history.csv", [r for x in results.values() for r in x[1]])
    write_csv("scf_audit.csv", [x[6] for x in results.values()])
    write_csv("parents.csv", [x[4] for x in results.values()])
    energy_rows = []
    for quantity in results["pairing"][2]:
        pairing, legacy = results["pairing"][2][quantity], results["legacy"][2][quantity]
        energy_rows.append(dict(quantity=quantity, pairing_total=pairing, legacy_total=legacy,
                               legacy_minus_pairing_total=legacy - pairing,
                               pairing_per_site=pairing / 128, legacy_per_site=legacy / 128,
                               legacy_minus_pairing_per_site=(legacy - pairing) / 128))
    write_csv("energy_decomposition.csv", energy_rows)
    spatial = []
    for key, (_, _, _, p, _, pp, _) in results.items():
        for role, data in (("chi400_endpoint", p), ("chi200_parent", pp)):
            for i in range(64):
                spatial.append(dict(lineage=key, role=role, rung=i+1, **{k: v[i] for k, v in data.items()}))
    write_csv("profiles.csv", spatial)
    pending = []
    for suffix in ("square_grid_smooth_pairing_chi200_loose", "cubic_unfrustrated_grid_smooth_pairing_chi200_loose", "square_t014_vm04_legacy_stripe_compare_chi200_loose"):
        run = ROOT / ("20260903_phase1_" + suffix)
        with (run / "jobs.tsv").open(encoding="utf-8", newline="") as stream:
            jobs = list(csv.DictReader(stream, delimiter="\t"))
        states = list(run.rglob("state.h5"))
        pending.append(dict(run=run.name, recorded_jobs=[r["job_id"] for r in jobs], local_terminal_states=len(states),
                            live_status_source="User reports this campaign pending; jobs.tsv is submission evidence only."))
    pair, legacy = endpoints
    summary = dict(endpoints=endpoints, pending_campaigns=pending,
                   legacy_minus_pairing_per_site=legacy["corrected_per_site"] - pair["corrected_per_site"],
                   legacy_minus_pairing_total=legacy["corrected_energy"] - pair["corrected_energy"],
                   sum_last10_ranges_per_site=sum(r["last10_energy_range_per_site"] for r in endpoints),
                   compact_checks_passed=len(checks), stored_fingerprints_match=True,
                   formal_ranking_allowed=all(r["accepted"] for r in endpoints),
                   evidence_boundary="Locally synced compact artifacts only; no full scratch, live scheduler or accounting verification.")
    (HERE / "summary.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    make_plots(results)
    print(json.dumps(json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
