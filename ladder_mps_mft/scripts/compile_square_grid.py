"""Compile the locally synced September 8 square chi=200 grid, without DMRG.

Run from any directory: python -B -X utf8 path/to/compile_square_grid.py
The HDF5 contains terminal snapshots, plot-ready arrays, candidate selection
evidence, source hashes, and every selected run's saved histories. No source is
modified. Legacy coverage and the flagged divergent endpoint are user-approved.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from audit_scf_numerics import _audit_state, _config_for_state, _scalar

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
ROOT = PROJECT / "output/phase1_gpu"
OUT = PROJECT / "output/square_grid_chi200_20260908"
REPORT = PROJECT / "docs/reports/square_grid_20260908"
CAMPAIGNS = (
    "20260830_phase1_square_t014_vm04_seed_chi200_loose",
    "20260902_phase1_square_t014_v000_seed_chi200_loose_cuda130",
    "20260903_phase1_square_grid_smooth_pairing_chi200_loose",
)
FINGERPRINTS = ("model_fingerprint", "numerical_fingerprint", "implementation_sha256", "ep_source_sha256")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def js(value):
    return json.dumps(safe(value), indent=2, allow_nan=False)


def point_id(t0, v):
    return f"t{round(t0 * 10):03d}_" + (f"vm{round(-v * 10):02d}" if v < 0 else "v000")


def verify_compact(path, f):
    branch = path.parents[2]
    with (branch / "stateless_manifest.tsv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream, delimiter="\t")
                if r["relative_path"].replace("\\", "/") == path.relative_to(branch).as_posix()]
    assert len(rows) == 1, path
    row = rows[0]
    actual = sha(path)
    assert actual == row["compact_sha256"], path
    assert path.stat().st_size == int(row["compact_bytes"]), path
    assert _scalar(f, "analysis_storage/full_artifact_sha256") == row["full_sha256"]
    assert bool(_scalar(f, "analysis_storage/is_stateless_copy"))
    names = []
    f.visit(names.append)
    assert not any(n.split("/")[-1] == "psi" for n in names)
    return actual, row["full_sha256"]


def candidate(run, path):
    config_path, config = _config_for_state(run, path)
    with h5py.File(path, "r") as f:
        compact_sha, full_sha = verify_compact(path, f)
        assert sha(config_path) == _scalar(f, "provenance/config_sha256")
        assert config["dmrg"]["maxdim"] == 200
        for k, expected in {"L": 64, "U": 8., "tp": .1, "density": .9375,
                            "transverse_geometry": "square"}.items():
            assert _scalar(f, "model/" + k) == expected
        assert _scalar(f, "provenance/initial_state_source") == "independent"
        assert 0 <= _scalar(f, "provenance/initial_amplitude") <= .001
        for k in ("parent_checkpoint", "resume_checkpoint", "inherit_from"):
            assert not _scalar(f, "provenance/" + k, "")
        audit = _audit_state(run, path)
        accepted = bool(_scalar(f, "accepted")) and audit["revised_accepted"]
        # This snapshot contains no accepted periodic solutions. Do not collapse
        # any future orbit into a single terminal field snapshot.
        if accepted:
            assert _scalar(f, "status") == "fixed_point" and _scalar(f, "fundamental_period") == 1
        stored = _scalar(f, "solution_target_density_corrected_variational_energy", np.nan)
        energy_kind = "stored_solution_target_density_corrected"
        if accepted and not np.isfinite(stored):
            canonical = _scalar(f, "solution_canonical_variational_energy")
            particles = np.sum(f["correlations/density_down"]) + np.sum(f["correlations/density_up"])
            stored = canonical + _scalar(f, "chemical_potential") * (120 - particles)
            energy_kind = "reconstructed_from_solution_canonical_energy"
        return dict(source=path.relative_to(REPO).as_posix(), run=run.name,
                    branch=path.relative_to(run).parts[1], t0=_scalar(f, "model/t0"),
                    V=_scalar(f, "model/V"), chi=200, status=_scalar(f, "status"),
                    accepted=accepted, stored_accepted=bool(_scalar(f, "accepted")),
                    seed=_scalar(f, "provenance/initial_seed"),
                    initial_amplitude=_scalar(f, "provenance/initial_amplitude"),
                    ancestry="independent small seed; no inherited fields/MPS",
                    energy=stored if accepted else np.nan, energy_kind=energy_kind if accepted else "not_rankable",
                    source_sha256=compact_sha, full_source_sha256=full_sha,
                    fingerprints={k: _scalar(f, "provenance/" + k) for k in FINGERPRINTS},
                    audit=audit, config_source=config_path.relative_to(REPO).as_posix(),
                    quality="accepted" if accepted else "unaccepted", selected=False)


def write_array(group, name, value):
    value = np.asarray(value)
    kwargs = dict(compression="gzip", shuffle=True) if value.ndim else {}
    group.create_dataset(name, data=value, **kwargs)


def plot_arrays(f, legacy):
    if legacy:
        arrays = {k: np.asarray(f[k]) for k in ("alpha", "beta")}
        # h5py dimension zero is Julia's final history dimension.
        arrays.update({k: np.asarray(f[k + "_list"][-1]) for k in ("C_pair", "C_exc_dn", "C_exc_up")})
        hartree = np.asarray(f["mu_cdw"])
    else:
        arrays = {k: np.asarray(f["fields/measured/" + k]) for k in ("alpha", "beta")}
        arrays.update({k: np.asarray(f["correlations/" + source]) for k, source in
                       (("C_pair", "pair"), ("C_exc_dn", "exchange_down"), ("C_exc_up", "exchange_up"))})
        hartree = np.asarray(f["fields/measured/mu_cdw"])
    # Same mapping as _p1_legacy_beta, in h5py's reversed dimension order.
    beta = arrays["beta"].copy()
    assert beta.shape == (2, 2, 64, 64, 2) and hartree.shape == (128, 2)
    for rung in range(64):
        for leg in range(2):
            beta[leg, leg, rung, rung, :] = hartree[2 * rung + leg, :]
    arrays["beta"] = beta
    arrays["mu_cdw"] = hartree
    assert arrays["alpha"].shape == (2, 2, 64, 64)
    for k, a in arrays.items():
        assert np.isfinite(a).all(), k
    return arrays


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    candidates = [candidate(ROOT / run, path) for run in CAMPAIGNS
                  for path in sorted((ROOT / run).rglob("state.h5"))]
    assert len(candidates) == 17
    selected = []
    for t0 in (1.0, 1.2, 1.4):
        for v in (-.4, -.2, 0.):
            pool = [r for r in candidates if r["t0"] == t0 and r["V"] == v]
            accepted = [r for r in pool if r["accepted"]]
            if accepted:
                for k in FINGERPRINTS:
                    values = {r["fingerprints"][k] for r in accepted}
                    assert len(values) == 1 and next(iter(values)), (t0, v, k)
                assert all(np.isfinite(r["energy"]) for r in accepted)
                choice = min(accepted, key=lambda r: (r["energy"], r["source"]))
                choice["selection_reason"] = "lowest corrected canonical energy among accepted, same-fingerprint small seeds"
            elif pool:
                assert (t0, v) == (1., -.4) and len(pool) == 1
                choice = pool[0]
                choice["quality"] = "diverging"
                choice["selection_reason"] = "user-approved terminal diagnostic; not a converged solution or energy-ranked choice"
            else:
                assert (t0, v) in ((1., 0.), (1.4, -.2))
                path = REPO / "stateless_data" / f"results_L_64_U_8.0_V_{v:.1f}_t0_{t0:.1f}_t_p_0.1_geometry_square_chi_200_density_0.9375_gpu.h5"
                with h5py.File(path) as f:
                    assert bool(_scalar(f, "completed")) and not bool(_scalar(f, "period2_cycle_detected"))
                choice = dict(source=path.relative_to(REPO).as_posix(), t0=t0, V=v, chi=200,
                              status="legacy_completed", accepted=False, quality="legacy",
                              source_sha256=sha(path), full_source_sha256="", seed="legacy fresh-run protocol; artifact ancestry not recorded",
                              ancestry="user-approved legacy coverage; not independently seed-certified",
                              selection_reason="only local coverage; legacy completion not recertified under Phase 1 gates",
                              energy=np.nan, energy_kind="legacy_E_not_ranked", selected=False)
            choice["selected"] = True
            choice["point_id"] = point_id(t0, v)
            selected.append(choice)
    bundle = OUT / "square_grid_chi200.h5"
    staging = OUT / "square_grid_chi200.building.h5"
    with h5py.File(staging, "w") as target:
        target["artifact_kind"] = "square_grid_bundle"
        target["schema_version"] = 2
        target["snapshot_date"] = "2026-09-08"
        target["t0_values"] = [1., 1.2, 1.4]
        target["V_values"] = [-.4, -.2, 0.]
        target["selection_json"] = js(selected)
        target["candidates_json"] = js(candidates)
        target["notes"] = ("6 accepted fixed points, 2 legacy-completed coverage points, 1 diverging diagnostic. "
                           "All chi=200. No converged-stripe inherited seeds. No cross-point energy ranking. "
                           "Legacy seed ancestry is not recorded; use is user-approved. HDF5 arrays follow Julia dimension order. "
                           "plot_data/beta uses mu_cdw on its diagonal; original fields are preserved under source_snapshot. "
                           "Every selected run's saved MF histories are retained, including recorded seeds; "
                           "legacy correlation histories are retained as well.")
        points = target.create_group("points")
        for row in selected:
            point = points.create_group(row["point_id"])
            point["metadata_json"] = js(row)
            for k in ("t0", "V", "chi", "status", "quality", "source", "source_sha256", "full_source_sha256"):
                point[k] = row[k]
            point["accepted"] = int(row["accepted"])
            point["plot_filename"] = f"results_L_64_U_8.0_V_{row['V']:.1f}_t0_{row['t0']:.1f}_t_p_0.1_geometry_square_chi_200_density_0.9375_gpu.h5"
            with h5py.File(REPO / row["source"]) as f:
                legacy = row["quality"] == "legacy"
                snapshot = point.create_group("source_snapshot")
                for k in f:
                    if isinstance(f[k], h5py.Dataset) and f[k].shape == ():
                        f.copy(k, snapshot)
                    elif not legacy and k in ("model", "energy", "provenance", "analysis_storage", "fields", "correlations", "history"):
                        f.copy(k, snapshot)
                    elif legacy and (k in ("alpha", "beta", "mu_cdw") or k.endswith("_list")):
                        f.copy(k, snapshot)
                plot = point.create_group("plot_data")
                for k, a in plot_arrays(f, legacy).items():
                    write_array(plot, k, a)
            if "config_source" in row:
                point["config_toml"] = (REPO / row["config_source"]).read_text(encoding="utf-8")
    # Read back every snapshot and compare its arrays with immutable inputs.
    with h5py.File(staging) as f:
        assert len(f["points"]) == 9
        for row in selected:
            with h5py.File(REPO / row["source"]) as source:
                for k, a in plot_arrays(source, row["quality"] == "legacy").items():
                    np.testing.assert_array_equal(f[f"points/{row['point_id']}/plot_data/{k}"], a)
                snapshot = f[f"points/{row['point_id']}/source_snapshot"]
                if row["quality"] == "legacy":
                    for k in source:
                        if k.endswith("_list"):
                            np.testing.assert_array_equal(snapshot[k], source[k])
                else:
                    names = []
                    source["history"].visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
                    for k in names:
                        np.testing.assert_array_equal(snapshot["history/"+k], source["history/"+k])
    staging.replace(bundle)
    (REPORT / "selection.json").write_text(js(dict(selected=selected, candidates=candidates, bundle_sha256=sha(bundle))) + "\n", encoding="utf-8")
    with (REPORT / "selection.csv").open("w", newline="", encoding="utf-8") as stream:
        keys = ("point_id", "t0", "V", "chi", "quality", "status", "seed", "energy", "energy_kind", "source", "source_sha256")
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    for row in selected:
        print(row["point_id"], row["quality"], row["seed"], row["energy"])
    print(f"Bundle: {bundle} ({bundle.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
