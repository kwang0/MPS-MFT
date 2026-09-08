"""Read-only local evidence extraction for the September 4 systematic review.

Outputs are review artifacts only. Never opens an input HDF5 for writing.
The existing recurrence audit is a diagnostic screen, not recertification.
"""
from pathlib import Path
import csv, hashlib, importlib.util, json, math, sys, tomllib
from collections import Counter
import h5py
import numpy as np
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = PROJECT / "output" / "phase1_gpu"
spec = importlib.util.spec_from_file_location("existing_numerical_audit", PROJECT / "scripts" / "audit_scf_numerics.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def scalar(handle, key, default=None):
    return audit._scalar(handle, key, default)

def write_csv(name, rows):
    if not rows:
        return
    with (HERE / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def run():
    rows, errors, screening = [], [], []
    for campaign in sorted(ROOT.iterdir()):
        if not campaign.is_dir():
            continue
        for path in sorted(campaign.rglob("state.h5")):
            try:
                with h5py.File(path, "r") as f:
                    L = int(scalar(f, "model/L", 0))
                    density = float(scalar(f, "model/density", math.nan))
                    canonical = float(scalar(f, "energy/canonical_variational_energy", math.nan))
                    corrected = float(scalar(f, "energy/target_density_corrected_variational_energy", math.nan))
                    mu = float(scalar(f, "chemical_potential", math.nan))
                    observed = float(np.mean(np.asarray(f["correlations/density_down"]) + np.asarray(f["correlations/density_up"]))) if "correlations/density_down" in f else math.nan
                    config_path, config = audit._config_for_state(campaign, path)
                    dmrg = config.get("dmrg", {})
                    final_sweeps = []
                    if "history/dmrg" in f and len(f["history/dmrg"]):
                        last_key = sorted(f["history/dmrg"])[-1]
                        final_sweeps = np.asarray(f[f"history/dmrg/{last_key}/sweep_energy"], dtype=float)
                    delta = abs(float(final_sweeps[-1] - final_sweeps[-2])) if len(final_sweeps) >= 2 else math.nan
                    row = dict(campaign=campaign.name, path=str(path.relative_to(PROJECT)),
                        compact_sha256=sha(path), full_sha256=scalar(f,"analysis_storage/full_artifact_sha256", ""),
                        status=scalar(f,"status", "unknown"), accepted=bool(scalar(f,"accepted", False)),
                        period=int(scalar(f,"fundamental_period",0)), schema=scalar(f,"schema_version",0),
                        geometry=scalar(f,"model/transverse_geometry", "unknown"), L=L,
                        t0=scalar(f,"model/t0",None), V=scalar(f,"model/V",None), chi=dmrg.get("maxdim"),
                        branch=scalar(f,"provenance/branch_label", "unknown"),
                        iterations=len(f["history/iteration"]) if "history/iteration" in f else 0,
                        density=observed, target_density=density, mu=mu,
                        canonical_per_site=canonical/(2*L) if L else math.nan,
                        corrected_per_site=corrected/(2*L) if L else math.nan,
                        density_correction_scale=abs(mu*(observed-density)),
                        final_sweeps=len(final_sweeps), last_sweep_change=delta,
                        dmrg_energy_tol=dmrg.get("energy_tol"),
                        model_fp=scalar(f,"provenance/model_fingerprint", ""),
                        numerical_fp=scalar(f,"provenance/numerical_fingerprint", ""),
                        implementation_fp=scalar(f,"provenance/implementation_sha256", ""),
                        registry_fp=scalar(f,"provenance/ep_source_sha256", ""),
                        stored_slow_rel=scalar(f,"fixed_point_extrapolated_rel_residual",None))
                    rows.append(row)
                try:
                    check = audit._audit_state(campaign,path)
                    screening.append(check)
                except Exception as exc:
                    errors.append(dict(path=str(path.relative_to(PROJECT)), stage="history_screen", error=str(exc)))
            except Exception as exc:
                errors.append(dict(path=str(path.relative_to(PROJECT)), stage="metadata_read", error=str(exc)))
    write_csv("state_inventory.csv", rows)
    write_csv("recurrence_screen.csv", screening)
    write_csv("read_gaps.csv", errors)
    # Independently verify the six current V=0 terminal compact files against their manifests.
    checks = []
    current = ROOT / "20260902_phase1_square_t014_v000_seed_chi200_loose_cuda130"
    for manifest in sorted(current.rglob("stateless_manifest.tsv")):
        with manifest.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if Path(row["relative_path"]).name != "state.h5":
                    continue
                path = manifest.parent / row["relative_path"]
                checks.append(dict(path=str(path.relative_to(PROJECT)), present=path.exists(),
                    size_match=path.exists() and path.stat().st_size == int(row["compact_bytes"]),
                    hash_match=path.exists() and sha(path) == row["compact_sha256"]))
    write_csv("current_terminal_hash_checks.csv", checks)
    campaigns=[]
    for name in sorted({r["campaign"] for r in rows}):
        local=[r for r in rows if r["campaign"] == name]
        screened=[r for r in screening if r["run"] == name]
        campaigns.append(dict(campaign=name, readable_states=len(local),
            stored_accepted=sum(r["accepted"] for r in local),
            screened_states=len(screened), screened_accepted=sum(r["revised_accepted"] for r in screened),
            changed_labels=sum(r["stored_status"] != r["revised_status"] for r in screened),
            statuses="; ".join(f"{k}:{v}" for k,v in Counter(r["status"] for r in local).items())))
    write_csv("campaign_inventory.csv", campaigns)
    latest=[r for r in rows if r["campaign"] == current.name]
    summary=dict(state_paths=len(rows), screenable_paths=len(screening), read_or_history_gaps=len(errors),
        stored_accepted=sum(r["accepted"] for r in rows),
        screen_changed_labels=sum(r["stored_status"] != r["revised_status"] for r in screening),
        unique_recorded_full_identities=len({r["full_sha256"] or r["compact_sha256"] for r in rows}),
        current_terminal_hash_checks=checks,
        current_corrected_energy_spread_per_site=max(r["corrected_per_site"] for r in latest)-min(r["corrected_per_site"] for r in latest),
        current_density_correction_range=[min(r["density_correction_scale"] for r in latest),max(r["density_correction_scale"] for r in latest)],
        source_boundary="Local compact HDF5 and manifests; full scratch hashes are recorded identity only, not verified existence.")
    (HERE/"evidence_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "current_terminal_hash_checks"},indent=2))
    print("Current terminal compact manifests:",len(checks),"all pass:",all(r["size_match"] and r["hash_match"] for r in checks))
    print("Current V=0 endpoints (last inner sweep difference is a diagnostic, not an error bar):")
    for r in latest:
        print(r["branch"],r["iterations"],r["last_sweep_change"],r["dmrg_energy_tol"],r["density_correction_scale"])

if __name__ == "__main__":
    run()
