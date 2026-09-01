#!/usr/bin/env python3
"""Read-only offline audit of SCF recurrence and slow-mode convergence.

The audit never edits an HDF5 artifact.  It reconstructs physical residuals
``r_k = f(x_k) - x_k`` from complete applied/measured histories, rejects a
putative period-two recurrence unless the measured sequence oscillates, and
estimates the remaining one-dimensional fixed-point error as
``r_k / (1 - lambda)`` when successive residuals are strongly aligned.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import tomllib

import h5py
import numpy as np


OSCILLATION_COSINE_MAX = -0.5
TWO_STEP_RATIO_MAX = 0.5
SLOW_MODE_COSINE_MIN = 0.9


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return value


def _scalar(handle, path, default=None):
    return _decode(handle[path][()]) if path in handle else default


def _julia_array(dataset):
    values = np.asarray(dataset)
    return values if values.ndim <= 1 else values.transpose(tuple(reversed(range(values.ndim))))


def _history_fields(handle, source):
    root = f"history/fields/{source}"
    names = ("alpha", "beta", "mu_cdw")
    missing = [f"{root}/{name}" for name in names if f"{root}/{name}" not in handle]
    if missing:
        raise ValueError("missing " + ", ".join(missing))
    return {name: np.asarray(_julia_array(handle[f"{root}/{name}"]), dtype=float) for name in names}


def _field_vector(fields, index):
    return np.concatenate([fields[name][..., index].reshape(-1) for name in ("alpha", "beta", "mu_cdw")])


def _residual_vector(applied, measured, index):
    return _field_vector(measured, index) - _field_vector(applied, index)


def _cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > np.finfo(float).eps else float("nan")


def _hybrid_residual(applied, measured, index):
    left = _field_vector(applied, index)
    right = _field_vector(measured, index)
    delta = right - left
    absolute = float(np.max(np.abs(delta))) if delta.size else 0.0
    relative = float(np.linalg.norm(delta) / max(np.linalg.norm(left), np.linalg.norm(right), np.finfo(float).eps))
    return absolute, relative


def _config_for_state(run_dir, state_path):
    relative = state_path.relative_to(run_dir)
    collection_index = next(
        (index for index, part in enumerate(relative.parts) if part in ("results", "stateless_results")),
        None,
    )
    if collection_index is None or collection_index + 1 >= len(relative.parts):
        return None, {}
    label = relative.parts[collection_index + 1]
    candidates = sorted((run_dir / "configs").glob(f"{label}.segment-*.toml"))
    if not candidates:
        return None, {}
    path = candidates[-1]
    with path.open("rb") as stream:
        return path, tomllib.load(stream)


def _audit_state(run_dir, state_path):
    config_path, config = _config_for_state(run_dir, state_path)
    convergence = config.get("convergence", {})
    field_abs_tol = float(convergence.get("field_abs_tol", 1e-6))
    field_rel_tol = float(convergence.get("field_rel_tol", 5e-3))
    period_repeats = int(convergence.get("period_repeats", 3))

    with h5py.File(state_path, "r") as handle:
        applied = _history_fields(handle, "applied")
        measured = _history_fields(handle, "measured")
        count = measured["alpha"].shape[-1]
        if count < 2:
            raise ValueError("fewer than two complete field-history records")
        status = str(_scalar(handle, "status", "unknown"))
        accepted = bool(_scalar(handle, "accepted", False))
        period = int(_scalar(handle, "fundamental_period", 0))
        branch = str(_scalar(handle, "provenance/branch_label", state_path.parts[-3]))
        full_sha = str(_scalar(handle, "analysis_storage/full_artifact_sha256", ""))
        update_modes = [str(_decode(item)) for item in np.asarray(handle["history/update_mode"])] \
            if "history/update_mode" in handle else []

        previous_residual = _residual_vector(applied, measured, count - 2)
        current_residual = _residual_vector(applied, measured, count - 1)
        residual_cosine = _cosine(current_residual, previous_residual)
        previous_norm_squared = float(np.dot(previous_residual, previous_residual))
        contraction = float(np.dot(current_residual, previous_residual) / previous_norm_squared) \
            if previous_norm_squared > np.finfo(float).eps else float("nan")
        raw_abs, raw_rel = _hybrid_residual(applied, measured, count - 1)
        slow_mode = math.isfinite(residual_cosine) and residual_cosine >= SLOW_MODE_COSINE_MIN
        if slow_mode and contraction >= 1.0:
            factor = float("inf")
        elif slow_mode and math.isfinite(contraction):
            factor = max(1.0, 1.0 / max(1.0 - contraction, np.finfo(float).eps))
        else:
            factor = 1.0
        extrap_abs = raw_abs * factor
        extrap_rel = raw_rel * factor
        extrapolated_gate = extrap_abs <= field_abs_tol or extrap_rel <= field_rel_tol

        oscillation_cosine = float("nan")
        two_step_ratio = float("nan")
        oscillatory_period2 = False
        if count >= 3:
            required = min(count, 2 * (period_repeats + 1))
            first = max(2, count - required + 2)
            cosines = []
            ratios = []
            for index in range(first, count):
                current_step = _field_vector(measured, index) - _field_vector(measured, index - 1)
                previous_step = _field_vector(measured, index - 1) - _field_vector(measured, index - 2)
                cosine = _cosine(current_step, previous_step)
                if math.isfinite(cosine):
                    cosines.append(cosine)
                denominator = np.linalg.norm(current_step)
                if denominator > np.finfo(float).eps:
                    ratios.append(float(np.linalg.norm(_field_vector(measured, index) - _field_vector(measured, index - 2)) / denominator))
            if cosines:
                oscillation_cosine = max(cosines)
            if ratios:
                two_step_ratio = max(ratios)
            oscillatory_period2 = bool(
                cosines and ratios and oscillation_cosine <= OSCILLATION_COSINE_MAX
                and two_step_ratio <= TWO_STEP_RATIO_MAX
            )

        stored_period2 = period == 2 and status in ("periodic_candidate", "periodic_solution")
        revised_status = status
        revised_accepted = accepted
        reason = "unchanged"
        if stored_period2 and not oscillatory_period2:
            revised_status = "iterating_monotone_drift"
            revised_accepted = False
            reason = "stored period-two class fails oscillation criterion"
        elif status == "fixed_point" and accepted and not extrapolated_gate:
            revised_status = "fixed_point_candidate_slow_mode"
            revised_accepted = False
            reason = "stored fixed point fails extrapolated residual gate"

        return {
            "run": run_dir.name,
            "branch": branch,
            "state_path": str(state_path.resolve()),
            "config_path": str(config_path.resolve()) if config_path else "",
            "full_artifact_sha256": full_sha,
            "records": count,
            "last_update_mode": update_modes[-1] if update_modes else "unknown",
            "stored_status": status,
            "stored_accepted": accepted,
            "stored_period": period,
            "revised_status": revised_status,
            "revised_accepted": revised_accepted,
            "revision_reason": reason,
            "field_abs_tol": field_abs_tol,
            "field_rel_tol": field_rel_tol,
            "raw_abs_residual": raw_abs,
            "raw_rel_residual": raw_rel,
            "residual_cosine": residual_cosine,
            "slow_mode_lambda": contraction,
            "extrapolation_factor": factor,
            "extrapolated_abs_residual": extrap_abs,
            "extrapolated_rel_residual": extrap_rel,
            "extrapolated_gate_pass": extrapolated_gate,
            "period2_oscillation_cosine_max": oscillation_cosine,
            "period2_two_step_ratio_max": two_step_ratio,
            "period2_oscillation_pass": oscillatory_period2,
        }


def _format(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "Inf" if math.isinf(value) else "NaN" if math.isnan(value) else f"{value:.17g}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Phase 1 output root containing campaign directories")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    rows = []
    errors = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for state_path in sorted(run_dir.rglob("state.h5")):
            try:
                rows.append(_audit_state(run_dir, state_path))
            except Exception as error:
                errors.append({"state_path": str(state_path.resolve()), "error": str(error)})

    fields = list(rows[0]) if rows else []
    with (output / "artifact_reaudit.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: _format(value) for key, value in row.items()} for row in rows)
    with (output / "audit_errors.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("state_path", "error"), delimiter="\t")
        writer.writeheader()
        writer.writerows(errors)

    changed = [row for row in rows if row["stored_status"] != row["revised_status"]]
    unique_keys = {
        row["full_artifact_sha256"] or row["state_path"]
        for row in rows
    }
    stage_v3 = [
        row for row in changed
        if "20260824_phase1_gpu_v3" in row["run"] or "recurrence_chi400" in row["run"]
    ]
    with (output / "summary.md").open("w", encoding="utf-8") as stream:
        stream.write("# Offline SCF numerical re-audit\n\n")
        stream.write("Read-only audit; no HDF5 status or acceptance field was modified.\n\n")
        stream.write(f"- State paths audited: `{len(rows)}`\n")
        stream.write(f"- Unique full-artifact identities or standalone paths: `{len(unique_keys)}`\n")
        stream.write(f"- Unreadable/incomplete paths: `{len(errors)}`\n")
        stream.write(f"- Stored classifications changed by the proposed numerical gates: `{len(changed)}`\n")
        stream.write(f"- Changed v3/Stage-A paths: `{len(stage_v3)}`\n\n")
        stream.write("The period-two criterion requires every recent step cosine to be at most "
                     f"`{OSCILLATION_COSINE_MAX}` and every two-step/one-step ratio to be at most "
                     f"`{TWO_STEP_RATIO_MAX}`. Slow-mode extrapolation is activated when residual "
                     f"cosine is at least `{SLOW_MODE_COSINE_MIN}`.\n\n")
        stream.write("## Reclassified v3 and Stage A states\n\n")
        stream.write("| run | branch | stored | revised | cos(step) | d2/d1 | lambda | extrapolated relative residual |\n")
        stream.write("|---|---|---|---|---:|---:|---:|---:|\n")
        for row in stage_v3:
            stream.write(
                f"| {row['run']} | {row['branch']} | {row['stored_status']} | {row['revised_status']} | "
                f"{_format(row['period2_oscillation_cosine_max'])} | "
                f"{_format(row['period2_two_step_ratio_max'])} | "
                f"{_format(row['slow_mode_lambda'])} | "
                f"{_format(row['extrapolated_rel_residual'])} |\n"
            )

    print(f"state_paths={len(rows)}")
    print(f"unique_artifacts={len(unique_keys)}")
    print(f"errors={len(errors)}")
    print(f"reclassified={len(changed)}")
    print(f"v3_stage_a_reclassified={len(stage_v3)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
