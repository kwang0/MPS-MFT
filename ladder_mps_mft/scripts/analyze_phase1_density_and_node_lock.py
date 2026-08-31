#!/usr/bin/env python3
"""Audit Phase 1 density-search cost and SDW-node/pairing-field alignment.

This script is local and read-only with respect to campaign artifacts.  It
parses saved Slurm logs and compact schema-v5 HDF5 mirrors, then writes bounded
TSV tables to a new analysis directory.  It does not alter convergence status,
HDF5 files, jobs, or the project budget ledger.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
import sys
import tomllib

import h5py
import numpy as np

from audit_spatial_phase_defects import (
    demodulated_envelope,
    discover_states,
    field_profiles,
    load_state,
    spatial_spectrum,
)


SWEEP_PATTERN = re.compile(r"^After sweep\s+\d+.*time=([0-9.]+)")
MF_PATTERN = re.compile(
    r"^MF\s+(\d+).*mu_evals=(\d+).*status=([^\s]+)"
)
PAIR_SIGNAL_FLOOR = 1.0e-6
SPIN_SIGNAL_FLOOR = 1.0e-5


def write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_config(run_directory: Path, label: str) -> dict:
    candidates = sorted((run_directory / "configs").glob(f"{label}.segment-*.toml"))
    if not candidates:
        return {}
    with candidates[-1].open("rb") as stream:
        return tomllib.load(stream)


def parse_log(path: Path) -> dict:
    sweeps = 0
    sweep_seconds = 0.0
    updates = []
    previous_sweeps = 0
    previous_seconds = 0.0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            sweep_match = SWEEP_PATTERN.match(line)
            if sweep_match:
                sweeps += 1
                sweep_seconds += float(sweep_match.group(1))
                continue
            mf_match = MF_PATTERN.match(line)
            if mf_match:
                updates.append(
                    {
                        "iteration": int(mf_match.group(1)),
                        "mu_evaluations": int(mf_match.group(2)),
                        "dmrg_sweeps": sweeps - previous_sweeps,
                        "logged_sweep_seconds": sweep_seconds - previous_seconds,
                        "status": mf_match.group(3),
                    }
                )
                previous_sweeps = sweeps
                previous_seconds = sweep_seconds
    return {
        "updates": updates,
        "dmrg_sweeps": sweeps,
        "logged_sweep_seconds": sweep_seconds,
    }


def density_work_rows(cohort: str, run_directory: Path, label_prefix: str) -> list[dict]:
    rows = []
    logs = sorted((run_directory / "logs").glob(f"{label_prefix}*.s1-*.out"))
    for log_path in logs:
        label = log_path.name.split(".s1-", 1)[0]
        parsed = parse_log(log_path)
        updates = parsed["updates"]
        if not updates:
            continue
        config = read_config(run_directory, label)
        dmrg = config.get("dmrg", {})
        run = config.get("run", {})
        total_evaluations = sum(row["mu_evaluations"] for row in updates)
        sweep_seconds = parsed["logged_sweep_seconds"]
        sweeps = parsed["dmrg_sweeps"]
        rows.append(
            {
                "cohort": cohort,
                "run": run_directory.name,
                "label": label,
                "chi": dmrg.get("maxdim", ""),
                "configured_sweeps": dmrg.get("nsweeps", ""),
                "dmrg_energy_tolerance": dmrg.get("energy_tol", ""),
                "mu_density_tolerance": dmrg.get("mu_density_tol", ""),
                "outer_updates": len(updates),
                "mu_evaluations": total_evaluations,
                "mu_evaluations_per_update": total_evaluations / len(updates),
                "printed_dmrg_sweeps": sweeps,
                "sweeps_per_mu_evaluation": sweeps / total_evaluations,
                "logged_sweep_hours": sweep_seconds / 3600.0,
                "seconds_per_printed_sweep": sweep_seconds / sweeps,
                "terminal_status": updates[-1]["status"],
                "configured_outer_update_limit": run.get("max_iterations", ""),
                "log_path": str(log_path.resolve()),
            }
        )
    return rows


def _bulk_slice(length: int, fraction: float = 0.75) -> slice:
    count = max(3, min(length, int(round(length * fraction))))
    start = (length - count) // 2
    return slice(start, start + count)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks


def _alignment_at_lag(pair_amplitude: np.ndarray, inverse_spin: np.ndarray, lag: int) -> float:
    if lag > 0:
        return _pearson(pair_amplitude[lag:], inverse_spin[:-lag])
    if lag < 0:
        return _pearson(pair_amplitude[:lag], inverse_spin[-lag:])
    return _pearson(pair_amplitude, inverse_spin)


def node_lock_metrics(spin: np.ndarray, pair_d: np.ndarray) -> dict:
    spin = np.asarray(spin, dtype=float)
    pair_d = np.asarray(pair_d, dtype=float)
    bulk = _bulk_slice(spin.size)
    _, _, spectrum = spatial_spectrum(spin, subtract_mean=True, fraction=0.75)
    q = spectrum["q"] if np.isfinite(spectrum["q"]) else 0.0
    spin_envelope = np.abs(demodulated_envelope(spin, q, subtract_mean=True))[bulk]
    pair_amplitude = np.abs(pair_d)[bulk]
    spin_rms = float(np.sqrt(np.mean((spin[bulk] - np.mean(spin[bulk])) ** 2)))
    pair_rms = float(np.sqrt(np.mean(pair_d[bulk] ** 2)))
    resolved = spin_rms >= SPIN_SIGNAL_FLOOR and pair_rms >= PAIR_SIGNAL_FLOOR
    if not resolved:
        return {
            "spin_q_over_pi": q / math.pi,
            "spin_rms": spin_rms,
            "pair_d_rms": pair_rms,
            "resolved": False,
            "node_lock_pearson": float("nan"),
            "node_lock_spearman": float("nan"),
            "node_enrichment_ratio": float("nan"),
            "best_lag_rungs": float("nan"),
            "best_lag_pearson": float("nan"),
        }

    inverse_spin = -spin_envelope
    pearson = _pearson(pair_amplitude, inverse_spin)
    spearman = _pearson(_rank(pair_amplitude), _rank(inverse_spin))
    lower = spin_envelope <= np.quantile(spin_envelope, 0.25)
    upper = spin_envelope >= np.quantile(spin_envelope, 0.75)
    low_pair = float(np.mean(pair_amplitude[lower]))
    high_pair = float(np.mean(pair_amplitude[upper]))
    enrichment = low_pair / high_pair if high_pair > np.finfo(float).eps else float("nan")
    lag_scores = {lag: _alignment_at_lag(pair_amplitude, inverse_spin, lag) for lag in range(-4, 5)}
    finite_lags = {lag: value for lag, value in lag_scores.items() if np.isfinite(value)}
    best_lag = max(finite_lags, key=finite_lags.get) if finite_lags else 0
    return {
        "spin_q_over_pi": q / math.pi,
        "spin_rms": spin_rms,
        "pair_d_rms": pair_rms,
        "resolved": True,
        "node_lock_pearson": pearson,
        "node_lock_spearman": spearman,
        "node_enrichment_ratio": enrichment,
        "best_lag_rungs": best_lag,
        "best_lag_pearson": finite_lags.get(best_lag, float("nan")),
    }


def node_lock_rows(
    cohort: str,
    run_directory: Path,
    label_pattern: re.Pattern,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    state_rows = []
    history_rows = []
    selected_profiles = {}
    for label, state_path in discover_states(run_directory):
        if not label_pattern.search(label):
            continue
        state = load_state(run_directory.name, label, state_path)
        profiles = field_profiles(state["measured"]["alpha"], state["measured"]["mu_cdw"])
        with h5py.File(state_path, "r") as handle:
            density_converged = np.asarray(
                handle["history/mu_density_converged"], dtype=bool
            )
            density_history = np.asarray(handle["history/density"], dtype=float)
            field_abs_history = np.asarray(handle["history/field_abs_residual"], dtype=float)
            field_rel_history = np.asarray(handle["history/field_rel_residual"], dtype=float)
            energy_history = np.asarray(handle["history/variational_energy"], dtype=float)
        if density_converged.size != len(state["iterations"]):
            raise ValueError(
                f"density-convergence history length mismatch in {state_path}"
            )
        valid_indices = np.flatnonzero(density_converged)
        selected_index = int(valid_indices[-1]) if valid_indices.size else len(state["iterations"]) - 1
        branch_history_start = len(history_rows)
        for index, iteration in enumerate(state["iterations"]):
            metrics = node_lock_metrics(profiles["spin_odd"][index], profiles["pair_d"][index])
            history_rows.append(
                {
                    "cohort": cohort,
                    "run": run_directory.name,
                    "label": label,
                    "iteration": int(iteration),
                    "density_converged": bool(density_converged[index]),
                    **metrics,
                }
            )
        final = history_rows[branch_history_start + selected_index].copy()
        config = read_config(run_directory, label)
        convergence = config.get("convergence", {})
        target_density = float(state["metadata"]["density"])
        energy_change_per_site = (
            abs(energy_history[selected_index] - energy_history[selected_index - 1])
            / (2.0 * state["metadata"]["L"])
            if selected_index > 0
            else float("inf")
        )
        field_gate = (
            field_abs_history[selected_index] <= convergence.get("field_abs_tol", 1.0e-6)
            or field_rel_history[selected_index] <= convergence.get("field_rel_tol", 5.0e-3)
        )
        density_error = abs(density_history[selected_index] - target_density)
        density_gate = density_error <= convergence.get("density_tol", 1.0e-5)
        energy_gate = energy_change_per_site <= convergence.get(
            "variational_energy_tol", 1.0e-7
        )
        final.update(
            {
                "status": state["metadata"]["status"],
                "accepted": state["metadata"]["accepted"],
                "period": state["metadata"]["period"],
                "geometry": state["metadata"]["geometry"],
                "chi": config.get("dmrg", {}).get("maxdim", state["metadata"]["chi"]),
                "measurement_selection": "last_density_converged",
                "terminal_iteration": int(state["iterations"][-1]),
                "field_abs_residual": float(field_abs_history[selected_index]),
                "field_rel_residual": float(field_rel_history[selected_index]),
                "density_error": density_error,
                "variational_energy_change_per_site": energy_change_per_site,
                "field_gate": field_gate,
                "density_gate": density_gate,
                "energy_gate": energy_gate,
                "state_path": state["metadata"]["state_path"],
            }
        )
        state_rows.append(final)
        selected_profiles[label] = {
            "iteration": int(state["iterations"][selected_index]),
            "spin": np.asarray(profiles["spin_odd"][selected_index], dtype=float),
            "pair_d": np.asarray(profiles["pair_d"][selected_index], dtype=float),
        }
    return state_rows, history_rows, selected_profiles


def _profile_signature(spin: np.ndarray, pair_d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bulk = _bulk_slice(spin.size)
    _, _, spectrum = spatial_spectrum(spin, subtract_mean=True, fraction=0.75)
    q = spectrum["q"] if np.isfinite(spectrum["q"]) else 0.0
    spin_envelope = np.abs(demodulated_envelope(spin, q, subtract_mean=True))[bulk]
    pair_amplitude = np.abs(pair_d)[bulk]
    return spin_envelope, pair_amplitude


def profile_similarity_rows(selected_profiles: dict[str, dict]) -> list[dict]:
    labels = sorted(selected_profiles)
    rows = []
    for left_index, left_label in enumerate(labels):
        left = selected_profiles[left_label]
        left_spin, left_pair = _profile_signature(left["spin"], left["pair_d"])
        for right_label in labels[left_index + 1 :]:
            right = selected_profiles[right_label]
            right_spin, right_pair = _profile_signature(right["spin"], right["pair_d"])
            rows.append(
                {
                    "left_label": left_label,
                    "left_iteration": left["iteration"],
                    "right_label": right_label,
                    "right_iteration": right["iteration"],
                    "spin_envelope_pearson": _pearson(left_spin, right_spin),
                    "pair_amplitude_pearson": _pearson(left_pair, right_pair),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--latest-run", required=True, type=Path)
    parser.add_argument("--chi200-run", required=True, type=Path)
    parser.add_argument("--recurrence-run", required=True, type=Path)
    parser.add_argument("--v3-run", required=True, type=Path)
    args = parser.parse_args(argv)

    for directory in (args.latest_run, args.chi200_run, args.recurrence_run, args.v3_run):
        if not directory.is_dir():
            parser.error(f"run directory does not exist: {directory}")

    density_rows = []
    density_rows.extend(density_work_rows("matched_chi400", args.latest_run, "unfrustrated__"))
    density_rows.extend(density_work_rows("independent_chi200", args.chi200_run, "unfrustrated__"))
    density_rows.extend(density_work_rows("prior_chi400", args.recurrence_run, "unfrustrated__"))

    node_rows = []
    node_history = []
    latest_profiles = {}
    cohorts = (
        ("matched_chi400", args.latest_run, re.compile(r"^unfrustrated__")),
        ("prior_chi400", args.recurrence_run, re.compile(r"^unfrustrated__")),
        ("v3_chi200_unfrustrated", args.v3_run, re.compile(r"^unfrustrated__")),
        ("v3_chi200_frustrated", args.v3_run, re.compile(r"^frustrated__")),
    )
    for cohort, run_directory, pattern in cohorts:
        final_rows, histories, selected_profiles = node_lock_rows(cohort, run_directory, pattern)
        node_rows.extend(final_rows)
        node_history.extend(histories)
        if cohort == "matched_chi400":
            latest_profiles = selected_profiles
    similarity_rows = profile_similarity_rows(latest_profiles)

    density_columns = [
        "cohort", "run", "label", "chi", "configured_sweeps",
        "dmrg_energy_tolerance", "mu_density_tolerance", "outer_updates",
        "mu_evaluations", "mu_evaluations_per_update", "printed_dmrg_sweeps",
        "sweeps_per_mu_evaluation", "logged_sweep_hours",
        "seconds_per_printed_sweep", "terminal_status",
        "configured_outer_update_limit", "log_path",
    ]
    node_columns = [
        "cohort", "run", "label", "iteration", "density_converged", "status",
        "accepted", "period", "geometry", "chi", "measurement_selection",
        "terminal_iteration", "field_abs_residual", "field_rel_residual",
        "density_error", "variational_energy_change_per_site", "field_gate",
        "density_gate", "energy_gate", "spin_q_over_pi", "spin_rms", "pair_d_rms", "resolved",
        "node_lock_pearson", "node_lock_spearman", "node_enrichment_ratio",
        "best_lag_rungs", "best_lag_pearson", "state_path",
    ]
    history_columns = [
        "cohort", "run", "label", "iteration", "density_converged",
        "spin_q_over_pi", "spin_rms", "pair_d_rms", "resolved",
        "node_lock_pearson", "node_lock_spearman", "node_enrichment_ratio",
        "best_lag_rungs", "best_lag_pearson",
    ]
    similarity_columns = [
        "left_label", "left_iteration", "right_label", "right_iteration",
        "spin_envelope_pearson", "pair_amplitude_pearson",
    ]
    write_tsv(args.output / "density_work.tsv", density_rows, density_columns)
    write_tsv(args.output / "node_lock_final.tsv", node_rows, node_columns)
    write_tsv(args.output / "node_lock_history.tsv", node_history, history_columns)
    write_tsv(args.output / "matched_profile_similarity.tsv", similarity_rows, similarity_columns)
    print(f"density_rows={len(density_rows)}")
    print(f"node_rows={len(node_rows)}")
    print(f"node_history_rows={len(node_history)}")
    print(f"similarity_rows={len(similarity_rows)}")
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
