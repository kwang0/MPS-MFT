#!/usr/bin/env python3
"""Compare the Phase 1 matched-mode seed pilot with its closest controls.

This is a local, read-only analysis of compact schema-v5 HDF5 states.  It
creates convergence figures and tabular snapshots, but it never changes an
HDF5 artifact, acceptance status, or the project budget ledger.  Energies are
deliberately omitted: the selected states are either unaccepted or have
different numerical/implementation fingerprints.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#20242A"
GREY = "#6B7280"
LIGHT_GREY = "#E5E7EB"
BLUE = "#2563EB"
ORANGE = "#D97706"
OLIVE = "#6B7A18"
PINK = "#B83280"
PURPLE = "#7C3AED"


DISPLAY_NAMES = {
    "unfrustrated__pairing_matched_m000_chi400": "matched pairing",
    "unfrustrated__sdw_matched_m058_chi400": "matched SDW",
    "unfrustrated__cdw_matched_m011_chi400": "matched CDW",
    "unfrustrated__pairing_s2_chi400": "prior broadband pairing s2",
    "unfrustrated__pairing_s1_phase001_chi400": "prior orbit parent p1",
    "unfrustrated__pairing_s1_phase002_chi400": "prior orbit parent p2",
    "unfrustrated__pairing_s1": "v3 pairing (chi=200)",
    "unfrustrated__sdw_s1": "v3 SDW (chi=200)",
    "unfrustrated__cdw_s1": "v3 CDW (chi=200)",
}


STYLE = {
    "unfrustrated__pairing_matched_m000_chi400": (BLUE, "o", "-"),
    "unfrustrated__sdw_matched_m058_chi400": (OLIVE, "s", "-"),
    "unfrustrated__cdw_matched_m011_chi400": (ORANGE, "^", "-"),
    "unfrustrated__pairing_s2_chi400": (PINK, "D", "--"),
    "unfrustrated__pairing_s1_phase001_chi400": (GREY, ">", ":"),
    "unfrustrated__pairing_s1_phase002_chi400": (PURPLE, "<", ":"),
}


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return value


def _scalar(handle, path, default=None):
    if path not in handle:
        return default
    return _decode(handle[path][()])


def discover_states(run_directory):
    run_directory = Path(run_directory).resolve()
    stateless = run_directory / "stateless_results"
    results = stateless if stateless.is_dir() else run_directory / "results"
    if not results.is_dir():
        raise ValueError("results directory does not exist: {}".format(results))
    states = {}
    for path in results.rglob("state.h5"):
        relative = path.relative_to(results)
        if len(relative.parts) < 2:
            continue
        label = relative.parts[0]
        key = (path.stat().st_mtime, str(path))
        if label not in states or key > states[label][0]:
            states[label] = (key, path)
    return {label: states[label][1] for label in sorted(states)}


def load_state(run_name, label, path):
    with h5py.File(path, "r") as handle:
        iterations = np.asarray(handle["history/iteration"], dtype=int)
        residual = np.asarray(handle["history/field_rel_residual"], dtype=float)
        density = np.asarray(handle["history/density"], dtype=float)
        wall_seconds = np.asarray(handle["history/wall_seconds"], dtype=float)
        density_target = float(_scalar(handle, "model/density"))
        mu_converged = np.asarray(handle["history/mu_density_converged"], dtype=bool)
        return {
            "run": run_name,
            "label": label,
            "display_name": DISPLAY_NAMES.get(label, label),
            "path": str(path),
            "status": str(_scalar(handle, "status", "")),
            "accepted": bool(_scalar(handle, "accepted", False)),
            "period": int(_scalar(handle, "fundamental_period", 0)),
            "seed_protocol": str(_scalar(handle, "provenance/initial_seed_protocol", "legacy")),
            "seed_label": str(_scalar(handle, "provenance/seed_label", "")),
            "model_fingerprint": str(_scalar(handle, "provenance/model_fingerprint", "")),
            "numerical_fingerprint": str(_scalar(handle, "provenance/numerical_fingerprint", "")),
            "implementation_fingerprint": str(_scalar(handle, "provenance/implementation_sha256", "")),
            "stateless": bool(_scalar(handle, "analysis_storage/is_stateless_copy", False)),
            "iterations": iterations,
            "residual": residual,
            "density_error": np.abs(density - density_target),
            "wall_seconds": wall_seconds,
            "cumulative_hours": np.cumsum(wall_seconds) / 3600.0,
            "mu_converged": mu_converged,
        }


def read_channel_summary(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            row["final_signal_rms"] = float(row["final_signal_rms"])
            rows.append(row)
    return rows


def write_tsv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis):
    axis.grid(True, color=LIGHT_GREY, linewidth=0.7)
    axis.tick_params(colors=INK)
    for spine in axis.spines.values():
        spine.set_color(GREY)


def plot_convergence(states, output_path):
    plotted_labels = [
        "unfrustrated__pairing_matched_m000_chi400",
        "unfrustrated__sdw_matched_m058_chi400",
        "unfrustrated__cdw_matched_m011_chi400",
        "unfrustrated__pairing_s2_chi400",
        "unfrustrated__pairing_s1_phase001_chi400",
        "unfrustrated__pairing_s1_phase002_chi400",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    residual_ax, density_ax, hours_ax, ratio_ax = axes.ravel()

    for label in plotted_labels:
        state = states[label]
        color, marker, linestyle = STYLE[label]
        residual_ax.plot(
            state["iterations"], state["residual"], color=color, marker=marker,
            linestyle=linestyle, linewidth=1.6, markersize=4,
            label=state["display_name"],
        )
        density_ax.plot(
            state["iterations"], np.maximum(state["density_error"], 1e-13),
            color=color, marker=marker, linestyle=linestyle, linewidth=1.6,
            markersize=4, label=state["display_name"],
        )
        hours_ax.plot(
            state["cumulative_hours"], state["residual"], color=color,
            marker=marker, linestyle=linestyle, linewidth=1.6, markersize=4,
            label=state["display_name"],
        )

    residual_ax.set_yscale("log")
    residual_ax.set_xlabel("Completed MF update")
    residual_ax.set_ylabel("Raw-map relative field residual")
    residual_ax.set_title("Convergence trajectory")
    residual_ax.legend(fontsize=8, ncol=2)

    density_ax.set_yscale("log")
    density_ax.set_xlabel("Completed MF update")
    density_ax.set_ylabel("Absolute density-target error")
    density_ax.set_title("Density targeting")

    hours_ax.set_yscale("log")
    hours_ax.set_xlabel("Cumulative recorded branch wall time (hours)")
    hours_ax.set_ylabel("Raw-map relative field residual")
    hours_ax.set_title("Convergence per elapsed branch time")

    matched = states["unfrustrated__pairing_matched_m000_chi400"]
    broadband = states["unfrustrated__pairing_s2_chi400"]
    common = min(len(matched["residual"]), len(broadband["residual"]))
    ratio = matched["residual"][:common] / broadband["residual"][:common]
    ratio_ax.axhline(1.0, color=GREY, linestyle="--", linewidth=1.2)
    ratio_ax.plot(
        matched["iterations"][:common], ratio, color=BLUE, marker="o",
        linewidth=1.8,
    )
    ratio_ax.fill_between(
        matched["iterations"][:common], ratio, 1.0, where=ratio <= 1.0,
        color=BLUE, alpha=0.13,
    )
    ratio_ax.set_xlabel("Completed MF update")
    ratio_ax.set_ylabel("Matched / broadband pairing residual")
    ratio_ax.set_title("Iteration-matched pairing-seed effect (<1 is lower)")
    ratio_ax.set_ylim(0.0, max(1.08, 1.05 * float(np.max(ratio))))

    for axis in axes.ravel():
        style_axis(axis)
    fig.suptitle(
        "Phase 1 chi=400 matched-mode pilot versus prior controls\n"
        "Raw-map diagnostics only; orbit parents remain unaccepted period-two candidates",
        fontsize=14, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def plot_channel_content(channel_rows, output_path):
    selected = [
        "unfrustrated__pairing_matched_m000_chi400",
        "unfrustrated__sdw_matched_m058_chi400",
        "unfrustrated__cdw_matched_m011_chi400",
        "unfrustrated__pairing_s2_chi400",
        "unfrustrated__pairing_s1_phase001_chi400",
        "unfrustrated__sdw_s1",
        "unfrustrated__cdw_s1",
    ]
    lookup = {(row["label"], row["group"]): row["final_signal_rms"] for row in channel_rows}
    groups = ("charge", "spin", "pairing")
    colors = (BLUE, ORANGE, OLIVE)
    x = np.arange(len(selected), dtype=float)
    width = 0.23
    fig, axis = plt.subplots(figsize=(13.2, 5.8))
    for index, (group, color) in enumerate(zip(groups, colors)):
        values = [max(lookup.get((label, group), math.nan), 1e-12) for label in selected]
        axis.bar(x + (index - 1) * width, values, width=width, color=color, label=group)
    axis.set_yscale("log")
    axis.set_ylabel("Final selected-channel RMS")
    axis.set_xticks(x)
    axis.set_xticklabels([DISPLAY_NAMES[label] for label in selected], rotation=23, ha="right")
    axis.set_title(
        "Terminal one-point field content (diagnostic, not thermodynamic order)"
    )
    axis.legend(ncol=3)
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(output_path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-run", required=True)
    parser.add_argument("--recurrence-run", required=True)
    parser.add_argument("--v3-run", required=True)
    parser.add_argument("--spatial-audit", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite existing output: {}".format(output))
    output.mkdir(parents=True)

    run_paths = {
        Path(args.matched_run).name: Path(args.matched_run),
        Path(args.recurrence_run).name: Path(args.recurrence_run),
        Path(args.v3_run).name: Path(args.v3_run),
    }
    states = {}
    for run_name, run_path in run_paths.items():
        for label, path in discover_states(run_path).items():
            if label in DISPLAY_NAMES:
                states[label] = load_state(run_name, label, path)

    required = set(DISPLAY_NAMES)
    missing = sorted(required.difference(states))
    if missing:
        raise ValueError("missing required comparison states: {}".format(", ".join(missing)))

    channel_rows = read_channel_summary(Path(args.spatial_audit) / "channel_summary.tsv")
    plot_convergence(states, output / "convergence_comparison.png")
    plot_channel_content(channel_rows, output / "terminal_channel_rms_comparison.png")

    history_rows = []
    state_rows = []
    for label in DISPLAY_NAMES:
        state = states[label]
        for index, iteration in enumerate(state["iterations"]):
            history_rows.append(
                {
                    "run": state["run"],
                    "label": label,
                    "display_name": state["display_name"],
                    "iteration": int(iteration),
                    "relative_residual": state["residual"][index],
                    "density_error": state["density_error"][index],
                    "wall_seconds": state["wall_seconds"][index],
                    "cumulative_hours": state["cumulative_hours"][index],
                    "mu_density_converged": bool(state["mu_converged"][index]),
                }
            )
        minimum_index = int(np.argmin(state["residual"]))
        state_rows.append(
            {
                "run": state["run"],
                "label": label,
                "display_name": state["display_name"],
                "status": state["status"],
                "accepted": state["accepted"],
                "period": state["period"],
                "completed_updates": len(state["iterations"]),
                "final_relative_residual": state["residual"][-1],
                "minimum_relative_residual": state["residual"][minimum_index],
                "minimum_residual_iteration": int(state["iterations"][minimum_index]),
                "final_density_error": state["density_error"][-1],
                "recorded_wall_hours": float(np.sum(state["wall_seconds"]) / 3600.0),
                "seed_protocol": state["seed_protocol"],
                "seed_label": state["seed_label"],
                "stateless": state["stateless"],
                "model_fingerprint": state["model_fingerprint"],
                "numerical_fingerprint": state["numerical_fingerprint"],
                "implementation_fingerprint": state["implementation_fingerprint"],
                "state_path": state["path"],
            }
        )

    write_tsv(
        output / "history_comparison.tsv",
        history_rows,
        (
            "run", "label", "display_name", "iteration", "relative_residual",
            "density_error", "wall_seconds", "cumulative_hours", "mu_density_converged",
        ),
    )
    write_tsv(
        output / "state_comparison.tsv",
        state_rows,
        (
            "run", "label", "display_name", "status", "accepted", "period",
            "completed_updates", "final_relative_residual", "minimum_relative_residual",
            "minimum_residual_iteration", "final_density_error", "recorded_wall_hours",
            "seed_protocol", "seed_label", "stateless", "model_fingerprint",
            "numerical_fingerprint", "implementation_fingerprint", "state_path",
        ),
    )

    matched = states["unfrustrated__pairing_matched_m000_chi400"]
    broadband = states["unfrustrated__pairing_s2_chi400"]
    common = min(len(matched["residual"]), len(broadband["residual"]))
    metrics = [
        {
            "metric": "pairing_minimum_residual_reduction_fraction",
            "value": 1.0 - float(np.min(matched["residual"]) / np.min(broadband["residual"])),
        },
        {
            "metric": "pairing_iteration_9_residual_reduction_fraction",
            "value": 1.0 - float(matched["residual"][common - 1] / broadband["residual"][common - 1]),
        },
        {
            "metric": "pairing_first_n_residual_ratio_maximum",
            "value": float(np.max(matched["residual"][:common] / broadband["residual"][:common])),
        },
        {
            "metric": "matched_sdw_minimum_relative_residual",
            "value": float(np.min(states["unfrustrated__sdw_matched_m058_chi400"]["residual"])),
        },
    ]
    write_tsv(output / "headline_metrics.tsv", metrics, ("metric", "value"))
    print("output_directory={}".format(output))
    print("states={}".format(len(states)))
    print("figures=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
