#!/usr/bin/env python3
"""Audit spatial phase slips, multi-q content, and residual localization.

This is a local, read-only diagnostic for compact Phase 1 schema-v5 states.
It never changes an HDF5 artifact, convergence status, or budget ledger.  The
phase-slip flags are deliberately diagnostic: open boundaries and beating can
produce the same amplitude-zero/phase-jump signature.

Example from ``ladder_mps_mft/``::

    python scripts/audit_spatial_phase_defects.py \
      --output output/phase1_gpu/20260826_phase1_unfrustrated_pairing_recurrence_chi400/spatial-defect-audit-20260828 \
      output/phase1_gpu/20260826_phase1_unfrustrated_pairing_recurrence_chi400 \
      output/phase1_gpu/20260824_phase1_gpu_v3_float64_history
"""

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


BLUE = "#2563EB"
BLUE_DARK = "#1E3A8A"
ORANGE = "#D97706"
ORANGE_DARK = "#92400E"
OLIVE = "#6B7A18"
PINK = "#B83280"
INK = "#20242A"
GREY = "#6B7280"
LIGHT_GREY = "#E5E7EB"
SIGNAL_FLOOR = 1e-6
BLUE_ORANGE = LinearSegmentedColormap.from_list(
    "blue_orange", [BLUE_DARK, "#F8FAFC", ORANGE_DARK]
)


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


def _julia_array(dataset):
    """Restore the dimension order reported by HDF5.jl.

    HDF5.jl reverses array dimensions when serializing to the C HDF5 layout;
    h5py consequently exposes those dimensions in the opposite order.
    """
    values = np.asarray(dataset)
    if values.ndim <= 1:
        return values
    return values.transpose(tuple(reversed(range(values.ndim))))


def sha256_file(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def discover_states(run_directory):
    """Return the newest final state below each branch label."""
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
    if not states:
        raise ValueError("no state.h5 found below {}".format(results))
    return [(label, states[label][1]) for label in sorted(states)]


def _bond_to_rungs(values):
    """Map L-1 nearest-neighbor bond values to L rung-centered values."""
    time_count, bond_count = values.shape
    if bond_count == 0:
        return np.zeros((time_count, 1), dtype=float)
    result = np.empty((time_count, bond_count + 1), dtype=float)
    result[:, 0] = values[:, 0]
    result[:, -1] = values[:, -1]
    if bond_count > 1:
        result[:, 1:-1] = 0.5 * (values[:, :-1] + values[:, 1:])
    return result


def field_profiles(alpha, mu_cdw):
    """Construct signed rung profiles in charge, spin, and pairing channels."""
    length = alpha.shape[0]
    time_count = alpha.shape[-1]
    if alpha.shape != (length, length, 2, 2, time_count):
        raise ValueError("unexpected alpha history shape {}".format(alpha.shape))
    if mu_cdw.shape != (2, 2 * length, time_count):
        raise ValueError("unexpected mu_cdw history shape {}".format(mu_cdw.shape))

    down_leg_1 = mu_cdw[0, 0::2, :].T
    up_leg_1 = mu_cdw[1, 0::2, :].T
    down_leg_2 = mu_cdw[0, 1::2, :].T
    up_leg_2 = mu_cdw[1, 1::2, :].T
    charge_1 = 0.5 * (down_leg_1 + up_leg_1)
    charge_2 = 0.5 * (down_leg_2 + up_leg_2)
    spin_1 = 0.5 * (up_leg_1 - down_leg_1)
    spin_2 = 0.5 * (up_leg_2 - down_leg_2)

    rungs = np.arange(length)
    onsite_1 = alpha[rungs, rungs, 0, 0, :].T
    onsite_2 = alpha[rungs, rungs, 1, 1, :].T
    rung_pair = alpha[rungs, rungs, 0, 1, :].T

    if length > 1:
        left = np.arange(length - 1)
        right = left + 1
        leg_1_bond = alpha[left, right, 0, 0, :].T
        leg_2_bond = alpha[left, right, 1, 1, :].T
        leg_even = _bond_to_rungs(0.5 * (leg_1_bond + leg_2_bond))
        leg_odd = _bond_to_rungs(0.5 * (leg_1_bond - leg_2_bond))
    else:
        leg_even = np.zeros((time_count, length))
        leg_odd = np.zeros((time_count, length))

    onsite_even = 0.5 * (onsite_1 + onsite_2)
    onsite_odd = 0.5 * (onsite_1 - onsite_2)
    return {
        "charge_even": 0.5 * (charge_1 + charge_2),
        "charge_odd": 0.5 * (charge_1 - charge_2),
        "spin_even": 0.5 * (spin_1 + spin_2),
        "spin_odd": 0.5 * (spin_1 - spin_2),
        "pair_onsite_even": onsite_even,
        "pair_onsite_odd": onsite_odd,
        "pair_rung": rung_pair,
        "pair_leg_even": leg_even,
        "pair_leg_odd": leg_odd,
        "pair_extended_s": leg_even + rung_pair,
        "pair_d": leg_even - rung_pair,
    }


GROUP_CANDIDATES = {
    "charge": ("charge_even", "charge_odd"),
    "spin": ("spin_even", "spin_odd"),
    "pairing": (
        "pair_onsite_even",
        "pair_onsite_odd",
        "pair_rung",
        "pair_leg_even",
        "pair_leg_odd",
        "pair_extended_s",
        "pair_d",
    ),
}


def select_group_channels(profiles):
    selected = {}
    for group, candidates in GROUP_CANDIDATES.items():
        scores = {}
        for name in candidates:
            values = profiles[name][-min(3, profiles[name].shape[0]) :]
            if group in ("charge", "spin"):
                values = values - np.mean(values, axis=1, keepdims=True)
            scores[name] = float(np.median(np.sqrt(np.mean(values * values, axis=1))))
        selected[group] = max(candidates, key=lambda name: (scores[name], name))
    return selected


def bulk_weights(length, fraction=0.75):
    if not (0.0 < fraction <= 1.0):
        raise ValueError("bulk fraction must be in (0, 1]")
    count = max(3, min(length, int(round(length * fraction))))
    start = (length - count) // 2
    result = np.zeros(length, dtype=float)
    window = np.hanning(count + 2)[1:-1]
    result[start : start + count] = window
    return result


def spatial_spectrum(values, subtract_mean, fraction=0.75):
    """Phase-independent half-spectrum on the open finite interval."""
    values = np.asarray(values, dtype=float)
    length = values.size
    x = np.arange(length, dtype=float)
    q = np.linspace(0.0, math.pi, length)
    weights = bulk_weights(length, fraction)
    work = values.copy()
    if subtract_mean:
        work -= np.average(work, weights=weights)
    weighted = weights * work
    transform = np.exp(-1j * np.outer(q, x)).dot(weighted)
    power = np.abs(transform) ** 2
    total = float(np.sum(power))
    if not np.isfinite(total) or total <= np.finfo(float).eps:
        return q, np.zeros_like(power), {
            "q": float("nan"),
            "q_over_pi": float("nan"),
            "peak_share": 0.0,
            "peak_band_share": 0.0,
            "second_peak_ratio": 0.0,
            "spectral_entropy": 0.0,
        }
    normalized = power / total
    order = np.argsort(normalized)[::-1]
    peak = int(order[0])
    separated = [index for index in order[1:] if abs(int(index) - peak) > 2]
    second = int(separated[0]) if separated else peak
    band_start = max(0, peak - 2)
    band_stop = min(len(normalized), peak + 3)
    positive = normalized[normalized > 0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(len(normalized))
    peak_share = float(normalized[peak])
    return q, normalized, {
        "q": float(q[peak]),
        "q_over_pi": float(q[peak] / math.pi),
        "peak_share": peak_share,
        "peak_band_share": float(np.sum(normalized[band_start:band_stop])),
        "second_peak_ratio": float(normalized[second] / peak_share),
        "spectral_entropy": entropy,
    }


def profile_signal_rms(values, subtract_mean):
    values = np.asarray(values, dtype=float)
    if subtract_mean:
        values = values - np.mean(values)
    return float(np.sqrt(np.mean(values * values)))


def demodulated_envelope(values, q, subtract_mean, half_width=None):
    values = np.asarray(values, dtype=float)
    length = values.size
    weights = bulk_weights(length, 0.75)
    work = values.copy()
    if subtract_mean:
        work -= np.average(work, weights=weights)
    if half_width is None:
        half_width = max(4, length // 12)
    x = np.arange(length, dtype=float)
    envelope = np.empty(length, dtype=complex)
    for center in range(length):
        distance = (x - center) / max(1.0, 0.55 * half_width)
        local = np.exp(-0.5 * distance * distance)
        local[np.abs(x - center) > half_width] = 0.0
        denominator = np.sum(local)
        envelope[center] = np.sum(local * work * np.exp(-1j * q * x)) / denominator
    return envelope


def wrapped_difference(right, left):
    return float(np.angle(np.exp(1j * (right - left))))


def phase_slip_candidates(envelope, bulk_fraction=0.75, amplitude_ratio=0.45, phase_jump=0.45 * math.pi):
    """Find diagnostic amplitude-zero/phase-jump coincidences."""
    amplitude = np.abs(envelope)
    phase = np.angle(envelope)
    length = amplitude.size
    bulk_count = max(3, int(round(length * bulk_fraction)))
    bulk_start = (length - bulk_count) // 2
    bulk_stop = bulk_start + bulk_count
    reference = float(np.median(amplitude[bulk_start:bulk_stop]))
    if not np.isfinite(reference) or reference <= 100 * np.finfo(float).eps:
        return []
    span = max(2, length // 24)
    candidates = []
    lower = max(bulk_start + span, 2)
    upper = min(bulk_stop - span, length - 2)
    for index in range(lower, upper):
        local = amplitude[index - 2 : index + 3]
        if amplitude[index] > np.min(local) + 10 * np.finfo(float).eps:
            continue
        ratio = float(amplitude[index] / reference)
        jump = abs(wrapped_difference(phase[index + span], phase[index - span]))
        if ratio <= amplitude_ratio and jump >= phase_jump:
            candidates.append(
                {
                    "rung": index + 1,
                    "amplitude_ratio": ratio,
                    "phase_jump_over_pi": float(jump / math.pi),
                    "score": float((1.0 - ratio) * jump / math.pi),
                }
            )
    return candidates


def _center_of_mass_mass(matrix_mass):
    """Bin T x L x L nonnegative pair mass by pair center of mass."""
    time_count, length, _ = matrix_mass.shape
    result = np.zeros((time_count, length), dtype=float)
    for left in range(length):
        for right in range(length):
            center = 0.5 * (left + right)
            lower = int(math.floor(center))
            upper = int(math.ceil(center))
            if lower == upper:
                result[:, lower] += matrix_mass[:, left, right]
            else:
                upper_weight = center - lower
                result[:, lower] += (1.0 - upper_weight) * matrix_mass[:, left, right]
                result[:, upper] += upper_weight * matrix_mass[:, left, right]
    return result


def residual_by_rung(applied, measured):
    delta_alpha = measured["alpha"] - applied["alpha"]
    delta_beta = measured["beta"] - applied["beta"]
    delta_mu = measured["mu_cdw"] - applied["mu_cdw"]
    length = delta_alpha.shape[0]

    alpha_matrix = np.sum(delta_alpha * delta_alpha, axis=(2, 3)).transpose(2, 0, 1)
    beta_matrix = np.sum(delta_beta * delta_beta, axis=(0, 3, 4)).transpose(2, 0, 1)
    alpha_mass = _center_of_mass_mass(alpha_matrix)
    beta_mass = _center_of_mass_mass(beta_matrix)
    mu_mass = np.sum(delta_mu * delta_mu, axis=0).reshape(length, 2, -1).sum(axis=1).T
    total = alpha_mass + beta_mass + mu_mass
    denominators = np.sum(total, axis=1, keepdims=True)
    normalized = np.divide(total, denominators, out=np.zeros_like(total), where=denominators > 0)
    return {
        "alpha_mass": alpha_mass,
        "beta_mass": beta_mass,
        "mu_mass": mu_mass,
        "total_mass": total,
        "normalized_total": normalized,
    }


def field_relative_distance(left, right):
    time_count = next(iter(left.values())).shape[-1]
    delta_squared = np.zeros(time_count, dtype=float)
    left_squared = np.zeros(time_count, dtype=float)
    right_squared = np.zeros(time_count, dtype=float)
    for name in ("alpha", "beta", "mu_cdw"):
        axes = tuple(range(left[name].ndim - 1))
        delta_squared += np.sum((right[name] - left[name]) ** 2, axis=axes)
        left_squared += np.sum(left[name] ** 2, axis=axes)
        right_squared += np.sum(right[name] ** 2, axis=axes)
    denominator = np.maximum(np.maximum(np.sqrt(left_squared), np.sqrt(right_squared)), np.finfo(float).eps)
    return np.sqrt(delta_squared) / denominator


def same_phase_residual_by_rung(fields, period):
    """Compare fields separated by one full candidate-orbit period."""
    time_count = next(iter(fields.values())).shape[-1]
    if not (1 <= period < time_count):
        raise ValueError("period must be positive and shorter than the history")
    earlier = {name: values[..., :-period] for name, values in fields.items()}
    later = {name: values[..., period:] for name, values in fields.items()}
    core = residual_by_rung(earlier, later)
    length = core["total_mass"].shape[1]
    padded = {}
    for name, values in core.items():
        target = np.full((time_count, length), np.nan, dtype=float)
        target[period:, :] = values
        padded[name] = target
    relative = np.full(time_count, np.nan, dtype=float)
    relative[period:] = field_relative_distance(earlier, later)
    padded["global_relative"] = relative
    return padded


def load_state(run_name, label, state_path):
    with h5py.File(state_path, "r") as handle:
        iterations = np.asarray(handle["history/iteration"], dtype=int)
        required = (
            "history/fields/applied/alpha",
            "history/fields/applied/beta",
            "history/fields/applied/mu_cdw",
            "history/fields/measured/alpha",
            "history/fields/measured/beta",
            "history/fields/measured/mu_cdw",
        )
        missing = [path for path in required if path not in handle]
        if missing:
            raise ValueError("{} lacks complete field history: {}".format(state_path, ", ".join(missing)))
        applied = {
            name: np.asarray(_julia_array(handle["history/fields/applied/{}".format(name)]), dtype=float)
            for name in ("alpha", "beta", "mu_cdw")
        }
        measured = {
            name: np.asarray(_julia_array(handle["history/fields/measured/{}".format(name)]), dtype=float)
            for name in ("alpha", "beta", "mu_cdw")
        }
        update_mode = [_decode(value) for value in np.asarray(handle["history/update_mode"])]
        metadata = {
            "run": run_name,
            "label": label,
            "state_path": str(state_path.resolve()),
            "status": str(_scalar(handle, "status", "unknown")),
            "accepted": bool(_scalar(handle, "accepted", False)),
            "period": int(_scalar(handle, "fundamental_period", 0)),
            "geometry": str(_scalar(handle, "model/transverse_geometry", "unknown")),
            "L": int(_scalar(handle, "model/L", applied["alpha"].shape[0])),
            "U": float(_scalar(handle, "model/U", float("nan"))),
            "V": float(_scalar(handle, "model/V", float("nan"))),
            "t": float(_scalar(handle, "model/t", float("nan"))),
            "t0": float(_scalar(handle, "model/t0", float("nan"))),
            "tp": float(_scalar(handle, "model/tp", float("nan"))),
            "density": float(_scalar(handle, "model/density", float("nan"))),
            "chi": int(_scalar(handle, "provenance/maximum_bond_dimension", 0) or 0),
            "seed": str(_scalar(handle, "provenance/initial_seed", "unknown")),
            "seed_label": str(_scalar(handle, "provenance/seed_label", "unknown")),
            "random_seed": int(_scalar(handle, "provenance/random_seed", 0)),
            "stateless": bool(_scalar(handle, "analysis_storage/is_stateless_copy", False)),
            "full_artifact_path": str(_scalar(handle, "analysis_storage/full_artifact_path", "")),
            "full_artifact_sha256": str(_scalar(handle, "analysis_storage/full_artifact_sha256", "")),
            "model_fingerprint": str(_scalar(handle, "provenance/model_fingerprint", "")),
            "numerical_fingerprint": str(_scalar(handle, "provenance/numerical_fingerprint", "")),
            "implementation_fingerprint": str(_scalar(handle, "provenance/implementation_sha256", "")),
        }
        histories = {
            "field_rel_residual": np.asarray(handle["history/field_rel_residual"], dtype=float),
            "variational_energy": np.asarray(handle["history/variational_energy"], dtype=float),
        }
    metadata["compact_sha256"] = sha256_file(state_path)
    return {
        "metadata": metadata,
        "iterations": iterations,
        "update_mode": update_mode,
        "applied": applied,
        "measured": measured,
        "histories": histories,
    }


def analyze_state(state, bulk_fraction):
    profiles = field_profiles(state["measured"]["alpha"], state["measured"]["mu_cdw"])
    selected = select_group_channels(profiles)
    raw_link_residual = residual_by_rung(state["applied"], state["measured"])
    if state["metadata"]["period"] == 2 and len(state["iterations"]) > 2:
        residual = same_phase_residual_by_rung(state["measured"], 2)
        residual_kind = "same_phase_period_2"
    else:
        residual = raw_link_residual
        residual["global_relative"] = state["histories"]["field_rel_residual"]
        residual_kind = "raw_map_link"
    iterations = state["iterations"]
    group_results = {}
    history_rows = []
    candidate_rows = []

    for group in ("charge", "spin", "pairing"):
        name = selected[group]
        values = profiles[name]
        subtract_mean = group in ("charge", "spin")
        spectra = []
        spectrum_metrics = []
        signal_rms_history = []
        for iteration_index, profile in enumerate(values):
            q, power, metrics = spatial_spectrum(profile, subtract_mean, bulk_fraction)
            signal_rms = profile_signal_rms(profile, subtract_mean)
            signal_rms_history.append(signal_rms)
            spectra.append(power)
            spectrum_metrics.append(metrics)
            history_rows.append(
                {
                    "run": state["metadata"]["run"],
                    "label": state["metadata"]["label"],
                    "group": group,
                    "channel": name,
                    "iteration": int(iterations[iteration_index]),
                    "q_over_pi": metrics["q_over_pi"],
                    "peak_share": metrics["peak_share"],
                    "peak_band_share": metrics["peak_band_share"],
                    "second_peak_ratio": metrics["second_peak_ratio"],
                    "spectral_entropy": metrics["spectral_entropy"],
                    "signal_rms": signal_rms,
                    "signal_resolved": signal_rms >= SIGNAL_FLOOR,
                }
            )
        final_metrics = spectrum_metrics[-1]
        final_q = final_metrics["q"]
        if not np.isfinite(final_q):
            final_q = 0.0
        envelopes = []
        candidates_by_iteration = []
        for iteration_index, profile in enumerate(values):
            envelope = demodulated_envelope(profile, final_q, subtract_mean)
            candidates = (
                phase_slip_candidates(envelope, bulk_fraction)
                if signal_rms_history[iteration_index] >= SIGNAL_FLOOR
                else []
            )
            envelopes.append(envelope)
            candidates_by_iteration.append(candidates)
            for candidate in candidates:
                row = {
                    "run": state["metadata"]["run"],
                    "label": state["metadata"]["label"],
                    "group": group,
                    "channel": name,
                    "iteration": int(iterations[iteration_index]),
                }
                row.update(candidate)
                candidate_rows.append(row)
        strongest_positions = [
            max(candidates, key=lambda item: item["score"])["rung"] if candidates else float("nan")
            for candidates in candidates_by_iteration
        ]
        finite_positions = np.asarray([value for value in strongest_positions if np.isfinite(value)])
        coverage = float(len(finite_positions) / len(strongest_positions))
        drift = float(finite_positions[-1] - finite_positions[0]) if len(finite_positions) >= 2 else float("nan")
        final_candidates = candidates_by_iteration[-1]
        final_envelope = envelopes[-1]
        group_results[group] = {
            "channel": name,
            "values": values,
            "q": q,
            "spectra": np.asarray(spectra),
            "spectrum_metrics": spectrum_metrics,
            "signal_rms_history": signal_rms_history,
            "final_signal_rms": signal_rms_history[-1],
            "signal_resolved": signal_rms_history[-1] >= SIGNAL_FLOOR,
            "final_q": final_q,
            "final_envelope": final_envelope,
            "candidates_by_iteration": candidates_by_iteration,
            "strongest_positions": strongest_positions,
            "final_candidates": final_candidates,
            "candidate_coverage": coverage,
            "candidate_drift": drift,
        }

    residual_map = residual["normalized_total"]
    residual_peak = np.asarray(
        [int(np.nanargmax(row)) + 1 if np.any(np.isfinite(row)) else float("nan") for row in residual_map],
        dtype=float,
    )
    residual_ipr = np.nansum(residual_map * residual_map, axis=1)
    final_residual_peak = int(residual_peak[-1])
    for result in group_results.values():
        result["final_candidate_residual_distance"] = (
            min(abs(candidate["rung"] - final_residual_peak) for candidate in result["final_candidates"])
            if result["final_candidates"]
            else float("nan")
        )
    return {
        "profiles": profiles,
        "selected": selected,
        "groups": group_results,
        "residual": residual,
        "raw_link_residual": raw_link_residual,
        "residual_kind": residual_kind,
        "residual_peak": residual_peak,
        "residual_ipr": residual_ipr,
        "history_rows": history_rows,
        "candidate_rows": candidate_rows,
    }


def _safe_slug(value):
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _heatmap(ax, values, iterations, x_extent, cmap, symmetric=False, log_power=False):
    data = np.asarray(values, dtype=float)
    if log_power:
        data = np.log10(np.maximum(data, 1e-12))
        vmin, vmax = -6.0, 0.0
    elif symmetric:
        limit = float(np.nanpercentile(np.abs(data), 99.0))
        limit = max(limit, np.finfo(float).eps)
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = None, None
    extent = [x_extent[0], x_extent[1], iterations[0] - 0.5, iterations[-1] + 0.5]
    image = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return image


def plot_state(state, analysis, output_path, bulk_fraction):
    metadata = state["metadata"]
    iterations = state["iterations"]
    length = metadata["L"]
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 14.0), constrained_layout=True)
    group_colors = {"charge": BLUE, "spin": ORANGE, "pairing": OLIVE}

    for row, group in enumerate(("charge", "spin", "pairing")):
        result = analysis["groups"][group]
        values = result["values"]
        profile_ax = axes[row, 0]
        image = _heatmap(
            profile_ax,
            values,
            iterations,
            (0.5, length + 0.5),
            BLUE_ORANGE,
            symmetric=True,
        )
        profile_ax.set_title("{} profile history: {}".format(group.capitalize(), result["channel"]))
        profile_ax.set_ylabel("MF iteration")
        profile_ax.set_xlabel("Rung")
        fig.colorbar(image, ax=profile_ax, shrink=0.78, label="Measured field")
        for iteration, candidates in zip(iterations, result["candidates_by_iteration"]):
            if candidates:
                strongest = max(candidates, key=lambda item: item["score"])
                profile_ax.plot(strongest["rung"], iteration, marker="o", ms=3.2, color=INK)

        spectrum_ax = axes[row, 1]
        image = _heatmap(
            spectrum_ax,
            result["spectra"],
            iterations,
            (0.0, 1.0),
            "Blues",
            log_power=True,
        )
        dominant = [item["q_over_pi"] for item in result["spectrum_metrics"]]
        spectrum_ax.plot(dominant, iterations, color=ORANGE_DARK, lw=1.4, marker=".", ms=3)
        spectrum_ax.set_title("Bulk-window spatial spectrum")
        spectrum_ax.set_xlabel("q / pi")
        spectrum_ax.set_ylabel("MF iteration")
        fig.colorbar(image, ax=spectrum_ax, shrink=0.78, label="log10 power share")

        envelope_ax = axes[row, 2]
        envelope = result["final_envelope"]
        amplitude = np.abs(envelope)
        bulk = bulk_weights(length, bulk_fraction) > 0
        reference = float(np.median(amplitude[bulk])) if np.any(bulk) else float(np.median(amplitude))
        normalized_amplitude = amplitude / reference if reference > np.finfo(float).eps else amplitude
        x = np.arange(1, length + 1)
        envelope_ax.plot(x, normalized_amplitude, color=group_colors[group], lw=1.8, label="Envelope amplitude / bulk median")
        envelope_ax.axhline(0.45, color=GREY, lw=1.0, ls=":", label="Candidate threshold")
        envelope_ax.set_xlabel("Rung")
        envelope_ax.set_ylabel("Relative amplitude")
        envelope_ax.set_ylim(bottom=0)
        phase_ax = envelope_ax.twinx()
        phase_ax.plot(x, np.unwrap(np.angle(envelope)) / math.pi, color=ORANGE_DARK, lw=1.2, ls="--", label="Unwrapped phase / pi")
        phase_ax.set_ylabel("Phase / pi")
        for candidate in result["final_candidates"]:
            envelope_ax.axvline(candidate["rung"], color=INK, lw=1.0, ls="-.")
        resolution_note = "resolved" if result["signal_resolved"] else "below {:.0e} floor".format(SIGNAL_FLOOR)
        envelope_ax.set_title(
            "Final demodulated envelope (q/pi={:.3f}, candidates={}, {})".format(
                result["final_q"] / math.pi, len(result["final_candidates"]), resolution_note
            )
        )
        lines = envelope_ax.get_lines()[:2] + phase_ax.get_lines()[:1]
        envelope_ax.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=7)

    residual_ax = axes[3, 0]
    image = _heatmap(
        residual_ax,
        analysis["residual"]["normalized_total"],
        iterations,
        (0.5, length + 0.5),
        "Oranges",
    )
    residual_ax.plot(analysis["residual_peak"], iterations, color=INK, lw=1.2, ls="--", label="Residual peak")
    residual_title = (
        "Same-phase period-2 change by rung"
        if analysis["residual_kind"] == "same_phase_period_2"
        else "Raw-map link residual mass by rung"
    )
    residual_ax.set_title(residual_title)
    residual_ax.set_xlabel("Rung")
    residual_ax.set_ylabel("MF iteration")
    residual_ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(image, ax=residual_ax, shrink=0.78, label="Fraction of squared field residual")

    convergence_ax = axes[3, 1]
    residual_history = state["histories"]["field_rel_residual"]
    convergence_ax.semilogy(iterations, residual_history, color=BLUE_DARK, marker="o", ms=3.5, label="Global relative residual")
    if analysis["residual_kind"] == "same_phase_period_2":
        convergence_ax.semilogy(
            iterations,
            analysis["residual"]["global_relative"],
            color=PINK,
            marker="^",
            ms=3.2,
            label="Same-phase 2-step field change",
        )
    convergence_ax.set_xlabel("MF iteration")
    convergence_ax.set_ylabel("Relative field residual")
    convergence_ax.grid(True, color=LIGHT_GREY, lw=0.7)
    energy_ax = convergence_ax.twinx()
    energies = state["histories"]["variational_energy"]
    finite = np.isfinite(energies)
    if np.any(finite):
        reference_energy = energies[np.flatnonzero(finite)[-1]]
        energy_shift = np.abs(energies - reference_energy) / max(1, length)
        energy_ax.semilogy(iterations, np.maximum(energy_shift, 1e-16), color=ORANGE_DARK, marker="s", ms=3.0, ls="--", label="|E-E_final| / L")
    energy_ax.set_ylabel("Absolute energy shift per rung")
    convergence_ax.set_title("Convergence history (diagnostic only)")
    lines = convergence_ax.get_lines() + energy_ax.get_lines()
    convergence_ax.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)

    trajectory_ax = axes[3, 2]
    for group, marker, color in (
        ("charge", "o", BLUE),
        ("spin", "s", ORANGE),
        ("pairing", "^", OLIVE),
    ):
        positions = np.asarray(analysis["groups"][group]["strongest_positions"], dtype=float)
        trajectory_ax.scatter(iterations, positions, marker=marker, color=color, s=24, label="{} phase-slip candidate".format(group))
    trajectory_ax.plot(iterations, analysis["residual_peak"], color=INK, lw=1.2, ls="--", label="Residual peak rung")
    trajectory_ax.set_xlabel("MF iteration")
    trajectory_ax.set_ylabel("Rung")
    trajectory_ax.set_ylim(0.5, length + 0.5)
    trajectory_ax.set_title("Candidate and residual locations")
    trajectory_ax.grid(True, color=LIGHT_GREY, lw=0.7)
    trajectory_ax.legend(loc="best", fontsize=7)

    subtitle = (
        "{} | {} | status={} accepted={} period={} | L={} U={} V={} t0={} tp={} density={} | {} records"
    ).format(
        metadata["run"],
        metadata["label"],
        metadata["status"],
        metadata["accepted"],
        metadata["period"],
        metadata["L"],
        metadata["U"],
        metadata["V"],
        metadata["t0"],
        metadata["tp"],
        metadata["density"],
        len(iterations),
    )
    fig.suptitle("Spatial phase-defect diagnostic\n" + subtitle, fontsize=13, color=INK)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_tsv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, states_and_analyses, source_rows, bulk_fraction):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("# Spatial phase-defect audit\n\n")
        stream.write("This is a local diagnostic of compact schema-v5 mean-field histories. It does not change or replace the canonical raw-map acceptance, recurrence, fingerprint, or variational-energy gates.\n\n")
        stream.write("## Question and diagnostic rule\n\n")
        stream.write("The audit asks whether an apparent modulation contains a localized envelope minimum plus a phase jump, and whether that feature persists or moves with the recurrence-aware spatial residual. Fixed-point searches use the raw link f(x)-x; period-two candidates use the same-phase two-step change. Such a coincidence supports a phase-slip/domain-wall interpretation, but does not prove one: open-boundary Friedel structure and beating between nearby wavevectors can generate the same signature.\n\n")
        stream.write("Spectra use a Hann-weighted central {:.0f}% window and the finite-interval half-spectrum 0 <= q <= pi. Charge and spin means are removed; pairing retains q=0. A phase-slip candidate requires signal RMS >= {:.0e}, envelope amplitude <= 0.45 of its bulk median, and a phase jump >= 0.45 pi. The reported second peak excludes the two bins adjacent to the dominant peak.\n\n".format(100 * bulk_fraction, SIGNAL_FLOOR))
        stream.write("## State/channel summary\n\n")
        stream.write("| Run | Branch | Outcome | Group | Selected profile | Signal RMS | q/pi | Peak band | Resolved 2nd/1st | Entropy | Final candidates | Residual distance | Coverage | Drift |\n")
        stream.write("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for state, analysis, _ in states_and_analyses:
            metadata = state["metadata"]
            outcome = "{} / accepted={}".format(metadata["status"], metadata["accepted"])
            for group in ("charge", "spin", "pairing"):
                result = analysis["groups"][group]
                metrics = result["spectrum_metrics"][-1]
                stream.write(
                    "| `{}` | `{}` | `{}` | {} | `{}` | {:.3e} | {:.4f} | {:.3f} | {:.3f} | {:.3f} | {} | {} | {:.2f} | {} |\n".format(
                        metadata["run"],
                        metadata["label"],
                        outcome,
                        group,
                        result["channel"],
                        result["final_signal_rms"],
                        metrics["q_over_pi"],
                        metrics["peak_band_share"],
                        metrics["second_peak_ratio"],
                        metrics["spectral_entropy"],
                        len(result["final_candidates"]),
                        "{:.1f}".format(result["final_candidate_residual_distance"]) if np.isfinite(result["final_candidate_residual_distance"]) else "n/a",
                        result["candidate_coverage"],
                        "{:.1f}".format(result["candidate_drift"]) if np.isfinite(result["candidate_drift"]) else "n/a",
                    )
                )
        focus_run = states_and_analyses[0][0]["metadata"]["run"]
        stream.write("\n## Focus-run observations\n\n")
        stream.write("The first supplied campaign, `{}`, is treated as the focus run. A moving-wall diagnosis would be strengthened by phase-slip coverage over at least half the history and a final candidate within four rungs of the recurrence-aware residual peak. This is a conservative diagnostic heuristic, not an acceptance gate.\n\n".format(focus_run))
        combined_support = False
        for state, analysis, _ in states_and_analyses:
            metadata = state["metadata"]
            if metadata["run"] != focus_run:
                continue
            total = float(np.nansum(analysis["residual"]["total_mass"][-1]))
            denominator = total if total > 0 else 1.0
            fractions = {
                name: float(np.nansum(analysis["residual"][name + "_mass"][-1]) / denominator)
                for name in ("alpha", "beta", "mu")
            }
            candidate_descriptions = []
            for group in ("charge", "spin", "pairing"):
                result = analysis["groups"][group]
                if result["final_candidates"]:
                    positions = ",".join(str(item["rung"]) for item in result["final_candidates"])
                    candidate_descriptions.append(
                        "{} at rung(s) {} (coverage {:.2f}, nearest residual distance {:.1f})".format(
                            group,
                            positions,
                            result["candidate_coverage"],
                            result["final_candidate_residual_distance"],
                        )
                    )
                    combined_support = combined_support or (
                        result["candidate_coverage"] >= 0.5
                        and result["final_candidate_residual_distance"] <= 4
                    )
            candidate_text = "; ".join(candidate_descriptions) if candidate_descriptions else "no final phase-slip candidates"
            stream.write(
                "- `{}`: stored period-one relative residual `{:.3e}`; `{}` diagnostic relative change `{:.3e}`, peak rung `{}`, IPR `{:.3f}`; squared diagnostic fractions alpha/beta/Hartree = `{:.3f}/{:.3f}/{:.3f}`; {}.\n".format(
                    metadata["label"],
                    state["histories"]["field_rel_residual"][-1],
                    analysis["residual_kind"],
                    analysis["residual"]["global_relative"][-1],
                    int(analysis["residual_peak"][-1]),
                    analysis["residual_ipr"][-1],
                    fractions["alpha"],
                    fractions["beta"],
                    fractions["mu"],
                    candidate_text,
                )
            )
        stream.write("\n")
        if combined_support:
            stream.write("At least one focus state passes the combined persistence/co-localization heuristic. This supports targeted follow-up, but open-boundary and multi-q alternatives remain unresolved.\n")
        else:
            stream.write("No focus state passes the combined persistence/co-localization heuristic. These histories therefore do not currently support a moving domain wall as the primary cause of their nonconvergence.\n")
        stream.write("\n## Verification boundary\n\n")
        stream.write("- Every input listed in `source_inventory.tsv` was opened read-only and SHA-256 hashed locally.\n")
        stream.write("- The analysis uses only stored fields and scalar histories; compact inputs contain no MPS tensors.\n")
        stream.write("- Recorded full scratch paths and hashes are provenance only. Full Perlmutter artifacts were not mounted or reverified locally.\n")
        stream.write("- The Fourier-like spectra are one-point field diagnostics, not connected structure factors or thermodynamic order parameters.\n")
        stream.write("- Central-window or phase-aligned diagnostics cannot be used for energy ranking. Only accepted, fingerprint-matched states at the same transverse geometry may enter the canonical variational comparison.\n")
        stream.write("- Legacy runs without complete stored field histories cannot be audited for wall motion from terminal profiles alone.\n")
        stream.write("\n## Outputs\n\n")
        stream.write("- `figures/*.png`: profile histories, spectra, final demodulated envelopes, convergence, and residual localization.\n")
        stream.write("- `channel_summary.tsv`: final quantitative diagnostics per selected channel.\n")
        stream.write("- `history_metrics.tsv`: dominant wavevector and spectral metrics at every stored MF iteration.\n")
        stream.write("- `residual_history.tsv`: recurrence-aware residual peak, localization, and field-component fractions at every stored MF iteration.\n")
        stream.write("- `phase_slip_candidates.tsv`: every thresholded amplitude-zero/phase-jump coincidence.\n")
        stream.write("- `state_summary.tsv`: state-level residual localization and provenance.\n")
        stream.write("- `source_inventory.tsv`: compact and recorded full-artifact hashes.\n")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", nargs="+", help="Phase 1 campaign directories")
    parser.add_argument("--output", required=True, help="New output directory; existing paths are refused")
    parser.add_argument("--bulk-fraction", type=float, default=0.75, help="central spectrum/envelope fraction (default: 0.75)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite existing output: {}".format(output))
    output.mkdir(parents=True)
    figure_directory = output / "figures"
    figure_directory.mkdir()

    states_and_analyses = []
    source_rows = []
    state_rows = []
    channel_rows = []
    history_rows = []
    residual_rows = []
    candidate_rows = []

    for run_directory in args.run_directories:
        run_path = Path(run_directory).resolve()
        run_name = run_path.name
        for label, state_path in discover_states(run_path):
            print("analyzing {} / {}".format(run_name, label), flush=True)
            state = load_state(run_name, label, state_path)
            analysis = analyze_state(state, args.bulk_fraction)
            figure_name = "{}__{}.png".format(_safe_slug(run_name), _safe_slug(label))
            figure_path = figure_directory / figure_name
            plot_state(state, analysis, figure_path, args.bulk_fraction)
            states_and_analyses.append((state, analysis, figure_path))
            metadata = state["metadata"]
            source_rows.append(
                {
                    "run": run_name,
                    "label": label,
                    "state_path": metadata["state_path"],
                    "compact_sha256": metadata["compact_sha256"],
                    "stateless": metadata["stateless"],
                    "full_artifact_path": metadata["full_artifact_path"],
                    "full_artifact_sha256": metadata["full_artifact_sha256"],
                }
            )
            final_total_residual_mass = float(np.nansum(analysis["residual"]["total_mass"][-1]))
            final_denominator = final_total_residual_mass if final_total_residual_mass > 0 else 1.0
            state_rows.append(
                {
                    "run": run_name,
                    "label": label,
                    "status": metadata["status"],
                    "accepted": metadata["accepted"],
                    "period": metadata["period"],
                    "geometry": metadata["geometry"],
                    "iterations": len(state["iterations"]),
                    "stored_final_period_one_relative_residual": state["histories"]["field_rel_residual"][-1],
                    "diagnostic_residual_kind": analysis["residual_kind"],
                    "final_diagnostic_relative_residual": analysis["residual"]["global_relative"][-1],
                    "final_residual_peak_rung": int(analysis["residual_peak"][-1]),
                    "final_residual_ipr": float(analysis["residual_ipr"][-1]),
                    "final_alpha_residual_fraction": float(np.nansum(analysis["residual"]["alpha_mass"][-1]) / final_denominator),
                    "final_beta_residual_fraction": float(np.nansum(analysis["residual"]["beta_mass"][-1]) / final_denominator),
                    "final_mu_residual_fraction": float(np.nansum(analysis["residual"]["mu_mass"][-1]) / final_denominator),
                    "model_fingerprint": metadata["model_fingerprint"],
                    "numerical_fingerprint": metadata["numerical_fingerprint"],
                    "implementation_fingerprint": metadata["implementation_fingerprint"],
                    "figure": str(figure_path),
                }
            )
            for group in ("charge", "spin", "pairing"):
                result = analysis["groups"][group]
                metrics = result["spectrum_metrics"][-1]
                channel_rows.append(
                    {
                        "run": run_name,
                        "label": label,
                        "group": group,
                        "channel": result["channel"],
                        "final_signal_rms": result["final_signal_rms"],
                        "signal_resolved": result["signal_resolved"],
                        "final_q_over_pi": metrics["q_over_pi"],
                        "final_peak_share": metrics["peak_share"],
                        "final_peak_band_share": metrics["peak_band_share"],
                        "final_second_peak_ratio": metrics["second_peak_ratio"],
                        "final_spectral_entropy": metrics["spectral_entropy"],
                        "final_phase_slip_candidates": len(result["final_candidates"]),
                        "final_candidate_residual_distance_rungs": result["final_candidate_residual_distance"],
                        "candidate_history_coverage": result["candidate_coverage"],
                        "strongest_candidate_drift_rungs": result["candidate_drift"],
                    }
                )
            history_rows.extend(analysis["history_rows"])
            candidate_rows.extend(analysis["candidate_rows"])
            for index, iteration in enumerate(state["iterations"]):
                total = float(np.nansum(analysis["residual"]["total_mass"][index]))
                denominator = total if total > 0 else float("nan")
                residual_rows.append(
                    {
                        "run": run_name,
                        "label": label,
                        "iteration": int(iteration),
                        "stored_period_one_relative_residual": state["histories"]["field_rel_residual"][index],
                        "diagnostic_residual_kind": analysis["residual_kind"],
                        "diagnostic_relative_residual": analysis["residual"]["global_relative"][index],
                        "residual_peak_rung": analysis["residual_peak"][index],
                        "residual_ipr": float(analysis["residual_ipr"][index]),
                        "alpha_residual_fraction": float(np.nansum(analysis["residual"]["alpha_mass"][index]) / denominator),
                        "beta_residual_fraction": float(np.nansum(analysis["residual"]["beta_mass"][index]) / denominator),
                        "mu_residual_fraction": float(np.nansum(analysis["residual"]["mu_mass"][index]) / denominator),
                    }
                )

    _write_tsv(
        output / "source_inventory.tsv",
        source_rows,
        ("run", "label", "state_path", "compact_sha256", "stateless", "full_artifact_path", "full_artifact_sha256"),
    )
    _write_tsv(
        output / "state_summary.tsv",
        state_rows,
        (
            "run", "label", "status", "accepted", "period", "geometry", "iterations",
            "stored_final_period_one_relative_residual", "diagnostic_residual_kind",
            "final_diagnostic_relative_residual", "final_residual_peak_rung", "final_residual_ipr",
            "final_alpha_residual_fraction", "final_beta_residual_fraction", "final_mu_residual_fraction",
            "model_fingerprint", "numerical_fingerprint", "implementation_fingerprint", "figure",
        ),
    )
    _write_tsv(
        output / "channel_summary.tsv",
        channel_rows,
        (
            "run", "label", "group", "channel", "final_signal_rms", "signal_resolved",
            "final_q_over_pi", "final_peak_share", "final_peak_band_share",
            "final_second_peak_ratio", "final_spectral_entropy", "final_phase_slip_candidates",
            "final_candidate_residual_distance_rungs",
            "candidate_history_coverage", "strongest_candidate_drift_rungs",
        ),
    )
    _write_tsv(
        output / "history_metrics.tsv",
        history_rows,
        (
            "run", "label", "group", "channel", "iteration", "q_over_pi", "peak_share",
            "peak_band_share", "second_peak_ratio", "spectral_entropy", "signal_rms", "signal_resolved",
        ),
    )
    _write_tsv(
        output / "residual_history.tsv",
        residual_rows,
        (
            "run", "label", "iteration", "stored_period_one_relative_residual",
            "diagnostic_residual_kind", "diagnostic_relative_residual", "residual_peak_rung",
            "residual_ipr", "alpha_residual_fraction", "beta_residual_fraction", "mu_residual_fraction",
        ),
    )
    _write_tsv(
        output / "phase_slip_candidates.tsv",
        candidate_rows,
        (
            "run", "label", "group", "channel", "iteration", "rung", "amplitude_ratio",
            "phase_jump_over_pi", "score",
        ),
    )
    write_report(output / "report.md", states_and_analyses, source_rows, args.bulk_fraction)
    print("output_directory={}".format(output))
    print("states={}".format(len(states_and_analyses)))
    print("figures={}".format(len(states_and_analyses)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
