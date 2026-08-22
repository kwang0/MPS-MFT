#!/usr/bin/env python3
"""Audit and summarize every run selected by plot_order_fourier_max_grid().

This script mirrors the Julia routine's default selection rules for three calls:

    plot_order_fourier_max_grid()
    plot_order_fourier_max_grid(transverse_geometry=:cubic_unfrustrated)
    plot_order_fourier_max_grid(transverse_geometry=:square)

It reconstructs Julia/HDF5 array order, reproduces the plotted Fourier maxima,
and adds geometry-neutral observables from the final correlation matrices.
Only NumPy and h5py are required.
"""

from __future__ import annotations

import argparse
import cmath
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Any, Iterable

import h5py
import numpy as np


RUN_RE = re.compile(
    r"_L_([0-9]+)_U_([-+0-9.eE]+)_V_([-+0-9.eE]+)_t0_([-+0-9.eE]+)"
    r"_t_p_([-+0-9.eE]+).*_density_([-+0-9.eE]+)_"
)
CHI_RE = re.compile(r"_chi_([0-9]+)_")
GEOMETRY_RE = re.compile(r"_geometry_([A-Za-z0-9_-]+)_chi_")
T0_GRID = (0.8, 1.0, 1.2, 1.4, 1.6)
GEOMETRIES = ("cubic_frustrated", "cubic_unfrustrated", "square")
ORDER_CHANNELS = ("cdw", "sdw", "swave", "extended_swave", "dwave")


@dataclass(frozen=True)
class RunParameters:
    path: Path
    L: int
    U: float
    V: float
    t0: float
    tp: float
    density: float
    chi: int | None
    filename_geometry: str | None


def load_pair_binding_reference(path: Path) -> dict[tuple[int, float, float, float, float], dict[str, Any]]:
    """Load the best available isolated-ladder pair-binding value for each parameter set.

    E_p_values.csv stores the hole-pairing convention, in which a negative value
    denotes binding.  The MF code uses the positive magnitude in its denominator.
    If more than one bond dimension exists, retain the largest-chi entry.
    """
    if not path.is_file():
        return {}
    output: dict[tuple[int, float, float, float, float], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["L"]),
                float(row["U"]),
                float(row["V"]),
                float(row["t0"]),
                float(row["density"]),
            )
            item = {
                "signed": float(row["E_p"]),
                "abs": abs(float(row["E_p"])),
                "chi": int(row["chi"]),
                "rel_diff": float(row["rel_diff"]),
            }
            if key not in output or item["chi"] > output[key]["chi"]:
                output[key] = item
    return output


def pair_binding_for_run(
    reference: dict[tuple[int, float, float, float, float], dict[str, Any]],
    params: RunParameters,
) -> dict[str, Any] | None:
    key = (params.L, params.U, params.V, params.t0, params.density)
    return reference.get(key)


def parse_run(path: Path) -> RunParameters | None:
    match = RUN_RE.search(path.name)
    if match is None:
        return None
    geometry_match = GEOMETRY_RE.search(path.name)
    chi_match = CHI_RE.search(path.name)
    return RunParameters(
        path=path,
        L=int(match.group(1)),
        U=float(match.group(2)),
        V=float(match.group(3)),
        t0=float(match.group(4)),
        tp=float(match.group(5)),
        density=float(match.group(6)),
        chi=int(chi_match.group(1)) if chi_match else None,
        filename_geometry=geometry_match.group(1) if geometry_match else None,
    )


def decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def result_geometry(path: Path, handle: h5py.File | None = None) -> str | None:
    match = GEOMETRY_RE.search(path.name)
    if match is not None:
        return match.group(1)
    if handle is not None:
        return decode_string(handle["transverse_geometry"][()]) if "transverse_geometry" in handle else None
    with h5py.File(path, "r") as h5:
        return decode_string(h5["transverse_geometry"][()]) if "transverse_geometry" in h5 else None


def is_grid_t0(value: float, atol: float = 1e-8) -> bool:
    return any(abs(value - target) <= atol for target in T0_GRID)


def select_runs(data_dir: Path) -> list[tuple[str, RunParameters]]:
    """Mirror the three Julia calls' suffix, geometry, and t0 filters."""
    selected: list[tuple[str, RunParameters]] = []
    for selection, suffix, requested_geometry in (
        ("no_explicit_geometry", "_nodamping.h5", None),
        ("cubic_unfrustrated", "_gpu.h5", "cubic_unfrustrated"),
        ("square", "_gpu.h5", "square"),
    ):
        for path in sorted(data_dir.iterdir()):
            if not path.is_file() or not path.name.endswith(suffix):
                continue
            params = parse_run(path)
            if params is None or not is_grid_t0(params.t0):
                continue
            if requested_geometry is not None and result_geometry(path) != requested_geometry:
                continue
            selected.append((selection, params))
    return selected


def julia_array(dataset: h5py.Dataset) -> np.ndarray:
    """Undo the dimension reversal used by HDF5.jl."""
    raw = np.asarray(dataset[()])
    if raw.ndim <= 1:
        return raw
    return np.transpose(raw, axes=tuple(range(raw.ndim - 1, -1, -1)))


def julia_history_slice(dataset: h5py.Dataset, index: int) -> np.ndarray:
    """Read one Julia history slice; HDF5's first axis is Julia's last."""
    raw = np.asarray(dataset[index])
    if raw.ndim <= 1:
        return raw
    return np.transpose(raw, axes=tuple(range(raw.ndim - 1, -1, -1)))


def scalar(handle: h5py.File, key: str, default: Any = None) -> Any:
    if key not in handle:
        return default
    value = handle[key][()]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def finite_or_none(value: Any) -> float | int | bool | str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(right.ravel())), 1e-300)
    return float(np.linalg.norm((left - right).ravel()) / denom)


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def shifted_indices(n: int) -> np.ndarray:
    half = n // 2
    if n % 2 == 0:
        return np.concatenate((np.arange(half, n), np.arange(0, half)))
    return np.concatenate((np.arange(half + 1, n), np.arange(0, half + 1)))


def shifted_integer_axis(n: int) -> np.ndarray:
    indices = shifted_indices(n)
    half = n / 2
    return np.asarray([idx if idx < half else idx - n for idx in indices], dtype=int)


def peak_summary(values: np.ndarray, *, ladder: bool) -> dict[str, Any]:
    flat_index = int(np.argmax(values.ravel(order="F")))
    row, col = np.unravel_index(flat_index, values.shape, order="F")
    period = values.shape[1]
    multiples = shifted_integer_axis(period)
    power = np.square(np.abs(values))
    power_sum = float(np.sum(power))
    return {
        "value": float(values[row, col]),
        "kx_multiple": int(multiples[col]),
        "kx_period": int(period),
        "kx_over_pi": float(2 * multiples[col] / period),
        "abs_kx_over_pi": float(abs(2 * multiples[col] / period)),
        "ky_multiple": int(row) if ladder else None,
        "ky_period": 2 if ladder else None,
        "ky_over_pi": float(row) if ladder else None,
        "peak_power_fraction": float(power[row, col] / power_sum) if power_sum > 0 else 0.0,
    }


def ladder_fourier(field: np.ndarray, *, subtract_average: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(field, dtype=float)
    if subtract_average:
        values = values - np.mean(values)
    n = values.shape[0]
    even = np.zeros(n, dtype=complex)
    odd = np.zeros(n, dtype=complex)
    # Match Julia's explicit summation order so conjugate-degenerate peak signs
    # follow the plotting routine's floating-point tie break.
    for k in range(n):
        even_sum = 0.0j
        odd_sum = 0.0j
        for x in range(n):
            phase = cmath.exp(-2j * math.pi * k * x / n)
            even_sum += (values[x, 0] + values[x, 1]) * phase
            odd_sum += (values[x, 0] - values[x, 1]) * phase
        even[k] = even_sum / (2 * n)
        odd[k] = odd_sum / (2 * n)
    spectrum = np.abs(np.vstack((even, odd))[:, shifted_indices(n)])
    return spectrum, peak_summary(spectrum, ladder=True)


def chain_fourier(field: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(field, dtype=float)
    n = len(values)
    raw = np.zeros(n, dtype=complex)
    for k in range(n):
        total = 0.0j
        for x in range(n):
            total += values[x] * cmath.exp(-2j * math.pi * k * x / n)
        raw[k] = total / n
    spectrum = np.abs(raw)[shifted_indices(n)].reshape(1, n)
    return spectrum, peak_summary(spectrum, ladder=False)


def order_fields_from_mf(alpha: np.ndarray, beta: np.ndarray, trim: int = 5) -> dict[str, np.ndarray]:
    alpha = alpha[trim:-trim, trim:-trim, :, :]
    beta = beta[:, trim:-trim, trim:-trim, :, :]
    L = alpha.shape[0]
    cdw = np.asarray(
        [[beta[1, i, i, leg, leg] + beta[0, i, i, leg, leg] for leg in range(2)] for i in range(L)]
    )
    sdw = np.asarray(
        [[beta[1, i, i, leg, leg] - beta[0, i, i, leg, leg] for leg in range(2)] for i in range(L)]
    )
    swave = np.asarray([[alpha[i, i, leg, leg] for leg in range(2)] for i in range(L)])
    leg_pair = np.asarray(
        [0.5 * (alpha[i, i + 1, 0, 0] + alpha[i, i + 1, 1, 1]) for i in range(L - 1)]
    )
    rung_pair = np.asarray([alpha[i, i, 0, 1] for i in range(L - 1)])
    return {
        "cdw": cdw,
        "sdw": sdw,
        "swave": swave,
        "extended_swave": leg_pair + rung_pair,
        "dwave": leg_pair - rung_pair,
    }


def order_fields_from_correlations(
    cpair: np.ndarray, c_dn: np.ndarray, c_up: np.ndarray, trim: int = 5
) -> dict[str, np.ndarray]:
    L_full = cpair.shape[0] // 2
    rungs = range(trim, L_full - trim)
    sites = [2 * rung + leg for rung in rungs for leg in range(2)]
    cpair = cpair[np.ix_(sites, sites)]
    c_dn = c_dn[np.ix_(sites, sites)]
    c_up = c_up[np.ix_(sites, sites)]
    L = len(rungs)
    cdw = np.asarray(
        [[c_up[2 * i + leg, 2 * i + leg] + c_dn[2 * i + leg, 2 * i + leg] for leg in range(2)] for i in range(L)]
    )
    sdw = np.asarray(
        [[c_up[2 * i + leg, 2 * i + leg] - c_dn[2 * i + leg, 2 * i + leg] for leg in range(2)] for i in range(L)]
    )
    swave = np.asarray([[cpair[2 * i + leg, 2 * i + leg] for leg in range(2)] for i in range(L)])
    extended: list[float] = []
    dwave: list[float] = []
    for i in range(L - 1):
        i0, i1, p0, p1 = 2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1
        leg0 = 0.5 * (cpair[i0, p0] + cpair[p0, i0])
        leg1 = 0.5 * (cpair[i1, p1] + cpair[p1, i1])
        rung = 0.5 * (cpair[i0, i1] + cpair[i1, i0])
        leg = 0.5 * (leg0 + leg1)
        extended.append(float(leg + rung))
        dwave.append(float(leg - rung))
    return {
        "cdw": cdw,
        "sdw": sdw,
        "swave": swave,
        "extended_swave": np.asarray(extended),
        "dwave": np.asarray(dwave),
    }


def fourier_maxima(fields: dict[str, np.ndarray]) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    spectra: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for channel in ORDER_CHANNELS:
        if channel in ("cdw", "sdw", "swave"):
            spectrum, summary = ladder_fourier(fields[channel], subtract_average=(channel == "cdw"))
        else:
            spectrum, summary = chain_fourier(fields[channel])
        spectra[channel] = spectrum
        summaries[channel] = summary
    return summaries, spectra


def unit_alpha_prediction(cpair: np.ndarray, geometry: str, r_range: int = 4) -> np.ndarray:
    L = cpair.shape[0] // 2
    prediction = np.zeros((L, L, 2, 2), dtype=float)
    for i in range(L):
        for ip in range(max(0, i - r_range), min(L, i + r_range + 1)):
            i0, i1, p0, p1 = 2 * i, 2 * i + 1, 2 * ip, 2 * ip + 1
            if geometry == "cubic_frustrated":
                values = (
                    cpair[p1, i1] + 2 * cpair[p0, i0],
                    cpair[p0, i0] + 2 * cpair[p1, i1],
                    2 * cpair[p0, i1],
                    2 * cpair[p1, i0],
                )
            elif geometry == "cubic_unfrustrated":
                values = (
                    3 * cpair[p1, i1],
                    3 * cpair[p0, i0],
                    2 * cpair[p1, i0],
                    2 * cpair[p0, i1],
                )
            elif geometry == "square":
                values = (cpair[p1, i1], cpair[p0, i0], 0.0, 0.0)
            else:
                raise ValueError(geometry)
            prediction[i, ip, 0, 0] = values[0]
            prediction[i, ip, 1, 1] = values[1]
            prediction[i, ip, 1, 0] = values[2]
            prediction[i, ip, 0, 1] = values[3]
    return prediction


def infer_geometry_and_prefactor(
    measured_alpha: np.ndarray, cpair: np.ndarray, tp: float, threshold: float = 1e-6
) -> tuple[str, float, float, dict[str, float]]:
    residuals: dict[str, float] = {}
    prefactors: dict[str, float] = {}
    for geometry in GEOMETRIES:
        unit = unit_alpha_prediction(cpair, geometry)
        nonzero = measured_alpha != 0
        denom = float(np.dot(unit[nonzero], unit[nonzero])) if np.any(nonzero) else 0.0
        pref = float(np.dot(unit[nonzero], measured_alpha[nonzero]) / denom) if denom > 0 else 0.0
        predicted = pref * unit
        predicted[np.abs(predicted) <= threshold] = 0.0
        residuals[geometry] = relative_l2(predicted, measured_alpha)
        prefactors[geometry] = pref
    best = min(GEOMETRIES, key=lambda geometry: residuals[geometry])
    pref = prefactors[best]
    inferred_ep = 2 * tp * tp / pref if pref != 0 else math.nan
    return best, pref, inferred_ep, residuals


def full_density_metrics(c_dn: np.ndarray, c_up: np.ndarray, L: int) -> dict[str, float]:
    n_dn = np.diag(c_dn)
    n_up = np.diag(c_up)
    density = n_dn + n_up
    spin = n_up - n_dn
    phases = np.repeat(np.exp(1j * math.pi * np.arange(L)), 2)
    return {
        "density_measured": float(np.mean(density)),
        "density_leg_imbalance_mean": float(np.mean(density[0::2] - density[1::2])),
        "density_leg_imbalance_rms": float(np.sqrt(np.mean(np.square(density[0::2] - density[1::2])))),
        "spin_rms": float(np.sqrt(np.mean(np.square(spin)))),
        "physical_cdw_qpi_full": float(abs(np.sum(density * phases) / (2 * L))),
        "physical_sdw_qpi_full": float(abs(np.sum(spin * phases) / (2 * L))),
    }


def fit_log_model(x: np.ndarray, y: np.ndarray, *, log_x: bool) -> tuple[float | None, float | None, int]:
    mask = np.isfinite(y) & (y > 1e-10) & np.isfinite(x) & (x > 0)
    x_fit = np.log(x[mask]) if log_x else x[mask]
    y_fit = np.log(y[mask])
    if len(x_fit) < 4:
        return None, None, int(len(x_fit))
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    predicted = slope * x_fit + intercept
    total = float(np.sum(np.square(y_fit - np.mean(y_fit))))
    residual = float(np.sum(np.square(y_fit - predicted)))
    r2 = 1 - residual / total if total > 0 else 1.0
    return float(slope), float(r2), int(len(x_fit))


def pair_correlation_rows(matrix: np.ndarray, run_id: str, selection: str, geometry: str, V: float, t0: float) -> list[dict[str, Any]]:
    L = matrix.shape[0]
    edge = 8
    rows: list[dict[str, Any]] = []
    for distance in range(1, L - 2 * edge):
        values = np.asarray([matrix[i, i + distance] for i in range(edge, L - edge - distance)])
        rows.append(
            {
                "run_id": run_id,
                "selection": selection,
                "geometry": geometry,
                "V": V,
                "t0": t0,
                "distance": distance,
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "mean_abs": float(np.mean(np.abs(values))),
                "std": float(np.std(values)),
            }
        )
    return rows


def pair_correlation_summary(matrix: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "pair_matrix_symmetry_max_abs": float(np.max(np.abs(matrix - matrix.T))),
        "pair_matrix_diagonal_mean": float(np.mean(np.diag(matrix))),
        "pair_structure_offdiag_per_rung": float((np.sum(matrix) - np.trace(matrix)) / matrix.shape[0]),
    }
    row_by_distance = {int(row["distance"]): row for row in rows}
    for distance in (1, 2, 4, 8, 12, 16, 20, 24):
        output[f"pair_corr_abs_r{distance}"] = (
            float(row_by_distance[distance]["mean_abs"]) if distance in row_by_distance else None
        )
    fit_rows = [row for row in rows if 4 <= int(row["distance"]) <= 20]
    x = np.asarray([row["distance"] for row in fit_rows], dtype=float)
    y = np.asarray([row["mean_abs"] for row in fit_rows], dtype=float)
    exp_slope, exp_r2, exp_n = fit_log_model(x, y, log_x=False)
    power_slope, power_r2, power_n = fit_log_model(x, y, log_x=True)
    output.update(
        {
            "pair_exp_xi_4_20": (-1 / exp_slope if exp_slope is not None and exp_slope < 0 else None),
            "pair_exp_r2_4_20": exp_r2,
            "pair_exp_n_4_20": exp_n,
            "pair_power_eta_4_20": (-power_slope if power_slope is not None else None),
            "pair_power_r2_4_20": power_r2,
            "pair_power_n_4_20": power_n,
        }
    )
    return output


def best_history_lag(dataset: h5py.Dataset, max_lag: int = 8) -> tuple[int | None, float | None]:
    n = dataset.shape[0]
    if n < 2:
        return None, None
    latest = julia_history_slice(dataset, -1)
    candidates: list[tuple[int, float]] = []
    for lag in range(1, min(max_lag, n - 1) + 1):
        previous = julia_history_slice(dataset, -1 - lag)
        candidates.append((lag, relative_l2(latest, previous)))
    return min(candidates, key=lambda item: item[1])


def closest_history_slice(dataset: h5py.Dataset, current: np.ndarray) -> tuple[int, int, float]:
    """Return the 1-based closest iteration, lag from the end, and relative L2 distance."""
    candidates: list[tuple[int, int, float]] = []
    n = dataset.shape[0]
    for index in range(n):
        distance = relative_l2(current, julia_history_slice(dataset, index))
        candidates.append((index + 1, n - (index + 1), distance))
    return min(candidates, key=lambda item: item[2])


def field_profile_metrics(fields: dict[str, np.ndarray], prefix: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for channel, values in fields.items():
        array = np.asarray(values, dtype=float)
        output[f"{prefix}_{channel}_mean"] = float(np.mean(array))
        output[f"{prefix}_{channel}_mean_abs"] = float(np.mean(np.abs(array)))
        output[f"{prefix}_{channel}_rms"] = float(np.sqrt(np.mean(np.square(array))))
        output[f"{prefix}_{channel}_max_abs"] = float(np.max(np.abs(array)))
    return output


def iteration_fourier_metrics(
    h5: h5py.File,
    *,
    run_id: str,
    selection: str,
    geometry: str,
    V: float,
    t0: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = min(h5["alpha_list"].shape[0], h5["beta_list"].shape[0])
    rows: list[dict[str, Any]] = []
    previous_alpha: np.ndarray | None = None
    previous_beta: np.ndarray | None = None
    for index in range(n):
        alpha = julia_history_slice(h5["alpha_list"], index)
        beta = julia_history_slice(h5["beta_list"], index)
        maxima, _ = fourier_maxima(order_fields_from_mf(alpha, beta))
        item: dict[str, Any] = {
            "run_id": run_id,
            "selection": selection,
            "geometry": geometry,
            "V": V,
            "t0": t0,
            "iteration": index + 1,
            "alpha_step_rel_l2": relative_l2(alpha, previous_alpha) if previous_alpha is not None else None,
            "beta_step_rel_l2": relative_l2(beta, previous_beta) if previous_beta is not None else None,
        }
        for channel in ORDER_CHANNELS:
            item[f"{channel}_value"] = maxima[channel]["value"]
            item[f"{channel}_abs_kx_over_pi"] = maxima[channel]["abs_kx_over_pi"]
            item[f"{channel}_ky_over_pi"] = maxima[channel]["ky_over_pi"]
        rows.append(item)
        previous_alpha = alpha
        previous_beta = beta

    summary: dict[str, Any] = {}
    recent = rows[-min(10, len(rows)) :]
    for channel in ORDER_CHANNELS:
        values = np.asarray([item[f"{channel}_value"] for item in recent], dtype=float)
        mean = float(np.mean(values))
        summary[f"recent10_{channel}_mean"] = mean
        summary[f"recent10_{channel}_std"] = float(np.std(values))
        summary[f"recent10_{channel}_cv"] = float(np.std(values) / mean) if mean > 0 else None
        summary[f"recent10_{channel}_relative_range"] = (
            float((np.max(values) - np.min(values)) / mean) if mean > 0 else None
        )
        momenta = [
            (item[f"{channel}_abs_kx_over_pi"], item[f"{channel}_ky_over_pi"])
            for item in recent
        ]
        summary[f"recent10_{channel}_momentum_mode_fraction"] = (
            Counter(momenta).most_common(1)[0][1] / len(momenta) if momenta else None
        )
    return rows, summary


def flatten_fourier(prefix: str, summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for channel, summary in summaries.items():
        for key, value in summary.items():
            output[f"{prefix}_{channel}_{key}"] = value
    dominant = max(ORDER_CHANNELS, key=lambda channel: summaries[channel]["value"])
    output[f"{prefix}_dominant_channel"] = dominant
    output[f"{prefix}_dominant_value"] = summaries[dominant]["value"]
    return output


def analyze_run(
    selection: str,
    params: RunParameters,
    pair_binding_reference: dict[tuple[int, float, float, float, float], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    path = params.path
    pair_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5:
        required = ("alpha", "beta", "alpha_list", "beta_list", "C_pair_list", "C_exc_dn_list", "C_exc_up_list")
        missing = [key for key in required if key not in h5]
        if missing:
            raise KeyError(f"{path.name}: missing required datasets {missing}")

        alpha = julia_array(h5["alpha"])
        beta = julia_array(h5["beta"])
        last_alpha = julia_history_slice(h5["alpha_list"], -1)
        last_beta = julia_history_slice(h5["beta_list"], -1)
        cpair = julia_history_slice(h5["C_pair_list"], -1)
        c_dn = julia_history_slice(h5["C_exc_dn_list"], -1)
        c_up = julia_history_slice(h5["C_exc_up_list"], -1)

        mf_fields = order_fields_from_mf(alpha, beta)
        corr_fields = order_fields_from_correlations(cpair, c_dn, c_up)
        mf_maxima, mf_spectra = fourier_maxima(mf_fields)
        corr_maxima, corr_spectra = fourier_maxima(corr_fields)

        inferred_geometry, prefactor, inferred_ep, geometry_residuals = infer_geometry_and_prefactor(
            last_alpha, cpair, params.tp
        )
        stored_geometry = result_geometry(path, h5)
        geometry = stored_geometry or inferred_geometry
        local_pair_binding = pair_binding_for_run(pair_binding_reference, params)
        run_id = f"{selection}__V_{params.V:+.1f}__t0_{params.t0:.1f}"
        iteration_rows, iteration_summary = iteration_fourier_metrics(
            h5,
            run_id=run_id,
            selection=selection,
            geometry=geometry,
            V=params.V,
            t0=params.t0,
        )

        alpha_lag, alpha_lag_rel = best_history_lag(h5["alpha_list"])
        beta_lag, beta_lag_rel = best_history_lag(h5["beta_list"])
        alpha_closest_iteration, alpha_closest_lag, alpha_closest_rel = closest_history_slice(
            h5["alpha_list"], alpha
        )
        beta_closest_iteration, beta_closest_lag, beta_closest_rel = closest_history_slice(
            h5["beta_list"], beta
        )
        alpha_previous = julia_history_slice(h5["alpha_list"], -2) if h5["alpha_list"].shape[0] >= 2 else None
        beta_previous = julia_history_slice(h5["beta_list"], -2) if h5["beta_list"].shape[0] >= 2 else None

        row: dict[str, Any] = {
            "run_id": run_id,
            "selection": selection,
            "filename": path.name,
            "file_size_bytes": path.stat().st_size,
            "file_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "L_rungs": params.L,
            "sites": 2 * params.L,
            "U": params.U,
            "V": params.V,
            "t0": params.t0,
            "tp": params.tp,
            "target_density": params.density,
            "chi": params.chi,
            "filename_geometry": params.filename_geometry,
            "stored_geometry": stored_geometry,
            "inferred_geometry": inferred_geometry,
            "analysis_geometry": geometry,
            "geometry_inference_residual": geometry_residuals[inferred_geometry],
            "geometry_residual_cubic_frustrated": geometry_residuals["cubic_frustrated"],
            "geometry_residual_cubic_unfrustrated": geometry_residuals["cubic_unfrustrated"],
            "geometry_residual_square": geometry_residuals["square"],
            "inferred_prefactor_2tp2_over_Ep": prefactor,
            "inferred_Ep": inferred_ep,
            "local_pair_binding_energy_signed": (
                local_pair_binding["signed"] if local_pair_binding is not None else None
            ),
            "local_pair_binding_energy_abs": (
                local_pair_binding["abs"] if local_pair_binding is not None else None
            ),
            "local_pair_binding_chi": (
                local_pair_binding["chi"] if local_pair_binding is not None else None
            ),
            "local_pair_binding_rel_diff": (
                local_pair_binding["rel_diff"] if local_pair_binding is not None else None
            ),
            "used_Ep_over_local_pair_binding_abs": (
                inferred_ep / local_pair_binding["abs"]
                if local_pair_binding is not None and local_pair_binding["abs"] > 0
                else None
            ),
            "used_mf_coupling_over_local_coupling": (
                local_pair_binding["abs"] / inferred_ep
                if local_pair_binding is not None and inferred_ep > 0
                else None
            ),
            "tp_over_local_pair_binding_abs": (
                params.tp / local_pair_binding["abs"]
                if local_pair_binding is not None and local_pair_binding["abs"] > 0
                else None
            ),
            "tp_below_local_pair_binding_abs": (
                params.tp < local_pair_binding["abs"] if local_pair_binding is not None else None
            ),
            "completed": bool(scalar(h5, "completed", False)),
            "period2_cycle_detected": bool(scalar(h5, "period2_cycle_detected", False)),
            "alpha_iterations": int(h5["alpha_list"].shape[0]),
            "beta_iterations": int(h5["beta_list"].shape[0]),
            "pair_iterations": int(h5["C_pair_list"].shape[0]),
            "mu_cdw_iterations": int(h5["mu_cdw_list"].shape[0]) if "mu_cdw_list" in h5 else 0,
            "alpha_shape_valid": tuple(alpha.shape) == (params.L, params.L, 2, 2),
            "beta_shape_valid": tuple(beta.shape) == (2, params.L, params.L, 2, 2),
            "arrays_finite": bool(
                np.all(np.isfinite(alpha))
                and np.all(np.isfinite(beta))
                and np.all(np.isfinite(cpair))
                and np.all(np.isfinite(c_dn))
                and np.all(np.isfinite(c_up))
            ),
            "stored_U_matches_filename": abs(float(scalar(h5, "U", math.nan)) - params.U) <= 1e-10,
            "stored_V_matches_filename": abs(float(scalar(h5, "V", math.nan)) - params.V) <= 1e-10,
            "stored_t0_matches_filename": abs(float(scalar(h5, "t0", math.nan)) - params.t0) <= 1e-10,
            "stored_tp_matches_filename": abs(float(scalar(h5, "t_p", math.nan)) - params.tp) <= 1e-10,
            "energy": finite_or_none(scalar(h5, "E")),
            "energy_per_site": finite_or_none(float(scalar(h5, "E")) / (2 * params.L)) if scalar(h5, "E") is not None else None,
            "gap": finite_or_none(scalar(h5, "gap")),
            "mu": finite_or_none(scalar(h5, "mu")),
            "onsite_pair_order_param": finite_or_none(scalar(h5, "order_param")),
            "dwave_order_param": finite_or_none(scalar(h5, "dwave_order_param")),
            "cdw_order_param": finite_or_none(scalar(h5, "cdw_order_param")),
            "plot_alpha_vs_last_measured_rel_l2": relative_l2(alpha, last_alpha),
            "plot_alpha_vs_last_measured_max_abs": max_abs_difference(alpha, last_alpha),
            "plot_beta_vs_last_measured_rel_l2": relative_l2(beta, last_beta),
            "plot_beta_vs_last_measured_max_abs": max_abs_difference(beta, last_beta),
            "alpha_current_max_abs": float(np.max(np.abs(alpha))),
            "alpha_last_measured_max_abs": float(np.max(np.abs(last_alpha))),
            "beta_current_max_abs": float(np.max(np.abs(beta))),
            "beta_last_measured_max_abs": float(np.max(np.abs(last_beta))),
            "alpha_current_last_jointly_below_5e3_fraction": float(
                np.mean((np.abs(alpha) < 5e-3) & (np.abs(last_alpha) < 5e-3))
            ),
            "alpha_current_closest_history_iteration": alpha_closest_iteration,
            "alpha_current_closest_history_lag": alpha_closest_lag,
            "alpha_current_closest_history_rel_l2": alpha_closest_rel,
            "beta_current_closest_history_iteration": beta_closest_iteration,
            "beta_current_closest_history_lag": beta_closest_lag,
            "beta_current_closest_history_rel_l2": beta_closest_rel,
            "last_alpha_step_rel_l2": relative_l2(last_alpha, alpha_previous) if alpha_previous is not None else None,
            "last_beta_step_rel_l2": relative_l2(last_beta, beta_previous) if beta_previous is not None else None,
            "alpha_best_lag_1_to_8": alpha_lag,
            "alpha_best_lag_rel_l2": alpha_lag_rel,
            "beta_best_lag_1_to_8": beta_lag,
            "beta_best_lag_rel_l2": beta_lag_rel,
        }
        row.update(full_density_metrics(c_dn, c_up, params.L))
        row["density_absolute_error"] = abs(row["density_measured"] - params.density)
        row["physical_cdw_vs_stored_abs_diff"] = (
            abs(row["physical_cdw_qpi_full"] - float(row["cdw_order_param"]))
            if row["cdw_order_param"] is not None
            else None
        )
        row.update(field_profile_metrics(mf_fields, "plot_field"))
        row.update(field_profile_metrics(corr_fields, "corr_field"))
        row.update(flatten_fourier("plot", mf_maxima))
        row.update(flatten_fourier("corr", corr_maxima))
        row.update(iteration_summary)

        for source, spectra in (("plot_mf", mf_spectra), ("physical_correlations", corr_spectra)):
            for channel, spectrum in spectra.items():
                multiples = shifted_integer_axis(spectrum.shape[1])
                for ky_index in range(spectrum.shape[0]):
                    for column, multiple in enumerate(multiples):
                        spectrum_rows.append(
                            {
                                "run_id": run_id,
                                "selection": selection,
                                "geometry": geometry,
                                "V": params.V,
                                "t0": params.t0,
                                "source": source,
                                "channel": channel,
                                "kx_multiple": int(multiple),
                                "kx_period": int(spectrum.shape[1]),
                                "kx_over_pi": float(2 * multiple / spectrum.shape[1]),
                                "ky_over_pi": float(ky_index) if spectrum.shape[0] == 2 else None,
                                "amplitude": float(spectrum[ky_index, column]),
                            }
                        )

        if "mu_cdw" in h5 and "mu_cdw_list" in h5 and h5["mu_cdw_list"].shape[0] > 0:
            mu_cdw = julia_array(h5["mu_cdw"])
            last_mu_cdw = julia_history_slice(h5["mu_cdw_list"], -1)
            row["plot_mu_cdw_vs_last_measured_rel_l2"] = relative_l2(mu_cdw, last_mu_cdw)
            row["plot_mu_cdw_vs_last_measured_max_abs"] = max_abs_difference(mu_cdw, last_mu_cdw)
        else:
            row["plot_mu_cdw_vs_last_measured_rel_l2"] = None
            row["plot_mu_cdw_vs_last_measured_max_abs"] = None

        if "D_rung_matrix" in h5:
            pair_matrix = julia_array(h5["D_rung_matrix"])
            pair_rows = pair_correlation_rows(
                pair_matrix, run_id, selection, geometry, params.V, params.t0
            )
            row.update(pair_correlation_summary(pair_matrix, pair_rows))
            row["has_D_rung_matrix"] = True
        else:
            row["has_D_rung_matrix"] = False

        return row, pair_rows, spectrum_rows, iteration_rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for selection in ("no_explicit_geometry", "cubic_unfrustrated", "square"):
        subset = [row for row in rows if row["selection"] == selection]
        groups[selection] = {
            "run_count": len(subset),
            "completed_count": sum(bool(row["completed"]) for row in subset),
            "incomplete_count": sum(not bool(row["completed"]) for row in subset),
            "V_values": sorted({row["V"] for row in subset}),
            "t0_values": sorted({row["t0"] for row in subset}),
            "max_density_absolute_error": max((row["density_absolute_error"] for row in subset), default=None),
            "inferred_geometries": sorted({row["inferred_geometry"] for row in subset}),
            "dominant_plot_channels": sorted({row["plot_dominant_channel"] for row in subset}),
            "dominant_correlation_channels": sorted({row["corr_dominant_channel"] for row in subset}),
            "local_pair_binding_available_count": sum(
                row["local_pair_binding_energy_abs"] is not None for row in subset
            ),
            "tp_not_below_local_pair_binding_count": sum(
                row["tp_below_local_pair_binding_abs"] is False for row in subset
            ),
            "used_mf_coupling_over_local_coupling_range": [
                min(
                    row["used_mf_coupling_over_local_coupling"]
                    for row in subset
                    if row["used_mf_coupling_over_local_coupling"] is not None
                ),
                max(
                    row["used_mf_coupling_over_local_coupling"]
                    for row in subset
                    if row["used_mf_coupling_over_local_coupling"] is not None
                ),
            ],
        }
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_contract": {
            "no_explicit_geometry": {"suffix": "_nodamping.h5", "geometry_filter": None},
            "cubic_unfrustrated": {"suffix": "_gpu.h5", "geometry_filter": "cubic_unfrustrated"},
            "square": {"suffix": "_gpu.h5", "geometry_filter": "square"},
            "t0_grid": list(T0_GRID),
            "trim_boundary_rungs": 5,
            "source": "mf",
            "fourier_value": "abs",
            "fourier_normalized": True,
        },
        "total_runs": len(rows),
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("stateless_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/fourier_max_grid_2026-08-14"))
    parser.add_argument("--pair-binding-csv", type=Path, default=Path("E_p_values.csv"))
    args = parser.parse_args()

    selected = select_runs(args.data_dir)
    pair_binding_reference = load_pair_binding_reference(args.pair_binding_csv)
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    for selection, params in selected:
        row, run_pair_rows, run_spectrum_rows, run_iteration_rows = analyze_run(
            selection, params, pair_binding_reference
        )
        rows.append(row)
        pair_rows.extend(run_pair_rows)
        spectrum_rows.extend(run_spectrum_rows)
        iteration_rows.extend(run_iteration_rows)
        print(
            f"{selection:22s} V={params.V:+.1f} t0={params.t0:.1f} "
            f"completed={row['completed']} inferred_geometry={row['inferred_geometry']}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_metrics.csv", rows)
    write_csv(args.output_dir / "pair_correlations.csv", pair_rows)
    write_csv(args.output_dir / "fourier_spectra.csv", spectrum_rows)
    write_csv(args.output_dir / "iteration_metrics.csv", iteration_rows)
    (args.output_dir / "run_metrics.json").write_text(
        json.dumps(json_safe(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(aggregate_summary(rows)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(rows)} run rows, {len(pair_rows)} pair-correlation rows, "
        f"{len(spectrum_rows)} spectrum rows, and {len(iteration_rows)} iteration rows"
    )


if __name__ == "__main__":
    main()
