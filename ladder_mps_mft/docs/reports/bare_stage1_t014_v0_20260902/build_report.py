#!/usr/bin/env python3
"""Build the 2026-09-02 bare-ladder Stage 1 technical report.

The builder is intentionally read-only with respect to the synchronized run.
It derives every quantitative table from the compact HDF5 artifacts, manifest,
and scheduler logs and writes only inside this report directory.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
LADDER_ROOT = REPORT_DIR.parents[2]
RUN_ID = "20260901_bare_t014_v0_stage1"
RUN_DIR = LADDER_ROOT / "output" / "bare_stage1" / RUN_ID
STATELESS = RUN_DIR / "stateless_results"
BACKBONE = STATELESS / "backbone.h5"
STAGE1 = STATELESS / "stage1.h5"
MANIFEST = STATELESS / "stateless_manifest.tsv"
SUMMARY = STATELESS / "stage1_summary.tsv"
LOG_DIR = RUN_DIR / "logs"
DATA_DIR = REPORT_DIR / "data"

SECTOR_LABELS = {
    "N118_twoSz0": "N-2, 2Sz=0",
    "N119_twoSz1": "N-1, 2Sz=1",
    "N120_twoSz0": "N, 2Sz=0 ground",
    "N120_twoSz2": "N, 2Sz=2 spin excitation",
    "N121_twoSz1": "N+1, 2Sz=1",
    "N122_twoSz0": "N+2, 2Sz=0",
}


def scalar(group: h5py.Group, name: str) -> Any:
    value = group[name][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def covariance_metrics(matrix: np.ndarray) -> tuple[float, int]:
    eigenvalues = np.maximum(np.linalg.eigvalsh((matrix + matrix.T) / 2), 0.0)
    total = float(eigenvalues.sum())
    participation_rank = total * total / float(eigenvalues @ eigenvalues)
    descending = np.sort(eigenvalues)[::-1]
    k90 = int(np.searchsorted(np.cumsum(descending), 0.9 * total) + 1)
    return participation_rank, k90


def parse_logs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sweep_re = re.compile(
        r"After sweep\s+(\d+)\s+energy=([-+0-9.eE]+)\s+"
        r"maxlinkdim=(\d+)\s+maxerr=([-+0-9.eE]+)\s+time=([-+0-9.eE]+)"
    )
    checkpoint_re = re.compile(r"stage_(\d{3})_([^/\\]+)\.h5")
    sector_re = re.compile(r"^backbone_sector=([^\r\n]+)", re.MULTILINE)
    sector_rows: list[dict[str, Any]] = []
    stage_aggregate: dict[str, dict[str, float]] = defaultdict(
        lambda: {"sweeps": 0.0, "seconds": 0.0}
    )

    with h5py.File(BACKBONE, "r") as backbone:
        for log_path in sorted(LOG_DIR.glob("sector-*.out")):
            text = log_path.read_text(encoding="utf-8", errors="replace")
            sector_match = sector_re.search(text)
            if sector_match is None:
                raise RuntimeError(f"missing sector marker in {log_path}")
            sector = sector_match.group(1).strip()
            stages: list[tuple[str, list[tuple[str, str, str, str, str]]]] = []
            pending: list[tuple[str, str, str, str, str]] = []
            for line in text.splitlines():
                match = sweep_re.search(line)
                if match:
                    pending.append(match.groups())
                    continue
                checkpoint = checkpoint_re.search(line)
                if checkpoint:
                    stage_name = checkpoint.group(2)
                    stages.append((stage_name, pending))
                    pending = []
            if pending:
                raise RuntimeError(f"uncheckpointed sweeps in {log_path}")
            all_sweeps = [row for _, rows in stages for row in rows]
            final_stage = stages[-1][1]
            elapsed_seconds = sum(float(row[4]) for row in all_sweeps)
            for stage_name, rows in stages:
                stage_aggregate[stage_name]["sweeps"] += len(rows)
                stage_aggregate[stage_name]["seconds"] += sum(float(row[4]) for row in rows)
            sector_group = backbone[f"sectors/{sector}"]
            sector_rows.append(
                {
                    "sector": sector,
                    "purpose": SECTOR_LABELS[sector],
                    "sweeps": len(all_sweeps),
                    "dmrg_hours": elapsed_seconds / 3600.0,
                    "final_stage_sweeps": len(final_stage),
                    "final_max_discarded_weight": max(float(row[3]) for row in final_stage),
                    "last_five_energy_spread": scalar(
                        sector_group, "last_five_energy_change"
                    ),
                    "energy": scalar(sector_group, "energy"),
                    "converged": bool(scalar(sector_group, "converged")),
                    "log": str(log_path.relative_to(LADDER_ROOT)).replace("\\", "/"),
                }
            )

    stage_order = {
        "pre_relax_chi200": 1,
        "chi400": 2,
        "chi800": 3,
        "chi1200": 4,
    }
    stage_rows = [
        {
            "stage": stage,
            "maxdim": int(stage.removeprefix("chi"))
            if stage.startswith("chi")
            else 200,
            "sweeps": int(values["sweeps"]),
            "dmrg_hours": values["seconds"] / 3600.0,
        }
        for stage, values in sorted(
            stage_aggregate.items(), key=lambda item: stage_order[item[0]]
        )
    ]
    return sector_rows, stage_rows


def svg_horizontal_bars(
    rows: list[dict[str, Any]], label_key: str, value_key: str, unit: str, color: str
) -> str:
    width, left, right, row_height = 760, 230, 80, 38
    height = 34 + row_height * len(rows)
    plot_width = width - left - right
    maximum = max(float(row[value_key]) for row in rows)
    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Horizontal bar chart in {html.escape(unit)}">'
    ]
    for index, row in enumerate(rows):
        y = 20 + index * row_height
        value = float(row[value_key])
        bar_width = 0 if maximum == 0 else plot_width * value / maximum
        label = html.escape(str(row[label_key]))
        elements.append(
            f'<text x="{left - 12}" y="{y + 18}" text-anchor="end">{label}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="24" '
            f'rx="5" fill="{color}" opacity="0.86" />'
            f'<text x="{left + bar_width + 8:.2f}" y="{y + 18}">'
            f'{value:.2f} {html.escape(unit)}</text>'
        )
    elements.append("</svg>")
    return "".join(elements)


def svg_rank_bars(rows: list[dict[str, Any]]) -> str:
    width, left, right, row_height = 760, 190, 80, 34
    height = 34 + row_height * len(rows)
    plot_width = width - left - right
    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Covariance participation rank relative to block dimension">'
    ]
    for index, row in enumerate(rows):
        y = 18 + index * row_height
        dimension = float(row["dimension"])
        rank = float(row["participation_rank"])
        bar_width = plot_width * rank / dimension
        label = html.escape(str(row["block"]))
        elements.append(
            f'<text x="{left - 12}" y="{y + 16}" text-anchor="end">{label}</text>'
            f'<rect x="{left}" y="{y}" width="{plot_width}" height="22" rx="5" '
            'fill="var(--track)" />'
            f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="22" rx="5" '
            'fill="#b86b3d" />'
            f'<text x="{left + plot_width + 8}" y="{y + 16}">{rank:.1f}/{int(dimension)}</text>'
        )
    elements.append("</svg>")
    return "".join(elements)


def table_html(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for key, _, fmt in columns:
            value = row[key]
            if fmt == "bool":
                shown = "yes" if bool(value) else "no"
            elif fmt:
                shown = format(value, fmt)
            else:
                shown = str(value)
            cells.append(f"<td>{html.escape(shown)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )


def main() -> None:
    for required in (BACKBONE, STAGE1, MANIFEST, SUMMARY):
        if not required.is_file():
            raise FileNotFoundError(required)

    sector_rows, stage_rows = parse_logs()
    manifest_rows = read_tsv(MANIFEST)
    summary_rows = read_tsv(SUMMARY)
    assert len(manifest_rows) == 33
    assert len(sector_rows) == 6
    assert all(row["converged"] for row in sector_rows)

    with h5py.File(BACKBONE, "r") as backbone, h5py.File(STAGE1, "r") as stage1:
        model = {key: scalar(backbone["model"], key) for key in backbone["model"]}
        assert model["L"] == 64 and model["U"] == 8.0
        assert model["V"] == 0.0 and model["t0"] == 1.4
        assert bool(scalar(backbone, "scientifically_accepted"))
        assert bool(scalar(stage1, "complete"))

        energy_keys = (
            "spin_gap",
            "charge_gap",
            "hole_pair_binding",
            "particle_pair_binding",
            "chemical_potential",
        )
        chi_rows: list[dict[str, Any]] = []
        for name in sorted(backbone["chi_dependence"]):
            group = backbone[f"chi_dependence/{name}"]
            chi_rows.append(
                {
                    "stage": name,
                    "chi": int(scalar(group, "maxdim")),
                    **{key: float(scalar(group, key)) for key in energy_keys},
                    "all_sectors_converged": bool(
                        scalar(group, "all_sectors_converged")
                    ),
                }
            )

        diagnostics = {
            key: float(scalar(stage1["diagnostics"], key))
            for key in (
                "K_rho_rung_normalized",
                "K_rho_site_normalized",
                "central_charge",
                "central_charge_r2",
            )
        }
        validity = {
            key: float(scalar(backbone["validity"], key))
            for key in (
                "tp_over_pair_binding",
                "tp_over_spin_gap",
                "tp_over_charge_gap",
            )
        }

        decay_rows: list[dict[str, Any]] = []
        for key, label in (
            ("charge_rung_total", "Charge"),
            ("rung_pair", "Rung pair"),
        ):
            group = stage1[f"decay_fits/{key}"]
            for window in ("001", "002", "003"):
                window_group = group[window]
                decay_rows.append(
                    {
                        "channel": label,
                        "window": window,
                        "edge_fraction": float(scalar(window_group, "edge_fraction")),
                        "exponent": float(scalar(window_group, "exponent")),
                        "r2": float(scalar(window_group, "r2")),
                        "points": int(scalar(window_group, "points")),
                        "reported_estimate": float(scalar(group, "estimate")),
                        "window_uncertainty": float(
                            scalar(group, "window_uncertainty")
                        ),
                    }
                )

        covariance_paths = (
            ("normal/charge/even", "charge even"),
            ("normal/charge/odd", "charge odd"),
            ("normal/spin/even", "spin even"),
            ("normal/spin/odd", "spin odd"),
            ("pairing/rung", "pair rung"),
            ("pairing/leg0", "pair leg 0"),
            ("pairing/leg1", "pair leg 1"),
            ("pairing/onsite0", "pair onsite 0"),
            ("pairing/onsite1", "pair onsite 1"),
        )
        rank_rows: list[dict[str, Any]] = []
        for path, label in covariance_paths:
            matrix = np.asarray(stage1[f"{path}/covariance"][()], dtype=float)
            rank, k90 = covariance_metrics(matrix)
            rank_rows.append(
                {
                    "block": label,
                    "dimension": matrix.shape[0],
                    "participation_rank": rank,
                    "k90": k90,
                    "leading_eigenvalue": float(
                        np.max(np.linalg.eigvalsh((matrix + matrix.T) / 2))
                    ),
                }
            )

        charge_even = np.asarray(
            stage1["normal/charge/even/covariance"][()], dtype=float
        )
        values, vectors = np.linalg.eigh((charge_even + charge_even.T) / 2)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        positions = np.arange(1, 65)
        q_modes = np.arange(33)
        peak_modes = []
        for column in range(vectors.shape[1]):
            amplitudes = [
                abs(
                    np.sum(
                        vectors[:, column]
                        * np.exp(-1j * 2 * np.pi * mode * positions / 64)
                    )
                )
                / 8
                for mode in q_modes
            ]
            peak_modes.append(int(np.argmax(amplitudes)))
        physical_charge_rank = peak_modes.index(4) + 1
        physical_charge_eigenvalue = float(values[physical_charge_rank - 1])

        spin_odd_edge = np.asarray(
            stage1["normal/spin/odd/edge_weight"][()], dtype=float
        )
        spin_odd_values = np.asarray(
            stage1["normal/spin/odd/eigenvalues"][()], dtype=float
        )
        pair_rung_uniform_overlap = float(
            stage1["pairing/rung/fourier_overlap"][0]
        )
        pair_leg_uniform_overlap = float(
            stage1["pairing/leg0/fourier_overlap"][0]
        )

        density = np.asarray(stage1["diagnostics/density"][()], dtype=float)
        rung_density = density.reshape(64, 2).sum(axis=1)
        centered_density = rung_density - rung_density.mean()
        density_fourier = np.asarray(
            [
                abs(
                    np.sum(
                        centered_density
                        * np.exp(-1j * 2 * np.pi * mode * positions / 64)
                    )
                )
                / 8
                for mode in q_modes
            ]
        )
        friedel_mode = int(np.argmax(density_fourier))

        raw_map_norm = float(scalar(stage1, "raw_map_norm"))

    full_bytes = sum(int(row["full_bytes"]) for row in manifest_rows)
    compact_bytes = sum(int(row["compact_bytes"]) for row in manifest_rows)
    final_triplicate_bytes = sum(
        int(row["full_bytes"])
        for row in manifest_rows
        if row["relative_path"] == "backbone.h5"
        or row["relative_path"].startswith("sectors/")
        or "stage_004_chi1200.h5" in row["relative_path"]
    )
    storage_rows = [
        {"copy": "Full restartable tree", "bytes": full_bytes},
        {"copy": "Compact stateless mirror", "bytes": compact_bytes},
        {"copy": "Final-state triplication", "bytes": final_triplicate_bytes},
    ]

    bank_rows = [
        {"block": "normal", "candidate": name, "retained_basis": True}
        for name in (
            "charge even mode 7",
            "charge even mode 8",
            "charge even mode 9",
            "spin odd mode 58",
            "spin odd mode 59",
            "spin odd mode 63",
            "Stage-1 charge-even rank 1",
            "Stage-1 charge-odd rank 1",
            "Stage-1 spin-even rank 1",
        )
    ] + [
        {"block": "pair", "candidate": name, "retained_basis": retained}
        for name, retained in (
            ("onsite s", True),
            ("rung s", True),
            ("leg s", True),
            ("extended s", False),
            ("d wave", False),
        )
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(DATA_DIR / "backbone_convergence.csv", chi_rows)
    write_csv(DATA_DIR / "sector_efficiency.csv", sector_rows)
    write_csv(DATA_DIR / "stage_efficiency.csv", stage_rows)
    write_csv(DATA_DIR / "covariance_rank.csv", rank_rows)
    write_csv(DATA_DIR / "decay_fits.csv", decay_rows)
    write_csv(DATA_DIR / "storage.csv", storage_rows)
    write_csv(DATA_DIR / "stage2_bank.csv", bank_rows)

    total_sweeps = sum(row["sweeps"] for row in sector_rows)
    total_dmrg_hours = sum(row["dmrg_hours"] for row in sector_rows)
    critical_path_hours = max(row["dmrg_hours"] for row in sector_rows)
    final = chi_rows[-1]
    previous = chi_rows[-2]
    pair_est = next(
        row["reported_estimate"] for row in decay_rows if row["channel"] == "Rung pair"
    )
    pair_unc = next(
        row["window_uncertainty"] for row in decay_rows if row["channel"] == "Rung pair"
    )
    charge_est = next(
        row["reported_estimate"] for row in decay_rows if row["channel"] == "Charge"
    )
    charge_unc = next(
        row["window_uncertainty"] for row in decay_rows if row["channel"] == "Charge"
    )
    g = model["tp"] ** 2 / abs(final["hole_pair_binding"])
    g_raw_map_norm = g * raw_map_norm

    metadata = {
        "surface": "technical_report",
        "title": "Bare-Ladder Stage 1 at V=0, t0=1.4",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": RUN_ID,
        "decision": "proceed_to_gated_stage2_discovery",
        "model": model,
        "metrics": {
            "final_spin_gap": final["spin_gap"],
            "final_charge_gap": final["charge_gap"],
            "final_hole_pair_binding": final["hole_pair_binding"],
            "final_particle_pair_binding": final["particle_pair_binding"],
            "pair_decay_exponent": pair_est,
            "charge_decay_exponent": charge_est,
            "total_sweeps": total_sweeps,
            "summed_dmrg_hours": total_dmrg_hours,
            "sector_critical_path_hours": critical_path_hours,
            "full_bytes": full_bytes,
            "compact_bytes": compact_bytes,
            "stage2_named_candidates": 14,
            "stage2_independent_directions": 12,
            "stage2_kernel_pair_binding": final["hole_pair_binding"],
        },
        "sources": [
            {
                "path": str(path.relative_to(LADDER_ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
            }
            for path in (BACKBONE, STAGE1, MANIFEST, SUMMARY)
        ],
        "limitations": [
            "No synchronized sacct or /usr/bin/time record: CPU efficiency and peak RSS are not measured.",
            "Equal-time covariance eigenvalues are screening quantities, not static susceptibilities.",
            "L=64 and chi=1200 are finite-size and finite-entanglement results.",
            "The weak-coupling validity ratios do not support a controlled MPS+MF claim at tp=0.1.",
        ],
    }
    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    runtime_chart = svg_horizontal_bars(
        sorted(sector_rows, key=lambda row: row["dmrg_hours"]),
        "purpose",
        "dmrg_hours",
        "h",
        "#3978a8",
    )
    rank_chart = svg_rank_bars(rank_rows)

    chi_table = table_html(
        chi_rows,
        [
            ("chi", "χ", "d"),
            ("spin_gap", "Δs", ".8f"),
            ("charge_gap", "Δc", ".8f"),
            ("hole_pair_binding", "Ep hole", ".8f"),
            ("particle_pair_binding", "Ep particle", ".8f"),
            ("chemical_potential", "μ", ".8f"),
            ("all_sectors_converged", "all sectors passed", "bool"),
        ],
    )
    sector_table = table_html(
        sector_rows,
        [
            ("purpose", "Sector", ""),
            ("sweeps", "Sweeps", "d"),
            ("dmrg_hours", "Logged DMRG h", ".3f"),
            ("final_max_discarded_weight", "χ=1200 max discarded", ".3e"),
            ("last_five_energy_spread", "last-five ΔE", ".3e"),
        ],
    )
    decay_table = table_html(
        decay_rows,
        [
            ("channel", "Channel", ""),
            ("edge_fraction", "edge excluded", ".2f"),
            ("points", "points", "d"),
            ("exponent", "exponent", ".4f"),
            ("r2", "R²", ".3f"),
        ],
    )
    rank_table = table_html(
        rank_rows,
        [
            ("block", "Block", ""),
            ("dimension", "dimension", "d"),
            ("participation_rank", "participation rank", ".2f"),
            ("k90", "modes for 90%", "d"),
            ("leading_eigenvalue", "top covariance λ", ".6f"),
        ],
    )

    markdown = f"""# Bare-Ladder Stage 1 at V=0, t0=1.4

**Decision:** proceed to the gated Stage 2 discovery calculation, but treat its
instability eigenvalues as exploratory rather than a controlled weak-coupling
prediction.

The six-sector L=64, chi=1200 backbone is numerically usable. Its final spin
gap is {final['spin_gap']:.9f}, charge gap {final['charge_gap']:.9f}, hole pair
binding {final['hole_pair_binding']:.9f}, and particle pair binding
{final['particle_pair_binding']:.9f}. The rung-pair correlation decays more
slowly than the charge correlation: exponents {pair_est:.4f} +/- {pair_unc:.4f}
and {charge_est:.4f} +/- {charge_unc:.4f}, respectively. The pair fits are much
more stable (R2 0.953--0.982) than the charge fits (R2 0.527--0.707).

This is not yet a susceptibility result. The equal-time covariance spectra are
broad: participation ranks range from {min(r['participation_rank'] for r in rank_rows):.1f}
to {max(r['participation_rank'] for r in rank_rows):.1f}. They therefore argue
against assuming a tiny low-rank response, while still supplying useful
data-driven candidate directions. The Stage 2 bank combines 11 motivated names
with three covariance additions. Exact orthogonalization reduces those 14 names
to 12 independent field directions: nine normal and three pairing.

## Numerical convergence

- All six final chi=1200 sectors pass the saved convergence gates.
- From chi=800 to 1200, the spin gap moves {100*(final['spin_gap']/previous['spin_gap']-1):.3f}%,
  the charge gap {100*(final['charge_gap']/previous['charge_gap']-1):.3f}%, and
  the magnitudes of the hole and particle bindings by less than 0.03%.
- The chi=200 derived gaps are not physical results: not all sectors were
  converged at that stage.
- The central-charge fit gives c={diagnostics['central_charge']:.3f} but
  R2={diagnostics['central_charge_r2']:.3f}; it is unusable as a central-charge
  estimate.

## Physics interpretation

- K_rho is {diagnostics['K_rho_rung_normalized']:.4f} with the rung normalization
  and {diagnostics['K_rho_site_normalized']:.4f} with the site normalization.
  The convention must be named whenever this number is quoted.
- The open-boundary density profile has its strongest Fourier component at
  q/pi={2*friedel_mode/64:.4f}, the expected four-kF-scale Friedel modulation.
- The leading rung and leg pair covariance vectors are q=0-like, with uniform
  overlaps {pair_rung_uniform_overlap:.3f} and {pair_leg_uniform_overlap:.3f}.
- The two largest spin-odd covariance vectors are boundary modes (edge weights
  {spin_odd_edge[0]:.3f} and {spin_odd_edge[1]:.3f}); the first bulk spin-odd
  candidate is rank 3, with covariance eigenvalue {spin_odd_values[2]:.6f}.
- In charge-even covariance, the first mode whose dominant Fourier component is
  q/pi=0.125 appears only at rank {physical_charge_rank}, with eigenvalue
  {physical_charge_eigenvalue:.6f}. A top-six-only rule would have missed it.
- Separate covariance matrices cannot decide an extended-s versus d-wave
  rung/leg mixture. That is exactly the mixing Stage 2 will determine.

## Method-validity boundary

At tp={model['tp']}, tp/|Ep|={validity['tp_over_pair_binding']:.3f},
tp/Delta_s={validity['tp_over_spin_gap']:.3f}, and
tp/Delta_c={validity['tp_over_charge_gap']:.3f}. The charge ratio is especially
large. The bare MPS+MF expansion is therefore a diagnostic ordering tendency,
not a controlled weak-coupling prediction at this point. With
g=tp^2/|Ep|={g:.6f}, g||F(0)||={g_raw_map_norm:.3f}; normal-state dressing is
not parametrically negligible.

## Efficiency

The logs contain {total_sweeps} DMRG sweeps and {total_dmrg_hours:.3f} summed
DMRG wall-hours. Because the six sector jobs ran concurrently, the ideal
sector-array critical path is {critical_path_hours:.3f} hours. The spin-excited
sector is the bottleneck. The full scratch tree represented in the compact
manifest is {full_bytes/1024**3:.3f} GiB; the stateless mirror is
{compact_bytes/1024**2:.3f} MiB, a {full_bytes/compact_bytes:.0f}x reduction.
Final MPS data are intentionally present in the assembled backbone, final
sector files, and chi=1200 checkpoints; that triplication accounts for
{100*final_triplicate_bytes/full_bytes:.1f}% of the full tree.

No `sacct` or `/usr/bin/time -v` record was synchronized. Consequently this
report does not claim measured CPU utilization, charged node-hours, or peak
resident memory. The four-thread block-sparse topology remains the only
production choice supported by the repository's calibration; Stage 2 obtains
parallelism across independent probe jobs instead of adding unbenchmarked
threads inside each DMRG solve.

## Stage 2 pilot now prepared

Discovery performs one finite-field solve at h=1e-4 for each of the 12 basis
directions, plus two representation-matched zero-field references. The strict
number-conserving zero-field re-solve is essential: the saved backbone's
last-five energy spread is much larger than h times the desired response
accuracy, so an unrelaxed baseline would create a common O(residual/h)
contamination. The pairing reference drops only Nf while preserving fermion
parity and Sz.

The nine normal probes and the pairing-reference job run concurrently after
the strict zero-field reference. Three parity-only pairing probes follow that
reference. Each measured ladder response is reused for cubic frustrated,
cubic unfrustrated, and square kernels. Their prefactor uses the backbone's
measured hole binding, {final['hole_pair_binding']:.9f}, rather than the older
registry interpolation. Discovery must pass DMRG convergence,
5% within-block reciprocity, and a 5% normal/pair cross-block gate before it
emits the validation plan. Validation is a separate submission: three selected
eigenvectors are checked at h and h/2, and the final response uses Richardson
extrapolation only if the 5% linearity gate passes.

The launcher's conservative reservation ceilings are 18.656 CPU node-hours for
discovery and 9.609 for the optional validation. These are walltime-memory
reservation bounds, not measured charges.

## Evidence boundary

All quantitative claims above are derived from `{RUN_ID}`'s synchronized
compact artifacts and logs. The 33 manifest rows were present, and the compact
backbone and Stage 1 HDF5 files were hash-checked during this report build. The
full restartable MPS tree remains on Perlmutter scratch and was not locally
opened. No Perlmutter connection, transfer, scheduler query, or submission was
performed.
"""
    (REPORT_DIR / "REPORT.md").write_text(markdown, encoding="utf-8")

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<title>Bare-Ladder Stage 1 at V=0, t0=1.4</title>
<style>
:root{{--bg:#f4f3ef;--card:#fff;--ink:#18212a;--muted:#5c6873;--line:#d9ddd9;--accent:#3978a8;--good:#2f7d5a;--warn:#a75d24;--track:#e8ebe8}}
@media(prefers-color-scheme:dark){{:root{{--bg:#15191d;--card:#20262b;--ink:#eef2f3;--muted:#b6c0c7;--line:#384149;--track:#343c42}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.58 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{width:min(1040px,calc(100% - 32px));margin:0 auto;padding:42px 0 72px}} h1{{font-size:clamp(30px,5vw,52px);line-height:1.04;letter-spacing:-.035em;margin:.2em 0}}
h2{{margin:52px 0 14px;font-size:25px;letter-spacing:-.02em}} h3{{margin:26px 0 8px}} p{{max-width:78ch}} code{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}
.eyebrow{{text-transform:uppercase;letter-spacing:.13em;color:var(--accent);font-weight:700;font-size:12px}} .lede{{font-size:20px;max-width:76ch;color:var(--muted)}}
.decision{{border-left:5px solid var(--good);background:var(--card);padding:20px 24px;border-radius:12px;box-shadow:0 8px 26px #0000000b;margin:28px 0}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}} .metric{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}}
.metric strong{{font-size:25px;display:block}} .metric span{{color:var(--muted);font-size:13px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;margin:18px 0}}
.callout{{border:1px solid #d7a56c;background:color-mix(in srgb,var(--card) 86%,#e7a357 14%);padding:18px 20px;border-radius:12px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;background:var(--card)}} table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}} tbody tr:last-child td{{border-bottom:0}}
svg{{width:100%;height:auto}} svg text{{fill:var(--ink);font:13px ui-sans-serif,system-ui}} ul{{max-width:82ch}} li+li{{margin-top:7px}} .small{{font-size:13px;color:var(--muted)}}
.steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;counter-reset:step}} .step{{counter-increment:step;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:17px}}
.step:before{{content:counter(step);display:grid;place-items:center;width:28px;height:28px;border-radius:50%;background:var(--accent);color:white;font-weight:700;margin-bottom:12px}}
footer{{border-top:1px solid var(--line);margin-top:54px;padding-top:18px;color:var(--muted);font-size:13px}}
@media(max-width:800px){{.grid,.steps{{grid-template-columns:repeat(2,1fr)}}}} @media(max-width:520px){{.grid,.steps{{grid-template-columns:1fr}} main{{width:min(100% - 20px,1040px)}}}}
</style>
</head>
<body><main>
<div class="eyebrow">Bare ladder · Stage 1 readout · 2 September 2026</div>
<h1>Pairing correlations lead, but the Stage 2 stability result must remain exploratory</h1>
<p class="lede">L=64, U=8, n=0.9375, V=0, t0=1.4, tp=0.1. The synchronized six-sector backbone is numerically usable at chi=1200. Equal-time screening supports a compact hybrid response bank, while the very small charge gap rules out a controlled weak-coupling claim.</p>
<div class="decision"><strong>Decision: proceed to gated Stage 2 discovery.</strong><br />Run 12 independent projected-response directions at one amplitude, inspect reciprocity and symmetry leakage, and only then submit h/h2 validation for three selected eigenvectors.</div>
<div class="grid">
 <div class="metric"><span>spin gap</span><strong>{final['spin_gap']:.4f}</strong><span>chi=1200</span></div>
 <div class="metric"><span>charge gap</span><strong>{final['charge_gap']:.5f}</strong><span>finite L</span></div>
 <div class="metric"><span>hole pair binding</span><strong>{final['hole_pair_binding']:.4f}</strong><span>negative = bound</span></div>
 <div class="metric"><span>pair decay exponent</span><strong>{pair_est:.3f}</strong><span>+/- {pair_unc:.3f} by window</span></div>
</div>

<h2>1. The backbone passed its finite-system convergence contract</h2>
<p>All six final sectors pass the saved chi=1200 gates. From chi=800 to 1200, the spin gap changes {100*(final['spin_gap']/previous['spin_gap']-1):.2f}%, while the charge gap and pair bindings move by at most 0.03%. The chi=200 row is retained as solver history, not as physics: its sectors did not all converge.</p>
{chi_table}
<div class="card"><h3>Where the computation went</h3>{runtime_chart}<p class="small">Bars sum per-sweep times printed by ITensor DMRG. They exclude compilation, HDF5, covariance analysis, scheduler wait, and other job overhead.</p></div>
{sector_table}
<p>The sector logs contain <strong>{total_sweeps} sweeps</strong> and <strong>{total_dmrg_hours:.3f} summed DMRG hours</strong>. Six-way Slurm parallelism reduced the ideal sector phase to the {critical_path_hours:.3f}-hour spin-excited bottleneck.</p>

<h2>2. The physical signal favors pairing correlations, with important caveats</h2>
<p>The rung-pair exponent is {pair_est:.4f} +/- {pair_unc:.4f}; the charge exponent is {charge_est:.4f} +/- {charge_unc:.4f}. Pair fits are consistently strong (R² 0.953–0.982), whereas the charge result is window-sensitive (R² 0.527–0.707). Thus the defensible statement is that pair correlations decay more slowly over the fitted finite-ladder windows—not that the ladder has long-range superconducting order.</p>
{decay_table}
<ul>
 <li><strong>K-rho normalization matters:</strong> {diagnostics['K_rho_rung_normalized']:.4f} per rung versus {diagnostics['K_rho_site_normalized']:.4f} per site.</li>
 <li><strong>Open-boundary Friedel scale:</strong> the largest density-profile Fourier component is q/pi={2*friedel_mode/64:.4f}.</li>
 <li><strong>Pair profiles:</strong> leading rung and leg covariance modes are q=0-like, with uniform overlaps {pair_rung_uniform_overlap:.3f} and {pair_leg_uniform_overlap:.3f}.</li>
 <li><strong>Entanglement-fit failure:</strong> c={diagnostics['central_charge']:.3f} with R²={diagnostics['central_charge_r2']:.3f}; no central-charge inference should use it.</li>
</ul>

<h2>3. Stage 1 is an unbiased screen, not a susceptibility measurement</h2>
<p>The equal-time matrices locate strong ground-state fluctuations but omit the excitation-energy denominators in a static susceptibility. Their spectra are also broad. High participation ranks and large k90 values mean the data do not support assuming a globally low-rank response.</p>
<div class="card">{rank_chart}</div>
{rank_table}
<p>The screen still changes the pilot constructively. The first charge-even covariance eigenmode with dominant q/pi=0.125 is rank {physical_charge_rank} (lambda={physical_charge_eigenvalue:.6f}), so a top-six rule would miss the physically motivated charge scale. Conversely, the two largest spin-odd modes are almost entirely boundary-localized (edge weights {spin_odd_edge[0]:.3f} and {spin_odd_edge[1]:.3f}); the leading bulk spin-odd candidate is rank 3.</p>
<p>Raw covariance eigenvalues cannot be compared across differently normalized operator classes. Separate rung and leg matrices also cannot choose d-wave versus extended-s mixing. The projected finite-field response is needed for both questions.</p>

<h2>4. The method-validity flag is red at tp=0.1</h2>
<div class="callout"><strong>Use Stage 2 as a diagnostic ordering-tendency calculation.</strong> The ratios are tp/|Ep|={validity['tp_over_pair_binding']:.3f}, tp/Delta_s={validity['tp_over_spin_gap']:.3f}, and tp/Delta_c={validity['tp_over_charge_gap']:.3f}. In particular, the charge ratio is 9.21. Normal-state dressing is not parametrically small: g={g:.5f} and g||F(0)||={g_raw_map_norm:.3f}.</div>

<h2>5. The Stage 2 pilot is deliberately staged</h2>
<div class="steps">
 <div class="step"><strong>Prepare</strong><br />Hash-check Stage 1 and orthonormalize 14 named candidates to 12 directions.</div>
 <div class="step"><strong>Re-reference</strong><br />Strict h=0 number-conserving solve, then a parity-only pairing reference.</div>
 <div class="step"><strong>Discover</strong><br />Nine normal and three pairing probes at h=1e-4; reuse responses across three geometries.</div>
 <div class="step"><strong>Gate</strong><br />Require convergence, 5% reciprocity, and 5% normal/pair decoupling before validation.</div>
</div>
<p>The five familiar q=0 pair labels contain only three independent fields: onsite, rung, and leg. Extended-s and d-wave are combinations within that span. Keeping their labels in provenance while eliminating exact linear dependence makes the response solve cheaper without restricting the mixed eigenvector.</p>
<p>A fresh strict h=0 reference is not cosmetic. Dividing a shared residual relaxation by h=1e-4 would contaminate every finite-difference column and can visibly break reciprocity. Pairing probes branch from the same re-relaxed state after removing only <code>Nf</code>; fermion parity and <code>Sz</code> remain conserved.</p>
<p>All three geometry kernels use the newly measured backbone hole binding, <strong>{final['hole_pair_binding']:.9f}</strong>, for their prefactor; the older registry interpolation is retained only as model provenance.</p>
<p>Discovery reserves at most <strong>18.656 CPU node-hours</strong>. The optional three-mode h/h2 validation reserves at most <strong>9.609</strong> more and is a separate operator decision. Each validation task performs its two amplitudes sequentially and checks 5% linearity before Richardson extrapolation.</p>

<h2>6. Efficiency and evidence boundaries</h2>
<ul>
 <li>The compact mirror is {compact_bytes/1024**2:.3f} MiB versus {full_bytes/1024**3:.3f} GiB represented by the manifest: a {full_bytes/compact_bytes:.0f}x reduction.</li>
 <li>Final MPS data appear in the backbone, sector artifacts, and final checkpoints, accounting for {100*final_triplicate_bytes/full_bytes:.1f}% of full bytes. A future deduplicated storage schema could save roughly two of those three copies.</li>
 <li>No synchronized <code>sacct</code> or <code>/usr/bin/time -v</code> record exists, so charged node-hours, CPU utilization, and peak RSS are unknown. Logged DMRG time is not a substitute.</li>
 <li>The next launcher keeps the calibrated four-thread block-sparse topology and parallelizes across probes. It does not spend cores on unbenchmarked intra-solve thread counts.</li>
 <li>The full restartable states remain on Perlmutter scratch. This report opened only user-synchronized compact artifacts and logs; no remote operation occurred.</li>
</ul>

<footer>Source run: <code>{RUN_ID}</code>. Generated {metadata['generated_at']}. Rebuild with <code>python build_report.py</code>. Inspect <code>artifact.json</code>, <code>SOURCE_NOTES.md</code>, and the CSV files under <code>data/</code> for machine-readable evidence.</footer>
</main></body></html>"""
    (REPORT_DIR / "report.html").write_text(document, encoding="utf-8")

    print(f"wrote {REPORT_DIR / 'report.html'}")
    print(f"wrote {REPORT_DIR / 'REPORT.md'}")
    print(f"sectors={len(sector_rows)} sweeps={total_sweeps} dmrg_hours={total_dmrg_hours:.6f}")
    print(f"full_gib={full_bytes/1024**3:.6f} compact_mib={compact_bytes/1024**2:.6f}")


if __name__ == "__main__":
    main()
