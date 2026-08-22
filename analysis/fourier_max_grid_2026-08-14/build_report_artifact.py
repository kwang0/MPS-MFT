#!/usr/bin/env python3
"""Build the canonical portable-report artifact from reviewed audit outputs."""

from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RUN_METRICS = HERE / "run_metrics.csv"
PAIR_CORRELATIONS = HERE / "pair_correlations.csv"
QUERY_FILE = HERE / "report_queries.sql"
OUTPUT = HERE / "artifact.json"


def number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    output = float(value)
    return output if math.isfinite(output) else None


def integer(row: dict[str, str], key: str) -> int | None:
    value = row.get(key, "")
    return None if value == "" else int(float(value))


def boolean(row: dict[str, str], key: str) -> int:
    return 1 if row.get(key, "").lower() == "true" else 0


def load_runs(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE runs (
          run_id TEXT PRIMARY KEY,
          selection TEXT NOT NULL,
          V REAL NOT NULL,
          t0 REAL NOT NULL,
          completed INTEGER NOT NULL,
          corr_sdw REAL NOT NULL,
          corr_dwave REAL NOT NULL,
          corr_cdw REAL NOT NULL,
          sdw_abs_kx_over_pi REAL,
          sdw_ky_over_pi REAL,
          pair_r4 REAL,
          pair_r8 REAL,
          pair_r24 REAL,
          pair_xi REAL,
          dwave_order REAL,
          local_ep REAL,
          tp REAL NOT NULL,
          tp_over_ep REAL,
          dominant TEXT NOT NULL,
          iterations INTEGER NOT NULL,
          density_error REAL NOT NULL,
          gap REAL,
          alpha_below_floor INTEGER NOT NULL
        )
        """
    )
    with RUN_METRICS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            jointly_below = number(row, "alpha_current_last_jointly_below_5e3_fraction") or 0.0
            alpha_max = number(row, "alpha_current_max_abs") or 0.0
            connection.execute(
                """
                INSERT INTO runs VALUES (
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    row["run_id"],
                    row["selection"],
                    number(row, "V"),
                    number(row, "t0"),
                    boolean(row, "completed"),
                    number(row, "corr_sdw_value"),
                    number(row, "corr_dwave_value"),
                    number(row, "corr_cdw_value"),
                    number(row, "corr_sdw_abs_kx_over_pi"),
                    number(row, "corr_sdw_ky_over_pi"),
                    number(row, "pair_corr_abs_r4"),
                    number(row, "pair_corr_abs_r8"),
                    number(row, "pair_corr_abs_r24"),
                    number(row, "pair_exp_xi_4_20"),
                    number(row, "dwave_order_param"),
                    number(row, "local_pair_binding_energy_abs"),
                    number(row, "tp"),
                    number(row, "tp_over_local_pair_binding_abs"),
                    row["corr_dominant_channel"],
                    integer(row, "alpha_iterations"),
                    number(row, "density_absolute_error"),
                    number(row, "gap"),
                    1 if jointly_below == 1.0 and alpha_max < 5.0e-3 else 0,
                ),
            )


def load_pair_correlations(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE pair_correlations (
          run_id TEXT NOT NULL,
          distance INTEGER NOT NULL,
          mean_abs REAL NOT NULL
        )
        """
    )
    with PAIR_CORRELATIONS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            connection.execute(
                "INSERT INTO pair_correlations VALUES (?, ?, ?)",
                (row["run_id"], integer(row, "distance"), number(row, "mean_abs")),
            )


def load_publication_assessment(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE publication_assessment (
          priority INTEGER PRIMARY KEY,
          finding TEXT NOT NULL,
          evidence TEXT NOT NULL,
          current_status TEXT NOT NULL,
          minimum_next_check TEXT NOT NULL
        )
        """
    )
    rows = [
        (
            1,
            "Frustrated stacking stabilizes a uniform d-wave MF solution for attractive V and t0 >= 1.2",
            "Six completed points; mean |D(r=24)| = 0.0118-0.0219 and the qx=0 d-wave Fourier component dominates.",
            "Strong preliminary result; suitable for an internal or methods report, not yet a phase claim.",
            "Length and bond-dimension scaling, two seeds, continuation in both t0 directions, and corrected MF energy comparison.",
        ),
        (
            2,
            "Square stacking weakens SDW order and preserves substantially longer rung-pair correlations than cubic unfrustrated stacking",
            "At common V=0 points the SDW maximum is 24.5%-46.5% smaller and |D(4)| is 3.43-7.20 times larger.",
            "Reportable geometry comparison; the sampled square points still show exponential pair decay.",
            "Extend t0 and transverse hopping scans, then extrapolate correlation length and pair correlations in chi and L.",
        ),
        (
            3,
            "The legacy frustrated V=0 data switch sharply from SDW to d-wave dominance between t0=1.0 and 1.2",
            "The physical SDW maximum changes from 0.357 at t0=1.0 to 0.000337 at 1.2 while the d-wave maximum rises from 0.00207 to 0.0816.",
            "Candidate first-order or hysteretic transition; low-t0 saved states are recurrent/stale.",
            "Scan t0 in steps of 0.025 with AF and d-wave seeds, bidirectional continuation, and a properly double-counted MF functional.",
        ),
        (
            4,
            "The explicit geometries share an incommensurate SDW maximum near the doped antiferromagnetic wavevector",
            "All ten explicit runs peak at |kx|/pi = 25/27 = 0.9259 and ky/pi = 1 after trimming to 54 rungs.",
            "Robust descriptive Fourier observation.",
            "Repeat at multiple L and density to test whether the peak tracks pi*n rather than the finite momentum grid.",
        ),
        (
            5,
            "The positive-V legacy sector is unsettled",
            "All six V=0.2 or 0.4 files are checkpoint-only; several recent Fourier histories remain strongly variable.",
            "Not publishable as converged data.",
            "Restart with improved mixing and fixed-point diagnostics before interpreting the apparent coexistence region.",
        ),
    ]
    connection.executemany("INSERT INTO publication_assessment VALUES (?, ?, ?, ?, ?)", rows)


def parse_queries(text: str) -> dict[str, str]:
    parts = re.split(r"(?m)^-- dataset: ([a-z0-9_]+)\s*$", text)
    output: dict[str, str] = {}
    for index in range(1, len(parts), 2):
        name = parts[index]
        query = parts[index + 1].strip()
        if query.endswith(";"):
            query = query[:-1].rstrip()
        output[name] = query
    return output


def execute_datasets(connection: sqlite3.Connection, queries: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    connection.row_factory = sqlite3.Row
    output: dict[str, list[dict[str, Any]]] = {}
    for name, query in queries.items():
        output[name] = [dict(row) for row in connection.execute(query).fetchall()]
    return output


def source_spec(query_text: str, generated_at: str) -> dict[str, Any]:
    return {
        "id": "run_audit_sql",
        "label": "Reviewed HDF5 run audit and report queries",
        "path": "analysis/fourier_max_grid_2026-08-14/report_queries.sql",
        "query": {
            "engine": "SQLite over reviewed CSV extracts",
            "sql": query_text,
            "executed_at": generated_at,
            "tables_used": ["run_metrics.csv", "pair_correlations.csv"],
            "filters": [
                "Exact plot-order suffix and geometry filters",
                "t0 in 0.8, 1.0, 1.2, 1.4, 1.6",
                "Five boundary rungs trimmed for Fourier observables",
            ],
            "metric_definitions": [
                "corr_sdw and corr_dwave are maxima of normalized absolute Fourier spectra reconstructed from the final measured correlation matrices.",
                "The plot itself uses stored alpha and beta fields; cross-geometry conclusions instead use geometry-neutral correlations.",
                "Pair D(r) is the mean absolute rung-singlet pair-pair correlation at separation r after excluding eight edge rungs.",
                "The reported gap is the saved orthogonal-state DMRG gap, not a sector-resolved spin or charge gap.",
            ],
        },
    }


def literature_sources() -> list[dict[str, Any]]:
    return [
        {
            "id": "bollmark_2023",
            "label": "Bollmark et al., Phys. Rev. X 13, 011039 (2023)",
            "href": "https://doi.org/10.1103/PhysRevX.13.011039",
        },
        {
            "id": "bollmark_2025",
            "label": "Bollmark, Koehler, and Kantian, Phys. Rev. B 111, 125141 (2025)",
            "href": "https://doi.org/10.1103/PhysRevB.111.125141",
        },
        {
            "id": "dolfi_2015",
            "label": "Dolfi et al., Phys. Rev. B 92, 195139 (2015)",
            "href": "https://doi.org/10.1103/PhysRevB.92.195139",
        },
        {
            "id": "shen_2023",
            "label": "Shen, Zhang, and Qin, Phys. Rev. B 108, 165113 (2023)",
            "href": "https://doi.org/10.1103/PhysRevB.108.165113",
        },
        {
            "id": "white_2002",
            "label": "White, Affleck, and Scalapino, Phys. Rev. B 65, 165122 (2002)",
            "href": "https://doi.org/10.1103/PhysRevB.65.165122",
        },
        {
            "id": "noack_1997",
            "label": "Noack et al., Phys. Rev. B 56, 7162 (1997)",
            "href": "https://doi.org/10.1103/PhysRevB.56.7162",
        },
        {
            "id": "jiang_2018",
            "label": "Jiang and Devereaux, arXiv:1806.01465",
            "href": "https://arxiv.org/abs/1806.01465",
        },
    ]


def chart_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "status_chart",
            "title": "Run completion by requested geometry mode",
            "subtitle": "The default no-option plot includes eight checkpoint-only files",
            "question": "How much of the plotted grid is marked complete?",
            "rationale": "A stacked bar makes the unequal completion coverage visible before any physics comparison.",
            "intent": "status",
            "type": "stackedBar",
            "dataset": "status_by_selection",
            "sourceId": "run_audit_sql",
            "encodings": {
                "x": {"field": "selection_label", "type": "nominal", "label": "Geometry mode"},
                "y": {
                    "fields": ["completed", "checkpoint_only"],
                    "type": "quantitative",
                },
            },
            "xAxisTitle": "Requested plot mode",
            "yAxisTitle": "Files",
            "settings": {"groupMode": "stacked", "showValues": True},
            "layout": "full",
        },
        {
            "id": "legacy_phase_chart",
            "title": "Completed legacy runs separate into SDW- and d-wave-dominant solutions",
            "subtitle": "Color is log10(max d-wave Fourier amplitude / max SDW Fourier amplitude); zero marks equal weight",
            "question": "Where does the cubic-frustrated legacy grid switch its dominant order?",
            "rationale": "The ratio emphasizes competition without conflating the geometry-weighted alpha and beta prefactors.",
            "intent": "comparison",
            "type": "heatmap",
            "dataset": "legacy_phase_map",
            "sourceId": "run_audit_sql",
            "encodings": {
                "x": {"field": "t0_label", "type": "ordinal", "label": "t0 / t"},
                "y": {"field": "V", "type": "quantitative", "label": "V / t"},
                "color": {
                    "field": "log10_dwave_to_sdw",
                    "type": "quantitative",
                    "label": "log10(d / SDW)",
                },
                "tooltip": [
                    {"field": "dominant_order", "type": "text", "label": "Dominant"},
                    {"field": "corr_dwave", "type": "quantitative", "label": "d-wave max"},
                    {"field": "corr_sdw", "type": "quantitative", "label": "SDW max"},
                ],
            },
            "palette": {"kind": "diverging", "midpoint": 0},
            "layout": "full",
        },
        {
            "id": "v0_sdw_chart",
            "title": "Transverse stacking strongly changes the V=0 SDW response",
            "subtitle": "Geometry-neutral SDW Fourier maximum; missing bars were not run",
            "question": "How does the magnetic response compare at common ladder parameters?",
            "rationale": "Grouped discrete bars are used because the parameter coverage is sparse and not a continuous trend series.",
            "intent": "comparison",
            "type": "bar",
            "dataset": "v0_sdw_by_geometry",
            "sourceId": "run_audit_sql",
            "encodings": {
                "x": {"field": "t0_label", "type": "ordinal", "label": "t0 / t"},
                "y": {
                    "fields": ["cubic_frustrated", "cubic_unfrustrated", "square"],
                    "type": "quantitative",
                },
            },
            "settings": {"groupMode": "grouped", "showValues": False},
            "layout": "full",
        },
        {
            "id": "pair_decay_chart",
            "title": "Frustrated order plateaus while explicit-geometry pair correlations decay",
            "subtitle": "log10 of mean absolute rung-singlet pair correlation; representative t0=1.4 runs",
            "question": "Does the pair-pair correlator approach a nonzero plateau or remain short ranged?",
            "rationale": "A logarithmic transform distinguishes exponential decay from the long-distance plateau over several decades.",
            "intent": "trend",
            "type": "line",
            "dataset": "pair_decay",
            "sourceId": "run_audit_sql",
            "encodings": {
                "x": {"field": "distance", "type": "quantitative", "label": "Rung separation r"},
                "y": {
                    "fields": [
                        "frustrated_sc_vminus02_t014",
                        "square_v0_t014",
                        "cubic_unfrustrated_v0_t014",
                    ],
                    "type": "quantitative",
                },
            },
            "xAxisTitle": "Rung separation r",
            "yAxisTitle": "log10 mean |D(r)|",
            "settings": {"showPoints": "always"},
            "layout": "full",
        },
    ]


def table_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "geometry_comparison_table",
            "title": "Square versus cubic-unfrustrated geometry at common V=0 points",
            "subtitle": "Geometry-neutral correlations; ratios at r=4 avoid interpreting numerical-floor values at long range",
            "dataset": "geometry_comparison",
            "sourceId": "run_audit_sql",
            "density": "dense",
            "layout": "full",
            "defaultSort": {"field": "t0", "direction": "asc"},
            "columns": [
                {"field": "t0", "label": "t0 / t", "format": "number"},
                {"field": "cubic_sdw", "label": "Cubic SDW", "format": "number"},
                {"field": "square_sdw", "label": "Square SDW", "format": "number"},
                {"field": "square_sdw_reduction_pct", "label": "SDW reduction (%)", "format": "number"},
                {"field": "cubic_pair_r4", "label": "Cubic |D(4)|", "format": "number"},
                {"field": "square_pair_r4", "label": "Square |D(4)|", "format": "number"},
                {"field": "square_to_cubic_pair_r4_ratio", "label": "Square / cubic D(4)", "format": "number"},
                {"field": "cubic_pair_xi", "label": "Cubic xi fit", "format": "number"},
                {"field": "square_pair_xi", "label": "Square xi fit", "format": "number"},
            ],
        },
        {
            "id": "pair_summary_table",
            "title": "Representative pair-correlation scales",
            "subtitle": "The large frustrated correlation length is a plateau diagnostic, not a controlled critical exponent",
            "dataset": "pair_summary",
            "sourceId": "run_audit_sql",
            "density": "dense",
            "layout": "full",
            "defaultSort": {"field": "case_label", "direction": "asc"},
            "columns": [
                {"field": "case_label", "label": "Case", "type": "text"},
                {"field": "V", "label": "V / t", "format": "number"},
                {"field": "t0", "label": "t0 / t", "format": "number"},
                {"field": "pair_r4", "label": "|D(4)|", "format": "number"},
                {"field": "pair_r8", "label": "|D(8)|", "format": "number"},
                {"field": "pair_r24", "label": "|D(24)|", "format": "number"},
                {"field": "pair_xi", "label": "Exp. xi, r=4..20", "format": "number"},
                {"field": "dwave_order", "label": "Saved d-wave order", "format": "number"},
            ],
        },
        {
            "id": "hierarchy_table",
            "title": "Runs that fail the necessary t_perp < |E_p| hierarchy",
            "subtitle": "The stronger requirement is t_perp below every single-ladder gap; spin-gap coverage is missing",
            "dataset": "perturbative_hierarchy_flags",
            "sourceId": "run_audit_sql",
            "density": "dense",
            "layout": "full",
            "defaultSort": {"field": "tp_over_ep", "direction": "desc"},
            "columns": [
                {"field": "geometry", "label": "Geometry", "type": "text"},
                {"field": "V", "label": "V / t", "format": "number"},
                {"field": "t0", "label": "t0 / t", "format": "number"},
                {"field": "local_ep", "label": "|E_p| / t", "format": "number"},
                {"field": "tp", "label": "t_perp / t", "format": "number"},
                {"field": "tp_over_ep", "label": "t_perp / |E_p|", "format": "number"},
            ],
        },
        {
            "id": "publication_table",
            "title": "Publication-readiness assessment",
            "subtitle": "Results are ranked by evidential strength, not visual prominence",
            "dataset": "publication_assessment",
            "sourceId": "run_audit_sql",
            "density": "spacious",
            "layout": "full",
            "defaultSort": {"field": "finding", "direction": "asc"},
            "columns": [
                {"field": "finding", "label": "Finding", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
                {"field": "current_status", "label": "Current status", "type": "text"},
                {"field": "minimum_next_check", "label": "Minimum next check", "type": "text"},
            ],
        },
        {
            "id": "all_runs_table",
            "title": "All 28 runs selected by the three plot calls",
            "subtitle": "Fourier quantities use final geometry-neutral correlation matrices; checkpoint rows are diagnostic only",
            "dataset": "all_runs",
            "sourceId": "run_audit_sql",
            "density": "dense",
            "layout": "full",
            "defaultSort": {"field": "geometry", "direction": "asc"},
            "columns": [
                {"field": "geometry", "label": "Geometry", "type": "text"},
                {"field": "V", "label": "V / t", "format": "number"},
                {"field": "t0", "label": "t0 / t", "format": "number"},
                {"field": "review_status", "label": "Review status", "type": "text"},
                {"field": "dominant", "label": "Dominant", "type": "text"},
                {"field": "corr_sdw", "label": "SDW max", "format": "number"},
                {"field": "corr_dwave", "label": "d-wave max", "format": "number"},
                {"field": "corr_cdw", "label": "CDW max", "format": "number"},
                {"field": "sdw_abs_kx_over_pi", "label": "|kx| / pi", "format": "number"},
                {"field": "sdw_ky_over_pi", "label": "ky / pi", "format": "number"},
                {"field": "pair_r8", "label": "|D(8)|", "format": "number"},
                {"field": "gap", "label": "Orthogonal gap", "format": "number"},
                {"field": "iterations", "label": "MF iterations", "format": "number"},
                {"field": "density_error", "label": "|n - target|", "format": "number"},
                {"field": "tp_over_ep", "label": "t_perp / |E_p|", "format": "number"},
            ],
        },
    ]


def card_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "selected_card",
            "dataset": "overview",
            "sourceId": "run_audit_sql",
            "metrics": [{"label": "Selected files", "field": "selected_runs", "format": "number"}],
        },
        {
            "id": "completed_card",
            "dataset": "overview",
            "sourceId": "run_audit_sql",
            "metrics": [{"label": "Marked complete", "field": "completed_runs", "format": "number"}],
        },
        {
            "id": "checkpoint_card",
            "dataset": "overview",
            "sourceId": "run_audit_sql",
            "metrics": [{"label": "Checkpoint only", "field": "checkpoint_runs", "format": "number"}],
        },
        {
            "id": "plateau_card",
            "dataset": "overview",
            "sourceId": "run_audit_sql",
            "metrics": [
                {
                    "label": "Frustrated SC plateaus",
                    "field": "frustrated_pair_plateau_runs",
                    "format": "number",
                }
            ],
        },
    ]


def report_blocks() -> list[dict[str, Any]]:
    return [
        {
            "id": "executive_summary",
            "type": "markdown",
            "sourceId": "run_audit_sql",
            "body": """## Executive readout

The most defensible result is a geometry hierarchy, not a finished phase diagram. The legacy no-option files reconstruct the **cubic-frustrated Appendix-E geometry exactly**. In that geometry, all six completed `V=-0.4,-0.2`, `t0=1.2,1.4,1.6` points are uniform d-wave-dominant solutions and their rung-singlet pair correlator approaches a large long-distance plateau. At `V=0`, the same geometry changes abruptly from a strong SDW solution at `t0=1.0` to a d-wave solution at `t0=1.2`.

The explicit **cubic-unfrustrated** and **square** runs remain SDW dominated. Square stacking nevertheless suppresses the SDW maximum by 24.5%-46.5% and enhances short/intermediate-distance pair correlations at every common `V=0` point. Its best sampled point, `t0=1.4`, has an exponential pair-fit length `xi=2.81` rungs, versus `0.87` for cubic unfrustrated; neither explicit geometry shows a nonzero pair plateau.

Overall verdict: **share with substantial numerical caveats**. The frustrated pairing plateau and square-versus-cubic contrast are reportable preliminary observations. A transition location, thermodynamic order, positive-`V` behavior, and cross-geometry energies are not yet publication-grade.""",
        },
        {
            "id": "metric_strip",
            "type": "metric-strip",
            "cardIds": ["selected_card", "completed_card", "checkpoint_card", "plateau_card"],
        },
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "run_audit_sql",
            "body": """## Scope and observable definitions

This audit mirrors three calls with their default `source=:mf`, five-rung boundary trim, absolute normalized Fourier transform, and `t0=0.8:0.2:1.6`: no geometry option, `:cubic_unfrustrated`, and `:square`. The exact suffix rules select 18 legacy `_nodamping.h5` files plus five files in each explicit geometry.

The plotted `alpha` and `beta` fields include geometry-dependent coordination factors, so their colors are not directly comparable across stackings. Cross-geometry conclusions here use the final measured anomalous and normal correlation matrices instead. `D(r)` denotes the distance-averaged absolute rung-singlet pair-pair correlator. The saved gap is only the energy to an orthogonal DMRG state; it is not a symmetry-sector spin or charge gap.""",
        },
        {"id": "status_chart_block", "type": "chart", "chartId": "status_chart"},
        {
            "id": "quality_text",
            "type": "markdown",
            "sourceId": "run_audit_sql",
            "body": """### Data-quality verdict

Eight legacy files have `completed=false`, yet the plotting function includes them. All six positive-`V` files are in this group. Their latest Fourier magnitudes can sometimes look stable even when the fields have not reached a fixed point, so they are shown only as diagnostics.

Two completed legacy states (`V=0`, `t0=0.8,1.0`) are recurrent/stale: the stored field used by the plot differs materially from the last measured correlations. Their broad AF-versus-SC placement is stable, but their exact saved amplitudes are not final fixed points. Conversely, every explicit-geometry pairing field lies entirely below the `5e-3` convergence masking scale; relative changes of order unity can therefore be labeled converged when their absolute magnitude is tiny. These runs securely establish SDW dominance, but they do not resolve an infinitesimal superconducting field.

The inferred denominator `E_p` agrees with the parameter-specific magnitude in `E_p_values.csv` to about `5e-5` relative error or better for every run, a useful provenance check. However, it is not stored in the HDF5 files.""",
        },
        {"id": "hierarchy_table_block", "type": "table", "tableId": "hierarchy_table"},
        {
            "id": "hierarchy_note",
            "type": "markdown",
            "body": """The perturbative MPS+MF construction requires transverse hopping to be weaker than every gap of the isolated ladder. Six selected files already fail the necessary `t_perp < |E_p|` test. At the baseline `U=8`, `n=0.9375`, `V=0`, `t0=1`, the published isolated-ladder spin gap is about `0.078t`, also below the present `t_perp=0.1t`. Parameter-specific spin gaps are absent for the rest of this grid, so the formal hierarchy is unverified even where `t_perp < |E_p|`.""",
        },
        {
            "id": "findings_heading",
            "type": "markdown",
            "body": """## Main numerical findings

The following panels use only completed rows unless stated otherwise. Momentum signs related by real-field conjugacy are reported as absolute values.""",
        },
        {"id": "legacy_phase_block", "type": "chart", "chartId": "legacy_phase_chart"},
        {
            "id": "legacy_phase_note",
            "type": "markdown",
            "sourceId": "run_audit_sql",
            "body": """For `V=0`, the correlation-based SDW maximum falls from `0.755` at `t0=0.8` and `0.357` at `1.0` to `3.37e-4` at `1.2`, while the d-wave maximum rises from `0.00207` at `1.0` to `0.0816` at `1.2`. The coarse grid and recurrent low-`t0` solutions cannot distinguish a first-order transition from hysteresis or a convergence basin change.

For attractive nearest-neighbor `V`, all completed `t0 >= 1.2` points are uniformly d-wave dominated at `qx=0`. Their mean `|D(24)|` increases from `0.0118-0.0132` at `t0=1.2` to `0.0205-0.0219` at `1.6`, directly corroborating broken-symmetry pairing within the self-consistent model.""",
        },
        {"id": "v0_sdw_block", "type": "chart", "chartId": "v0_sdw_chart"},
        {
            "id": "v0_sdw_note",
            "type": "markdown",
            "sourceId": "run_audit_sql",
            "body": """All ten explicit-geometry runs have their SDW maximum at `|kx|/pi=25/27=0.9259`, `ky/pi=1`, consistent with the nearest available momentum to the doped antiferromagnetic scale `pi*n` for `n=0.9375`. The legacy `V=0,t0=0.8` point is the exception, peaking at `(pi,0)`.

At common `V=0` points, square stacking lowers the SDW maximum by 24.5%, 27.7%, and 46.5% at `t0=0.8,1.0,1.4`, respectively. This is a robust geometry-neutral contrast, although the two low-`t0` square/cubic comparisons include `t_perp >= |E_p|` at `t0=0.8`.""",
        },
        {"id": "geometry_table_block", "type": "table", "tableId": "geometry_comparison_table"},
        {"id": "pair_decay_block", "type": "chart", "chartId": "pair_decay_chart"},
        {
            "id": "pair_decay_note",
            "type": "markdown",
            "sourceId": "run_audit_sql",
            "body": """The frustrated `V=-0.2,t0=1.4` correlator is nearly flat beyond roughly 12 rungs. Square `V=0,t0=1.4` retains much more pair weight than cubic unfrustrated—`|D(4)|=0.0103` versus `0.00163`—but decays to `1.25e-5` by 24 rungs and is well described by an exponential over `r=4..20`. Thus the square data show enhanced pairing susceptibility/correlation length, not superconducting long-range order at the sampled coupling.""",
        },
        {"id": "pair_table_block", "type": "table", "tableId": "pair_summary_table"},
        {
            "id": "symmetry_warning",
            "type": "markdown",
            "body": """### A plotting-specific symmetry warning

In square geometry, the cross-leg `alpha` components are exactly zero by construction. The plot's extended-s and d-wave combinations can therefore become equal (or both be thresholded to zero). That visual equality is **not evidence for a d-wave representation**. Pair symmetry should instead be diagnosed from physical leg/rung pair correlations and their relative signs.""",
        },
        {
            "id": "literature_context",
            "type": "markdown",
            "body": """## Context within the numerical literature

The closest benchmark is [Bollmark et al., PRX 13, 011039 (2023)](https://doi.org/10.1103/PhysRevX.13.011039). It studies the same plain ladder baseline (`U/t=8`, `n=0.9375`), reports isolated-ladder `Delta E_s≈0.078t` and `Delta E_p≈0.134t`, and uses the Appendix-E self-consistency equations reconstructed here for the legacy files. That work deliberately excluded spatially varying density order and warned that converged MF amplitudes retain non-negligible bond-dimension dependence. Its ladder extrapolations used lengths `80-112` and substantially larger bond dimensions (a representative production ground state used `chi=1000`, `L=96`), whereas this grid is uniformly `L=64`, `chi=200`. The present SDW-versus-SC competition is therefore a scientifically interesting extension of the original single-channel calculation, but currently below its numerical control level.

[Bollmark, Koehler, and Kantian, PRB 111, 125141 (2025)](https://doi.org/10.1103/PhysRevB.111.125141) demonstrates a two-channel MPS+MF treatment of CDW and superconductivity for negative-`U` chain arrays and explicitly frames repulsive doped ladders as the demanding next target. The current grid sits naturally in that program, with SDW added as a further competitor.

For the isolated repulsive two-leg ladder, [Dolfi et al., PRB 92, 195139 (2015)](https://doi.org/10.1103/PhysRevB.92.195139) found dominant superconducting correlations at the same filling `n=0.9375`, but only after extrapolating bond dimension and reaching `L=192`; their quoted `K_rho≈1.54` implies algebraic pairing in the Luther-Emery regime. [Shen, Zhang, and Qin, PRB 108, 165113 (2023)](https://doi.org/10.1103/PhysRevB.108.165113) likewise stresses that pair exponents are highly sensitive to reference-bond position, open boundaries, finite size, and truncation. The exponential square/cubic-unfrustrated correlations at `L=64,chi=200` therefore should not be read as contradicting isolated-ladder Luther-Emery physics: the self-consistent magnetic field, geometry, and limited numerical control all differ.

The low-q charge peaks are compatible with boundary-induced density oscillations at a hole scale near `2*pi*delta`, but [White, Affleck, and Scalapino, PRB 65, 165122 (2002)](https://doi.org/10.1103/PhysRevB.65.165122) shows why open-boundary Friedel oscillations can mimic CDW order. A single length cannot establish a stripe/CDW phase. In the broader width-4 Hubbard-cylinder literature, [Jiang and Devereaux (2018)](https://arxiv.org/abs/1806.01465) similarly finds a delicate competition among spin/charge textures and pairing, controlled by band parameters and requiring aggressive truncation and length extrapolations.""",
        },
        {"id": "publication_block", "type": "table", "tableId": "publication_table"},
        {
            "id": "directions",
            "type": "markdown",
            "body": """## Recommended next directions, in priority order

1. **Resolve the frustrated transition first.** Scan `t0=1.00..1.20` in steps of `0.025` for `V=-0.4,-0.2,0`, using independent AF/SDW and d-wave seeds plus continuation in both directions. Save fixed-point residuals and a recurrence label.
2. **Do controlled `L` and `chi` scaling.** At minimum use `L=32,48,64,96,128` and `chi=200,400,800,1200+`, extrapolating energy and long-distance correlations against discarded weight. Refit `D(r)` only in a boundary-safe window that grows with `L`.
3. **Compute the missing isolated-ladder scales at every `(V,t0)`.** Store `E_p`, the spin gap, and the perturbative ratios in each HDF5 file. Reduce `t_perp` where it is not below both gaps, then verify the predicted coupling scaling.
4. **Make the phase comparison variational.** Add the MF double-counting constants and compare a common energy/free-energy functional across converged SDW, CDW, and SC seeds. The saved effective-Hamiltonian eigenvalue alone is not comparable across geometries or branches.
5. **Upgrade convergence.** Require a true fixed point for physically large fields; treat period-2 or longer recurrences as separately labeled solutions; use damping/Anderson or Broyden mixing. Do not let a `5e-3` absolute mask certify tiny pairing fields without a separate relative/absolute criterion.
6. **Turn the square enhancement into a targeted scan.** Extend `t0=1.2..1.8` and vary `t_perp` below the isolated gaps. Track `D(r)`, pair correlation length, and whether the square solution ever develops a plateau before magnetic order disappears.
7. **Add ladder diagnostics used in controlled DMRG.** Measure sector-resolved spin/charge gaps, pair binding, charge and spin structure factors, `K_rho`, entanglement entropy/central charge, and sign-resolved leg/rung pair correlations. Test whether charge and spin wavevectors track density with `L`.
8. **Repair provenance and selection.** Store geometry, `E_p`, `r_range`, git commit, seed lineage, convergence reason, and last measured versus applied fields; make plotting exclude `completed=false` by default or visibly hatch those cells.""",
        },
        {
            "id": "methods",
            "type": "markdown",
            "sourceId": "run_audit_sql",
            "body": """## Methods and reproducibility

`analyze_runs.py` exactly mirrors the three file-selection contracts, reverses HDF5.jl array axes, reproduces the plotting routine's explicit Fourier summation order, infers the transverse geometry and `2*t_perp^2/E_p` prefactor from the final measured correlations, and exports run-, iteration-, spectrum-, and pair-distance-level tables. The reconstructed Fourier maximum was cross-checked against Julia on a representative legacy file.

The report datasets are materialized by the actual SQLite statements in `report_queries.sql` from the reviewed extracts `run_metrics.csv` and `pair_correlations.csv`. No rows were silently dropped: the detail table below contains all 28 selected files. Missing cells indicate datasets not present in the source HDF5 (not zeros).""",
        },
        {"id": "all_runs_block", "type": "table", "tableId": "all_runs_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": """### Interpretation boundary

This analysis establishes reproducible finite-system behavior of the saved self-consistent calculations. It does not establish a thermodynamic phase diagram, a transition temperature, or a sector-resolved excitation spectrum. Mean-field transverse fluctuations are omitted; the no-option low-`t0` states are recurrent; explicit pairing fields are below the current convergence floor; and all production points share one modest `L` and `chi`.""",
        },
    ]


def build_artifact(datasets: dict[str, list[dict[str, Any]]], query_text: str, generated_at: str) -> dict[str, Any]:
    run_source = source_spec(query_text, generated_at)
    sources = [run_source, *literature_sources()]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Fourier-Max Grid Audit: Transverse Geometry, Magnetism, and Pairing",
            "description": "A source-backed technical audit of all runs selected by plot_order_fourier_max_grid() in three transverse-geometry modes.",
            "generatedAt": generated_at,
            "cards": card_specs(),
            "charts": chart_specs(),
            "tables": table_specs(),
            "sources": sources,
            "blocks": report_blocks(),
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
            "accessIssues": [],
        },
        "sources": sources,
    }


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    query_text = QUERY_FILE.read_text(encoding="utf-8")
    connection = sqlite3.connect(":memory:")
    load_runs(connection)
    load_pair_correlations(connection)
    load_publication_assessment(connection)
    datasets = execute_datasets(connection, parse_queries(query_text))
    artifact = build_artifact(datasets, query_text, generated_at)
    OUTPUT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(datasets)} reviewed datasets")


if __name__ == "__main__":
    main()
