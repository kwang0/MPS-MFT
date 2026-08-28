#!/usr/bin/env python3
"""Build the bounded MCP report artifact for the 2026-08-26 Phase 1 readout.

The builder consumes only already-audited Phase 1 tables and the saved reviewed
legacy analysis artifact.  It does not open or modify immutable HDF5 states.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPORT_DIR = Path(__file__).resolve().parent
LADDER_ROOT = REPORT_DIR.parents[2]
REPOSITORY_ROOT = LADDER_ROOT.parent
DATA_DIR = REPORT_DIR / "data"

V2_AUDIT = (
    LADDER_ROOT
    / "output/phase1_gpu/20260823_phase1_gpu_v2/audit-report-20260826/states.tsv"
)
V3_AUDIT = (
    LADDER_ROOT
    / "output/phase1_gpu/20260824_phase1_gpu_v3_float64_history"
    / "audit-win-nextprep-20260825/states.tsv"
)
INDEPENDENT_AUDIT = (
    LADDER_ROOT / "output/phase1_gpu/RUN_ID/audit-local-20260825/states.tsv"
)
LEGACY_ARTIFACT = (
    REPOSITORY_ROOT / "analysis/fourier_max_grid_2026-08-14/artifact.json"
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def finite_float(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def classify_outcome(row: dict[str, str]) -> str:
    if as_bool(row["accepted"]):
        return "accepted"
    if row["class"] == "raw_map_candidate":
        return "raw_map_candidate"
    if row["class"] == "mixer_dependent_candidate":
        return "mixer_dependent_candidate"
    return "excluded"


def outcome_rows(
    campaign_label: str, rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    geometry_labels = {
        "cubic_frustrated": "Frustrated cubic",
        "cubic_unfrustrated": "Unfrustrated cubic",
        "square": "Square",
    }
    short_campaign = {
        "v2 Float32": "v2",
        "v3 Float64": "v3",
        "independent chi=200": "independent",
    }[campaign_label]
    short_geometry = {
        "cubic_frustrated": "frustrated",
        "cubic_unfrustrated": "unfrustrated",
        "square": "square",
    }
    result: list[dict[str, Any]] = []
    for geometry, geometry_label in geometry_labels.items():
        selected = [row for row in rows if row["geometry"] == geometry]
        counts = {
            key: sum(classify_outcome(row) == key for row in selected)
            for key in (
                "accepted",
                "raw_map_candidate",
                "mixer_dependent_candidate",
                "excluded",
            )
        }
        result.append(
            {
                "panel_label": f"{short_campaign} · {short_geometry[geometry]}",
                "campaign": campaign_label,
                "geometry": geometry_label,
                "accepted": counts["accepted"],
                "raw_map_candidate": counts["raw_map_candidate"],
                "mixer_dependent_candidate": counts["mixer_dependent_candidate"],
                "excluded": counts["excluded"],
                "total_branches": len(selected),
            }
        )
    return result


def unfrustrated_pairing_rows(
    v3_rows: list[dict[str, str]], independent_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    campaigns = (
        ("v3 parent lineages", "v3", v3_rows),
        ("independent chi=200", "independent", independent_rows),
    )
    seed_order = {"pairing": 1, "sdw": 2, "cdw": 3}
    result: list[dict[str, Any]] = []
    for campaign_index, (campaign, campaign_short, rows) in enumerate(campaigns):
        selected = sorted(
            (row for row in rows if row["geometry"] == "cubic_unfrustrated"),
            key=lambda row: seed_order[row["seed"]],
        )
        for row in selected:
            alpha = float(row["alpha_max"])
            result.append(
                {
                    "state_label": f"{campaign_short} · {row['seed']}",
                    "campaign": campaign,
                    "seed": row["seed"],
                    "classification": row["class"],
                    "accepted": as_bool(row["accepted"]),
                    "alpha_max": alpha,
                    "log10_alpha_max": math.log10(alpha),
                    "stage_b_floor": 1.0e-4,
                    "log10_stage_b_floor": -4.0,
                    "iterations": int(row["iterations"]),
                    "residual_rel": finite_float(row["residual_rel"]),
                    "energy_gate": as_bool(row["energy_gate"]),
                    "implementation_comparable": False,
                }
            )
    return result


def artifact_source(
    source_id: str,
    label: str,
    *,
    path: str | None = None,
    description: str,
    tables_used: list[str],
    filters: list[str],
    definitions: list[str],
    executed_at: str,
    engine: str,
    sql: str,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "id": source_id,
        "label": label,
        "query": {
            "engine": engine,
            "language": "SQL",
            "sql": sql,
            "description": description,
            "executed_at": executed_at,
            "tables_used": tables_used,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }
    if path is not None:
        source["path"] = path
    return source


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    v2_rows = read_tsv(V2_AUDIT)
    v3_rows = read_tsv(V3_AUDIT)
    independent_rows = read_tsv(INDEPENDENT_AUDIT)
    with LEGACY_ARTIFACT.open("r", encoding="utf-8") as handle:
        legacy = json.load(handle)

    assert len(v2_rows) == len(v3_rows) == len(independent_rows) == 9
    assert sum(as_bool(row["accepted"]) for row in v2_rows) == 0
    assert sum(as_bool(row["accepted"]) for row in v3_rows) == 8
    assert sum(as_bool(row["accepted"]) for row in independent_rows) == 8
    assert sum(row["class"] == "raw_map_candidate" for row in v3_rows) == 1

    phase1_outcomes = (
        outcome_rows("v2 Float32", v2_rows)
        + outcome_rows("v3 Float64", v3_rows)
        + outcome_rows("independent chi=200", independent_rows)
    )
    unfrustrated_sensitivity = unfrustrated_pairing_rows(v3_rows, independent_rows)
    legacy_phase_map = legacy["snapshot"]["datasets"]["legacy_phase_map"]
    legacy_pair_decay = legacy["snapshot"]["datasets"]["pair_decay"]
    legacy_overview = legacy["snapshot"]["datasets"]["overview"][0]
    assert legacy_overview["selected_runs"] == 28
    assert legacy_overview["completed_runs"] == 20
    assert len(legacy_phase_map) == 10
    assert len(legacy_pair_decay) == 24

    rankings = [
        {
            "campaign": "v3 Float64 recovery",
            "geometry": "Frustrated cubic",
            "authorized_order": "CDW seed < SDW seed < pairing seed",
            "relative_offsets_total": "0; +0.007368263458; +0.008447412380",
            "excluded": "None",
            "interpretation": "All three accepted; all retain max|alpha| ≈ 0.011.",
        },
        {
            "campaign": "v3 Float64 recovery",
            "geometry": "Unfrustrated cubic",
            "authorized_order": "CDW seed < SDW seed",
            "relative_offsets_total": "0; +0.000312604756",
            "excluded": "Pairing seed: unaccepted raw-map period-2 candidate",
            "interpretation": "The accepted splitting is tiny; pairing cannot be ranked.",
        },
        {
            "campaign": "v3 Float64 recovery",
            "geometry": "Square",
            "authorized_order": "pairing seed < SDW seed < CDW seed",
            "relative_offsets_total": "0; +0.001045316109; +0.058892533574",
            "excluded": "None",
            "interpretation": "Accepted finite-chi seed-basin ordering only.",
        },
        {
            "campaign": "independent chi=200",
            "geometry": "Frustrated cubic",
            "authorized_order": "SDW seed < CDW seed < pairing seed",
            "relative_offsets_total": "0; +0.000468121858; +0.002485188371",
            "excluded": "None",
            "interpretation": "Winner reverses relative to v3; fingerprints differ across campaigns.",
        },
        {
            "campaign": "independent chi=200",
            "geometry": "Unfrustrated cubic",
            "authorized_order": "SDW seed < pairing seed",
            "relative_offsets_total": "0; +0.000730901361",
            "excluded": "CDW seed: stagnated",
            "interpretation": "Pairing seed converges with max|alpha| = 6.45e-7.",
        },
        {
            "campaign": "independent chi=200",
            "geometry": "Square",
            "authorized_order": "SDW seed < CDW seed < pairing seed",
            "relative_offsets_total": "0; +0.058855958961; +0.059173902674",
            "excluded": "None",
            "interpretation": "Winner reverses relative to v3; no cross-run energy comparison.",
        },
    ]

    supplement_matrix = [
        {
            "question": "Where are the broad signals?",
            "legacy_evidence": "28 selected grid files; 20 marked complete across t0, V, and three geometry modes.",
            "phase1_addition": "One matched representative point with three seed lineages per geometry.",
            "current_boundary": "Phase 1 deepens control; it does not replace the legacy parameter scan.",
        },
        {
            "question": "Is a terminal file scientifically accepted?",
            "legacy_evidence": "A completed flag; eight checkpoint-only files and two recurrent/stale completed states remain in the reviewed grid.",
            "phase1_addition": "Explicit density, energy, Hamiltonian-identity, effective-energy, and recurrence gates.",
            "current_boundary": "Accepted means self-consistent at finite L and chi, not thermodynamic convergence.",
        },
        {
            "question": "Is recurrence physical or mixer-induced?",
            "legacy_evidence": "Recurrent/stale behavior was visible but not phase-resolved or cleanly classified.",
            "phase1_addition": "Twenty-update raw-map probe, separate orbit phases, and mixer-dependent recurrence labels.",
            "current_boundary": "The v3 unfrustrated pairing orbit is still a candidate because its energy gate fails.",
        },
        {
            "question": "Can branches be ranked variationally?",
            "legacy_evidence": "Saved effective-Hamiltonian energies lacked a common double-counted functional.",
            "phase1_addition": "Canonical variational energy with double-counting terms plus fingerprint and geometry gates.",
            "current_boundary": "Only accepted, fingerprint-matched states within one geometry and campaign are ranked.",
        },
        {
            "question": "Can the iteration be reconstructed?",
            "legacy_evidence": "Sparse snapshots; missing history cannot be reconstructed after the fact.",
            "phase1_addition": "Schema-v5 stores complete applied and measured fields at every MF iteration.",
            "current_boundary": "Full MPS restart artifacts remain scratch-only and are not locally verified.",
        },
        {
            "question": "How sensitive is the result to the seed?",
            "legacy_evidence": "Mostly single-basin exploratory runs.",
            "phase1_addition": "Pairing, SDW, and CDW lineages plus a later independent campaign expose ordering reversals and pairing collapse.",
            "current_boundary": "The two chi=200 campaigns have different implementation fingerprints and cannot be energy-ranked together.",
        },
        {
            "question": "What physical observables are available?",
            "legacy_evidence": "Fourier maxima and rung-pair correlations over a broad grid.",
            "phase1_addition": "Gated one-point fields, complete histories, energy decomposition, recurrence, and provenance.",
            "current_boundary": "Connected structure factors, sector gaps, L scaling, and chi scaling are still pending.",
        },
    ]

    verification_boundaries = [
        {
            "boundary": "Compact-data integrity",
            "verified": "v2: 42 artifacts; v3: 50 artifacts; independent campaign: 48 artifacts (saved audit).",
            "not_verified": "No local full-MPS byte/hash comparison for scratch artifacts.",
            "authority": "Compact verifier locally; full verification must run on Perlmutter.",
        },
        {
            "boundary": "Hamiltonian and effective-energy identities",
            "verified": "v3 and independent campaign: zero failures in nine states each.",
            "not_verified": "v2 fails all nine and is screening-only.",
            "authority": "Saved Phase 1 audit tables.",
        },
        {
            "boundary": "Scientific convergence",
            "verified": "Finite-L, finite-chi self-consistency for accepted states.",
            "not_verified": "No L extrapolation, chi extrapolation, discarded-weight extrapolation, or thermodynamic phase claim.",
            "authority": "Phase 1 acceptance contract and convergence documentation.",
        },
        {
            "boundary": "Current allocation control",
            "verified": "Latest user-supplied Perlmutter status: 114.625 node-hours conservatively reserved after the Stage A smoke.",
            "not_verified": "The local synced ledger ends at 114.500 and is not the current accounting authority.",
            "authority": "Live Perlmutter ledger and accounting output.",
        },
        {
            "boundary": "Legacy-grid reproduction in this report",
            "verified": "Saved reviewed artifact, extracts, and query outputs from the 28-file audit were reused exactly.",
            "not_verified": "The original legacy HDF5 grid was not reprocessed during this report build.",
            "authority": "Saved 2026-08-14 legacy analysis artifact.",
        },
    ]

    budget_scenarios = [
        {
            "scenario": "Current: Stage A smoke complete",
            "reserved": 114.625,
            "unreserved": 285.375,
            "new_reserve_from_current": 0.0,
            "hard_cap": 400.0,
            "status": "Observed in user-supplied Perlmutter status",
        },
        {
            "scenario": "After Stage A first branch segments",
            "reserved": 123.625,
            "unreserved": 276.375,
            "new_reserve_from_current": 9.0,
            "hard_cap": 400.0,
            "status": "Plan-only; three chi=400 branches",
        },
        {
            "scenario": "After conditional Stage B first segments",
            "reserved": 129.75,
            "unreserved": 270.25,
            "new_reserve_from_current": 15.125,
            "hard_cap": 400.0,
            "status": "Plan-only; only if the two-lineage pairing gate passes",
        },
    ]

    headline = [{
        "v3_accepted": 8,
        "v3_total": 9,
        "v3_raw_candidate": 1,
        "independent_accepted": 8,
        "independent_total": 9,
        "budget_reserved": 114.625,
        "budget_remaining": 285.375,
        "budget_cap": 400.0,
    }]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets_to_write = {
        "phase1_outcomes": phase1_outcomes,
        "unfrustrated_pairing_sensitivity": unfrustrated_sensitivity,
        "authorized_rankings": rankings,
        "legacy_phase_map": legacy_phase_map,
        "legacy_pair_decay": legacy_pair_decay,
        "supplement_matrix": supplement_matrix,
        "verification_boundaries": verification_boundaries,
        "budget_scenarios": budget_scenarios,
        "headline": headline,
    }
    for name, rows in datasets_to_write.items():
        write_csv(DATA_DIR / f"{name}.csv", rows)

    source_inventory = {
        "generated_at": generated_at,
        "branch": "codex/mps-mft-phase0-refactor",
        "commit": "ec969b3690f2acae05306807c5c9c6195d755f7f",
        "inputs": [
            str(V2_AUDIT.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
            str(V3_AUDIT.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
            str(INDEPENDENT_AUDIT.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
            str(LEGACY_ARTIFACT.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
        ],
        "current_perlmutter_budget_snapshot": {
            "reserved": 114.625,
            "unreserved": 285.375,
            "hard_cap": 400.0,
            "basis": "User-supplied phase1_gpu.sh status output after Stage A smoke job 57620629 completed",
        },
    }
    (DATA_DIR / "source_inventory.json").write_text(
        json.dumps(source_inventory, indent=2) + "\n", encoding="utf-8"
    )

    sources = [
        artifact_source(
            "phase1_v2_audit",
            "Phase 1 v2 screening audit",
            path="ladder_mps_mft/output/phase1_gpu/20260823_phase1_gpu_v2/audit-report-20260826/states.tsv",
            description="Reproduced compact-state campaign audit for the Float32 v2 screening run.",
            tables_used=["20260823_phase1_gpu_v2/audit-report-20260826/states.tsv"],
            filters=["Nine final state.h5 artifacts", "No acceptance overrides"],
            definitions=["Accepted requires every configured scientific gate; v2 has zero accepted states."],
            executed_at="2026-08-26T20:03:19.351Z",
            engine="DuckDB over Julia audit output",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'ladder_mps_mft/output/phase1_gpu/20260823_phase1_gpu_v2/"
                "audit-report-20260826/states.tsv', delim='\\t', header=true);"
            ),
        ),
        artifact_source(
            "phase1_v3_audit",
            "Phase 1 v3 Float64-history audit",
            path="ladder_mps_mft/output/phase1_gpu/20260824_phase1_gpu_v3_float64_history/audit-win-nextprep-20260825/states.tsv",
            description="Audited nine v3 representative-point state lineages from the verified compact mirror.",
            tables_used=["20260824_phase1_gpu_v3_float64_history/audit-win-nextprep-20260825/states.tsv"],
            filters=["Nine final state.h5 artifacts", "Acceptance and recurrence classifications preserved"],
            definitions=[
                "Solution energy is the canonical variational energy; an accepted orbit would use its phase average.",
                "Accepted states pass density, variational-energy, Hamiltonian-identity, and effective-energy gates.",
            ],
            executed_at="2026-08-25T20:22:58.857Z",
            engine="DuckDB over Julia audit output",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'ladder_mps_mft/output/phase1_gpu/"
                "20260824_phase1_gpu_v3_float64_history/"
                "audit-win-nextprep-20260825/states.tsv', delim='\\t', header=true);"
            ),
        ),
        artifact_source(
            "phase1_independent_audit",
            "Independent chi=200 campaign audit and review",
            path="ladder_mps_mft/output/phase1_gpu/RUN_ID/audit-local-20260825/states.tsv",
            description="Audited the accidental standard nine-branch campaign retained as independent-seed evidence.",
            tables_used=["RUN_ID/audit-local-20260825/states.tsv", "RUN_ID/audit-local-20260825/review.md"],
            filters=["Nine final state.h5 artifacts", "Within-campaign rankings only"],
            definitions=[
                "Cross-campaign energy ranking against v3 is refused because the implementation fingerprints differ.",
                "Seed labels identify lineages, not thermodynamic phases.",
            ],
            executed_at="2026-08-26T06:36:02.377Z",
            engine="DuckDB over Julia audit and comparator output",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'ladder_mps_mft/output/phase1_gpu/RUN_ID/"
                "audit-local-20260825/states.tsv', delim='\\t', header=true);"
            ),
        ),
        artifact_source(
            "legacy_review",
            "Reviewed legacy Fourier-grid analysis",
            path="analysis/fourier_max_grid_2026-08-14/artifact.json",
            description="Saved reviewed artifact and SQLite-derived datasets from the 28-file legacy-grid audit.",
            tables_used=["run_metrics.csv", "pair_correlations.csv", "report_queries.sql"],
            filters=[
                "Exact legacy plot-selection contracts",
                "Completed rows only for physics charts",
                "Five boundary rungs trimmed for Fourier observables",
            ],
            definitions=[
                "Legacy heatmap color is log10(max d-wave Fourier amplitude / max SDW Fourier amplitude).",
                "D(r) is the distance-averaged absolute rung-singlet pair-pair correlator.",
            ],
            executed_at="2026-08-15T00:34:14.434Z",
            engine="DuckDB over saved Python/SQLite legacy extracts",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'ladder_mps_mft/docs/reports/phase1_progress_20260826/"
                "data/legacy_phase_map.csv', header=true); "
                "SELECT * FROM read_csv_auto("
                "'ladder_mps_mft/docs/reports/phase1_progress_20260826/"
                "data/legacy_pair_decay.csv', header=true);"
            ),
        ),
        artifact_source(
            "perlmutter_budget",
            "Latest user-supplied Perlmutter launcher status",
            path="ladder_mps_mft/docs/reports/phase1_progress_20260826/data/budget_scenarios.csv",
            description="Authoritative current project-control snapshot printed after the Stage A smoke completed.",
            tables_used=["Perlmutter output/project_budget/additional_node_hours.tsv"],
            filters=["Hard cap counts requested upper bounds", "Early completion is not reclaimed"],
            definitions=[
                "Reserved node-hours are conservative requested upper bounds summed across CPU and GPU jobs.",
                "Unreserved allowance equals the 400-node-hour hard cap minus reserved node-hours.",
            ],
            executed_at="2026-08-26",
            engine="DuckDB over materialized user-supplied Perlmutter status",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'ladder_mps_mft/docs/reports/phase1_progress_20260826/"
                "data/budget_scenarios.csv', header=true);"
            ),
        ),
        artifact_source(
            "report_synthesis",
            "Phase 1 progress report source inventory",
            path="ladder_mps_mft/docs/reports/phase1_progress_20260826/data/source_inventory.json",
            description="Deterministic synthesis of audited Phase 1 tables, the reviewed legacy artifact, and the latest Perlmutter budget snapshot.",
            tables_used=[
                "v2 audit states.tsv",
                "v3 audit states.tsv",
                "independent audit states.tsv",
                "legacy artifact.json",
                "Perlmutter budget status",
            ],
            filters=[
                "No cross-geometry energy ranking",
                "No cross-campaign energy ranking when fingerprints differ",
                "Candidates and stagnated states excluded from energy rankings",
            ],
            definitions=[
                "Relative energy offsets are zeroed separately inside each authorized same-geometry, same-campaign comparison.",
                "Pairing-bearing for the conditional chi=400 gate means max|alpha| >= 1e-4 in every accepted orbit phase.",
            ],
            executed_at=generated_at,
            engine="DuckDB over Python-built report datasets",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'ladder_mps_mft/docs/reports/phase1_progress_20260826/"
                "data/phase1_outcomes.csv', header=true); "
                "SELECT * FROM read_csv_auto("
                "'ladder_mps_mft/docs/reports/phase1_progress_20260826/"
                "data/unfrustrated_pairing_sensitivity.csv', header=true); "
                "SELECT * FROM read_csv_auto("
                "'ladder_mps_mft/docs/reports/phase1_progress_20260826/"
                "data/authorized_rankings.csv', header=true); "
                "SELECT * FROM read_csv_auto("
                "'ladder_mps_mft/docs/reports/phase1_progress_20260826/"
                "data/supplement_matrix.csv', header=true); "
                "SELECT * FROM read_csv_auto("
                "'ladder_mps_mft/docs/reports/phase1_progress_20260826/"
                "data/verification_boundaries.csv', header=true);"
            ),
        ),
    ]

    cards = [
        {
            "id": "v3_accepted_card",
            "dataset": "headline",
            "sourceId": "phase1_v3_audit",
            "description": "Accepted finite-L, finite-chi states in the Float64-history campaign.",
            "metrics": [
                {"label": "v3 accepted states", "field": "v3_accepted", "format": "number"},
                {"label": "out of", "field": "v3_total", "format": "number"},
            ],
        },
        {
            "id": "v3_candidate_card",
            "dataset": "headline",
            "sourceId": "phase1_v3_audit",
            "description": "Unaccepted raw-map period-two candidate retained phase by phase.",
            "metrics": [
                {"label": "unresolved raw candidate", "field": "v3_raw_candidate", "format": "number"}
            ],
        },
        {
            "id": "independent_accepted_card",
            "dataset": "headline",
            "sourceId": "phase1_independent_audit",
            "description": "Accepted states in the later independent-seed standard campaign.",
            "metrics": [
                {"label": "independent accepted", "field": "independent_accepted", "format": "number"},
                {"label": "out of", "field": "independent_total", "format": "number"},
            ],
        },
        {
            "id": "budget_card",
            "dataset": "headline",
            "sourceId": "perlmutter_budget",
            "description": "Conservative project-control ledger after the completed Stage A smoke.",
            "metrics": [
                {"label": "node-hours reserved", "field": "budget_reserved", "format": "number"},
                {"label": "remaining", "field": "budget_remaining", "format": "number"},
                {"label": "hard cap", "field": "budget_cap", "format": "number"},
            ],
        },
    ]

    charts = [
        {
            "id": "phase1_outcome_chart",
            "title": "Phase 1 branch outcomes by campaign and geometry",
            "subtitle": "Each campaign contains three seed lineages per geometry; acceptance rules differ sharply from scheduler completion",
            "question": "Which branches became accepted states rather than candidates or incomplete outcomes?",
            "rationale": "A stacked bar preserves the three-branch denominator while separating acceptance from recurrence and failure modes.",
            "intent": "composition",
            "type": "stackedBar",
            "dataset": "phase1_outcomes",
            "sourceId": "report_synthesis",
            "encodings": {
                "x": {"field": "panel_label", "type": "nominal", "label": "Campaign and geometry"},
                "y": {
                    "fields": ["accepted", "raw_map_candidate", "mixer_dependent_candidate", "excluded"],
                    "type": "quantitative",
                },
            },
            "xAxisTitle": "Campaign and geometry",
            "yAxisTitle": "Branches",
            "settings": {"groupMode": "stacked", "showValues": True},
            "layout": "full",
        },
        {
            "id": "unfrustrated_pairing_chart",
            "title": "Unfrustrated pairing-field magnitude across audited lineages",
            "subtitle": "log10(max|alpha|); the conditional chi=400 survival floor is -4, and cross-campaign energies are not comparable",
            "question": "Does the pairing-bearing basin survive changes in lineage and implementation?",
            "rationale": "A log-transformed categorical bar resolves the eight-order-of-magnitude spread without hiding excluded outcomes.",
            "intent": "comparison",
            "type": "bar",
            "dataset": "unfrustrated_pairing_sensitivity",
            "sourceId": "report_synthesis",
            "encodings": {
                "x": {"field": "state_label", "type": "nominal", "label": "Audited lineage"},
                "y": {"field": "log10_alpha_max", "type": "quantitative", "label": "log10 max|alpha|"},
            },
            "xAxisTitle": "Audited lineage",
            "yAxisTitle": "log10 max|alpha|",
            "settings": {"showValues": False},
            "layout": "full",
        },
        {
            "id": "legacy_phase_chart",
            "title": "Completed legacy frustrated-grid order balance",
            "subtitle": "log10(max d-wave Fourier amplitude / max SDW Fourier amplitude); zero denotes equal weight",
            "question": "Where did the reviewed legacy cubic-frustrated grid switch its dominant channel?",
            "rationale": "The heatmap shows the sparse two-parameter pattern without treating missing or checkpoint-only cells as data.",
            "intent": "matrix",
            "type": "heatmap",
            "dataset": "legacy_phase_map",
            "sourceId": "legacy_review",
            "encodings": {
                "x": {"field": "t0_label", "type": "ordinal", "label": "t0 / t"},
                "y": {"field": "V", "type": "quantitative", "label": "V / t"},
                "color": {"field": "log10_dwave_to_sdw", "type": "quantitative", "label": "log10(d-wave / SDW)"},
                "tooltip": [
                    {"field": "dominant_order", "type": "text", "label": "Dominant"},
                    {"field": "corr_dwave", "type": "quantitative", "label": "d-wave maximum"},
                    {"field": "corr_sdw", "type": "quantitative", "label": "SDW maximum"},
                ],
            },
            "palette": {"kind": "diverging", "midpoint": 0},
            "layout": "full",
        },
        {
            "id": "legacy_pair_decay_chart",
            "title": "Representative legacy rung-pair correlations",
            "subtitle": "log10 mean absolute D(r), r=1..24; the frustrated trace is V=-0.2 while explicit-geometry traces are V=0",
            "question": "Which saved legacy examples plateau and which decay over the available ladder length?",
            "rationale": "A 24-point line comparison exposes a long-distance plateau versus exponential decay over several decades.",
            "intent": "trend",
            "type": "line",
            "dataset": "legacy_pair_decay",
            "sourceId": "legacy_review",
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
        {
            "id": "budget_scenarios_chart",
            "title": "Conservative node-hour envelope for the staged chi=400 calculation",
            "subtitle": "Requested upper bounds under the 400-node-hour project cap; the current row includes the completed smoke",
            "question": "How much allowance remains after each authorized decision gate?",
            "rationale": "A stacked bar keeps the fixed cap visible while separating reserved and deliberately uncommitted allowance.",
            "intent": "status",
            "type": "stackedBar",
            "dataset": "budget_scenarios",
            "sourceId": "perlmutter_budget",
            "encodings": {
                "x": {"field": "scenario", "type": "nominal", "label": "Scenario"},
                "y": {"fields": ["reserved", "unreserved"], "type": "quantitative"},
            },
            "xAxisTitle": "Scenario",
            "yAxisTitle": "Node-hours",
            "settings": {"groupMode": "stacked", "showValues": True},
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "authorized_rankings_table",
            "title": "Authorized same-geometry rankings",
            "subtitle": "Offsets are total canonical variational energies zeroed separately within each campaign and geometry",
            "dataset": "authorized_rankings",
            "sourceId": "report_synthesis",
            "density": "spacious",
            "layout": "full",
            "defaultSort": {"field": "geometry", "direction": "asc"},
            "columns": [
                {"field": "campaign", "label": "Campaign", "type": "text"},
                {"field": "geometry", "label": "Geometry", "type": "text"},
                {"field": "authorized_order", "label": "Accepted order", "type": "text"},
                {"field": "relative_offsets_total", "label": "Relative total-energy offsets", "type": "text"},
                {"field": "excluded", "label": "Excluded", "type": "text"},
                {"field": "interpretation", "label": "Interpretation", "type": "text"},
            ],
        },
        {
            "id": "supplement_matrix_table",
            "title": "How Phase 1 supplements the legacy grid",
            "subtitle": "The two evidence layers answer different parts of the same scientific program",
            "dataset": "supplement_matrix",
            "sourceId": "report_synthesis",
            "density": "spacious",
            "layout": "full",
            "defaultSort": {"field": "question", "direction": "asc"},
            "columns": [
                {"field": "question", "label": "Question", "type": "text"},
                {"field": "legacy_evidence", "label": "Legacy evidence", "type": "text"},
                {"field": "phase1_addition", "label": "Phase 1 addition", "type": "text"},
                {"field": "current_boundary", "label": "Current boundary", "type": "text"},
            ],
        },
        {
            "id": "verification_boundaries_table",
            "title": "Verification boundaries at this report cutoff",
            "subtitle": "Local compact checks, Perlmutter authority, and scientific convergence are distinct claims",
            "dataset": "verification_boundaries",
            "sourceId": "report_synthesis",
            "density": "spacious",
            "layout": "full",
            "defaultSort": {"field": "boundary", "direction": "asc"},
            "columns": [
                {"field": "boundary", "label": "Boundary", "type": "text"},
                {"field": "verified", "label": "Verified", "type": "text"},
                {"field": "not_verified", "label": "Not verified", "type": "text"},
                {"field": "authority", "label": "Authority", "type": "text"},
            ],
        },
    ]

    title = "Phase 1 Progress and the Legacy Ladder MPS+MF Grid"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "## Technical summary\n\n"
                "The legacy grid supplied the broad hypothesis: frustrated stacking supports a pairing-dominant basin for attractive `V` at larger `t0`, while explicit cubic-unfrustrated and square runs remain SDW-dominant, with square stacking preserving longer short-range pair correlations. Phase 1 has **not yet confirmed a thermodynamic phase**. What it adds is a controlled finite-system test at `L=64`, `chi=200`, `U=8`, `V=-0.2`, `t0=1.1`, `t_perp=0.1`, and density `0.9375`.\n\n"
                "The v3 Float64-history campaign yields **eight accepted fixed points and one unaccepted raw-map period-two candidate**. It makes within-geometry canonical-energy rankings possible for the first time, retains complete mean-field histories, and separates raw-map physics from Anderson acceleration. The later independent campaign also yields eight accepted states, but reverses several seed-lineage winners and collapses the unfrustrated pairing seed to `max|alpha|=6.45e-7`. Because its implementation fingerprint differs from v3, that contrast is seed-sensitivity evidence—not a cross-run energy ranking.\n\n"
                "The immediate open question is therefore narrow and valuable: at `chi=400`, does the unfrustrated pairing basin survive both a phase-resolved parent continuation and a fresh independent seed, as an accepted fixed point or accepted period-two orbit? The Stage A smoke has passed; no Stage A branch result is included in this report cutoff."
            ),
        },
        {
            "id": "metric_strip",
            "type": "metric-strip",
            "cardIds": [
                "v3_accepted_card",
                "v3_candidate_card",
                "independent_accepted_card",
                "budget_card",
            ],
        },
        {
            "id": "definitions",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "## How to read the evidence\n\n"
                "An **accepted state** is a period-one fixed point or explicitly enabled unmixed orbit that passes density, recurrence, variational-energy, Hamiltonian-identity, and effective-energy gates. Scheduler completion alone is not acceptance. A **seed label** records lineage; it is not a phase name.\n\n"
                "Every energy offset below uses the canonical zero-temperature variational functional, including mean-field double-counting terms. Periodic solutions would use the orbit-phase average. Energies are ranked only among accepted states with matching fingerprints inside one transverse geometry. No absolute or relative energy comparison across geometries is made."
            ),
        },
        {
            "id": "outcomes_heading",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "## Float64 and explicit gates convert screening into accepted finite-system states\n\n"
                "The v2 jobs all reached terminal scheduler outcomes, but none passed the physics gates: Float32 storage produced Hamiltonian and effective-energy inconsistencies, while premature recurrence handling blurred raw and mixer-dependent behavior. The v3 recovery changes the evidential status, not merely the success count: eight branches are accepted under the repaired Float64 and complete-history contract. The independent campaign reproduces eight accepted fixed points but exposes strong seed sensitivity."
            ),
        },
        {"id": "outcomes_chart_block", "type": "chart", "chartId": "phase1_outcome_chart"},
        {
            "id": "outcomes_note",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "### Interpretation\n\n"
                "The acceptance gain is strongest evidence of what Phase 1 contributes. It does not mean every accepted branch is physically robust: accepted states can still move with bond dimension, length, or seed. Conversely, the unaccepted v3 raw-map orbit remains scientifically interesting because it is observed before mixing and preserved phase by phase."
            ),
        },
        {
            "id": "rankings_heading",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "## Same-geometry energy winners change with lineage, so chi=200 rankings are provisional\n\n"
                "Within v3, the frustrated CDW-seeded branch is lowest while all three accepted frustrated states retain nearly the same pairing-field magnitude. The later independent campaign instead places the SDW-seeded branch lowest. Square likewise changes from a pairing-seeded v3 winner to an SDW-seeded independent winner. These are not contradictory energy measurements: the two campaigns have different implementation fingerprints and cannot be ranked against one another. They are a warning that small finite-`chi` seed-basin splittings are not yet robust."
            ),
        },
        {"id": "rankings_table_block", "type": "table", "tableId": "authorized_rankings_table"},
        {
            "id": "rankings_note",
            "type": "markdown",
            "body": (
                "### Reading the table\n\n"
                "Each row is a self-contained authorized comparison. Offsets start from zero again in every row. Do not compare their absolute energies, winners, or offset magnitudes across geometries or campaigns as if they belonged to one Hamiltonian or one numerical fingerprint."
            ),
        },
        {
            "id": "recurrence_heading",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "## The unfrustrated pairing basin is the decisive unresolved Phase 1 question\n\n"
                "The v3 pairing lineage retains `max|alpha|=0.01925` and exhibits an unmixed period-two recurrence, but its orbit-energy spread and variational-energy recurrence fail acceptance. In the independent campaign, the pairing seed instead reaches an accepted period-one fixed point with `max|alpha|=6.45e-7`; its CDW seed retains a large pairing field but stagnates and is excluded. The chart is therefore diagnostic only: it compares field magnitudes and outcomes, not energies, across different implementation fingerprints."
            ),
        },
        {"id": "recurrence_chart_block", "type": "chart", "chartId": "unfrustrated_pairing_chart"},
        {
            "id": "recurrence_note",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "### Why Stage A is phase-resolved\n\n"
                "On this logarithmic scale, `-4` corresponds to the conditional Stage B survival floor `max|alpha|=1e-4`. Stage A separately continues v3 orbit phases 001 and 002 from their hash-pinned full MPS parents and adds an independent pairing seed. It performs twenty raw updates and stops before Anderson. Anderson may later accelerate a recurrence-free fixed point; it cannot certify, average, or erase a physical raw-map orbit."
            ),
        },
        {
            "id": "legacy_heading",
            "type": "markdown",
            "sourceId": "legacy_review",
            "body": (
                "## The legacy grid supplies breadth that Phase 1 has not yet reproduced\n\n"
                "Among completed legacy cubic-frustrated rows, attractive `V=-0.4` and `-0.2` at `t0=1.2,1.4,1.6` are d-wave-dominant by the reviewed correlation-based Fourier measure. At `V=0`, the sampled grid switches sharply from SDW dominance at `t0=1.0` to d-wave dominance at `1.2`. This locates the motivation for the Phase 1 representative point at `t0=1.1`, but the coarse legacy grid cannot determine whether the switch is a transition, hysteresis, or a basin/convergence change."
            ),
        },
        {"id": "legacy_phase_block", "type": "chart", "chartId": "legacy_phase_chart"},
        {
            "id": "legacy_pair_intro",
            "type": "markdown",
            "sourceId": "legacy_review",
            "body": (
                "### Correlation evidence behind the broad legacy picture\n\n"
                "A representative frustrated `V=-0.2,t0=1.4` rung-pair correlator approaches a long-distance plateau over the available `L=64` ladder, whereas square and cubic-unfrustrated `V=0,t0=1.4` examples decay. Square retains substantially more short- and intermediate-distance pair weight than cubic unfrustrated, but neither explicit geometry shows a nonzero plateau in these saved runs."
            ),
        },
        {"id": "legacy_pair_block", "type": "chart", "chartId": "legacy_pair_decay_chart"},
        {
            "id": "legacy_pair_note",
            "type": "markdown",
            "sourceId": "legacy_review",
            "body": (
                "### Interpretation boundary\n\n"
                "The three traces are illustrative, not a matched phase competition: the frustrated example uses `V=-0.2`, the explicit-geometry examples use `V=0`, and their mean-field field normalizations differ by geometry. The figure supports the legacy qualitative hypotheses—frustrated plateau and square enhancement—not a cross-geometry energy claim or thermodynamic extrapolation."
            ),
        },
        {
            "id": "supplement_heading",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "## Phase 1 is a control layer on top of the legacy discovery layer\n\n"
                "The legacy runs remain valuable because they cover parameter space and contain correlation observables that the current representative-point audit does not yet replace. Phase 1 contributes the numerical contract needed to decide which saved solutions can support a variational comparison. The combined evidence is stronger than either layer alone, but it is still pre-scaling evidence."
            ),
        },
        {"id": "supplement_table_block", "type": "table", "tableId": "supplement_matrix_table"},
        {
            "id": "methodology_heading",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "## Scope, data, and methodology\n\n"
                "Report cutoff: 26 August 2026. The Phase 1 evidence uses the locally synced compact mirrors and saved audits on branch `codex/mps-mft-phase0-refactor` at commit `ec969b3`. Compact verification was rerun for v2 and v3 without `--full`; the legacy figures reuse the reviewed datasets saved in the 14 August analysis artifact. No HDF5 state was modified.\n\n"
                "The representative-point cohort is `L=64`, `U=8`, `V=-0.2`, `t0=1.1`, `t_perp=0.1`, density `0.9375`, and `chi=200`, across frustrated cubic, unfrustrated cubic, and square transverse geometries. V3 lineages inherit v2 fields; the later standard campaign uses independent seeds. The two campaigns share the model point but not the implementation fingerprint.\n\n"
                "The report preserves exact statuses and excludes candidates and stagnated branches from all energy rankings. It reports seed lineages rather than assigning thermodynamic phase names."
            ),
        },
        {
            "id": "verification_heading",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "## Verification is strong at the compact finite-system level and deliberately incomplete beyond it\n\n"
                "Local verification establishes compact hashes, sizes, stateless markers, provenance links, and MPS omission for the v2 and v3 mirrors. Saved audits establish the configured finite-system gates. They do not verify the Perlmutter full scratch artifacts, actual allocation charge, or scientific convergence under increasing `L` and `chi`."
            ),
        },
        {"id": "verification_table_block", "type": "table", "tableId": "verification_boundaries_table"},
        {
            "id": "limitations_heading",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "## Why these results still do not establish a thermodynamic phase\n\n"
                "All current controlled states are at one length and `chi=200`; the χ=400 scientific branches are not yet in the report. Accepted self-consistency does not bound finite-entanglement or open-boundary error. The legacy grid includes checkpoint-only and recurrent/stale rows, and its saved effective energies are not branch-comparable through the new canonical functional. The refactored campaigns add one-point fields and recurrence control, but connected spin/charge structure factors, sign-resolved pair correlations, sector gaps, entanglement scaling, and controlled `L`/`chi` extrapolations remain pending.\n\n"
                "Accordingly, the report supports finite-system observations and a prioritized calculation. It does not claim superconducting, SDW, or CDW long-range order; a transition location; or a ranking between transverse geometries."
            ),
        },
        {
            "id": "next_heading",
            "type": "markdown",
            "sourceId": "perlmutter_budget",
            "body": (
                "## Recommended next calculation: complete only Stage A's first chi=400 segments\n\n"
                "The exact next calculation is the prepared three-branch unfrustrated recurrence campaign: v3 orbit phases 001 and 002 as separate full-MPS parents plus independent pairing seed s2, with `chi=400`, 16 sweeps, cutoff `1e-11`, energy tolerance `1e-9`, and a twenty-update unmixed probe that stops before Anderson. The smoke is already complete. Submitting the three first branch segments would reserve **9.000 additional node-hours**, moving the conservative ledger from `114.625` to `123.625` and leaving `276.375` under the 400-node-hour project cap.\n\n"
                "Do not pre-authorize continuation or Stage B. First audit all three results. Prepare the two chi=400 SDW/CDW controls only if an accepted pairing-bearing phase-parent result and the independent s2 result both satisfy `max|alpha| >= 1e-4`, phase by phase for any accepted orbit. That conditional first-segment step would reserve another `6.125` node-hours and still leave `270.250` for the larger bond dimensions and scaling runs the project will need."
            ),
        },
        {"id": "budget_chart_block", "type": "chart", "chartId": "budget_scenarios_chart"},
        {
            "id": "budget_note",
            "type": "markdown",
            "sourceId": "perlmutter_budget",
            "body": (
                "### Ledger interpretation\n\n"
                "These are conservative requested upper bounds, not elapsed charges, and early completion does not return project allowance. CPU and GPU reservations are summed for project control even though NERSC accounts them in separate pools. Live Perlmutter measurements and accounting remain authoritative."
            ),
        },
        {
            "id": "questions_heading",
            "type": "markdown",
            "sourceId": "report_synthesis",
            "body": (
                "## Questions the next evidence must answer\n\n"
                "1. Does either v3 orbit phase close into an accepted period-two orbit at `chi=400`, or do both flow to the same period-one fixed point?\n"
                "2. Does the independent pairing seed retain `max|alpha| >= 1e-4`, making the pairing-bearing basin reproducible?\n"
                "3. If Stage B is triggered, do matched SDW/CDW controls preserve the same-geometry energy ordering at `chi=400`?\n"
                "4. Which accepted survivors merit `chi=800+`, length scaling, connected structure factors, pair correlations, sector gaps, and entanglement analysis?\n"
                "5. After numerical controls are stable, does the frustrated transition remain between `t0=1.0` and `1.2` under bidirectional continuation and fine spacing?"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Technical progress report joining controlled Phase 1 audits to the reviewed legacy ladder MPS+MF grid.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets_to_write,
            "accessIssues": [],
        },
        "sources": sources,
        "version": 1,
    }
    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"artifact={REPORT_DIR / 'artifact.json'}")
    print(f"datasets={len(datasets_to_write)}")
    print(f"blocks={len(blocks)}")
    print(f"charts={len(charts)}")
    print(f"tables={len(tables)}")


if __name__ == "__main__":
    main()
