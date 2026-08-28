# Phase 1 progress report source notes

## Reporting job

- Question: What has Phase 1 established so far, and how does it supplement the
  earlier legacy-grid evidence?
- Audience: technical research collaborators.
- Evidence cutoff: 2026-08-26, after the chi=400 Stage A smoke completed and
  before any Stage A scientific branch result was available in the synced data.
- Primary comparison basis: accepted finite-system Phase 1 states within one
  geometry and one matching-fingerprint campaign; legacy observables remain a
  separate historical evidence layer.
- Success criterion: a reader can distinguish robust finite-system results,
  diagnostic candidates, historical hypotheses, and unverified boundaries
  without inferring a thermodynamic phase.

## Required technical-report structure mapping

1. Title: `Phase 1 Progress and the Legacy Ladder MPS+MF Grid`.
2. Technical summary: `Technical summary`.
3. Key findings with visual evidence: campaign outcomes, authorized rankings,
   unfrustrated seed sensitivity, legacy order map, and legacy pair decay.
4. Scope, data, and definitions: `How to read the evidence` and
   `Scope, data, and methodology`.
5. Methodology: the latter section plus the canonical source metadata.
6. Limitations and robustness: verification-boundary table and the explicit
   no-phase-claim section.
7. Recommended next step: the three-branch chi=400 Stage A calculation and
   staged ledger figure.
8. Further questions: the final five-question section.

No required role was omitted. Definitions are moved ahead of detailed findings
so acceptance, lineage, recurrence, and energy offsets are not used before they
are explained.

## Evidence inventory

- `output/phase1_gpu/20260823_phase1_gpu_v2/audit-report-20260826/`:
  freshly reproduced compact-state audit; zero accepted states.
- `output/phase1_gpu/20260824_phase1_gpu_v3_float64_history/`:
  verified 50-artifact compact mirror and saved nine-state audit.
- `output/phase1_gpu/RUN_ID/audit-local-20260825/`:
  saved independent-campaign audit and canonical-comparator review; its nine
  branch compact manifests had already passed verification.
- `analysis/fourier_max_grid_2026-08-14/artifact.json` at the repository root:
  reviewed legacy datasets. It is read-only input; this report modifies only
  files inside `ladder_mps_mft/`.
- User-supplied Perlmutter `phase1_gpu.sh status` output after smoke job
  `57620629`: 114.625 node-hours reserved and 285.375 unreserved. This supersedes
  the locally synced ledger ending at 114.500 for current accounting.

## Chart map

| Section | Analytical question | Family / type | Fields | Supported takeaway | Palette policy |
|---|---|---|---|---|---|
| Phase 1 outcomes | Which scheduler-complete branches became accepted states? | Composition / stacked bar | campaign-geometry; accepted, raw candidate, mixer candidate, excluded | Float64 plus repaired gates changes v3 from screening to eight accepted states | Relaxed four-state categorical palette; labels and stacking provide non-color distinction |
| Unfrustrated sensitivity | Does the pairing-bearing basin survive lineage changes? | Comparison / categorical bar | lineage; log10 max pairing field | Pairing magnitude is highly seed-sensitive; the chi=400 two-lineage test is necessary | Single-root bars; exact lineage labels and classification remain in the data table |
| Legacy order map | Where did the completed frustrated grid switch dominant channel? | Matrix / heatmap | t0, V, log10(d-wave/SDW) | Legacy data motivate the 1.0--1.2 transition bracket | Diverging palette centered at zero |
| Legacy pair decay | Which representative correlations plateau or decay? | Trend / multi-series line | distance; three saved cases | Frustrated example plateaus; square is enhanced over cubic unfrustrated but decays | Three approved categorical roots plus direct legend; line separation remains visible in shape |
| Node-hour envelope | What allowance remains after each gate? | Composition / stacked bar | scenario; reserved, unreserved | Conditional staging preserves at least 270.250 node-hours after first segments | Hard two-root cap plus exact values |

No trend uses fewer than 24 observed distances. The two categorical charts have
six or nine meaningful categories. The three-row budget chart is retained
because each discrete gate is decision-relevant and the constant 400-node-hour
denominator is the analytical point.

## Explicit interpretation exclusions

- No energy is compared across transverse geometries.
- V3 and the independent `RUN_ID` campaign are not energy-ranked together
  because their implementation fingerprints differ.
- Candidates and stagnated states never enter the ranking table.
- Seed names remain lineage labels, not phase labels.
- Legacy and refactored energies are not put on one scale.
- Compact verification is not represented as full-scratch verification.
- Acceptance is not represented as length, bond-dimension, or thermodynamic
  convergence.
- The chi=400 Stage A smoke is a runtime validation, not a scientific result.
