---
title: "Task 1 revised: direct attraction versus transverse feedback"
subtitle: "A postprocessing study of the measured square-array boundary snapshots"
date: "Revision 2 - September 26, 2026"
---

## 1. Decision: change the anchor, preserve the mechanism question

This revision supersedes the earlier $V=0$-pair plan. No new mechanism calculation has yet been executed.

**Use the measured stripe-like and paired-like snapshots at $t_0/t=1.4$, $V_*/t=-0.05$, $\chi=200$. Do not assume two surviving paired/striped solutions at $V=0$, and do not make recovery of historical $\chi=400$ MPSs a prerequisite.**

The revised question is: for two distinct trial textures actually observed at the same near-boundary Hamiltonian, does making $V$ more attractive favor pairing through direct intraladder bond-density energetics, through the reduction of the transverse scale $g=t_\perp^2/|E_p|$, or through both? This is a fixed-trial sensitivity analysis, not a proof of two stable phases or of the relaxed transition mechanism.

The September 22 retrospective campaign already measured the required density-correlation matrices for these two snapshots. Compact states contain the pair, exchange and density arrays needed for transverse energy reconstruction. **The intended computation is array postprocessing: zero new optimized states, zero new MPS contractions, and no new production campaign.** The required compact files are not present in this Mac checkout; obtaining those existing files is the explicit first dependency.

### What changed from revision 1

- Replace the historical $V=0$, $\chi=400$ pair by `backfill_37` and `backfill_38`, both at $V_*=-0.05$, $\chi=200$ and the same 60-step controls.
- Extract bond-density coefficients from already completed diagnostics instead of preparing fresh measurements.
- Preserve the actual interpolated $E_p$ used by the fine-cut campaign. The older exact-entry-only premise cannot describe these source states.
- Describe the result as a decomposition for two measured finite-state snapshots. Source convergence, density mismatch and end-localized spin qualify the physical interpretation.
- Keep future collaborator scans as independent contextual evidence. Do not duplicate their seeding, convergence, correlation or geometry work.

### Latest pushed evidence checked

On September 26, a live `git ls-remote --heads origin` check found the collaborator's working branch `codex/mps-mft-phase0-refactor` still at **7da085b**, matching this checkout. Other advertised branch tips also matched cached refs; our separate plan branch is not new collaborator evidence. No newer pushed scan results were found.

Recent work includes the September 24 material review, September 23 stripe/pairing correlation grids and density-constraint discussion, September 22 completed correlations and accepted square A/B controls, and preparation of the square $t_\perp$ and trellis campaigns. Project records call the latter ongoing based on user reports; this check does not establish live scheduler state or completion.

\newpage

## 2. What has been done and the exact usable pair

The September 16 coarse-grid update shows both $V=0,t_0=1.4$ seed lineages becoming striped. Their final physical leg-pair RMS values are $1.04\times10^{-9}$ and $4.42\times10^{-8}$. A seed name is not an endpoint phase label. The old September 5 report does record distinct $\chi=400$ snapshots, but full-state availability has not been established locally.

The newer fine cut supplies the appropriate same-Hamiltonian contrast:

| Quantity | Stripe-like trial $s$ | Paired-like trial $p$ |
|---|---:|---:|
| Retrospective diagnostic ID | backfill_37 | backfill_38 |
| Source job | 58387972 | 58387973 |
| Raw evaluations | 60 | 60 |
| Physical leg-pair RMS | $3.67\times10^{-5}$ | $0.032991$ |
| Physical spin RMS | $0.144695$ | $0.005606$ |
| Stored status | maximum_iterations | maximum_iterations |
| Accepted | false | false |

Both have square geometry, $L=64$, $U/t=8$, $t_0/t=1.4$, $V_*/t=-0.05$, $t_\perp/t=0.1$, target $n=15/16$, $\chi=200$ and raw-map evolution. Validate stored range, implementation and numerical fingerprints from the source files. The paired snapshot has 96.7% of its full-chain spin-squared weight in the outer 14 rungs at each end; it is not an established bulk coexistence state.

Archived applied-field values give $\Delta e=e_s-e_p=+4.9757\times10^{-5}\,t$/site canonically, or $+5.9520\times10^{-5}$ after the stored target-density correction. The difference, about $9.76\times10^{-6}$, matters at this scale. Neither number is yet our rebuilt frozen-trial baseline or an accepted phase-energy ranking.

### Source map: no guessed filenames or new measurements

All paths here are relative to `ladder_mps_mft/`.

1. Read `coverage.csv` in the September 22 correlation report directory, rows with `parent_index` 37 and 38. These specify campaign, labels, compact/full source hashes and config hashes.
2. Resolve exact compact-state paths from the $t_0=1.4,V=-0.05$ rows of `variational_energy_cuts.csv` in the September 18 fine-cut report directory. Its `source_sha256` is the **compact** hash, not the full-MPS hash.
3. Resolve exact diagnostic paths and hashes by IDs `backfill_37/38` in `source_validation.json` in the September 22 correlation report directory.
4. Both diagnostics are in retry1, campaign `20260915_square_two_basin_fine_cuts_95_5_60`. Retrieve their adjacent measurement receipts, relevant manifest rows and original run configs. Use manifest/config metadata to resolve config paths.

The minimal transfer is two compact `state.h5` files, two `diagnostics.h5` files and their small provenance/config records. Full MPS tensors are unnecessary. Git versions the reports, not these output files. The user performs any transfer from Perlmutter; Codex does not connect there.

`data/two_basin_references.h5` is only a seeding template: its pair reference is $(1.4,-0.4)$ and stripe reference $(1.0,0)$, and it contains no MPS. It is not an analysis substitute. Also, `data/README.md`'s older “never interpolates” statement is superseded for fine cuts by the explicit preparer and campaign records.

\newpage

## 3. Mechanism definition and why V=0 is not required

For each fixed trial, all parameters except $V$ and $g$ held fixed, write

\[
e_b(V,g)=A_b+V B_b+g Q_b,\qquad \Delta X=X_s-X_p.
\]

Measure $B_b$ from density correlations and $Q_b$ from the unthresholded square transverse map. At the actual source point $(V_*,g_*)$, the exact frozen change is

\[
\Delta e_{\rm frozen}(V,g)=\Delta e_*+(V-V_*)\Delta B+(g-g_*)\Delta Q.
\]

This is exact for these fixed trials within the implemented functional; it is not an expansion assuming small parameter changes. It is not exact for reoptimized states. Branch-independent offsets cancel in the difference.

### Direct intraladder coefficient

\[
B_b=\frac{1}{128}\sum_{\langle a,c\rangle_{\rm lad}}\langle n_a n_c\rangle_b.
\]

Use 126 unique leg bonds plus 64 rung bonds. Save leg/rung sums, the profile product contribution $\langle n_a\rangle\langle n_c\rangle$, and the connected remainder. Reuse the full-chain energy normalization. A bulk-only diagnostic may supplement the result, but must never replace the full-chain $B$ in the energy identity. No Wick factorization.

### Transverse coefficient and consistent baseline

Rebuild interaction fields from each trial's current correlations with `mean_fields_from_correlations(...; threshold=0.0)`. Reuse the existing variational/channel routines to compute $e_{\perp,b}$ and $Q_b=e_{\perp,b}/g_*$. Keep pair, exchange, density and, if already supported, density charge/spin pieces.

The single-ladder solver's stored energy uses applied input fields, which differ from rebuilt fields away from self-consistency. Preserve applied fields for the effective-Hamiltonian identity; replace only interaction fields in this analysis-side evaluation. Set $e_{*,b}=e_{{\rm bare},b}^{\rm stored}+e_{\perp,b}^{\rm rebuilt}$. The stored bare term already contains $V_*B_b$; do not add it twice. If needed, $A_b=e_{{\rm bare},b}^{\rm stored}-V_*B_b$.

### Two independent effects and the actual campaign path

$\Delta B$ and $\Delta Q$ are the primary outputs. For $\delta V<0$, $\Delta B>0$ means direct attraction favors the stripe trial, while $\Delta B<0$ favors the paired trial. A decrease in $g$ favors pairing if $\Delta Q<0$. A favorable shift need not reverse the total gap.

For the minimum figure, evaluate only $V/t=0,-0.05,-0.2$, using the archived source $g_*$ and exact registry endpoint scales. Show direct-only $(V,g_*)$, feedback-only $(V_*,g(V))$, and combined changes. No new fine scan or smooth derivative is needed.

The user's $V=0$ intuition remains mathematically correct: two distinct $V=0$ trials would suffice to measure both sensitivities, even though $VB$ vanishes there. The problem was our unverified choice of such a pair. **Any common source point works; the measured split at $V_*=-0.05$ is better supported.** Existing $E_p(V)$ values supply the change in $g$; no target-$V$ wavefunction is required for frozen comparisons.

\newpage

## 4. Implementation sequence and data contract

### Step A: build an input inventory before numerical code

Resolve the two IDs through the versioned report tables. Produce a manifest containing exact compact/diagnostic/config paths and hashes, full-source identity, source status, iteration, geometry, model/numerical fingerprints and code versions. Report which files exist locally. Missing inputs produce a precise transfer list, not an attempt to rerun the measurement campaign.

The current local folder lacks `ladder_mps_mft/output/`. Accordingly, this plan is implementable from already measured data, but the numerical execution is **blocked on obtaining that existing compact bundle**. Do not describe it as immediately runnable here or silently substitute legacy HDF5 files.

### Step B: reuse array loaders and the physics implementation

Adapt the source/receipt checks in `scripts/analyze_pair_correlations_20260922.py` for only these two rows. Do not invoke its all-56-branch loader. Reuse its array-orientation handling: the Julia-written HDF5 arrays must be interpreted consistently.

Required diagnostic arrays are `density`, `charge_correlation` and `charge_connected`. Required compact-state arrays are spin-resolved densities, `pair`, `exchange_up` and `exchange_down` under `correlations`, plus applied fields, model/config metadata, bare energy and original energy components. Confirm actual group paths through the existing storage/loader code before implementation. Check consistency between the two files' densities and source hashes.

Use Julia library calls for the transverse map and energy rather than independently reimplementing the physics in Python. A small Julia wrapper can export the energy components and bond summaries; a small Python script can make the final table/figure. Existing primitives need no production change. If the archived implementation differs from current code, audit the relevant map/energy diff before reuse; record both hashes.

### Step C: retain the actual denominator convention

The fine-cut preparer pins registry SHA-256 beginning `2209bd2c` and linearly interpolates signed $E_p$ between $V=-0.2$ and 0. The source uses

\[
E_p(V_*)/t=-0.16160336393289043,\qquad g_*/t=0.06187990000104661.
\]

The target endpoint values are $g(0)/t=0.06824181005993883$ and $g(-0.2)/t=0.04835583792876438$. Verify all values against the source config and archived registry; do not replace the source denominator with a new estimate. Record interpolation weight 0.25 when measured from $V=0$ toward $-0.2$; the implementation may store the complementary weight because its bracket order is reversed.

The prior fine-cut report already gives a polynomial-guide sensitivity of +0.615% in $g_*$, not an error bar. As a cheap optional check, reevaluate fixed $Q$ at that source scale, label the alternative model, and rebuild its baseline consistently. Endpoint frozen gaps at fixed target $g$ must remain invariant to mere changes of reference parametrization; do not manufacture endpoint uncertainty by changing $g_*$ while holding the baseline fixed. True unmeasured $E_p(V_*)$ uncertainty remains unbounded by that guide.

### Proposed implementation boundary

Add at most `scripts/analyze_frozen_mechanism.jl`, a plotting wrapper and focused tests. Proposed output folder: `docs/reports/task1_frozen_mechanism/`. A future driver should accept an explicit manifest and output directory, with an inventory-only mode. These are proposed interfaces, not already existing commands. Deliver a tested reproduction command once implemented. No solver, seeding, launcher or acceptance changes.

\newpage

## 5. Validation, acceptance and stopping rules

### Required correctness gates

- **Provenance:** every compact and diagnostic hash matches its own recorded hash; diagnostic full-source identity matches the compact metadata; model, iteration and statuses agree. Preserve both false acceptance flags. Source and analysis code hashes remain distinct.
- **Geometry and sums:** assert 128 sites and exactly 190 unique bonds. Verify total = leg + rung = profile + connected. On a small occupation-product fixture, reproduce analytic products and zero connected correlations. Compare a small correlated fixture against an independently assembled bond-density MPO if any new indexing/measurement code is introduced.
- **Array consistency:** charge matrix is symmetric within numerical tolerance; its diagonal satisfies $\langle n_i^2\rangle=\langle n_i\rangle+2\langle n_{i\uparrow}n_{i\downarrow}\rangle$ where the stored double occupancy is available. Check connected subtraction and match spin-resolved compact densities to diagnostic densities.
- **Map scaling:** at fixed arrays, verify every transverse channel scales linearly at two positive $g$ values. Suggested numerical tolerance is $10^{-10}$ relative or $10^{-12}\,t$/site absolute, whichever is looser. Preserve any original identity failures rather than relaxing gates to make this task pass.
- **Energy bookkeeping:** reproduce archived applied-field channel values first, then report the rebuilt-minus-applied changes. Check $e_{\rm bare}(V_*)-V_*B=A$ algebraically and verify that separate direct/feedback evaluations sum to the combined change. No new full-state energy contraction is required.
- **Density:** show actual densities, canonical gaps and the archived target-density corrections separately. The paired correction is about $-9.75\times10^{-6}\,t$/site. An old chemical potential is not an exact correction for a new target Hamiltonian. Do not advertise exact common-density phase ranking.

### Levels of interpretation

**Calculation complete:** provenance, array checks, rebuilt energies, $B,Q$ and the figure reproduce. Nonstationary trials still have well-defined trial energies. Completion does not promote them to accepted solutions.

**Mechanism resolved for these snapshots:** a contribution has a definite sign above arithmetic/extraction tolerances. Report its magnitude and sensitivity to applied-versus-rebuilt bookkeeping, density correction and the stated denominator convention. Profile/connected and leg/rung pieces identify what carries the direct preference. The conclusion explicitly names the two saved trials.

**Robust beyond these snapshots:** requires existing, comparably measured later snapshots with matched controls and persisting contrasting textures. If such data arrive from the collaborator, repeat the same wrapper and retain old results. Do not use convergence residuals or last energy steps as certified error bars, and do not infer $B$ stability from energy stability. Absence of later measurements leaves this level untested.

**Phase or relaxed-mechanism claim:** not a deliverable. Unaccepted source states, finite $L,\chi$, restricted transverse ansatz, density mismatch and end-weighted spin prohibit a claim of equilibrium coexistence or a resolved transition. Cancellation alone establishes competing frozen effects, not that relaxation caused the observed phase change.

Stop after one figure if the mechanism is clear for the trials. If the apparent conclusion depends on bookkeeping or density sensitivity, publish only the coefficients and qualify the ambiguity. If the inputs cannot be supplied, finish the manifest and report the blocker. Do not initiate new optimization or measurement campaigns to rescue this plan.

\newpage

## 6. Deliverable, effort and collaboration strategy

### One figure and auditable tables

Panel A shows the rebuilt baseline stripe-minus-pair energy by bare and transverse channel at $V_*=-0.05$. Panel B shows $(V-V_*)\Delta B$, $(g(V)-g_*)\Delta Q$, their sum and the total frozen gap at the three specified coordinates. Use discrete markers/bars; any line is a guide, not a phase boundary. Add a small direct-term decomposition only if physically informative.

Export a source manifest, per-bond values, per-trial energy components, frozen changes and a validation receipt. A short report states the result, source limitations and stop/go decision. Append completed analysis actions to `docs/RUN_LOG.md`; update project state when actual evidence changes. Do not rewrite manuscript conclusions before numerical output exists.

### Effort after the compact bundle is available

| Work | Active effort |
|---|---:|
| Resolve and validate two-state input bundle | 0.5-1 hour |
| Extract density bonds and wrap existing transverse routines | 1-2 hours |
| Focused tests and bookkeeping checks | 1-2 hours |
| Figure, provenance and interpretation | 1-2 hours |
| **Total** | **3.5-7 hours** |

The array calculations should be inexpensive; first-run Julia loading and environment setup may dominate. No GPU allocation or DMRG is required. Transfer/queue time is not included, and no scheduler action is assumed. Report dependency problems instead of broadening scope.

### Complement the collaborator's likely next work

The collaborator's recent pattern is basin tests, selective continuations, transverse-cell controls, terminal correlations and integrated reporting. Assume that pattern continues for planning purposes; do not treat it as evidence that any particular future dataset already exists.

- Their work determines which textures persist and which cells or parameters change the outcome. This task supplies an energetic explanation using the same recorded outputs.
- The existing square $t_\perp$ scan varies $g$ at fixed $V$; use synchronized results as an independent directional check, not as a duplicate campaign. Classify states by measured texture, not seed name or MF-field magnitude alone.
- If a later continuation makes both $V_*=-0.05$ starts striped or paired, retain this analysis as a historical trial comparison. Withdraw any language suggesting persistent competing branches. Do not automatically search many points until a desired contrast appears.
- If the collaborator supplies a genuinely stationary contrasting pair, add it as a clearly versioned validation set after checking equal model/numerical controls. New data never silently replace the fixed primary manifest.
- Trellis and accepted square A/B results provide context, not a substitute for the one-ladder same-model pair. Full Task 2 correlation analysis is already substantially done and is not repeated here.

### Evidence locations for the implementer

All report folders below are under `ladder_mps_mft/docs/reports/`:

- `square_fine_cuts_20260918`: split textures and energies.
- `pair_correlations_20260922`: exact measured-state identities.
- `two_basin_grid_20260915`: the later $V=0$ collapse.
- `two_basin_fine_cuts_20260915`: interpolation sensitivity.
- `square_tp_scan_20260922`: the complementary prepared scan.

The current project-state and active-plan documents distinguish completed evidence from user-reported ongoing campaigns.

