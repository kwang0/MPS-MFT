---
title: "Task 1: what stabilizes the stripe, and what attraction changes"
subtitle: "A minimal frozen-state mechanism study for MPS+MFT"
date: "September 26, 2026"
---

## 1. Decision and intended contribution

Proceed with a **two-state, measurement-only analysis** at the existing square-array $V=0$ point. Separate (i) direct intraladder density-interaction preference from (ii) the change in transverse energy caused by the prescribed $E_p(V)$ denominator. Use existing registry values at $V/t=-0.2,-0.4$ only to evaluate the physical parameter path. No new self-consistent optimization is required for the primary result.

The strongest attainable statement is:

> For two specified competing $V=0$ trial states, quantify whether attraction directly favors pairing, weakens the transverse stabilization of the stripe, or produces competing frozen effects.

This adds a mechanism to the existing order-selection results. It is not a phase-boundary calculation, a claim of global stability, or a proof that either trial remains stationary at negative $V$.

**Deliverable:** one two-panel mechanism figure, a compact numerical table, reproducible provenance and a short interpretation. The direct bond-density coefficient is the main missing measurement. The transverse reconstruction should mostly reuse saved two-point correlations and existing energy routines.

### Changes to the original Task 1

| Original proposal | Revised minimal plan |
|---|---|
| Treat two negative-$V$ evaluations as the primary framing | First extract independent $V$ and $g$ sensitivities from the two $V=0$ states; then map them onto the existing registry path. |
| Use stored transverse energy divided by $g$ | Rebuild interaction fields from each trial's own correlations, then evaluate the same square functional. Quantify the difference from stored applied-field energies. |
| One pass over two full MPSs if needed | First inspect compact correlations/sidecars. If missing, measure only the 190 nearest-neighbor density bonds per state. |
| Cancellation implies reorganization is important | Cancellation establishes competing frozen effects. Reorganization needs a separate, reliable relaxed-state comparison. |
| Estimate 0.5-1 day | Budget 4-8 hours of active work once inputs are available; measurement runtime and user-managed transfer/queue time are separate. |
| Optional broad contextual comparisons | Keep Task 2, new scans and new convergence campaigns outside this work. Existing results may provide labeled context. |

The source proposal remains unchanged. This plan refines Task 1 only. Repository evidence was reviewed at commit `7da085b`; this is a local evidence boundary, not a live remote or scheduler check.

\newpage

## 2. Why V=0 can be enough

**The user's intuition is correct for frozen-state mechanism separation.** A state at $V=0$ has a measurable density-density expectation even though its contribution to the energy is multiplied by zero. The old energy table alone omits this coefficient; the state does not.

Let $s$ and $p$ denote the saved stripe and paired trials. Use $\Delta X=X_s-X_p$, energies per physical site, and $g=t_\perp^2/|E_p|$. For a fixed trial $b$, write

\[
e_b(V,g)=A_b+V B_b+g Q_b.
\]

Here $A_b$ is its $V=0$ bare-ladder energy per site, $B_b$ its intraladder nearest-neighbor density coefficient, and $Q_b$ its transverse quadratic contraction in the specified square ansatz. Geometry, range, density convention and the state itself are held fixed. Branch-independent offsets cancel in differences.

The two independent frozen sensitivities are already available at $V=0$:

\[
\left.\frac{\partial\Delta e}{\partial V}\right|_{g,\psi}=\Delta B,
\qquad
\left.\frac{\partial\Delta e}{\partial g}\right|_{V,\psi}=\Delta Q.
\]

These answer which trial gains from direct attraction and which gains from stronger transverse coupling. **No negative-$V$ wavefunction is needed.** Moreover, linearity makes the finite frozen change exact within this functional, not just a first-order approximation in $V$ or $g$:

\[
\Delta e_{\rm frozen}(V_1,g_1)
=\Delta e_0+V_1\Delta B+(g_1-g_0)\Delta Q.
\]

There are three different questions:

1. **What stabilizes the two trials at $V=0$?** The baseline bare/transverse decomposition answers this.
2. **How would direct attraction and transverse rescaling separately change their relative energies?** $\Delta B$ and $\Delta Q$ measured on those same trials answer this.
3. **What actually happens when the system relaxes at negative $V$?** Frozen trials alone cannot answer this; the wavefunctions, textures and even existence of distinct branches can change.

To put the second question on the project's physical path, we need $g(V_1)-g_0$. One value $E_p(0)$ cannot supply that change. Existing tabulated $E_p(V_1)$ does supply it, without new optimization. Do not infer a smooth derivative from sparse registry entries. A hypothetical fixed-$g$ attraction change remains a valid diagnostic even though it is not the default fixed-$t_\perp$ campaign path.

The old $V=0$ energy table shows a transverse stripe advantage. It does **not** contain $\Delta B$, because the direct $V B$ energy is zero there. This is the precise sense in which the table is insufficient, while the two $V=0$ states are sufficient for our primary task. My earlier assessment should be read with this distinction.

Even on a smooth stationary branch, where variational stationarity can remove state-response terms from a first derivative at fixed density, finite changes still need evolving coefficients. These endpoints are not formally accepted stationary solutions, so this plan does not invoke that shortcut.

\newpage

## 3. Inputs, existing evidence and first gate

Use the two endpoints in `docs/reports/chi400_comparison_20260905/endpoints.csv`, relative to `ladder_mps_mft/`. The physical controls are square geometry, $L=64$, $U/t=8$, $t_0/t=1.4$, $n=15/16$, $V=0$, $t_\perp/t=0.1$, and $\chi=400$. Verify the stored longitudinal interaction range and every relevant fingerprint; do not assume them from this prose.

- **Paired trial:** lineage `pairing`, source job 57905744; stored status `stagnated`, accepted false.
- **Stripe trial:** lineage `legacy`, source job 57905745; stored status `time_limit`, accepted false.
- Resolve exact paths and full-artifact SHA-256 values from the endpoint manifest. Do not select a different state by a filename or texture resemblance. Later $\chi=200$ campaign endpoints are not substitutes for these two trials.

The existing endpoint CSV records matching model, numerical, implementation and registry fingerprints. Verify these against available source metadata before analysis. Both states reach bond dimension 400; neither carries a certified truncation-error bound. The paired trial's residual density differs from target by about $3.58\times10^{-6}$ per site.

### Quantitative motivation, not a new result

The archived applied-field energy table gives stripe-minus-pair differences of $+0.0034783$ in bare energy and $-0.0037276$ in total transverse energy, yielding $-0.0002493\,t$/site. The density channel's spin component supplies the largest transverse stripe advantage. These numbers motivate recomputation; they are not yet the consistent frozen-trial baseline.

At $t_\perp/t=0.1$, existing registry entries give:

| $V/t$ | Signed $E_p/t$ | $g/t$ |
|---:|---:|---:|
| 0 | -0.14653773091916378 | 0.06824181 |
| -0.2 | -0.2068002629740704 | 0.04835584 |
| -0.4 | -0.24962435880865996 | 0.04006019 |

Thus attraction decreases $g$ on this path. Rescaling the archived transverse difference alone gives positive shifts of roughly $0.00109$ and $0.00154\,t$/site, toward the paired trial. The direct $V\Delta B$ term is unknown. No crossing or mechanism conclusion follows until it and the consistent baseline are available.

### Gate A: inventory before coding

Allow 30-60 minutes to locate the exact two compact states, full-state identities and any matching diagnostic sidecars. This checkout currently has no `ladder_mps_mft/output/` directory. The historical report's compact mirrors excluded MPS tensors; full scratch availability has not been verified. The September 22 backfill concerns later $\chi=200$ states.

If saved correlations suffice for $Q$, compute it without loading MPSs. If a verified charge-correlation sidecar exists, extract $B$ from it. Otherwise request only the missing two saved-state measurements through the established user-managed workflow. If full states are unavailable, stop with an input manifest and a precise missing-data list. Do not replace them or regenerate them silently.

\newpage

## 4. Minimal implementation and measurements

### A. Reconstruct the energy of each specified trial

Reuse `mean_fields_from_correlations(...; threshold=0.0)` and the existing variational/channel contraction routines. Build interaction fields from the saved trial's current pair, exchange and density correlators in the same square ansatz. Preserve the actually applied fields separately for the effective-Hamiltonian identity.

The production single-ladder solver stores energy with `interaction_fields=fields`, the applied input fields. Away from a fixed point this is not generally the energy of identical copies of the outgoing trial. Use the rebuilt fields for the frozen transverse functional, retain the original bare-ladder expectation, and report the old/new difference by channel. This is an analysis-side evaluation, not a change to production energy bookkeeping or solver acceptance.

Record pair, exchange, density and total transverse energies, and, where the existing decomposition supports it, density charge/spin pieces. The latter is a useful low-cost explanation of what decreasing $g$ removes. Define $Q_b=e_{\perp,b}/g_0$ from the recomputed value.

### B. Measure only the missing bond-density information

For each state,

\[
B_b=\frac{1}{2L}\sum_{\langle a,c\rangle_{\rm lad}}
\langle n_a n_c\rangle_b
=B_b^{\rm leg}+B_b^{\rm rung}
=B_b^{\rm prof}+B_b^{\rm conn}.
\]

There are 126 leg bonds and 64 rung bonds: 190 total, divided by 128 physical sites. Compute the profile part from $\langle n_a\rangle\langle n_c\rangle$ and the connected part by subtraction. Preserve individual bond values so all sums are independently auditable. Use the interacting MPS expectation; no Wick factorization.

Prefer existing saved `charge_correlation` and density arrays. If they are absent, reuse existing contraction primitives in a small density-only measurement wrapper. Do not call the full pair-correlation diagnostic pipeline merely to obtain these bonds. No four-fermion pair matrices, new DMRG solves, or production-map changes are needed.

### C. Produce the three counterfactual evaluations

For each target $V_1/t=-0.2,-0.4$, tabulate:

\[
D_V=V_1\Delta B,\qquad D_g=(g_1-g_0)\Delta Q,\qquad
\Delta e_{\rm frozen}=\Delta e_0+D_V+D_g.
\]

Evaluate direct-only $(V_1,g_0)$, feedback-only $(0,g_1)$ and combined $(V_1,g_1)$. Positive changes in $\Delta e$ favor the paired trial relative to the stripe. In particular, $\Delta B>0$ means negative $V$ directly favors the stripe; $\Delta B<0$ means it favors pairing. “Favors” describes the direction of the change, not necessarily a reversal of the baseline ordering.

### Coding ceiling

One small Julia analysis/measurement wrapper, one lightweight plotting/export script if needed, and focused tests. Prefer adapting existing loaders, `Variational.jl`, `Diagnostics.jl`, `EpRegistry.jl` and the retrospective measurement workflow. Proposed code belongs under `ladder_mps_mft/scripts/`; analysis outputs under `ladder_mps_mft/docs/reports/task1_frozen_mechanism/`. Names are proposed, not existing commands. If substantial library redesign or a new production campaign becomes necessary, stop and report why.

\newpage

## 5. Validation and acceptance criteria

### Implementation correctness: required before interpretation

1. **Identity and compatibility.** Verify source hashes, lineages, stored statuses, model/geometry/range and numerical fingerprints. Record source and analysis code versions separately. Confirm the exact campaign-compatible registry selections and bound-state sign convention.
2. **Bond accounting.** Assert 126 unique leg and 64 unique rung bonds with the project's site mapping. Check total = leg + rung = profile + connected. On an occupation product state, reproduce analytic density products and zero connected density correlations.
3. **Independent energy check.** If introducing a bond diagnostic, compare the summed bond observable against an independently constructed $\sum n_a n_c$ MPO on a small test MPS. If the full source states are measured, also check the same frozen state's direct bare energy difference at one nonzero $V$ against $2L\,V B$; this is a contraction, not optimization.
4. **Transverse scaling.** Rebuild unthresholded fields at two distinct positive $g$ values from the same correlators. Verify every channel scales with $g$ and $Q$ is unchanged. Suggested algebraic target: $10^{-10}$ relative or $10^{-12}\,t$/site absolute, whichever is looser. Test failures must be explained, not hidden by changing the target after inspection.
5. **Energy conventions.** Reproduce archived applied-field bookkeeping before replacing only the interaction fields in the analysis. Check reconstructed/direct consistency against the original tolerances and preserve existing failures. The new trial evaluation must not be presented as a new Hamiltonian-identity certification of the old run.
6. **Units and density.** Use $1/(2L)$ everywhere. Keep exact frozen canonical values at each state's actual particle expectation as primary. Show the original target-density correction as a separate sensitivity diagnostic (about $5.93\times10^{-6}\,t$/site in the archived gap); do not carry an old chemical potential into negative-$V$ evaluations as an exact fixed-density correction.

### Distinguish calculation acceptance from physical claims

**Accept the analysis as completed** when the two trials, $B$, $Q$, all channel sums, registry path, tests and provenance are reproducible. The states need not become accepted fixed points to define these trial expectations. Their original acceptance flags remain unchanged, and the accepted-solution branch ranker is not bypassed.

**Allow a robust frozen-trial mechanism statement** when signs survive numerical tolerance checks and any available nearby-checkpoint recomputation. Report applied-versus-rebuilt differences, late energy drift, density sensitivity and known $\chi$ limitations separately; they are not independent statistical errors and must not be combined into an invented confidence interval.

As a screening rule, require a claimed contribution to exceed three times its available empirical sensitivity scale and retain its sign under the tested variants. This factor is a planning heuristic, not a certified error bound. If nearby-checkpoint $B,Q$ are unavailable, limit the claim to the exact saved trials and explicitly leave state-selection sensitivity untested. Small Hamiltonian-identity errors do not bound MPS optimization/truncation error.

**Allow a frozen ordering reversal statement** only when the consistent $\Delta e_0+D_V+D_g$ changes sign robustly under the same checks. An individual positive contribution is insufficient. **Do not claim a relaxed transition or thermodynamic mechanism** from this task alone.

If contributions cancel, report the cancellation and its sensitivity. If all effects are unresolved, retain a concise null-result table and stop; do not automatically launch more states, higher $\chi$, or a $V$ scan.

\newpage

## 6. Output, effort and optional follow-through

### One figure with two complementary panels

**Panel A: the $V=0$ balance.** Show stripe-minus-pair bare energy and transverse channels, plus the recomputed total. This answers the user's baseline question directly. Distinguish the new frozen-trial evaluation from archived applied-field values in the accompanying table.

**Panel B: the change under attraction.** At $V/t=0,-0.2,-0.4$, show direct $D_V$, feedback $D_g$, their sum and the resulting $\Delta e_{\rm frozen}$ relative to zero. Use discrete markers or grouped bars; connecting lines are guides, not evidence for a smooth $E_p(V)$ interpolation. Add a leg/rung or profile/connected inset only if it changes the interpretation.

Provide `sources.json` with hashes, statuses and registry records. Export the measurements to `bond_density.csv`, the energies to `trial_energies.csv`, and the counterfactuals to `frozen_changes.csv`. Include a validation receipt, one PDF/PNG figure and a brief report. Each table must identify its source trials and normalization. Include one verified reproduction command after implementation. Append executed work to `docs/RUN_LOG.md`; update current project state when the analysis changes the evidence. Durable methods need revision only if an actual method changes.

### Effort budget after inputs are available

| Work | Active effort | Decision |
|---|---:|---|
| Resolve exact artifacts and saved observables | 0.5-1 hour | Stop early if inputs are missing. |
| Rebuild transverse energies and registry path | 1-2 hours | Reuse existing map/energy routines. |
| Extract $B$, or prepare a density-only wrapper | 1-2 hours | Measure only the missing bonds. |
| Focused checks, figure and interpretation | 1.5-3 hours | Stop at one figure and evidence table. |
| **Total** | **4-8 hours** | **Zero new optimized states.** |

Saved-array analysis should be inexpensive. Full-MPS contraction time and memory must be estimated from the actual state and environment; no runtime promise is justified yet. User-managed transfer, queue and scheduler time is additional. Do not start a local operation expected to exceed five minutes without reporting its expected cost. All Perlmutter transfers and actions remain user-operated.

### Optional evidence, not a prerequisite

When existing $t_\perp$ scan results arrive, compare their qualitative behavior with the sign of $\Delta Q$: varying $t_\perp$ at fixed $V$ changes $g$ without the direct $V B$ term. These are complementary relaxed-state observations, not an exact fixed-$g$ or matched-state causal control. No new scan is part of this plan.

A relaxation residual requires both appropriately tracked competing branches at the target point, evaluated under the same functional with comparable controls and adequate convergence. The accepted negative-$V$ paired A/B states do not by themselves supply a matched relaxed stripe-minus-pair gap for the $\chi=400$, one-ladder trials. Omit this residual unless those requirements are already met. Absence of a surviving competing branch is not a numerical value for the missing energy.

### Local evidence used

Relative to `ladder_mps_mft/`: the September 5 chi400 report and endpoint manifest; the Julia mean-field, geometry, energy, solver, storage, diagnostics and registry modules; `data/E_p_values.csv`; the variational-method, project-state and active-plan documents; and the September 22 pair-correlation and square $t_\perp$ reports. These identify the source states, existing routines and complementary evidence.

This is an execution plan, not a report of new contractions, newly validated endpoints or completed mechanism results.
