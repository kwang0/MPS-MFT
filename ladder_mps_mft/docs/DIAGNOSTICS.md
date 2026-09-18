# Ladder diagnostics

## Terminal measurements and September 18 backfill

Future SCF templates now enable both `quick_diagnostics=true` (the historical
master switch) and `full_pair_correlations=true`. Square, cubic and trellis
drivers measure accepted endpoints **and `maximum_iterations` endpoints**.
Measurements do not change the saved `accepted`, status, solution kind or
period. Each diagnostic records the original state SHA-256, MPS location,
iteration, model fingerprint, measurement implementation hash, completion
marker and measurement time. Unaccepted results are `terminal_snapshot`s.
Accepted temporal cycles retain separate phase files; trellis A/B retain
separate **spatial** ladder files. There is no temporal or spatial averaging.

The default equal-time dataset contains:

- Full raw and connected density-density, longitudinal spin-spin and
  density-spin matrices; transverse `<S+_i S-_j>` and its connected matrix.
- Connected charge/spin structure factors on both transverse momentum sectors.
- Spin-resolved single-particle Green matrices, anomalous `<Cup_i Cdn_j>`,
  density, magnetization, transverse-spin expectation and double occupancy.
- Full singlet pair addition/removal Gram matrices, including **cross-channel
  entries** between onsite, rung and nearest-neighbor leg pairs. Both raw and
  connected matrices are saved with complex values, operator expectations,
  site coordinates and class labels. Historical class-specific datasets remain.
- Von Neumann/second-Renyi entropy on every MPS bond and the existing finite-size
  central-charge and small-q charge-slope estimates.

For pair expectations `f_a=<Delta_a>`, the connected matrices are
`removal_connected[a,b]=<Delta_a^dagger Delta_b>-conj(f_a)*f_b` and
`addition_connected[a,b]=<Delta_a Delta_b^dagger>-f_a*conj(f_b)`.
The stored `basis_site1`, `basis_site2`, and `basis_class` identify every row
and column (Julia's one-based rung-major site order). Onsite operators are
`Cup*Cdn`, without the extra factor of two of a coincident-site bond singlet.
These cross-channel data permit a chosen d-wave form-factor projection later,
without discarding sign information or imposing uniformity during measurement.
Schema 2 also saves the raw charge/spin matrices that the old writer computed
but did not persist. Connected subtraction separates anomalous coherence from
pair fluctuations; neither alone establishes a thermodynamic phase.

The measurements use normalized host MPSs, even for GPU-produced states.
They cost additional terminal wall time; the immutable state is saved first.
An allocation timeout can interrupt measurement, in which case the saved full
MPS can be postprocessed. Deadline/nonfinite stops do not automatically start
an expensive measurement pass. The explicit offline option allows finite
`time_limit`/`stagnated` endpoints as well as `maximum_iterations`.
Calibration fixtures and archived prepared configs retain their explicit
diagnostic switches; use the updated scientific templates for new preparations.
The four deferred finite-size configs must be regenerated from the updated
base template before submission. Already queued source checkouts are unchanged.

### User-run Perlmutter backfill

After the final trellis run and its ordinary compact export finish, run on
**Perlmutter** (Codex does not run these commands):

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
git pull --ff-only
module load julia
bash slurm/phase1_gpu.sh reconcile
bash slurm/measure_latest_campaigns.sh plan
bash slurm/measure_latest_campaigns.sh submit
```

The default run ID is `20260918_latest_correlations`. The plan is read-only;
`submit` repeats its gates and prepares an immutable manifest and source copy.
It requires all 56 latest terminal branch states and their full MPS paths:

| Set | Endpoints | MPS measurements |
|---|---:|---:|
| Square 3x3, two seeds, including latest (1.4,0) continuations | 18 | 18 |
| Cubic unfrustrated 3x3, two seeds | 18 | 18 |
| Square finer cuts, two seeds | 12 | 12 |
| Square (1.2,+0.2), four seeds | 4 | 4 |
| Trellis one-/two-ladder cells, two seeds | 4 | 6 |
| Total | 56 | 58 |

The two superseded square V=0 parent endpoints are excluded. Discovery uses
the synchronized campaign manifests, selects the latest timestamped terminal
state within each branch, checks config/model provenance, and resolves the
full source recorded in each compact mirror. Full source size is checked
before submission and its SHA-256 is checked by the worker before contraction.
Missing results, mismatched provenance or missing scratch sources stop
preparation; the tool never silently substitutes a rolling checkpoint.

There are 56 CPU shared jobs, each with four Julia threads, eight Slurm logical
CPUs and 32 GiB. The default wall ceiling is two hours per MPS, four hours for
a two-ladder trellis endpoint. Using the repository's memory-aware CPU billing
convention (nine physical cores out of 128), the total requested ceiling is
**8.15625 CPU node-hours**, not an observed runtime estimate. The launcher
enforces a nine-node-hour measurement cap and the existing 400-additional-
node-hour project cap, with reservations in the shared append-only ledger.
It runs the required read-only Phase 0 CPU plan before submission. No DMRG
iteration, sector-gap solve, GPU job or acceptance-threshold change is made.

```bash
bash slurm/measure_latest_campaigns.sh status
bash slurm/measure_latest_campaigns.sh reconcile
```

`status` verifies small measurement receipts and output hashes; it is a data
completion check, not a live queue view. `reconcile` obtains actual Slurm
accounting and releases unused ceilings through the existing append-only
ledger. Repeating `submit` completes an interrupted submission and skips
already recorded jobs; it does not automatically resubmit failed jobs.

Results are compact, MPS-free files directly under
`output/phase1_diagnostics/20260918_latest_correlations/results/CAMPAIGN/LABEL/`:
`diagnostics.h5`, or `diagnostics_ladder_A.h5`/`diagnostics_ladder_B.h5`.
Sync that directory together with its manifest, receipts and logs for analysis.
Original scratch and CFS simulation artifacts are never modified.

For a selected row inside a user-provided CPU allocation, a partial measurement
can be resumed with the **saved** source and manifest:

```bash
RUN="$CFS/m4863/MPS-MFT/ladder_mps_mft/output/phase1_diagnostics/20260918_latest_correlations"
# Set INDEX to the manifest row being retried; this performs contractions.
JULIA_NUM_THREADS=4 julia --startup-file=no --project="$RUN/source" \
  "$RUN/source/scripts/measure_latest_campaigns.jl" run "$RUN/manifest.tsv" "$INDEX"
```

Completed files are reused only when source and measurement hashes match.
If preparing from a different checkout, set `DIAGNOSTICS_CAMPAIGN_ROOT` to the
original `output/phase1_gpu` and `PHASE1_BUDGET_ROOT` to its `output/project_budget`.
The default handoff above uses the original checkout and needs no overrides.
For an individual full state, `scripts/run_diagnostics.jl CONFIG STATE
--allow-unaccepted --output=NEW_DIR` uses the same complete suite by default.
`--basic` explicitly omits pair-pair matrices; `--sector-gaps` remains a separate,
opt-in set of new DMRG calculations and is never invoked by this backfill.

## Conventions and interpretation

All Fourier axes use rung momentum `qx = 2 pi m/L` and transverse momentum `ky = 0,pi`. Charge and spin structure factors use

```text
S_O(qx,ky) = (1/(2L)) sum_ij exp[i q.(r_i-r_j)]
              ( <O_i O_j> - <O_i><O_j> ).
```

The saved peak includes `qx`, `ky`, and both values divided by pi. The reference `pi*density` is stored for tracking, but a two-leg ladder can have multiple bands, so mismatch to that reference is not by itself a phase diagnostic.

`K_rho` is estimated from the first up to three nonzero `ky=0` charge modes. Because the stored structure factor is normalized by `2L`, the file reports both conventions

```text
K_rho,site = pi * dS_charge(q,0)/dq
K_rho,rung = 2 pi * dS_charge(q,0)/dq.
```

The second corresponds to first converting the total `ky=0` density structure factor to a per-rung `1/L` normalization. Match the normalization used by a comparison paper explicitly. Both finite-OBC estimators require L and chi scaling before interpretation.

The entanglement profile stores von Neumann and second-Renyi entropies for every MPS bond. The central-charge fit uses only even MPS bonds, which are cuts between complete rungs, and fits the open-boundary Calabrese-Cardy form away from the edges. Report the fit window, R-squared, L and chi; parity oscillations and gapped crossovers can make a finite-size central charge unreliable.

Sign-resolved singlet-pair matrices use the unnormalized convention `Delta_ab = c_up,a c_dn,b - c_dn,a c_up,b`. The complete short-range basis is enabled by default for SCF diagnostics. Cached transfer environments avoid an independent full-ladder MPO contraction for every matrix entry. Report the convention, signs, and spatial decay, not only a maximum Fourier component.

The separate fixed-number calculations produce:

```text
spin gap       = E(N,2Sz=2) - E(N,0)
charge gap     = [E(N+2,0)+E(N-2,0)-2E(N,0)]/2
hole binding   = E(N-2,0)+E(N,0)-2E(N-1,1)
particle bind. = E(N+2,0)+E(N,0)-2E(N+1,1).
```

Their HDF5 artifact is separate from the number-parity SCF state. Use the measured spin and charge gaps together with |E_p| when checking whether t_perp is perturbatively small. The registry lookup alone cannot certify weak coupling.

The current bundled formulas require an even target particle number and an Sz=0 reference sector. Odd-N reference sectors need an explicit spin-sector convention before extension.
