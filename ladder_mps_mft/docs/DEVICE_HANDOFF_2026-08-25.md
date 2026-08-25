# Device handoff: ladder MPS+MF project

This file is the durable continuation record for moving the project to another
Codex desktop. Read it together with `AGENTS.md`; the operating rules there are
mandatory. The handoff was prepared on 2026-08-25 from branch
`codex/mps-mft-phase0-refactor`, with `52c4ad9` as the code baseline immediately
before these handoff documents. On the destination device, use the newest
remote branch head that contains this file.

## What GitHub transfers

GitHub transfers the refactored solver, Julia environments, launchers, tests,
plotting/audit utilities, documentation, and this conversational handoff. It
does **not** transfer numerical output:

- `ladder_mps_mft/output/**` is intentionally ignored;
- all `*.h5` files are ignored; and
- full Phase 1 MPS/checkpoint/orbit artifacts are intentionally scratch-first,
  not repository data.

This separation is deliberate. The code and context move through GitHub; only
the compact, MPS-free analysis mirrors need to move to a workstation. Full
restartable artifacts stay on Perlmutter scratch and selected irreplaceable
ones should be archived to HPSS.

## Destination-device Git setup

In the existing checkout on the other desktop:

```bash
cd /path/to/MPS-MFT
git fetch origin
git status --short
git switch codex/mps-mft-phase0-refactor
git pull --ff-only origin codex/mps-mft-phase0-refactor
git log -1 --oneline
```

Do not discard destination-side changes if `git status` is nonempty. Commit
them on their own branch or move them aside before switching. If the branch
does not yet exist locally, replace the `git switch` line with
`git switch --track -c codex/mps-mft-phase0-refactor origin/codex/mps-mft-phase0-refactor`.
Then open
`ladder_mps_mft/` as the Codex workspace and start a new task with the contents
of `docs/NEW_DEVICE_CHAT_PROMPT.md`.

## Required reading order for the new task

1. `AGENTS.md`
2. this file
3. `docs/PHASE1_V3_AUDIT.md`
4. the tail of append-only `docs/RUN_LOG.md`
5. `docs/PERLMUTTER_STORAGE.md`
6. `docs/CONVERGENCE.md`
7. `docs/VARIATIONAL_FUNCTIONAL.md`
8. `docs/PROVENANCE_AND_SELECTION.md`
9. `docs/PHASES_0_TO_4.md`

The new task should not infer submission authority from these files. Planning,
local audit, and plotting are safe; Perlmutter submission, cancellation,
migration, pruning, or deletion still require an explicit current request.

## Current implementation state

- Phase 0 is closed. CPU calibration was useful for backend equivalence and
  screening, but the tested CPU production path was roughly two orders of
  magnitude slower than the legacy GPU evidence. Production therefore uses the
  refactored dense-CUDA path, not a wholesale return to the legacy solver.
- Dense CUDA explicitly disables tensor-block `S_z`, particle-number, and
  fermion-parity QNs. The Hamiltonian symmetries are unchanged. Fixed-sector
  gap and `E_p` jobs are separate QN-conserving CPU calculations.
- Float64 is mandatory for Phase 1. v2 accidentally used Float32 through the
  old CUDA adaptor and failed all Hamiltonian-consistency gates. It remains
  screening/warm-start evidence only.
- The raw, unmixed MF map is probed before acceleration. A genuine period-two
  orbit is an allowed physical result and is stored phase by phase. Anderson is
  only a fixed-point accelerator; it does not certify, average, or erase a raw
  recurrence. A mixer-dependent recurrence triggers a fresh raw probe.
- Schema-v5 states store the exact seed plus complete applied and measured MF
  fields for every iteration. `inherit_from` remains available for legacy and
  refactored field-only starts, including stateless mirrors. It creates a fresh
  MPS/sites. `parent_checkpoint` and `resume_checkpoint` reuse a full MPS and
  therefore require the hash-linked full artifact on scratch.
- Branch energies are comparable only through the common zero-temperature
  canonical variational functional, including the MF double-counting terms,
  after acceptance and fingerprint gates. Effective-Hamiltonian eigenvalues
  alone are not comparable. Different transverse geometries are different
  Hamiltonians and are not ranked as phases of one another.
- The project has a hard cap of 400 additional node-hours relative to the
  user-reported allocation snapshot of 1,000 total and 277 used. The launcher
  ledger conservatively charges requested upper bounds and does not reclaim
  early completion. Its live Perlmutter value, not an old chat estimate, is
  authoritative before any new plan or submission.

## Numerical campaigns available

### v2: screening and parent data

`20260823_phase1_gpu_v2` completed all nine scheduler jobs, but has zero
scientifically accepted states because its Float32 MPSs fail the consistency
gates. Two interrupted-transfer parent states were repaired before the Float64
recovery; truncated auxiliary checkpoint/orbit files were later removed with
the guarded cleaner. See `docs/PHASE1_V2_AUDIT.md`.

The local stateless verifier passed 42 compact artifacts on 2026-08-25:

```text
full bytes represented: 3,512,377,436
compact bytes:          113,990,381
local tree size:        about 109 MiB
```

### v3: current Float64 representative-point result

`20260824_phase1_gpu_v3_float64_history` includes the original nine branch
jobs plus repaired reruns of frustrated CDW and unfrustrated SDW. The relevant
job IDs are recorded in `jobs.tsv`; the repair jobs are `57554463` and
`57554464`. A fresh local audit on 2026-08-25 found eight accepted fixed points,
one unresolved unfrustrated-pairing raw-map period-two candidate, and no
Hamiltonian-identity or effective-energy gate failures. The exact table and
authorized same-geometry comparisons are in `docs/PHASE1_V3_AUDIT.md`.

The local stateless verifier passed 50 compact artifacts:

```text
full bytes represented: 7,385,759,171
compact bytes:          666,136,203
local tree size:        about 635 MiB
```

These local checks validate compact hashes, sizes, provenance links, and MPS
removal. They do not validate the full files on Perlmutter scratch. Both runs
record this authoritative full root pattern:

```text
/pscratch/sd/k/kwang98/MPS-MFT/ladder_mps_mft/phase1_gpu/RUN_ID
```

### Prior exploratory legacy-grid analysis

Git also carries `analysis/fourier_max_grid_2026-08-14/`, including its HTML
report, tabular extracts, queries, and reproducer. That exploratory audit of the
legacy grid motivated the recurrence, variational-energy, scaling, and
provenance work in this refactor. It is useful historical evidence, but it is
not a substitute for the v3 acceptance gates and must not be presented as a
completed thermodynamic phase comparison.

## Transfer the lightweight analysis data

The destination Mac does not need the full MPS artifacts. Package only the two
CFS campaign-control trees and their stateless mirrors on Perlmutter, excluding
the `results` scratch symlinks and any pending full-CFS holding directories:

```bash
PROJECT=/global/cfs/cdirs/m4863/MPS-MFT/ladder_mps_mft
ARCHIVE=/global/cfs/cdirs/m4863/MPS-MFT/phase1-lightweight-20260825.tar.gz

cd "$PROJECT"
julia --project=. scripts/verify_stateless_results.jl \
  output/phase1_gpu/20260823_phase1_gpu_v2/stateless_results --full
julia --project=. scripts/verify_stateless_results.jl \
  output/phase1_gpu/20260824_phase1_gpu_v3_float64_history/stateless_results --full

tar -czf "$ARCHIVE" \
  --exclude='*/.*' \
  --exclude='output/phase1_gpu/*/results*' \
  output/phase1_gpu/20260823_phase1_gpu_v2 \
  output/phase1_gpu/20260824_phase1_gpu_v3_float64_history
sha256sum "$ARCHIVE"
```

Transfer that one archive from the **NERSC DTN** collection to the destination
Mac's Globus Connect Personal collection. Do not target
`perlmutter.nersc.gov` with `rsync`; that is not the supported large-file
transfer path. Record the printed SHA-256, verify it on the Mac, and extract
from inside `ladder_mps_mft/`:

```bash
cd /path/to/MPS-MFT/ladder_mps_mft
shasum -a 256 /path/to/phase1-lightweight-20260825.tar.gz
tar -xzf /path/to/phase1-lightweight-20260825.tar.gz
```

If the destination already has either ignored campaign directory, rename it
before extraction rather than overwriting it blindly. After extraction:

```bash
julia --project=. scripts/verify_stateless_results.jl \
  output/phase1_gpu/20260823_phase1_gpu_v2/stateless_results

julia --project=. scripts/verify_stateless_results.jl \
  output/phase1_gpu/20260824_phase1_gpu_v3_float64_history/stateless_results
```

Do not add `--full` on a Mac unless the recorded scratch paths are actually
mounted. On Perlmutter, use `--full` to verify every compact file against its
full source. A recursive Globus transfer of the whole checkout is unnecessary
and risks ambiguity around the `results` symlinks; the archive above is the
deterministic non-dotfile, lightweight payload.

## Reproduce the current analysis on the destination

Audit v3 into a new, ignored output directory:

```bash
cd /path/to/MPS-MFT/ladder_mps_mft
RUN=output/phase1_gpu/20260824_phase1_gpu_v3_float64_history
julia --project=. scripts/audit_phase1_campaign.jl "$RUN" "$RUN/audit-device"
```

Render all complete MF profiles/histories and exact seed profiles:

```bash
MPLBACKEND=Agg julia --project=. plot_phase1_mf_observables.jl \
  "$RUN" "$RUN/plots/mf_profiles"
```

This should produce 18 figures: one profile/history figure and one seed figure
for each of nine states. For interactive inspection, include the plotting file
in Julia and use `plot_phase1_mf_profiles_and_middle_histories(state)` and
`plot_phase1_seed_profiles(state)` as shown in its header.

## Recommended continuation order

1. Inspect the v3 histories and both phases of the unfrustrated pairing
   period-two candidate. Design a recurrence-focused continuation that does not
   average the two phases or let mixing hide them.
2. Run second independent seeds for the lowest accepted branch in each
   geometry; include both unfrustrated CDW and SDW because their splitting is
   only `0.000312604756` total.
3. Use accepted survivors for `L`, `chi`, discarded-weight, correlation,
   structure-factor, gap, and entanglement scaling.
4. Resolve the frustrated transition with `t0=1.00:0.025:1.20` at
   `V=-0.4,-0.2,0`, independent seeds, and forward/reverse continuation.
5. Extend the square scan only after the representative-point histories and
   recurrence are understood. Validate interpolated `E_p` with selected exact
   fixed-sector calculations rather than computing an unnecessarily dense
   expensive grid.

Before allocating anything, run the relevant launcher `plan`, inspect the live
budget ledger and Perlmutter status, and record the command/job IDs in
`docs/RUN_LOG.md`. The first new-device task should stop at a readiness report
unless the user explicitly asks to submit.

## Storage and provenance invariants

- Full MPS states, checkpoints, orbit files, and `psi_N_*` sectors live on
  scratch. Stateless CFS/local copies are analysis-only.
- Never restart from a stateless file. Follow its recorded full path and
  SHA-256.
- Scratch is not a backup and may purge unaccessed files after eight weeks.
  Archive irreplaceable restart states to HPSS.
- Never overwrite or silently repair immutable `state.h5`. Stage transfers,
  verify SHA-256, and retain or explicitly identify a corrupt source before
  installation.
- Do not delete, prune, submit, cancel, or migrate merely because this handoff
  describes the command. Those remain explicit operator actions.
