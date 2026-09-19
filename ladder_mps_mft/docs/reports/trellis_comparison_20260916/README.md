# Four trellis comparison runs

Final results update, September 19: all four terminal states are synchronized.
[Both one-ladder starts are paired; both two-ladder starts develop stripes](../trellis_progress_20260918/README.md),
with large alternating relaxation remaining in the latter. See the
[combined campaign review](../campaign_review_20260918/README.md). The preparation
contract and original checksum-repair history below are retained; they are
not instructions to resubmit these jobs.

Prepared locally September 16, 2026, at the user's request. The user reports
that other campaigns are still running. That is user-reported live status;
no scheduler query, transfer, submission or modification of those campaigns
was performed here.

| Implementation | Stripe-dominant seed | Pairing-dominant seed |
|---|---|---|
| Fixed reciprocal one-ladder map | 95% stripe + 5% pairing | 95% pairing + 5% stripe |
| Explicit rectangular two-ladder cell | Same template on A/B | Same template on A/B |

All four use t=1, U=8, t0=1, tau0=tau1=0.1, V=0, L=64 per ladder,
n=0.9375 per site on each ladder, and chi=200. They reuse the established
reference bundle SHA-256
`e01a1ea7d6be813110870d26377db0df529d1584816c946af78584e1e1fbddc1`
and random seed 1404. Each run starts a fresh MPS (two for the explicit cell),
with fields rebuilt using its own trellis kernel. These are initial fields,
not accepted solutions or inherited square MPS states.

The highest-chi exact registry row is E_p=-0.13251724 t at bare chi=1000;
Delta=0.13251724 t. No interpolation or new pair-binding job is needed.
Each individual hopping is below Delta, but the coherent two-path scale
tau0+tau1=0.2 t exceeds it. The requested effective model is therefore an
exploratory parameter comparison; this preparation does not establish that
the second-order approximation is quantitatively controlled for a material.

## Controls and interpretation

- Fixed coordinates, simultaneous raw updates, no Anderson or damping.
- At most 60 cell sweeps, at least 40 before acceptance, ten stable records.
- One-ladder sweep: one ladder solve; two-ladder sweep: two ladder solves.
  Across four jobs this is at most 360 density-targeted ladder solves
  (each may require multiple chemical-potential evaluations).
- r_range=4 with reciprocal input/output projection. All zigzag cross terms
  and the centered normal one-body terms are retained within that projection.
- Field absolute/relative tolerances 1e-7/1e-4; channel floor 5e-7;
  energy-window tolerance 1e-7 t/site; inner-DMRG tolerance 1e-7 t total.
- Density tolerance 1e-5 on each ladder separately. A/B can develop different
  profiles but have equal constrained average fillings.
- A/B are spatial labels. Only stationary cell solutions are accepted;
  temporal cycles remain unaccepted. All MPS, field and correlation histories
  are kept separately. Compare energies per physical site, with the
  half-rung coordinate offset when comparing profiles.
- Each job: one GPU, shared QOS, 32 CPU cores, 12-hour allocation ceiling,
  11.5-hour solver deadline, one segment, no automatic extension. Four jobs
  reserve at most **12 node-hours** through the existing shared ledger.
  Two-ladder sweeps cost roughly twice the one-ladder work; actual timings
  and convergence remain unmeasured on Perlmutter.

See [the exact spatial maps and energy convention](../../TRELLIS_MEAN_FIELD.md).
The single-ladder skew repetition and rectangular two-ladder cell impose
different transverse stripe arrangements and end cuts. Differences between
them need that physical interpretation; they are not a test of an arbitrary
iteration convention. Keep unaccepted endpoints as transient evidence.

## User-run Perlmutter handoff

Use a separate source directory so that the existing queued/running jobs
continue to load their original files. The local source bundle includes
the reference HDF5, solver, launchers, manifests, tests, and this handoff.
It does not depend on the working changes having been pushed to Git.

1. Transfer `output/source_bundles/trellis_comparison_20260916.zip` and its
   `.sha256` sidecar yourself to
   `$CFS/m4863/MPS-MFT/ladder_mps_mft/output/source_bundles/`.
2. Run the following on **Perlmutter**, not on local Windows:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
(
set -euo pipefail
(cd output/source_bundles && tr -d '\r' < trellis_comparison_20260916.sha256 | sha256sum -c -)
trellis_project="$CFS/m4863/MPS-MFT-trellis-20260916/ladder_mps_mft"
test ! -e "$trellis_project" || { echo "Choose a new empty source directory"; exit 1; }
mkdir -p "$trellis_project"
unzip -q output/source_bundles/trellis_comparison_20260916.zip -d "$trellis_project"
cd "$trellis_project"
module load julia
bash slurm/submit_trellis_comparison.sh
)
```

The wrapper reuses the original anchor run.env account, shared result roots,
scratch root and append-only budget ledgers. It prepares exactly four
configs, reconciles accounting, and submits under the existing budget gates.
It refuses execution in the original checkout. Default run ID:
`20260916_trellis_two_basin_comparison_60`.

September 17 checksum repair: the original Windows-written `.sha256` file
had CRLF line endings. GNU sha256sum interpreted the trailing carriage return
as part of the ZIP filename and stopped before extraction/submission. The
command above strips that carriage return before verification. The ZIP is
unchanged; no Git pull or retransmission is needed. The local checksum writer
now explicitly emits LF. Copy shell code with ordinary underscores, not `\_`.

The local [four-branch receipt](prepared_branches.csv) records the preview;
host-specific configs, seed hashes and source fingerprints are regenerated
on Perlmutter. No local preview path should be submitted directly.

## Local validation

The focused checks cover direct zigzag paths and OBC ends, the tau1=0 square
limit, reciprocal range truncation, energy derivatives including affine
normal fields, four matched seed preparations, and the complete spatial-cell
driver/storage/compact/resume path on tiny CPU ladders. Launcher tests use
local fake commands and never contact Slurm. These are implementation checks,
not scientific convergence or GPU performance measurements.

Validation completed: 150 focused algebra/preparation/stationarity assertions,
51 tiny CPU driver/storage assertions, and six local launcher checks passed.
The repository regression pass covered 784 assertions (including the 150
trellis unit assertions), with two existing Windows-specific shell skips.
Its first attempt stopped at two pre-existing unqualified fingerprint calls
in tests; those were corrected and the remaining testsets completed without
repeating the earlier passing portion. A checkpoint Boolean-storage error
found by the first tiny trellis run was fixed and the complete tiny smoke
was rerun successfully. No GPU or scientific L64 solve ran locally.
