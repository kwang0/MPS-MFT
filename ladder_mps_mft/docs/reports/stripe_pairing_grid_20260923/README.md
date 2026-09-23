# Stripe amplitude versus pair correlations

These grids show the requested descriptive associations across different
model parameters, using the saved September 22 pair-correlation measurements.
No fixed-parameter restriction is imposed.

- [Charge-modulation grid (PNG)](charge_pairing_grid.png) / [PDF](charge_pairing_grid.pdf).
- [Spin-order grid (PNG)](spin_pairing_grid.png) / [PDF](spin_pairing_grid.pdf).
- [Every plotted point and its source](points.csv).
- [Validation and diagnostic hashes](validation.json).

## How to read the grids

Rows are square, cubic unfrustrated and trellis geometries. Columns show
short-distance full, short-distance connected, long-distance full and
long-distance connected **rung-singlet** correlations. Color encodes V;
marker shape encodes t0. A larger unfilled ring marks an accepted result.
Trellis 1L and 2L labels distinguish the one- and two-ladder spatial cells.

Each mark is one spatial MPS: 42 square, 18 cubic and 6 trellis, from 60 source
branches. The 58 retrospective MPSs remain unaccepted terminal snapshots;
the eight square A/B MPSs belong to four accepted fixed-point runs. Seeds
and spatial A/B ladders remain separate, even when marks overlap. They are
not independent statistical replicates. No points are jittered, averaged
across seeds or filtered according to whether they are paired.

All plotted states have L=64 rungs, chi=200, U=8 and target density 15/16.
The older isolated chi=1200 reference is excluded. Prospective t_perp/V=-1
campaigns without measurements in this source set are not represented.

## Definitions

Both stripe-amplitude proxies use rungs 9–56, matching the pair-correlation
bulk window:

```text
n_rung(i) = [<n(i,0)> + <n(i,1)>] / 2
charge RMS = sqrt(mean_i [n_rung(i) - mean_bulk(n_rung)]^2)

m_rung(i) = [<Sz(i,0)> - <Sz(i,1)>] / 2
spin RMS = sqrt(mean_i m_rung(i)^2)
```

Charge RMS includes residual open-boundary density modulation. Spin RMS is
the static leg-odd magnetic amplitude; by itself it does not distinguish
stripe antiphase structure from ordinary antiferromagnetic order. Neither
is a fluctuation spectrum or an infinite-system order parameter.
The older correlation summary's spin RMS used rungs 6–59; this grid
recomputes it over 9–56 for a common bulk definition.

For the stored unnormalized rung singlet D:

```text
P(i,j) = <D_i^dagger D_j>
P_connected(i,j) = P(i,j) - <D_i>* <D_j>
short = mean |P(i,j)| for 2 <= |i-j| <= 4
long  = mean |P(i,j)| for 16 <= |i-j| <= 24
```

In the subtraction, `<D_i>*` means complex conjugation. Both endpoints must
be in rungs 9–56. Every eligible ordered bond pair receives equal weight;
short and long definitions are reused from the existing analysis. The
manuscript's normalized singlet gives correlations half as large.
Short-distance axes are linear; long-distance axes are logarithmic. Scales
are common across geometries and across full/connected columns at the same
distance range. Tiny tails have no independent truncation-error bound.

## What the plots show

The square and cubic clouds broadly associate larger static amplitudes with
weaker pairing correlations. Square short-distance connected correlations
are not globally monotonic: the weak-stripe paired cluster has a smaller
connected short-distance value than some intermediate-stripe snapshots,
even though its full pair correlation is larger. Subtracting the anomalous
component therefore changes the apparent association.

Trellis one- and two-ladder snapshots have very similar short-distance
connected correlations, while the two-ladder stripe endpoints have much
smaller long-distance correlations. With only two spatial ansatzes at one
parameter coordinate, this is a comparison of endpoints rather than a
continuous trend.

These are cross-parameter and cross-state associations, not an estimate of
the causal effect of stripes. Changing V or t0 also changes pairing physics
directly. Acceptance labels, finite-L/chi limits, and the distinction between
full and connected correlations are retained; no phase ranking is inferred.

## Reproduction and focused validation

From the repository root on local Windows:

```powershell
& C:/Python313/python.exe -B ladder_mps_mft/scripts/plot_stripe_pairing_grid_20260923.py
```

The script reads the existing summary/source manifest, verifies all 66
diagnostic SHA-256 hashes and source-state lineage hashes, checks stored
status and model labels, verifies Hermiticity and connected subtraction,
and recomputes all four pair measures against the existing summary. The
source manifest supplies the chi=200 provenance. Charge/spin RMS values
are calculated directly from the saved density/spin profiles. The exported
CSV retains source IDs, parameters, spatial cells, status and hashes.
Both PNG grids were visually inspected after rendering. No new DMRG,
measurement backfill, source-state edits, scheduler actions or transfers
were performed.
