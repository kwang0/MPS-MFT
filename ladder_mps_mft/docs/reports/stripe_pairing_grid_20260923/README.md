# Stripe and uniform-pairing correlation grids

The requested cross-parameter scatter plots now separate the stripe and
uniformly paired endpoints. In the stripe group, pair correlations are
plotted against static stripe amplitude. In the paired group, the axes
reverse roles: connected charge/spin correlations are plotted against the
uniform pair amplitude. The original mixed-state grids are replaced.

## Plots and data

**42 stripe-state MPSs, with all uniformly paired states removed:**

- [Charge RMS versus pair correlations (PNG)](charge_pairing_grid.png) / [PDF](charge_pairing_grid.pdf).
- [Spin RMS versus pair correlations (PNG)](spin_pairing_grid.png) / [PDF](spin_pairing_grid.pdf).

**24 uniformly paired MPSs, with pairing strength on the horizontal axis:**

- [Stripe-wavevector connected weights (PNG)](paired_stripe_weights.png) / [PDF](paired_stripe_weights.pdf).
- [Short-/long-distance connected charge/spin correlations (PNG)](paired_stripe_distance_correlations.png) / [PDF](paired_stripe_distance_correlations.pdf).

Tables: [all 66 points](points.csv), [stripe subset](stripe_points.csv),
[uniformly paired subset](paired_points.csv), and
[validation/source hashes](validation.json).

Color encodes V; marker shape encodes t0. An outer ring marks an accepted
MPS. Every seed and A/B ladder remains a separate mark; overlaps are retained
without jitter or averaging. They are not independent statistical replicates.
All states have L=64 rungs, chi=200, U=8 and target density 15/16.
The older isolated chi1200 reference and prospective campaigns lacking
measurements in this source set are excluded.

## State separation

For the stored unnormalized rung singlet D and bulk rungs 9–56, let

```text
F_i = <D_i>
F_rms = sqrt(mean_i |F_i|^2)
F_uniform = |mean_i F_i|
uniform weight fraction = F_uniform^2 / F_rms^2
```

The paired group has F_rms > 0.01 and uniform weight fraction >= 0.99.
This separates a clear measured gap: the largest stripe-group F_rms is
0.00011033, versus 0.06578888 for the smallest paired value. Thresholds
0.001, 0.01 and 0.05 give exactly the same partition. Every selected paired
profile has uniform weight fraction >= 0.9984596. "Uniform" allows the
small amplitude variations caused by open boundaries; it does not require
an exactly constant finite-ladder profile. Classification uses measured
pairing, not the seed name or the charge/spin correlations being plotted.

The stripe group contains 20 square, 18 cubic and 4 trellis MPSs, all
unaccepted endpoints. The paired group contains 22 square and 2 trellis
MPSs; 16 are unaccepted and eight are the A/B MPSs from four accepted square
runs. There are no paired cubic endpoints in this data set. Trellis has only
one paired parameter coordinate, with two nearly identical seed outcomes;
its panels do not constitute a parameter trend. No scientific acceptance
status is changed by this plotting classification.

## Stripe-group axes

The static amplitude proxies use the common bulk rungs 9–56:

```text
n_+(i) = [<n(i,0)> + <n(i,1)>] / 2
charge RMS = sqrt(mean_i [n_+(i) - mean_bulk(n_+)]^2)

m_-(i) = [<Sz(i,0)> - <Sz(i,1)>] / 2
spin RMS = sqrt(mean_i m_-(i)^2)
```

Charge RMS includes residual boundary density modulation. Spin RMS alone
does not distinguish antiphase stripes from ordinary antiferromagnetism.
The earlier summary's spin RMS used rungs 6–59; these grids recompute it on
9–56 to use the same bulk as the pair correlations.

```text
P(i,j) = <D_i^dagger D_j>
P_connected(i,j) = P(i,j) - conjugate(F_i) F_j
short = mean |P(i,j)| for 2 <= |i-j| <= 4
long  = mean |P(i,j)| for 16 <= |i-j| <= 24
```

Both endpoints must lie in the bulk; every eligible ordered bond pair is
weighted equally. Definitions are reused from the September 22 analysis.
The manuscript's normalized singlet gives correlations half as large.
Short-distance axes are linear; long-distance axes are logarithmic. Scales
are common across geometries and full/connected columns at a given distance.

## Paired-group axes

The horizontal coordinate is F_uniform, not the mean-field source alpha.
The charge and longitudinal-spin operators are the rung average n_+ and
leg difference m_- defined above. From the saved site covariance matrices,

```text
C_O(i,j) = <O_i O_j> - <O_i><O_j>
S_O_connected(q) = (1/N_bulk) sum_ij exp[i q(i-j)] C_O(i,j)
N_bulk = 48
```

Each rung covariance carries the factor 1/4 from the two factors of 1/2 in
the operator definition. This is a rung-average normalization, not the
stored 1/(2L) site-normalized structure factor. All comparisons within the
new plots use this same normalization.

The fixed stripe-channel wavevectors are q_c=pi/8 (charge period 16, leg
even) and q_s=15pi/16 (antiferromagnetic spin modulation with period-32
envelope, leg odd). They reference the period-16/32 stripe pattern already
identified in this data set, including the trellis longitudinal harmonic.
For real covariances, the other magnetic satellite 17pi/16 is equivalent.
These are evaluated weights, not fitted peaks or claims that a stripe peak
is present in every paired state. Onsite/contact terms are included; the
table also records their contribution and the remaining signed offsite
weight. A finite value can include broad short-range backgrounds.

The companion distance grid averages |C_O(i,j)| for the same 2–4 and 16–24
windows, with both endpoints in the bulk. It excludes contact terms and
shows the spatial range of charge/spin correlations, but is not specific
to one wavevector. Signed averages are also retained in the table.
Only intraladder longitudinal-spin and charge covariances are used; these
are not dynamical spectra or interladder correlation measurements.

Paired panels have explicitly different axis ranges to show variation
within each geometry/measure. The tiny trellis seed differences are not
magnified: its x span matches the square row. Long-distance distance-grid
axes are logarithmic. No points are fitted or connected as a causal curve.

## Descriptive reading

In the stripe-only grids, larger static amplitudes broadly accompany weaker
pair correlations. In the paired square subset, the spin-weight association
depends on which parameter changes: along V=-0.4, increasing t0 increases
pairing and decreases the weight at q_s; along t0=1.4, increasing pairing
as V becomes more attractive instead increases that weight. The distance
grid also distinguishes these parameter paths. There is no single universal
"stronger pairing means weaker stripe correlations" trend across the cloud.

Parameter variation directly changes both channels. Classification and
selection do not establish phase coexistence or causal competition. Tiny
tails lack independent truncation-error bounds; finite L/chi and the original
acceptance labels remain relevant. No phase ranking is inferred.

## Reproduction and focused validation

From the repository root on local Windows:

```powershell
& C:/Python313/python.exe -B ladder_mps_mft/scripts/plot_stripe_pairing_grid_20260923.py
```

The script verifies all 66 diagnostic hashes, source-state lineage, status,
model labels, pair Hermiticity/subtraction and all four old pair-window
values. New checks verify charge/spin connected subtraction, rung-channel
projection, Fourier normalization against an independent explicit sum,
and the complete disjoint 42/24 partition. The source manifest supplies
chi200 provenance. Both subset CSVs retain every source ID and status.
All four PNG grids were visually inspected after rendering. No new DMRG,
backfill, source-state modification, transfer or scheduler action was needed.
