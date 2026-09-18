# Cubic unfrustrated: complete two-seed grid

Local synchronized snapshot, 18 September 2026. Part of the
[combined campaign review](../campaign_review_20260918/README.md).

**Both seeds reach essentially unpaired CDW/SDW stripes at all nine points.**
This includes (1.4,−0.4) and (1.4,−0.2), which were paired on square.
There is no paired cubic endpoint in this tested grid. This is a statement
about the observed basins, not exclusion of every possible metastable state.

![Preliminary cubic phase diagram](preliminary_phase_diagram.png)

## Coverage and physical order

Campaign `20260915_cubic_unfrustrated_two_basin_95_5_60` has all 18 terminal
states: 60 raw evaluations each, **1080 total**, all `maximum_iterations`,
`accepted=false`, period zero. The reciprocal 95%/5% correlation templates,
L=64, U=8, n=0.9375, tp=0.1 and chi=200 match the prepared campaign contract.
No Anderson or damping was used.

Physical spin below is the RMS of (Sz₁−Sz₂)/2 on rungs 6–59. It is extracted
from the geometry-dependent fields and checked against stored correlations;
it is not a Hartree-field amplitude. Leg-pair RMS uses bonds 6–58.

| t0 | V | Spin RMS, stripe seed | Spin RMS, pairing seed | E(pair seed) − E(stripe seed), t/site |
|---:|---:|---:|---:|---:|
| 1.0 | 0.0 | 0.356220 | 0.356209 | −5.02e−9 |
| 1.0 | −0.2 | 0.348660 | 0.348670 | −5.98e−9 |
| 1.0 | −0.4 | 0.341539 | 0.341617 | −4.84e−8 |
| 1.2 | 0.0 | 0.326114 | 0.326116 | −2.86e−9 |
| 1.2 | −0.2 | 0.316349 | 0.316369 | −5.50e−9 |
| 1.2 | −0.4 | 0.307315 | 0.307413 | −1.11e−7 |
| 1.4 | 0.0 | 0.322006 | 0.322013 | +1.67e−9 |
| 1.4 | −0.2 | 0.296067 | 0.296083 | −8.46e−9 |
| 1.4 | −0.4 | 0.280224 | 0.280357 | −1.80e−7 |

The energy column reports signed **unaccepted endpoint differences**, not
energetic winners. The largest leg-pair RMS anywhere in these 18 states is
2.21e−11. All have dominant full-length charge mode m=4 and spin mode m=30:
nominal charge period 16 and staggered spin-envelope period 32 rungs.
Open boundaries and finite length qualify those wavelength assignments.

At t0=1.4, the pairing-seeded leg-pair RMS stays below 1e−4 from evaluations
15, 11 and 8 for V=−0.4, −0.2 and 0 respectively. These are well-developed
stripe trajectories after the loss of the initial pairing, rather than
ambiguous endpoints with comparable surviving anomalous order.

The comparison with square is physically meaningful because the order
parameters are correlations, not field units. Cubic changes the interaction
kernel, including a threefold density/same-leg coefficient and different
rung channels. These data demonstrate geometry dependence but do not isolate
one kernel term as its cause. Energies are never ranked across geometries.

## Why no point is formally accepted

Every endpoint passes the final observation-window density, inner-DMRG,
energy, Hamiltonian-identity and effective-energy checks used in this audit.
The final ten energy ranges are only 6.76e−10–1.75e−8 t/site, below 1e−7.
The remaining failures concern field stationarity, channel profile spans,
and/or slow-mode extrapolation. They do not represent appreciable pairing.

The stripe seed at (1.0,−0.2) is the closest to acceptance: its only failing
gate is the charge-modulation window, with relative span 1.01306e−4 versus
1e−4, a 1.31% excess. Other states, especially at (1.4,−0.4), still fail
several field/profile gates. A flat energy does not certify the full spatial
fixed point, so no acceptance flag or tolerance was changed.

## Figures and inspectable data

Energy panels retain separate y scales, as requested for the square report.
Iterations start at the first measured MF evaluation; an initial seed is not
inserted as an extra measured point.

- [Complete energy histories](energy_grid.png) and [PDF](energy_grid.pdf).
- [Late energy histories](energy_late_grid.png) and [PDF](energy_late_grid.pdf).
- [Spin histories](spin_rms_grid.png) and [pairing histories](pairing_rms_grid.png).
- [Run summary](run_summary.csv), [iteration histories](iteration_history.csv),
  [terminal physical profiles](terminal_profiles.csv), and [source hashes](sources.csv).
- [Full audit JSON](analysis.json), including every failed gate and cost record.

## Cost and verification

Saved solver work totals **13.886995 fractional node-hours**, using one GPU
as one quarter node. This excludes allocation startup/cleanup and is not an
exact allocation charge. The synced reconciliation ledger covers **12/18
jobs**, the t0=1.0/1.2 subset, totaling **9.441597 actual node-hours**. The
remaining six t0=1.4 jobs account for 4.642934 solver-only node-hours; their
actual allocation cost is unavailable in this snapshot. The original
54-node-hour reservation ceiling is not a measured cost.

The read-only analysis verifies compact hashes, config/seed hashes,
model/numerical/implementation/registry fingerprints, raw-map adjacency,
stdout iteration counts, physical-profile reconstruction, target-density
energy correction and channel-window calculations. It does not verify full
scratch MPS availability, alter simulation artifacts or query Perlmutter.

Reproduce locally from the repository root:

```powershell
C:/Python313/python.exe -B -X utf8 ladder_mps_mft/scripts/analyze_two_basin_campaigns_20260918.py
```

This also regenerates the accompanying square-cut and positive-V reports'
data and figures. Narrative README files are maintained separately.
