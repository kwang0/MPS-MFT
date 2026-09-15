# Preliminary square two-basin phase diagram — September 15, 2026

**The nine-point grid supports seven stripe assignments and two d-wave-like
paired assignments.** Both seeds reach the same phase family at eight points.
At `(t0,V)=(1.4,0)`, the pairing start is still converting toward stripes;
that assignment is marked `S*`. All 18 terminal artifacts are synchronized,
but none passed its original formal acceptance checks. These are preliminary
basin assignments from spatial order and full raw histories, not a converged
ground-state energy selection.

![Preliminary phase diagram](preliminary_phase_diagram.png)

[Phase-diagram PDF](preliminary_phase_diagram.pdf) ·
[All 18 endpoint diagnostics](run_summary.csv) · [Analysis and provenance](analysis.json)

## Coverage and phase evidence

Square geometry, `L=64` rungs / 128 sites, `chi=200`, density `0.9375`,
`tp/t=0.1`. Every point has reciprocal 95%/5% mixtures of the stripe and
paired references, reconstructed at its target couplings. The four original
anchors use their archived 80-step controls; the fourteen new runs use the
revised 40-step controls. Neither campaign uses Anderson.

| V / t | t0 / t = 1.0 | t0 / t = 1.2 | t0 / t = 1.4 |
|---|---|---|---|
| 0.0 | Stripe, both seeds | Stripe, both seeds | Stripe; pairing start still converting (`S*`) |
| -0.2 | Stripe, both seeds | Stripe, both seeds | Paired, both seeds |
| -0.4 | Stripe, both seeds | Stripe, both seeds; recent conversion | Paired, both seeds |

The observed separation is large. At the six points with `t0<=1.2`, final
bulk physical leg-odd spin RMS is `0.175–0.261`, while leg-pair RMS is at most
`9.85e-7`. All twelve of these stripe endpoints have dominant full-length
charge mode `m=4` (`q/pi=0.125`) and spin mode `m=30` (`q/pi=0.9375`). These
identify the familiar charge modulation and antiphase spin envelope on this
finite open ladder; they are not a stripe-wavelength or length extrapolation.

At the two paired points, physical leg-pair RMS is `0.0336–0.0352`, with
opposite-sign rung means `-0.0502` and `-0.0532`. Spin RMS is only
`3.11e-7–4.57e-6`. Leg pairing has about 2.1–2.6% bulk spatial variation
relative to its mean. Thus "d-wave-like paired" allows finite open-boundary
modulation; neither exact uniformity nor a thermodynamic order parameter is
being inferred. Weak residual charge structure alone is not classified as
a competing stripe phase.

The coarse grid brackets a change between `t0=1.2` and `1.4` at both negative
V values, and between `V=-0.2` and `0` at `t0=1.4`. No curve is interpolated
between points, and its location/order cannot be extracted from this grid.

## What the histories add

![Pairing histories](pairing_rms_grid.png)

![Spin histories](spin_rms_grid.png)

- **New paired control `(1.4,-0.2)`:** the stripe start loses spin while
  pairing grows to the same profile as the pairing start. Their final leg
  pair means agree to about `4.6e-8`, and their corrected endpoint energies
  differ by only `2.12e-9 t/site`. The pairing-start energy spans
  `3.04e-10 t/site` over its last ten records. Remaining spin decays, rather
  than growing as in the problematic loose V=0 states.
- **Delayed conversion `(1.2,-0.4)`:** the pairing-start MF leg-pair RMS is
  `0.00203` at evaluation 10 and `0.00195` at 20, while spin is already
  increasing. Pairing subsequently falls to `0.000639` at 30 and
  `7.83e-8` at 40. Its final ten values lose 99.976% of their initial
  pairing amplitude while spin grows 13.6%. This is another explicit
  example of a long paired transient ending in a stripe texture. The
  late energy span is still `1.56e-4 t/site`, so quantitative convergence
  needs more time despite the clear endpoint phase family.
- **Unfinished conversion `(1.4,0)`:** the stripe start remains essentially
  unpaired. At its deadline the pairing start still has physical leg-pair
  RMS `0.00524`, but its final five MF records lose 65.6% of pairing while
  spin grows 12%. This is the starred provisional assignment, not an
  established coexistence phase. The final density miss is also retained.
- **Other stripe endpoints:** strong CDW/SDW and tiny pairing are established,
  but slowly changing spatial profiles prevent strict self-consistency.
  `(1.0,-0.4)` has roughly `0.0034` relative field residuals and both
  energies are still drifting. At `t0=1.0/1.2, V=-0.2/0`, scalar amplitudes
  and energies look flat while the full spatial residuals remain resolved.

Large relative fluctuations in pairing after it reaches a tiny amplitude
are not interpreted as renewed pairing growth. Likewise, tiny spin noise
at the paired V=-0.4 endpoints is distinguished from a resolved instability.

## Acceptance and energetics

There are **zero formally accepted endpoints**. This is preserved in every
table and the figure. The new paired `(1.4,-0.2)` case illustrates why phase
identification and numerical acceptance need separate reporting:

- Pairing start: all other reconstructed terminal configured-window gates
  pass, but the ten-record spin-profile span is `1.074e-6 t`, above the
  `5e-7 t` floor. Its final individual spin update is already only
  `3.49e-8 t`; the window retains earlier resolved decay.
- Stripe start: spin is still decaying with contraction about 0.775;
  its final individual spin residual is `5.13e-7 t`. The ten-record global
  field window and spin/exchange-spin/pairing spans also fail. This is not
  evidence that the paired state is becoming unstable.

The stripe endpoints fail resolved field/slow-mode and profile-span gates,
often alongside energy and inner-DMRG windows. The original V=-0.4 anchors
retain the already documented weak-channel and marginal energy failures.
No thresholds were changed and no endpoint was relabeled by this analysis.

![Full corrected energy histories](energy_grid.png)

![Expanded late energy histories](energy_late_grid.png)

Every energy panel compares only the two runs at the same Hamiltonian.
Their four comparison fingerprints match within each point. The two
campaigns have different archived numerical/implementation fingerprints,
so this report does not silently treat all stopping controls as identical.
Reported energy is stored target-density-corrected canonical energy per
site, independently reconstructed from canonical energy, chemical potential,
and density. Early off-self-consistent energy dips are not candidate minima.

For the four stripe points at `t0=1.0/1.2, V=-0.2/0`, the two terminal
energies differ by `2.64e-7–4.38e-7 t/site`, despite similar phase labels.
Their charge profiles still differ by up to about `0.010–0.011` per site.
The V=-0.4 stripe points have terminal separations of `2.61e-6` and
`5.88e-6 t/site`. These differences describe unresolved trajectories and
texture relaxation; no winning branch or phase gap is selected from them.

The tested seeds therefore give a useful preliminary basin diagram without
yet supplying the intended accepted-state energetic phase comparison.
The most informative existing continuations would resolve `(1.4,0)` and
the late `(1.2,-0.4)` conversion; `(1.0,-0.4)` also needs spatial relaxation.
Any such compute remains a separate user decision.

## Iterations and allocation cost

All eighteen jobs have completed-job accounting in the user-synchronized
append-only reconciliation ledger. Exact fractional node-hours use saved
Slurm elapsed seconds multiplied by a 0.25-node share and divided by 3600.

| Campaign / point | MF evaluations | Allocation node-hours |
|---|---:|---:|
| Original `(1.4,-0.4)` anchors | 160 | 3.763125 |
| Original `(1.4,0)` anchors | 142 | 4.444722 |
| Fourteen remaining starts | 560 | 18.590486 |
| **Full 18-run grid** | **862** | **26.798333** |

Seventeen runs ended at their iteration cap; one ended at its solver time
limit after 62 evaluations. The scheduler accounting states are all
`COMPLETED`, which does not imply scientific convergence. Summed saved MF
time alone would estimate `26.474287` node-hours; the allocation figure
includes startup/finalization. No live scheduler query or ledger mutation
was performed. The V=0 actual cost now replaces the earlier estimate.

## Reproduction and verification

Run locally from the repository root:

```powershell
python -B -X utf8 ladder_mps_mft/scripts/analyze_two_basin_grid.py
```

The existing anchor loader was extended only with optional `t0` and campaign
directory arguments; its defaults and prior anchor summaries are retained.
All eighteen unique point/seed combinations are required. The script checks
compact/config/seed hashes, target model and chi, job IDs and provenance,
seed readback, every raw input/output handoff, recorded channel residuals,
and configured terminal profile-window spans. It independently reconciles
the physical spin with the square-geometry Hartree kernel, whose leg swap
reverses the leg-odd sign. Energy reconstruction, source hashes after reading,
log/HDF5 counts, and allocation arithmetic are checked as well.

Classification uses descriptive thresholds inside the large observed
amplitude gaps: stripe has spin RMS `>1e-3` and leg-pair RMS `<1e-4`;
paired has spin RMS `<1e-4`, leg-pair RMS `>1e-3`, and opposite leg/rung
mean signs. A mixed endpoint is marked as converting only when the recent
history shows a large pairing decrease and spin increase. These are display
rules, never substitute solver acceptance conditions. Varying the amplitude
separators tenfold leaves the settled-family assignments unchanged.

[Source inventory and hashes](sources.csv) · [862 iteration records](iteration_history.csv) ·
[Terminal charge, spin and rung-pair profiles](terminal_profiles.csv) ·
[Full gates, metrics and accounting](analysis.json)

All five PNG figures were visually inspected; each has a matching PDF.
The raw HDF5 states, numerical controls, manuscript and remote systems were
left unchanged. This is analysis of locally synchronized results only.
