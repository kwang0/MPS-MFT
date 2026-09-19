# Finer square cuts near the stripe–pairing boundary

Local synchronized snapshot, 18 September 2026. Part of the
[combined campaign review](../campaign_review_20260918/README.md).

**Two points retain distinct textures from the two seeds:** (1.25,−0.4)
and (1.4,−0.05). The other four new points reach paired trajectories from
both starts, with residual spin still decaying in one (1.4,−0.10) run.
These are useful candidate metastability points, but no endpoint is formally
accepted and the runs do not yet establish the transition's order.

![Fine-cut order parameters and energy differences](cut_summary.png)

## Endpoints after 60 raw evaluations

All twelve runs in `20260915_square_two_basin_fine_cuts_95_5_60` completed
their 60-evaluation cap: **720 saved evaluations**, zero accepted endpoints.
L=64, chi=200, U=8, n=0.9375, tp=0.1, reciprocal 95%/5% reference seeds,
no Anderson or damping, minimum 40 evaluations and a ten-record window.

S/P in the table denote the stripe-dominated and pairing-dominated initial
seeds. Spin is physical leg-odd RMS on rungs 6–59; pairing is physical
leg-bond RMS on bonds 6–58. D denotes pairing with opposite leg/rung signs,
not a proof of a thermodynamic superconducting phase.

| (t0,V) | Spin RMS S / P | Leg-pair RMS S / P | Observed trajectories | E(P)−E(S), t/site |
|---|---:|---:|---|---:|
| (1.4,−0.15) | 3.24e−6 / 2.03e−6 | 0.033329 / 0.033329 | Both D | +1.22e−9 |
| (1.4,−0.10) | 4.38e−4 / 2.60e−5 | 0.033139 / 0.033139 | Both paired; spin remainder decays | −5.50e−11 |
| (1.4,−0.05) | 0.144695 / 0.005606 | 3.67e−5 / 0.032991 | Stripe / paired with weak end-weighted spin | −5.95e−5 |
| (1.25,−0.4) | 0.141970 / 9.60e−5 | 1.87e−7 / 0.028445 | Stripe / D | −3.19e−5 |
| (1.30,−0.4) | 1.05e−6 / 3.70e−7 | 0.031061 / 0.031061 | Both D | −1.78e−8 |
| (1.35,−0.4) | 6.97e−7 / 6.57e−7 | 0.033383 / 0.033383 | Both D | −1.62e−9 |

Along t0=1.4, both starts are paired through V=−0.10, split at −0.05, and
are striped at the earlier V=0 endpoint. Along V=−0.4, both earlier t0=1.2
starts are striped, the new t0=1.25 starts split, and both are paired from
1.30 upward. The coarse endpoint markers in the figure have different
stopping controls; connecting lines guide the eye and are not fitted phase
boundaries.

The paired (1.25,−0.4) spin amplitude falls by 52.1% from evaluations 51
to 60. At (1.4,−0.10), spin falls by about 55% over the same interval in
both trajectories. Those remnants show continued decay rather than the
late SDW growth previously seen in the square V=0 pairing lineage.
Relative growth fits for amplitudes around 1e−6 are not treated as resolved
instabilities.

## The subtle (1.4,−0.05) paired trajectory

![Boundary spin diagnostic](boundary_spin.png)

Its usual spin RMS bottoms out around evaluation 30 and rises slightly:
0.005523 at 51 to 0.005606 at 60, a 1.50% increase. That statistic alone
could suggest renewed stripe growth. The spatial profile gives a different
qualification: **96.7% of the final full-chain spin-squared weight lies in
the outer 14 rungs at each end**. The central 32-rung spin RMS is 0.001703
and still falls by 0.385% over the last ten records. Pairing remains 0.032991.

This is a paired trajectory with weak, strongly end-weighted spin. It does
not yet demonstrate bulk coexistence, and the late increase in the wider
window does not establish a bulk return to stripes. Full spatial profiles
are essential here. The stripe-seeded competitor has strong spin and only
3.67e−5 leg pairing, which is still falling at the endpoint.

![Two seed-dependent spatial profiles](split_seed_profiles.png)

## Full variational energies across the transition

![Full variational energy along both cuts](variational_energy_cuts.png)

These are **energies versus Hamiltonian parameter**, including all three new
coordinates and both coarse endpoints on each cut. They use the full stored
canonical variational functional, including the field-dependent mean-field
double-counting terms, with the target-density correction
`E_target/N = E_var/N + mu*(n_target-n)`. The effective-Hamiltonian eigenvalue
alone is not plotted. All ladders have 128 sites, so total energies are these
values multiplied by 128; both forms, and the uncorrected canonical energies,
are exported in [the endpoint table](variational_energy_cuts.csv).
The convention is the [implemented functional](../../VARIATIONAL_FUNCTIONAL.md);
no additional field-independent perturbative offset has been introduced.

On the full scale, **the V cut looks almost linear and the t0 cut smoothly
curved**. The seed differences are too small to distinguish at this scale.
The coarse V=0 markers use the latest 100/82-evaluation continuation endpoints,
not the earlier parent states. Open markers distinguish coarse controls from
the new 60-step fine runs. Each curve connects independent runs sharing a seed
family; a seed can end in a different phase at different coordinates.

![Energy curvature and adjacent-interval slopes](variational_energy_shape.png)

To expose small features, the upper panels subtract **the same straight chord
from both seed curves in a cut**. The chord joins the mean of the two seed
energies at each outer endpoint. This removes only a linear background; it
does not change seed gaps or slope changes. The lower panels give actual
adjacent secants `Delta(E/N)/Delta p`, with p=V or t0, rather than a fitted
derivative. Their bars show the sum of the two endpoints' final-ten energy
ranges divided by the interval width: a recent-drift diagnostic, not an
uncertainty bound. The two cuts have independent vertical scales.

| Cut and interval | Stripe-seed secant | Pairing-seed secant |
|---|---:|---:|
| V: −0.20 → −0.15 | 1.266231 | 1.266231 |
| V: −0.15 → −0.10 | 1.266144 | 1.266144 |
| V: −0.10 → −0.05 | 1.267403 | 1.266212 |
| V: −0.05 → 0 | 1.260138 | 1.261367 |
| t0: 1.20 → 1.25 | −0.339825 | −0.340515 |
| t0: 1.25 → 1.30 | −0.369494 | −0.368857 |
| t0: 1.30 → 1.35 | −0.393242 | −0.393241 |
| t0: 1.35 → 1.40 | −0.417749 | −0.417749 |

**There is a small candidate bend on the V cut, but no resolved first-order
kink.** The pairing-seed secant is nearly constant at 1.2662 through V=−0.05,
then falls to 1.26137 in the final interval (a 0.38% decrease). The stripe-seed
curve also turns downward. The common-background plot makes this visible
near the change from paired to stripe outcomes. A first-order crossing can
produce a continuous energy with a discontinuous first derivative; it need
not produce a jump in energy. Here the interval-average slope change is
compatible with such a weak crossing, but also with a smooth change of
curvature. Straight segments joining five samples cannot resolve that distinction.

**The t0 cut does not show an obvious sharp cusp.** Its pairing-seed slopes
evolve −0.3405, −0.3689, −0.3932, −0.4177, with nearly regular increments.
The broad arch after linear subtraction reflects that curvature; its polygonal
appearance is a sampling effect, not evidence for a cusp at t0=1.30.
Absence of a visible kink does not exclude a weak first-order transition.

The energy shape and distinct textures together make first order plausible,
especially on the V cut, but do not establish it. Only one coordinate on
each cut retains two distinct textures; these curves do not trace two
stationary phase branches through a crossing. Every endpoint remains
unaccepted. The coarse t0=1.2 pairing lineage has a final-ten energy range
1.56e−4 t/site, much larger than most others, and contributes the visible
drift bar to the first t0 secant. Varying the interpolated E_p also varies
g=tp²/|E_p|, so these slopes include that smooth coupling dependence and are
not the bare nearest-neighbor density correlator or rung kinetic energy alone.

Exact [secants](variational_energy_slopes.csv), [chords and source hashes](variational_energy_analysis.json),
and PDF versions of [full curves](variational_energy_cuts.pdf) and
[shape diagnostics](variational_energy_shape.pdf) accompany the plots.
Regenerate just these figures with
`python scripts/analyze_two_basin_campaigns_20260918.py --energy-cuts-only`
from the ladder subproject. This verifies all 20 endpoint hashes, summary
energies and density corrections without rerunning the campaign analysis.

## What the seed-to-seed energy differences establish

At the two split points the signed endpoint gaps are −3.189e−5 t/site
(t0=1.25) and −5.952e−5 t/site (V=−0.05). The sums of the two final-ten
energy ranges are 1.574e−7 and 7.461e−8 respectively. The gaps are therefore
well resolved relative to recent energy drift, but those ranges are **not
error bars on the converged variational energies**. They do not bound
remaining field relaxation, finite-chi error, or seed-dependent transients.
The stripe t0=1.25 run also misses the energy and inner-DMRG gates.

We retain these signed diagnostics without selecting an energetic winner.
Persistent distinct branches and sizable order-parameter differences make a
first-order scenario worth testing. They do not prove it: the endpoints
are unfinished, no accepted energy crossing was measured, these independent
starts are not parameter-continuation hysteresis, and the weak spin near the
V=−0.05 boundary is not established bulk coexistence. A continuous transition
with slow relaxation is not excluded by this snapshot.

## Pair-binding interpolation and remaining qualification

All six signed E_p values match the user-requested straight-line
interpolation and recorded bracket/weight in the manifest. The V cut uses
V=−0.2 and 0 at t0=1.4; the t0 cut uses t0=1.2 and 1.4 at V=−0.4.
These are interpolated couplings, not new bare-ladder measurements.
The [earlier bare-ladder check](../two_basin_fine_cuts_20260915/README.md)
found a model-dependent interpolation sensitivity of at most 0.62% in the
coupling along V and about 5% along t0. Those checks are not uncertainty
bounds; they matter before quoting a precise boundary location.

Controls, all failed gates, density errors, last-sweep energy changes and
discarded weights are preserved in the [audit JSON](analysis.json) and
[run table](run_summary.csv). Qualitative paired/stripe assignment remains
separate from fixed-point acceptance. No thresholds were relaxed.

## Full histories, provenance and cost

- [Energy from the beginning](energy_grid.png), [late energy](energy_late_grid.png),
  [spin](spin_rms_grid.png), and [pairing](pairing_rms_grid.png); PDF versions
  use the same filenames. Energy panels retain individual y scales.
- [All iteration data](iteration_history.csv), [terminal profiles](terminal_profiles.csv),
  and [state/config hashes](sources.csv).
- Reproduction: [analysis script](../../../scripts/analyze_two_basin_campaigns_20260918.py).

The twelve completed histories contain **13.030485 solver-only fractional
node-hours**. The September 19 sync now includes all twelve reconciliations,
giving **13.274722 actual node-hours**. The 36-node-hour reservation remains
a requested ceiling, not actual cost. See the [completed accounting audit](../campaign_review_20260918/completion_accounting.json).
Compact/config/seed hashes, pointwise fingerprints, interpolation, raw-map
adjacency, logs, physical correlations and energy/channel calculations were
checked locally. No simulation or accounting artifact was changed.

## Matched physical spin/pairing cuts — September 19 addition

![Physical spin and pairing along both transition cuts](order_parameter_cuts.png)

This figure uses exactly the same twenty hashed endpoints as the full
variational-energy curves. Spin is physical leg-odd Sz RMS on rungs 6–59;
leg pairing is symmetrized and averaged over the two legs before its RMS
on bonds with left rungs 6–58. Coarse endpoints are recomputed with that
same convention. Dotted curves use the central 32 spin rungs. These are
anomalous amplitudes, not four-fermion pair–pair correlations.

Finite pairing and sizable stripe order remain in distinct trajectories
at one point per cut. Their abrupt sampled changes motivate a first-order
test; lines between independent starts do not measure a discontinuity,
hysteresis or a stationary coexistence region. The small end-weighted spin
at paired V=−0.05 is better resolved in the [logarithmic companion](order_parameter_cuts_log.png).

[PDF](order_parameter_cuts.pdf), [log PDF](order_parameter_cuts_log.pdf),
[values and provenance](order_parameter_cuts.csv), [definitions and audit](order_parameter_analysis.json).
This figure is included in manuscript Section 3.10. Reproduce locally with
`python scripts/complete_campaign_analysis_20260919.py` from the ladder
subproject. No state, threshold or acceptance flag was changed.
