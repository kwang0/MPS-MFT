# Trellis: the first completed stripe seed evolves toward pairing

Local synchronized snapshot, 18 September 2026. Part of the
[combined campaign review](../campaign_review_20260918/README.md).

**The fixed one-ladder trellis map has a paired trajectory at (t0,V)=(1,0),
even when initialized with 95% stripe and 5% pairing.** Its substantial
initial spin order disappears while opposite-sign leg/rung pairing grows.
This is qualitatively different from the square and cubic-unfrustrated
results at the same bare-ladder coordinate, which reach stripes.

![Trellis full and late histories](histories.png)

## Completed run and physical profiles

Campaign `20260916_trellis_two_basin_comparison_60`, job **58468871**:
one-ladder cell, U=8, t0=1, V=0, tau0=tau1=0.1, L=64, n=0.9375, chi=200,
r_range=4. The exact signed E_p is −0.13251724. Sixty simultaneous raw cell
sweeps here are sixty ladder solves; no Anderson or damping was used.
The saved terminal status is `maximum_iterations`, `accepted=false`.

| Physical diagnostic | First measured evaluation | Evaluation 60 |
|---|---:|---:|
| Leg-odd spin RMS, rungs 6–59 | 0.0627150 | 1.21484e−5 |
| Leg-pair RMS, bonds 6–58 | 0.00965140 | 0.01796892 |
| Charge standard deviation, rungs 6–59 | 0.0230562 | 0.00565120 |

The final mean leg and rung pair amplitudes are +0.017956 and −0.032923.
The pairing RMS changes by only 6.59e−6 fractionally from evaluations 51
to 60; the maximum bond-profile change over those endpoints is 6.90e−7.
Spin is reduced by more than three orders of magnitude from the first
measurement and has no resolved late return to a strong stripe texture.

![Trellis spatial profiles](profiles.png)

Charge oscillations are strongest near the open ends and weaken in the
center (central 32-rung standard deviation 0.001572). They do not by
themselves demonstrate a bulk CDW intertwined with pairing. Opposite
leg/rung signs identify d-wave-like internal structure, not a spatially
sign-changing pair-density wave. Spin at about 1e−5 is too small/noisy here
to assign a robust spin wavelength from its largest Fourier bin.

Trellis fields mix density with normal bond correlators. This analysis
therefore uses the **stored raw correlation histories**, not the
square/cubic Hartree-field inversion. See the
[trellis map and energy contract](../../TRELLIS_MEAN_FIELD.md).

## Why it remains unaccepted

The last ten global relative residuals all pass 1e−4; their maximum is
8.46e−5 and the last is 2.10e−5. The global slow-mode, density,
Hamiltonian-identity and effective-energy checks pass too. However:

- The corrected energy range is **2.844e−7 t/site**, versus 1e−7 allowed.
- Inner-DMRG last-sweep differences fail at evaluations 55, 59 and 60;
  the final difference is **3.36e−7 t total** versus 1e−7 allowed.
- Spin/exchange-spin residuals and the spin, pairing and charge-modulation
  profile windows fail. The final spin absolute residual is 1.47e−6;
  weak-channel residuals are not simply below the 5e−7 floor.

The last discarded weight is 6.30e−5. A stable qualitative paired texture
is well supported, but finite-chi/inner-solver variability and the recorded
profile spans still prevent a fixed-point claim. The corrected endpoint
energy is **−0.518820336467 t/site**. No threshold or acceptance flag changed.

## Other trellis evidence available in the sync

These are partial stdout records only; none has a synced spatial artifact.
They are not terminal outcomes or live scheduler observations.

| Cell / initial family | Job | Complete logged cell sweeps | Latest corrected energy, t/site |
|---|---:|---:|---:|
| One ladder / pairing | 58468873 | 21 | −0.518820331322 |
| Two ladders / stripe | 58468875 | 18 | −0.519257203947 |
| Two ladders / pairing | 58468876 | 1 | −0.518667545597 |

The one-ladder pairing log is already very close in energy to the completed
one-ladder trajectory, but its profiles are needed before concluding that
both seeds merge. The two-ladder results are unfinished. A/B are spatial
states solved simultaneously, not alternating temporal states. The skew
one-ladder repetition and rectangular A/B cell impose different transverse
patterns and end cuts; no accepted cell comparison is available yet.

The interesting result now is the **change of observed basin with geometry**.
It is not a cross-geometry energy ranking and does not yet determine whether
the explicit two-ladder cell supports the same pairing or a competing texture.

## Reproduction, evidence and cost

The completed run records 28,609.98 solver seconds, or **1.986804 fractional
node-hours** at one quarter node. No actual allocation reconciliation for
it is in the local ledger. Partial-job costs are unavailable and excluded.

- [Audit JSON](analysis.json), [source hashes](sources.csv), [full iteration history](iteration_history.csv),
  [terminal profiles](terminal_profiles.csv), and [partial log histories](partial_log_histories.csv).
- [Preparation contract](../trellis_comparison_20260916/README.md).
- [Analysis script](../../../scripts/analyze_trellis_progress_20260918.py).

```powershell
C:/Python313/python.exe -B -X utf8 ladder_mps_mft/scripts/analyze_trellis_progress_20260918.py
```

Checks cover compact/config/seed hashes, fingerprints, raw-map adjacency,
stored correlation endpoints/densities, canonical energy reconstruction,
target-density correction, cell normalization, channel windows and stdout
counts. The audit does not modify results or inspect the remote scratch MPS.
