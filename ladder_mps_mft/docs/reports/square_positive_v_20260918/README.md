# Square (t0,V)=(1.2,+0.2): all four seeds lose pairing

**Updated 19 September 2026 from user-synchronized final states.** This
supersedes the September 18 three-state/partial-log assessment. All four
starts reach 60 raw MF evaluations at L=64, chi=200, with
`status=maximum_iterations` and `accepted=false`.

The period-eight intertwined seed also loses pairing. It leaves a different
magnetic texture, however, so the four runs have not merged into one spatial
fixed point. No tested seed sustains the legacy frustrated-cubic coexistence
of appreciable anomalous pairing and stripes at this square coordinate.

| Initial seed | Job | Spin RMS | Leg-pair RMS | Corrected energy, t/site | Final-ten energy range, t/site |
|---|---:|---:|---:|---:|---:|
| 95% stripe | 58394103 | 0.233268 | 1.54e−10 | −0.333744046217 | 3.57e−8 |
| 95% pairing | 58394104 | 0.233212 | 1.72e−10 | −0.333744379065 | 5.10e−8 |
| Intertwined period 8 | 58394105 | 0.207004 | 7.01e−9 | −0.332811625875 | 5.38e−8 |
| Intertwined period 16 | 58394106 | 0.233581 | 8.29e−10 | −0.333743848121 | 2.66e−7 |

Spin is the physical leg-odd Sz RMS on rungs 6–59. Pairing is the symmetric
nearest-leg anomalous amplitude averaged over the two legs before taking
the RMS on bonds with left rungs 6–58. These are anomalous amplitudes,
not four-fermion pair–pair correlations. Energies include the canonical MF
functional and target-density correction, normalized by 128 sites.

![Figure 1. All four full histories and late energy detail](histories.png)

**Figure 1** shows pairing disappearing from both intertwined starts.
The period-eight remainder falls another 87.8% over evaluations 51–60.
Its value of 7e−9 does not support coexistence. Small fluctuations in the
other seeds after pairing reaches tiny values have the same limitation.

## The shorter-period seed leaves magnetic defects

![Figure 2. All four terminal spatial profiles](terminal_profiles.png)

In **Figure 2**, the stripe, pairing and period-16 seeds have four
antiphase spin-envelope nodes, near rungs 10, 25, 40 and 55, up to wall
shifts and a global spin reversal. Their dominant charge/spin Fourier
bins are m=4/30: nominal charge period 16 and spin-envelope period 32.

The period-eight start begins with eight envelope nodes and ends with six,
near **10.16, 23.21, 26.81, 38.30, 41.89 and 54.91**. Two closely spaced
node pairs remain around the inner hole-rich regions. Charge nevertheless
has dominant m=4; the initial charge wavelength eight has not simply survived.
The spin spectrum is mixed: m=31 has amplitude 0.09264, m=30 has 0.08450,
and m=28 has 0.06761. Calling this a clean period-64 spin state from its
largest Fourier bin would conceal the nonuniform wall spacings.

The profile is consistent with incomplete coarsening or a trapped magnetic
defect texture. That is an interpretation of the observed nodes, not a
demonstrated stationary metastable state. Its endpoint lies **9.3242e−4
t/site above the stripe-seeded endpoint**, versus a sum of recent energy
ranges of 8.96e−8. Recent drift is not a bound on remaining relaxation or
finite-chi error. No accepted energetic ranking is made.

## Why a flat energy is insufficient

All four fail global field/slow-mode and spatial charge, spin and exchange
checks. Stripe and period-eight starts additionally fail an inner-DMRG
window check; period 16 fails the 1e−7 t/site energy window. Over the last
ten evaluations, maximum pointwise spin changes remain **0.00358–0.00655**,
despite nearly stationary spin RMS and often flat energy. Wall positions
can move while a norm barely changes. The failures are not solely tiny
pairing-channel noise. Original thresholds and acceptance flags are preserved.

This completed test strengthens the conclusion that **repulsive V alone
does not guarantee intertwining** in the tested geometries. It neither
rules out other square basins/parameters nor measures whether local pair–pair
correlations survive after anomalous order disappears. That separate
measurement is enabled for future runs and the prepared retrospective pass;
no new four-fermion measurements are used in this report.

## Evidence, cost and reproduction

The four states contain **240 MF evaluations**. Recorded solver time totals
4.523293 fractional node-hours. The newly synced completed-job accounting
gives **4.602431 actual node-hours** for all four jobs (one GPU = quarter node).

- [Audited states and gates](analysis.json), [source hashes](sources.csv),
  [full histories](iteration_history.csv), [terminal profiles](terminal_profiles.csv),
  [nodes and endpoint differences](endpoint_comparison.json).
- The old `intertwined_lambda08_log_history.csv` remains a historical
  September 18 partial-log snapshot, superseded by the complete state.
- [Seed construction](../square_positive_v_seeds_20260915/README.md),
  [combined report](../campaign_review_20260918/README.md).

Run locally from the repository root:

```powershell
C:/Python313/python.exe -B -X utf8 ladder_mps_mft/scripts/analyze_two_basin_campaigns_20260918.py --positive-only
C:/Python313/python.exe -B -X utf8 ladder_mps_mft/scripts/complete_campaign_analysis_20260919.py
```

Checks cover compact/config/seed hashes, matching fingerprints, raw-map
adjacency, canonical reconstruction, physical profiles, stored gates, stdout
counts and accounting. State files remain read-only; no new simulations or
scheduler operations are performed.
