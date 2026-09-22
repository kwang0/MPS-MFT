# Pair correlations and completed square A/B results

The [report](correlation_report.pdf) analyzes all 58 retrospective MPS
measurements, the eight new square A/B diagnostics, and the earlier isolated
ladder reference. [Editable report source](correlation_report.tex).

**Stripes retain local pair correlations, concentrated near hole-rich magnetic
walls, while long-distance correlations are strongly reduced.** Full versus
connected comparisons distinguish this from the anomalous plateau in paired
states. Rung–leg signs predominantly survive; trellis leg–leg correlations
also show separation-dependent signs. Reference and window sensitivity rule
out quoting one reliable asymptotic exponent from these data.

**All four square A/B runs are accepted paired fixed points.** Stripe/pairing
seeds converge in 40/40 sweeps at V=-0.4 and 55/44 at V=-0.2, with matching
profiles, energies and intraladder pair correlations. This supports survival
in the tested cell, without establishing general transverse stability.

The report defines every operator and averaging window. Numerical values use
the stored unnormalized bond singlet; divide correlations by two for the
manuscript's normalized singlet. The isolated reference uses chi=1200 versus
chi=200 in the coupled states and lacks rung–leg cross-channel data. All 56
retrospective branches retain their unaccepted status.

Reproduce from `ladder_mps_mft/`:

```powershell
C:/Python313/python.exe -B scripts/analyze_pair_correlations_20260922.py
```

The script checks receipts/source hashes, model/config provenance,
Hermiticity, Gram positivity, connected subtraction and state/diagnostic
agreement. It separately rechecks the square A/B fixed-point gates and
canonical-energy bookkeeping. Outputs include complete CSV/JSON evidence and
seven PDF/PNG figures. No DMRG or source-state changes are performed.

Boundary sensitivity is motivated by
[Shen, Zhang and Qin, PRB 108, 165113 (2023)](https://doi.org/10.1103/PhysRevB.108.165113),
whose primary arXiv record was checked on September 22. The model and numerical
controls differ from this project; no literature exponent is transferred here.
