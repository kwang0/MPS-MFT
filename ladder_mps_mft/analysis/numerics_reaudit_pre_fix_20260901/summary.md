# Offline SCF numerical re-audit

Read-only audit; no HDF5 status or acceptance field was modified.

- State paths audited: `36`
- Unique full-artifact identities or standalone paths: `36`
- Unreadable/incomplete paths: `9`
- Stored classifications changed by the proposed numerical gates: `14`
- Changed v3/Stage-A paths: `8`

The period-two criterion requires every recent step cosine to be at most `-0.5` and every two-step/one-step ratio to be at most `0.5`. Slow-mode extrapolation is activated when residual cosine is at least `0.9`.

## Reclassified v3 and Stage A states

| run | branch | stored | revised | cos(step) | d2/d1 | lambda | extrapolated relative residual |
|---|---|---|---|---:|---:|---:|---:|
| 20260824_phase1_gpu_v3_float64_history | cdw | fixed_point | fixed_point_candidate_slow_mode | 0.99889741004814681 | 2.0873757464010478 | 0.9523704302273488 | 0.013909894575277 |
| 20260824_phase1_gpu_v3_float64_history | sc | fixed_point | fixed_point_candidate_slow_mode | 0.99940051886043091 | 2.0899994076762831 | 0.95879896944954934 | 0.012314823968365874 |
| 20260824_phase1_gpu_v3_float64_history | sdw | fixed_point | fixed_point_candidate_slow_mode | 0.99993855515438623 | 5.0277684398633609 | 0.99817024773105589 | 1.9728713806789551 |
| 20260824_phase1_gpu_v3_float64_history | cdw | fixed_point | fixed_point_candidate_slow_mode | 0.99968433029583925 | 2.0141765384044068 | 0.98555589106134123 | 0.07524277703485506 |
| 20260824_phase1_gpu_v3_float64_history | sc | periodic_candidate | iterating_monotone_drift | 0.99993103888527779 | 1.9965748090419897 | 1.0043724326812662 | Inf |
| 20260824_phase1_gpu_v3_float64_history | sdw | fixed_point | fixed_point_candidate_slow_mode | 0.99990237609468513 | 2.0078761077645768 | 0.99230334587718216 | 0.1069701079463936 |
| 20260826_phase1_unfrustrated_pairing_recurrence_chi400 | sc | periodic_candidate | iterating_monotone_drift | 0.99985246121578431 | 1.9834954366342037 | 1.0268817338812797 | Inf |
| 20260826_phase1_unfrustrated_pairing_recurrence_chi400 | sc | periodic_candidate | iterating_monotone_drift | 0.99985501309382385 | 1.9812502908336411 | 1.0284151547286009 | Inf |
