# Square chi=200 grid, September 8 snapshot

The full 3 x 3 square grid is compiled in one self-contained
[HDF5 file](../../../output/square_grid_chi200_20260908/square_grid_chi200.h5)
(104.76 MiB, including full saved histories). It contains **six software-accepted fixed points, two historical legacy
results, and one explicitly unaccepted, diverging endpoint**. The latter three
coverage choices were approved by the user on September 8. No converged-stripe
inheritance campaign is used. Legacy files lack recorded seed ancestry; their
inclusion is user-approved coverage, not an independent ancestry certification.

**September 8 interpretation update:** the restored histories reveal coherent
SDW growth in several loose accepted runs and its suppression by Anderson
mixing. Software acceptance does not certify a stable paired basin. See the
[basin assessment and exact jump replay](BASIN_ASSESSMENT.md) for the evidence
and proposed two-family seed comparison. The original flags and selections
remain intact.

The local five-point square campaign is terminal: four fixed points and one
`diverging` result. The user reports that the cubic campaign and the separate
`(t0,V)=(1.4,-0.4)` stripe/control comparison are still running. This snapshot
does not verify live scheduler state or accounting.

## Plotting

Open [the physical-correlation Fourier grid](../../../output/square_grid_chi200_20260908/fourier_grid_correlations.png)
or its [PDF](../../../output/square_grid_chi200_20260908/fourier_grid_correlations.pdf).
The [MF-proxy grid](../../../output/square_grid_chi200_20260908/fourier_grid_mf.png)
and [PDF](../../../output/square_grid_chi200_20260908/fourier_grid_mf.pdf) are also available.

Interactive Julia, from the repository root:

```julia
include("ladder_mps_mft/plot_square_grid.jl")
grid = plot_square_grid()
# For the old default's field-proxy view:
grid_mf = plot_square_grid(; source=:mf)
```

The default `source=:correlations` compares the physical density, spin and
pairing expectation values. Hover shows the Fourier maximum and wavevector;
click opens that cell's Fourier maps and the full five-row, two-column
MF profiles and middle histories, with the iteration slider. New Phase 1
points use the existing measured-field history adapter: the recorded time-zero
seed is displayed as iteration 1, followed by every saved update. Legacy
points retain their original correlation histories (or MF histories with
`source=:mf`). The grid's `source` selects the Fourier quantities; new-run
history panels display MF fields because those are the saved per-update data.
A different bundle
path may be passed as the first positional argument. Use the Julia environment
with HDF5, PyCall and PyPlot, as for the legacy plotter.

This wrapper reuses `plot_ladder_mf_observables.jl` without changing it. It
exports temporary plotting data from the bundle, which remain until Julia exits.
Full histories are extracted lazily on click. Reload `plot_square_grid.jl` and
recreate the grid after an update; existing figure callbacks retain their old
behavior.
The original five-region cell design, color maps, shared logarithmic scale,
yellow maximum outline, DFT normalization and default five-rung boundary trim
are retained. CDW has its spatial mean removed. The ladder transforms use 54
rungs and divide by 108 sites; extended-s/d-wave transforms use 53 bonds and
divide by 53. Values below `1e-4` share the color floor, not a numerical zero.

For MF plotting, the spin-resolved `mu_cdw` is mapped onto the plotting copy's
`beta` diagonal, following the existing Phase 1 adapter. The original tensors
are preserved. MF proxies carry coupling factors and should not be interpreted
as physical correlation amplitudes; the two plot types can therefore highlight
different channels. These finite-size order-field Fourier amplitudes are not
connected structure factors or thermodynamic phase assignments.

## Selection

All points have square geometry, L=64, U=8, tp=0.1, density=0.9375 and chi=200.

| t0 | V | Selected source | Status |
|---:|---:|---|---|
| 1.0 | -0.4 | September 3 smooth-pairing grid, terminal record 30 | **Diverging**, hatched |
| 1.0 | -0.2 | September 3 smooth-pairing grid | Accepted |
| 1.0 | 0.0 | Legacy stateless result | Legacy completed |
| 1.2 | -0.4 | September 3 smooth-pairing grid | Accepted |
| 1.2 | -0.2 | September 3 smooth-pairing grid | Accepted |
| 1.2 | 0.0 | September 3 smooth-pairing grid | Accepted |
| 1.4 | -0.4 | August 30 `stripe_pairing_m004_chi200_loose` | Accepted |
| 1.4 | -0.2 | Legacy stateless result | Legacy completed |
| 1.4 | 0.0 | September 2 `stripe_pairing_m004_chi200_loose` | Accepted |

The two `stripe_pairing` starts are independent **small** sources of field norm
per physical site `1e-3`, with no parent, restart or inherited fields. They are
eligible under the requested seed rule. The smooth pairing starts have the same
norm, zero initial Hartree/exchange fields and no inherited state.

Within each multi-seed point, retain stored acceptance only if the existing
history-based slow-mode screen also passes, then select the smallest
target-density-corrected canonical solution energy after model, numerical,
implementation and E_p-registry fingerprints match. For August 30, reconstruct
the correction from stored solution canonical energy, chemical potential and
measured particle number, as in `src/Selection.jl`. No effective-Hamiltonian
eigenvalue or cross-coordinate energy comparison enters selection.

At `(1.4,-0.4)`, pure `stripe_m004` fails the slow-mode screen and is excluded.
The five eligible energies span only about `9.71e-9 t/site`; the chosen
`stripe_pairing_m004` is below the d-wave pairing start by `1.58e-9 t/site`.
At `(1.4,0)`, all six pass, and their energy spread is `6.0274e-7 t/site`;
the chosen start is below the d-wave pairing start by `1.8836e-7 t/site`.
These are deterministic choices among nearly tied finite-accuracy solutions,
not resolved physical energy orderings. The unaccepted August 31 tight-five
probes, frozen legacy-stripe evaluations, chi=400 runs, and pending inherited
stripe comparison do not replace the accepted chi=200 small-seed endpoints.

[selection.csv](selection.csv) gives the nine choices and hashes.
[selection.json](selection.json) retains all 17 candidate states, the existing
history screen, fingerprints and selection decisions. The two legacy points
are not recertified under the refactored acceptance or energy-ranking gates.

## Divergence analysis

See [ANALYSIS.md](ANALYSIS.md), [history figure](divergence_history.png), and
[spatial profiles](divergence_profiles.png). The terminal file records a
residual-based divergence stop after a slow striped trajectory; the fields
remain finite, pairing has decayed, and the final spike is in Hartree/exchange
channels. A converged solution or physical periodic orbit is not established.

## Bundle schema and reproduction

`points/<point_id>/plot_data` contains terminal `alpha`, mapped `beta`,
`mu_cdw`, `C_pair`, `C_exc_dn`, and `C_exc_up`. Arrays are readable directly in
Julia's conventions; h5py sees reversed dimensions, as for HDF5.jl source
files. In schema v2, `source_snapshot` preserves original terminal fields,
correlations, energies, provenance, full applied/measured field histories,
recorded seeds and per-update diagnostics for every Phase 1 point. Legacy
points retain their original fields/scalars and all six saved history arrays.
No MPS is included. Source paths, source/recorded-full
hashes, status flags, configs and selection evidence are embedded in the bundle;
plotting does not require the original source tree.

Schema v2 supersedes the initial terminal-only compilation at the same path.
The selected runs and Fourier maxima are unchanged; the compiler verifies
every retained history dataset against its immutable source before replacing
the bundle. The latest bundle hash is recorded in `selection.json`.

Local PowerShell commands from the repository root:

```powershell
python -B -X utf8 ladder_mps_mft/scripts/compile_square_grid.py
julia --startup-file=no ladder_mps_mft/plot_square_grid.jl
python -B -X utf8 ladder_mps_mft/scripts/analyze_square_grid_divergence.py
python -B -X utf8 ladder_mps_mft/scripts/analyze_square_basin_stability.py
```

The compiler is an explicit dated selection recipe; it does not silently add
future campaigns. The HDF5 and plot exports live in ignored `output/`, so they
must be copied explicitly if needed on another computer. No Perlmutter action
is needed to reproduce this analysis locally.

Validation: all 17 candidate compact-state hashes, sizes, recorded full hashes
and config hashes checked against local manifests; no-MPS/stateless checks;
all nine bundled plotting snapshots compared exactly against their sources.
A focused Julia check passed 95 assertions covering all five physical order
fields and Fourier maxima at every point, cell flags, and the click callback.
Both grid variants and both divergence figures were rendered and visually
inspected. Full scratch artifacts and scientific convergence beyond the stated
stored flags/history screen are outside this local verification. No DMRG ran.

The history restoration additionally passed 109 Julia assertions across all
nine click targets (all five middle-history traces, sample counts, seed indexing
and slider axes), plus 10 checks exercising both the Phase 1 and legacy sliders
from the final iteration to the first and back. Every retained source history
dataset was compared exactly during the rebuild. Histories contain 4–31
plotted samples depending on the selected run; the new runs include the seed.
