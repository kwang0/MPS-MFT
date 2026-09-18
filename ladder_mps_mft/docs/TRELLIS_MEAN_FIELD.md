# Reciprocal trellis mean field and spatial cells

The September 16 comparison extends the existing real, spin-diagonal,
second-order approximation with one denominator Delta=|E_p|. It implements
the two zigzag hoppings discussed in the trellis conversation. It is not a
new microscopic derivation for the Uehara material.

## Kernel and finite range

With ladder label m, rung i and legs 0/1, the skew-coordinate links are

```text
(m,i,1) --tau0-- (m+1,i,0)
(m,i,1) --tau1-- (m+1,i+1,0).
```

Let S[i,j]=delta[j,i+1] on the finite open ladder and A=tau0 I+tau1 S.
For each physical neighboring ladder separately, the fields are

```text
alpha = (2/Delta) P [T (P F) T^T]
B     = (2/Delta) P [T (P (X - I/2)) T^T].
```

F uses (down,up) indices, the transpose of the stored `pair` convention.
B's off-diagonal entries are beta; its diagonal is mu_cdw. P retains rung
separations at most r_range, including every diagonal. Projecting both input
and output preserves the adjoint relation when the shifted paths cross the
cutoff. Dropping only the out-of-range output would break reciprocity.

Normal fields include interference with the neighbor's bond correlations.
The `-T T^T/Delta` term has off-diagonal entries. The density cannot be
reconstructed by inverting a two-by-two rung density kernel. Raw correlation
histories are therefore stored in trellis checkpoints. No thresholding or
ad hoc end renormalization is applied. Missing OBC links are simply absent.
Different neighboring ladders never supply the two legs of one pair transfer.

## The two spatial ansatzes

`trellis_cell="one_ladder"` uses one representative MPS repeated in skew
coordinates. Leg 1 receives A F00 A^T, while leg 0 receives A^T F11 A.
The analogous centered normal maps use the same T matrices. The two leg
maps are adjoints. Tau1=0 recovers the square map, including its normalization.

`trellis_cell="two_ladder"` uses a rectangular transverse cell with ladder
A at longitudinal origin 0 and B at origin -1/2 rung. The A-to-B maps are
A on target leg 1 and C=tau1 I+tau0 S on target leg 0, sourcing the opposite
leg. B uses A^T on target leg 0 and C^T on target leg 1. Thus for equal
hoppings both A legs use forward offsets and both B legs backward offsets.
This is a fixed spatial assignment; it never changes with iteration parity.

The rectangular cell and the skew single-ladder repetition permit different
transverse arrangements of a longitudinal stripe. They must not be interpreted
as two numerically identical coordinate descriptions of the same restricted
ansatz. For OBC they also cut the array's ends differently. Compare bulk
profiles in physical coordinates, with the half-rung displacement recorded,
and distinguish end effects from a change of bulk order.

Each spatial ladder has a separate MPS, applied fields, measured fields,
chemical potential and history. One Jacobi cell sweep solves every ladder
against the frozen incoming fields, then rebuilds all outgoing fields from
the new correlation states. A and B use the same initial correlation
template and initial product-state configuration for each seed family.

The present comparison constrains the mean density on each ladder to
0.9375 separately. It permits different spatial profiles but does not test
charge transfer between ladders with different average fillings.

## Energy and acceptance

For a simultaneous trial cell, calculate its interaction fields from all
current MPS correlations, and evaluate

```text
Ecell = sum_m [ <H_ladder,m>
              - sum(alpha_m F_m)
              + (1/2) sum_sigma Tr(B_m,sigma (X_m,sigma-I/2)) ].
```

The existing centered density/exchange energy implementation gives exactly
this expression. Keeping both density-to-bond and bond-to-density terms
recovers the full derivative, including the affine normal term. Finite
differences test the derivative against the applied linear Hamiltonian for
both cells, unequal hoppings, OBC and several cutoffs. No separate half/full
correction should be added to the off-diagonal one-body term.

The Hamiltonian identity still uses the fields actually applied to each MPS.
The cell energy uses current simultaneous correlations, including before
convergence. These roles are distinct during a raw iteration. Cell totals
are stored together with energies per physical site: divide by 2L for one
ladder and 4L for two. Target-density corrections sum each ladder's mu times
its residual particle-number error. The common field-independent perturbative
offset remains omitted, as in the existing model; for the equal-hop comparison
its value per site is the same in both finite cells.

Acceptance requires every ladder to pass the existing field, slow-mode,
channel-window, density, inner-DMRG, energy and Hamiltonian-identity gates
in the same sweep. A stationary A/B pattern is period one of the cell
iteration. A temporal two-cycle is saved as diagnostic evidence and remains
unaccepted. No iteration alternation, phase average or weakened gate converts
it into a stationary spatial solution.

`trellis_mps_mft_state` stores MPS objects under `ladders/A/psi` and, for the
two-ladder cell, `ladders/B/psi`. The existing compact mirror recursively
removes them and keeps every history. Resuming requires the complete cell
checkpoint and its SHA-256. The existing branch comparator can rank accepted
seeds within the same cell fingerprint; comparison across cell ansatzes
requires explicit per-site normalization and profile/embedding assessment.

## Plotting saved MF histories

The Phase 1 plotting adapter reads each trellis ladder's nested fields and
complete history directly, including its embedded initial seed. From the
`ladder_mps_mft` directory:

```julia
include("plot_phase1_mf_observables.jl")
plot_phase1_mf_profiles_and_middle_histories(state_file)             # ladder A
plot_phase1_mf_profiles_and_middle_histories(state_file; ladder=:B)  # two-ladder cell
plot_phase1_seed_profiles(state_file; ladder=:A)
```

For two-ladder states, inspect A and B separately; they are spatial states,
not alternate iterations. Titles identify the cell and selected ladder.
The default includes the seed as plotted iteration 1, so a 60-sweep state has
61 slider positions. Use `include_seed=false` to show only the 60 measured
records. No external seed or `parent_path` is needed for these saved states.

These are MF-field plots, consistent with the existing square/cubic adapter.
Trellis `mu_cdw` contains normal-bond contributions as well as densities, and
its pairing kernel mixes bond channels. For physical density, spin and pair
profiles, use `ladders/A/history/correlations` (or B), as in the
[September 18 physical-correlation analysis](reports/trellis_progress_20260918/README.md).
