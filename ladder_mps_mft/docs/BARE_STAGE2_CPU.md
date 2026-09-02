# Bare-ladder Stage 2 projected-response CPU pilot

This workflow implements Stage 2 of the hybrid response search described in
the linked linear-response notes. It consumes the completed `L=64`, `V=0`,
`t0=1.4` backbone and Stage 1 covariance screen and computes the true
finite-field ladder response in a small, physics-informed subspace.

The Stage 1 analysis and the decision to proceed are recorded in
`docs/reports/bare_stage1_t014_v0_20260902/REPORT.md` and the self-contained
`report.html` beside it.

Codex never authenticates to NERSC/Perlmutter, transfers this repository, or
operates Slurm. Every command below is a handoff for the user to run after
synchronizing the repository. The user also performs all status checks, result
synchronization, live accounting, and cancellation.

## Scientific design

The named bank contains the 11 physically motivated directions from the
survey plus three additions chosen from the Stage 1 real-space covariance
screen:

- charge even modes 7, 8, and 9;
- spin odd modes 58, 59, and 63;
- uniform onsite, rung, leg, extended-s, and d-wave pairing names;
- Stage 1 rank-1 charge-even, charge-odd, and spin-even covariance vectors.

All vectors use the same field metric and are orthonormalized with two-pass
modified Gram--Schmidt. The 14 names yield 12 independent fields: nine normal
and three pairing. At q=0, extended-s and d-wave are exact linear combinations
of the onsite/rung/leg span. Their names and expansion coefficients remain in
the candidate-bank provenance; eliminating their duplicate columns does not
remove either form factor from the projected eigenvectors.

Discovery measures one finite-difference action for every independent field,

```text
Z = chi_ladder Q,
chi_Q = Q' Z,
J_geometry,Q = Q' K_geometry Z.
```

The expensive ladder responses are computed once and reused for the cubic
frustrated, cubic unfrustrated, and square kernels. The complete projected
matrices are diagonalized, so the candidate templates may mix where the
symmetries permit; the run does not interpret 12 independent scalar Rayleigh
quotients as eigenvalues. Response outside the candidate span, including the
exchange-field component, is stored as a leakage diagnostic for the operator
review before validation.

The geometry prefactor uses the hole pair binding measured by the new backbone,
not the older registry value used to initialize the model metadata. Both the
signed value and its exact HDF5 source path are stored in the candidate bank and
discovery artifact.

## Representation-matched zero-field references

The saved backbone state is a valid Stage 1 state but is not a sufficiently
strict finite-difference baseline at `h=1e-4`: a shared residual relaxation
would be divided by `h` and contaminate every response column. Stage 2 first
re-relaxes the number-conserving state at `chi=1200` with the response solver's
`1e-9` energy tolerance.

Normal probes preserve `Nf`, `NfParity`, and `Sz` and subtract observables from
that strict reference. The pairing branch then calls `removeqn(psi, "Nf")`,
preserving `NfParity` and `Sz`, and solves the parity-only Hamiltonian again at
zero field. Every pairing probe subtracts this representation-matched
reference. The source term and conjugate expectation are generated through the
same `build_mf_mpo` convention, including its signs and orientations.

## Discovery acceptance gates

The discovery assembly is scientifically accepted only when:

- every strict reference and finite-field DMRG solve passes the configured
  sweep, energy, discarded-weight, and time gates;
- the projected raw susceptibility satisfies the 5 percent reciprocity gate
  independently in the normal and pairing blocks;
- normal/pair response leakage is at most 5 percent, as required by U(1) at the
  unpaired reference; and
- every probe, candidate bank, reference, and source backbone has consistent
  model, implementation, bank, and SHA-256 provenance.

The code saves the measured unsymmetrized matrix. It only writes a symmetric
copy after measuring and passing reciprocity; it never hides a failure by
symmetrizing first.

After those gates pass, discovery selects up to three distinct leading
eigenvectors across the three geometry kernels. Validation is deliberately a
separate submission. Each selected vector is solved at `h=1e-4` and `h/2=5e-5`.
The two derivatives must agree within 5 percent before the code writes the
Richardson estimate `2 D(h/2) - D(h)`.

## Job graph and CPU allocation

```text
prepare
   |
strict normal h=0 reference
   |-------------------------|
normal probes [1:9]          parity-only pair h=0 reference
                             |
                             pair probes [1:3]
   |-------------------------|
discovery assembly + compact mirror

operator review
   |
validation probes [1:3], each h then h/2
   |
validation assembly + compact mirror
```

The DMRG jobs retain the repository's measured `blocksparse-t4` topology:
four Julia block-sparse threads, eight Slurm logical CPUs, one BLAS thread, and
one Strided thread. The 12 probe solves are independent Slurm array elements,
which is the useful level of parallelism. Increasing intra-solve threads beyond
four would be an unbenchmarked allocation change; the Phase 0 matrix showed
that 8 and 16 block-sparse threads were slower on its screening payload.

Normal jobs request 48 GiB and 12 hours. Pairing jobs request 64 GiB because
removing `Nf` merges blocks and raises memory pressure. Validation requests 24
hours because each array element runs two amplitudes sequentially. Preparation
and assembly use one Julia thread and request 8 GiB; they stream or copy HDF5
objects and do not materialize the large MPS tensors. No allocated DMRG job waits for another
probe; `Dependency` jobs remain pending without running until their parents
succeed.

The launcher's memory- and walltime-based conservative ceilings are:

| action | maximum CPU node-hours |
|---|---:|
| discovery, including both zero-field references | 18.65625 |
| optional three-mode validation | 9.609375 |

These are reservation bounds, not predicted or measured charges. The Stage 1
sync did not include `sacct` or `/usr/bin/time -v`, so the current memory values
remain conservative. After this pilot, the user should sync scheduler elapsed,
`AllocCPUS`, `TotalCPU`, and `MaxRSS` before changing memory or thread defaults.

## User-run Perlmutter handoff

Start every command from the persistent checkout path:

```bash
cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"
```

After synchronizing these changes, inspect the discovery plan without making a
scheduler change:

```bash
bash slurm/bare_stage2_cpu.sh plan 20260901_bare_t014_v0_stage1
```

The plan verifies that the Stage 1 control directory points to a scratch tree
containing both `backbone.h5` and `stage1.h5`, prints the exact job topology,
and refuses a reservation above the 24-node-hour discovery cap.

Submission is explicit:

```bash
bash slurm/bare_stage2_cpu.sh submit-discovery \
  20260902_bare_t014_v0_stage2 \
  20260901_bare_t014_v0_stage1
```

Inspect the scheduler state or the compact discovery summary with:

```bash
bash slurm/bare_stage2_cpu.sh status 20260902_bare_t014_v0_stage2
bash slurm/bare_stage2_cpu.sh show 20260902_bare_t014_v0_stage2
```

The expected control output is
`output/bare_stage2/20260902_bare_t014_v0_stage2/`. Full MPS references and
probe states live under the matching `$PSCRATCH` tree; `stateless_results/` is
the compact CFS mirror intended for synchronization back here.

Do not submit validation merely because discovery completed. First synchronize
the compact discovery result and analyze its spectrum, reciprocity error,
cross-block leakage, convergence, and geometry dependence. If that review
supports the three proposed modes, the user-run commands are:

```bash
bash slurm/bare_stage2_cpu.sh plan-validation \
  20260902_bare_t014_v0_stage2
bash slurm/bare_stage2_cpu.sh submit-validation \
  20260902_bare_t014_v0_stage2
```

Stage 3 random residual probes are not part of this launcher. They should be
added only if the measured Stage 2 leakage, near-degeneracy, or geometry
dependence shows that the 12-dimensional candidate span is inadequate.

## Local validation boundary

Windows is suitable for package compilation, unit tests, candidate-bank
construction from the synchronized compact Stage 1 artifacts, and the tiny
end-to-end fixture. The fixture uses a deliberately tiny ladder and loose gates
to test mechanics; its response spectrum and reciprocity numbers are not
physical evidence.

The local suite validates:

- exact removal of the two dependent pair labels;
- metric orthogonality of the resulting 12-direction production bank;
- equivalence of the refactored all-geometry map to the prior map;
- equality between each field's conjugate expectation and the Hamiltonian MPO
  convention;
- number-conserving and parity-only reference/probe workflows; and
- discovery, validation selection, two-amplitude checks, and Richardson
  assembly on the tiny fixture.
