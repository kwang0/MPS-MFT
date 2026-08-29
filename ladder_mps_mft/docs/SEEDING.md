# Independent-field seeding

Independent starts support two explicit protocols. Existing configurations that
omit `initial_seed_protocol` retain `legacy` behavior exactly.

## Legacy protocol

`initial_seed_protocol = "legacy"` reproduces the established seeds:

- pairing fills every retained real `alpha` entry with independent uniform
  noise and then enforces the stored transpose symmetry;
- SDW uses a deterministic rung- and leg-staggered Hartree field; and
- CDW uses a deterministic rung-staggered, leg-even Hartree field.

This path is required to reproduce prior campaigns, but it is not a
channel-equivalent basin probe. In particular, its pairing seed contains many
center-of-mass wavevectors and sign changes while its Hartree seeds do not.

## Matched-mode protocol

`initial_seed_protocol = "matched_mode"` maps one declared real spatial mode
into exactly one order channel. For a ladder of `L` rungs,

```text
q / pi = initial_mode_number / (L - 1)
g(i)   = cos(pi * ((q/pi) * (i - 1) + initial_mode_phase_pi)).
```

Nonzero modes are mean-controlled on the rung grid so a small accidental
uniform component is not injected. Pair fields on leg bonds evaluate the same
mode at the bond midpoint. The complete stored field vector is then rescaled
to satisfy

```text
sqrt(sum(abs2, alpha) + sum(abs2, beta) + sum(abs2, mu_cdw)) / (2L)
    == initial_amplitude.
```

Thus pairing, SDW, and CDW starts have identical total source strength in the
same field-vector metric even though the number of nonzero tensor entries and
their componentwise maxima differ. Transpose-related pairing entries count in
the norm because they are also present in the solver's mixing vector.

The available pairing templates are `onsite_s`, `rung_s`, `leg_s`,
`extended_s`, and `d_wave`. Hartree seeds use `initial_leg_parity = "even"` or
`"odd"`; `"auto"` resolves to odd for SDW and even for CDW. A uniform,
leg-even CDW source is rejected because it is redundant with chemical-potential
targeting.

Example `[run]` entries are:

```toml
initial_seed = "pairing"
initial_amplitude = 1.0e-3
initial_seed_protocol = "matched_mode"
initial_mode_number = 10
initial_mode_phase_pi = 0.0
initial_pairing_form_factor = "d_wave"
initial_leg_parity = "auto"
random_seed = 404
```

With a matched-mode base config, `scripts/prepare_branch_scan.jl` gives the SC,
SDW, and CDW branches the same `random_seed`, mode, phase, and norm. The common
random seed makes their independent product-MPS start identical. The field
template itself is deterministic. Inspect any generated config without a DMRG
solve:

```bash
julia --project=. scripts/inspect_initial_seed.jl \
  path/to/branch.toml path/to/branch_seed_profile.tsv
```

The TSV is lightweight and contains charge/spin leg-parity profiles and the
five pairing proxies. The command consumes no node-hours.

The guarded standard Phase 1 preparation path also honors a matched-mode base:
within each transverse geometry, `scripts/prepare_phase1_gpu.jl` assigns all
three channels the same product-state seed and records the protocol plus seed
fingerprint in `manifest.tsv`. The checked-in Phase 1 base remains legacy, so
existing plans and prepared campaigns do not change implicitly. Preparation is
still separate from smoke and matrix submission, and all ledger checks remain
authoritative on Perlmutter.

## What “fair” does and does not mean

The matched protocol removes the known roughness and norm asymmetry between
channels. It does not make one chosen wavevector or pairing form factor
unbiased. Basin-accessibility work must predeclare a small common mode/phase
bank, run every enabled order channel through that bank, and retain every
accepted distinct solution. Mode or phase choices may not be selected after
looking at which branch has the lowest energy.

Seed settings have their own `initial_seed_fingerprint` and full provenance.
They deliberately remain outside `numerical_fingerprint`, because branches
started from different seeds must still be comparable after convergence. The
model, numerical, implementation, pair-binding-source, acceptance, and
same-geometry gates remain unchanged. A seed is only an initial condition; it
never waives raw-map recurrence validation or the canonical variational-energy
comparison.

Parent, resume, and field-inheritance configurations reuse their declared
lineage state rather than constructing an independent seed. Their exact applied
initial fields remain stored in `fields/initial`.
