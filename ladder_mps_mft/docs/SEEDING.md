# Independent-field seeding

Independent starts support two explicit protocols. Existing configurations that
omit `initial_seed_protocol` retain the refactor's established `legacy`
behavior exactly.

## Legacy protocol

`initial_seed_protocol = "legacy"` reproduces the refactor's established
pre-matched-mode seeds:

- pairing fills every retained real `alpha` entry with independent uniform
  noise and then enforces the stored transpose symmetry;
- SDW uses a deterministic rung- and leg-staggered Hartree field; and
- CDW uses a deterministic rung-staggered, leg-even Hartree field.

This path is required to reproduce prior campaigns, but it is not a
channel-equivalent basin probe. In particular, its pairing seed contains many
center-of-mass wavevectors and sign changes while its Hartree seeds do not.
The name of this compatibility protocol must not be confused with an exact
reconstruction of the original monolithic driver's fresh-run pairing source.

## Matched-mode protocol

`initial_seed_protocol = "matched_mode"` maps declared real spatial templates
into normalized field sources. The elementary single-channel profile for a
ladder of `L` rungs is

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

### Harmonic stripe and coexistence seeds

`initial_seed = "stripe"` constructs one combined SDW/CDW texture rather than
two unrelated Hartree seeds. Here `initial_mode_number=m` denotes the slow
signed spin-envelope mode. The spin source has both longitudinal and
transverse antiferromagnetic staggering,

```text
s(i,leg) = (-1)^(i-1) (-1)^(leg-1) g_m(i),
```

so its high finite-ladder mode is `n_s=L-1-m`. The leg-even charge source uses
the locked second harmonic `g_2m(i)` with doubled phase, giving
`n_c=2m`. Charge maxima then coincide with maxima of `abs(s)`, while charge
minima coincide with spin antiphase nodes. The configuration requires
`m>0`, `2m<=L-1`, and `initial_leg_parity="auto"` because the mixed parity is
part of the stripe definition.

Spin and charge arrays are separately normalized in the full-field metric
before applying `initial_stripe_charge_to_spin_ratio`; the complete seed is
then normalized to `initial_amplitude`. Therefore the ratio is a source-norm
ratio, not an assumed equality between source and observable amplitudes.

`initial_seed = "stripe_pairing"` adds a separately normalized uniform pairing
template, selected by `initial_pairing_form_factor`, before the same final
normalization. `initial_stripe_pairing_to_spin_ratio` declares its norm relative
to the spin source. A pure `stripe` seed is a useful normal, `alpha=0`
symmetry-subspace control. It cannot by itself test whether pairing coexists,
because an exactly number-conserving Hamiltonian can preserve `alpha=0` under
the map. At least one `stripe_pairing` start is therefore required for an
unrestricted coexistence comparison; pairing is allowed to decay if it is not
supported.

For the square reconnaissance, the predeclared bank is `m=4,5`, phase zero,
charge:spin ratio `0.2`, and pairing:spin ratio `1.0` in coexistence starts.
At `L=64` these give `(n_s,n_c)=(59,8)` and `(58,10)`. Both modes are run before
examining their new energies.

### Legacy-like translation-invariant pairing control

The original monolithic fresh-run code did not initialize random `alpha` and
`beta` fields. It set `beta=0` and `mu_cdw=0`, then drew one Gaussian `alpha`
coefficient per relative rung-offset/leg-pair class and copied it along every
allowed rung. Its randomness therefore mixed relative pairing form factors but
introduced no center-of-mass phase slips or domain walls.

`initial_seed = "legacy_pairing"` reconstructs that structure under
`initial_seed_protocol = "matched_mode"`. It uses a dedicated field RNG stream
keyed by `random_seed`, leaves `beta` and `mu_cdw` exactly zero, and normalizes
the complete `alpha` field to `initial_amplitude`. Because each coefficient is
already constant along the ladder, no extra spatial filter is applied. The
matched norm deliberately differs from the original driver's
`1e-3 * (2 t_perp^2 / E_p)` per-coefficient scale; this branch tests the legacy
basin geometry without giving it a different source strength.

This is a broadband *relative-form-factor* pairing control, not a broadband
spatial seed and not an independent SDW/CDW control. The square representative
bank runs it alongside the explicit uniform d-wave source so survival of a
pairing basin can be checked against pairing-form-factor initialization.

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
unbiased. Basin-accessibility work must predeclare a small physically related
mode/phase bank, include unrestricted multi-channel starts whenever a
single-channel seed defines an invariant symmetry subspace, and retain every
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
initial fields remain stored in `fields/initial`. Schema-v7 artifacts also
store that exact field under `history/fields/seed` with
`history/fields/seed_iteration=0`; the complete-history plotting adapter
prepends it by default.
