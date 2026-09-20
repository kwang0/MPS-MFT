# Transverse stripes, leg parity, and MF iteration cycles

20 September 2026. This interprets the four synchronized trellis states and
the existing square protocol; no new simulation is involved.

The two-ladder trellis retains frustration. Its striped trajectories carry
a large charge component odd between spatial ladders, while charge on the
two legs of each ladder is almost identical. This motivates a targeted A/B
test at the paired square points. It does not establish that those paired
states are artifacts of the one-ladder calculation.

## Alternating legs and alternating ladders are different modes

Write charge deviations as `(A0,A1 ; B0,B1)` at a common physical longitudinal
phase, where A/B are spatial ladders and 0/1 their legs:

- Leg-odd charge: `(rho,-rho ; rho,-rho)`.
- Leg-even charge alternating between ladders: `(rho,rho ; -rho,-rho)`.

Here rho varies along the ladder; these are local deviations, not different
average fillings. The second pattern preserves the same charge on both sites
of each rung. Its transverse period spans four legs. Thus `k_y=pi` in the old
**two-leg Fourier basis** and an odd A/B component in a **two-ladder cell**
are different labels. A microscopic momentum also needs the leg basis and
trellis offsets.

The solver permits leg-odd charge. It measures each site's density
independently and does not average leg densities before updating the MF
fields. Rung-averaged plotting hides that component but does not remove it
from the calculation. The square kernel's leg-swap matrix has both even and
odd eigenvectors. The tested seeds are not a dedicated stability search in
every leg-odd direction.

The two modes need not have comparable susceptibilities. In the rung
bonding/antibonding basis, summed over spin,

```text
n0+n1 = n_bonding+n_antibonding
n0-n1 = c_bonding^dagger c_antibonding + h.c.
```

Leg-odd charge polarizes the rung and mixes these sectors. Leg-even charge
can modulate total rung density while retaining internal charge symmetry.
Strong rung hopping and correlated rung structure can therefore give the
two channels different responses. This is a qualitative mechanism, not a
measured relative-charge gap or a proof that leg-odd CDW is impossible.
Leg-odd spin is a separate channel.

Direct central-rung charge measurements show the distinction in these runs:

| Cell / seed / spatial ladder | std[(n0+n1)/2] | RMS[(n0-n1)/2] |
|---|---:|---:|
| One / stripe / A | 0.00157175 | 1.0723e-5 |
| One / pairing / A | 0.00157257 | 1.0719e-5 |
| Two / stripe / A | 0.04684307 | 5.8781e-6 |
| Two / stripe / B | 0.04666694 | 6.4857e-6 |
| Two / pairing / A | 0.04698354 | 5.0058e-6 |
| Two / pairing / B | 0.04667287 | 6.5061e-6 |

These use rungs 17-48 at sweep 60. The observed two-ladder charge stripe is
overwhelmingly leg-even. Small observed leg-odd charge alone does not establish
stability against an explicitly applied leg-odd perturbation.

## Frustration and the trellis repetition convention

Both implementations retain tau0 and tau1 zigzag hoppings. Together with
an intraladder leg bond, they form triangles. Enlarging the MF cell removes
none of these links. Independent spatial profiles can relieve some ordering
constraints without eliminating the geometric frustration. Two separate
two-leg MPSs remain coupled by MF; this is not one quantum four-leg MPS.

The skew one-ladder map uses backward T^T on target leg 0 and forward T on
target leg 1, where T=tau0 I+tau1 S and S shifts one rung. The rectangular
two-ladder map at equal hoppings uses forward T on both A legs and backward
T^T on both B legs. B has physical longitudinal origin -1/2 rung. Repetition
and finite open-end cuts both change, so the comparison does not isolate
cell size within one identical finite ansatz.

In an infinite-ladder coordinate illustration, put x=i-m/2 for skew ladder
index m. A repeated raw-index harmonic exp(i q i) then has physical phase
exp(i q x) exp(i q m/2). A common representative profile ties longitudinal
modulation to transverse registration. A period-two sequence in skew
coordinates still repeats after the sheared translation (-1,2); the
rectangular A/B cell repeats after (0,2). At nonzero q these restrictions
differ. A temporal two-cycle of the skew map therefore need not reproduce
the rectangular A/B solution.

## Physical meaning of the different outcomes

The relevant freedom is the relative registration of order on adjacent
ladders. Moving every stripe by the same distance changes its absolute
position (and can matter at open ends). Moving A relative to B changes which
charge maxima and spin domains face each other and therefore changes the
interladder energy even in the bulk. A two-ladder cell tests this physical
degree of freedom, beyond the original concern about numerical translation
drift. Cell dependence should have been part of the initial phase-selection
qualification, before treating one-ladder pairing as a geometry-level result.

For a stationary skew one-ladder harmonic, the physical stripe phase advances
by q/2 per neighboring ladder; at q=pi/8 this is 11.25 degrees. Independent
rectangular A/B profiles can choose a different relative phase. The measured
endpoint charge phases reported below are about 125-131 degrees depending on
seed/window, and remain nonstationary. They illustrate a different stripe
arrangement, not a demonstrated optimal phase angle. A skew temporal
two-cycle still imposes its own repetition and does not remove this distinction.

This restriction affects the competing orders differently. For homogeneous
bulk pairing, correlations depend on rung separation and commute with
longitudinal translations. Forward/backward trellis kernels then agree in
the infinite bulk. A stripe with a nonzero longitudinal wavevector is
sensitive to their directional shifts. Restricting its transverse alignment
can therefore disadvantage it relative to uniform pairing. The detailed
instability also involves the ladder's spin/charge response and normal-bond
feedback; no susceptibility calculation here proves which channel initiates
the change. The strong central stripes show the observed difference extends
into the bulk, although open ends could still select or nucleate that branch.

For phase competition, independent A/B profiles should be part of the minimum
comparison. This is a better test of possible broken transverse translation,
not a new microscopic lattice, removal of frustration, or an improvement of
interladder quantum entanglement. The two existing finite-cell ansatzes are
not strictly nested for arbitrary profiles, and both still constrain longer
transverse patterns. Their current nonconverged endpoints do not certify
which phase minimizes the functional, but their actual trial energies can
be compared after evaluating both in the same cell, as done below.

The focused discriminating test is to initialize the rectangular cell from
the actual paired trellis endpoint, let its boundary fields adjust, and test
weak relative stripe perturbations. Compare any surviving paired branch
with a stationary stripe branch in that same cell and energy convention.
Growth from small perturbations supports local instability of the paired
branch; decay with a separate stable stripe branch indicates competing basins
whose converged energies decide preference. A strict one-to-two-cell
variational inequality instead requires an extension with matching repetition
and boundaries, rather than assuming the current rectangular cell contains
every finite skew one-ladder state.

## Direct energy comparison in the rectangular cell

Different restrictions do not prohibit comparing specified trial states.
The September 20 audit evaluates the product of two identical copies of
each saved paired endpoint in the actual rectangular A/B functional. Stored
bare-ladder expectation values and correlation matrices suffice: all
interaction fields are recomputed for that cell. This is an energy
evaluation, not a new stationary paired solution or MPS optimization.

The two paired-copy energies are -0.518817872689 and -0.518817859078 t/site,
versus -0.521125816792 and -0.521055560230 for the two saved striped
endpoints. Reembedding raises each paired energy by only 2.46378e-6 t/site.
The same-cell stripe advantage is 0.002238-0.002308 t/site, about 900 times
larger. Target-density corrections are below 8.33e-6 t/site and do not
change the ordering; the uncorrected canonical gaps are 0.002229-0.002301.
Residual density differences are small but this tangent correction is not
an exact particle-number projection or rigorous fixed-density error bound.

Thus a lower-energy striped trial state has been found in a common finite
functional. Large residuals qualify stationarity and optimality, not the
existence of this lower trial energy. Nonnesting prevents an automatic
inequality between separately optimized ansatz families; it does not erase
the measured comparison. The preferred converged stripe arrangement, a
possibly distinct optimized paired branch, and larger-cell minima remain
open. The original accepted=false flags are preserved.

The [numeric audit](same_cell_energy_audit_20260920.toml) records source and
kernel hashes, original and embedded energy components, density corrections
and all four pairwise gaps. Recomputed original fields agree exactly and
energies agree within 1e-12 t/site. Reproduce from the repository root:

```powershell
julia --startup-file=no --compiled-modules=existing --project=ladder_mps_mft ladder_mps_mft/scripts/audit_trellis_same_cell_energy_20260920.jl
```

## Measured A/B charge Fourier sectors

At the dominant charge harmonic q=2 pi (4/64)=pi/8, use

```text
n_l(q) = mean_i[(n_l(i)-mean(n_l))*exp(-i*q*x_l(i))]
x_A(i)=i, x_B(i)=i-1/2
n_even=(n_A+n_B)/2; n_odd=(n_A-n_B)/2
w_odd=|n_odd|^2/(|n_even|^2+|n_odd|^2).
```

This A/B decomposition fixes the physical longitudinal phase and uses
simultaneous spatial profiles. It is independent of MF sweep parity.

| Two-ladder seed | Full-window odd weight | Central-window odd weight | Central B-minus-A phase |
|---|---:|---:|---:|
| Stripe | 81.46% | 82.98% | 131.27 degrees |
| Pairing | 78.81% | 79.45% | 126.08 degrees |

B's origin correction multiplies its raw-index coefficient by exp(i q/2).
Both even and odd components remain: neither endpoint is a pure pi phase
shift. Over the last ten sweeps, the central odd weights span 76.3-88.7%
and 71.6-85.5%, respectively. Large alternating relaxation persists. These
are finite, unaccepted textures, not a locked ordering wavevector; OBC
Fourier components need not be exact translation eigenstates.

## Diagonal stripes and larger transverse cells

The user's bottom-right slide sketch suggests diagonal or oblique
hole-rich walls separating antiphase magnetic domains. This is a plausible
competitor, not an orientation established by the two-ladder data. At the
measured charge harmonic, repeating A/B gives phase changes +theta, -theta,
+theta, -theta. With theta about 126-131 degrees, it does not continue a
uniform phase advance across successive ladders. Such repetition may
describe staggered or zigzag wall registration. The special phases 0 and
pi can close a constant advance in one or two ladders, respectively.

For an ideal bulk texture, write the charge modulation at physical x as

```text
delta n_m(x) = rho cos(q_x x + m theta),  lambda_c = 2 pi / q_x.
```

An ordinary rectangular n-ladder cell requires n theta = 2 pi p. If the
wall shifts by d rungs per ladder, theta = -q_x d, so n d must be an integer
multiple of lambda_c. Thus n depends on the tilt, not just the longitudinal
period, and one ladder contains two microscopic rows. The spin texture
must also close. For a rigidly translated pattern with nominal charge/spin
periods 16/32 and d=2 rungs per ladder, charge closes after eight ladders
but the full spin texture requires sixteen, absent an additional spin
transformation. These are illustrative closure conditions, not inferred
optimal cell sizes; the present OBC walls are not exactly periodic.

More generally every ordering harmonic must obey Q dot T = 2 pi times an
integer for the actual cell translation T. This includes the trellis
half-rung registration and the internal leg/spin basis. A translated or
"screw" cell can represent some tilted patterns with fewer independent
ladders by combining a transverse step with a longitudinal shift. The
existing skew cell already imposes one particular registration; it does
not explore arbitrary tilts. OBC makes a new translated boundary a kernel
and endpoint problem, not simply a circular roll of saved arrays.

The next useful distinction is between a preferred A/B registration and
a stripe that would lower its energy by continuing a phase advance through
a larger cell. A controlled search over registrations and modest cell
sizes can identify candidate tilts before committing to a large commensurate
cell. No larger-cell implementation or campaign is prepared here.

The slide cites Miyazaki, Yanagisawa and Yamaji, JPSJ 73, 1643 (2004).
Their [primary paper](https://staff.aist.go.jp/t-yanagisawa/activity/JPSJ-Miyazaki04.pdf)
finds bond-centered diagonal stripes favored over vertical stripes around
hole doping 1/16 in a square-lattice VMC calculation with U=8 and t'=-0.2.
The matching doping motivates the hypothesis, but its hopping geometry and
variational treatment differ from this weakly coupled trellis model.

## Relation to Bollmark's two-period construction

Bollmark, Koehler and Kantian describe repulsive transverse density
feedback for which consecutive MF solutions avoid one another's density
maxima. A two-cycle encodes the checkerboard-like competitor in their
chain-array calculation. See Sec. III A and Fig. 2 of the
[accessible preprint](https://arxiv.org/pdf/2301.08116), associated with
[Phys. Rev. B 111, 125141 (2025)](https://doi.org/10.1103/PhysRevB.111.125141).

The mathematical condition is explicit. If R is the raw one-subsystem
response, including neighbor-field construction, and the physical bipartite
array uses the same neighbor map on both sublattices, then

```text
X_A=R(X_B), X_B=R(X_A)
```

defines both a stationary A/B state and a two-cycle of R. This is static
order, not physical time evolution. A numerical cycle alone does not verify
the spatial equations. In particular the different trellis repetition
conventions require their actual forward/backward kernels.

Trellis accepts stationary cells only but continues raw updates; its
acceptance gate does not project oscillations away. The rectangular runs
already contain two spatial ladders and still alternate in iteration space.
Their damped transient cannot automatically be relabeled a stationary A/B
solution or evidence for a four-ladder phase.

## Implications for square and cubic

A correctly embedded square A/B extension with only opposite-sublattice
neighbors contains the existing identical-ladder fixed point exactly. For
a deterministic, accurately solved raw response with derivative J there,

```text
J_cell = [0 J; J 0].
```

A one-ladder eigenvalue lambda becomes +lambda for A/B-even perturbations
and -lambda for A/B-odd ones. R composed with itself instead has eigenvalue
lambda^2. Complete raw-map linear stability therefore implies stability to
both spatial parities in this particular extension. Doubling alone does not
create a linear instability hidden from a fully tested raw map.

The square anchors and remainder used damping=1, no Anderson, and
accepted_periods=[1,2]. Both seed families approach pairing at (1.4,-0.4)
and (1.4,-0.2). These trajectories provide useful evidence but are neither
a complete Jacobian measurement nor an exclusion of a finite-amplitude
metastable stripe basin. Run length, seeded modes and DMRG resolution matter.

Recommended first comparison: these two square points, each starting from
pairing plus weak A/B-odd charge/spin perturbations and from a finite-amplitude
stripe with a longitudinally translated partner. Preserve chi, L, density
and target-coupling reconstruction, and let all channels evolve independently.
Leg-odd charge is a useful separate perturbation. A stripe point such as
(1.4,0) can subsequently check registration and energy. This is a targeted
proposal, not a prepared or authorized full-grid rerun.

Cubic unfrustrated is already striped throughout, so a cell test is lower
priority for the paired/stripe boundary but may change registration,
coexistence or energy. Its kernel sums several physical neighbor directions:
specify the sublattice of each before extending it. The legacy frustrated
connectivity may require more than a generic two-sublattice assignment.

Implementation prerequisite: verify that A=B in the square spatial map
reproduces the existing fields and per-site functional. Merely selecting
rectangular trellis and setting tau1=0 is not that check: its
C=tau1 I+tau0 S interface retains a shifted tau0 bond. The documented
tau1=0 square limit belongs to the skew one-ladder map.

## Provenance and validation

The [numeric audit](transverse_sectors_20260920.json) records all four compact
source paths and SHA-256 values, endpoint/window results, and complete
60-sweep Fourier histories. Reproduce locally from the repository root:

```powershell
C:/Python313/python.exe -B -X utf8 ladder_mps_mft/scripts/analyze_trellis_transverse_sectors_20260920.py
```

Checks cover hashes before/after, density normalization, endpoint/history
equality, the origin correction, even/odd Parseval identity and pure
even/odd algebraic examples. Acceptance flags remain false. No MPS
optimization, susceptibility measurement, geometry change, correlation
backfill, budget change or Perlmutter action was performed. Kernel analysis
uses src/Trellis.jl, src/MeanField.jl, src/Geometry.jl and the trellis method
contract. Square controls are read from the synchronized September 8 anchor
and September 10 remainder configs, including both paired-point families.
