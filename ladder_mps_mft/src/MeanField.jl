function make_sites(model::ModelSettings, runtime::RuntimeSettings=RuntimeSettings())
    return siteinds(
        "Electron",
        2 * model.L;
        conserve_sz=runtime.conserve_sz,
        conserve_nfparity=runtime.conserve_nfparity,
    )
end

const MATCHED_PAIRING_FORM_FACTORS = (:onsite_s, :rung_s, :leg_s, :extended_s, :d_wave)

function resolved_initial_leg_parity(seed::Symbol, requested::Symbol)
    requested in (:auto, :even, :odd) || throw(ArgumentError(
        "initial leg parity must be auto, even, or odd",
    ))
    requested != :auto && return requested
    seed in (:stripe, :stripe_pairing) && return :mixed_even_charge_odd_spin
    return seed == :sdw ? :odd : :even
end

function initial_mode_wavevector_pi(model::ModelSettings, mode_number::Integer)
    0 <= mode_number <= model.L - 1 || throw(ArgumentError(
        "initial mode number must lie between 0 and L-1",
    ))
    return Float64(mode_number) / (model.L - 1)
end

function _matched_mode_offset(model::ModelSettings, mode_number::Integer, phase_pi::Real)
    mode_number == 0 && return 0.0
    q_pi = initial_mode_wavevector_pi(model, mode_number)
    return mean(cospi(q_pi * (rung - 1) + phase_pi) for rung in 1:model.L)
end

function _matched_mode_value(
    model::ModelSettings,
    position::Real,
    mode_number::Integer,
    phase_pi::Real,
    offset::Real,
)
    q_pi = initial_mode_wavevector_pi(model, mode_number)
    return cospi(q_pi * (position - 1) + phase_pi) - offset
end

"""Return the mean-controlled rung profile used by the opt-in matched-mode seed."""
function matched_mode_profile(
    model::ModelSettings;
    mode_number::Integer=0,
    phase_pi::Real=0.0,
)
    isfinite(phase_pi) || throw(ArgumentError("initial mode phase must be finite"))
    offset = _matched_mode_offset(model, mode_number, phase_pi)
    return [
        _matched_mode_value(model, rung, mode_number, phase_pi, offset)
        for rung in 1:model.L
    ]
end

function _set_symmetric_alpha!(alpha, left::Int, right::Int, leg::Int, other_leg::Int, value::Real)
    alpha[left, right, leg, other_leg] = value
    alpha[right, left, other_leg, leg] = value
    return alpha
end

function _matched_pairing_template!(
    alpha,
    model::ModelSettings,
    rung_profile,
    mode_number::Integer,
    phase_pi::Real,
    form_factor::Symbol,
)
    form_factor in MATCHED_PAIRING_FORM_FACTORS || throw(ArgumentError(
        "unknown matched pairing form factor '$form_factor'",
    ))
    offset = _matched_mode_offset(model, mode_number, phase_pi)

    if form_factor == :onsite_s
        for rung in 1:model.L, leg in 1:2
            _set_symmetric_alpha!(alpha, rung, rung, leg, leg, rung_profile[rung])
        end
        return alpha
    elseif form_factor == :rung_s
        for rung in 1:model.L
            _set_symmetric_alpha!(alpha, rung, rung, 1, 2, rung_profile[rung])
        end
        return alpha
    end

    model.r_range >= 1 || throw(ArgumentError(
        "matched pairing form factor '$form_factor' requires model.r_range >= 1",
    ))
    for rung in 1:(model.L - 1)
        bond_value = _matched_mode_value(
            model,
            rung + 0.5,
            mode_number,
            phase_pi,
            offset,
        )
        for leg in 1:2
            _set_symmetric_alpha!(alpha, rung, rung + 1, leg, leg, bond_value)
        end
    end
    form_factor == :leg_s && return alpha

    rung_sign = form_factor == :d_wave ? -1.0 : 1.0
    for rung in 1:model.L
        _set_symmetric_alpha!(alpha, rung, rung, 1, 2, rung_sign * rung_profile[rung])
    end
    return alpha
end

function field_l2_per_physical_site(fields::FieldState, model::ModelSettings)
    squared_norm = sum(abs2, fields.alpha) + sum(abs2, fields.beta) + sum(abs2, fields.mu_cdw)
    return sqrt(squared_norm / (2 * model.L))
end

function _normalize_matched_seed!(fields::FieldState, model::ModelSettings, amplitude::Real)
    target = Float64(amplitude)
    if iszero(target)
        fill!(fields.alpha, 0.0)
        fill!(fields.beta, 0.0)
        fill!(fields.mu_cdw, 0.0)
        return fields
    end
    current = field_l2_per_physical_site(fields, model)
    current > eps(Float64) || throw(ArgumentError(
        "the requested matched-mode seed is identically zero; choose another mode or phase",
    ))
    scale = target / current
    fields.alpha .*= scale
    fields.beta .*= scale
    fields.mu_cdw .*= scale
    return fields
end

function _normalize_component_per_site!(component, model::ModelSettings)
    current = sqrt(sum(abs2, component) / (2 * model.L))
    current > eps(Float64) || throw(ArgumentError(
        "the requested stripe component is identically zero",
    ))
    component ./= current
    return component
end

"""
Construct a harmonically locked ladder stripe source.

`mode_number=m` is the slow signed SDW-envelope mode. The physical spin source
has longitudinal and transverse antiferromagnetic staggering, so its dominant
high wavevector is `(L-1-m)/(L-1) * pi`. The leg-even charge source uses the
second harmonic `2m`, placing charge maxima at maxima of the absolute spin
envelope and charge minima at its antiphase nodes. Spin, charge, and optional
uniform pairing templates are normalized separately before their declared
ratios are applied; the complete field is normalized once more by the caller.
"""
function _matched_stripe_template!(
    fields::FieldState,
    model::ModelSettings;
    mode_number::Integer,
    mode_phase_pi::Real,
    charge_to_spin_ratio::Real,
    pairing_to_spin_ratio::Real,
    pairing_form_factor::Symbol,
)
    1 <= mode_number || throw(ArgumentError("stripe envelope mode must be positive"))
    2 * mode_number <= model.L - 1 || throw(ArgumentError(
        "stripe charge harmonic 2m must not exceed L-1",
    ))
    charge_to_spin_ratio > 0 || throw(ArgumentError(
        "stripe charge-to-spin ratio must be positive",
    ))
    pairing_to_spin_ratio >= 0 || throw(ArgumentError(
        "stripe pairing-to-spin ratio must be nonnegative",
    ))

    spin_profile = matched_mode_profile(
        model;
        mode_number,
        phase_pi=mode_phase_pi,
    )
    charge_profile = matched_mode_profile(
        model;
        mode_number=2 * mode_number,
        phase_pi=2 * mode_phase_pi,
    )
    spin_mu = zeros(Float64, size(fields.mu_cdw))
    charge_mu = zeros(Float64, size(fields.mu_cdw))
    for rung in 1:model.L, leg in 1:2
        site = rung_leg_to_site(rung, leg - 1)
        longitudinal_sign = isodd(rung - 1) ? -1.0 : 1.0
        transverse_sign = leg == 2 ? -1.0 : 1.0
        spin_value = spin_profile[rung] * longitudinal_sign * transverse_sign
        charge_value = charge_profile[rung]
        spin_mu[1, site] = spin_value
        spin_mu[2, site] = -spin_value
        charge_mu[:, site] .= charge_value
    end
    _normalize_component_per_site!(spin_mu, model)
    _normalize_component_per_site!(charge_mu, model)
    fields.mu_cdw .= spin_mu .+ charge_to_spin_ratio .* charge_mu

    if pairing_to_spin_ratio > 0
        uniform_profile = matched_mode_profile(model; mode_number=0, phase_pi=0.0)
        _matched_pairing_template!(
            fields.alpha,
            model,
            uniform_profile,
            0,
            0.0,
            pairing_form_factor,
        )
        _normalize_component_per_site!(fields.alpha, model)
        fields.alpha .*= pairing_to_spin_ratio
    end
    return fields
end

"""
Reconstruct the spatial structure of the legacy fresh-run pairing source.

The legacy script drew one Gaussian coefficient for each relative rung offset
and ordered leg-pair class, then copied that coefficient along every allowed
center-of-mass position. Thus the source is random across relative pairing
form factors but already maximally smoothed (translation-invariant away from
open-boundary truncation) along the ladder. The refactored control uses a
dedicated RNG stream and the common matched full-field normalization.
"""
function _legacy_pairing_template!(
    alpha,
    model::ModelSettings,
    random_seed::Integer,
)
    rng = MersenneTwister(random_seed)
    for offset in 0:model.r_range, leg in 1:2, other_leg in 1:2
        offset == 0 && other_leg < leg && continue
        coefficient = randn(rng)
        for rung in 1:(model.L - offset)
            _set_symmetric_alpha!(
                alpha,
                rung,
                rung + offset,
                leg,
                other_leg,
                coefficient,
            )
        end
    end
    return alpha
end

function _legacy_initial_fields(
    model::ModelSettings;
    seed::Symbol=:pairing,
    amplitude::Real=1e-3,
    rng=MersenneTwister(1),
)
    alpha = zeros(Float64, model.L, model.L, 2, 2)
    beta = zeros(Float64, 2, model.L, model.L, 2, 2)
    mu_cdw = zeros(Float64, 2, 2 * model.L)
    if seed == :pairing
        for rung in 1:model.L, offset in 0:model.r_range
            target = rung + offset
            target <= model.L || continue
            for leg in 1:2, other_leg in 1:2
                offset == 0 && other_leg < leg && continue
                value = amplitude * (2rand(rng) - 1)
                alpha[rung, target, leg, other_leg] = value
                alpha[target, rung, other_leg, leg] = value
            end
        end
    elseif seed == :sdw
        for rung in 1:model.L, leg in 0:1
            site = rung_leg_to_site(rung, leg)
            phase = isodd(rung + leg) ? -1.0 : 1.0
            mu_cdw[1, site] = amplitude * phase
            mu_cdw[2, site] = -amplitude * phase
        end
    elseif seed == :cdw
        for site in 1:(2 * model.L)
            phase = isodd(div(site - 1, 2)) ? 1.0 : -1.0
            mu_cdw[:, site] .= amplitude * phase
        end
    elseif seed != :zero
        throw(ArgumentError("unknown seed '$seed'"))
    end
    return FieldState(alpha, beta, mu_cdw)
end

function _matched_mode_initial_fields(
    model::ModelSettings;
    seed::Symbol,
    amplitude::Real,
    mode_number::Integer,
    mode_phase_pi::Real,
    pairing_form_factor::Symbol,
    leg_parity::Symbol,
    stripe_charge_to_spin_ratio::Real,
    stripe_pairing_to_spin_ratio::Real,
    random_seed::Integer,
)
    alpha = zeros(Float64, model.L, model.L, 2, 2)
    beta = zeros(Float64, 2, model.L, model.L, 2, 2)
    mu_cdw = zeros(Float64, 2, 2 * model.L)
    fields = FieldState(alpha, beta, mu_cdw)
    seed == :zero && return fields

    profile = matched_mode_profile(
        model;
        mode_number,
        phase_pi=mode_phase_pi,
    )
    parity = resolved_initial_leg_parity(seed, leg_parity)
    if seed == :pairing
        _matched_pairing_template!(
            alpha,
            model,
            profile,
            mode_number,
            mode_phase_pi,
            pairing_form_factor,
        )
    elseif seed == :legacy_pairing
        mode_number == 0 || throw(ArgumentError(
            "legacy_pairing requires mode_number=0",
        ))
        leg_parity == :auto || throw(ArgumentError(
            "legacy_pairing draws all leg-pair classes; use auto leg parity",
        ))
        _legacy_pairing_template!(alpha, model, random_seed)
    elseif seed == :sdw || seed == :cdw
        seed == :cdw && parity == :even && mode_number == 0 && amplitude > 0 &&
            throw(ArgumentError(
                "uniform even-parity matched CDW is redundant with chemical-potential targeting",
            ))
        for rung in 1:model.L, leg in 1:2
            site = rung_leg_to_site(rung, leg - 1)
            transverse_sign = parity == :odd && leg == 2 ? -1.0 : 1.0
            value = profile[rung] * transverse_sign
            if seed == :sdw
                mu_cdw[1, site] = value
                mu_cdw[2, site] = -value
            else
                mu_cdw[:, site] .= value
            end
        end
    elseif seed in (:stripe, :stripe_pairing)
        leg_parity == :auto || throw(ArgumentError(
            "stripe seeds fix odd spin and even charge leg parity; use auto",
        ))
        _matched_stripe_template!(
            fields,
            model;
            mode_number,
            mode_phase_pi,
            charge_to_spin_ratio=stripe_charge_to_spin_ratio,
            pairing_to_spin_ratio=seed == :stripe_pairing ?
                stripe_pairing_to_spin_ratio : 0.0,
            pairing_form_factor,
        )
    else
        throw(ArgumentError("unknown seed '$seed'"))
    end
    return _normalize_matched_seed!(fields, model, amplitude)
end

"""
Construct initial mean fields.

`protocol=:legacy` retains the refactor's pre-existing random-pairing and
deterministic staggered-Hartree behavior. Under `protocol=:matched_mode`, the
special `legacy_pairing` channel reconstructs the actual legacy fresh-run
structure (one random coefficient per relative bond/leg class, constant along
the ladder). Other matched channels map explicit finite-ladder templates into
a selected channel or harmonically locked stripe/stripe-pairing field. Every
nonzero matched seed is normalized so
`field_l2_per_physical_site(fields, model) == amplitude`.
"""
function initial_fields(
    model::ModelSettings;
    seed::Symbol=:pairing,
    amplitude::Real=1e-3,
    rng=MersenneTwister(1),
    protocol::Symbol=:legacy,
    mode_number::Integer=0,
    mode_phase_pi::Real=0.0,
    pairing_form_factor::Symbol=:onsite_s,
    leg_parity::Symbol=:auto,
    stripe_charge_to_spin_ratio::Real=0.2,
    stripe_pairing_to_spin_ratio::Real=1.0,
    random_seed::Integer=1,
)
    isfinite(amplitude) && amplitude >= 0 || throw(ArgumentError(
        "initial amplitude must be finite and nonnegative",
    ))
    protocol == :legacy && return _legacy_initial_fields(model; seed, amplitude, rng)
    protocol == :matched_mode || throw(ArgumentError("unknown initial seed protocol '$protocol'"))
    return _matched_mode_initial_fields(
        model;
        seed,
        amplitude,
        mode_number,
        mode_phase_pi,
        pairing_form_factor,
        leg_parity,
        stripe_charge_to_spin_ratio,
        stripe_pairing_to_spin_ratio,
        random_seed,
    )
end

function initial_seed_metadata(model::ModelSettings, run::RunSettings)
    matched = run.initial_seed_protocol == :matched_mode
    stripe = run.initial_seed in (:stripe, :stripe_pairing)
    legacy_pairing = run.initial_seed == :legacy_pairing
    resolved_leg_parity = run.initial_seed in (:sdw, :cdw, :stripe, :stripe_pairing) ?
        resolved_initial_leg_parity(run.initial_seed, run.initial_leg_parity) : :not_applicable
    stripe_envelope_mode = stripe ? run.initial_mode_number : 0
    stripe_spin_mode = stripe ? model.L - 1 - stripe_envelope_mode : 0
    stripe_charge_mode = stripe ? 2 * stripe_envelope_mode : 0
    return (
        protocol=run.initial_seed_protocol,
        mode_number=run.initial_mode_number,
        mode_wavevector_pi=initial_mode_wavevector_pi(model, run.initial_mode_number),
        mode_phase_pi=run.initial_mode_phase_pi,
        mode_basis=stripe ?
            "antiferromagnetic stripe with mean-controlled envelope and charge second harmonic" :
            (legacy_pairing ?
                "legacy translation-invariant random relative-bond pairing mixture" :
                (matched ? "mean-controlled finite-open-ladder cosine" : "legacy")),
        pairing_form_factor=legacy_pairing ?
            :mixed_relative_bond_classes : run.initial_pairing_form_factor,
        requested_leg_parity=run.initial_leg_parity,
        resolved_leg_parity,
        normalization=matched ? "full_field_l2_per_sqrt_physical_site" : "legacy_per_entry",
        target_field_l2_per_physical_site=matched && run.initial_seed != :zero ?
            run.initial_amplitude : NaN,
        stripe_envelope_mode_number=stripe_envelope_mode,
        stripe_spin_mode_number=stripe_spin_mode,
        stripe_spin_wavevector_pi=stripe ? initial_mode_wavevector_pi(model, stripe_spin_mode) : NaN,
        stripe_charge_mode_number=stripe_charge_mode,
        stripe_charge_wavevector_pi=stripe ? initial_mode_wavevector_pi(model, stripe_charge_mode) : NaN,
        stripe_charge_to_spin_ratio=stripe ? run.initial_stripe_charge_to_spin_ratio : NaN,
        stripe_pairing_to_spin_ratio=run.initial_seed == :stripe_pairing ?
            run.initial_stripe_pairing_to_spin_ratio : 0.0,
        legacy_pairing_random_seed=legacy_pairing ? run.random_seed : 0,
        legacy_pairing_center_of_mass_structure=legacy_pairing ?
            "constant_by_relative_offset_and_leg_pair" : "not_applicable",
        legacy_pairing_beta_initialization=legacy_pairing ? "zero" : "not_applicable",
        legacy_pairing_mu_cdw_initialization=legacy_pairing ? "zero" : "not_applicable",
    )
end

function configure_threading!(runtime::RuntimeSettings)
    BLAS.set_num_threads(runtime.blas_threads)
    try
        ITensors.NDTensors.Strided.set_num_threads(runtime.strided_threads)
    catch err
        @warn "could not set ITensor Strided threads" exception=(err, catch_backtrace())
    end
    blocksparse = try
        runtime.threaded_blocksparse ? ITensors.enable_threaded_blocksparse() : ITensors.disable_threaded_blocksparse()
        ITensors.using_threaded_blocksparse()
    catch err
        @warn "could not configure ITensor block-sparse threading" exception=(err, catch_backtrace())
        false
    end
    strided = try
        ITensors.NDTensors.Strided.get_num_threads()
    catch
        runtime.strided_threads
    end
    return (; julia=Threads.nthreads(), blas=BLAS.get_num_threads(), strided, blocksparse)
end

function build_mf_mpo(
    sites,
    model::ModelSettings,
    fields::FieldState,
    chemical_potential::Real;
    backend::Union{Symbol,RuntimeSettings}=:cpu,
)
    os = OpSum()
    for site in 1:(2 * model.L)
        add!(os, -chemical_potential, "Ntot", site)
        add!(os, fields.mu_cdw[1, site], "Ndn", site)
        add!(os, fields.mu_cdw[2, site], "Nup", site)
        add!(os, model.U, "Nupdn", site)
    end
    for rung in 1:(model.L - 1), leg in 0:1
        left = rung_leg_to_site(rung, leg)
        right = rung_leg_to_site(rung + 1, leg)
        for spin in ("up", "dn")
            add!(os, -model.t, "Cdag$spin", left, "C$spin", right)
            add!(os, -model.t, "Cdag$spin", right, "C$spin", left)
        end
    end
    for rung in 1:model.L
        left = rung_leg_to_site(rung, 0)
        right = rung_leg_to_site(rung, 1)
        for spin in ("up", "dn")
            add!(os, -model.t0, "Cdag$spin", left, "C$spin", right)
            add!(os, -model.t0, "Cdag$spin", right, "C$spin", left)
        end
    end
    if model.V != 0
        for rung in 1:(model.L - 1), leg in 0:1
            left = rung_leg_to_site(rung, leg)
            right = rung_leg_to_site(rung + 1, leg)
            add!(os, model.V, "Ntot", left, "Ntot", right)
        end
        for rung in 1:model.L
            add!(os, model.V, "Ntot", rung_leg_to_site(rung, 0), "Ntot", rung_leg_to_site(rung, 1))
        end
    end
    for i in 1:model.L, ip in 1:model.L
        abs(i - ip) <= model.r_range || continue
        for leg in 0:1, other_leg in 0:1
            site_i = rung_leg_to_site(i, leg)
            site_ip = rung_leg_to_site(ip, other_leg)
            alpha = fields.alpha[i, ip, leg + 1, other_leg + 1]
            if alpha != 0
                add!(os, -alpha, "Cup", site_ip, "Cdn", site_i)
                add!(os, -alpha, "Cdagdn", site_i, "Cdagup", site_ip)
            end
            site_i == site_ip && continue
            beta_down = fields.beta[1, i, ip, leg + 1, other_leg + 1]
            beta_up = fields.beta[2, i, ip, leg + 1, other_leg + 1]
            beta_down != 0 && add!(os, beta_down, "Cdagdn", site_i, "Cdn", site_ip)
            beta_up != 0 && add!(os, beta_up, "Cdagup", site_i, "Cup", site_ip)
        end
    end
    return move_to_backend(MPO(os, sites), backend)
end

function _threshold(value::Real, threshold::Real)
    return abs(value) > threshold ? Float64(value) : 0.0
end

function mean_fields_from_correlations(
    correlations::CorrelationState,
    model::ModelSettings;
    threshold::Real=0.0,
)
    pair = correlations.pair
    exchange_down = correlations.exchange_down
    exchange_up = correlations.exchange_up
    density_down = correlations.density_down
    density_up = correlations.density_up
    prefactor = 2 * model.tp^2 / model.ep
    alpha = zeros(Float64, model.L, model.L, 2, 2)
    beta = zeros(Float64, 2, model.L, model.L, 2, 2)
    for i in 1:model.L, ip in 1:model.L
        abs(i - ip) <= model.r_range || continue
        i0, i1 = rung_leg_to_site(i, 0), rung_leg_to_site(i, 1)
        ip0, ip1 = rung_leg_to_site(ip, 0), rung_leg_to_site(ip, 1)
        if model.geometry == :cubic_frustrated
            avals = (
                prefactor * (pair[ip1, i1] + 2 * pair[ip0, i0]),
                prefactor * (pair[ip0, i0] + 2 * pair[ip1, i1]),
                2 * prefactor * pair[ip0, i1],
                2 * prefactor * pair[ip1, i0],
            )
            bvals_down = (
                prefactor * (exchange_down[i1, ip1] + 2 * exchange_down[i0, ip0]),
                prefactor * (exchange_down[i0, ip0] + 2 * exchange_down[i1, ip1]),
                2 * prefactor * exchange_down[ip0, i1],
                2 * prefactor * exchange_down[ip1, i0],
            )
            bvals_up = (
                prefactor * (exchange_up[i1, ip1] + 2 * exchange_up[i0, ip0]),
                prefactor * (exchange_up[i0, ip0] + 2 * exchange_up[i1, ip1]),
                2 * prefactor * exchange_up[ip0, i1],
                2 * prefactor * exchange_up[ip1, i0],
            )
        elseif model.geometry == :cubic_unfrustrated
            avals = (3 * prefactor * pair[ip1, i1], 3 * prefactor * pair[ip0, i0], 2 * prefactor * pair[ip1, i0], 2 * prefactor * pair[ip0, i1])
            bvals_down = (3 * prefactor * exchange_down[i1, ip1], 3 * prefactor * exchange_down[i0, ip0], 2 * prefactor * exchange_down[ip1, i0], 2 * prefactor * exchange_down[ip0, i1])
            bvals_up = (3 * prefactor * exchange_up[i1, ip1], 3 * prefactor * exchange_up[i0, ip0], 2 * prefactor * exchange_up[ip1, i0], 2 * prefactor * exchange_up[ip0, i1])
        else
            avals = (prefactor * pair[ip1, i1], prefactor * pair[ip0, i0], 0.0, 0.0)
            bvals_down = (prefactor * exchange_down[i1, ip1], prefactor * exchange_down[i0, ip0], 0.0, 0.0)
            bvals_up = (prefactor * exchange_up[i1, ip1], prefactor * exchange_up[i0, ip0], 0.0, 0.0)
        end
        for (index, (leg, other_leg)) in enumerate(((1, 1), (2, 2), (2, 1), (1, 2)))
            alpha[i, ip, leg, other_leg] = _threshold(avals[index], threshold)
            beta[1, i, ip, leg, other_leg] = _threshold(bvals_down[index], threshold)
            beta[2, i, ip, leg, other_leg] = _threshold(bvals_up[index], threshold)
            if rung_leg_to_site(i, leg - 1) == rung_leg_to_site(ip, other_leg - 1)
                beta[:, i, ip, leg, other_leg] .= 0.0
            end
        end
    end
    mu_cdw = zeros(Float64, 2, 2 * model.L)
    kernel = density_kernel(model.geometry, model.tp, model.ep)
    for rung in 1:model.L
        sites = [rung_leg_to_site(rung, 0), rung_leg_to_site(rung, 1)]
        mu_cdw[1, sites] .= kernel * (density_down[sites] .- 0.5)
        mu_cdw[2, sites] .= kernel * (density_up[sites] .- 0.5)
    end
    mu_cdw[abs.(mu_cdw) .<= threshold] .= 0.0
    return FieldState(alpha, beta, mu_cdw)
end

function calculate_mean_fields(psi::MPS, model::ModelSettings; threshold::Real=0.0)
    pair = real.(correlation_matrix(psi, "Cup", "Cdn"))
    exchange_down = real.(correlation_matrix(psi, "Cdagdn", "Cdn"))
    exchange_up = real.(correlation_matrix(psi, "Cdagup", "Cup"))
    density_down = real.(expect(psi, "Ndn"))
    density_up = real.(expect(psi, "Nup"))
    correlations = CorrelationState(pair, exchange_down, exchange_up, density_down, density_up)
    return mean_fields_from_correlations(correlations, model; threshold), correlations
end
