using Test
using HDF5
using ITensorMPS
using LadderMPSMFT
using LinearAlgebra
using Random
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))

function test_model(; geometry=:cubic_frustrated)
    return ModelSettings(;
        L=2,
        U=2.0,
        t0=1.0,
        tp=0.1,
        density=1.0,
        r_range=1,
        geometry,
        ep=0.2,
        ep_signed=-0.2,
        ep_source="synthetic",
    )
end

function test_fields(value=0.0)
    alpha = zeros(2, 2, 2, 2)
    beta = zeros(2, 2, 2, 2, 2)
    mu = zeros(2, 4)
    mu[1, 1] = value
    return FieldState(alpha, beta, mu)
end

function test_correlations()
    return CorrelationState(zeros(4, 4), zeros(4, 4), zeros(4, 4), fill(0.5, 4), fill(0.5, 4))
end

function test_energy(value=0.0)
    return EnergyBreakdown(
        effective_eigenvalue=value,
        effective_expectation=value,
        effective_eigenvalue_error=0.0,
        bare_ladder_energy=value,
        reconstructed_bare_ladder_energy=value,
        hamiltonian_identity_error=0.0,
        chemical_potential_term=0.0,
        pair_field_energy=0.0,
        exchange_field_energy=0.0,
        density_field_energy=0.0,
        pair_transverse_energy=0.0,
        exchange_transverse_energy=0.0,
        density_transverse_energy=0.0,
        double_counting_correction=0.0,
        reconstructed_variational_energy=value,
        direct_variational_energy=value,
        variational_consistency_error=0.0,
        canonical_variational_energy=value,
        target_density_correction=0.0,
        target_density_corrected_variational_energy=value,
        grand_potential=value,
    )
end

function test_record(iteration, applied, measured; energy=0.0, density=1.0, update_mode=:unknown, correlations=test_correlations())
    absolute, relative = LadderMPSMFT.hybrid_distance(measured, applied)
    return IterationRecord(;
        iteration=iteration,
        update_mode,
        applied,
        measured,
        correlations,
        density,
        chemical_potential=0.0,
        mu_search_status=:density_tolerance,
        mu_evaluations=1,
        mu_density_converged=true,
        effective_energy=energy,
        variational=test_energy(energy),
        field_abs_residual=absolute,
        field_rel_residual=relative,
        wall_seconds=0.1,
        dmrg_max_discarded_weight=1.0e-9,
        dmrg_maxlinkdim=4,
        dmrg_sweep_energies=[Float64(energy)],
        dmrg_sweep_max_discarded_weights=[1.0e-9],
        dmrg_sweep_maxlinkdims=[4],
    )
end

@testset "geometry and E_p registry" begin
    @test normalize_geometry("cubic-frustrated") == :cubic_frustrated
    @test normalize_geometry(:square) == :square
    @test_throws ArgumentError normalize_geometry(:triangular)
    @test density_kernel(:cubic_frustrated, 0.1, 0.2) ≈ [0.2 0.1; 0.1 0.2]
    @test density_kernel(:cubic_unfrustrated, 0.1, 0.2) ≈ [0.0 0.3; 0.3 0.0]
    @test density_kernel(:square, 0.1, 0.2) ≈ [0.0 0.1; 0.1 0.0]

    registry = joinpath(ROOT, "data", "E_p_values.csv")
    selection = lookup_ep(registry; L=64, U=8, V=0, t0=1, density=0.9375, tp=0.1)
    @test selection.record.chi == 1000
    @test selection.record.E_p ≈ -0.13251724
    @test selection.denominator ≈ 0.13251724
    @test selection.bound_pair
    @test selection.tp_below_pair_binding
    @test selection.mode == :exact
    @test selection.lower_record === selection.upper_record
    @test_throws ArgumentError lookup_ep(registry; L=64, U=8, V=0, t0=2, density=0.9375, tp=0.1)

    interpolated = lookup_ep(
        registry;
        L=64,
        U=8,
        V=-0.2,
        t0=1.1,
        density=0.9375,
        tp=0.1,
        allow_interpolation=true,
    )
    @test interpolated.mode == :linear_t0
    @test interpolated.interpolation_weight ≈ 0.5
    @test interpolated.lower_record.t0 ≈ 1.0
    @test interpolated.upper_record.t0 ≈ 1.2
    @test interpolated.record.E_p ≈ -0.18452309659153343
    @test interpolated.denominator ≈ 0.18452309659153343
    @test_throws ArgumentError lookup_ep(
        registry;
        L=64, U=8, V=-0.2, t0=0.9, density=0.9375, tp=0.1,
        allow_interpolation=true,
    )

    sign_changing = [
        EpRecord(4, 2.0, 0.0, 1.0, 1.0, 10, -1.0, -0.1, 1e-6),
        EpRecord(4, 2.0, 0.0, 1.2, 1.0, 10, -1.1, 0.1, 1e-6),
    ]
    @test_throws ArgumentError lookup_ep(
        sign_changing;
        L=4, U=2, V=0, t0=1.1, density=1.0, tp=0.01,
        source_path="synthetic", require_bound=false, allow_interpolation=true,
    )
end

@testset "configuration and deterministic seeds" begin
    settings = load_settings(joinpath(ROOT, "configs", "phase0_timing.toml"))
    @test settings.model.geometry == :cubic_frustrated
    @test settings.model.ep_signed < 0
    @test settings.run.initial_seed == :pairing
    @test settings.run.initial_seed_protocol == :legacy
    @test settings.run.initial_mode_number == 0
    @test settings.run.initial_mode_phase_pi == 0.0
    @test settings.run.initial_pairing_form_factor == :onsite_s
    @test settings.run.initial_leg_parity == :auto
    @test settings.runtime.backend == :cpu
    @test settings.convergence.unmixed_cycle_probe
    @test settings.convergence.accepted_periods == [1, 2]
    @test settings.convergence.orbit_bulk_fraction == 0.5
    @test settings.model.mu_initial == 0.0
    @test settings.dmrg.mu_density_tol == 5e-4
    @test settings.dmrg.mu_max_iterations == 16
    @test settings.dmrg.mu_bracket_step == 0.05
    @test settings.dmrg.mu_warm_start_noise == 1e-8
    @test settings.convergence.period2_oscillation_cosine_max == -0.5
    @test settings.convergence.period2_two_step_ratio_max == 0.5
    @test settings.convergence.slow_mode_cosine_min == 0.9
    @test !settings.run.require_accepted_solution
    gpu_settings = load_settings(joinpath(ROOT, "configs", "phase1_gpu_base.toml"))
    @test gpu_settings.runtime.backend == :gpu
    @test gpu_settings.runtime.tensor_scalar_type == :float64
    @test !gpu_settings.runtime.conserve_sz
    @test !gpu_settings.runtime.conserve_nfparity
    @test gpu_settings.convergence.cycle_action == :continue
    @test gpu_settings.model.ep_mode == :linear_t0
    @test gpu_settings.model.ep_signed ≈ -0.18452309659153343
    @test gpu_settings.model.tp^2 / gpu_settings.model.ep ≈ 0.05419375777188662
    recurrence_settings = load_settings(joinpath(ROOT, "configs", "phase1_gpu_recurrence_chi400.toml"))
    @test recurrence_settings.model.geometry == :cubic_unfrustrated
    @test recurrence_settings.dmrg.maxdim == 400
    @test recurrence_settings.dmrg.nsweeps == 16
    @test recurrence_settings.convergence.cycle_action == :stop
    @test recurrence_settings.run.max_iterations == recurrence_settings.convergence.probe_iterations + 1
    conflicting_lineage = ProjectSettings(
        model=test_model(),
        run=RunSettings(
            inherit_from="legacy.h5",
            inherit_sha256=repeat("0", 64),
            parent_checkpoint="parent.h5",
            parent_sha256=repeat("1", 64),
        ),
    )
    @test_throws ArgumentError validate_settings(conflicting_lineage)
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=test_model(),
        run=RunSettings(parent_orbit_phase=1),
    ))
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=test_model(),
        run=RunSettings(
            parent_checkpoint="parent.h5",
            parent_sha256=repeat("0", 64),
            parent_orbit_phase=0,
        ),
    ))
    invalid_gpu = ProjectSettings(
        model=test_model(),
        runtime=RuntimeSettings(backend=:gpu, threaded_blocksparse=false),
    )
    @test_throws ArgumentError validate_settings(invalid_gpu)
    cpu_fingerprint = LadderMPSMFT.numerical_fingerprint(ProjectSettings(model=test_model()))
    dense_cpu_fingerprint = LadderMPSMFT.numerical_fingerprint(ProjectSettings(
        model=test_model(),
        runtime=RuntimeSettings(threaded_blocksparse=false, conserve_sz=false, conserve_nfparity=false),
    ))
    @test cpu_fingerprint != dense_cpu_fingerprint
    float32_fingerprint = LadderMPSMFT.numerical_fingerprint(ProjectSettings(
        model=test_model(),
        runtime=RuntimeSettings(tensor_scalar_type=:float32),
    ))
    @test cpu_fingerprint != float32_fingerprint
    performance_only_fingerprint = LadderMPSMFT.numerical_fingerprint(ProjectSettings(
        model=test_model(),
        dmrg=DMRGSettings(max_time_seconds=123.0, output_level=7),
    ))
    @test cpu_fingerprint == performance_only_fingerprint
    first_seed = initial_fields(test_model(); seed=:pairing, rng=MersenneTwister(7))
    second_seed = initial_fields(test_model(); seed=:pairing, rng=MersenneTwister(7))
    explicit_legacy_seed = initial_fields(
        test_model();
        seed=:pairing,
        rng=MersenneTwister(7),
        protocol=:legacy,
    )
    @test first_seed.alpha == second_seed.alpha
    @test first_seed.alpha == explicit_legacy_seed.alpha
    @test first_seed.alpha == permutedims(first_seed.alpha, (2, 1, 4, 3))
    @test all(iszero, first_seed.beta)
    sdw_seed = initial_fields(test_model(); seed=:sdw, amplitude=1.0)
    @test sdw_seed.mu_cdw[1, :] == [-1.0, 1.0, 1.0, -1.0]
    @test sdw_seed.mu_cdw[2, :] == -sdw_seed.mu_cdw[1, :]
    cdw_seed = initial_fields(test_model(); seed=:cdw, amplitude=1.0)
    @test cdw_seed.mu_cdw[1, :] == [-1.0, -1.0, 1.0, 1.0]
    @test cdw_seed.mu_cdw[2, :] == cdw_seed.mu_cdw[1, :]

    matched_model = ModelSettings(;
        L=8,
        U=2.0,
        t0=1.0,
        tp=0.1,
        density=1.0,
        r_range=2,
        geometry=:cubic_frustrated,
        ep=0.2,
        ep_signed=-0.2,
        ep_source="synthetic",
    )
    matched_kwargs = (
        amplitude=2.5e-3,
        protocol=:matched_mode,
        mode_number=2,
        mode_phase_pi=0.25,
    )
    pairing_matched = initial_fields(
        matched_model;
        seed=:pairing,
        pairing_form_factor=:d_wave,
        matched_kwargs...,
    )
    pairing_matched_other_rng = initial_fields(
        matched_model;
        seed=:pairing,
        pairing_form_factor=:d_wave,
        rng=MersenneTwister(999),
        matched_kwargs...,
    )
    extended_matched = initial_fields(
        matched_model;
        seed=:pairing,
        pairing_form_factor=:extended_s,
        matched_kwargs...,
    )
    sdw_matched = initial_fields(matched_model; seed=:sdw, matched_kwargs...)
    cdw_matched = initial_fields(matched_model; seed=:cdw, matched_kwargs...)
    for fields in (pairing_matched, sdw_matched, cdw_matched)
        @test field_l2_per_physical_site(fields, matched_model) ≈ matched_kwargs.amplitude
        @test all(iszero, fields.beta)
    end
    @test pairing_matched.alpha == pairing_matched_other_rng.alpha
    @test pairing_matched.alpha == permutedims(pairing_matched.alpha, (2, 1, 4, 3))
    @test all(
        extended_matched.alpha[rung, rung, 1, 2] ≈ -pairing_matched.alpha[rung, rung, 1, 2]
        for rung in 1:matched_model.L
    )
    @test all(
        extended_matched.alpha[rung, rung + 1, leg, leg] ≈
            pairing_matched.alpha[rung, rung + 1, leg, leg]
        for rung in 1:(matched_model.L - 1), leg in 1:2
    )
    @test all(iszero, pairing_matched.mu_cdw)
    @test any(value -> !iszero(value), pairing_matched.alpha)
    @test all(iszero, sdw_matched.alpha)
    @test all(iszero, cdw_matched.alpha)
    for rung in 1:matched_model.L
        first_site = rung_leg_to_site(rung, 0)
        second_site = rung_leg_to_site(rung, 1)
        @test sdw_matched.mu_cdw[1, second_site] ≈ -sdw_matched.mu_cdw[1, first_site]
        @test sdw_matched.mu_cdw[2, first_site] ≈ -sdw_matched.mu_cdw[1, first_site]
        @test cdw_matched.mu_cdw[1, second_site] ≈ cdw_matched.mu_cdw[1, first_site]
        @test cdw_matched.mu_cdw[2, first_site] ≈ cdw_matched.mu_cdw[1, first_site]
    end
    @test sum(matched_mode_profile(
        matched_model;
        mode_number=matched_kwargs.mode_number,
        phase_pi=matched_kwargs.mode_phase_pi,
    )) ≈ 0 atol=1e-14
    @test initial_mode_wavevector_pi(matched_model, 2) ≈ 2 / 7
    @test_throws ArgumentError initial_fields(
        matched_model;
        seed=:cdw,
        protocol=:matched_mode,
        mode_number=0,
    )

    legacy_pairing_matched = initial_fields(
        matched_model;
        seed=:legacy_pairing,
        amplitude=2.5e-3,
        protocol=:matched_mode,
        mode_number=0,
        random_seed=77,
    )
    legacy_pairing_matched_repeat = initial_fields(
        matched_model;
        seed=:legacy_pairing,
        amplitude=2.5e-3,
        protocol=:matched_mode,
        mode_number=0,
        random_seed=77,
    )
    legacy_pairing_matched_other_seed = initial_fields(
        matched_model;
        seed=:legacy_pairing,
        amplitude=2.5e-3,
        protocol=:matched_mode,
        mode_number=0,
        random_seed=78,
    )
    @test legacy_pairing_matched.alpha == legacy_pairing_matched_repeat.alpha
    @test legacy_pairing_matched.alpha != legacy_pairing_matched_other_seed.alpha
    @test legacy_pairing_matched.alpha ==
        permutedims(legacy_pairing_matched.alpha, (2, 1, 4, 3))
    @test field_l2_per_physical_site(legacy_pairing_matched, matched_model) ≈ 2.5e-3
    @test all(iszero, legacy_pairing_matched.beta)
    @test all(iszero, legacy_pairing_matched.mu_cdw)
    legacy_coefficients = Float64[]
    for offset in 0:matched_model.r_range, leg in 1:2, other_leg in 1:2
        offset == 0 && other_leg < leg && continue
        values = [
            legacy_pairing_matched.alpha[rung, rung + offset, leg, other_leg]
            for rung in 1:(matched_model.L - offset)
        ]
        @test all(value -> value == first(values), values)
        push!(legacy_coefficients, first(values))
    end
    @test length(unique(legacy_coefficients)) > 1

    stripe_matched = initial_fields(
        matched_model;
        seed=:stripe,
        protocol=:matched_mode,
        mode_number=2,
        mode_phase_pi=0.0,
        amplitude=2.5e-3,
        stripe_charge_to_spin_ratio=0.2,
        stripe_pairing_to_spin_ratio=0.0,
    )
    stripe_pairing_matched = initial_fields(
        matched_model;
        seed=:stripe_pairing,
        protocol=:matched_mode,
        mode_number=2,
        mode_phase_pi=0.0,
        amplitude=2.5e-3,
        pairing_form_factor=:d_wave,
        stripe_charge_to_spin_ratio=0.2,
        stripe_pairing_to_spin_ratio=1.0,
    )
    for fields in (stripe_matched, stripe_pairing_matched)
        @test field_l2_per_physical_site(fields, matched_model) ≈ 2.5e-3
        @test all(iszero, fields.beta)
    end
    @test all(iszero, stripe_matched.alpha)
    @test any(value -> !iszero(value), stripe_pairing_matched.alpha)
    function stripe_components(fields)
        charge = Float64[]
        spin_demodulated = Float64[]
        charge_mu = zeros(size(fields.mu_cdw))
        spin_mu = zeros(size(fields.mu_cdw))
        for rung in 1:matched_model.L, leg in 1:2
            site = rung_leg_to_site(rung, leg - 1)
            down = fields.mu_cdw[1, site]
            up = fields.mu_cdw[2, site]
            charge_value = (down + up) / 2
            spin_value = (down - up) / 2
            charge_mu[:, site] .= charge_value
            spin_mu[1, site] = spin_value
            spin_mu[2, site] = -spin_value
            if leg == 1
                push!(charge, charge_value)
                push!(spin_demodulated, spin_value * (isodd(rung - 1) ? -1 : 1))
            end
        end
        return (; charge, spin_demodulated, charge_mu, spin_mu)
    end
    stripe_components_only = stripe_components(stripe_matched)
    spin_profile = matched_mode_profile(matched_model; mode_number=2, phase_pi=0.0)
    charge_profile = matched_mode_profile(matched_model; mode_number=4, phase_pi=0.0)
    spin_scale = dot(stripe_components_only.spin_demodulated, spin_profile) / dot(spin_profile, spin_profile)
    charge_scale = dot(stripe_components_only.charge, charge_profile) / dot(charge_profile, charge_profile)
    @test stripe_components_only.spin_demodulated ≈ spin_scale .* spin_profile
    @test stripe_components_only.charge ≈ charge_scale .* charge_profile
    stripe_charge_norm = sqrt(sum(abs2, stripe_components_only.charge_mu) / (2 * matched_model.L))
    stripe_spin_norm = sqrt(sum(abs2, stripe_components_only.spin_mu) / (2 * matched_model.L))
    @test stripe_charge_norm / stripe_spin_norm ≈ 0.2
    mixed_components = stripe_components(stripe_pairing_matched)
    mixed_spin_norm = sqrt(sum(abs2, mixed_components.spin_mu) / (2 * matched_model.L))
    mixed_pairing_norm = sqrt(sum(abs2, stripe_pairing_matched.alpha) / (2 * matched_model.L))
    @test mixed_pairing_norm / mixed_spin_norm ≈ 1.0

    stripe_settings = ProjectSettings(
        model=matched_model,
        run=RunSettings(
            random_seed=404,
            initial_seed=:stripe,
            initial_seed_protocol=:matched_mode,
            initial_mode_number=2,
            initial_mode_phase_pi=0.0,
            initial_stripe_charge_to_spin_ratio=0.2,
            initial_stripe_pairing_to_spin_ratio=0.0,
        ),
    )
    validate_settings(stripe_settings)
    stripe_metadata = initial_seed_metadata(matched_model, stripe_settings.run)
    @test stripe_metadata.stripe_envelope_mode_number == 2
    @test stripe_metadata.stripe_spin_mode_number == 5
    @test stripe_metadata.stripe_charge_mode_number == 4
    @test stripe_metadata.stripe_spin_wavevector_pi ≈ 5 / 7
    @test stripe_metadata.stripe_charge_wavevector_pi ≈ 4 / 7
    @test stripe_metadata.resolved_leg_parity == :mixed_even_charge_odd_spin
    stripe_provenance = collect_provenance(stripe_settings)
    @test stripe_provenance["initial_stripe_spin_mode_number"] == 5
    @test stripe_provenance["initial_stripe_charge_mode_number"] == 4
    @test stripe_provenance["initial_stripe_charge_to_spin_ratio"] == 0.2
    @test initial_seed_fingerprint(stripe_settings) != initial_seed_fingerprint(ProjectSettings(
        model=matched_model,
        run=RunSettings(
            random_seed=404,
            initial_seed=:stripe_pairing,
            initial_seed_protocol=:matched_mode,
            initial_mode_number=2,
            initial_pairing_form_factor=:d_wave,
            initial_stripe_charge_to_spin_ratio=0.2,
            initial_stripe_pairing_to_spin_ratio=1.0,
        ),
    ))
    legacy_pairing_settings = ProjectSettings(
        model=matched_model,
        run=RunSettings(
            random_seed=77,
            initial_seed=:legacy_pairing,
            initial_seed_protocol=:matched_mode,
            initial_mode_number=0,
            initial_pairing_form_factor=:onsite_s,
        ),
    )
    validate_settings(legacy_pairing_settings)
    legacy_pairing_metadata = initial_seed_metadata(
        matched_model,
        legacy_pairing_settings.run,
    )
    @test legacy_pairing_metadata.pairing_form_factor == :mixed_relative_bond_classes
    @test legacy_pairing_metadata.resolved_leg_parity == :not_applicable
    @test legacy_pairing_metadata.legacy_pairing_random_seed == 77
    @test legacy_pairing_metadata.legacy_pairing_center_of_mass_structure ==
        "constant_by_relative_offset_and_leg_pair"
    legacy_pairing_provenance = collect_provenance(legacy_pairing_settings)
    @test legacy_pairing_provenance["initial_legacy_pairing_random_seed"] == 77
    @test legacy_pairing_provenance["initial_legacy_pairing_beta_initialization"] == "zero"
    @test initial_seed_fingerprint(legacy_pairing_settings) != initial_seed_fingerprint(
        ProjectSettings(
            model=matched_model,
            run=RunSettings(
                random_seed=78,
                initial_seed=:legacy_pairing,
                initial_seed_protocol=:matched_mode,
            ),
        ),
    )
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=matched_model,
        run=RunSettings(initial_seed=:legacy_pairing, initial_seed_protocol=:legacy),
    ))
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=matched_model,
        run=RunSettings(
            initial_seed=:legacy_pairing,
            initial_seed_protocol=:matched_mode,
            initial_mode_number=1,
        ),
    ))
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=matched_model,
        run=RunSettings(
            initial_seed=:legacy_pairing,
            initial_seed_protocol=:matched_mode,
            initial_pairing_form_factor=:d_wave,
        ),
    ))
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=matched_model,
        run=RunSettings(initial_seed=:stripe, initial_seed_protocol=:legacy, initial_mode_number=2),
    ))
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=matched_model,
        run=RunSettings(initial_seed=:stripe, initial_seed_protocol=:matched_mode, initial_mode_number=0),
    ))
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=matched_model,
        run=RunSettings(initial_seed=:stripe, initial_seed_protocol=:matched_mode, initial_mode_number=4),
    ))
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=matched_model,
        run=RunSettings(
            initial_seed=:stripe_pairing,
            initial_seed_protocol=:matched_mode,
            initial_mode_number=2,
            initial_stripe_pairing_to_spin_ratio=0.0,
        ),
    ))

    matched_settings = ProjectSettings(
        model=matched_model,
        run=RunSettings(
            random_seed=404,
            initial_seed=:pairing,
            initial_seed_protocol=:matched_mode,
            initial_mode_number=2,
            initial_mode_phase_pi=0.25,
            initial_pairing_form_factor=:d_wave,
        ),
    )
    validate_settings(matched_settings)
    @test LadderMPSMFT.numerical_fingerprint(matched_settings) == LadderMPSMFT.numerical_fingerprint(ProjectSettings(
        model=matched_model,
        run=RunSettings(random_seed=999, initial_seed=:cdw),
    ))
    @test initial_seed_fingerprint(matched_settings) != initial_seed_fingerprint(ProjectSettings(
        model=matched_model,
        run=RunSettings(random_seed=999, initial_seed=:cdw),
    ))
    @test initial_seed_fingerprint(matched_settings) == initial_seed_fingerprint(ProjectSettings(
        model=matched_model,
        run=RunSettings(
            random_seed=404,
            initial_seed=:pairing,
            initial_seed_protocol=:matched_mode,
            initial_mode_number=2,
            initial_mode_phase_pi=0.25,
            initial_pairing_form_factor=:d_wave,
            initial_leg_parity=:odd,
        ),
    ))
    provenance = collect_provenance(matched_settings)
    @test provenance["initial_seed_protocol"] == "matched_mode"
    @test provenance["initial_mode_number"] == 2
    @test provenance["initial_pairing_form_factor"] == "d_wave"
    @test provenance["initial_leg_parity_resolved"] == "not_applicable"
    @test provenance["initial_seed_normalization"] == "full_field_l2_per_sqrt_physical_site"
    @test provenance["initial_seed_fingerprint"] == initial_seed_fingerprint(matched_settings)

    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=matched_model,
        run=RunSettings(
            initial_seed=:cdw,
            initial_seed_protocol=:matched_mode,
            initial_mode_number=0,
        ),
    ))
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=matched_model,
        run=RunSettings(initial_pairing_form_factor=:unsupported),
    ))

    mktempdir() do directory
        raw = TOML.parsefile(joinpath(ROOT, "configs", "phase0_timing.toml"))
        run_raw = raw["run"]
        run_raw["initial_seed_protocol"] = "matched_mode"
        run_raw["initial_mode_number"] = 1
        run_raw["initial_mode_phase_pi"] = 0.25
        run_raw["initial_pairing_form_factor"] = "extended_s"
        run_raw["initial_leg_parity"] = "even"
        run_raw["initial_stripe_charge_to_spin_ratio"] = 0.25
        run_raw["initial_stripe_pairing_to_spin_ratio"] = 0.75
        path = joinpath(directory, "matched.toml")
        open(path, "w") do io
            TOML.print(io, raw; sorted=true)
        end
        loaded = load_settings(path)
        @test loaded.run.initial_seed_protocol == :matched_mode
        @test loaded.run.initial_mode_number == 1
        @test loaded.run.initial_mode_phase_pi == 0.25
        @test loaded.run.initial_pairing_form_factor == :extended_s
        @test loaded.run.initial_leg_parity == :even
        @test loaded.run.initial_stripe_charge_to_spin_ratio == 0.25
        @test loaded.run.initial_stripe_pairing_to_spin_ratio == 0.75

        scan_directory = joinpath(directory, "matched_scan")
        prepare_script = joinpath(ROOT, "scripts", "prepare_branch_scan.jl")
        run(`$(Base.julia_cmd()) --startup-file=no --project=$ROOT $prepare_script $path $scan_directory`)
        generated = [TOML.parsefile(joinpath(scan_directory, "$branch.toml")) for branch in ("sc", "sdw", "cdw")]
        @test [config["run"]["initial_seed"] for config in generated] == ["pairing", "sdw", "cdw"]
        @test all(config["run"]["random_seed"] == run_raw["random_seed"] for config in generated)
        @test all(config["run"]["initial_seed_protocol"] == "matched_mode" for config in generated)
        @test all(config["run"]["initial_mode_number"] == 1 for config in generated)
        @test occursin("does not sample wavevector", read(joinpath(
            scan_directory,
            "BRANCH_MANIFEST.md",
        ), String))

        profile_path = joinpath(directory, "seed_profile.tsv")
        inspect_script = joinpath(ROOT, "scripts", "inspect_initial_seed.jl")
        inspection = read(
            `$(Base.julia_cmd()) --startup-file=no --project=$ROOT $inspect_script $(joinpath(scan_directory, "sc.toml")) $profile_path`,
            String,
        )
        @test occursin("field_l2_per_physical_site=0.001", inspection)
        @test startswith(read(profile_path, String), "rung\tcharge_even\tcharge_odd")

        control_directory = joinpath(directory, "phase1_matched_control")
        full_directory = joinpath(directory, "phase1_matched_full")
        phase1_prepare_script = joinpath(ROOT, "scripts", "prepare_phase1_gpu.jl")
        run(`$(Base.julia_cmd()) --startup-file=no --project=$ROOT $phase1_prepare_script $path $control_directory $full_directory matched_test`)
        for geometry in ("frustrated", "unfrustrated", "square")
            configs = [TOML.parsefile(joinpath(
                control_directory,
                "configs",
                "$(geometry)__$(channel)_s1.segment-001.toml",
            )) for channel in ("pairing", "sdw", "cdw")]
            @test length(unique([config["run"]["random_seed"] for config in configs])) == 1
            @test all(config["run"]["initial_seed_protocol"] == "matched_mode" for config in configs)
        end
        manifest_header = split(first(readlines(joinpath(control_directory, "manifest.tsv"))), '\t')
        @test manifest_header[end - 1:end] == ["initial_seed_protocol", "initial_seed_fingerprint"]
    end
end

@testset "solver-only and full-tree fingerprints" begin
    mktempdir() do directory
        source_directory = joinpath(directory, "src")
        launcher_directory = joinpath(directory, "slurm")
        test_directory = joinpath(directory, "test")
        mkpath.((source_directory, launcher_directory, test_directory))
        manifest = joinpath(directory, "Manifest.toml")
        source = joinpath(source_directory, "Solver.jl")
        launcher = joinpath(launcher_directory, "run.sh")
        test_file = joinpath(test_directory, "runtests.jl")
        write(manifest, "manifest_format = \"2.0\"\n")
        write(source, "solver_value() = 1\n")
        write(launcher, "#!/bin/bash\necho first\n")
        write(test_file, "using Test\n@test true\n")

        implementation_before = LadderMPSMFT.implementation_fingerprint(directory)
        tree_before = LadderMPSMFT.tree_fingerprint(directory)
        write(launcher, "#!/bin/bash\necho second\n")
        @test LadderMPSMFT.implementation_fingerprint(directory) == implementation_before
        @test LadderMPSMFT.tree_fingerprint(directory) != tree_before
        write(test_file, "using Test\n@test 1 == 1\n")
        @test LadderMPSMFT.implementation_fingerprint(directory) == implementation_before
        write(source, "solver_value() = 2\n")
        @test LadderMPSMFT.implementation_fingerprint(directory) != implementation_before
    end
end

@testset "Phase 0 run environment round trip" begin
    script = joinpath(ROOT, "slurm", "phase0_calibrate_cpu.sh")
    if Sys.which("bash") === nothing
        @test_skip "bash-only Perlmutter launcher round trip"
    else
      mktempdir() do directory
        command = `bash -c 'source "$1" plan >/dev/null; write_environment "$2/run.env"; load_environment "$2"; printf "%s\n" "$PHASE0_RUN_SCRIPT_VERSION"' bash $script $directory`
        loaded_version = read(command, String)
        environment = read(joinpath(directory, "run.env"), String)
        @test occursin(r"(?m)^PHASE0_RUN_SCRIPT_VERSION=1\.3\.1$", environment)
        @test !occursin(r"(?m)^PHASE0_SCRIPT_VERSION=", environment)
        @test strip(loaded_version) == "1.3.1"

        compatible_environment = replace(environment,
            "PHASE0_RUN_SCRIPT_VERSION=1.3.1" => "PHASE0_RUN_SCRIPT_VERSION=1.3.0",
        )
        write(joinpath(directory, "run.env"), compatible_environment)
        compatible_version = read(
            `bash -c 'source "$1" plan >/dev/null; load_environment "$2"; printf "%s\n" "$PHASE0_RUN_SCRIPT_VERSION"' bash $script $directory`,
            String,
        )
        @test strip(compatible_version) == "1.3.0"

        script_source = read(script, String)
        @test occursin("submit_matrix_jobs \"\$run_dir\" \"\$seed_id\" pending", script_source)
        @test occursin("submit_matrix_jobs \"\$run_dir\" \"\$seed_id\" completed", script_source)
      end
    end
end

@testset "Phase 1 guarded GPU launcher" begin
    script = joinpath(ROOT, "slurm", "phase1_gpu.sh")
    preference_script = joinpath(ROOT, "scripts", "check_gpu_preferences.jl")
    gpu_project = TOML.parsefile(joinpath(ROOT, "gpu", "Project.toml"))
    migration_script = joinpath(ROOT, "slurm", "migrate_phase1_to_scratch.sh")
    corrupt_cleaner = joinpath(ROOT, "scripts", "prune_corrupt_auxiliary_hdf5.jl")
    script_source = read(script, String)
    preference_source = read(preference_script, String)
    migration_source = read(migration_script, String)
    cleaner_source = read(corrupt_cleaner, String)
    @test !occursin(r"(?m)^\s*module load cudatoolkit\s*$", script_source)
    @test occursin("module unload cudatoolkit", script_source)
    @test occursin("sanitize_cuda_runtime_environment", script_source)
    @test gpu_project["extras"]["CUDA_Runtime_jll"] ==
        "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
    @test gpu_project["preferences"]["CUDA_Runtime_jll"]["local"] == "false"
    @test gpu_project["preferences"]["CUDA_Runtime_jll"]["version"] == "13.0"
    @test gpu_project["extras"]["MPIPreferences"] ==
        "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
    @test gpu_project["preferences"]["MPIPreferences"]["binary"] == "MPICH_jll"
    @test isempty(gpu_project["preferences"]["MPIPreferences"]["preloads"])
    @test occursin("require_current_run_version", script_source)
    @test occursin("require_worker_compatible_run_version", script_source)
    @test occursin("PHASE1_SCRIPT_VERSION=\"1.18.0\"", script_source)
    @test occursin("check-gpu-preferences)", script_source)
    @test occursin("validate_gpu_runtime_preferences", script_source)
    @test occursin("Base.get_preferences", preference_source)
    @test !occursin(r"(?m)^\s*using\s+(CUDA|HDF5|MPI)", preference_source)
    @test occursin("reconcile)", script_source)
    @test occursin("additional_node_hours_reconciliations.tsv", script_source)
    @test occursin("prepare-recovery)", script_source)
    @test occursin("prepare-recurrence)", script_source)
    @test occursin("prepare-recurrence-competitors)", script_source)
    @test occursin("prepare-matched-seed-pilot)", script_source)
    @test occursin("plan-matched-seed-pilot)", script_source)
    @test occursin("prepare-square-seed-pilot)", script_source)
    @test occursin("plan-square-seed-pilot)", script_source)
    @test occursin("prepare-square-v0-seed-pilot)", script_source)
    @test occursin("plan-square-v0-seed-pilot)", script_source)
    @test occursin("prepare-square-tight5)", script_source)
    @test occursin("plan-square-tight5)", script_source)
    @test occursin("prepare-square-v0-chi400-compare)", script_source)
    @test occursin("plan-square-v0-chi400-compare)", script_source)
    @test occursin("prepare-square-smooth-pairing-grid)", script_source)
    @test occursin("plan-square-smooth-pairing-grid)", script_source)
    @test occursin("prepare-cubic-unfrustrated-smooth-pairing-grid)", script_source)
    @test occursin("plan-cubic-unfrustrated-smooth-pairing-grid)", script_source)
    @test occursin("prepare-square-legacy-stripe-compare)", script_source)
    @test occursin("plan-square-legacy-stripe-compare)", script_source)
    @test occursin("require_continuation_compatible_run_version", script_source)
    @test occursin("prepare-standard)", script_source)
    @test occursin("--licenses=scratch,cfs", script_source)
    @test occursin("compact_results.jl", script_source)
    @test !occursin("transfer_files.py", migration_source)
    @test occursin("cp -a --", migration_source)
    @test occursin("sha256sum -c", migration_source)
    @test occursin("direct_scratch_copy_sha256_verified=true", migration_source)
    @test occursin("--prune-corrupt-auxiliary", migration_source)
    @test occursin("critical_corrupt_hdf5", cleaner_source)
    @test occursin("lowercase(name) == \"state.h5\"", cleaner_source)
    @test occursin("verify_stateless_results.jl", migration_source)
    @test occursin("--prune-cfs", migration_source)
    @test occursin("gpu_linalg_preflight!", read(joinpath(ROOT, "scripts", "gpu_smoke.jl"), String))
    @test occursin("gpu_linalg_preflight!", read(joinpath(ROOT, "scripts", "run_scf_gpu.jl"), String))
    compare_settings = load_settings(joinpath(
        ROOT,
        "configs",
        "phase1_gpu_square_v0_chi400_tight_compare.toml",
    ))
    @test compare_settings.model.geometry == :square
    @test compare_settings.model.V == 0.0
    @test compare_settings.model.t0 == 1.4
    @test compare_settings.dmrg.maxdim == 400
    @test compare_settings.dmrg.mu_density_tol == 1.0e-4
    @test compare_settings.convergence.density_tol == 1.0e-4
    @test compare_settings.convergence.field_abs_tol == 1.0e-7
    @test compare_settings.convergence.field_rel_tol == 1.0e-4
    @test compare_settings.convergence.variational_energy_tol == 1.0e-7
    @test compare_settings.convergence.probe_iterations == 20
    @test compare_settings.run.max_iterations == 80
    grid_settings = load_settings(joinpath(
        ROOT,
        "configs",
        "phase1_gpu_square_grid_smooth_pairing_chi200_loose.toml",
    ))
    @test grid_settings.model.geometry == :square
    @test grid_settings.model.V == -0.4
    @test grid_settings.model.t0 == 1.0
    @test grid_settings.model.ep_mode == :exact
    @test grid_settings.model.ep_signed == -0.17882744409052975
    @test grid_settings.dmrg.maxdim == 200
    @test grid_settings.dmrg.nsweeps == 12
    @test grid_settings.dmrg.mu_density_tol == 1.0e-3
    @test grid_settings.dmrg.mu_warm_start_noise == 1.0e-8
    @test grid_settings.convergence.density_tol == 1.0e-3
    @test grid_settings.convergence.field_abs_tol == 1.0e-6
    @test grid_settings.convergence.field_rel_tol == 5.0e-3
    @test grid_settings.convergence.probe_iterations == 20
    @test grid_settings.run.initial_seed == :legacy_pairing
    @test grid_settings.run.initial_seed_protocol == :matched_mode
    @test grid_settings.run.initial_amplitude == 1.0e-3
    @test grid_settings.run.random_seed == 1404
    grid_seed = initial_fields(
        grid_settings.model;
        seed=grid_settings.run.initial_seed,
        amplitude=grid_settings.run.initial_amplitude,
        protocol=grid_settings.run.initial_seed_protocol,
        mode_number=grid_settings.run.initial_mode_number,
        mode_phase_pi=grid_settings.run.initial_mode_phase_pi,
        pairing_form_factor=grid_settings.run.initial_pairing_form_factor,
        leg_parity=grid_settings.run.initial_leg_parity,
        stripe_charge_to_spin_ratio=grid_settings.run.initial_stripe_charge_to_spin_ratio,
        stripe_pairing_to_spin_ratio=grid_settings.run.initial_stripe_pairing_to_spin_ratio,
        random_seed=grid_settings.run.random_seed,
    )
    @test all(iszero, grid_seed.beta)
    @test all(iszero, grid_seed.mu_cdw)
    @test any(value -> !iszero(value), grid_seed.alpha)
    @test field_l2_per_physical_site(grid_seed, grid_settings.model) ≈ 1.0e-3
    for offset in 0:grid_settings.model.r_range, leg in 1:2, other_leg in 1:2
        values = [
            grid_seed.alpha[rung, rung + offset, leg, other_leg]
            for rung in 1:(grid_settings.model.L - offset)
        ]
        @test maximum(abs.(values .- first(values))) <= 1.0e-14
    end
    cubic_grid_settings = load_settings(joinpath(
        ROOT,
        "configs",
        "phase1_gpu_cubic_unfrustrated_grid_smooth_pairing_chi200_loose.toml",
    ))
    @test cubic_grid_settings.model.geometry == :cubic_unfrustrated
    @test cubic_grid_settings.model.V == grid_settings.model.V
    @test cubic_grid_settings.model.t0 == grid_settings.model.t0
    @test cubic_grid_settings.model.ep_signed == grid_settings.model.ep_signed
    @test numerical_fingerprint(cubic_grid_settings) == numerical_fingerprint(grid_settings)
    @test initial_seed_fingerprint(cubic_grid_settings) == initial_seed_fingerprint(grid_settings)
    stripe_compare_settings = load_settings(joinpath(
        ROOT,
        "configs",
        "phase1_gpu_square_t014_vm04_legacy_stripe_compare_chi200_loose.toml",
    ))
    @test stripe_compare_settings.model.geometry == :square
    @test stripe_compare_settings.model.V == -0.4
    @test stripe_compare_settings.model.t0 == 1.4
    @test stripe_compare_settings.model.ep_mode == :exact
    @test stripe_compare_settings.model.ep_signed == -0.24962435880865996
    @test stripe_compare_settings.dmrg.maxdim == 200
    @test stripe_compare_settings.dmrg.mu_density_tol == 1.0e-3
    @test stripe_compare_settings.convergence.field_rel_tol == 5.0e-3
    @test stripe_compare_settings.convergence.probe_iterations == 20
    @test numerical_fingerprint(stripe_compare_settings) == numerical_fingerprint(grid_settings)
    stripe_prepare_source = read(joinpath(
        ROOT,
        "scripts",
        "prepare_phase1_square_legacy_stripe_compare.jl",
    ), String)
    @test occursin(
        "ae6a3bfe76ca8f06f2396fd731b18bca8539e0b7ee68df016cc9156fdceeb074",
        stripe_prepare_source,
    )
    @test occursin("zero_inactive_same_physical_site_beta_only_v1", stripe_prepare_source)
    bash_executable = Sys.which("bash")
    if bash_executable === nothing
        @test_skip "bash-only Perlmutter launcher execution tests"
    else
    sanitized_environment = read(
        `$bash_executable -c 'source "$1" plan >/dev/null; export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2; export LD_LIBRARY_PATH=/safe/one:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/math_libs/13.2/lib64:/safe/two; sanitize_cuda_runtime_environment; printf "%s|%s|%s\n" "${LD_LIBRARY_PATH-unset}" "${CUDA_HOME-unset}" "${CUDA_PATH-unset}"' bash $script`,
        String,
    )
    @test strip(sanitized_environment) == "/safe/one:/safe/two|unset|unset"
    constraint_policy = read(
        `$bash_executable -c 'source "$1" budget >/dev/null; gpu_constraint_for_chi 1199; gpu_constraint_for_chi 1200' bash $script`,
        String,
    )
    @test split(chomp(constraint_policy), '\n') == ["gpu", "gpu&hbm80g"]
    function bash_path(path::AbstractString)
        normalized = replace(abspath(path), '\\' => '/')
        match_result = match(r"^([A-Za-z]):/(.*)$", normalized)
        return match_result === nothing ? normalized :
            "/$(lowercase(match_result.captures[1]))/$(match_result.captures[2])"
    end
    bash_environment_path = strip(read(
        `$bash_executable -lc 'printf "%s" "${PATH}"'`,
        String,
    ))
    mock_bin = joinpath(ROOT, "test", "fixtures", "mock_slurm")
    julia_executable = Base.julia_cmd().exec[1]
    mktempdir() do directory
        run_root = joinpath(directory, "runs")
        scratch_root = joinpath(directory, "scratch")
        budget_root = joinpath(directory, "budget")
        ledger = joinpath(budget_root, "ledger.tsv")
        environment = (
            "PATH" => string(bash_path(mock_bin), ":", bash_environment_path),
            "MOCK_SLURM_STATE_DIR" => bash_path(joinpath(directory, "slurm")),
            "PHASE1_RUN_ROOT" => bash_path(run_root),
            "PHASE1_SCRATCH_ROOT" => bash_path(scratch_root),
            "PHASE1_BUDGET_ROOT" => bash_path(budget_root),
            "PHASE1_LEDGER_PATH" => bash_path(ledger),
            "PHASE1_JULIA" => bash_path(julia_executable),
        )
        run(pipeline(
            addenv(`$bash_executable $script prepare-standard mock_phase1`, environment...),
            stdout=devnull,
        ))
        @test !isfile(ledger)
        @test strip(read(joinpath(run_root, "mock_phase1", "campaign_kind.txt"), String)) == "standard"
        run_environment_path = joinpath(run_root, "mock_phase1", "run.env")
        write(
            run_environment_path,
            replace(
                read(run_environment_path, String),
                "PHASE1_RUN_SCRIPT_VERSION=1.18.0" => "PHASE1_RUN_SCRIPT_VERSION=1.12.0",
            ),
        )
        run(pipeline(
            addenv(`$bash_executable $script submit mock_phase1`, environment...),
            stdout=devnull,
        ))
        ledger_rows = readlines(ledger)[2:end]
        @test length(ledger_rows) == 9
        @test sum(parse(Float64, split(row, '\t')[7]) for row in ledger_rows) ≈ 27.0
        submission_arguments = read(joinpath(directory, "slurm", "submitted_args.tsv"), String)
        @test count(line -> occursin("--constraint=gpu", line), split(chomp(submission_arguments), '\n')) == 9
        @test !occursin("hbm80g", submission_arguments)
        @test length(readlines(joinpath(run_root, "mock_phase1", "manifest.tsv"))) == 10
        @test length(readlines(joinpath(run_root, "mock_phase1", "jobs.tsv"))) == 10
        status_output = read(
            addenv(`$bash_executable $script status mock_phase1`, environment...),
            String,
        )
        @test occursin("frustrated__pairing_s1", status_output)
        @test occursin("COMPLETED", status_output)
        @test isdir(joinpath(scratch_root, "mock_phase1", "results"))
        prepared_config = TOML.parsefile(joinpath(
            run_root,
            "mock_phase1",
            "configs",
            "frustrated__pairing_s1.segment-001.toml",
        ))
        @test startswith(
            prepared_config["run"]["output_directory"],
            joinpath(scratch_root, "mock_phase1", "results"),
        )
        @test isempty(readdir(joinpath(run_root, "mock_phase1", "results")))

        source_run = joinpath(run_root, "source_recurrence")
        source_full_root = joinpath(scratch_root, "source_recurrence")
        source_full_state = joinpath(source_full_root, "results", "candidate", "state.h5")
        recurrence_base = load_settings(joinpath(
            ROOT,
            "configs",
            "phase1_gpu_recurrence_chi400.toml",
        ))
        mkpath(dirname(source_full_state))
        h5open(source_full_state, "w") do file
            file["status"] = "periodic_candidate"
            file["accepted"] = false
            file["fundamental_period"] = 2
            provenance = create_group(file, "provenance")
            provenance["model_fingerprint"] = LadderMPSMFT.model_fingerprint(recurrence_base.model)
            provenance["numerical_fingerprint"] = "source-numerical-fingerprint"
            provenance["tensor_scalar_type"] = "float64"
            cycles = create_group(file, "cycle_members")
            for (index, phase_name) in enumerate(("001", "002"))
                phase = create_group(cycles, phase_name)
                phase["psi"] = Float64[parse(Int, phase_name)]
                create_group(phase, "applied")
                create_group(phase, "measured")
                phase["chemical_potential"] = 1.0
                phase["iteration"] = 47 + index
                phase["update_mode"] = "unmixed_probe"
            end
        end
        source_compact_state = joinpath(
            source_run,
            "stateless_results",
            "unfrustrated__pairing_s1",
            "candidate",
            "state.h5",
        )
        mkpath(dirname(source_compact_state))
        h5open(source_compact_state, "w") do file
            analysis = create_group(file, "analysis_storage")
            analysis["is_stateless_copy"] = true
            analysis["full_artifact_path"] = source_full_state
            analysis["full_artifact_sha256"] = LadderMPSMFT.sha256_file(source_full_state)
            file["status"] = "periodic_candidate"
            file["accepted"] = false
            file["fundamental_period"] = 2
            provenance = create_group(file, "provenance")
            provenance["model_fingerprint"] = LadderMPSMFT.model_fingerprint(recurrence_base.model)
            provenance["numerical_fingerprint"] = "source-numerical-fingerprint"
            provenance["tensor_scalar_type"] = "float64"
            cycles = create_group(file, "cycle_members")
            for (index, phase_name) in enumerate(("001", "002"))
                phase = create_group(cycles, phase_name)
                create_group(phase, "applied")
                create_group(phase, "measured")
                phase["iteration"] = 47 + index
                phase["update_mode"] = "unmixed_probe"
            end
        end
        mkpath(source_run)
        write(joinpath(source_run, "full_storage_path.txt"), source_full_root * "\n")
        ledger_before_recurrence = read(ledger, String)
        recurrence_plan = read(
            addenv(`$bash_executable $script plan-recurrence`, environment...),
            String,
        )
        @test occursin("9.000000000 node-hours", recurrence_plan)
        run(pipeline(
            addenv(
                `$bash_executable $script prepare-recurrence source_recurrence mock_recurrence`,
                environment...,
            ),
            stdout=devnull,
        ))
        @test read(ledger, String) == ledger_before_recurrence
        recurrence_run = joinpath(run_root, "mock_recurrence")
        @test strip(read(joinpath(recurrence_run, "branch_count.txt"), String)) == "3"
        @test length(readlines(joinpath(recurrence_run, "manifest.tsv"))) == 4
        @test length(filter(
            name -> endswith(name, ".segment-001.toml"),
            readdir(joinpath(recurrence_run, "configs")),
        )) == 3
        phase_config = TOML.parsefile(joinpath(
            recurrence_run,
            "configs",
            "unfrustrated__pairing_s1_phase001_chi400.segment-001.toml",
        ))
        @test phase_config["run"]["parent_orbit_phase"] == 1
        @test phase_config["run"]["parent_sha256"] == LadderMPSMFT.sha256_file(source_full_state)
        @test phase_config["dmrg"]["maxdim"] == 400
        @test phase_config["convergence"]["cycle_action"] == "stop"
        @test strip(read(joinpath(recurrence_run, "campaign_kind.txt"), String)) == "recurrence"
        # The mock launcher runs through Git Bash while Julia is native Windows;
        # rewrite the locator in the host-native form used by synthetic HDF5 links.
        write(
            joinpath(recurrence_run, "full_storage_path.txt"),
            joinpath(scratch_root, "mock_recurrence") * "\n",
        )

        recurrence_numerical = LadderMPSMFT.numerical_fingerprint(recurrence_base)
        recurrence_implementation = LadderMPSMFT.implementation_fingerprint(recurrence_base)
        recurrence_ep_hash = LadderMPSMFT.sha256_file(recurrence_base.model.ep_source)
        function write_gate_state(
            label::AbstractString;
            accepted::Bool,
            status::AbstractString,
            period::Int,
            unmixed::Bool,
            alpha_max::Float64,
        )
            path = joinpath(recurrence_run, "results", label, "result", "state.h5")
            full_path = joinpath(
                scratch_root,
                "mock_recurrence",
                "results",
                label,
                "result",
                "state.h5",
            )
            mkpath(dirname(path))
            mkpath(dirname(full_path))
            h5open(full_path, "w") do file
                file["synthetic_full_state"] = true
            end
            h5open(path, "w") do file
                file["status"] = status
                file["accepted"] = accepted
                file["fundamental_period"] = period
                file["unmixed_cycle_probe"] = unmixed
                fields = create_group(file, "fields")
                measured = create_group(fields, "measured")
                measured["alpha"] = fill(alpha_max, 2, 4)
                if status == "periodic_solution"
                    cycles = create_group(file, "cycle_members")
                    for (index, phase_name) in enumerate(("001", "002"))
                        phase = create_group(cycles, phase_name)
                        phase_measured = create_group(phase, "measured")
                        phase_measured["alpha"] = fill(alpha_max / index, 2, 4)
                    end
                end
                model = create_group(file, "model")
                model["transverse_geometry"] = "cubic_unfrustrated"
                provenance = create_group(file, "provenance")
                provenance["initial_seed"] = "pairing"
                provenance["model_fingerprint"] = LadderMPSMFT.model_fingerprint(recurrence_base.model)
                provenance["numerical_fingerprint"] = recurrence_numerical
                provenance["implementation_sha256"] = recurrence_implementation
                provenance["ep_source_sha256"] = recurrence_ep_hash
                provenance["tensor_scalar_type"] = "float64"
                analysis = create_group(file, "analysis_storage")
                analysis["is_stateless_copy"] = true
                analysis["full_artifact_path"] = full_path
                analysis["full_artifact_sha256"] = LadderMPSMFT.sha256_file(full_path)
            end
            return path
        end
        write_gate_state(
            "unfrustrated__pairing_s1_phase001_chi400";
            accepted=true,
            status="periodic_solution",
            period=2,
            unmixed=true,
            alpha_max=1.0e-2,
        )
        write_gate_state(
            "unfrustrated__pairing_s1_phase002_chi400";
            accepted=false,
            status="maximum_iterations",
            period=0,
            unmixed=true,
            alpha_max=1.1e-2,
        )
        write_gate_state(
            "unfrustrated__pairing_s2_chi400";
            accepted=true,
            status="fixed_point",
            period=1,
            unmixed=true,
            alpha_max=2.0e-3,
        )
        controls_plan = read(
            addenv(`$bash_executable $script plan-recurrence-controls`, environment...),
            String,
        )
        @test occursin("Conditional Stage B first segments: 6.000000000 node-hours", controls_plan)
        @test occursin("Combined first-segment envelope: 15.000000000 node-hours", controls_plan)
        ledger_before_controls = read(ledger, String)
        run(pipeline(
            addenv(
                `$bash_executable $script prepare-recurrence-competitors mock_recurrence mock_recurrence_controls`,
                environment...,
            ),
            stdout=devnull,
        ))
        @test read(ledger, String) == ledger_before_controls
        controls_run = joinpath(run_root, "mock_recurrence_controls")
        @test strip(read(joinpath(controls_run, "campaign_kind.txt"), String)) == "recurrence_competitors"
        @test strip(read(joinpath(controls_run, "branch_count.txt"), String)) == "2"
        @test length(readlines(joinpath(controls_run, "manifest.tsv"))) == 3
        @test length(readlines(joinpath(controls_run, "conditional_gate.tsv"))) == 4
        sdw_control = TOML.parsefile(joinpath(
            controls_run,
            "configs",
            "unfrustrated__sdw_s2_chi400.segment-001.toml",
        ))
        @test sdw_control["run"]["max_iterations"] == 80
        @test sdw_control["run"]["random_seed"] == 1203
        @test sdw_control["convergence"]["cycle_action"] == "stop"
        @test LadderMPSMFT.numerical_fingerprint(load_settings(joinpath(
            controls_run,
            "configs",
            "unfrustrated__sdw_s2_chi400.segment-001.toml",
        ))) == recurrence_numerical

        matched_plan = read(
            addenv(`$bash_executable $script plan-matched-seed-pilot`, environment...),
            String,
        )
        @test occursin("First-segment envelope: 9.000000000 node-hours", matched_plan)
        @test occursin("mode n=0, d_wave", matched_plan)
        @test occursin("mode n=58, odd leg parity", matched_plan)
        @test occursin("mode n=11, even leg parity", matched_plan)
        ledger_before_matched = read(ledger, String)
        run(pipeline(
            addenv(
                `$bash_executable $script prepare-matched-seed-pilot mock_matched_seed`,
                environment...,
            ),
            stdout=devnull,
        ))
        @test read(ledger, String) == ledger_before_matched
        matched_run = joinpath(run_root, "mock_matched_seed")
        @test strip(read(joinpath(matched_run, "campaign_kind.txt"), String)) ==
            "matched_seed_pilot"
        @test strip(read(joinpath(matched_run, "branch_count.txt"), String)) == "3"
        @test length(readlines(joinpath(matched_run, "manifest.tsv"))) == 4
        matched_labels = (
            "unfrustrated__pairing_matched_m000_chi400",
            "unfrustrated__sdw_matched_m058_chi400",
            "unfrustrated__cdw_matched_m011_chi400",
        )
        matched_settings = [load_settings(joinpath(
            matched_run,
            "configs",
            "$label.segment-001.toml",
        )) for label in matched_labels]
        @test [settings.run.initial_seed for settings in matched_settings] ==
            [:pairing, :sdw, :cdw]
        @test [settings.run.initial_mode_number for settings in matched_settings] == [0, 58, 11]
        @test [settings.run.initial_leg_parity for settings in matched_settings] ==
            [:auto, :odd, :even]
        @test matched_settings[1].run.initial_pairing_form_factor == :d_wave
        @test all(settings.run.initial_seed_protocol == :matched_mode for settings in matched_settings)
        @test all(settings.run.initial_amplitude == 1.0e-3 for settings in matched_settings)
        @test all(settings.run.initial_mode_phase_pi == 0.0 for settings in matched_settings)
        @test all(settings.run.random_seed == 1404 for settings in matched_settings)
        @test all(settings.dmrg.maxdim == 400 for settings in matched_settings)
        @test all(settings.run.max_iterations == 21 for settings in matched_settings)
        @test all(settings.convergence.probe_iterations == 20 for settings in matched_settings)
        @test all(settings.convergence.cycle_action == :stop for settings in matched_settings)
        @test length(unique(
            LadderMPSMFT.numerical_fingerprint(settings) for settings in matched_settings
        )) == 1
        @test length(unique(
            LadderMPSMFT.model_fingerprint(settings.model) for settings in matched_settings
        )) == 1
        @test length(unique(
            LadderMPSMFT.initial_seed_fingerprint(settings) for settings in matched_settings
        )) == 3
        matched_header = split(first(readlines(joinpath(matched_run, "manifest.tsv"))), '\t')
        @test "initial_mode_wavevector_pi" in matched_header
        @test "initial_seed_normalization" in matched_header

        square_plan = read(
            addenv(`$bash_executable $script plan-square-seed-pilot`, environment...),
            String,
        )
        @test occursin("First-segment envelope: 18.000000000 node-hours", square_plan)
        @test occursin("envelope m=4 -> AF spin n=59 and charge harmonic n=8", square_plan)
        @test occursin("envelope m=5 -> AF spin n=58 and charge harmonic n=10", square_plan)
        @test occursin("representative six-branch bank plus eight later three-branch points:  90.000000000", square_plan)
        @test occursin("eight later points only (conditional, seed bank not yet locked):       72.000000000", square_plan)
        @test occursin("repeating all six branches at all nine points is not recommended:     162.000000000", square_plan)
        ledger_before_square = read(ledger, String)
        run(pipeline(
            addenv(
                `$bash_executable $script prepare-square-seed-pilot mock_square_seed`,
                environment...,
            ),
            stdout=devnull,
        ))
        @test read(ledger, String) == ledger_before_square
        square_run = joinpath(run_root, "mock_square_seed")
        @test strip(read(joinpath(square_run, "campaign_kind.txt"), String)) ==
            "square_seed_pilot"
        @test strip(read(joinpath(square_run, "branch_count.txt"), String)) == "6"
        @test length(readlines(joinpath(square_run, "manifest.tsv"))) == 7
        square_labels = (
            "square__pairing_dwave_m000_chi200_loose",
            "square__legacy_pairing_mixed_chi200_loose",
            "square__stripe_m004_chi200_loose",
            "square__stripe_m005_chi200_loose",
            "square__stripe_pairing_m004_chi200_loose",
            "square__stripe_pairing_m005_chi200_loose",
        )
        square_paths = [joinpath(
            square_run,
            "configs",
            "$label.segment-001.toml",
        ) for label in square_labels]
        square_settings = load_settings.(square_paths)
        @test [settings.run.initial_seed for settings in square_settings] ==
            [:pairing, :legacy_pairing, :stripe, :stripe, :stripe_pairing, :stripe_pairing]
        @test [settings.run.initial_mode_number for settings in square_settings] == [0, 0, 4, 5, 4, 5]
        @test all(settings.run.initial_leg_parity == :auto for settings in square_settings)
        @test square_settings[1].run.initial_pairing_form_factor == :d_wave
        @test [settings.run.initial_pairing_form_factor for settings in square_settings] ==
            [:d_wave, :onsite_s, :onsite_s, :onsite_s, :d_wave, :d_wave]
        @test all(settings.model.geometry == :square for settings in square_settings)
        @test all(settings.model.V == -0.4 for settings in square_settings)
        @test all(settings.model.t0 == 1.4 for settings in square_settings)
        @test all(settings.model.mu_initial == 0.55 for settings in square_settings)
        @test all(settings.model.ep_mode == :exact for settings in square_settings)
        @test all(settings.model.ep_signed == -0.24962435880865996 for settings in square_settings)
        @test all(settings.run.initial_seed_protocol == :matched_mode for settings in square_settings)
        @test all(settings.run.initial_amplitude == 1.0e-3 for settings in square_settings)
        @test all(settings.run.initial_mode_phase_pi == 0.0 for settings in square_settings)
        @test all(settings.run.random_seed == 1404 for settings in square_settings)
        @test all(settings.run.initial_stripe_charge_to_spin_ratio == 0.2 for settings in square_settings)
        @test [settings.run.initial_stripe_pairing_to_spin_ratio for settings in square_settings] ==
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        square_seed_fields = [initial_fields(
            settings.model;
            seed=settings.run.initial_seed,
            amplitude=settings.run.initial_amplitude,
            protocol=settings.run.initial_seed_protocol,
            mode_number=settings.run.initial_mode_number,
            mode_phase_pi=settings.run.initial_mode_phase_pi,
            pairing_form_factor=settings.run.initial_pairing_form_factor,
            leg_parity=settings.run.initial_leg_parity,
            stripe_charge_to_spin_ratio=settings.run.initial_stripe_charge_to_spin_ratio,
            stripe_pairing_to_spin_ratio=settings.run.initial_stripe_pairing_to_spin_ratio,
            random_seed=settings.run.random_seed,
        ) for settings in square_settings]
        @test all(field_l2_per_physical_site(fields, square_settings[1].model) ≈ 1.0e-3 for fields in square_seed_fields)
        @test any(value -> !iszero(value), square_seed_fields[1].alpha)
        @test all(iszero, square_seed_fields[1].mu_cdw)
        @test any(value -> !iszero(value), square_seed_fields[2].alpha)
        @test all(iszero, square_seed_fields[2].beta)
        @test all(iszero, square_seed_fields[2].mu_cdw)
        @test all(iszero, square_seed_fields[3].alpha)
        @test all(iszero, square_seed_fields[4].alpha)
        @test all(any(value -> !iszero(value), fields.mu_cdw) for fields in square_seed_fields[3:6])
        @test all(any(value -> !iszero(value), fields.alpha) for fields in square_seed_fields[5:6])
        square_seed_metadata = [initial_seed_metadata(settings.model, settings.run) for settings in square_settings]
        @test [metadata.stripe_spin_mode_number for metadata in square_seed_metadata] == [0, 0, 59, 58, 59, 58]
        @test [metadata.stripe_charge_mode_number for metadata in square_seed_metadata] == [0, 0, 8, 10, 8, 10]
        @test square_seed_metadata[2].legacy_pairing_random_seed == 1404
        @test square_seed_metadata[2].legacy_pairing_center_of_mass_structure ==
            "constant_by_relative_offset_and_leg_pair"
        @test all(settings.dmrg.nsweeps == 12 for settings in square_settings)
        @test all(settings.dmrg.maxdim == 200 for settings in square_settings)
        @test all(settings.dmrg.cutoff == 1.0e-10 for settings in square_settings)
        @test all(settings.dmrg.energy_tol == 1.0e-6 for settings in square_settings)
        @test all(settings.dmrg.mu_density_tol == 1.0e-3 for settings in square_settings)
        @test all(settings.dmrg.mu_bracket_step == 0.01 for settings in square_settings)
        @test all(settings.dmrg.mu_bracket_growth == 3.0 for settings in square_settings)
        @test all(settings.dmrg.mu_warm_start_noise == 1.0e-8 for settings in square_settings)
        @test all(settings.convergence.density_tol == 1.0e-3 for settings in square_settings)
        @test all(settings.convergence.variational_energy_tol == 1.0e-6 for settings in square_settings)
        @test all(settings.convergence.period2_oscillation_cosine_max == -0.5 for settings in square_settings)
        @test all(settings.convergence.period2_two_step_ratio_max == 0.5 for settings in square_settings)
        @test all(settings.convergence.slow_mode_cosine_min == 0.9 for settings in square_settings)
        @test all(settings.convergence.probe_iterations == 20 for settings in square_settings)
        @test all(settings.convergence.cycle_action == :continue for settings in square_settings)
        @test all(settings.run.max_iterations == 80 for settings in square_settings)
        @test all(startswith(
            settings.run.output_directory,
            joinpath(scratch_root, "mock_square_seed", "results"),
        ) for settings in square_settings)
        @test length(unique(
            LadderMPSMFT.numerical_fingerprint(settings) for settings in square_settings
        )) == 1
        @test length(unique(
            LadderMPSMFT.model_fingerprint(settings.model) for settings in square_settings
        )) == 1
        @test length(unique(
            LadderMPSMFT.initial_seed_fingerprint(settings) for settings in square_settings
        )) == 6
        for path in square_paths
            raw = TOML.parsefile(path)["run"]
            @test !haskey(raw, "inherit_from")
            @test !haskey(raw, "parent_checkpoint")
            @test !haskey(raw, "resume_checkpoint")
        end
        square_header = split(first(readlines(joinpath(square_run, "manifest.tsv"))), '\t')
        @test "analysis_role" in square_header
        @test "preliminary_energy_only" in square_header
        @test "mu_bracket_growth" in square_header
        @test "stripe_spin_mode_number" in square_header
        @test "stripe_charge_mode_number" in square_header
        @test "stripe_pairing_to_spin_ratio" in square_header
        @test "legacy_pairing_center_of_mass_structure" in square_header
        @test "mu_warm_start_noise" in square_header
        @test "period2_oscillation_cosine_max" in square_header
        @test "slow_mode_cosine_min" in square_header

        square_v0_plan = read(
            addenv(`$bash_executable $script plan-square-v0-seed-pilot`, environment...),
            String,
        )
        @test occursin("L=64, U=8, V=0, t0=1.4", square_v0_plan)
        @test occursin("exact registry E_p=-0.14653773091916378", square_v0_plan)
        @test occursin("First-segment envelope: 18.000000000 node-hours", square_v0_plan)
        @test occursin("step cosine<=-0.5 and d2/d1<=0.5", square_v0_plan)
        @test occursin("apply r/(1-lambda)", square_v0_plan)
        ledger_before_square_v0 = read(ledger, String)
        run(pipeline(
            addenv(
                `$bash_executable $script prepare-square-v0-seed-pilot mock_square_v0_seed`,
                environment...,
            ),
            stdout=devnull,
        ))
        @test read(ledger, String) == ledger_before_square_v0
        square_v0_run = joinpath(run_root, "mock_square_v0_seed")
        @test strip(read(joinpath(square_v0_run, "campaign_kind.txt"), String)) ==
            "square_seed_pilot_v0"
        @test strip(read(joinpath(square_v0_run, "branch_count.txt"), String)) == "6"
        @test length(readlines(joinpath(square_v0_run, "manifest.tsv"))) == 7
        square_v0_paths = [joinpath(
            square_v0_run,
            "configs",
            "$label.segment-001.toml",
        ) for label in square_labels]
        square_v0_settings = load_settings.(square_v0_paths)
        @test [settings.run.initial_seed for settings in square_v0_settings] ==
            [:pairing, :legacy_pairing, :stripe, :stripe, :stripe_pairing, :stripe_pairing]
        @test [settings.run.initial_mode_number for settings in square_v0_settings] == [0, 0, 4, 5, 4, 5]
        @test all(settings.model.geometry == :square for settings in square_v0_settings)
        @test all(settings.model.V == 0.0 for settings in square_v0_settings)
        @test all(settings.model.t0 == 1.4 for settings in square_v0_settings)
        @test all(settings.model.mu_initial == 0.55 for settings in square_v0_settings)
        @test all(settings.model.ep_mode == :exact for settings in square_v0_settings)
        @test all(settings.model.ep_signed == -0.14653773091916378 for settings in square_v0_settings)
        @test all(settings.dmrg.mu_warm_start_noise == 1.0e-8 for settings in square_v0_settings)
        @test length(unique(
            LadderMPSMFT.model_fingerprint(settings.model) for settings in square_v0_settings
        )) == 1
        @test length(unique(
            LadderMPSMFT.numerical_fingerprint(settings) for settings in square_v0_settings
        )) == 1
        @test length(unique(
            LadderMPSMFT.initial_seed_fingerprint(settings) for settings in square_v0_settings
        )) == 6
        @test all(startswith(
            settings.run.output_directory,
            joinpath(scratch_root, "mock_square_v0_seed", "results"),
        ) for settings in square_v0_settings)
        square_v0_header = split(first(readlines(joinpath(square_v0_run, "manifest.tsv"))), '\t')
        @test all(column in square_v0_header for column in (
            "point_id", "L", "U", "V", "t0", "tp", "density", "chi",
        ))
        square_v0_rows = split.(readlines(joinpath(square_v0_run, "manifest.tsv"))[2:end], '\t')
        v0_index = findfirst(==("V"), square_v0_header)
        point_index = findfirst(==("point_id"), square_v0_header)
        @test all(row[v0_index] == "0.0" for row in square_v0_rows)
        @test all(row[point_index] == "square_t014_v000" for row in square_v0_rows)
        parsed_submission_rows = split(chomp(read(
            addenv(
                `$bash_executable -c 'source "$1" budget >/dev/null; manifest_submission_rows "$2"' bash $script $(joinpath(square_v0_run, "manifest.tsv"))`,
                environment...,
            ),
            String,
        )), '\n')
        @test length(parsed_submission_rows) == 6
        @test Set(last(split(row, '\t'; limit=2)) for row in parsed_submission_rows) ==
            Set(square_v0_paths)

        # Build six synthetic full/compact accepted parents under the prepared
        # square pilot. The tight-five preparer must rehash the full parents,
        # retain fresh-history lineage, and leave the budget ledger untouched.
        write(
            joinpath(square_run, "full_storage_path.txt"),
            joinpath(scratch_root, "mock_square_seed") * "\n",
        )
        square_model_fingerprint = LadderMPSMFT.model_fingerprint(first(square_settings).model)
        square_numerical_fingerprint = LadderMPSMFT.numerical_fingerprint(first(square_settings))
        square_implementation = LadderMPSMFT.implementation_fingerprint(first(square_settings))
        square_ep_hash = LadderMPSMFT.sha256_file(first(square_settings).model.ep_source)
        square_parent_hashes = Dict{String,String}()
        for (index, label) in enumerate(square_labels)
            full_path = joinpath(
                scratch_root,
                "mock_square_seed",
                "results",
                label,
                "result",
                "state.h5",
            )
            mkpath(dirname(full_path))
            h5open(full_path, "w") do file
                file["psi"] = Float64[1.0]
                fields = create_group(file, "fields")
                restart = create_group(fields, "restart")
                restart["alpha"] = zeros(2, 2)
                restart["beta"] = zeros(2, 2)
                restart["mu_cdw"] = zeros(2, 2)
                file["chemical_potential"] = 0.55
                file["status"] = "fixed_point"
                file["accepted"] = true
                file["solution_kind"] = "fixed_point"
                file["fundamental_period"] = 1
                provenance = create_group(file, "provenance")
                provenance["model_fingerprint"] = square_model_fingerprint
                provenance["numerical_fingerprint"] = square_numerical_fingerprint
                provenance["implementation_sha256"] = square_implementation
                provenance["ep_source_sha256"] = square_ep_hash
                provenance["tensor_scalar_type"] = "float64"
            end
            full_hash = LadderMPSMFT.sha256_file(full_path)
            square_parent_hashes[label] = full_hash
            compact_path = joinpath(square_run, "results", label, "result", "state.h5")
            mkpath(dirname(compact_path))
            h5open(compact_path, "w") do file
                analysis = create_group(file, "analysis_storage")
                analysis["is_stateless_copy"] = true
                analysis["full_artifact_path"] = full_path
                analysis["full_artifact_sha256"] = full_hash
                file["status"] = "fixed_point"
                file["accepted"] = true
                file["solution_kind"] = "fixed_point"
                file["fundamental_period"] = 1
                file["solution_canonical_variational_energy"] = -149.0 - index / 1000
                file["fixed_point_rel_residual"] = 1.0e-3 / index
                file["density_error"] = 5.0e-4
                history = create_group(file, "history")
                history["iteration"] = collect(1:6)
                provenance = create_group(file, "provenance")
                provenance["model_fingerprint"] = square_model_fingerprint
                provenance["numerical_fingerprint"] = square_numerical_fingerprint
                provenance["implementation_sha256"] = square_implementation
                provenance["ep_source_sha256"] = square_ep_hash
                provenance["tensor_scalar_type"] = "float64"
            end
        end
        tight_plan = read(
            addenv(`$bash_executable $script plan-square-tight5`, environment...),
            String,
        )
        @test occursin("First-segment envelope: 4.500000000 node-hours", tight_plan)
        @test occursin("one of four GPUs, 03:00:00", tight_plan)
        @test occursin("physical map threshold=0", tight_plan)
        ledger_before_tight = read(ledger, String)
        run(pipeline(
            addenv(
                `$bash_executable $script prepare-square-tight5 mock_square_seed mock_square_tight5`,
                environment...,
            ),
            stdout=devnull,
        ))
        @test read(ledger, String) == ledger_before_tight
        tight_run = joinpath(run_root, "mock_square_tight5")
        @test strip(read(joinpath(tight_run, "campaign_kind.txt"), String)) == "square_tight5"
        @test strip(read(joinpath(tight_run, "branch_count.txt"), String)) == "6"
        @test occursin("PHASE1_GPU_TIME=03:00:00", read(joinpath(tight_run, "run.env"), String))
        tight_labels = replace.(collect(square_labels), "_loose" => "_tight5")
        tight_paths = [joinpath(
            tight_run,
            "configs",
            "$label.segment-001.toml",
        ) for label in tight_labels]
        tight_settings = load_settings.(tight_paths)
        @test all(settings.dmrg.nsweeps == 16 for settings in tight_settings)
        @test all(settings.dmrg.maxdim == 200 for settings in tight_settings)
        @test all(settings.dmrg.cutoff == 1.0e-11 for settings in tight_settings)
        @test all(settings.dmrg.energy_tol == 1.0e-9 for settings in tight_settings)
        @test all(settings.dmrg.max_time_seconds == 9000.0 for settings in tight_settings)
        @test all(settings.dmrg.mu_density_tol == 1.0e-4 for settings in tight_settings)
        @test all(settings.convergence.density_tol == 1.0e-4 for settings in tight_settings)
        @test all(settings.convergence.field_abs_tol == 1.0e-7 for settings in tight_settings)
        @test all(settings.convergence.field_rel_tol == 1.0e-4 for settings in tight_settings)
        @test all(settings.convergence.variational_energy_tol == 1.0e-7 for settings in tight_settings)
        @test all(settings.convergence.probe_iterations == 9 for settings in tight_settings)
        @test all(settings.convergence.cycle_action == :stop for settings in tight_settings)
        @test all(settings.run.max_iterations == 5 for settings in tight_settings)
        @test all(settings.run.resume_checkpoint === nothing for settings in tight_settings)
        @test all(settings.run.parent_checkpoint !== nothing for settings in tight_settings)
        @test [settings.run.parent_sha256 for settings in tight_settings] ==
            [square_parent_hashes[label] for label in square_labels]
        @test length(unique(
            LadderMPSMFT.numerical_fingerprint(settings) for settings in tight_settings
        )) == 1
        @test only(unique(
            LadderMPSMFT.numerical_fingerprint(settings) for settings in tight_settings
        )) != square_numerical_fingerprint
        tight_header = split(first(readlines(joinpath(tight_run, "manifest.tsv"))), '\t')
        @test "map_field_threshold" in tight_header
        @test "analysis_field_floor_scan" in tight_header
        @test "parent_implementation_sha256" in tight_header
        @test "preliminary_energy_only" in tight_header

        placeholder_rejected = run(pipeline(
            ignorestatus(addenv(
                `$bash_executable $script prepare-standard RUN_ID`,
                environment...,
            )),
            stdout=devnull,
            stderr=devnull,
        ))
        @test !success(placeholder_rejected)
        @test !ispath(joinpath(run_root, "RUN_ID"))

        missing_submit_rejected = run(pipeline(
            ignorestatus(addenv(
                `$bash_executable $script submit not_prepared`,
                environment...,
            )),
            stdout=devnull,
            stderr=devnull,
        ))
        @test !success(missing_submit_rejected)
        @test !ispath(joinpath(run_root, "not_prepared"))

        run(pipeline(
            addenv(
                `$bash_executable $script prepare-standard cap_rejection`,
                environment...,
                "PHASE1_ADDITIONAL_NODE_HOUR_CAP" => "27.1",
            ),
            stdout=devnull,
        ))
        rejected = run(pipeline(
            ignorestatus(addenv(
                `$bash_executable $script submit cap_rejection`,
                environment...,
            )),
            stdout=devnull,
            stderr=devnull,
        ))
        @test !success(rejected)
        @test isdir(joinpath(run_root, "cap_rejection"))
        @test length(readlines(joinpath(run_root, "cap_rejection", "jobs.tsv"))) == 1
        @test strip(read(joinpath(directory, "slurm", "next_job_id"), String)) == "700009"

        ledger_before_reconcile = read(ledger, String)
        reconciliation = joinpath(budget_root, "additional_node_hours_reconciliations.tsv")
        run(pipeline(
            addenv(`$bash_executable $script reconcile mock_phase1`, environment...),
            stdout=devnull,
        ))
        @test read(ledger, String) == ledger_before_reconcile
        reconciliation_rows = readlines(reconciliation)[2:end]
        @test length(reconciliation_rows) == 9
        @test all(split(row, '\t')[14] == "COMPLETED" for row in reconciliation_rows)
        @test all(parse(Float64, split(row, '\t')[11]) ≈ 0.25 for row in reconciliation_rows)
        @test sum(parse(Float64, split(row, '\t')[13]) for row in reconciliation_rows) ≈ 24.75
        budget_after_reconcile = read(
            addenv(`$bash_executable $script budget`, environment...),
            String,
        )
        @test occursin("Requested upper bounds in ledger: 27.000000000", budget_after_reconcile)
        @test occursin("Released after sacct reconcile:   24.750000000", budget_after_reconcile)
        @test occursin("Active project accounting:        2.250000000", budget_after_reconcile)
        reconciliation_before_repeat = read(reconciliation, String)
        run(pipeline(
            addenv(`$bash_executable $script reconcile mock_phase1`, environment...),
            stdout=devnull,
        ))
        @test read(reconciliation, String) == reconciliation_before_repeat
    end
    end
end

@testset "fixed-mu Phase 0 payload and report" begin
    config = joinpath(ROOT, "test", "fixtures", "phase0_tiny.toml")
    seed_script = joinpath(ROOT, "scripts", "phase0_prepare_seed.jl")
    payload_script = joinpath(ROOT, "scripts", "phase0_payload.jl")
    report_script = joinpath(ROOT, "scripts", "phase0_report.jl")
    julia = Base.julia_cmd()
    mktempdir() do directory
        seed_path = joinpath(directory, "seed_state.h5")
        run(pipeline(
            `$julia --startup-file=no --project=$ROOT $seed_script $config $seed_path`,
            stdout=devnull,
        ))
        seed = h5open(seed_path, "r") do file
            return (
                schema=Int(read(file, "schema_version")),
                benchmark_kind=String(read(file, "benchmark_kind")),
                density=Float64(read(file, "density")),
                target=Float64(read(file, "target_density")),
                chemical_potential=Float64(read(file, "chemical_potential")),
                maximum_bond_dimension=Int(read(file, "maximum_bond_dimension")),
            )
        end
        @test seed.schema == 3
        @test seed.benchmark_kind == "fixed_mu_dmrg"
        @test 0.0 <= seed.density <= 2.0
        @test isfinite(seed.chemical_potential)
        @test 1 <= seed.maximum_bond_dimension <= 4

        metrics_directory = joinpath(directory, "metrics")
        mkpath(metrics_directory)
        metric_path = joinpath(metrics_directory, "serial-t1.toml")
        payload_command = addenv(
            `$julia --startup-file=no --project=$ROOT $payload_script $config $metric_path serial-t1 serial`,
            "PHASE0_SEED_STATE" => seed_path,
            "PHASE0_REPETITIONS" => "1",
            "PHASE0_COMPILE_WARMUP" => "0",
        )
        run(pipeline(payload_command, stdout=devnull))
        metric = TOML.parsefile(metric_path)
        @test metric["schema_version"] == 4
        @test metric["benchmark_kind"] == "fixed_mu_dmrg"
        @test metric["timed_region"] == "run_dmrg_ground_only"
        @test metric["dmrg_solves"] == [1]
        @test !metric["mpo_construction_timed"]
        @test !metric["initial_mps_copy_timed"]
        @test !metric["garbage_collection_timed"]
        @test !metric["compile_warmup_timed"]
        @test metric["seed_chemical_potential"] ≈ seed.chemical_potential
        @test metric["seed_config_sha256"] == metric["config_sha256"]
        @test metric["seed_git_commit"] == metric["git_commit"]
        @test metric["seed_implementation_sha256"] == metric["implementation_sha256"]
        @test metric["benchmark_chemical_potential"] ≈ seed.chemical_potential
        @test length(metric["seconds"]) == 1

        write(joinpath(directory, "candidates.tsv"),
            "label\tjulia_threads\tbackend\tslurm_logical_cpus\nserial-t1\t1\tserial\t2\n")
        write(joinpath(metrics_directory, "serial-t1.time"),
            "Maximum resident set size (kbytes): 1024\nExit status: 0\n")
        run(pipeline(
            `$julia --startup-file=no --project=$ROOT $report_script $directory`,
            stdout=devnull,
        ))
        recommendation = read(joinpath(directory, "recommendation.md"), String)
        @test occursin("Median fixed-mu DMRG time", recommendation)
        @test occursin("run_dmrg_ground", recommendation)
        @test occursin("Estimated comparison with the legacy GPU path", recommendation)

        bad_directory = joinpath(directory, "bad-workload")
        bad_metrics = joinpath(bad_directory, "metrics")
        mkpath(bad_metrics)
        write(joinpath(bad_directory, "candidates.tsv"),
            "label\tjulia_threads\tbackend\tslurm_logical_cpus\nserial-t1\t1\tserial\t2\n")
        bad_metric = deepcopy(metric)
        bad_metric["timed_region"] = "density_search"
        open(joinpath(bad_metrics, "serial-t1.toml"), "w") do io
            TOML.print(io, bad_metric; sorted=true)
        end
        write(joinpath(bad_metrics, "serial-t1.time"),
            "Maximum resident set size (kbytes): 1024\nExit status: 0\n")
        bad_report = run(pipeline(
            ignorestatus(`$julia --startup-file=no --project=$ROOT $report_script $bad_directory`),
            stdout=devnull,
            stderr=devnull,
        ))
        @test !success(bad_report)
    end
end

@testset "variational double-counting functional" begin
    model = test_model()
    fields = test_fields()
    fields.alpha[1, 1, 1, 1] = 0.3
    correlations = test_correlations()
    correlations.pair[1, 1] = 0.4
    energy = variational_energy(-10.0, 2.0, fields, correlations, model)
    @test energy.pair_field_energy ≈ -0.24
    @test energy.pair_transverse_energy ≈ -0.12
    @test energy.double_counting_correction ≈ 0.12
    @test energy.chemical_potential_term ≈ 8.0
    @test energy.canonical_variational_energy ≈ -1.88

    direct = variational_energy(
        -10.0,
        2.0,
        fields,
        correlations,
        model;
        effective_expectation=-10.0,
        bare_ladder_energy=-1.76,
    )
    @test direct.hamiltonian_identity_error ≈ 0.0
    @test direct.variational_consistency_error ≈ 0.0
    @test direct.direct_variational_energy ≈ -1.88

    measured_fields = copy(fields)
    measured_fields.alpha[1, 1, 1, 1] = 0.6
    off_fixed_point = variational_energy(
        -10.0,
        2.0,
        fields,
        correlations,
        model;
        interaction_fields=measured_fields,
    )
    @test off_fixed_point.pair_field_energy ≈ -0.24
    @test off_fixed_point.pair_transverse_energy ≈ -0.24

    density_fields = test_fields()
    density_fields.mu_cdw[1, 1] = 0.2
    density_correlations = test_correlations()
    density_correlations.density_down[1] = 0.7
    density_energy = variational_energy(-3.0, 0.0, density_fields, density_correlations, model)
    @test density_energy.density_field_energy ≈ 0.14
    @test density_energy.density_transverse_energy ≈ 0.02
    @test density_energy.double_counting_correction ≈ -0.12
    @test density_energy.canonical_variational_energy ≈ -3.12

    off_target = test_correlations()
    off_target.density_down .= 0.45
    off_target.density_up .= 0.45
    corrected = variational_energy(-3.0, 2.0, test_fields(), off_target, model)
    @test corrected.target_density_correction ≈ 0.8
    @test corrected.target_density_corrected_variational_energy ≈
        corrected.canonical_variational_energy + 0.8
end

@testset "period-resolved convergence" begin
    settings = ConvergenceSettings(
        field_abs_tol=1e-8,
        field_rel_tol=1e-8,
        density_tol=1e-8,
        variational_energy_tol=1e-8,
        stable_iterations=2,
        max_period=4,
        period_repeats=2,
        period_abs_tol=1e-10,
        period_rel_tol=1e-10,
        stagnation_window=20,
    )
    @test detect_period(test_fields.([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]), settings).period == 2
    @test detect_period(test_fields.([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]), settings).period == 3
    @test detect_period(test_fields.([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]), settings).period == 4
    # Repeating only one phase is insufficient: every phase must recur.
    @test detect_period(test_fields.([0.0, 1.0, 0.0, 2.0, 0.0, 3.0]), settings).period == 0

    fixed_records = [
        test_record(1, test_fields(1.0), test_fields(1.0 + 1e-10)),
        test_record(2, test_fields(1.0), test_fields(1.0 + 1e-12)),
    ]
    fixed = assess_convergence(fixed_records, settings, 1.0)
    @test fixed.status == :fixed_point
    @test fixed.accepted
    @test fixed.fixed_point_contraction_estimate ≈ 0.01 rtol=1e-4
    @test fixed.fixed_point_extrapolated_rel_residual <= settings.field_rel_tol

    drift_settings = ConvergenceSettings(
        max_period=2,
        period_repeats=3,
        period_abs_tol=1e-12,
        period_rel_tol=1e-3,
        period2_oscillation_cosine_max=-0.5,
        period2_two_step_ratio_max=0.5,
    )
    monotone_history = test_fields.([1.0 + 1e-4 * index for index in 0:7])
    monotone_recurrence = detect_period(
        monotone_history,
        drift_settings;
        min_period=2,
        max_period=2,
    )
    @test monotone_recurrence.period == 0

    slow_settings = ConvergenceSettings(
        field_abs_tol=1e-12,
        field_rel_tol=5e-3,
        density_tol=1e-8,
        variational_energy_tol=1e-8,
        stable_iterations=2,
        slow_mode_cosine_min=0.9,
        stagnation_window=20,
    )
    slow_records = [
        test_record(1, test_fields(1.0), test_fields(1.004)),
        test_record(2, test_fields(1.0), test_fields(1.00392)),
    ]
    slow = assess_convergence(slow_records, slow_settings, 1.0)
    @test !slow.accepted
    @test slow.status == :iterating
    @test slow.fixed_point_contraction_estimate ≈ 0.98 rtol=1e-10
    @test slow.fixed_point_extrapolation_factor ≈ 50.0 rtol=1e-10
    @test slow.fixed_point_extrapolated_rel_residual > slow_settings.field_rel_tol

    values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    applied_values = [1.0, values[1:end - 1]...]
    cycle_records = IterationRecord[]
    for (index, value) in enumerate(values)
        correlations = test_correlations()
        profile = value == 0.0 ? [0.6, 0.4, 0.6, 0.4] : [0.4, 0.6, 0.4, 0.6]
        correlations.density_down .= profile
        correlations.density_up .= profile
        push!(cycle_records, test_record(
            index,
            test_fields(applied_values[index]),
            test_fields(value);
            energy=isodd(index) ? 0.0 : 2.0,
            update_mode=:unmixed_probe,
            correlations,
        ))
    end
    cycle = assess_convergence(cycle_records, settings, 1.0)
    @test cycle.status == :periodic_solution
    @test cycle.fundamental_period == 2
    @test cycle.accepted
    @test cycle.orbit_validated
    @test cycle.solution_canonical_variational_energy ≈ 1.0
    @test cycle.orbit_energy_spread ≈ 2.0
    @test cycle.orbit_density_contrast ≈ 0.4
    @test cycle.cycle_oscillation_cosine <= settings.period2_oscillation_cosine_max
    @test cycle.cycle_two_step_ratio <= settings.period2_two_step_ratio_max

    mixed_cycle_records = [
        test_record(index, test_fields(applied_values[index]), test_fields(value); update_mode=:anderson)
        for (index, value) in enumerate(values)
    ]
    mixed_cycle = assess_convergence(mixed_cycle_records, settings, 1.0)
    @test mixed_cycle.status == :periodic_candidate
    @test !mixed_cycle.accepted
    @test !mixed_cycle.orbit_validated

    raw_candidate = ConvergenceDiagnostic(
        status=:periodic_candidate,
        accepted=false,
        solution_kind=:periodic_orbit,
        fundamental_period=2,
        unmixed_probe=true,
    )
    continue_settings = ConvergenceSettings(probe_iterations=20, cycle_action=:continue)
    @test LadderMPSMFT._recurrence_action(
        raw_candidate,
        continue_settings;
        probe_steps=8,
        probe_origin=:initial,
        mixer_probe_completed=false,
    ) == :continue_probe
    @test LadderMPSMFT._recurrence_action(
        raw_candidate,
        continue_settings;
        probe_steps=20,
        probe_origin=:initial,
        mixer_probe_completed=false,
    ) == :enter_mixing
    @test LadderMPSMFT._recurrence_action(
        raw_candidate,
        continue_settings;
        probe_steps=20,
        probe_origin=:mixer_recurrence,
        mixer_probe_completed=true,
    ) == :stop_candidate
    mixed_candidate = ConvergenceDiagnostic(
        status=:periodic_candidate,
        accepted=false,
        solution_kind=:periodic_orbit,
        fundamental_period=2,
        unmixed_probe=false,
    )
    @test LadderMPSMFT._recurrence_action(
        mixed_candidate,
        continue_settings;
        probe_steps=0,
        probe_origin=:none,
        mixer_probe_completed=false,
    ) == :start_mixer_probe
    @test LadderMPSMFT._recurrence_action(
        mixed_candidate,
        continue_settings;
        probe_steps=0,
        probe_origin=:none,
        mixer_probe_completed=true,
    ) == :stop_candidate
end

@testset "mixing" begin
    applied = test_fields(0.0)
    measured = test_fields(1.0)
    settings = MixingSettings(method=:linear, damping=0.25, minimum_damping=0.1, maximum_damping=0.8, adaptive=false)
    state = LadderMPSMFT.MixingState(settings)
    mixed, metadata = mix_fields!(state, applied, measured, settings)
    @test mixed.mu_cdw[1, 1] ≈ 0.25
    @test metadata.method == :linear

    anderson = MixingSettings(method=:anderson, damping=0.5, minimum_damping=0.1,
        maximum_damping=0.8, adaptive=false, regularization=1e-12)
    anderson_state = LadderMPSMFT.MixingState(anderson)
    mix_fields!(anderson_state, test_fields(0.0), test_fields(1.0), anderson)
    midpoint, metadata = mix_fields!(anderson_state, test_fields(1.0), test_fields(0.0), anderson)
    @test metadata.method == :anderson
    @test midpoint.mu_cdw[1, 1] ≈ 0.5 atol=1e-10
end

@testset "DMRG stopping evidence and carried compressibility" begin
    observer = LadderMPSMFT.EnergyTimerObserver(1e-4, Inf, 3)
    @test !ITensorMPS.checkdone!(observer; energy=-10.0, sweep=1)
    @test !ITensorMPS.checkdone!(observer; energy=-10.00001, sweep=2)
    @test ITensorMPS.checkdone!(observer; energy=-10.00002, sweep=3)
    @test observer.converged
    @test observer.sweep_energies == [-10.0, -10.00001, -10.00002]

    left = (mu=0.1, density=0.9)
    right = (mu=0.2, density=1.0)
    @test LadderMPSMFT._positive_density_slope(left, right) ≈ 1.0
    @test LadderMPSMFT._positive_density_slope(right, left) ≈ 1.0
    nonmonotone = (mu=0.3, density=0.8)
    @test LadderMPSMFT._positive_density_slope(right, nonmonotone, 0.75) == 0.75
end

@testset "ladder diagnostics primitives" begin
    correlation = Matrix{Float64}(I, 4, 4)
    expectation = zeros(4)
    grid = LadderMPSMFT.structure_factor_grid(correlation, expectation, 2)
    @test size(grid.values) == (2, 2)
    @test all(grid.values .≈ 1.0)
    states = LadderMPSMFT.fixed_sector_product_state(1:8, 6, 0; rng=MersenneTwister(4))
    @test length(states) == 8
    @test count(==("Up"), states) + count(==("UpDn"), states) == 3
    @test count(==("Dn"), states) + count(==("UpDn"), states) == 3
    mktempdir() do directory
        evidence = (;
            sweep_energies=[-1.0, -1.1],
            sweep_max_discarded_weights=[1.0e-5, 2.0e-7],
            sweep_maxlinkdims=[4, 8],
            max_discarded_weight=1.0e-5,
            maximum_link_dimension=8,
            energy_converged=true,
        )
        gaps = (;
            particle_number=4,
            spin_gap=0.1,
            charge_gap=0.2,
            hole_pair_binding=-0.05,
            particle_pair_binding=-0.04,
            energies=Dict((4, 0) => -1.1),
            dmrg_evidence=Dict((4, 0) => evidence),
        )
        path = joinpath(directory, "sector_gaps.h5")
        write_sector_gaps(path, gaps; immutable=true)
        h5open(path, "r") do file
            @test Int(read(file, "schema_version")) == 2
            @test read(file, "sector_dmrg_evidence/N4_twoSz0/sweep_energy") == [-1.0, -1.1]
            @test read(file, "sector_dmrg_evidence/N4_twoSz0/sweep_max_discarded_weight") ==
                [1.0e-5, 2.0e-7]
            @test read(file, "sector_dmrg_evidence/N4_twoSz0/sweep_maxlinkdim") == [4, 8]
            @test Int(read(file, "sector_dmrg_evidence/N4_twoSz0/maximum_link_dimension")) == 8
        end
    end
end

@testset "isolated-ladder backbone and Stage 1 primitives" begin
    model = test_model()
    backbone = load_ladder_backbone_settings(joinpath(ROOT, "test", "fixtures", "phase0_tiny.toml"))
    stage1 = load_bare_stage1_settings(joinpath(ROOT, "test", "fixtures", "phase0_tiny.toml"))
    @test backbone.chi_ladder == [4, 8]
    @test stage1.top_modes == 2
    @test backbone_sector_keys(model) == [(4, 0), (4, 2), (3, 1), (2, 0), (5, 1), (6, 0)]

    sites = backbone_siteinds(model)
    product = spread_fixed_sector_product_state(sites, 4, 0)
    @test count(==("Up"), product) + count(==("UpDn"), product) == 2
    @test count(==("Dn"), product) + count(==("UpDn"), product) == 2
    psi = productMPS(sites, product)
    parity_state = ITensorMPS.removeqn(psi, "Nf")
    parity_space = sprint(show, ITensorMPS.ITensors.space(siteind(parity_state, 1)))
    @test occursin("NfParity", parity_space)
    @test !occursin("Nf,", parity_space)

    pair_diagnostics = LadderMPSMFT.sign_resolved_pair_correlations(psi, model)
    rung_bonds = [
        (rung_leg_to_site(rung, 0), rung_leg_to_site(rung, 1))
        for rung in 1:model.L
    ]
    direct_rung_addition = [
        real(inner(
            psi',
            LadderMPSMFT.singlet_pair_mpo(sites, bond_left..., bond_right...),
            psi,
        ))
        for bond_left in rung_bonds, bond_right in rung_bonds
    ]
    @test pair_diagnostics.rung ≈ direct_rung_addition atol = 1e-12
    @test minimum(eigvals(Symmetric(pair_diagnostics.rung_field))) >= -1e-12
    @test minimum(eigvals(Symmetric(pair_diagnostics.onsite0_field))) >= -1e-12

    mktempdir() do directory
        stage_record = (;
            kind=:chi,
            maxdim=8,
            energy=-2.0,
            timed_out=false,
            energy_converged=true,
            sweep_energies=[-1.9, -2.0],
            sweep_max_discarded_weights=[1e-5, 1e-7],
            sweep_maxlinkdims=[4, 8],
            max_discarded_weight=1e-5,
            maximum_link_dimension=8,
            last_five_energy_change=0.1,
            scientifically_converged=true,
        )
        checkpoint = joinpath(directory, "stage.h5")
        write_backbone_stage_checkpoint(
            checkpoint,
            psi,
            [stage_record],
            4,
            0,
            model,
            joinpath(ROOT, "test", "fixtures", "phase0_tiny.toml"),
        )
        restored = read_backbone_stage_checkpoint(checkpoint)
        @test restored.particle_number == 4
        @test restored.twice_sz == 0
        @test restored.stages[1].maxdim == 8
        @test restored.stages[1].sweep_energies == [-1.9, -2.0]
        @test length(restored.psi) == length(psi)
    end

    covariance = connected_covariance_matrix(Matrix{Float64}(I, 4, 4), zeros(4))
    blocks = leg_parity_covariance(covariance, 2)
    @test blocks.even ≈ Matrix{Float64}(I, 2, 2)
    @test blocks.odd ≈ Matrix{Float64}(I, 2, 2)
    @test blocks.cross_relative_norm ≈ 0.0
    spectrum = covariance_eigensystem(blocks.even, model; top_modes=2)
    @test spectrum.eigenvalues ≈ ones(2)
    @test maximum(spectrum.residuals) < 1e-12
    @test last_five_sweep_change([-2.0, -2.1, -2.11]) ≈ 0.11

    stage2 = load_bare_stage2_settings(
        joinpath(ROOT, "test", "fixtures", "phase0_tiny.toml"),
        model,
    )
    @test stage2.charge_even_modes == [1]
    kernel_model = LadderMPSMFT._model_with_geometry(model, :square; ep_signed=-0.5)
    @test kernel_model.ep == 0.5
    @test kernel_model.ep_signed == -0.5
    zero = zero_field_state(model)
    @test field_metric_norm(zero, model) == 0.0
    stage1_result = compute_bare_stage1(psi, model, stage1)
    mktempdir() do directory
        stage1_path = joinpath(directory, "stage1.h5")
        write_bare_stage1(stage1_path, stage1_result, model; immutable=true)
        candidates = build_stage2_candidates(stage1_path, model, stage2)
        @test length(candidates) == 10
        @test count(candidate -> candidate.block == :pair, candidates) == 5
        pair_bank = orthonormalize_stage2_candidates(
            filter(candidate -> candidate.block == :pair, candidates),
            model;
            tolerance=stage2.orthogonalization_tol,
        )
        @test length(pair_bank.basis) == 3
        @test pair_bank.maximum_orthogonality_error < 1e-12
        @test pair_bank.candidate_retained_basis_index[4:5] == [0, 0]

        direction = first(filter(candidate -> candidate.block == :normal, candidates)).fields
        @test field_metric_norm(direction, model) ≈ 1.0 atol=1e-12
        measured_fields, raw_correlations = calculate_mean_fields(psi, model)
        @test mean_fields_from_correlations(raw_correlations, model).mu_cdw ≈
            measured_fields.mu_cdw
        h0 = build_mf_mpo(sites, model, zero, 0.0)
        h1 = build_mf_mpo(sites, model, direction, 0.0)
        direct_field_expectation = real(inner(psi', h1, psi) - inner(psi', h0, psi))
        @test field_conjugate_expectation(direction, raw_correlations, model) ≈
            direct_field_expectation atol=1e-11
    end

    sectors = [
        (particle_number=number, twice_sz=sz, energy=energy)
        for ((number, sz), energy) in zip(
            backbone_sector_keys(model),
            (-10.0, -9.5, -8.0, -6.5, -7.8, -6.0),
        )
    ]
    summary = backbone_energy_summary(sectors, model)
    @test summary.spin_gap ≈ 0.5
    @test summary.chemical_potential ≈ 0.125
    @test summary.hole_pair_binding ≈ -0.5
    @test summary.particle_pair_binding ≈ -0.4
end

@testset "checkpoint schema and strict selection" begin
    mktempdir() do directory
        model = test_model()
        run = RunSettings(output_directory=directory, quick_diagnostics=false)
        settings = ProjectSettings(model=model, run=run)
        sites = LadderMPSMFT.make_sites(model)
        psi = productMPS(sites, density_product_state(4, 1.0; rng=MersenneTwister(2)))
        psi_float32 = LadderMPSMFT.convert_tensor_scalar_type(psi, :float32)
        psi_float64 = move_to_backend(
            psi_float32,
            RuntimeSettings(backend=:cpu, tensor_scalar_type=:float64),
        )
        @test all(tensor -> eltype(tensor) == Float64, psi_float64)
        record = test_record(1, test_fields(0.0), test_fields(0.5))
        history_record = test_record(2, test_fields(0.25), test_fields(0.75))
        diagnostic = ConvergenceDiagnostic(
            status=:fixed_point,
            accepted=true,
            reason="test fixed point",
            solution_kind=:fixed_point,
            fundamental_period=1,
            solution_canonical_variational_energy=0.0,
            orbit_energy_spread=0.0,
            orbit_density_contrast=0.0,
            fixed_point_abs_residual=0.0,
            fixed_point_rel_residual=0.0,
            cycle_abs_residual=0.0,
            cycle_rel_residual=0.0,
            density_error=0.0,
            variational_energy_change=0.0,
            hamiltonian_identity_error_per_site=0.0,
            effective_eigenvalue_error_per_site=0.0,
            best_iteration=1,
        )
        run_directory = joinpath(directory, "nested", "run")
        path = joinpath(run_directory, "state.h5")
        write_checkpoint(path; settings, psi, records=[record, history_record], diagnostic,
            restart_fields=test_fields(0.25), chemical_potential=0.4, immutable=true)
        checkpoint = read_checkpoint(path)
        @test checkpoint.accepted
        @test checkpoint.chemical_potential ≈ 0.4
        @test checkpoint.restart.mu_cdw[1, 1] ≈ 0.25
        measured_history = read_field_history(path; source=:measured)
        applied_history = read_field_history(path; source=:applied)
        measured_with_seed = read_field_history(path; source=:measured, include_seed=true)
        @test measured_history.iterations == [1, 2]
        @test size(measured_history.alpha) == (2, 2, 2, 2, 2)
        @test size(measured_history.beta) == (2, 2, 2, 2, 2, 2)
        @test size(measured_history.mu_cdw) == (2, 4, 2)
        @test measured_history.mu_cdw[1, 1, :] == [0.5, 0.75]
        @test applied_history.mu_cdw[1, 1, :] == [0.0, 0.25]
        @test measured_with_seed.iterations == [0, 1, 2]
        @test size(measured_with_seed.alpha) == (2, 2, 2, 2, 3)
        @test size(measured_with_seed.beta) == (2, 2, 2, 2, 2, 3)
        @test size(measured_with_seed.mu_cdw) == (2, 4, 3)
        @test measured_with_seed.mu_cdw[1, 1, :] == [0.0, 0.5, 0.75]
        refactored_inherit = read_inherited_fields(path)
        @test refactored_inherit.format == :refactored
        @test refactored_inherit.fields.mu_cdw[1, 1] == 0.25
        @test refactored_inherit.chemical_potential == 0.4
        h5open(path, "r") do file
            @test Int(read(file, "schema_version")) == 7
            @test read(file, "fields/initial/mu_cdw")[1, 1] == 0.0
            @test Int(read(file, "history/fields/seed_iteration")) == 0
            @test read(file, "history/fields/seed/mu_cdw")[1, 1] == 0.0
            @test haskey(file, "solution_target_density_corrected_variational_energy")
            @test haskey(file, "fixed_point_extrapolated_rel_residual")
            @test haskey(file, "history/mu_density_slope")
            @test haskey(file, "history/target_density_corrected_variational_energy")
            @test haskey(file, "history/dmrg/0001/sweep_energy")
            @test haskey(file, "history/dmrg/0002/sweep_max_discarded_weight")
            @test haskey(file, "history/dmrg/0002/sweep_maxlinkdim")
        end
        @test_throws ArgumentError write_checkpoint(path; settings, psi, records=[record], diagnostic, immutable=true)

        legacy_path = joinpath(directory, "legacy_fields.h5")
        h5open(legacy_path, "w") do file
            file["alpha"] = test_fields(0.0).alpha
            file["beta"] = test_fields(0.0).beta
            file["mu"] = 1.25
        end
        inherited = read_inherited_fields(legacy_path)
        @test inherited.format == :legacy
        @test inherited.chemical_potential == 1.25
        @test size(inherited.fields.mu_cdw) == (2, 4)
        @test all(iszero, inherited.fields.mu_cdw)
        inherit_settings = ProjectSettings(
            model=model,
            run=RunSettings(
                output_directory=directory,
                inherit_from=legacy_path,
                inherit_sha256=LadderMPSMFT.sha256_file(legacy_path),
                quick_diagnostics=false,
            ),
        )
        inherited_start = LadderMPSMFT._initial_state(inherit_settings)
        @test inherited_start.source == "field_inherit"
        @test inherited_start.inherit_format == "legacy"
        @test inherited_start.chemical_potential == 1.25
        @test length(inherited_start.psi) == 4
        bad_hash_settings = ProjectSettings(
            model=model,
            run=RunSettings(
                inherit_from=legacy_path,
                inherit_sha256=repeat("0", 64),
                quick_diagnostics=false,
            ),
        )
        @test_throws ArgumentError LadderMPSMFT._initial_state(bad_hash_settings)
        gpu_resume_settings = ProjectSettings(
            model=model,
            runtime=RuntimeSettings(
                backend=:gpu,
                threaded_blocksparse=false,
                conserve_sz=false,
                conserve_nfparity=false,
            ),
            run=RunSettings(
                output_directory=directory,
                resume_checkpoint=path,
                resume_sha256=LadderMPSMFT.sha256_file(path),
                quick_diagnostics=false,
            ),
        )
        @test_throws ArgumentError LadderMPSMFT._initial_state(gpu_resume_settings)
        selected = select_completed_runs(directory)
        @test length(selected) == 1
        @test only(selected).path == path
        @test only(selected).plot_style == "solid"

        second_settings = ProjectSettings(
            model=model,
            run=RunSettings(output_directory=directory, branch_label="sdw", quick_diagnostics=false),
        )
        second_path = joinpath(directory, "nested", "second", "state.h5")
        second_record = test_record(1, test_fields(0.0), test_fields(0.0); energy=-0.1)
        second_diagnostic = ConvergenceDiagnostic(
            status=:fixed_point,
            accepted=true,
            reason="test fixed point",
            solution_kind=:fixed_point,
            fundamental_period=1,
            solution_canonical_variational_energy=-0.1,
            orbit_energy_spread=0.0,
            orbit_density_contrast=0.0,
            fixed_point_abs_residual=0.0,
            fixed_point_rel_residual=0.0,
            cycle_abs_residual=0.0,
            cycle_rel_residual=0.0,
            density_error=0.0,
            variational_energy_change=0.0,
            hamiltonian_identity_error_per_site=0.0,
            effective_eigenvalue_error_per_site=0.0,
            best_iteration=1,
        )
        write_checkpoint(second_path; settings=second_settings, psi, records=[second_record], diagnostic=second_diagnostic,
            restart_fields=test_fields(0.0), chemical_potential=0.0, immutable=true)
        ranking = compare_variational_branches([path, second_path])
        @test first(ranking).path == second_path
        @test first(ranking).energy ≈ -0.1

        periodic_path = joinpath(directory, "nested", "periodic", "state.h5")
        periodic_records = [
            test_record(1, test_fields(1.0), test_fields(0.0); energy=-0.2, update_mode=:unmixed_probe),
            test_record(2, test_fields(0.0), test_fields(1.0); energy=-0.4, update_mode=:unmixed_probe),
        ]
        periodic_diagnostic = ConvergenceDiagnostic(
            status=:periodic_solution,
            accepted=true,
            reason="test periodic solution",
            solution_kind=:periodic_orbit,
            fundamental_period=2,
            orbit_validated=true,
            unmixed_probe=true,
            solution_canonical_variational_energy=-0.3,
            orbit_energy_spread=0.2,
            orbit_density_contrast=0.5,
            cycle_abs_residual=0.0,
            cycle_rel_residual=0.0,
            density_error=0.0,
            variational_energy_change=0.0,
            hamiltonian_identity_error_per_site=0.0,
            effective_eigenvalue_error_per_site=0.0,
            best_iteration=2,
        )
        phase_psis = Dict(1 => deepcopy(psi), 2 => deepcopy(psi))
        write_checkpoint(
            periodic_path;
            settings,
            psi,
            records=periodic_records,
            diagnostic=periodic_diagnostic,
            restart_fields=test_fields(0.0),
            chemical_potential=0.0,
            immutable=true,
            phase_psis,
        )
        periodic_checkpoint = read_checkpoint(periodic_path)
        @test periodic_checkpoint.accepted
        @test periodic_checkpoint.fundamental_period == 2
        @test periodic_checkpoint.orbit_validated
        first_phase = read_checkpoint(periodic_path; orbit_phase=1)
        second_phase = read_checkpoint(periodic_path; orbit_phase=2)
        @test first_phase.applied.alpha == test_fields(1.0).alpha
        @test first_phase.measured.alpha == test_fields(0.0).alpha
        @test first_phase.restart.alpha == first_phase.measured.alpha
        @test second_phase.applied.alpha == test_fields(0.0).alpha
        @test second_phase.measured.alpha == test_fields(1.0).alpha
        @test_throws ArgumentError read_checkpoint(periodic_path; orbit_phase=3)
        phase_parent_settings = ProjectSettings(
            model=model,
            run=RunSettings(
                output_directory=directory,
                parent_checkpoint=periodic_path,
                parent_sha256=LadderMPSMFT.sha256_file(periodic_path),
                parent_orbit_phase=1,
                quick_diagnostics=false,
            ),
        )
        phase_parent_start = LadderMPSMFT._initial_state(phase_parent_settings)
        @test phase_parent_start.source == "parent_orbit_phase_001"
        @test phase_parent_start.fields.alpha == test_fields(0.0).alpha
        @test length(read_orbit_phase_states(periodic_path)) == 2
        stateless_path = joinpath(directory, "stateless_periodic.h5")
        stateless = write_stateless_copy(periodic_path, stateless_path)
        @test stateless.source_bytes > stateless.compact_bytes
        h5open(stateless_path, "r") do file
            @test !haskey(file, "psi")
            @test !haskey(file, "cycle_members/001/psi")
            @test !haskey(file, "cycle_members/002/psi")
            @test haskey(file, "history/fields/applied/alpha")
            @test haskey(file, "history/fields/seed/alpha")
            @test Bool(read(file, "analysis_storage/is_stateless_copy"))
            @test !Bool(read(file, "analysis_storage/restartable"))
            @test String(read(file, "analysis_storage/full_artifact_sha256")) ==
                LadderMPSMFT.sha256_file(periodic_path)
        end
        @test read_field_history(stateless_path; source=:measured).iterations == [1, 2]
        @test read_field_history(
            stateless_path;
            source=:measured,
            include_seed=true,
        ).iterations == [0, 1, 2]
        @test read_inherited_fields(stateless_path).format == :refactored
        @test_throws ArgumentError read_checkpoint(stateless_path)
        @test_throws ArgumentError read_checkpoint(stateless_path; orbit_phase=1)
        @test_throws ArgumentError read_orbit_phase_states(stateless_path)

        full_tree = joinpath(directory, "full_tree")
        compact_tree = joinpath(directory, "compact_tree")
        mkpath(joinpath(full_tree, "nested"))
        cp(periodic_path, joinpath(full_tree, "nested", "artifact.h5"))
        write(joinpath(full_tree, "nested", "run_summary.md"), "test summary\n")
        mirror = mirror_stateless_tree(full_tree, compact_tree)
        @test length(mirror.records) == 2
        @test isfile(joinpath(compact_tree, "stateless_manifest.tsv"))
        @test isfile(joinpath(compact_tree, "nested", "run_summary.md"))
        h5open(joinpath(compact_tree, "nested", "artifact.h5"), "r") do file
            @test !haskey(file, "psi")
            @test haskey(file, "history/fields/measured/beta")
        end

        pair_binding_path = joinpath(directory, "pair_binding.h5")
        h5open(pair_binding_path, "w") do file
            file["E_N_120"] = -10.0
            file["psi_N_120"] = Float64[1, 2, 3]
        end
        pair_binding_stateless = joinpath(directory, "pair_binding_stateless.h5")
        write_stateless_copy(pair_binding_path, pair_binding_stateless)
        h5open(pair_binding_stateless, "r") do file
            @test haskey(file, "E_N_120")
            @test !haskey(file, "psi_N_120")
        end
        ranking = compare_variational_branches([path, second_path, periodic_path])
        @test first(ranking).path == periodic_path
        @test first(ranking).energy ≈ -0.3

        invalid_periodic_path = joinpath(directory, "nested", "invalid-periodic", "state.h5")
        invalid_periodic_diagnostic = ConvergenceDiagnostic(
            status=:periodic_solution,
            accepted=true,
            reason="deliberately inconsistent test artifact",
            solution_kind=:periodic_orbit,
            fundamental_period=2,
            orbit_validated=false,
            unmixed_probe=false,
            solution_canonical_variational_energy=-1.0,
            best_iteration=2,
        )
        write_checkpoint(
            invalid_periodic_path;
            settings,
            psi,
            records=periodic_records,
            diagnostic=invalid_periodic_diagnostic,
            restart_fields=test_fields(0.0),
            chemical_potential=0.0,
            immutable=true,
        )
        @test length(select_completed_runs(directory)) == 3
        included = select_completed_runs(directory; include_incomplete=true)
        invalid_row = only(filter(row -> row.path == invalid_periodic_path, included))
        @test invalid_row.plot_style == "hatched"
        @test_throws ArgumentError compare_variational_branches([path, invalid_periodic_path])
    end
end
