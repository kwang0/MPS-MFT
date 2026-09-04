#!/usr/bin/env julia

using LadderMPSMFT
using Random
using TOML

length(ARGS) in (4, 5) || error(
    "usage: julia --project=. scripts/prepare_phase1_square_smooth_pairing_grid.jl " *
    "BASE_CONFIG.toml CONTROL_RUN FULL_RUN RUN_ID [square|cubic_unfrustrated]",
)

const COMMON_RANDOM_SEED = 1404
const INITIAL_AMPLITUDE = 1.0e-3
const TARGET_GEOMETRY = length(ARGS) == 5 ? Symbol(ARGS[5]) : :square
TARGET_GEOMETRY in (:square, :cubic_unfrustrated) || error(
    "unsupported smooth-pairing grid geometry: $TARGET_GEOMETRY",
)
const SQUARE_POINT_CONTRACTS = (
    (
        point_id="square_t010_vm04",
        label="square__smooth_pairing_t010_vm04_chi200_loose",
        V=-0.4,
        t0=1.0,
        mu_initial=0.55,
        mu_initial_basis="rounded_current_t014_vm04_endpoint",
        ep_signed=-0.17882744409052975,
    ),
    (
        point_id="square_t010_vm02",
        label="square__smooth_pairing_t010_vm02_chi200_loose",
        V=-0.2,
        t0=1.0,
        mu_initial=1.10,
        mu_initial_basis="midpoint_of_current_t014_vm04_and_v000_endpoints",
        ep_signed=-0.1545120066237189,
    ),
    (
        point_id="square_t012_vm04",
        label="square__smooth_pairing_t012_vm04_chi200_loose",
        V=-0.4,
        t0=1.2,
        mu_initial=0.55,
        mu_initial_basis="rounded_current_t014_vm04_endpoint",
        ep_signed=-0.25124588461187614,
    ),
    (
        point_id="square_t012_vm02",
        label="square__smooth_pairing_t012_vm02_chi200_loose",
        V=-0.2,
        t0=1.2,
        mu_initial=1.10,
        mu_initial_basis="midpoint_of_current_t014_vm04_and_v000_endpoints",
        ep_signed=-0.21453418655934797,
    ),
    (
        point_id="square_t012_v000",
        label="square__smooth_pairing_t012_v000_chi200_loose",
        V=0.0,
        t0=1.2,
        mu_initial=1.65,
        mu_initial_basis="rounded_current_t014_v000_endpoint",
        ep_signed=-0.17989619749147323,
    ),
)
const CUBIC_UNFRUSTRATED_POINT_CONTRACTS = (
    (
        point_id="cubic_unfrustrated_t010_vm04",
        label="cubic_unfrustrated__smooth_pairing_t010_vm04_chi200_loose",
        V=-0.4,
        t0=1.0,
        mu_initial=0.55,
        mu_initial_basis="V_informed_grid_bracket_guide",
        ep_signed=-0.17882744409052975,
    ),
    (
        point_id="cubic_unfrustrated_t010_vm02",
        label="cubic_unfrustrated__smooth_pairing_t010_vm02_chi200_loose",
        V=-0.2,
        t0=1.0,
        mu_initial=1.10,
        mu_initial_basis="V_informed_grid_bracket_guide",
        ep_signed=-0.1545120066237189,
    ),
    (
        point_id="cubic_unfrustrated_t012_vm04",
        label="cubic_unfrustrated__smooth_pairing_t012_vm04_chi200_loose",
        V=-0.4,
        t0=1.2,
        mu_initial=0.55,
        mu_initial_basis="V_informed_grid_bracket_guide",
        ep_signed=-0.25124588461187614,
    ),
    (
        point_id="cubic_unfrustrated_t012_vm02",
        label="cubic_unfrustrated__smooth_pairing_t012_vm02_chi200_loose",
        V=-0.2,
        t0=1.2,
        mu_initial=1.10,
        mu_initial_basis="V_informed_grid_bracket_guide",
        ep_signed=-0.21453418655934797,
    ),
    (
        point_id="cubic_unfrustrated_t012_v000",
        label="cubic_unfrustrated__smooth_pairing_t012_v000_chi200_loose",
        V=0.0,
        t0=1.2,
        mu_initial=1.65,
        mu_initial_basis="V_informed_grid_bracket_guide",
        ep_signed=-0.17989619749147323,
    ),
    (
        point_id="cubic_unfrustrated_t014_vm04",
        label="cubic_unfrustrated__smooth_pairing_t014_vm04_chi200_loose",
        V=-0.4,
        t0=1.4,
        mu_initial=0.55,
        mu_initial_basis="V_informed_grid_bracket_guide",
        ep_signed=-0.24962435880865996,
    ),
    (
        point_id="cubic_unfrustrated_t014_vm02",
        label="cubic_unfrustrated__smooth_pairing_t014_vm02_chi200_loose",
        V=-0.2,
        t0=1.4,
        mu_initial=1.10,
        mu_initial_basis="V_informed_grid_bracket_guide",
        ep_signed=-0.2068002629740704,
    ),
    (
        point_id="cubic_unfrustrated_t014_v000",
        label="cubic_unfrustrated__smooth_pairing_t014_v000_chi200_loose",
        V=0.0,
        t0=1.4,
        mu_initial=1.65,
        mu_initial_basis="V_informed_grid_bracket_guide",
        ep_signed=-0.14653773091916378,
    ),
)
const POINT_CONTRACTS = if TARGET_GEOMETRY == :square
    SQUARE_POINT_CONTRACTS
else
    CUBIC_UNFRUSTRATED_POINT_CONTRACTS
end

base_path = abspath(ARGS[1])
control_run = abspath(ARGS[2])
full_run = abspath(ARGS[3])
run_id = ARGS[4]

isfile(base_path) || error("smooth-pairing grid base configuration not found: $base_path")
isdir(full_run) || error("full scratch output directory not found: $full_run")
occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID: $run_id")

base = load_settings(base_path)
base.model.geometry == TARGET_GEOMETRY || error(
    "grid campaign must use $TARGET_GEOMETRY geometry",
)
base.model.L == 64 || error("grid campaign must use L=64")
base.model.U == 8.0 || error("grid campaign must use U=8")
base.model.V == -0.4 || error("base grid configuration must use V=-0.4")
base.model.t0 == 1.0 || error("base grid configuration must use t0=1.0")
base.model.tp == 0.1 || error("grid campaign must use t_perp=0.1")
base.model.density == 0.9375 || error("grid campaign must use density=0.9375")
base.model.mu_initial == 0.55 || error("base grid configuration must begin at mu=0.55")
base.model.ep_mode == :exact || error("grid campaign requires exact E_p registry rows")
base.model.ep_signed == first(POINT_CONTRACTS).ep_signed || error(
    "base grid configuration resolved an unexpected E_p=$(base.model.ep_signed)",
)
base.runtime.backend == :gpu || error("grid campaign must use the GPU backend")
base.runtime.tensor_scalar_type == :float64 || error("grid campaign must use Float64 tensors")
base.dmrg.nsweeps == 12 || error("grid campaign must use 12 DMRG sweeps")
base.dmrg.maxdim == 200 || error("grid campaign must use chi=200")
base.dmrg.cutoff == 1.0e-10 || error("grid campaign must use cutoff=1e-10")
base.dmrg.energy_tol == 1.0e-6 || error("grid campaign must use DMRG energy_tol=1e-6")
base.dmrg.max_time_seconds == 41400.0 || error("grid campaign DMRG deadline changed")
base.dmrg.mu_density_tol == 1.0e-3 || error("inner density tolerance must be 1e-3")
base.dmrg.mu_max_iterations == 16 || error("mu search must allow 16 evaluations")
base.dmrg.mu_bracket_step == 0.01 || error("initial mu bracket step must be 0.01")
base.dmrg.mu_bracket_growth == 3.0 || error("mu bracket growth must be 3")
base.dmrg.mu_warm_start_noise == 1.0e-8 || error(
    "warm-started mu re-solves must begin with noise 1e-8",
)
base.convergence.density_tol == 1.0e-3 || error("outer density tolerance must be 1e-3")
base.convergence.field_abs_tol == 1.0e-6 || error("field absolute tolerance must be 1e-6")
base.convergence.field_rel_tol == 5.0e-3 || error("field relative tolerance must be 5e-3")
base.convergence.variational_energy_tol == 1.0e-6 || error(
    "variational energy tolerance must be 1e-6 per physical site",
)
base.convergence.period2_oscillation_cosine_max == -0.5 || error(
    "period-two classification must require a negative step cosine",
)
base.convergence.period2_two_step_ratio_max == 0.5 || error(
    "period-two classification must require d2/d1 <= 0.5",
)
base.convergence.slow_mode_cosine_min == 0.9 || error(
    "fixed-point acceptance must retain the slow-mode extrapolation gate",
)
base.convergence.unmixed_cycle_probe || error("grid campaign must begin with a raw-map probe")
base.convergence.probe_iterations == 20 || error("raw-map probe must allow 20 updates")
base.convergence.cycle_action == :continue || error(
    "unaccepted raw recurrence must be archived before optional Anderson acceleration",
)
base.run.max_iterations == 80 || error("grid campaign must allow up to 80 MF updates")
base.run.initial_seed == :legacy_pairing || error("grid campaign must use legacy_pairing")
base.run.initial_seed_protocol == :matched_mode || error(
    "smooth mixed pairing requires the matched_mode protocol",
)
base.run.initial_amplitude == INITIAL_AMPLITUDE || error(
    "smooth mixed-pairing norm must be 1e-3",
)
base.run.initial_mode_number == 0 || error("smooth mixed pairing must use mode number zero")
base.run.initial_mode_phase_pi == 0.0 || error("smooth mixed pairing must use phase zero")
base.run.initial_pairing_form_factor == :onsite_s || error(
    "legacy_pairing requires onsite_s as its unused form-factor sentinel",
)
base.run.initial_leg_parity == :auto || error("smooth mixed pairing must draw all leg-pair classes")
base.run.random_seed == COMMON_RANDOM_SEED || error(
    "grid campaign must use common random seed $COMMON_RANDOM_SEED",
)
base.run.inherit_from === nothing || error("grid campaign must not inherit fields")
base.run.parent_checkpoint === nothing || error("grid campaign must not use parent MPS states")
base.run.resume_checkpoint === nothing || error("grid campaign must not resume checkpoints")

config_directory = joinpath(control_run, "configs")
ispath(config_directory) && !isempty(readdir(config_directory)) && error(
    "refusing to overwrite nonempty configuration directory: $config_directory",
)
mkpath(config_directory)
mkpath(joinpath(control_run, "results"))
mkpath(joinpath(full_run, "results"))

manifest_path = joinpath(control_run, "manifest.tsv")
ispath(manifest_path) && error("refusing to overwrite manifest: $manifest_path")
model_fingerprints = String[]
numerical_fingerprints = String[]
seed_fingerprints = String[]
implementation_fingerprints = String[]
ep_source_hashes = String[]

open(manifest_path, "w") do io
    println(io, join((
        "label", "geometry", "point_id", "L", "U", "V", "t0", "tp", "density", "chi",
        "grid_role", "config", "config_sha256", "model_fingerprint",
        "numerical_fingerprint", "implementation_sha256", "ep_source_sha256",
        "ep_mode", "ep_signed", "ep_abs", "ep_t0_lower", "ep_t0_upper",
        "ep_lower_signed", "ep_upper_signed", "ep_weight", "tp2_over_ep",
        "initial_mu", "initial_mu_basis", "full_output_directory",
        "stateless_output_directory", "initial_seed", "initial_seed_protocol",
        "initial_seed_fingerprint", "initial_amplitude", "random_seed",
        "initial_mode_number", "initial_mode_phase_pi", "initial_pairing_form_factor",
        "initial_leg_parity_requested", "initial_leg_parity_resolved",
        "initial_seed_normalization", "legacy_pairing_center_of_mass_structure",
        "legacy_pairing_beta_initialization", "legacy_pairing_mu_cdw_initialization",
        "center_of_mass_spatial_noise", "pairing_sector_open", "normal_sector_lock",
        "parent_or_inherit", "cross_point_energy_ranking_authorized",
        "dmrg_sweeps", "dmrg_energy_tol", "dmrg_cutoff", "mu_density_tol",
        "outer_density_tol", "field_abs_tol", "field_rel_tol", "variational_energy_tol",
        "mu_bracket_step", "mu_bracket_growth", "mu_warm_start_noise",
        "period2_oscillation_cosine_max", "period2_two_step_ratio_max",
        "slow_mode_cosine_min", "raw_probe_updates", "max_mf_updates",
    ), '\t'))

    for point in POINT_CONTRACTS
        raw = TOML.parsefile(base_path)
        model_raw = raw["model"]
        model_raw["V"] = point.V
        model_raw["t0"] = point.t0
        model_raw["mu_initial"] = point.mu_initial

        run = raw["run"]
        full_output_directory = joinpath(full_run, "results", point.label)
        stateless_output_directory = joinpath(control_run, "results", point.label)
        run["output_directory"] = full_output_directory
        run["branch_label"] = "smooth_mixed_pairing_grid_control"
        run["preparation"] = "$(TARGET_GEOMETRY)_smooth_mixed_pairing_independent_grid_fill"
        run["direction"] = "none"
        run["seed_label"] = "smooth_mixed_pairing_common_s$(COMMON_RANDOM_SEED)"
        run["random_seed"] = COMMON_RANDOM_SEED
        run["initial_seed"] = "legacy_pairing"
        run["initial_amplitude"] = INITIAL_AMPLITUDE
        run["initial_seed_protocol"] = "matched_mode"
        run["initial_mode_number"] = 0
        run["initial_mode_phase_pi"] = 0.0
        run["initial_pairing_form_factor"] = "onsite_s"
        run["initial_leg_parity"] = "auto"
        run["initial_stripe_pairing_to_spin_ratio"] = 0.0
        for key in (
            "inherit_from", "inherit_sha256", "parent_checkpoint", "parent_sha256",
            "parent_orbit_phase", "resume_checkpoint", "resume_sha256",
        )
            pop!(run, key, nothing)
        end

        config_path = joinpath(config_directory, "$(point.label).segment-001.toml")
        ispath(config_path) && error("refusing to overwrite grid config: $config_path")
        open(config_path, "w") do config_io
            TOML.print(config_io, raw; sorted=true)
        end

        settings = load_settings(config_path)
        model = settings.model
        metadata = initial_seed_metadata(model, settings.run)
        model.geometry == TARGET_GEOMETRY || error(
            "prepared point changed geometry: $(point.point_id)",
        )
        model.V == point.V || error("prepared point changed V: $(point.point_id)")
        model.t0 == point.t0 || error("prepared point changed t0: $(point.point_id)")
        model.mu_initial == point.mu_initial || error(
            "prepared point changed initial mu: $(point.point_id)",
        )
        model.ep_mode == :exact || error("prepared point did not resolve exact E_p: $(point.point_id)")
        model.ep_signed == point.ep_signed || error(
            "prepared point resolved E_p=$(model.ep_signed), expected $(point.ep_signed)",
        )
        settings.run.initial_seed == :legacy_pairing || error(
            "prepared point changed seed channel: $(point.point_id)",
        )
        settings.run.random_seed == COMMON_RANDOM_SEED || error(
            "prepared point changed common RNG seed: $(point.point_id)",
        )
        metadata.legacy_pairing_center_of_mass_structure ==
            "constant_by_relative_offset_and_leg_pair" || error(
                "prepared point introduced center-of-mass pairing noise: $(point.point_id)",
            )
        metadata.legacy_pairing_beta_initialization == "zero" || error(
            "prepared point must initialize beta to zero: $(point.point_id)",
        )
        metadata.legacy_pairing_mu_cdw_initialization == "zero" || error(
            "prepared point must initialize mu_cdw to zero: $(point.point_id)",
        )

        seed_fields = initial_fields(
            model;
            seed=settings.run.initial_seed,
            amplitude=settings.run.initial_amplitude,
            rng=MersenneTwister(settings.run.random_seed),
            protocol=settings.run.initial_seed_protocol,
            mode_number=settings.run.initial_mode_number,
            mode_phase_pi=settings.run.initial_mode_phase_pi,
            pairing_form_factor=settings.run.initial_pairing_form_factor,
            leg_parity=settings.run.initial_leg_parity,
            stripe_charge_to_spin_ratio=settings.run.initial_stripe_charge_to_spin_ratio,
            stripe_pairing_to_spin_ratio=settings.run.initial_stripe_pairing_to_spin_ratio,
            random_seed=settings.run.random_seed,
        )
        all(iszero, seed_fields.beta) || error("smooth pairing seed has nonzero beta")
        all(iszero, seed_fields.mu_cdw) || error("smooth pairing seed has nonzero mu_cdw")
        any(value -> !iszero(value), seed_fields.alpha) || error(
            "smooth pairing seed has zero alpha",
        )
        isapprox(
            field_l2_per_physical_site(seed_fields, model),
            INITIAL_AMPLITUDE;
            atol=1.0e-14,
            rtol=1.0e-12,
        ) || error("smooth pairing seed lost its matched total norm")
        for offset in 0:model.r_range, leg in 1:2, other_leg in 1:2
            values = [
                seed_fields.alpha[rung, rung + offset, leg, other_leg]
                for rung in 1:(model.L - offset)
            ]
            maximum(abs.(values .- first(values))) <= 1.0e-14 || error(
                "smooth pairing seed varies along the ladder at offset=$offset, legs=$leg,$other_leg",
            )
        end

        model_fp = LadderMPSMFT.model_fingerprint(model)
        numerical_fp = LadderMPSMFT.numerical_fingerprint(settings)
        seed_fp = LadderMPSMFT.initial_seed_fingerprint(settings)
        implementation_fp = implementation_fingerprint(settings)
        ep_source_sha256 = LadderMPSMFT.sha256_file(model.ep_source)
        push!(model_fingerprints, model_fp)
        push!(numerical_fingerprints, numerical_fp)
        push!(seed_fingerprints, seed_fp)
        push!(implementation_fingerprints, implementation_fp)
        push!(ep_source_hashes, ep_source_sha256)

        println(io, join((
            point.label,
            String(model.geometry),
            point.point_id,
            string(model.L),
            string(model.U),
            string(model.V),
            string(model.t0),
            string(model.tp),
            string(model.density),
            string(settings.dmrg.maxdim),
            "missing_3x3_grid_point_smooth_pairing_control",
            config_path,
            LadderMPSMFT.sha256_file(config_path),
            model_fp,
            numerical_fp,
            implementation_fp,
            ep_source_sha256,
            String(model.ep_mode),
            string(model.ep_signed),
            string(model.ep),
            string(model.ep_t0_lower),
            string(model.ep_t0_upper),
            string(model.ep_lower_signed),
            string(model.ep_upper_signed),
            string(model.ep_interpolation_weight),
            string(model.tp^2 / model.ep),
            string(model.mu_initial),
            point.mu_initial_basis,
            full_output_directory,
            stateless_output_directory,
            String(settings.run.initial_seed),
            String(settings.run.initial_seed_protocol),
            seed_fp,
            string(settings.run.initial_amplitude),
            string(settings.run.random_seed),
            string(metadata.mode_number),
            string(metadata.mode_phase_pi),
            String(metadata.pairing_form_factor),
            String(metadata.requested_leg_parity),
            String(metadata.resolved_leg_parity),
            metadata.normalization,
            metadata.legacy_pairing_center_of_mass_structure,
            metadata.legacy_pairing_beta_initialization,
            metadata.legacy_pairing_mu_cdw_initialization,
            "false",
            "true",
            "false",
            "none",
            "false",
            string(settings.dmrg.nsweeps),
            string(settings.dmrg.energy_tol),
            string(settings.dmrg.cutoff),
            string(settings.dmrg.mu_density_tol),
            string(settings.convergence.density_tol),
            string(settings.convergence.field_abs_tol),
            string(settings.convergence.field_rel_tol),
            string(settings.convergence.variational_energy_tol),
            string(settings.dmrg.mu_bracket_step),
            string(settings.dmrg.mu_bracket_growth),
            string(settings.dmrg.mu_warm_start_noise),
            string(settings.convergence.period2_oscillation_cosine_max),
            string(settings.convergence.period2_two_step_ratio_max),
            string(settings.convergence.slow_mode_cosine_min),
            string(settings.convergence.probe_iterations),
            string(settings.run.max_iterations),
        ), '\t'))
    end
end

length(unique(model_fingerprints)) == length(POINT_CONTRACTS) || error(
    "the $(length(POINT_CONTRACTS)) distinct physical grid points did not receive distinct model fingerprints",
)
length(unique(numerical_fingerprints)) == 1 || error(
    "grid points do not share one numerical fingerprint",
)
length(unique(seed_fingerprints)) == 1 || error(
    "grid points do not share one initial-seed fingerprint",
)
length(unique(implementation_fingerprints)) == 1 || error(
    "grid points do not share one solver implementation fingerprint",
)
length(unique(ep_source_hashes)) == 1 || error("grid points do not share one E_p registry")

println("branch_count=$(length(POINT_CONTRACTS))")
println("geometry=$TARGET_GEOMETRY")
println("common_random_seed=$COMMON_RANDOM_SEED")
println("initial_seed=legacy_pairing")
println("initial_amplitude=$INITIAL_AMPLITUDE")
println("center_of_mass_spatial_noise=false")
println("numerical_fingerprint=$(only(unique(numerical_fingerprints)))")
println("initial_seed_fingerprint=$(only(unique(seed_fingerprints)))")
println("implementation_sha256=$(only(unique(implementation_fingerprints)))")
println("ep_source_sha256=$(only(unique(ep_source_hashes)))")
println("manifest_path=$manifest_path")
