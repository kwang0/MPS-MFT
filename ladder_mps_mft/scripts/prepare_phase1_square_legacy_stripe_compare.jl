#!/usr/bin/env julia

using HDF5
using LadderMPSMFT
using Random
using Statistics
using TOML

length(ARGS) == 5 || error(
    "usage: julia --project=. scripts/prepare_phase1_square_legacy_stripe_compare.jl " *
    "BASE_CONFIG.toml LEGACY_T010_V000_STATE.h5 CONTROL_RUN FULL_RUN RUN_ID",
)

const TARGET_L = 64
const TARGET_U = 8.0
const TARGET_V = -0.4
const TARGET_T0 = 1.4
const TARGET_TP = 0.1
const TARGET_DENSITY = 0.9375
const TARGET_EP_SIGNED = -0.24962435880865996
const COMMON_RANDOM_SEED = 1404
const INITIAL_AMPLITUDE = 1.0e-3
const EXPECTED_LEGACY_SHA256 = "ae6a3bfe76ca8f06f2396fd731b18bca8539e0b7ee68df016cc9156fdceeb074"
const HISTORICAL_TARGET_RUN = "20260830_phase1_square_t014_vm04_seed_chi200_loose"
const SOURCE_POINT = "square_t010_v000_legacy_terminal_stripe"
const SANITIZATION_POLICY = "zero_inactive_same_physical_site_beta_only_v1"

base_path = abspath(ARGS[1])
legacy_path = abspath(ARGS[2])
control_run = abspath(ARGS[3])
full_run = abspath(ARGS[4])
run_id = ARGS[5]

isfile(base_path) || error("comparison base configuration not found: $base_path")
isfile(legacy_path) || error("legacy stripe source not found: $legacy_path")
isdir(full_run) || error("full scratch output directory not found: $full_run")
occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID: $run_id")

function require_scalar(file, key::AbstractString, expected::Real; atol::Real=0.0)
    haskey(file, key) || error("legacy source has no top-level $key")
    actual = Float64(read(file, key))
    isapprox(actual, Float64(expected); atol=Float64(atol), rtol=0.0) || error(
        "legacy source $key=$actual, expected $expected",
    )
    return actual
end

function maximum_abs(values)
    isempty(values) && return 0.0
    return maximum(abs, values)
end

function inactive_onsite_beta_stats(beta::AbstractArray, L::Integer)
    values = Float64[]
    for spin in 1:2, rung in 1:L, leg in 1:2
        push!(values, beta[spin, rung, rung, leg, leg])
    end
    return (
        maximum=maximum_abs(values),
        nonzero_count=count(value -> !iszero(value), values),
        total_count=length(values),
    )
end

function active_beta_maximum(beta::AbstractArray, L::Integer, r_range::Integer)
    result = 0.0
    for spin in 1:2, rung in 1:L, other_rung in 1:L, leg in 1:2, other_leg in 1:2
        abs(rung - other_rung) <= r_range || continue
        rung == other_rung && leg == other_leg && continue
        result = max(result, abs(beta[spin, rung, other_rung, leg, other_leg]))
    end
    return result
end

function check_only_inactive_onsite_beta_changed(
    original::AbstractArray,
    sanitized::AbstractArray,
    L::Integer,
)
    size(original) == size(sanitized) || error("sanitized beta shape changed")
    for index in CartesianIndices(original)
        spin, rung, other_rung, leg, other_leg = Tuple(index)
        inactive_onsite = rung == other_rung && leg == other_leg
        if inactive_onsite
            iszero(sanitized[index]) || error("inactive same-site beta was not zeroed at $index")
        else
            sanitized[index] == original[index] || error(
                "physical beta field changed during sanitization at $index",
            )
        end
    end
    return nothing
end

base = load_settings(base_path)
base.model.geometry == :square || error("comparison must use square geometry")
base.model.L == TARGET_L || error("comparison must use L=$TARGET_L")
base.model.U == TARGET_U || error("comparison must use U=$TARGET_U")
base.model.V == TARGET_V || error("comparison must use V=$TARGET_V")
base.model.t0 == TARGET_T0 || error("comparison must use t0=$TARGET_T0")
base.model.tp == TARGET_TP || error("comparison must use t_perp=$TARGET_TP")
base.model.density == TARGET_DENSITY || error("comparison must use density=$TARGET_DENSITY")
base.model.mu_initial == 0.55 || error("comparison base must use initial mu=0.55")
base.model.ep_mode == :exact || error("comparison requires an exact E_p registry row")
base.model.ep_signed == TARGET_EP_SIGNED || error(
    "comparison resolved E_p=$(base.model.ep_signed), expected $TARGET_EP_SIGNED",
)
base.runtime.backend == :gpu || error("comparison must use the GPU backend")
base.runtime.tensor_scalar_type == :float64 || error("comparison must use Float64 tensors")
base.dmrg.nsweeps == 12 || error("comparison must use 12 DMRG sweeps")
base.dmrg.maxdim == 200 || error("comparison must use chi=200")
base.dmrg.cutoff == 1.0e-10 || error("comparison must use cutoff=1e-10")
base.dmrg.energy_tol == 1.0e-6 || error("comparison must use DMRG energy_tol=1e-6")
base.dmrg.max_time_seconds == 41400.0 || error("comparison DMRG deadline changed")
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
base.convergence.unmixed_cycle_probe || error("comparison must begin with a raw-map probe")
base.convergence.probe_iterations == 20 || error("raw-map probe must allow 20 updates")
base.convergence.cycle_action == :continue || error(
    "unaccepted raw recurrence must be archived before optional Anderson acceleration",
)
base.run.max_iterations == 80 || error("comparison must allow up to 80 MF updates")
base.run.initial_seed == :legacy_pairing || error("control must use legacy_pairing")
base.run.initial_seed_protocol == :matched_mode || error(
    "smooth mixed pairing requires the matched_mode protocol",
)
base.run.initial_amplitude == INITIAL_AMPLITUDE || error(
    "smooth mixed-pairing norm must be 1e-3",
)
base.run.initial_mode_number == 0 || error("smooth mixed pairing must use mode zero")
base.run.initial_mode_phase_pi == 0.0 || error("smooth mixed pairing must use phase zero")
base.run.random_seed == COMMON_RANDOM_SEED || error(
    "comparison must use common random seed $COMMON_RANDOM_SEED",
)
base.run.inherit_from === nothing || error("base comparison configuration must not inherit fields")
base.run.parent_checkpoint === nothing || error("base comparison configuration must not use a parent MPS")
base.run.resume_checkpoint === nothing || error("base comparison configuration must not resume an MPS")

legacy_sha256 = LadderMPSMFT.sha256_file(legacy_path)
legacy_sha256 == EXPECTED_LEGACY_SHA256 || error(
    "legacy source SHA-256 is $legacy_sha256, expected $EXPECTED_LEGACY_SHA256",
)
source_metadata = h5open(legacy_path, "r") do file
    require_scalar(file, "U", TARGET_U)
    require_scalar(file, "V", 0.0)
    require_scalar(file, "t0", 1.0)
    require_scalar(file, "t_p", TARGET_TP)
    if haskey(file, "L")
        require_scalar(file, "L", TARGET_L)
    end
    haskey(file, "transverse_geometry") || error(
        "legacy source has no top-level transverse_geometry",
    )
    geometry = normalize_geometry(String(read(file, "transverse_geometry")))
    geometry == :square || error("legacy source geometry is $geometry, expected square")
    if haskey(file, "completed")
        Bool(read(file, "completed")) || error("legacy stripe source is not marked completed")
    end
    if haskey(file, "period2_cycle_detected")
        !Bool(read(file, "period2_cycle_detected")) || error(
            "legacy stripe source is marked as a period-two cycle",
        )
    end
    (
        source_mu=Float64(read(file, "mu")),
        legacy_effective_energy=haskey(file, "E") ? Float64(read(file, "E")) : NaN,
        legacy_completed=haskey(file, "completed") ? Bool(read(file, "completed")) : false,
        legacy_period2=haskey(file, "period2_cycle_detected") ?
            Bool(read(file, "period2_cycle_detected")) : false,
    )
end

inherited = read_inherited_fields(legacy_path)
inherited.format == :legacy || error("expected a legacy top-level field file")
normalize_geometry(something(inherited.source_geometry, "")) == :square || error(
    "legacy field reader did not recover square geometry",
)
fields = inherited.fields
size(fields.alpha) == (TARGET_L, TARGET_L, 2, 2) || error(
    "legacy alpha has shape $(size(fields.alpha))",
)
size(fields.beta) == (2, TARGET_L, TARGET_L, 2, 2) || error(
    "legacy beta has shape $(size(fields.beta))",
)
size(fields.mu_cdw) == (2, 2 * TARGET_L) || error(
    "legacy mu_cdw has shape $(size(fields.mu_cdw))",
)
all(isfinite, fields.alpha) || error("legacy alpha contains nonfinite values")
all(isfinite, fields.beta) || error("legacy beta contains nonfinite values")
all(isfinite, fields.mu_cdw) || error("legacy mu_cdw contains nonfinite values")
isfinite(inherited.chemical_potential) || error("legacy chemical potential is nonfinite")

source_alpha_max = maximum_abs(fields.alpha)
source_beta_max = maximum_abs(fields.beta)
source_mu_cdw_max = maximum_abs(fields.mu_cdw)
source_active_beta_max = active_beta_maximum(fields.beta, TARGET_L, base.model.r_range)
source_onsite_beta = inactive_onsite_beta_stats(fields.beta, TARGET_L)
source_active_beta_max >= 1.0e-3 || error(
    "legacy source lacks a high-amplitude active normal stripe field",
)
source_mu_cdw_max >= 1.0e-3 || error(
    "legacy source lacks a high-amplitude Hartree stripe field",
)

sanitized_fields = copy(fields)
for spin in 1:2, rung in 1:TARGET_L, leg in 1:2
    sanitized_fields.beta[spin, rung, rung, leg, leg] = 0.0
end
check_only_inactive_onsite_beta_changed(fields.beta, sanitized_fields.beta, TARGET_L)
sanitized_onsite_beta = inactive_onsite_beta_stats(sanitized_fields.beta, TARGET_L)
iszero(sanitized_onsite_beta.maximum) || error("sanitized inactive beta remains nonzero")

seed_directory = joinpath(control_run, "seeds")
ispath(seed_directory) && !isempty(readdir(seed_directory)) && error(
    "refusing to overwrite nonempty seed directory: $seed_directory",
)
mkpath(seed_directory)
seed_path = joinpath(seed_directory, "legacy_square_t010_v000_terminal_physical_fields.h5")
ispath(seed_path) && error("refusing to overwrite derived field seed: $seed_path")
h5open(seed_path, "w") do file
    file["alpha"] = sanitized_fields.alpha
    file["beta"] = sanitized_fields.beta
    file["mu_cdw"] = sanitized_fields.mu_cdw
    file["mu"] = inherited.chemical_potential
    file["transverse_geometry"] = "square"
    file["source_path"] = legacy_path
    file["source_sha256"] = legacy_sha256
    file["source_point"] = SOURCE_POINT
    file["sanitization_policy"] = SANITIZATION_POLICY
    file["source_U"] = TARGET_U
    file["source_V"] = 0.0
    file["source_t0"] = 1.0
    file["source_t_p"] = TARGET_TP
end
seed_sha256 = LadderMPSMFT.sha256_file(seed_path)
seed_readback = read_inherited_fields(seed_path)
seed_readback.fields.alpha == sanitized_fields.alpha || error("derived seed alpha readback failed")
seed_readback.fields.beta == sanitized_fields.beta || error("derived seed beta readback failed")
seed_readback.fields.mu_cdw == sanitized_fields.mu_cdw || error("derived seed mu_cdw readback failed")
seed_readback.chemical_potential == inherited.chemical_potential || error(
    "derived seed chemical-potential readback failed",
)

branches = (
    (
        label="square__smooth_pairing_current_t014_vm04_chi200_loose",
        role="matched_current_pairing_control",
        lineage="fresh_product_mps_plus_smooth_pairing_fields",
        inherit=false,
    ),
    (
        label="square__legacy_t010_v000_stripe_inherit_t014_vm04_chi200_loose",
        role="legacy_stripe_basin_probe",
        lineage="fresh_product_mps_plus_exact_legacy_terminal_physical_fields",
        inherit=true,
    ),
)

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
implementation_fingerprints = String[]
ep_source_hashes = String[]

open(manifest_path, "w") do io
    println(io, join((
        "label", "geometry", "L", "U", "V", "t0", "tp", "density", "chi",
        "comparison_role", "lineage", "config", "config_sha256", "model_fingerprint",
        "numerical_fingerprint", "implementation_sha256", "ep_source_sha256",
        "ep_mode", "ep_signed", "tp2_over_ep", "full_output_directory",
        "stateless_output_directory", "actual_initial_condition", "configured_fallback_seed",
        "random_seed", "source_point", "original_legacy_path", "original_legacy_sha256",
        "derived_field_seed_path", "derived_field_seed_sha256", "sanitization_policy",
        "sanitized_beta_entry_count", "source_mu", "source_alpha_max_abs",
        "source_beta_max_abs", "source_active_beta_max_abs", "source_mu_cdw_max_abs",
        "legacy_effective_energy_not_rankable", "historical_target_run",
        "historical_cross_campaign_energy_ranking_authorized",
        "same_campaign_energy_ranking_authorized_if_accepted_and_fingerprints_match",
        "fresh_mps", "parent_mps", "raw_map_probe_updates", "anderson_after_raw_probe",
        "dmrg_sweeps", "dmrg_energy_tol", "dmrg_cutoff", "mu_density_tol",
        "outer_density_tol", "field_abs_tol", "field_rel_tol", "variational_energy_tol",
        "max_mf_updates",
    ), '\t'))

    for branch in branches
        raw = TOML.parsefile(base_path)
        run = raw["run"]
        full_output_directory = joinpath(full_run, "results", branch.label)
        stateless_output_directory = joinpath(control_run, "results", branch.label)
        run["output_directory"] = full_output_directory
        run["branch_label"] = branch.role
        run["preparation"] = "square_t014_vm04_current_loose_legacy_stripe_basin_comparison"
        run["direction"] = branch.inherit ? "from_legacy_square_t010_v000_terminal_stripe" : "none"
        run["seed_label"] = branch.inherit ?
            "legacy_square_t010_v000_terminal_physical_fields" :
            "smooth_mixed_pairing_common_s$(COMMON_RANDOM_SEED)"
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
        if branch.inherit
            run["inherit_from"] = seed_path
            run["inherit_sha256"] = seed_sha256
        end

        config_path = joinpath(config_directory, "$(branch.label).segment-001.toml")
        ispath(config_path) && error("refusing to overwrite comparison config: $config_path")
        open(config_path, "w") do config_io
            TOML.print(config_io, raw; sorted=true)
        end

        settings = load_settings(config_path)
        settings.model.geometry == :square || error("prepared branch changed geometry")
        settings.model.V == TARGET_V || error("prepared branch changed V")
        settings.model.t0 == TARGET_T0 || error("prepared branch changed t0")
        settings.model.ep_mode == :exact || error("prepared branch lost exact E_p")
        settings.model.ep_signed == TARGET_EP_SIGNED || error("prepared branch changed E_p")
        settings.run.random_seed == COMMON_RANDOM_SEED || error("prepared branch changed RNG seed")
        if branch.inherit
            settings.run.inherit_from == seed_path || error("prepared inherit path changed")
            settings.run.inherit_sha256 == seed_sha256 || error("prepared inherit hash changed")
            settings.run.parent_checkpoint === nothing || error("inherited branch acquired a parent MPS")
        else
            settings.run.inherit_from === nothing || error("fresh control unexpectedly inherits fields")
            seed_fields = initial_fields(
                settings.model;
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
            all(iszero, seed_fields.beta) || error("fresh pairing control has nonzero beta")
            all(iszero, seed_fields.mu_cdw) || error("fresh pairing control has nonzero mu_cdw")
            any(value -> !iszero(value), seed_fields.alpha) || error(
                "fresh pairing control has zero alpha",
            )
            isapprox(
                field_l2_per_physical_site(seed_fields, settings.model),
                INITIAL_AMPLITUDE;
                atol=1.0e-14,
                rtol=1.0e-12,
            ) || error("fresh pairing control lost its matched total norm")
            for offset in 0:settings.model.r_range, leg in 1:2, other_leg in 1:2
                values = [
                    seed_fields.alpha[rung, rung + offset, leg, other_leg]
                    for rung in 1:(settings.model.L - offset)
                ]
                maximum(abs.(values .- first(values))) <= 1.0e-14 || error(
                    "fresh pairing control contains center-of-mass spatial noise",
                )
            end
        end

        model_fp = LadderMPSMFT.model_fingerprint(settings.model)
        numerical_fp = LadderMPSMFT.numerical_fingerprint(settings)
        implementation_fp = implementation_fingerprint(settings)
        ep_source_sha256 = LadderMPSMFT.sha256_file(settings.model.ep_source)
        push!(model_fingerprints, model_fp)
        push!(numerical_fingerprints, numerical_fp)
        push!(implementation_fingerprints, implementation_fp)
        push!(ep_source_hashes, ep_source_sha256)

        println(io, join((
            branch.label,
            String(settings.model.geometry),
            string(settings.model.L),
            string(settings.model.U),
            string(settings.model.V),
            string(settings.model.t0),
            string(settings.model.tp),
            string(settings.model.density),
            string(settings.dmrg.maxdim),
            branch.role,
            branch.lineage,
            config_path,
            LadderMPSMFT.sha256_file(config_path),
            model_fp,
            numerical_fp,
            implementation_fp,
            ep_source_sha256,
            String(settings.model.ep_mode),
            string(settings.model.ep_signed),
            string(settings.model.tp^2 / settings.model.ep),
            full_output_directory,
            stateless_output_directory,
            branch.inherit ? "derived_exact_legacy_terminal_fields" : "smooth_translation_invariant_pairing",
            String(settings.run.initial_seed),
            string(settings.run.random_seed),
            branch.inherit ? SOURCE_POINT : "none",
            branch.inherit ? legacy_path : "none",
            branch.inherit ? legacy_sha256 : "none",
            branch.inherit ? seed_path : "none",
            branch.inherit ? seed_sha256 : "none",
            branch.inherit ? SANITIZATION_POLICY : "none",
            branch.inherit ? string(source_onsite_beta.total_count) : "0",
            branch.inherit ? string(source_metadata.source_mu) : string(settings.model.mu_initial),
            branch.inherit ? string(source_alpha_max) : "not_applicable",
            branch.inherit ? string(source_beta_max) : "not_applicable",
            branch.inherit ? string(source_active_beta_max) : "not_applicable",
            branch.inherit ? string(source_mu_cdw_max) : "not_applicable",
            branch.inherit ? string(source_metadata.legacy_effective_energy) : "not_applicable",
            HISTORICAL_TARGET_RUN,
            "false",
            "true",
            "true",
            "none",
            string(settings.convergence.probe_iterations),
            "true",
            string(settings.dmrg.nsweeps),
            string(settings.dmrg.energy_tol),
            string(settings.dmrg.cutoff),
            string(settings.dmrg.mu_density_tol),
            string(settings.convergence.density_tol),
            string(settings.convergence.field_abs_tol),
            string(settings.convergence.field_rel_tol),
            string(settings.convergence.variational_energy_tol),
            string(settings.run.max_iterations),
        ), '\t'))
    end
end

length(unique(model_fingerprints)) == 1 || error("comparison branches have different model fingerprints")
length(unique(numerical_fingerprints)) == 1 || error(
    "comparison branches have different numerical fingerprints",
)
length(unique(implementation_fingerprints)) == 1 || error(
    "comparison branches have different implementation fingerprints",
)
length(unique(ep_source_hashes)) == 1 || error("comparison branches do not share one E_p registry")

println("branch_count=$(length(branches))")
println("target=L64_U8_V-0.4_t0-1.4_tp0.1_density0.9375_square_chi200")
println("legacy_source=$legacy_path")
println("legacy_source_sha256=$legacy_sha256")
println("legacy_source_mu=$(source_metadata.source_mu)")
println("legacy_source_alpha_max_abs=$source_alpha_max")
println("legacy_source_beta_max_abs=$source_beta_max")
println("legacy_source_active_beta_max_abs=$source_active_beta_max")
println("legacy_source_mu_cdw_max_abs=$source_mu_cdw_max")
println("inactive_onsite_beta_max_abs=$(source_onsite_beta.maximum)")
println("inactive_onsite_beta_entries_zeroed=$(source_onsite_beta.total_count)")
println("derived_field_seed=$seed_path")
println("derived_field_seed_sha256=$seed_sha256")
println("sanitization_policy=$SANITIZATION_POLICY")
println("numerical_fingerprint=$(only(unique(numerical_fingerprints)))")
println("implementation_sha256=$(only(unique(implementation_fingerprints)))")
println("ep_source_sha256=$(only(unique(ep_source_hashes)))")
println("same_campaign_energy_ranking=only_if_both_final_states_are_accepted_and_fingerprints_match")
println("historical_target_run_energy_ranking=false_fingerprint_mismatch")
println("manifest_path=$manifest_path")
