#!/usr/bin/env julia

using LadderMPSMFT
using TOML

length(ARGS) == 4 || error(
    "usage: julia --project=. scripts/prepare_phase1_matched_seed_pilot.jl BASE_CONFIG.toml CONTROL_RUN FULL_RUN RUN_ID",
)

const COMMON_RANDOM_SEED = 1404
const INITIAL_AMPLITUDE = 1.0e-3

base_path = abspath(ARGS[1])
control_run = abspath(ARGS[2])
full_run = abspath(ARGS[3])
run_id = ARGS[4]

isfile(base_path) || error("matched-seed pilot base configuration not found: $base_path")
occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID: $run_id")

base_settings = load_settings(base_path)
base_settings.model.geometry == :cubic_unfrustrated || error(
    "matched-seed pilot must use cubic_unfrustrated geometry",
)
base_settings.model.L == 64 || error("matched-seed pilot must use L=64")
base_settings.model.U == 8.0 || error("matched-seed pilot must use U=8")
base_settings.model.V == -0.2 || error("matched-seed pilot must use V=-0.2")
base_settings.model.t0 == 1.1 || error("matched-seed pilot must use t0=1.1")
base_settings.model.tp == 0.1 || error("matched-seed pilot must use t_perp=0.1")
base_settings.model.density == 0.9375 || error("matched-seed pilot must use density=0.9375")
base_settings.runtime.backend == :gpu || error("matched-seed pilot must use the GPU backend")
base_settings.runtime.tensor_scalar_type == :float64 || error(
    "matched-seed pilot must request Float64 tensors",
)
base_settings.dmrg.maxdim == 400 || error("matched-seed pilot must use chi=400")
base_settings.run.initial_seed_protocol == :matched_mode || error(
    "matched-seed pilot base must opt in to matched_mode",
)
base_settings.run.initial_amplitude == INITIAL_AMPLITUDE || error(
    "matched-seed pilot base must use initial_amplitude=1e-3",
)
base_settings.run.random_seed == COMMON_RANDOM_SEED || error(
    "matched-seed pilot base must use the declared common random seed $COMMON_RANDOM_SEED",
)
base_settings.convergence.unmixed_cycle_probe || error(
    "matched-seed pilot must begin with the unmixed raw-map probe",
)
base_settings.convergence.probe_iterations == 20 || error(
    "matched-seed pilot must use exactly 20 raw-map updates",
)
base_settings.convergence.cycle_action == :stop || error(
    "matched-seed pilot must stop rather than enter Anderson acceleration",
)
base_settings.run.max_iterations == base_settings.convergence.probe_iterations + 1 || error(
    "raw-only matched-seed pilot requires max_iterations=probe_iterations+1",
)

config_directory = joinpath(control_run, "configs")
ispath(config_directory) && !isempty(readdir(config_directory)) && error(
    "refusing to overwrite nonempty configuration directory: $config_directory",
)
mkpath(config_directory)
mkpath(joinpath(control_run, "results"))
mkpath(joinpath(full_run, "results"))

branches = (
    (
        label="unfrustrated__pairing_matched_m000_chi400",
        branch="sc",
        seed="pairing",
        mode=0,
        pairing_form_factor="d_wave",
        leg_parity="auto",
    ),
    (
        label="unfrustrated__sdw_matched_m058_chi400",
        branch="sdw",
        seed="sdw",
        mode=58,
        pairing_form_factor="onsite_s",
        leg_parity="odd",
    ),
    (
        label="unfrustrated__cdw_matched_m011_chi400",
        branch="cdw",
        seed="cdw",
        mode=11,
        pairing_form_factor="onsite_s",
        leg_parity="even",
    ),
)

manifest_path = joinpath(control_run, "manifest.tsv")
ispath(manifest_path) && error("refusing to overwrite manifest: $manifest_path")
model_fingerprints = String[]
numerical_fingerprints = String[]
seed_fingerprints = String[]

open(manifest_path, "w") do io
    println(io, join((
        "label", "geometry", "branch", "seed", "random_seed", "config", "config_sha256",
        "ep_mode", "ep_signed", "ep_abs", "ep_t0_lower", "ep_t0_upper",
        "ep_lower_signed", "ep_upper_signed", "ep_weight", "tp2_over_ep",
        "full_output_directory", "stateless_output_directory",
        "initial_seed_protocol", "initial_seed_fingerprint", "initial_amplitude",
        "initial_mode_number", "initial_mode_wavevector_pi", "initial_mode_phase_pi",
        "initial_pairing_form_factor", "initial_leg_parity_requested",
        "initial_leg_parity_resolved", "initial_seed_normalization",
    ), '\t'))

    for branch in branches
        raw = TOML.parsefile(base_path)
        run = raw["run"]
        full_output_directory = joinpath(full_run, "results", branch.label)
        stateless_output_directory = joinpath(control_run, "results", branch.label)
        run["output_directory"] = full_output_directory
        run["branch_label"] = branch.branch
        run["preparation"] = "matched_mode_independent_seed"
        run["direction"] = "none"
        run["seed_label"] = "matched_$(branch.seed)_m$(lpad(branch.mode, 3, '0'))"
        run["random_seed"] = COMMON_RANDOM_SEED
        run["initial_seed"] = branch.seed
        run["initial_amplitude"] = INITIAL_AMPLITUDE
        run["initial_seed_protocol"] = "matched_mode"
        run["initial_mode_number"] = branch.mode
        run["initial_mode_phase_pi"] = 0.0
        run["initial_pairing_form_factor"] = branch.pairing_form_factor
        run["initial_leg_parity"] = branch.leg_parity
        for key in (
            "inherit_from", "inherit_sha256",
            "parent_checkpoint", "parent_sha256", "parent_orbit_phase",
            "resume_checkpoint", "resume_sha256",
        )
            pop!(run, key, nothing)
        end

        config_path = joinpath(config_directory, "$(branch.label).segment-001.toml")
        ispath(config_path) && error("refusing to overwrite matched-seed config: $config_path")
        open(config_path, "w") do config_io
            TOML.print(config_io, raw; sorted=true)
        end

        settings = load_settings(config_path)
        model = settings.model
        seed_metadata = initial_seed_metadata(model, settings.run)
        settings.run.random_seed == COMMON_RANDOM_SEED || error(
            "prepared branch lost the common product-state random seed: $(branch.label)",
        )
        settings.run.initial_seed == Symbol(branch.seed) || error(
            "prepared branch changed seed channel: $(branch.label)",
        )
        settings.run.initial_seed_protocol == :matched_mode || error(
            "prepared branch changed seed protocol: $(branch.label)",
        )
        settings.run.initial_mode_number == branch.mode || error(
            "prepared branch changed carrier mode: $(branch.label)",
        )
        settings.run.initial_mode_phase_pi == 0.0 || error(
            "prepared branch changed carrier phase: $(branch.label)",
        )
        settings.run.initial_pairing_form_factor == Symbol(branch.pairing_form_factor) || error(
            "prepared branch changed pairing form factor: $(branch.label)",
        )
        seed_metadata.resolved_leg_parity == (
            branch.seed == "pairing" ? :not_applicable : Symbol(branch.leg_parity)
        ) || error("prepared branch changed transverse parity: $(branch.label)")

        push!(model_fingerprints, LadderMPSMFT.model_fingerprint(model))
        push!(numerical_fingerprints, LadderMPSMFT.numerical_fingerprint(settings))
        push!(seed_fingerprints, LadderMPSMFT.initial_seed_fingerprint(settings))
        println(io, join((
            branch.label,
            String(model.geometry),
            branch.branch,
            branch.seed,
            string(COMMON_RANDOM_SEED),
            config_path,
            LadderMPSMFT.sha256_file(config_path),
            String(model.ep_mode),
            string(model.ep_signed),
            string(model.ep),
            string(model.ep_t0_lower),
            string(model.ep_t0_upper),
            string(model.ep_lower_signed),
            string(model.ep_upper_signed),
            string(model.ep_interpolation_weight),
            string(model.tp^2 / model.ep),
            full_output_directory,
            stateless_output_directory,
            String(settings.run.initial_seed_protocol),
            LadderMPSMFT.initial_seed_fingerprint(settings),
            string(settings.run.initial_amplitude),
            string(seed_metadata.mode_number),
            string(seed_metadata.mode_wavevector_pi),
            string(seed_metadata.mode_phase_pi),
            String(seed_metadata.pairing_form_factor),
            String(seed_metadata.requested_leg_parity),
            String(seed_metadata.resolved_leg_parity),
            seed_metadata.normalization,
        ), '\t'))
    end
end

length(unique(model_fingerprints)) == 1 || error(
    "matched-seed pilot branches do not share one model fingerprint",
)
length(unique(numerical_fingerprints)) == 1 || error(
    "matched-seed pilot branches do not share one numerical fingerprint",
)
length(unique(seed_fingerprints)) == length(branches) || error(
    "matched-seed pilot branches do not have distinct seed fingerprints",
)

println("branch_count=$(length(branches))")
println("common_random_seed=$COMMON_RANDOM_SEED")
println("model_fingerprint=$(only(unique(model_fingerprints)))")
println("numerical_fingerprint=$(only(unique(numerical_fingerprints)))")
println("manifest_path=$manifest_path")
