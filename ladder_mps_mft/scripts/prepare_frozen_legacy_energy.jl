#!/usr/bin/env julia

using HDF5
using LadderMPSMFT
using TOML

length(ARGS) == 5 || error(
    "usage: julia --project=. scripts/prepare_frozen_legacy_energy.jl " *
    "SOURCE_RUN LEGACY_STATE.h5 CONTROL_RUN FULL_RUN RUN_ID",
)

const SOURCE_LABELS = [
    "square__pairing_dwave_m000_chi200_loose",
    "square__legacy_pairing_mixed_chi200_loose",
    "square__stripe_m004_chi200_loose",
    "square__stripe_m005_chi200_loose",
    "square__stripe_pairing_m004_chi200_loose",
    "square__stripe_pairing_m005_chi200_loose",
]
const BASE_LABEL = first(SOURCE_LABELS)
const TARGET_LABEL = "square__legacy_terminal_fields_frozen_dmrg_chi200"

source_run, legacy_path, control_run, full_run = realpath.(ARGS[1:4])
run_id = ARGS[5]
isfile(legacy_path) || error("legacy state not found: $legacy_path")

function only_analysis_state(run::AbstractString, label::AbstractString)
    root_name = isdir(joinpath(run, "stateless_results")) ? "stateless_results" : "results"
    root = joinpath(run, root_name, label)
    isdir(root) || error("source result directory not found: $root")
    states = String[]
    for (directory, _, names) in walkdir(root)
        "state.h5" in names && push!(states, realpath(joinpath(directory, "state.h5")))
    end
    unique!(states)
    length(states) == 1 || error(
        "expected exactly one analysis state for $label, found $(length(states))",
    )
    return only(states)
end

function accepted_state_metadata(path::AbstractString)
    return h5open(path, "r") do file
        required = (
            "accepted", "completed", "status", "solution_kind", "fundamental_period",
            "orbit_validated", "model/transverse_geometry",
            "solution_canonical_variational_energy",
            "solution_target_density_corrected_variational_energy",
            "provenance/model_fingerprint", "provenance/numerical_fingerprint",
            "provenance/implementation_sha256", "provenance/ep_source_sha256",
            "provenance/tensor_scalar_type",
        )
        for name in required
            haskey(file, name) || error("source state has no $name: $path")
        end
        Bool(read(file, "accepted")) || error("source state is not accepted: $path")
        Bool(read(file, "completed")) || error("source state is not completed: $path")
        status = String(read(file, "status"))
        solution_kind = String(read(file, "solution_kind"))
        period = Int(read(file, "fundamental_period"))
        orbit_validated = Bool(read(file, "orbit_validated"))
        status == "fixed_point" || error("source state is $status rather than fixed_point: $path")
        solution_kind == "fixed_point" || error(
            "source solution kind is $solution_kind rather than fixed_point: $path",
        )
        period == 1 || error("source state has period $period rather than 1: $path")
        !orbit_validated || error("period-one source unexpectedly claims an orbit: $path")
        tensor_type = lowercase(String(read(file, "provenance/tensor_scalar_type")))
        tensor_type == "float64" || error("source state is $tensor_type rather than Float64: $path")
        geometry = String(read(file, "model/transverse_geometry"))
        geometry == "square" || error("source state geometry is $geometry rather than square: $path")
        target_energy = Float64(read(
            file,
            "solution_target_density_corrected_variational_energy",
        ))
        isfinite(target_energy) || error("source target-density energy is not finite: $path")
        return (
            model_fingerprint=String(read(file, "provenance/model_fingerprint")),
            numerical_fingerprint=String(read(file, "provenance/numerical_fingerprint")),
            implementation_sha256=String(read(file, "provenance/implementation_sha256")),
            ep_source_sha256=String(read(file, "provenance/ep_source_sha256")),
            tensor_type,
            canonical_energy=Float64(read(file, "solution_canonical_variational_energy")),
            target_energy,
            state_sha256=LadderMPSMFT.sha256_file(path),
        )
    end
end

function validate_legacy(path::AbstractString, model::ModelSettings)
    inherited = read_inherited_fields(path)
    inherited.format == :legacy || error("requested source is not a legacy top-level field file")
    size(inherited.fields.alpha) == (model.L, model.L, 2, 2) || error(
        "legacy alpha shape $(size(inherited.fields.alpha)) does not match the requested model",
    )
    size(inherited.fields.beta) == (2, model.L, model.L, 2, 2) || error(
        "legacy beta shape $(size(inherited.fields.beta)) does not match the requested model",
    )
    size(inherited.fields.mu_cdw) == (2, 2 * model.L) || error(
        "legacy mu_cdw shape $(size(inherited.fields.mu_cdw)) does not match the requested model",
    )
    inherited.source_geometry === nothing && error("legacy file has no transverse_geometry")
    normalize_geometry(inherited.source_geometry) == model.geometry || error(
        "legacy geometry $(inherited.source_geometry) does not match $(model.geometry)",
    )
    h5open(path, "r") do file
        for (name, expected) in (
            ("U", model.U), ("V", model.V), ("t0", model.t0), ("t_p", model.tp),
        )
            haskey(file, name) || error("legacy file has no $name")
            actual = Float64(read(file, name))
            isapprox(actual, expected; atol=0, rtol=1e-12) || error(
                "legacy $name=$actual does not match requested $expected",
            )
        end
        for name in (
            "E", "mu", "completed", "period2_cycle_detected", "alpha_list", "beta_list",
            "mu_cdw_list", "C_pair_list", "C_exc_dn_list", "C_exc_up_list",
        )
            haskey(file, name) || error("legacy file has no required dataset $name")
        end
    end
    return inherited
end

reference_paths = [only_analysis_state(source_run, label) for label in SOURCE_LABELS]
reference_metadata = accepted_state_metadata.(reference_paths)
for field in (
    :model_fingerprint, :numerical_fingerprint, :implementation_sha256,
    :ep_source_sha256, :tensor_type,
)
    length(unique(getfield(row, field) for row in reference_metadata)) == 1 || error(
        "the six source states do not share one $field",
    )
end

base_config_path = joinpath(source_run, "configs", "$BASE_LABEL.segment-001.toml")
isfile(base_config_path) || error("source base configuration not found: $base_config_path")
base_settings = load_settings(base_config_path)
base_settings.model.geometry == :square || error("source base configuration is not square")
base_settings.model.L == 64 || error("source base L changed")
base_settings.model.U == 8.0 || error("source base U changed")
base_settings.model.V == 0.0 || error("source base V changed")
base_settings.model.t0 == 1.4 || error("source base t0 changed")
base_settings.model.tp == 0.1 || error("source base t_perp changed")
base_settings.model.density == 0.9375 || error("source base density changed")
base_settings.dmrg.maxdim == 200 || error("source base maxdim changed")
base_settings.runtime.backend == :gpu || error("source base backend is not GPU")
base_settings.runtime.tensor_scalar_type == :float64 || error(
    "source base configuration does not request Float64 tensors",
)
model_fingerprint = LadderMPSMFT.model_fingerprint(base_settings.model)
numerical_fingerprint = LadderMPSMFT.numerical_fingerprint(base_settings)
implementation_sha256 = implementation_fingerprint(base_settings)
model_fingerprint == first(reference_metadata).model_fingerprint || error(
    "source config and state model fingerprints differ",
)
numerical_fingerprint == first(reference_metadata).numerical_fingerprint || error(
    "source config and state numerical fingerprints differ",
)
implementation_sha256 == first(reference_metadata).implementation_sha256 || error(
    "current src/ plus GPU Manifest differ from the six source states",
)
validate_legacy(legacy_path, base_settings.model)

config_directory = joinpath(control_run, "configs")
ispath(config_directory) && !isempty(readdir(config_directory)) && error(
    "refusing to overwrite nonempty configuration directory: $config_directory",
)
mkpath(config_directory)
mkpath(joinpath(control_run, "results"))
mkpath(joinpath(full_run, "results"))

raw = TOML.parsefile(base_config_path)
run = raw["run"]
dmrg = raw["dmrg"]
full_output_directory = joinpath(full_run, "results", TARGET_LABEL)
stateless_output_directory = joinpath(control_run, "results", TARGET_LABEL)
legacy_sha256 = LadderMPSMFT.sha256_file(legacy_path)
run["output_directory"] = full_output_directory
run["branch_label"] = "legacy_frozen_energy"
run["preparation"] = "legacy_terminal_fields_single_frozen_dmrg"
run["direction"] = "from_legacy_terminal_fields"
run["seed_label"] = "exact_legacy_terminal_fields"
run["inherit_from"] = legacy_path
run["inherit_sha256"] = legacy_sha256
run["max_iterations"] = 1
run["require_accepted_solution"] = false
run["quick_diagnostics"] = true
run["full_pair_correlations"] = false
for key in (
    "parent_checkpoint", "parent_sha256", "parent_orbit_phase",
    "resume_checkpoint", "resume_sha256",
)
    pop!(run, key, nothing)
end
# Leave all accuracy controls identical to the six-state campaign. Only the
# performance-only deadline is shortened to finish before the three-hour job.
dmrg["max_time_seconds"] = 9000.0
raw["frozen_legacy"] = Dict(
    "policy" => "single_fresh_dmrg_at_saved_legacy_fields_and_mu_no_mf_update_no_mu_search",
    "selection_eligible" => false,
    "source_path" => legacy_path,
    "source_sha256" => legacy_sha256,
    "reference_run" => basename(source_run),
    "reference_labels" => SOURCE_LABELS,
    "reference_states" => reference_paths,
    "reference_state_sha256" => getfield.(reference_metadata, :state_sha256),
    "expected_model_fingerprint" => model_fingerprint,
    "expected_numerical_fingerprint" => numerical_fingerprint,
    "expected_implementation_sha256" => implementation_sha256,
    "expected_ep_source_sha256" => first(reference_metadata).ep_source_sha256,
)

config_path = joinpath(config_directory, "$TARGET_LABEL.segment-001.toml")
ispath(config_path) && error("refusing to overwrite frozen-field config: $config_path")
open(config_path, "w") do io
    TOML.print(io, raw; sorted=true)
end
settings = load_settings(config_path)
LadderMPSMFT.model_fingerprint(settings.model) == model_fingerprint || error(
    "prepared frozen-field model fingerprint changed",
)
LadderMPSMFT.numerical_fingerprint(settings) == numerical_fingerprint || error(
    "prepared frozen-field numerical fingerprint changed",
)
implementation_fingerprint(settings) == implementation_sha256 || error(
    "prepared frozen-field implementation fingerprint changed",
)
settings.run.inherit_from == legacy_path || error("prepared config lost the legacy source path")
settings.run.inherit_sha256 == legacy_sha256 || error("prepared config lost the legacy source hash")
settings.run.max_iterations == 1 || error("prepared frozen-field config must allow one evaluation")
settings.dmrg.max_time_seconds == 9000.0 || error("prepared DMRG deadline changed")

manifest_path = joinpath(control_run, "manifest.tsv")
ispath(manifest_path) && error("refusing to overwrite manifest: $manifest_path")
open(manifest_path, "w") do io
    println(io, join((
        "label", "geometry", "L", "U", "V", "t0", "tp", "density", "chi",
        "config", "config_sha256", "legacy_source", "legacy_source_sha256",
        "reference_run", "reference_count", "model_fingerprint", "numerical_fingerprint",
        "implementation_sha256", "ep_source_sha256", "full_output_directory",
        "stateless_output_directory", "dmrg_solves", "mf_updates", "mu_searches",
        "selection_eligible", "requested_time", "requested_node_hours",
    ), '\t'))
    model = settings.model
    println(io, join((
        TARGET_LABEL, String(model.geometry), string(model.L), string(model.U), string(model.V),
        string(model.t0), string(model.tp), string(model.density), string(settings.dmrg.maxdim),
        config_path, LadderMPSMFT.sha256_file(config_path), legacy_path, legacy_sha256,
        basename(source_run), string(length(reference_paths)), model_fingerprint,
        numerical_fingerprint, implementation_sha256, first(reference_metadata).ep_source_sha256,
        full_output_directory, stateless_output_directory, "1", "0", "0", "false",
        "03:00:00", "0.750000000",
    ), '\t'))
end

println("legacy_source=$legacy_path")
println("legacy_source_sha256=$legacy_sha256")
println("reference_run=$(basename(source_run))")
println("reference_count=$(length(reference_paths))")
println("branch_count=1")
println("dmrg_solves=1")
println("mf_updates=0")
println("mu_searches=0")
println("selection_eligible=false")
println("model_fingerprint=$model_fingerprint")
println("numerical_fingerprint=$numerical_fingerprint")
println("implementation_sha256=$implementation_sha256")
println("manifest_path=$manifest_path")
