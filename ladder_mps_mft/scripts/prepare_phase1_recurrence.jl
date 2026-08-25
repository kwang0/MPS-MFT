#!/usr/bin/env julia

using HDF5
using LadderMPSMFT
using TOML

length(ARGS) == 5 || error(
    "usage: julia --project=. scripts/prepare_phase1_recurrence.jl BASE_CONFIG.toml SOURCE_CONTROL_RUN CONTROL_RUN FULL_RUN RUN_ID",
)

base_path = abspath(ARGS[1])
source_run = abspath(ARGS[2])
control_run = abspath(ARGS[3])
full_run = abspath(ARGS[4])
run_id = ARGS[5]

isfile(base_path) || error("recurrence base configuration not found: $base_path")
isdir(source_run) || error("source control run not found: $source_run")
occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID: $run_id")

function only_candidate_state(source_run::AbstractString)
    root = joinpath(source_run, "stateless_results", "unfrustrated__pairing_s1")
    isdir(root) || error("source run has no unfrustrated pairing stateless branch: $root")
    states = String[]
    for (directory, _, names) in walkdir(root)
        "state.h5" in names && push!(states, joinpath(directory, "state.h5"))
    end
    length(states) == 1 || error(
        "expected exactly one unfrustrated pairing state, found $(length(states)) below $root",
    )
    return only(states)
end

function source_metadata(compact_path::AbstractString, source_run::AbstractString)
    metadata = h5open(compact_path, "r") do file
        haskey(file, "analysis_storage/is_stateless_copy") || error(
            "source must be a verified stateless analysis state: $compact_path",
        )
        Bool(read(file, "analysis_storage/is_stateless_copy")) || error(
            "source is not marked as a stateless analysis state: $compact_path",
        )
        status = Symbol(read(file, "status"))
        status == :periodic_candidate || error(
            "source status must be periodic_candidate, found $status",
        )
        Bool(read(file, "accepted")) && error("source candidate is unexpectedly accepted")
        period = Int(read(file, "fundamental_period"))
        period == 2 || error("source candidate period must be 2, found $period")
        haskey(file, "cycle_members") || error("source candidate has no stored cycle members")
        phase_names = sort!(String.(collect(keys(file["cycle_members"]))))
        phase_names == ["001", "002"] || error(
            "source candidate must have exactly phases 001 and 002, found $(join(phase_names, ','))",
        )
        phase_iterations = Int[]
        phase_modes = String[]
        for phase_name in phase_names
            phase = file["cycle_members/$phase_name"]
            for required in ("applied", "measured", "iteration", "update_mode")
                haskey(phase, required) || error("source phase $phase_name has no $required")
            end
            mode = String(read(phase, "update_mode"))
            mode == "unmixed_probe" || error(
                "source phase $phase_name is $mode rather than unmixed_probe",
            )
            push!(phase_iterations, Int(read(phase, "iteration")))
            push!(phase_modes, mode)
        end
        return (
            full_path=String(read(file, "analysis_storage/full_artifact_path")),
            full_sha256=lowercase(String(read(file, "analysis_storage/full_artifact_sha256"))),
            model_fingerprint=String(read(file, "provenance/model_fingerprint")),
            numerical_fingerprint=String(read(file, "provenance/numerical_fingerprint")),
            tensor_scalar_type=String(read(file, "provenance/tensor_scalar_type")),
            status=String(status),
            phase_iterations,
            phase_modes,
        )
    end

    full_root_path = joinpath(source_run, "full_storage_path.txt")
    isfile(full_root_path) || error("source run has no full_storage_path.txt")
    full_root = realpath(strip(read(full_root_path, String)))
    isfile(metadata.full_path) || error(
        "source full orbit artifact is unavailable: $(metadata.full_path)",
    )
    full_path = realpath(metadata.full_path)
    relative = relpath(full_path, full_root)
    (relative == ".." || startswith(relative, "..$(Base.Filesystem.path_separator)")) && error(
        "source full artifact escapes the recorded scratch run: $full_path",
    )
    actual_sha256 = LadderMPSMFT.sha256_file(full_path)
    actual_sha256 == metadata.full_sha256 || error(
        "source full orbit artifact SHA-256 differs from its stateless link",
    )
    full_metadata = h5open(full_path, "r") do file
        for required in (
            "status", "accepted", "fundamental_period",
            "provenance/model_fingerprint",
            "provenance/numerical_fingerprint",
            "provenance/tensor_scalar_type",
            "cycle_members",
        )
            haskey(file, required) || error("source full artifact has no $required")
        end
        phase_names = sort!(String.(collect(keys(file["cycle_members"]))))
        phase_names == ["001", "002"] || error(
            "source full artifact must have exactly phases 001 and 002",
        )
        phase_iterations = Int[]
        phase_modes = String[]
        for phase_name in phase_names
            phase = file["cycle_members/$phase_name"]
            for required in (
                "psi", "applied", "measured", "chemical_potential",
                "iteration", "update_mode",
            )
                haskey(phase, required) || error(
                    "source full orbit phase $phase_name has no $required",
                )
            end
            push!(phase_iterations, Int(read(phase, "iteration")))
            push!(phase_modes, String(read(phase, "update_mode")))
        end
        return (
            status=Symbol(read(file, "status")),
            accepted=Bool(read(file, "accepted")),
            period=Int(read(file, "fundamental_period")),
            model_fingerprint=String(read(file, "provenance/model_fingerprint")),
            numerical_fingerprint=String(read(file, "provenance/numerical_fingerprint")),
            tensor_scalar_type=String(read(file, "provenance/tensor_scalar_type")),
            phase_iterations,
            phase_modes,
        )
    end
    full_metadata.status == Symbol(metadata.status) || error(
        "full and stateless source statuses differ",
    )
    !full_metadata.accepted || error("source full candidate is unexpectedly accepted")
    full_metadata.period == 2 || error(
        "source full candidate period must be 2, found $(full_metadata.period)",
    )
    full_metadata.model_fingerprint == metadata.model_fingerprint || error(
        "full and stateless source model fingerprints differ",
    )
    full_metadata.numerical_fingerprint == metadata.numerical_fingerprint || error(
        "full and stateless source numerical fingerprints differ",
    )
    lowercase(full_metadata.tensor_scalar_type) == lowercase(metadata.tensor_scalar_type) ||
        error("full and stateless source tensor scalar types differ")
    full_metadata.phase_iterations == metadata.phase_iterations || error(
        "full and stateless source phase iterations differ",
    )
    full_metadata.phase_modes == metadata.phase_modes || error(
        "full and stateless source phase update modes differ",
    )
    return merge(metadata, (; full_path))
end

source_state = only_candidate_state(source_run)
source = source_metadata(source_state, source_run)
base_settings = load_settings(base_path)
base_settings.model.geometry == :cubic_unfrustrated || error(
    "recurrence base must use cubic_unfrustrated geometry",
)
LadderMPSMFT.model_fingerprint(base_settings.model) == source.model_fingerprint || error(
    "source candidate and recurrence base have different model fingerprints",
)
base_settings.runtime.backend == :gpu || error("recurrence base must use the GPU backend")
base_settings.runtime.tensor_scalar_type == :float64 || error(
    "recurrence base must request Float64 tensors",
)
lowercase(source.tensor_scalar_type) == "float64" || error(
    "recurrence source must contain Float64 tensors",
)
base_settings.dmrg.maxdim == 400 || error("recurrence base must use chi=400")
base_settings.convergence.cycle_action == :stop || error(
    "recurrence base must stop rather than enter fixed-point acceleration",
)
base_settings.run.max_iterations == base_settings.convergence.probe_iterations + 1 || error(
    "raw-only recurrence run requires max_iterations=probe_iterations+1",
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
        label="unfrustrated__pairing_s1_phase001_chi400",
        seed_label="pairing_s1_phase001",
        random_seed=1101,
        phase=1,
        preparation="orbit_phase_parent",
        direction="from_20260824_phase1_gpu_v3_float64_history_phase001",
    ),
    (
        label="unfrustrated__pairing_s1_phase002_chi400",
        seed_label="pairing_s1_phase002",
        random_seed=1101,
        phase=2,
        preparation="orbit_phase_parent",
        direction="from_20260824_phase1_gpu_v3_float64_history_phase002",
    ),
    (
        label="unfrustrated__pairing_s2_chi400",
        seed_label="pairing_s2",
        random_seed=1102,
        phase=nothing,
        preparation="independent_seed",
        direction="none",
    ),
)

manifest_path = joinpath(control_run, "manifest.tsv")
ispath(manifest_path) && error("refusing to overwrite manifest: $manifest_path")
numerical_fingerprints = String[]
open(manifest_path, "w") do io
    println(io, join((
        "label", "geometry", "branch", "seed", "random_seed", "config", "config_sha256",
        "ep_mode", "ep_signed", "ep_abs", "ep_t0_lower", "ep_t0_upper",
        "ep_lower_signed", "ep_upper_signed", "ep_weight", "tp2_over_ep",
        "parent_checkpoint", "parent_sha256", "parent_orbit_phase", "parent_status",
        "parent_numerical_fingerprint", "parent_tensor_scalar_type",
        "source_phase_iteration", "source_phase_update_mode",
        "full_output_directory", "stateless_output_directory",
    ), '\t'))
    for branch in branches
        raw = TOML.parsefile(base_path)
        run = raw["run"]
        full_output_directory = joinpath(full_run, "results", branch.label)
        stateless_output_directory = joinpath(control_run, "results", branch.label)
        run["output_directory"] = full_output_directory
        run["branch_label"] = "sc"
        run["preparation"] = branch.preparation
        run["direction"] = branch.direction
        run["seed_label"] = branch.seed_label
        run["random_seed"] = branch.random_seed
        run["initial_seed"] = "pairing"
        for key in (
            "inherit_from", "inherit_sha256",
            "parent_checkpoint", "parent_sha256", "parent_orbit_phase",
            "resume_checkpoint", "resume_sha256",
        )
            pop!(run, key, nothing)
        end
        parent_path = ""
        parent_sha256 = ""
        parent_phase = 0
        parent_status = ""
        parent_numerical_fingerprint = ""
        parent_tensor_scalar_type = ""
        source_phase_iteration = 0
        source_phase_update_mode = ""
        if branch.phase !== nothing
            parent_path = source.full_path
            parent_sha256 = source.full_sha256
            parent_phase = branch.phase
            parent_status = source.status
            parent_numerical_fingerprint = source.numerical_fingerprint
            parent_tensor_scalar_type = source.tensor_scalar_type
            source_phase_iteration = source.phase_iterations[branch.phase]
            source_phase_update_mode = source.phase_modes[branch.phase]
            run["parent_checkpoint"] = parent_path
            run["parent_sha256"] = parent_sha256
            run["parent_orbit_phase"] = parent_phase
        end

        config_path = joinpath(config_directory, "$(branch.label).segment-001.toml")
        ispath(config_path) && error("refusing to overwrite recurrence config: $config_path")
        open(config_path, "w") do config_io
            TOML.print(config_io, raw; sorted=true)
        end
        settings = load_settings(config_path)
        push!(numerical_fingerprints, LadderMPSMFT.numerical_fingerprint(settings))
        settings.model.geometry == :cubic_unfrustrated || error(
            "prepared recurrence branch changed geometry: $(branch.label)",
        )
        settings.run.parent_orbit_phase == branch.phase || error(
            "prepared recurrence branch lost its orbit-phase lineage: $(branch.label)",
        )
        model = settings.model
        println(io, join((
            branch.label,
            String(model.geometry),
            "sc",
            "pairing",
            string(branch.random_seed),
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
            parent_path,
            parent_sha256,
            string(parent_phase),
            parent_status,
            parent_numerical_fingerprint,
            parent_tensor_scalar_type,
            string(source_phase_iteration),
            source_phase_update_mode,
            full_output_directory,
            stateless_output_directory,
        ), '\t'))
    end
end

length(unique(numerical_fingerprints)) == 1 || error(
    "recurrence branches do not share one numerical fingerprint",
)
println("source_state=$source_state")
println("source_full_artifact=$(source.full_path)")
println("source_full_sha256=$(source.full_sha256)")
println("branch_count=$(length(branches))")
println("numerical_fingerprint=$(only(unique(numerical_fingerprints)))")
println("manifest_path=$manifest_path")
