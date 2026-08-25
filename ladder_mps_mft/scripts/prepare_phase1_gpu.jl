#!/usr/bin/env julia

using LadderMPSMFT
using HDF5
using TOML

(length(ARGS) == 4 || length(ARGS) == 6) || error(
    "usage: julia --project=. scripts/prepare_phase1_gpu.jl BASE_CONFIG.toml CONTROL_RUN_DIRECTORY FULL_RUN_DIRECTORY RUN_ID [SOURCE_RUN_DIRECTORY SOURCE_RESULTS_DIRECTORY]",
)
base_path = abspath(ARGS[1])
run_directory = abspath(ARGS[2])
full_run_directory = abspath(ARGS[3])
run_id = ARGS[4]
source_run_directory = length(ARGS) == 6 ? abspath(ARGS[5]) : nothing
source_results_directory = length(ARGS) == 6 ? abspath(ARGS[6]) : nothing
isfile(base_path) || error("base configuration not found: $base_path")
occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID: $run_id")
config_directory = joinpath(run_directory, "configs")
ispath(config_directory) && !isempty(readdir(config_directory)) && error(
    "refusing to overwrite nonempty configuration directory: $config_directory",
)
mkpath(config_directory)
mkpath(joinpath(run_directory, "results"))
mkpath(joinpath(full_run_directory, "results"))
base = TOML.parsefile(base_path)

function latest_source_state(source_directory::AbstractString, label::AbstractString)
    root = joinpath(source_directory, label)
    isdir(root) || error("source campaign has no result directory for $label: $root")
    paths = String[]
    for (directory, _, names) in walkdir(root)
        "state.h5" in names && push!(paths, joinpath(directory, "state.h5"))
    end
    isempty(paths) && error("source campaign has no state.h5 for $label")
    sort!(paths; by=path -> (mtime(path), path))
    return last(paths)
end

function source_metadata(path::AbstractString)
    return h5open(path, "r") do file
        tensor_path = "psi/MPS[1]/storage/data"
        haskey(file, tensor_path) || error("source MPS tensor storage is missing: $path")
        return (
            status=String(read(file, "status")),
            model_fingerprint=String(read(file, "provenance/model_fingerprint")),
            numerical_fingerprint=String(read(file, "provenance/numerical_fingerprint")),
            tensor_scalar_type=string(eltype(read(file, tensor_path))),
        )
    end
end

geometries = (
    (name="cubic_frustrated", short="frustrated", offset=0),
    (name="cubic_unfrustrated", short="unfrustrated", offset=1000),
    (name="square", short="square", offset=2000),
)
seeds = (
    (branch="sc", initial="pairing", short="pairing", random=101),
    (branch="sdw", initial="sdw", short="sdw", random=202),
    (branch="cdw", initial="cdw", short="cdw", random=303),
)

manifest_path = joinpath(run_directory, "manifest.tsv")
ispath(manifest_path) && error("refusing to overwrite manifest: $manifest_path")
open(manifest_path, "w") do io
    println(io, join((
        "label", "geometry", "branch", "seed", "random_seed", "config", "config_sha256",
        "ep_mode", "ep_signed", "ep_abs", "ep_t0_lower", "ep_t0_upper",
        "ep_lower_signed", "ep_upper_signed", "ep_weight", "tp2_over_ep",
        "parent_checkpoint", "parent_sha256", "parent_status",
        "parent_numerical_fingerprint", "parent_tensor_scalar_type",
        "full_output_directory", "stateless_output_directory",
    ), '\t'))
    for geometry in geometries, seed in seeds
        label = "$(geometry.short)__$(seed.short)_s1"
        raw = deepcopy(base)
        raw["model"]["geometry"] = geometry.name
        run = raw["run"]
        full_output_directory = joinpath(full_run_directory, "results", label)
        stateless_output_directory = joinpath(run_directory, "results", label)
        run["output_directory"] = full_output_directory
        run["branch_label"] = seed.branch
        run["preparation"] = "independent_seed"
        run["direction"] = "none"
        run["seed_label"] = "$(seed.short)_s1"
        run["random_seed"] = seed.random + geometry.offset
        run["initial_seed"] = seed.initial
        for key in (
            "inherit_from", "inherit_sha256",
            "parent_checkpoint", "parent_sha256", "parent_orbit_phase",
            "resume_checkpoint", "resume_sha256",
        )
            pop!(run, key, nothing)
        end
        parent_path = ""
        parent_sha256 = ""
        parent_status = ""
        parent_numerical_fingerprint = ""
        parent_tensor_scalar_type = ""
        if source_run_directory !== nothing
            parent_path = latest_source_state(source_results_directory, label)
            metadata = source_metadata(parent_path)
            parent_sha256 = LadderMPSMFT.sha256_file(parent_path)
            parent_status = metadata.status
            parent_numerical_fingerprint = metadata.numerical_fingerprint
            parent_tensor_scalar_type = metadata.tensor_scalar_type
            run["preparation"] = "float64_recovery"
            run["direction"] = "from_$(basename(source_run_directory))"
            run["parent_checkpoint"] = parent_path
            run["parent_sha256"] = parent_sha256
        end
        config_path = joinpath(config_directory, "$label.segment-001.toml")
        open(config_path, "w") do config_io
            TOML.print(config_io, raw; sorted=true)
        end
        settings = load_settings(config_path)
        model = settings.model
        if source_run_directory !== nothing
            metadata = source_metadata(parent_path)
            metadata.model_fingerprint == LadderMPSMFT.model_fingerprint(model) || error(
                "source-state model fingerprint differs for $label",
            )
            settings.runtime.tensor_scalar_type == :float64 || error(
                "recovery campaign must request float64 tensors",
            )
        end
        println(io, join((
            label,
            geometry.name,
            seed.branch,
            seed.initial,
            string(run["random_seed"]),
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
            parent_status,
            parent_numerical_fingerprint,
            parent_tensor_scalar_type,
            full_output_directory,
            stateless_output_directory,
        ), '\t'))
    end
end
println("manifest_path=$manifest_path")
