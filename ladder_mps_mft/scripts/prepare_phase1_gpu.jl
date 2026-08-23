#!/usr/bin/env julia

using LadderMPSMFT
using TOML

length(ARGS) == 3 || error(
    "usage: julia --project=. scripts/prepare_phase1_gpu.jl BASE_CONFIG.toml RUN_DIRECTORY RUN_ID",
)
base_path = abspath(ARGS[1])
run_directory = abspath(ARGS[2])
run_id = ARGS[3]
isfile(base_path) || error("base configuration not found: $base_path")
occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID: $run_id")
config_directory = joinpath(run_directory, "configs")
ispath(config_directory) && !isempty(readdir(config_directory)) && error(
    "refusing to overwrite nonempty configuration directory: $config_directory",
)
mkpath(config_directory)
mkpath(joinpath(run_directory, "results"))
base = TOML.parsefile(base_path)

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
    ), '\t'))
    for geometry in geometries, seed in seeds
        label = "$(geometry.short)__$(seed.short)_s1"
        raw = deepcopy(base)
        raw["model"]["geometry"] = geometry.name
        run = raw["run"]
        run["output_directory"] = joinpath(run_directory, "results", label)
        run["branch_label"] = seed.branch
        run["preparation"] = "independent_seed"
        run["direction"] = "none"
        run["seed_label"] = "$(seed.short)_s1"
        run["random_seed"] = seed.random + geometry.offset
        run["initial_seed"] = seed.initial
        config_path = joinpath(config_directory, "$label.segment-001.toml")
        open(config_path, "w") do config_io
            TOML.print(config_io, raw; sorted=true)
        end
        settings = load_settings(config_path)
        model = settings.model
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
        ), '\t'))
    end
end
println("manifest_path=$manifest_path")
