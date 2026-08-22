#!/usr/bin/env julia

using TOML

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/prepare_branch_scan.jl BASE_CONFIG.toml OUTPUT_DIRECTORY",
)
base_path = abspath(ARGS[1])
output_directory = abspath(ARGS[2])
isfile(base_path) || error("base configuration not found: $base_path")
mkpath(output_directory)
raw = TOML.parsefile(base_path)
run = get!(raw, "run", Dict{String,Any}())
run["output_directory"] = joinpath(output_directory, "results")
run["preparation"] = "independent_seed"
run["direction"] = "none"

branches = (
    (name="sc", initial_seed="pairing", seed_label="independent_sc", random_seed=101),
    (name="sdw", initial_seed="sdw", seed_label="independent_sdw", random_seed=202),
    (name="cdw", initial_seed="cdw", seed_label="independent_cdw", random_seed=303),
)
paths = String[]
for branch in branches
    branch_raw = deepcopy(raw)
    branch_run = branch_raw["run"]
    branch_run["branch_label"] = branch.name
    branch_run["initial_seed"] = branch.initial_seed
    branch_run["seed_label"] = branch.seed_label
    branch_run["random_seed"] = branch.random_seed
    path = joinpath(output_directory, "$(branch.name).toml")
    ispath(path) && error("refusing to overwrite branch configuration: $path")
    open(path, "w") do io
        TOML.print(io, branch_raw; sorted=true)
    end
    push!(paths, path)
end

manifest_path = joinpath(output_directory, "BRANCH_MANIFEST.md")
ispath(manifest_path) && error("refusing to overwrite branch manifest: $manifest_path")
open(manifest_path, "w") do io
    println(io, "# Independent phase-seed manifest")
    println(io)
    println(io, "Base configuration: `$base_path`")
    println(io)
    println(io, "Run each branch independently:")
    println(io)
    for path in paths
        println(io, "- `julia --project=. scripts/run_scf.jl $path`")
    end
    println(io)
    println(io, "Only after all branches are accepted fixed points or unmixed validated periodic solutions, compare their immutable `state.h5` files with `scripts/compare_branches.jl`. The comparison script rejects mismatched fingerprints and mixer-dependent recurrences.")
end
println("manifest_path=$manifest_path")
