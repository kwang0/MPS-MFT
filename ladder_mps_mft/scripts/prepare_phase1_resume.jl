#!/usr/bin/env julia

using HDF5
using LadderMPSMFT
using TOML

length(ARGS) == 3 || error(
    "usage: julia --project=. scripts/prepare_phase1_resume.jl SOURCE_STATE.h5 PREVIOUS_CONFIG.toml OUTPUT_CONFIG.toml",
)
source_path = abspath(ARGS[1])
previous_path = abspath(ARGS[2])
output_path = abspath(ARGS[3])
isfile(source_path) || error("source state not found: $source_path")
isfile(previous_path) || error("previous configuration not found: $previous_path")
ispath(output_path) && error("refusing to overwrite resume configuration: $output_path")
state = h5open(source_path, "r") do file
    return (
        accepted=Bool(read(file, "accepted")),
        status=Symbol(read(file, "status")),
        model_fingerprint=String(read(file, "provenance/model_fingerprint")),
    )
end
state.accepted && error("source state is already accepted; no continuation is needed")
allowed = Set((:time_limit, :maximum_iterations))
state.status in allowed || get(ENV, "PHASE1_ALLOW_UNSAFE_CONTINUE", "0") == "1" || error(
    "automatic continuation is allowed only for time_limit or maximum_iterations, not $(state.status); " *
    "set PHASE1_ALLOW_UNSAFE_CONTINUE=1 only after inspecting the branch",
)
settings = load_settings(previous_path)
state.model_fingerprint == LadderMPSMFT.model_fingerprint(settings.model) || error(
    "source-state model fingerprint differs from the previous configuration",
)
raw = TOML.parsefile(previous_path)
run = raw["run"]
for key in (
    "inherit_from", "inherit_sha256",
    "parent_checkpoint", "parent_sha256", "parent_orbit_phase",
    "resume_checkpoint", "resume_sha256",
)
    pop!(run, key, nothing)
end
run["preparation"] = "same_model_resume"
run["resume_checkpoint"] = source_path
run["resume_sha256"] = LadderMPSMFT.sha256_file(source_path)
mkpath(dirname(output_path))
open(output_path, "w") do io
    TOML.print(io, raw; sorted=true)
end
load_settings(output_path)
println("source_status=$(state.status)")
resume_sha256 = run["resume_sha256"]
println("source_sha256=$resume_sha256")
println("config_path=$output_path")
