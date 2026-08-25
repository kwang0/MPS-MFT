#!/usr/bin/env julia

using LadderMPSMFT
using TOML

length(ARGS) == 3 || error(
    "usage: julia --project=. scripts/prepare_field_inherit.jl SOURCE_STATE.h5 BASE_CONFIG.toml OUTPUT_CONFIG.toml",
)

source_path = abspath(ARGS[1])
base_path = abspath(ARGS[2])
output_path = abspath(ARGS[3])
isfile(source_path) || error("field-inherit source not found: $source_path")
isfile(base_path) || error("base configuration not found: $base_path")
ispath(output_path) && error("refusing to overwrite field-inherit configuration: $output_path")

inherited = read_inherited_fields(source_path)
raw = TOML.parsefile(base_path)
run = get!(raw, "run", Dict{String,Any}())
for key in (
    "inherit_from", "inherit_sha256",
    "parent_checkpoint", "parent_sha256", "parent_orbit_phase",
    "resume_checkpoint", "resume_sha256",
)
    pop!(run, key, nothing)
end
run["preparation"] = "field_inherit_$(String(inherited.format))"
run["direction"] = "from_$(basename(source_path))"
run["inherit_from"] = source_path
run["inherit_sha256"] = LadderMPSMFT.sha256_file(source_path)

mkpath(dirname(output_path))
open(output_path, "w") do io
    TOML.print(io, raw; sorted=true)
end

settings = load_settings(output_path)
model = settings.model
fields = inherited.fields
size(fields.alpha) == (model.L, model.L, 2, 2) || error(
    "inherited alpha shape $(size(fields.alpha)) does not match L=$(model.L)",
)
size(fields.beta) == (2, model.L, model.L, 2, 2) || error(
    "inherited beta shape $(size(fields.beta)) does not match L=$(model.L)",
)
size(fields.mu_cdw) == (2, 2 * model.L) || error(
    "inherited mu_cdw shape $(size(fields.mu_cdw)) does not match L=$(model.L)",
)

println("inherit_format=$(inherited.format)")
println("inherit_source=$source_path")
inherit_sha256 = run["inherit_sha256"]
println("inherit_sha256=$inherit_sha256")
println("config_path=$output_path")
