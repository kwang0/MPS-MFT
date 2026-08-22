#!/usr/bin/env julia

using LadderMPSMFT
using Printf

isempty(ARGS) && error("usage: julia --project=. scripts/summarize_runs.jl DIRECTORY [--include-incomplete] [--geometry=NAME]")
directory = first(ARGS)
include_incomplete = "--include-incomplete" in ARGS[2:end]
geometry_arg = findfirst(arg -> startswith(arg, "--geometry="), ARGS[2:end])
geometry = geometry_arg === nothing ? nothing : split(ARGS[geometry_arg + 1], "="; limit=2)[2]
rows = select_completed_runs(directory; include_incomplete, geometry)
println("accepted,status,solution_kind,period,orbit_validated,style,geometry,branch,canonical_variational_energy,path")
for row in rows
    @printf("%s,%s,%s,%d,%s,%s,%s,%s,%.16g,%s\n", row.accepted, row.status,
        row.solution_kind, row.period, row.orbit_validated, row.plot_style, row.geometry,
        row.branch, row.energy, row.path)
end
