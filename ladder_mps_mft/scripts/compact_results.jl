#!/usr/bin/env julia

using LadderMPSMFT

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/compact_results.jl FULL_RESULTS_DIRECTORY STATELESS_RESULTS_DIRECTORY",
)

full_results = abspath(ARGS[1])
stateless_results = abspath(ARGS[2])
result = mirror_stateless_tree(full_results, stateless_results; force=true)

full_bytes = sum(record.source_bytes for record in result.records; init=0)
compact_bytes = sum(record.compact_bytes for record in result.records; init=0)
println("full_results_directory=$(result.source)")
println("stateless_results_directory=$(result.destination)")
println("artifact_count=$(length(result.records))")
println("full_bytes=$full_bytes")
println("compact_bytes=$compact_bytes")
println("stateless_manifest=$(result.manifest_path)")
