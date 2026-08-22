#!/usr/bin/env julia

using LadderMPSMFT

length(ARGS) == 1 || error("usage: julia --project=. scripts/run_scf.jl CONFIG.toml")
settings = load_settings(ARGS[1])
result = run_scf(settings)
println("state_path=$(result.state_path)")
println("summary_path=$(result.summary_path)")
println("status=$(result.diagnostic.status)")
println("accepted=$(result.diagnostic.accepted)")
if settings.run.require_accepted_solution && !result.diagnostic.accepted
    exit(2)
end
