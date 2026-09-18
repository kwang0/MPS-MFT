#!/usr/bin/env julia

using LadderMPSMFT

length(ARGS) >= 2 || error(
    "usage: julia --project=. scripts/run_diagnostics.jl CONFIG.toml STATE.h5 [--allow-unaccepted] [--basic] [--reuse] [--output=DIR] [--sector-gaps]",
)
settings = load_settings(ARGS[1])
state_path = abspath(ARGS[2])
options = ARGS[3:end]
all(arg -> arg in ("--full-pair", "--basic", "--allow-unaccepted", "--reuse", "--sector-gaps") ||
    startswith(arg, "--output="), options) || error("unknown option")
configure_threading!(RuntimeSettings())
outputs = filter(arg -> startswith(arg, "--output="), options)
length(outputs) <= 1 || error("give --output only once")
destination = isempty(outputs) ? dirname(state_path) : split(only(outputs), '='; limit=2)[2]
measure_state_diagnostics(state_path; output_directory=destination,
    full_pair_correlations=!("--basic" in options), allow_unaccepted="--allow-unaccepted" in options,
    reuse="--reuse" in options, expected_model_fingerprint=LadderMPSMFT.model_fingerprint(settings.model))
if "--sector-gaps" in ARGS[3:end]
    gaps = sector_resolved_gaps(settings.model, settings.dmrg)
    gaps_path = joinpath(destination, "sector_gaps.h5")
    write_sector_gaps(gaps_path, gaps; immutable=true)
    println("sector_gaps_path=$gaps_path")
    println("weak_coupling=$(validate_weak_coupling(lookup_ep(settings.model.ep_source; L=settings.model.L, U=settings.model.U, V=settings.model.V, t0=settings.model.t0, density=settings.model.density, tp=settings.model.tp); spin_gap=gaps.spin_gap, charge_gap=gaps.charge_gap))")
end
