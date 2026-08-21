#!/usr/bin/env julia

using LadderMPSMFT

length(ARGS) >= 2 || error(
    "usage: julia --project=. scripts/run_diagnostics.jl CONFIG.toml STATE.h5 [--full-pair] [--sector-gaps]",
)
settings = load_settings(ARGS[1])
state_path = abspath(ARGS[2])
checkpoint = read_checkpoint(state_path)
checkpoint.accepted || error("diagnostics require an accepted fixed-point state; got $(checkpoint.status)")
checkpoint.model_fingerprint == LadderMPSMFT.model_fingerprint(settings.model) ||
    error("state and configuration model fingerprints differ")
configure_threading!(settings.runtime)
full_pair = "--full-pair" in ARGS[3:end]
diagnostics = compute_ladder_diagnostics(checkpoint.psi, settings.model; full_pair_correlations=full_pair)
diagnostic_path = joinpath(dirname(state_path), full_pair ? "diagnostics_full_pair.h5" : "diagnostics.h5")
write_diagnostics(diagnostic_path, diagnostics; state_sha256=LadderMPSMFT.sha256_file(state_path), immutable=true)
println("diagnostics_path=$diagnostic_path")
if "--sector-gaps" in ARGS[3:end]
    gaps = sector_resolved_gaps(settings.model, settings.dmrg)
    gaps_path = joinpath(dirname(state_path), "sector_gaps.h5")
    write_sector_gaps(gaps_path, gaps; immutable=true)
    println("sector_gaps_path=$gaps_path")
    println("weak_coupling=$(validate_weak_coupling(lookup_ep(settings.model.ep_source; L=settings.model.L, U=settings.model.U, V=settings.model.V, t0=settings.model.t0, density=settings.model.density, tp=settings.model.tp); spin_gap=gaps.spin_gap, charge_gap=gaps.charge_gap))")
end
