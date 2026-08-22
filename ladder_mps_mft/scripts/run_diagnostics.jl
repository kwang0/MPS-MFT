#!/usr/bin/env julia

using LadderMPSMFT

length(ARGS) >= 2 || error(
    "usage: julia --project=. scripts/run_diagnostics.jl CONFIG.toml STATE.h5 [--full-pair] [--sector-gaps]",
)
settings = load_settings(ARGS[1])
state_path = abspath(ARGS[2])
checkpoint = read_checkpoint(state_path)
checkpoint.accepted || error("diagnostics require an accepted fixed-point or periodic state; got $(checkpoint.status)")
checkpoint.model_fingerprint == LadderMPSMFT.model_fingerprint(settings.model) ||
    error("state and configuration model fingerprints differ")
configure_threading!(settings.runtime)
full_pair = "--full-pair" in ARGS[3:end]
state_hash = LadderMPSMFT.sha256_file(state_path)
if checkpoint.fundamental_period == 1
    diagnostics = compute_ladder_diagnostics(checkpoint.psi, settings.model; full_pair_correlations=full_pair)
    diagnostic_path = joinpath(dirname(state_path), full_pair ? "diagnostics_full_pair.h5" : "diagnostics.h5")
    write_diagnostics(
        diagnostic_path,
        diagnostics;
        state_sha256=state_hash,
        metadata=Dict("solution_kind" => "fixed_point", "phase" => 1, "period" => 1),
        immutable=true,
    )
    println("diagnostics_path=$diagnostic_path")
else
    checkpoint.orbit_validated || error("periodic state is not an unmixed validated orbit")
    phases = read_orbit_phase_states(state_path)
    length(phases) == checkpoint.fundamental_period || error("state does not contain every orbit-phase MPS")
    for phase in phases
        diagnostics = compute_ladder_diagnostics(phase.psi, settings.model; full_pair_correlations=full_pair)
        suffix = full_pair ? "_full_pair" : ""
        diagnostic_path = joinpath(dirname(state_path), "diagnostics_phase_$(lpad(phase.phase, 3, '0'))$suffix.h5")
        write_diagnostics(
            diagnostic_path,
            diagnostics;
            state_sha256=state_hash,
            metadata=Dict(
                "solution_kind" => "periodic_orbit",
                "phase" => phase.phase,
                "period" => checkpoint.fundamental_period,
                "iteration" => phase.iteration,
            ),
            immutable=true,
        )
        println("diagnostics_path=$diagnostic_path")
    end
end
if "--sector-gaps" in ARGS[3:end]
    gaps = sector_resolved_gaps(settings.model, settings.dmrg)
    gaps_path = joinpath(dirname(state_path), "sector_gaps.h5")
    write_sector_gaps(gaps_path, gaps; immutable=true)
    println("sector_gaps_path=$gaps_path")
    println("weak_coupling=$(validate_weak_coupling(lookup_ep(settings.model.ep_source; L=settings.model.L, U=settings.model.U, V=settings.model.V, t0=settings.model.t0, density=settings.model.density, tp=settings.model.tp); spin_gap=gaps.spin_gap, charge_gap=gaps.charge_gap))")
end
