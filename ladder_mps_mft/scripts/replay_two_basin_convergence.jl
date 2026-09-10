#!/usr/bin/env julia
#= Replay stored field/energy/density/inner-sweep gates without DMRG or relabeling.

Per-iteration Hamiltonian identity/eigenvalue errors are not in these histories;
only terminal values are available. A history-gate pass is not new acceptance.
=#
using LadderMPSMFT, HDF5, CSV, Statistics, TOML

const REPLAY_ROOT = dirname(@__DIR__)
const REPLAY_OUT = joinpath(REPLAY_ROOT, "docs/reports/two_basin_remainder_20260910")

function replay_records(path)
    h5open(path, "r") do f
        h = f["history"]
        iterations = Int.(read(h, "iteration"))
        fields = Dict(source => Dict(key => read(h, "fields/$source/$key")
                      for key in ("alpha", "beta", "mu_cdw")) for source in ("applied", "measured"))
        values = Dict(key => read(h, key) for key in ("density", "chemical_potential",
            "field_abs_residual", "field_rel_residual", "variational_energy", "effective_energy",
            "wall_seconds", "update_mode"))
        corrected = haskey(h, "target_density_corrected_variational_energy") ?
            read(h, "target_density_corrected_variational_energy") :
            values["variational_energy"] .+ 128 .* values["chemical_potential"] .* (.9375 .- values["density"])
        # Correlators and missing energy components remain unknown, not fabricated.
        unknown = CorrelationState(fill(NaN,128,128), fill(NaN,128,128), fill(NaN,128,128), fill(NaN,128), fill(NaN,128))
        records = IterationRecord[]
        for (j, iteration) in enumerate(iterations)
            pair = map(("applied", "measured")) do source
                FieldState((Array(selectdim(fields[source][key], ndims(fields[source][key]), j))
                            for key in ("alpha", "beta", "mu_cdw"))...)
            end
            energies = (; (key => NaN for key in fieldnames(EnergyBreakdown))...)
            energy = EnergyBreakdown(; merge(energies, (
                canonical_variational_energy=values["variational_energy"][j],
                target_density_corrected_variational_energy=corrected[j]))...)
            sweep_path = "dmrg/$(lpad(iteration, 4, '0'))/sweep_energy"
            sweeps = haskey(h, sweep_path) ? Float64.(read(h, sweep_path)) : Float64[]
            push!(records, IterationRecord(iteration=iteration, applied=pair[1], measured=pair[2],
                update_mode=Symbol(values["update_mode"][j]), correlations=unknown,
                density=values["density"][j], chemical_potential=values["chemical_potential"][j],
                mu_search_status=:unknown, mu_evaluations=0, mu_density_converged=false,
                effective_energy=values["effective_energy"][j], variational=energy,
                field_abs_residual=values["field_abs_residual"][j],
                field_rel_residual=values["field_rel_residual"][j], wall_seconds=values["wall_seconds"][j],
                dmrg_sweep_energies=sweeps))
        end
        terminal_consistency = (
            identity_error=Float64(read(f, "hamiltonian_identity_error_per_site")),
            effective_error=Float64(read(f, "effective_eigenvalue_error_per_site")))
        return records, terminal_consistency
    end
end

function history_gates(records, settings)
    recent = @view records[max(1, end-settings.stable_iterations+1):end]
    channels = LadderMPSMFT._channel_window_pass(records, settings)
    slow = LadderMPSMFT._slow_mode_diagnostic(records, settings)
    energy_span = LadderMPSMFT._energy_change(records; window=settings.stable_iterations)
    gates = (observation=length(records) >= max(settings.minimum_iterations, settings.stable_iterations),
        global_fields=all(r -> LadderMPSMFT._field_pass(r, settings), recent), channels,
        global_slow_mode=slow.passes, energy=energy_span <= settings.variational_energy_tol,
        density=all(r -> abs(r.density-.9375) <= settings.density_tol, recent),
        inner_dmrg=all(r -> LadderMPSMFT._dmrg_sweep_pass(r, settings), recent))
    return merge(gates, (all_available_gates=all(values(gates)), energy_span_per_site=energy_span))
end

function main()
    settings = load_settings(joinpath(REPLAY_ROOT, "configs/phase1_gpu_square_two_basin_chi200_raw40.toml")).convergence
    inputs = NamedTuple[]
    anchor_root = joinpath(REPLAY_ROOT, "output/phase1_gpu/20260908_square_two_basin_95_5_80_anchors/results")
    for family in ("stripe", "pairing")
        branch = joinpath(anchor_root, "square__$(family)_weak_other_t014_vm04_chi200_raw")
        path = only([joinpath(root, "state.h5") for (root, _, names) in walkdir(branch) if "state.h5" in names])
        manifest = CSV.File(joinpath(branch, "stateless_manifest.tsv"); delim='\t')
        row = only(filter(row -> row.relative_path == replace(relpath(path, branch), '\\'=>'/'), collect(manifest)))
        push!(inputs, (label="anchor_$(family)", path=path, sha=String(row.compact_sha256), anchor=true))
    end
    for row in CSV.File(joinpath(REPLAY_ROOT, "docs/reports/square_grid_20260908/basin_seed_summary.csv"))
        row.t0 == 1.4 && row.V == 0 || continue
        push!(inputs, (label=String(row.branch), path=joinpath(dirname(REPLAY_ROOT), row.source),
                       sha=String(row.source_sha256), anchor=false))
    end
    output, summary = NamedTuple[], NamedTuple[]
    for input in inputs
        LadderMPSMFT.sha256_file(input.path) == input.sha || error("source hash mismatch: $(input.label)")
        records, terminal = replay_records(input.path)
        raw_end = findfirst(r -> !(r.update_mode in (:initial, :unmixed_probe)), records)
        last_raw = raw_end === nothing ? length(records) : raw_end - 1
        range = input.anchor ? (1:length(records)) : (last_raw:last_raw)
        passes = Int[]
        for i in range
            prefix = @view records[1:i]
            gates = history_gates(prefix, settings)
            gates.all_available_gates && push!(passes, i)
            failed = join([String(r.name) for r in LadderMPSMFT.channel_diagnostics(prefix, settings) if !r.passes], ";")
            window_failed = join([String(r.name) for r in LadderMPSMFT.channel_window_diagnostics(prefix, settings) if !r.passes], ";")
            push!(output, merge((label=input.label, iteration=i), gates,
                (current_failed_channels=failed, span_failed_channels=window_failed)))
        end
        push!(summary, (label=input.label, records=length(records), raw_records=last_raw,
            first_available_gate_pass=isempty(passes) ? 0 : first(passes),
            first_pass_by_40=any(i -> i <= 40, passes), source_sha256=input.sha,
            source=replace(relpath(input.path, REPLAY_ROOT), '\\'=>'/'),
            terminal_identity_pass=terminal.identity_error <= settings.hamiltonian_identity_tol,
            terminal_effective_pass=terminal.effective_error <= settings.effective_energy_consistency_tol))
        LadderMPSMFT.sha256_file(input.path) == input.sha || error("source changed during replay")
        println(last(summary))
    end
    mkpath(REPLAY_OUT)
    CSV.write(joinpath(REPLAY_OUT, "history_gate_replay.csv"), output)
    CSV.write(joinpath(REPLAY_OUT, "replay_sources.csv"), summary)
    open(joinpath(REPLAY_OUT, "replay_contract.toml"), "w") do io
        TOML.print(io, Dict(
            "config_sha256" => LadderMPSMFT.sha256_file(joinpath(REPLAY_ROOT, "configs/phase1_gpu_square_two_basin_chi200_raw40.toml")),
            "implementation_sha256_local" => implementation_fingerprint(),
            "boundary" => "History gates only; per-iteration identity and eigenvalue errors unavailable; original flags unchanged."))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
