"""Number of spatial ladder states, never a period of the iteration."""
trellis_ladders(model::ModelSettings) = model.trellis_cell == :one_ladder ? 1 : 2

function _trellis_hopping(model::ModelSettings)
    A = Matrix(Diagonal(fill(model.tau0, model.L)))
    for i in 1:(model.L - 1)
        A[i, i + 1] = model.tau1
    end
    return A
end

"""
Apply the reciprocal OBC trellis kernel with the same range projection on
input and output. Each leg faces a different physical neighbor; the two
neighbors are never combined before squaring the hopping.

One ladder: leg 1 receives A*C00*A', leg 0 receives A'*C11*A.
Rectangular two-ladder cell: A has origin 0 and B origin -1/2. Its two
interfaces are A_1 -> (tau0 B_i0 + tau1 B_(i+1)0) and
A_0 -> (tau1 B_i1 + tau0 B_(i+1)1). B uses their transposes.
"""
function trellis_mean_fields(correlations::AbstractVector{CorrelationState}, model::ModelSettings)
    model.geometry == :trellis || throw(ArgumentError("expected trellis geometry"))
    model.trellis_cell in (:one_ladder, :two_ladder) || throw(ArgumentError("unknown trellis cell"))
    count = trellis_ladders(model)
    length(correlations) == count || throw(DimensionMismatch("one correlation state per spatial ladder is required"))
    mask = [abs(i - j) <= model.r_range for i in 1:model.L, j in 1:model.L]
    A = _trellis_hopping(model)
    C = Matrix(Diagonal(fill(model.tau1, model.L)))
    for i in 1:(model.L - 1)
        C[i, i + 1] = model.tau0
    end
    fields = [FieldState(zeros(model.L, model.L, 2, 2),
                        zeros(2, model.L, model.L, 2, 2), zeros(2, 2model.L)) for _ in 1:count]
    for target in 1:count, leg in 0:1
        neighbor = count == 1 ? 1 : 3 - target
        T = if count == 1
            leg == 0 ? transpose(A) : A
        elseif target == 1
            leg == 0 ? C : A
        else
            leg == 0 ? transpose(A) : transpose(C)
        end
        source_sites = [rung_leg_to_site(i, 1 - leg) for i in 1:model.L]
        target_sites = [rung_leg_to_site(i, leg) for i in 1:model.L]
        corr = correlations[neighbor]
        # Stored pair indices are (up,down); alpha indices are (down,up).
        F = transpose(corr.pair[source_sites, source_sites]) .* mask
        fields[target].alpha[:, :, leg + 1, leg + 1] .= (2 / model.ep) .* (T * F * transpose(T)) .* mask
        for (spin, X, density) in ((1, corr.exchange_down, corr.density_down),
                                    (2, corr.exchange_up, corr.density_up))
            centered = X[source_sites, source_sites] .* mask
            for i in 1:model.L
                centered[i, i] = density[source_sites[i]] - 0.5
            end
            # Includes the off-diagonal -T*T'/Delta one-body contribution.
            B = (2 / model.ep) .* (T * centered * transpose(T)) .* mask
            fields[target].mu_cdw[spin, target_sites] .= diag(B)
            for i in 1:model.L
                B[i, i] = 0.0
            end
            fields[target].beta[spin, :, :, leg + 1, leg + 1] .= B
        end
    end
    return fields
end

_trellis_read_correlations(group) = CorrelationState(
    (Float64.(read(group, String(key))) for key in fieldnames(CorrelationState))...)

function _trellis_initial_states(settings::ProjectSettings)
    model = settings.model
    count = trellis_ladders(model)
    if settings.run.resume_checkpoint !== nothing
        verify_resume!(settings)
        return h5open(settings.run.resume_checkpoint, "r") do file
            read(file, "artifact_kind") == "trellis_mps_mft_state" || error("expected a complete trellis cell checkpoint")
            read(file, "provenance/model_fingerprint") == model_fingerprint(model) || error("trellis resume model mismatch")
            read(file, "spatial_ladders") == count || error("trellis resume cell mismatch")
            map(1:count) do index
                group = file["ladders/$(index == 1 ? "A" : "B")"]
                psi = read(group, "psi", MPS)
                fields = _read_fields(group["fields/restart"])
                _field_shapes_match(fields, model) && length(psi) == 2model.L || error("trellis resume shape mismatch")
                settings.runtime.backend == :gpu && any(ITensors.hasqns, siteinds(psi)) && error("GPU resume requires dense MPS")
                (; sites=siteinds(psi), psi=move_to_backend(psi, settings.runtime), fields,
                   chemical_potential=Float64(read(group, "chemical_potential")))
            end
        end
    end
    settings.run.inherit_from !== nothing || throw(ArgumentError("trellis starts require a correlation-template seed"))
    verify_inherit!(settings)
    fields = h5open(settings.run.inherit_from, "r") do file
        read(file, "artifact_kind") == "trellis_correlation_seed" || error("wrong trellis seed kind")
        read(file, "model_fingerprint") == model_fingerprint(model) || error("trellis seed model mismatch")
        template = _trellis_read_correlations(file["template_correlations"])
        generated = trellis_mean_fields(fill(template, count), model)
        for index in 1:count
            stored = _read_fields(file["ladders/$(index == 1 ? "A" : "B")/fields"])
            all(getfield(stored, key) == getfield(generated[index], key) for key in fieldnames(FieldState)) ||
                error("trellis seed kernel/readback mismatch")
        end
        generated
    end
    return map(1:count) do index
        sites = make_sites(model, settings.runtime)
        # Same initial product-state configuration in both members. Differences
        # are generated by the spatial kernel, not an extra random seed.
        rng = MersenneTwister(settings.run.random_seed)
        psi = productMPS(sites, density_product_state(2model.L, model.density; rng))
        (; sites, psi=move_to_backend(psi, settings.runtime), fields=fields[index],
           chemical_potential=model.mu_initial)
    end
end

function _trellis_cell_diagnostic(histories, settings)
    members = [assess_convergence(records, settings.convergence, settings.model.density) for records in histories]
    accepted = all(d -> d.accepted && d.status == :fixed_point, members)
    current = last.(histories)
    diagnostic = ConvergenceDiagnostic(;
        status=accepted ? :fixed_point : any(d -> d.status == :nonfinite, members) ? :nonfinite : :iterating,
        accepted,
        reason=accepted ? "every spatial ladder passed all stationary raw-map gates in the same cell sweep" :
            "spatial cell is not stationary; temporal recurrence does not constitute a spatial solution",
        solution_kind=accepted ? :fixed_point : :none,
        fundamental_period=accepted ? 1 : 0,
        unmixed_probe=true,
        solution_canonical_variational_energy=accepted ? sum(r.variational.canonical_variational_energy for r in current) : NaN,
        solution_target_density_corrected_variational_energy=accepted ? sum(r.variational.target_density_corrected_variational_energy for r in current) : NaN,
        fixed_point_abs_residual=maximum(d.fixed_point_abs_residual for d in members),
        fixed_point_rel_residual=maximum(d.fixed_point_rel_residual for d in members),
        fixed_point_extrapolated_abs_residual=maximum(d.fixed_point_extrapolated_abs_residual for d in members),
        fixed_point_extrapolated_rel_residual=maximum(d.fixed_point_extrapolated_rel_residual for d in members),
        density_error=maximum(d.density_error for d in members),
        variational_energy_change=maximum(d.variational_energy_change for d in members),
        hamiltonian_identity_error_per_site=maximum(d.hamiltonian_identity_error_per_site for d in members),
        effective_eigenvalue_error_per_site=maximum(d.effective_eigenvalue_error_per_site for d in members),
    )
    return diagnostic, members
end

function _write_trellis_checkpoint(path; settings, psis, histories, diagnostic, members,
                                    restart_fields, provenance, immutable=false)
    immutable && ispath(path) && throw(ArgumentError("refusing to overwrite immutable artifact: $path"))
    mkpath(dirname(path))
    temporary = tempname(dirname(path))
    count = length(histories)
    sites = 2settings.model.L * count
    h5open(temporary, "w") do file
        _write_dict(file, Dict(
            "schema_version" => 1, "artifact_kind" => "trellis_mps_mft_state",
            "spatial_ladders" => count, "physical_sites" => sites,
            "process_completed" => diagnostic.status != :iterating,
            "accepted" => diagnostic.accepted, "completed" => diagnostic.accepted,
            "status" => diagnostic.status, "convergence_reason" => diagnostic.reason,
            "solution_kind" => diagnostic.solution_kind, "fundamental_period" => diagnostic.fundamental_period,
            "orbit_validated" => false,
            "solution_canonical_variational_energy" => diagnostic.solution_canonical_variational_energy,
            "solution_target_density_corrected_variational_energy" => diagnostic.solution_target_density_corrected_variational_energy,
            "energy_normalization" => "total over spatial cell; divide by physical_sites",
        ))
        _write_dict(create_group(file, "model"), Dict(
            String(key) => getfield(settings.model, key) for key in fieldnames(ModelSettings)))
        file["model/transverse_geometry"] = "trellis"
        _write_dict(create_group(file, "provenance"), provenance)
        history = create_group(file, "history")
        history["cell_sweep"] = getfield.(first(histories), :iteration)
        for key in (:canonical_variational_energy, :target_density_corrected_variational_energy)
            totals = [sum(getfield(records[i].variational, key) for records in histories) for i in eachindex(first(histories))]
            history[String(key)] = totals
            history["$(key)_per_site"] = totals ./ sites
        end
        history["wall_seconds"] = [sum(records[i].wall_seconds for records in histories) for i in eachindex(first(histories))]
        ladders = create_group(file, "ladders")
        for index in 1:count
            records = histories[index]
            current = last(records)
            group = create_group(ladders, index == 1 ? "A" : "B")
            group["psi"] = move_to_cpu(psis[index])
            group["chemical_potential"] = current.chemical_potential
            _write_dict(create_group(group, "convergence"), Dict(
                String(key) => getfield(members[index], key) for key in fieldnames(ConvergenceDiagnostic)))
            _write_fields(create_group(group, "fields/initial"), first(records).applied)
            _write_fields(create_group(group, "fields/applied"), current.applied)
            _write_fields(create_group(group, "fields/measured"), current.measured)
            _write_fields(create_group(group, "fields/restart"), restart_fields[index])
            _write_correlations(create_group(group, "correlations"), current.correlations)
            _write_energy(create_group(group, "energy"), current.variational)
            h = create_group(group, "history")
            for key in (:iteration, :density, :chemical_potential, :mu_evaluations,
                        :mu_density_converged, :field_abs_residual, :field_rel_residual, :wall_seconds)
                h[String(key)] = [getfield(record, key) for record in records]
            end
            h["update_mode"] = String.(getfield.(records, :update_mode))
            h["mu_search_status"] = String.(getfield.(records, :mu_search_status))
            for key in fieldnames(EnergyBreakdown)
                h["energy/$(key)"] = [getfield(record.variational, key) for record in records]
            end
            _write_field_history(create_group(h, "fields"), records)
            _write_dmrg_history(create_group(h, "dmrg"), records)
            # Trellis densities mix with bond correlations: store measured
            # correlations rather than reconstructing profiles from mu_cdw.
            for key in fieldnames(CorrelationState)
                values = [getfield(record.correlations, key) for record in records]
                h["correlations/$(key)"] = cat(values...; dims=ndims(first(values)) + 1)
            end
            if settings.convergence.channel_residuals
                diagnostics = [channel_diagnostics(@view(records[1:i]), settings.convergence) for i in eachindex(records)]
                windows = [channel_window_diagnostics(@view(records[1:i]), settings.convergence) for i in eachindex(records)]
                for channel in eachindex(first(diagnostics))
                    g = create_group(h, "channels/$(first(diagnostics)[channel].name)")
                    for key in (:absolute, :relative, :cosine, :contraction, :factor, :passes, :applied_rms, :measured_rms)
                        g[String(key)] = [getproperty(rows[channel], key) for rows in diagnostics]
                    end
                    for key in (:absolute, :relative, :passes)
                        g["window_$(key)"] = [getproperty(rows[channel], key) for rows in windows]
                    end
                end
            end
        end
    end
    mv(temporary, path; force=!immutable)
    return path
end

"""Simultaneous spatial-cell SCF; one sweep solves all members against frozen old fields."""
function run_trellis_scf(settings::ProjectSettings)
    validate_settings(settings)
    ensure_backend!(settings.runtime)
    threading = configure_threading!(settings.runtime)
    starts = _trellis_initial_states(settings)
    count = length(starts)
    output_directory = _run_directory(settings)
    deadline = time() + settings.dmrg.max_time_seconds
    histories = [IterationRecord[] for _ in 1:count]
    psis = [s.psi for s in starts]
    fields = [s.fields for s in starts]
    mus = [s.chemical_potential for s in starts]
    slopes = Union{Nothing,Float64}[nothing for _ in 1:count]
    rngs = [MersenneTwister(settings.run.random_seed) for _ in 1:count]
    bare = [build_bare_ladder_mpo(s.sites, settings.model; backend=settings.runtime) for s in starts]
    provenance = collect_provenance(settings)
    provenance["trellis_update"] = "simultaneous raw spatial-cell map; fixed coordinates"
    provenance["trellis_density_constraint"] = "target density on each ladder separately"
    provenance["trellis_energy"] = "current simultaneous cell correlations; centered normal functional"
    provenance["threading"] = Dict(string(key) => value for (key, value) in pairs(threading))
    provenance["device"] = backend_metadata(settings.runtime)
    diagnostic = ConvergenceDiagnostic()
    members = ConvergenceDiagnostic[]
    for iteration in 1:settings.run.max_iterations
        results = map(1:count) do index
            started = time()
            result = find_mu_for_density(starts[index].sites, settings.model, fields[index], mus[index], settings.dmrg;
                runtime=settings.runtime, psi_init=psis[index], rng=rngs[index], deadline, density_slope=slopes[index])
            psis[index] = result.psi
            mus[index] = result.mu
            isfinite(result.density_slope) && result.density_slope > 0 && (slopes[index] = result.density_slope)
            correlations = measure_correlations(result.psi)
            effective = Float64(real(inner(result.psi', result.hamiltonian, result.psi)))
            bare_energy = Float64(real(inner(result.psi', bare[index], result.psi)))
            (; result, correlations, effective, bare_energy, seconds=time() - started)
        end
        measured = trellis_mean_fields([r.correlations for r in results], settings.model)
        for index in 1:count
            r = results[index]
            absolute, relative = hybrid_distance(measured[index], fields[index])
            energy = variational_energy(r.result.energy, mus[index], fields[index], r.correlations, settings.model;
                interaction_fields=measured[index], effective_expectation=r.effective, bare_ladder_energy=r.bare_energy)
            push!(histories[index], IterationRecord(;
                iteration, update_mode=:unmixed_probe, applied=copy(fields[index]), measured=measured[index],
                correlations=r.correlations, density=r.result.density, chemical_potential=mus[index],
                mu_search_status=r.result.status, mu_evaluations=r.result.evaluations,
                mu_density_converged=r.result.converged, effective_energy=r.result.energy, variational=energy,
                field_abs_residual=absolute, field_rel_residual=relative, wall_seconds=r.seconds,
                mu_density_slope=r.result.density_slope, dmrg_max_discarded_weight=r.result.max_discarded_weight,
                dmrg_maxlinkdim=r.result.maximum_link_dimension, dmrg_sweep_energies=r.result.sweep_energies,
                dmrg_sweep_max_discarded_weights=r.result.sweep_max_discarded_weights,
                dmrg_sweep_maxlinkdims=r.result.sweep_maxlinkdims))
        end
        diagnostic, members = _trellis_cell_diagnostic(histories, settings)
        if any(r -> r.result.timed_out, results) || time() >= deadline
            diagnostic = _copy_diagnostic(diagnostic; status=:time_limit, accepted=false, solution_kind=:none,
                fundamental_period=0, solution_canonical_variational_energy=NaN,
                solution_target_density_corrected_variational_energy=NaN, reason="spatial-cell wall-time deadline reached")
        elseif iteration == settings.run.max_iterations && !diagnostic.accepted && diagnostic.status != :nonfinite
            diagnostic = _copy_diagnostic(diagnostic; status=:maximum_iterations, reason="maximum spatial-cell sweeps reached without stationary self-consistency")
        end
        for index in 1:count
            print("trellis_ladder=$(index == 1 ? "A" : "B") ")
            _print_iteration(last(histories[index]), members[index], results[index].result)
        end
        @printf("TRELLIS cell_sweep=%d ladders=%d Evar_target/site=%.12f status=%s\n", iteration, count,
            sum(last(h).variational.target_density_corrected_variational_energy for h in histories) / (2settings.model.L * count),
            String(diagnostic.status))
        fields = measured
        terminal = diagnostic.status != :iterating
        if iteration % settings.run.save_every == 0 || terminal
            _write_trellis_checkpoint(joinpath(output_directory, "checkpoint_latest.h5"); settings, psis, histories,
                diagnostic, members, restart_fields=fields, provenance)
        end
        terminal && break
    end
    state_path = _write_trellis_checkpoint(joinpath(output_directory, "state.h5"); settings, psis, histories,
        diagnostic, members, restart_fields=fields, provenance, immutable=true)
    summary_path = joinpath(output_directory, "run_summary.md")
    open(summary_path, "w") do io
        println(io, "# Trellis spatial-cell result\n")
        println(io, "Cell: $(settings.model.trellis_cell); $(count) ladder(s), $(length(first(histories))) simultaneous raw sweeps.")
        println(io, "\nStatus: $(diagnostic.status); accepted: $(diagnostic.accepted). $(diagnostic.reason).")
        energy = sum(last(h).variational.target_density_corrected_variational_energy for h in histories) / (2settings.model.L * count)
        println(io, "\nEndpoint target-density-corrected canonical energy: $(energy) t/site. Unaccepted endpoints are diagnostic only.")
        println(io, "\nA/B are spatial states. Each has its own MPS, fields, history and fixed mean density. Temporal recurrences are not accepted.")
    end
    return (; diagnostic, records=histories, state_path, summary_path, output_directory)
end
