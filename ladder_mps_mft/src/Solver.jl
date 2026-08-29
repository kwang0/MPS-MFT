mutable struct EnergyTimerObserver <: AbstractObserver
    energy_tol::Float64
    last_energy::Float64
    deadline::Float64
    timed_out::Bool
    converged::Bool
end

EnergyTimerObserver(energy_tol::Real, deadline::Real) =
    EnergyTimerObserver(Float64(energy_tol), NaN, Float64(deadline), false, false)

function ITensorMPS.checkdone!(observer::EnergyTimerObserver; energy, sweep, kwargs...)
    value = Float64(real(energy))
    if observer.energy_tol > 0 && isfinite(observer.last_energy)
        relative_change = abs(value - observer.last_energy) / max(abs(value), 1.0)
        if relative_change <= observer.energy_tol
            observer.converged = true
            return true
        end
    end
    observer.last_energy = value
    if time() >= observer.deadline
        observer.timed_out = true
        return true
    end
    return false
end

function density_product_state(number_sites::Integer, density::Real; rng=MersenneTwister(1))
    sites = Int(number_sites)
    0 < density <= 2 || throw(ArgumentError("density must lie in (0,2] per site"))
    particle_number = clamp(round(Int, Float64(density) * sites), 0, 2 * sites)
    n_up = div(particle_number, 2)
    n_down = particle_number - n_up
    doublons = max(0, n_up + n_down - sites)
    up_only = n_up - doublons
    down_only = n_down - doublons
    empty = sites - doublons - up_only - down_only
    states = vcat(
        fill("UpDn", doublons),
        fill("Up", up_only),
        fill("Dn", down_only),
        fill("Emp", empty),
    )
    shuffle!(rng, states)
    return states
end

function average_density(psi::MPS)
    return sum(real, expect(psi, "Ntot")) / length(psi)
end

function _extend_schedule(values::AbstractVector, count::Integer)
    count >= 1 || throw(ArgumentError("schedule length must be positive"))
    return [values[min(index, length(values))] for index in 1:count]
end

function run_dmrg_ground(
    sites,
    hamiltonian::MPO,
    target_density::Real,
    settings::DMRGSettings;
    psi_init=nothing,
    rng=MersenneTwister(1),
    deadline::Real=Inf,
    backend::Union{Symbol,RuntimeSettings}=:cpu,
)
    warm_start = psi_init !== nothing
    psi0_cpu = warm_start ? psi_init : productMPS(sites, density_product_state(length(sites), target_density; rng))
    psi0 = move_to_backend(psi0_cpu, backend)
    sweeps = Sweeps(settings.nsweeps)
    maxdims = warm_start ? fill(settings.maxdim, settings.nsweeps) :
        _extend_schedule([min(10, settings.maxdim), min(20, settings.maxdim), min(100, settings.maxdim), settings.maxdim], settings.nsweeps)
    noises = warm_start ? _extend_schedule([1e-3, 1e-4, 1e-5, 1e-6, 0.0], settings.nsweeps) :
        _extend_schedule([1e-5, 1e-6, 1e-7, 1e-8, 0.0], settings.nsweeps)
    maxdim!(sweeps, maxdims...)
    noise!(sweeps, noises...)
    cutoff!(sweeps, settings.cutoff)
    observer = EnergyTimerObserver(settings.energy_tol, deadline)
    energy, psi = dmrg(
        hamiltonian,
        psi0,
        sweeps;
        observer,
        outputlevel=settings.output_level,
        eigsolve_krylovdim=settings.eigsolve_krylovdim,
    )
    return (
        energy=Float64(real(energy)),
        psi,
        timed_out=observer.timed_out,
        energy_converged=observer.converged,
    )
end

function _solve_mu_point(
    sites,
    model::ModelSettings,
    fields::FieldState,
    mu::Real,
    settings::DMRGSettings;
    runtime::RuntimeSettings,
    psi_init,
    rng,
    deadline,
)
    hamiltonian = build_mf_mpo(sites, model, fields, mu; backend=runtime)
    result = run_dmrg_ground(
        sites,
        hamiltonian,
        model.density,
        settings;
        psi_init,
        rng,
        deadline,
        backend=runtime,
    )
    return (
        mu=Float64(mu),
        density=average_density(result.psi),
        energy=result.energy,
        psi=result.psi,
        hamiltonian,
        timed_out=result.timed_out,
        energy_converged=result.energy_converged,
    )
end

_density_error(point, target) = abs(point.density - target)

function find_mu_for_density(
    sites,
    model::ModelSettings,
    fields::FieldState,
    mu_initial::Real,
    settings::DMRGSettings;
    runtime::RuntimeSettings=RuntimeSettings(),
    psi_init=nothing,
    rng=MersenneTwister(1),
    deadline::Real=Inf,
)
    target = model.density
    evaluations = 0
    point = _solve_mu_point(
        sites,
        model,
        fields,
        mu_initial,
        settings;
        runtime,
        psi_init,
        rng,
        deadline,
    )
    evaluations += 1
    best = point
    if point.timed_out
        return merge(point, (; converged=false, status=:time_limit, evaluations))
    elseif _density_error(point, target) <= settings.mu_density_tol
        return merge(point, (; converged=true, status=:density_tolerance, evaluations))
    end

    direction = point.density < target ? 1.0 : -1.0
    step = settings.mu_bracket_step
    previous = point
    bracket = nothing
    while evaluations < settings.mu_max_iterations
        candidate_mu = previous.mu + direction * step
        candidate = _solve_mu_point(
            sites,
            model,
            fields,
            candidate_mu,
            settings;
            runtime,
            psi_init=previous.psi,
            rng,
            deadline,
        )
        evaluations += 1
        _density_error(candidate, target) < _density_error(best, target) && (best = candidate)
        if candidate.timed_out
            return merge(best, (; converged=false, status=:time_limit, evaluations, timed_out=true))
        elseif _density_error(candidate, target) <= settings.mu_density_tol
            return merge(candidate, (; converged=true, status=:density_tolerance, evaluations))
        elseif (previous.density - target) * (candidate.density - target) <= 0
            bracket = (previous, candidate)
            break
        end
        previous = candidate
        step *= settings.mu_bracket_growth
    end

    bracket === nothing && return merge(best, (; converged=false, status=:unbracketed, evaluations))
    left, right = bracket
    left.mu > right.mu && ((left, right) = (right, left))
    while evaluations < settings.mu_max_iterations
        width = right.mu - left.mu
        width <= settings.mu_interval_tol && return merge(best, (; converged=false, status=:mu_interval_tolerance, evaluations))
        left_residual = left.density - target
        right_residual = right.density - target
        denominator = right_residual - left_residual
        candidate_mu = abs(denominator) > 1e-12 ?
            right.mu - right_residual * width / denominator :
            0.5 * (left.mu + right.mu)
        guard = 0.1 * width
        if !isfinite(candidate_mu) || candidate_mu <= left.mu + guard || candidate_mu >= right.mu - guard
            candidate_mu = 0.5 * (left.mu + right.mu)
        end
        initial = abs(candidate_mu - left.mu) <= abs(candidate_mu - right.mu) ? left.psi : right.psi
        candidate = _solve_mu_point(
            sites,
            model,
            fields,
            candidate_mu,
            settings;
            runtime,
            psi_init=initial,
            rng,
            deadline,
        )
        evaluations += 1
        _density_error(candidate, target) < _density_error(best, target) && (best = candidate)
        if candidate.timed_out
            return merge(best, (; converged=false, status=:time_limit, evaluations, timed_out=true))
        elseif _density_error(candidate, target) <= settings.mu_density_tol
            return merge(candidate, (; converged=true, status=:density_tolerance, evaluations))
        elseif (left.density - target) * (candidate.density - target) <= 0
            right = candidate
        else
            left = candidate
        end
    end
    return merge(best, (; converged=false, status=:maximum_mu_iterations, evaluations))
end

function _field_shapes_match(fields::FieldState, model::ModelSettings)
    return size(fields.alpha) == (model.L, model.L, 2, 2) &&
        size(fields.beta) == (2, model.L, model.L, 2, 2) &&
        size(fields.mu_cdw) == (2, 2 * model.L)
end

function _initial_state(settings::ProjectSettings)
    model = settings.model
    if settings.run.inherit_from !== nothing
        verify_inherit!(settings)
        inherited = read_inherited_fields(settings.run.inherit_from)
        _field_shapes_match(inherited.fields, model) || throw(DimensionMismatch(
            "inherited fields do not match model.L=$(model.L)",
        ))
        if inherited.source_geometry !== nothing
            source_geometry = normalize_geometry(inherited.source_geometry)
            source_geometry == model.geometry || @warn(
                "field-only inheritance crosses transverse geometries",
                inherited_geometry=source_geometry,
                requested_geometry=model.geometry,
            )
        end
        rng = MersenneTwister(settings.run.random_seed)
        sites = make_sites(model, settings.runtime)
        psi = move_to_backend(
            productMPS(sites, density_product_state(2 * model.L, model.density; rng)),
            settings.runtime,
        )
        return (
            sites,
            psi,
            fields=inherited.fields,
            chemical_potential=inherited.chemical_potential,
            source="field_inherit",
            inherit_format=String(inherited.format),
            inherit_source_geometry=something(inherited.source_geometry, ""),
        )
    end
    checkpoint_path = settings.run.resume_checkpoint === nothing ?
        settings.run.parent_checkpoint : settings.run.resume_checkpoint
    if checkpoint_path !== nothing
        settings.run.resume_checkpoint === nothing ? verify_parent!(settings) : verify_resume!(settings)
        checkpoint = read_checkpoint(
            checkpoint_path;
            orbit_phase=settings.run.resume_checkpoint === nothing ?
                settings.run.parent_orbit_phase : nothing,
        )
        length(checkpoint.psi) == 2 * model.L || throw(DimensionMismatch("checkpoint MPS length does not match model.L"))
        _field_shapes_match(checkpoint.restart, model) || throw(DimensionMismatch("checkpoint fields do not match model.L"))
        if settings.run.resume_checkpoint !== nothing
            checkpoint.model_fingerprint == model_fingerprint(model) || throw(ArgumentError(
                "resume checkpoint model fingerprint differs from the requested model; use parent_checkpoint for a continuation seed",
            ))
        end
        checkpoint_sites = siteinds(checkpoint.psi)
        if settings.runtime.backend == :gpu && any(ITensors.hasqns, checkpoint_sites)
            throw(ArgumentError(
                "the dense GPU backend cannot resume a QN block-sparse checkpoint; " *
                "start from a dense checkpoint or an independent GPU seed",
            ))
        end
        return (
            sites=checkpoint_sites,
            psi=move_to_backend(checkpoint.psi, settings.runtime),
            fields=checkpoint.restart,
            chemical_potential=checkpoint.chemical_potential,
            source=settings.run.resume_checkpoint !== nothing ? "resume" :
                settings.run.parent_orbit_phase === nothing ? "parent" :
                "parent_orbit_phase_$(lpad(string(settings.run.parent_orbit_phase), 3, '0'))",
            inherit_format="none",
            inherit_source_geometry="",
        )
    end
    rng = MersenneTwister(settings.run.random_seed)
    sites = make_sites(model, settings.runtime)
    psi = move_to_backend(
        productMPS(sites, density_product_state(2 * model.L, model.density; rng)),
        settings.runtime,
    )
    fields = initial_fields(
        model;
        seed=settings.run.initial_seed,
        amplitude=settings.run.initial_amplitude,
        rng,
        protocol=settings.run.initial_seed_protocol,
        mode_number=settings.run.initial_mode_number,
        mode_phase_pi=settings.run.initial_mode_phase_pi,
        pairing_form_factor=settings.run.initial_pairing_form_factor,
        leg_parity=settings.run.initial_leg_parity,
    )
    return (;
        sites,
        psi,
        fields,
        chemical_potential=model.mu_initial,
        source="independent",
        inherit_format="none",
        inherit_source_geometry="",
    )
end

function _copy_diagnostic(
    diagnostic::ConvergenceDiagnostic;
    status::Symbol=diagnostic.status,
    accepted::Bool=diagnostic.accepted,
    reason::String=diagnostic.reason,
    solution_kind::Symbol=diagnostic.solution_kind,
    fundamental_period::Int=diagnostic.fundamental_period,
    orbit_validated::Bool=diagnostic.orbit_validated,
    unmixed_probe::Bool=diagnostic.unmixed_probe,
    solution_canonical_variational_energy::Float64=diagnostic.solution_canonical_variational_energy,
    orbit_energy_spread::Float64=diagnostic.orbit_energy_spread,
    orbit_density_contrast::Float64=diagnostic.orbit_density_contrast,
)
    return ConvergenceDiagnostic(;
        status=status,
        accepted=accepted,
        reason=reason,
        solution_kind=solution_kind,
        fundamental_period=fundamental_period,
        orbit_validated=orbit_validated,
        unmixed_probe=unmixed_probe,
        solution_canonical_variational_energy=solution_canonical_variational_energy,
        orbit_energy_spread=orbit_energy_spread,
        orbit_density_contrast=orbit_density_contrast,
        fixed_point_abs_residual=diagnostic.fixed_point_abs_residual,
        fixed_point_rel_residual=diagnostic.fixed_point_rel_residual,
        cycle_abs_residual=diagnostic.cycle_abs_residual,
        cycle_rel_residual=diagnostic.cycle_rel_residual,
        density_error=diagnostic.density_error,
        variational_energy_change=diagnostic.variational_energy_change,
        hamiltonian_identity_error_per_site=diagnostic.hamiltonian_identity_error_per_site,
        effective_eigenvalue_error_per_site=diagnostic.effective_eigenvalue_error_per_site,
        best_iteration=diagnostic.best_iteration,
    )
end

function _clear_solution_diagnostic(diagnostic::ConvergenceDiagnostic, reason::String)
    return _copy_diagnostic(
        diagnostic;
        status=:iterating,
        accepted=false,
        reason,
        solution_kind=:none,
        fundamental_period=0,
        orbit_validated=false,
        unmixed_probe=false,
        solution_canonical_variational_energy=NaN,
        orbit_energy_spread=NaN,
        orbit_density_contrast=NaN,
    )
end

"""
Choose how the SCF driver responds to a detected recurrence.

Raw-map candidates are allowed to use the full configured probe before they
can stop or hand off to mixing. A mixer-dependent recurrence always receives
one fresh raw-map probe; if that controlled probe remains unaccepted, the run
stops for inspection instead of repeatedly damping or re-probing the orbit.
"""
function _recurrence_action(
    diagnostic::ConvergenceDiagnostic,
    settings::ConvergenceSettings;
    probe_steps::Integer,
    probe_origin::Symbol,
    mixer_probe_completed::Bool,
)
    diagnostic.status in (:periodic_solution, :periodic_candidate) || return :none
    diagnostic.accepted && return :accept
    if diagnostic.unmixed_probe
        probe_steps < settings.probe_iterations && return :continue_probe
        probe_origin == :mixer_recurrence && return :stop_candidate
        return settings.cycle_action == :continue ? :enter_mixing : :stop_candidate
    end
    return settings.unmixed_cycle_probe && !mixer_probe_completed ?
        :start_mixer_probe : :stop_candidate
end

function _run_directory(settings::ProjectSettings)
    label = join((
        String(settings.model.geometry),
        settings.run.branch_label,
        settings.run.preparation,
        settings.run.direction,
        settings.run.seed_label,
    ), "__")
    safe_label = replace(label, r"[^A-Za-z0-9_.-]+" => "-")
    stamp = Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS")
    suffix = first(model_fingerprint(settings.model), 12)
    directory = joinpath(settings.run.output_directory, safe_label, "$(stamp)_$(getpid())_$(suffix)")
    mkpath(directory)
    return directory
end

function _print_iteration(record::IterationRecord, diagnostic::ConvergenceDiagnostic, mu_result)
    @printf(
        "MF %3d  n=%.9f  mu=% .8f  r_abs=%.3e  r_rel=%.3e  Evar/site=% .12f  mu_evals=%d  status=%s\n",
        record.iteration,
        record.density,
        record.chemical_potential,
        record.field_abs_residual,
        record.field_rel_residual,
        record.variational.canonical_variational_energy / (2 * size(record.applied.alpha, 1)),
        mu_result.evaluations,
        String(diagnostic.status),
    )
end

function run_scf(settings::ProjectSettings)
    validate_settings(settings)
    ensure_backend!(settings.runtime)
    threading = configure_threading!(settings.runtime)
    start = _initial_state(settings)
    rng = MersenneTwister(settings.run.random_seed)
    output_directory = _run_directory(settings)
    deadline = time() + settings.dmrg.max_time_seconds
    records = IterationRecord[]
    mixing_state = MixingState(settings.mixing)
    fields = start.fields
    psi = start.psi
    chemical_potential = start.chemical_potential
    diagnostic = ConvergenceDiagnostic()
    best_residual = Inf
    update_mode = :initial
    probe_active = settings.convergence.unmixed_cycle_probe
    probe_steps = 0
    probe_origin = probe_active ? :initial : :none
    mixer_probe_completed = false
    phase_psis = Dict{Int,MPS}()
    provenance = collect_provenance(settings)
    provenance["initial_state_source"] = start.source
    provenance["inherit_format"] = start.inherit_format
    provenance["inherit_source_geometry"] = start.inherit_source_geometry
    provenance["threading"] = Dict(string(key) => value for (key, value) in pairs(threading))
    provenance["device"] = backend_metadata(settings.runtime)
    bare_hamiltonian = build_bare_ladder_mpo(
        start.sites,
        settings.model;
        backend=settings.runtime,
    )

    for iteration in 1:settings.run.max_iterations
        iteration_start = time()
        mu_result = find_mu_for_density(
            start.sites,
            settings.model,
            fields,
            chemical_potential,
            settings.dmrg;
            runtime=settings.runtime,
            psi_init=psi,
            rng,
            deadline,
        )
        psi = mu_result.psi
        chemical_potential = mu_result.mu
        measured, correlations = calculate_mean_fields(psi, settings.model; threshold=0.0)
        absolute_residual, relative_residual = hybrid_distance(measured, fields)
        effective_expectation = Float64(real(inner(psi', mu_result.hamiltonian, psi)))
        bare_energy = Float64(real(inner(psi', bare_hamiltonian, psi)))
        energy = variational_energy(
            mu_result.energy,
            chemical_potential,
            fields,
            correlations,
            settings.model,
            interaction_fields=fields,
            effective_expectation=effective_expectation,
            bare_ladder_energy=bare_energy,
        )
        record = IterationRecord(;
            iteration=iteration,
            update_mode,
            applied=copy(fields),
            measured=measured,
            correlations=correlations,
            density=mu_result.density,
            chemical_potential=chemical_potential,
            mu_search_status=mu_result.status,
            mu_evaluations=mu_result.evaluations,
            mu_density_converged=mu_result.converged,
            effective_energy=mu_result.energy,
            variational=energy,
            field_abs_residual=absolute_residual,
            field_rel_residual=relative_residual,
            wall_seconds=time() - iteration_start,
        )
        push!(records, record)
        if update_mode == :unmixed_probe
            probe_steps += 1
            phase_psis[iteration] = deepcopy(psi)
            while length(phase_psis) > settings.convergence.probe_max_period
                delete!(phase_psis, minimum(keys(phase_psis)))
            end
        end
        diagnostic = assess_convergence(records, settings.convergence, settings.model.density)
        mu_result.timed_out && (diagnostic = _copy_diagnostic(
            diagnostic;
            status=:time_limit,
            accepted=false,
            reason="wall-time deadline reached during density-targeted DMRG",
        ))
        _print_iteration(record, diagnostic, mu_result)

        terminal = diagnostic.status in (
            :fixed_point,
            :periodic_solution,
            :stagnated,
            :diverging,
            :nonfinite,
            :time_limit,
        )
        force_unmixed_probe = false
        force_mixing = false
        if diagnostic.status in (:periodic_solution, :periodic_candidate)
            recurrence_action = _recurrence_action(
                diagnostic,
                settings.convergence;
                probe_steps,
                probe_origin,
                mixer_probe_completed,
            )
            if recurrence_action == :stop_candidate && !diagnostic.unmixed_probe && mixer_probe_completed
                diagnostic = _copy_diagnostic(
                    diagnostic;
                    reason="mixer-dependent recurrence persisted after its controlled raw-map probe; no physical orbit was accepted",
                )
            end
            if recurrence_action != :continue_probe
                cycle_path = joinpath(
                    output_directory,
                    @sprintf("orbit_period_%02d_iter_%04d.h5", diagnostic.fundamental_period, iteration),
                )
                write_checkpoint(
                    cycle_path;
                    settings,
                    psi,
                    records,
                    diagnostic,
                    restart_fields=measured,
                    chemical_potential,
                    provenance,
                    immutable=true,
                    phase_psis,
                )
            end
            if recurrence_action == :accept || recurrence_action == :stop_candidate
                terminal = true
            elseif recurrence_action == :continue_probe
                diagnostic = _clear_solution_diagnostic(
                    diagnostic,
                    "raw-map recurrence detected before the probe budget was exhausted; continuing the unmixed probe",
                )
            elseif recurrence_action == :enter_mixing
                probe_active = false
                probe_origin = :none
                empty!(phase_psis)
                empty!(mixing_state.x_history)
                empty!(mixing_state.f_history)
                empty!(mixing_state.residual_norms)
                mixing_state.damping = max(settings.mixing.minimum_damping, 0.5 * mixing_state.damping)
                force_mixing = true
                diagnostic = _clear_solution_diagnostic(
                    diagnostic,
                    "exhaustive initial raw-map probe archived an unaccepted recurrence; entering accelerated mixing with reduced damping",
                )
            elseif recurrence_action == :start_mixer_probe
                probe_active = true
                probe_steps = 0
                probe_origin = :mixer_recurrence
                mixer_probe_completed = true
                empty!(phase_psis)
                empty!(mixing_state.x_history)
                empty!(mixing_state.f_history)
                empty!(mixing_state.residual_norms)
                force_unmixed_probe = true
                diagnostic = _clear_solution_diagnostic(
                    diagnostic,
                    "mixer-dependent recurrence archived; starting a fresh unmixed raw-map probe",
                )
            end
        end
        if !terminal && update_mode == :unmixed_probe && probe_active &&
                probe_steps >= settings.convergence.probe_iterations &&
                diagnostic.status == :iterating && !force_mixing
            completed_origin = probe_origin
            probe_active = false
            probe_origin = :none
            empty!(phase_psis)
            empty!(mixing_state.x_history)
            empty!(mixing_state.f_history)
            empty!(mixing_state.residual_norms)
            force_mixing = true
            diagnostic = _clear_solution_diagnostic(
                diagnostic,
                "$(completed_origin) unmixed cycle probe completed without an accepted recurrence; entering accelerated mixing",
            )
        end
        if iteration == settings.run.max_iterations && !terminal
            diagnostic = _copy_diagnostic(
                diagnostic;
                status=:maximum_iterations,
                accepted=false,
                reason="maximum SCF iterations reached without an accepted fixed point or periodic solution",
            )
            terminal = true
        end

        next_fields = measured
        next_update_mode = update_mode
        if !terminal
            if force_unmixed_probe || (probe_active && probe_steps < settings.convergence.probe_iterations)
                next_fields = measured
                next_update_mode = :unmixed_probe
            else
                next_fields, mixing_metadata = mix_fields!(mixing_state, fields, measured, settings.mixing)
                next_update_mode = mixing_metadata.method
            end
        end
        if iteration % settings.run.save_every == 0 || terminal
            write_checkpoint(
                joinpath(output_directory, "checkpoint_latest.h5");
                settings,
                psi,
                records,
                diagnostic,
                restart_fields=next_fields,
                chemical_potential,
                provenance,
            )
        end
        if relative_residual < best_residual
            best_residual = relative_residual
            write_checkpoint(
                joinpath(output_directory, "checkpoint_best.h5");
                settings,
                psi,
                records,
                diagnostic,
                restart_fields=next_fields,
                chemical_potential,
                provenance,
            )
        end
        fields = next_fields
        update_mode = next_update_mode
        terminal && break
    end

    final_path = joinpath(output_directory, "state.h5")
    write_checkpoint(
        final_path;
        settings,
        psi,
        records,
        diagnostic,
        restart_fields=fields,
        chemical_potential,
        provenance,
        immutable=true,
        phase_psis,
    )
    summary_path = write_run_summary_markdown(
        joinpath(output_directory, "run_summary.md"),
        settings,
        diagnostic,
        records,
    )
    diagnostics_path = nothing
    diagnostics_paths = String[]
    if diagnostic.accepted && settings.run.quick_diagnostics
        state_hash = sha256_file(final_path)
        if diagnostic.fundamental_period == 1
            diagnostics = compute_ladder_diagnostics(
                psi,
                settings.model;
                full_pair_correlations=settings.run.full_pair_correlations,
            )
            diagnostics_path = write_diagnostics(
                joinpath(output_directory, "diagnostics.h5"),
                diagnostics;
                state_sha256=state_hash,
                metadata=Dict("solution_kind" => "fixed_point", "phase" => 1, "period" => 1),
                immutable=true,
            )
            push!(diagnostics_paths, diagnostics_path)
        else
            phase_records = records[(end - diagnostic.fundamental_period + 1):end]
            for (phase, phase_record) in enumerate(phase_records)
                haskey(phase_psis, phase_record.iteration) || error(
                    "accepted periodic solution is missing the MPS for iteration $(phase_record.iteration)",
                )
                diagnostics = compute_ladder_diagnostics(
                    phase_psis[phase_record.iteration],
                    settings.model;
                    full_pair_correlations=settings.run.full_pair_correlations,
                )
                path = write_diagnostics(
                    joinpath(output_directory, @sprintf("diagnostics_phase_%03d.h5", phase)),
                    diagnostics;
                    state_sha256=state_hash,
                    metadata=Dict(
                        "solution_kind" => "periodic_orbit",
                        "phase" => phase,
                        "period" => diagnostic.fundamental_period,
                        "iteration" => phase_record.iteration,
                    ),
                    immutable=true,
                )
                push!(diagnostics_paths, path)
            end
        end
    end
    return (
        diagnostic,
        records,
        state_path=final_path,
        summary_path,
        diagnostics_path,
        diagnostics_paths,
        output_directory,
    )
end
