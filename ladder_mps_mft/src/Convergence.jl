function _period_two_oscillation(history::AbstractVector{FieldState}, required::Integer, settings::ConvergenceSettings)
    count = length(history)
    count >= 3 || return (; passes=false, cosine=NaN, two_step_ratio=NaN)
    first_index = max(3, count - Int(required) + 3)
    worst_cosine = -Inf
    worst_ratio = 0.0
    comparisons = 0
    for index in first_index:count
        current = field_vector(history[index])
        previous = field_vector(history[index - 1])
        two_back = field_vector(history[index - 2])
        current_step = current .- previous
        previous_step = previous .- two_back
        denominator = norm(current_step) * norm(previous_step)
        denominator > eps(Float64) || continue
        cosine = dot(current_step, previous_step) / denominator
        ratio = norm(current .- two_back) / max(norm(current_step), eps(Float64))
        worst_cosine = max(worst_cosine, cosine)
        worst_ratio = max(worst_ratio, ratio)
        comparisons += 1
    end
    passes = comparisons > 0 &&
        worst_cosine <= settings.period2_oscillation_cosine_max &&
        worst_ratio <= settings.period2_two_step_ratio_max
    return (;
        passes,
        cosine=comparisons > 0 ? worst_cosine : NaN,
        two_step_ratio=comparisons > 0 ? worst_ratio : NaN,
    )
end

function detect_period(
    history::AbstractVector{FieldState},
    settings::ConvergenceSettings;
    min_period::Integer=1,
    max_period::Integer=settings.max_period,
)
    count = length(history)
    first_period = max(1, Int(min_period))
    last_period = min(settings.max_period, Int(max_period))
    for period in first_period:last_period
        # Check every phase of the orbit across `period_repeats` recurrence
        # links. For p=2 and three links this requires four complete cycles.
        required = period * (settings.period_repeats + 1)
        count >= required || continue
        worst_abs = 0.0
        worst_rel = 0.0
        passes = true
        for phase_offset in 0:(period - 1), repeat in 0:(settings.period_repeats - 1)
            right_index = count - phase_offset - repeat * period
            left_index = right_index - period
            absolute, relative = hybrid_distance(history[right_index], history[left_index])
            worst_abs = max(worst_abs, absolute)
            worst_rel = max(worst_rel, relative)
            if !(absolute <= settings.period_abs_tol || relative <= settings.period_rel_tol)
                passes = false
                break
            end
        end
        oscillation = period == 2 ? _period_two_oscillation(history, required, settings) :
            (; passes=true, cosine=NaN, two_step_ratio=NaN)
        passes && oscillation.passes && return (;
            period,
            absolute=worst_abs,
            relative=worst_rel,
            oscillation_cosine=oscillation.cosine,
            two_step_ratio=oscillation.two_step_ratio,
        )
    end
    return (;
        period=0,
        absolute=Inf,
        relative=Inf,
        oscillation_cosine=NaN,
        two_step_ratio=NaN,
    )
end

_field_pass(record::IterationRecord, settings::ConvergenceSettings) =
    record.field_abs_residual <= settings.field_abs_tol ||
    record.field_rel_residual <= settings.field_rel_tol

function _energy_change(records::AbstractVector{IterationRecord})
    length(records) >= 2 || return Inf
    current = records[end].variational.target_density_corrected_variational_energy
    previous = records[end - 1].variational.target_density_corrected_variational_energy
    sites = 2 * size(records[end].applied.alpha, 1)
    return abs(current - previous) / sites
end

function _slow_mode_diagnostic(records::AbstractVector{IterationRecord}, settings::ConvergenceSettings)
    isempty(records) && return (;
        cosine=NaN,
        contraction=NaN,
        factor=1.0,
        absolute=Inf,
        relative=Inf,
        passes=false,
    )
    current = last(records)
    factor = 1.0
    cosine = NaN
    contraction = NaN
    if length(records) >= 2
        previous = records[end - 1]
        current_residual = field_vector(current.measured) .- field_vector(current.applied)
        previous_residual = field_vector(previous.measured) .- field_vector(previous.applied)
        current_norm = norm(current_residual)
        previous_norm = norm(previous_residual)
        if current_norm > eps(Float64) && previous_norm > eps(Float64)
            cosine = dot(current_residual, previous_residual) / (current_norm * previous_norm)
            contraction = dot(current_residual, previous_residual) / (previous_norm^2)
            if cosine >= settings.slow_mode_cosine_min
                factor = contraction >= 1 ? Inf :
                    max(1.0, 1 / max(1 - contraction, eps(Float64)))
            end
        end
    end
    absolute = current.field_abs_residual * factor
    relative = current.field_rel_residual * factor
    passes = absolute <= settings.field_abs_tol || relative <= settings.field_rel_tol
    return (; cosine, contraction, factor, absolute, relative, passes)
end

function _trailing_records(records::AbstractVector{IterationRecord}, predicate)
    first_index = length(records) + 1
    while first_index > 1 && predicate(records[first_index - 1])
        first_index -= 1
    end
    return records[first_index:end]
end

function _map_closure(records::AbstractVector{IterationRecord})
    length(records) >= 2 || return (; absolute=Inf, relative=Inf)
    worst_abs = 0.0
    worst_rel = 0.0
    for index in 2:length(records)
        absolute, relative = hybrid_distance(records[index].applied, records[index - 1].measured)
        worst_abs = max(worst_abs, absolute)
        worst_rel = max(worst_rel, relative)
    end
    return (; absolute=worst_abs, relative=worst_rel)
end

function _periodic_energy_change(records::AbstractVector{IterationRecord}, period::Integer, repeats::Integer)
    count = length(records)
    required = period * (repeats + 1)
    count >= required || return Inf
    sites = 2 * size(records[end].applied.alpha, 1)
    worst = 0.0
    for phase_offset in 0:(period - 1), repeat in 0:(repeats - 1)
        right_index = count - phase_offset - repeat * period
        left_index = right_index - period
        right_energy = records[right_index].variational.target_density_corrected_variational_energy
        left_energy = records[left_index].variational.target_density_corrected_variational_energy
        worst = max(worst, abs(right_energy - left_energy) / sites)
    end
    return worst
end

function _orbit_density_contrast(
    records::AbstractVector{IterationRecord},
    period::Integer,
    bulk_fraction::Real,
)
    period > 1 || return 0.0
    phases = records[(end - period + 1):end]
    sites_count = length(phases[1].correlations.density_down)
    iseven(sites_count) || throw(DimensionMismatch("ladder orbit density requires an even site count"))
    rungs = div(sites_count, 2)
    kept_rungs = clamp(round(Int, Float64(bulk_fraction) * rungs), 1, rungs)
    first_rung = fld(rungs - kept_rungs, 2) + 1
    last_rung = first_rung + kept_rungs - 1
    bulk_sites = collect(Iterators.flatten((2 * rung - 1, 2 * rung) for rung in first_rung:last_rung))
    contrasts = Float64[]
    for phase in 1:period
        next_phase = phase == period ? 1 : phase + 1
        left = phases[phase].correlations.density_down .+ phases[phase].correlations.density_up
        right = phases[next_phase].correlations.density_down .+ phases[next_phase].correlations.density_up
        push!(contrasts, sum(abs, left[bulk_sites] .- right[bulk_sites]) / length(bulk_sites))
    end
    return mean(contrasts)
end

function _periodic_diagnostic(
    records::AbstractVector{IterationRecord},
    settings::ConvergenceSettings,
    target_density::Real,
    recurrence;
    unmixed_probe::Bool,
    best_iteration::Int,
)
    period = recurrence.period
    phases = records[(end - period + 1):end]
    sites = 2 * size(phases[end].applied.alpha, 1)
    density_error = maximum(abs(record.density - target_density) for record in phases)
    identity_error = maximum(abs(record.variational.hamiltonian_identity_error) / sites for record in phases)
    effective_error = maximum(abs(record.variational.effective_eigenvalue_error) / sites for record in phases)
    energy_change = _periodic_energy_change(records, period, settings.period_repeats)
    energies = [record.variational.canonical_variational_energy for record in phases]
    corrected_energies = [
        record.variational.target_density_corrected_variational_energy for record in phases
    ]
    energy_mean = mean(energies)
    corrected_energy_mean = mean(corrected_energies)
    energy_spread = maximum(energies) - minimum(energies)
    corrected_energy_spread = maximum(corrected_energies) - minimum(corrected_energies)
    density_contrast = _orbit_density_contrast(records, period, settings.orbit_bulk_fraction)
    closure = unmixed_probe ? _map_closure(records) : (; absolute=Inf, relative=Inf)
    closure_pass = unmixed_probe &&
        (closure.absolute <= settings.period_abs_tol || closure.relative <= settings.period_rel_tol)
    period_allowed = period in settings.accepted_periods
    numerical_gates = density_error <= settings.density_tol &&
        energy_change <= settings.variational_energy_tol &&
        identity_error <= settings.hamiltonian_identity_tol &&
        effective_error <= settings.effective_energy_consistency_tol
    accepted = closure_pass && period_allowed && numerical_gates

    reason = if !unmixed_probe
        "mixer-dependent period-$period recurrence detected; an unmixed orbit probe is required"
    elseif !closure_pass
        "period-$period recurrence failed the raw-map closure gate"
    elseif !period_allowed
        "validated period-$period orbit is not in accepted_periods=$(settings.accepted_periods)"
    elseif !numerical_gates
        "period-$period orbit failed density, phase-energy recurrence, or Hamiltonian-consistency gates"
    else
        "validated unmixed period-$period mean-field solution; every phase and raw-map link passed"
    end

    return ConvergenceDiagnostic(;
        status=accepted ? :periodic_solution : :periodic_candidate,
        accepted,
        reason,
        solution_kind=:periodic_orbit,
        fundamental_period=period,
        orbit_validated=closure_pass && numerical_gates,
        unmixed_probe,
        solution_canonical_variational_energy=energy_mean,
        solution_target_density_corrected_variational_energy=corrected_energy_mean,
        orbit_energy_spread=energy_spread,
        orbit_target_density_corrected_energy_spread=corrected_energy_spread,
        orbit_density_contrast=density_contrast,
        fixed_point_abs_residual=last(phases).field_abs_residual,
        fixed_point_rel_residual=last(phases).field_rel_residual,
        cycle_abs_residual=max(recurrence.absolute, closure.absolute),
        cycle_rel_residual=max(recurrence.relative, closure.relative),
        cycle_oscillation_cosine=recurrence.oscillation_cosine,
        cycle_two_step_ratio=recurrence.two_step_ratio,
        density_error,
        variational_energy_change=energy_change,
        hamiltonian_identity_error_per_site=identity_error,
        effective_eigenvalue_error_per_site=effective_error,
        best_iteration,
    )
end

function assess_convergence(
    records::AbstractVector{IterationRecord},
    settings::ConvergenceSettings,
    target_density::Real,
)
    isempty(records) && return ConvergenceDiagnostic()
    current = last(records)
    density_error = abs(current.density - target_density)
    energy_change = _energy_change(records)
    sites = 2 * size(current.applied.alpha, 1)
    identity_error = abs(current.variational.hamiltonian_identity_error) / sites
    effective_error = abs(current.variational.effective_eigenvalue_error) / sites
    slow_mode = _slow_mode_diagnostic(records, settings)
    best_iteration = findmin(record.field_rel_residual for record in records)[2]
    stable_count = min(settings.stable_iterations, length(records))
    recent = records[(end - stable_count + 1):end]
    fixed = length(records) >= settings.stable_iterations &&
        all(record -> _field_pass(record, settings), recent) &&
        slow_mode.passes &&
        all(record -> abs(record.density - target_density) <= settings.density_tol, recent) &&
        energy_change <= settings.variational_energy_tol &&
        identity_error <= settings.hamiltonian_identity_tol &&
        effective_error <= settings.effective_energy_consistency_tol
    if fixed
        return ConvergenceDiagnostic(;
            status=:fixed_point,
            accepted=true,
            reason="fixed-point raw and slow-mode-extrapolated residual, density, target-density-corrected energy, and Hamiltonian-consistency gates passed",
            solution_kind=:fixed_point,
            fundamental_period=1,
            orbit_validated=false,
            unmixed_probe=current.update_mode == :unmixed_probe,
            solution_canonical_variational_energy=current.variational.canonical_variational_energy,
            solution_target_density_corrected_variational_energy=
                current.variational.target_density_corrected_variational_energy,
            orbit_energy_spread=0.0,
            orbit_target_density_corrected_energy_spread=0.0,
            orbit_density_contrast=0.0,
            fixed_point_abs_residual=current.field_abs_residual,
            fixed_point_rel_residual=current.field_rel_residual,
            fixed_point_residual_cosine=slow_mode.cosine,
            fixed_point_contraction_estimate=slow_mode.contraction,
            fixed_point_extrapolation_factor=slow_mode.factor,
            fixed_point_extrapolated_abs_residual=slow_mode.absolute,
            fixed_point_extrapolated_rel_residual=slow_mode.relative,
            cycle_abs_residual=current.field_abs_residual,
            cycle_rel_residual=current.field_rel_residual,
            density_error,
            variational_energy_change=energy_change,
            hamiltonian_identity_error_per_site=identity_error,
            effective_eigenvalue_error_per_site=effective_error,
            best_iteration,
        )
    end

    probe_records = _trailing_records(records, record -> record.update_mode == :unmixed_probe)
    probe_recurrence = detect_period(
        [record.measured for record in probe_records],
        settings;
        min_period=2,
        max_period=settings.probe_max_period,
    )
    if probe_recurrence.period > 1
        return _periodic_diagnostic(
            probe_records,
            settings,
            target_density,
            probe_recurrence;
            unmixed_probe=true,
            best_iteration,
        )
    end

    accelerated_records = _trailing_records(
        records,
        record -> record.update_mode in (:linear, :anderson, :linear_fallback),
    )
    recurrence = detect_period(
        [record.measured for record in accelerated_records],
        settings;
        min_period=2,
    )
    if recurrence.period > 1
        return _periodic_diagnostic(
            accelerated_records,
            settings,
            target_density,
            recurrence;
            unmixed_probe=false,
            best_iteration,
        )
    end

    status_records = isempty(accelerated_records) ? records : accelerated_records
    residuals = [record.field_rel_residual for record in status_records]
    finite_residuals = filter(isfinite, residuals)
    if !isfinite(current.field_rel_residual)
        status, reason = :nonfinite, "nonfinite field residual"
    elseif current.update_mode == :unmixed_probe
        status, reason = :iterating, "unmixed cycle probe in progress"
    elseif length(finite_residuals) >= 3 && current.field_rel_residual > settings.divergence_factor * minimum(finite_residuals)
        status, reason = :diverging, "field residual exceeded the best value by the configured divergence factor"
    elseif length(status_records) >= settings.stagnation_window
        window = residuals[(end - settings.stagnation_window + 1):end]
        improvement = (first(window) - minimum(window)) / max(abs(first(window)), eps(Float64))
        if improvement < settings.stagnation_min_relative_improvement
            status, reason = :stagnated, "field residual failed the configured windowed-improvement gate"
        else
            status, reason = :iterating, "not yet converged"
        end
    else
        status, reason = :iterating, "not yet converged"
    end
    return ConvergenceDiagnostic(;
        status,
        accepted=false,
        reason,
        solution_kind=:none,
        fundamental_period=0,
        orbit_validated=false,
        unmixed_probe=current.update_mode == :unmixed_probe,
        solution_canonical_variational_energy=NaN,
        solution_target_density_corrected_variational_energy=NaN,
        orbit_energy_spread=NaN,
        orbit_target_density_corrected_energy_spread=NaN,
        orbit_density_contrast=NaN,
        fixed_point_abs_residual=current.field_abs_residual,
        fixed_point_rel_residual=current.field_rel_residual,
        fixed_point_residual_cosine=slow_mode.cosine,
        fixed_point_contraction_estimate=slow_mode.contraction,
        fixed_point_extrapolation_factor=slow_mode.factor,
        fixed_point_extrapolated_abs_residual=slow_mode.absolute,
        fixed_point_extrapolated_rel_residual=slow_mode.relative,
        cycle_abs_residual=Inf,
        cycle_rel_residual=Inf,
        density_error,
        variational_energy_change=energy_change,
        hamiltonian_identity_error_per_site=identity_error,
        effective_eigenvalue_error_per_site=effective_error,
        best_iteration,
    )
end
