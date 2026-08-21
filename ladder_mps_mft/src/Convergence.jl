function detect_period(history::AbstractVector{FieldState}, settings::ConvergenceSettings)
    count = length(history)
    for period in 1:settings.max_period
        required = period * settings.period_repeats + 1
        count >= required || continue
        worst_abs = 0.0
        worst_rel = 0.0
        passes = true
        for repeat in 0:(settings.period_repeats - 1)
            right_index = count - repeat * period
            left_index = right_index - period
            absolute, relative = hybrid_distance(history[right_index], history[left_index])
            worst_abs = max(worst_abs, absolute)
            worst_rel = max(worst_rel, relative)
            if !(absolute <= settings.period_abs_tol || relative <= settings.period_rel_tol)
                passes = false
                break
            end
        end
        passes && return (; period, absolute=worst_abs, relative=worst_rel)
    end
    return (; period=0, absolute=Inf, relative=Inf)
end

_field_pass(record::IterationRecord, settings::ConvergenceSettings) =
    record.field_abs_residual <= settings.field_abs_tol ||
    record.field_rel_residual <= settings.field_rel_tol

function _energy_change(records::AbstractVector{IterationRecord})
    length(records) >= 2 || return Inf
    current = records[end].variational.canonical_variational_energy
    previous = records[end - 1].variational.canonical_variational_energy
    sites = 2 * size(records[end].applied.alpha, 1)
    return abs(current - previous) / sites
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
    best_iteration = findmin(record.field_rel_residual for record in records)[2]
    stable_count = min(settings.stable_iterations, length(records))
    recent = records[(end - stable_count + 1):end]
    fixed = length(records) >= settings.stable_iterations &&
        all(record -> _field_pass(record, settings), recent) &&
        all(record -> abs(record.density - target_density) <= settings.density_tol, recent) &&
        energy_change <= settings.variational_energy_tol &&
        identity_error <= settings.hamiltonian_identity_tol &&
        effective_error <= settings.effective_energy_consistency_tol
    if fixed
        return ConvergenceDiagnostic(;
            status=:fixed_point,
            accepted=true,
            reason="fixed-point, density, variational-energy, and Hamiltonian-consistency gates passed",
            fundamental_period=1,
            fixed_point_abs_residual=current.field_abs_residual,
            fixed_point_rel_residual=current.field_rel_residual,
            cycle_abs_residual=current.field_abs_residual,
            cycle_rel_residual=current.field_rel_residual,
            density_error,
            variational_energy_change=energy_change,
            hamiltonian_identity_error_per_site=identity_error,
            effective_eigenvalue_error_per_site=effective_error,
            best_iteration,
        )
    end

    measured_history = [record.measured for record in records]
    recurrence = detect_period(measured_history, settings)
    if recurrence.period > 1
        return ConvergenceDiagnostic(;
            status=:periodic_cycle,
            accepted=false,
            reason="stable period-$(recurrence.period) SCF cycle detected; saved as a diagnostic, not a fixed point",
            fundamental_period=recurrence.period,
            fixed_point_abs_residual=current.field_abs_residual,
            fixed_point_rel_residual=current.field_rel_residual,
            cycle_abs_residual=recurrence.absolute,
            cycle_rel_residual=recurrence.relative,
            density_error,
            variational_energy_change=energy_change,
            hamiltonian_identity_error_per_site=identity_error,
            effective_eigenvalue_error_per_site=effective_error,
            best_iteration,
        )
    end

    residuals = [record.field_rel_residual for record in records]
    finite_residuals = filter(isfinite, residuals)
    if !isfinite(current.field_rel_residual)
        status, reason = :nonfinite, "nonfinite field residual"
    elseif length(finite_residuals) >= 3 && current.field_rel_residual > settings.divergence_factor * minimum(finite_residuals)
        status, reason = :diverging, "field residual exceeded the best value by the configured divergence factor"
    elseif length(records) >= settings.stagnation_window
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
        status=status,
        accepted=false,
        reason=reason,
        fundamental_period=recurrence.period,
        fixed_point_abs_residual=current.field_abs_residual,
        fixed_point_rel_residual=current.field_rel_residual,
        cycle_abs_residual=recurrence.absolute,
        cycle_rel_residual=recurrence.relative,
        density_error=density_error,
        variational_energy_change=energy_change,
        hamiltonian_identity_error_per_site=identity_error,
        effective_eigenvalue_error_per_site=effective_error,
        best_iteration=best_iteration,
    )
end
