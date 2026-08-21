mutable struct MixingState
    x_history::Vector{Vector{Float64}}
    f_history::Vector{Vector{Float64}}
    residual_norms::Vector{Float64}
    damping::Float64
end

MixingState(settings::MixingSettings) = MixingState(Vector{Float64}[], Vector{Float64}[], Float64[], settings.damping)

function field_vector(fields::FieldState)
    return vcat(vec(fields.alpha), vec(fields.beta), vec(fields.mu_cdw))
end

function fields_from_vector(template::FieldState, values::AbstractVector{<:Real})
    na = length(template.alpha)
    nb = length(template.beta)
    nm = length(template.mu_cdw)
    length(values) == na + nb + nm || throw(DimensionMismatch("field vector has the wrong length"))
    cursor = 1
    alpha = reshape(Float64.(values[cursor:cursor + na - 1]), size(template.alpha)); cursor += na
    beta = reshape(Float64.(values[cursor:cursor + nb - 1]), size(template.beta)); cursor += nb
    mu_cdw = reshape(Float64.(values[cursor:cursor + nm - 1]), size(template.mu_cdw))
    return FieldState(alpha, beta, mu_cdw)
end

function hybrid_distance(left::FieldState, right::FieldState)
    lv = field_vector(left)
    rv = field_vector(right)
    delta = lv .- rv
    absolute = isempty(delta) ? 0.0 : maximum(abs, delta)
    relative = norm(delta) / max(norm(lv), norm(rv), eps(Float64))
    return absolute, relative
end

function _adapt_damping!(state::MixingState, settings::MixingSettings, residual_norm::Float64)
    if settings.adaptive && !isempty(state.residual_norms)
        previous = last(state.residual_norms)
        if residual_norm > 1.25 * previous
            state.damping = max(settings.minimum_damping, 0.5 * state.damping)
        elseif residual_norm < 0.8 * previous
            state.damping = min(settings.maximum_damping, 1.1 * state.damping)
        end
    end
    push!(state.residual_norms, residual_norm)
    return state.damping
end

function _anderson_coefficients(residuals::AbstractMatrix, regularization::Real)
    count = size(residuals, 2)
    gram = residuals' * residuals + Float64(regularization) * I
    kkt = [gram ones(count); ones(count)' 0.0]
    rhs = vcat(zeros(count), 1.0)
    return (kkt \ rhs)[1:count]
end

function mix_fields!(state::MixingState, applied::FieldState, measured::FieldState, settings::MixingSettings)
    x = field_vector(applied)
    f = field_vector(measured)
    residual = f .- x
    damping = _adapt_damping!(state, settings, norm(residual))
    push!(state.x_history, x)
    push!(state.f_history, f)
    keep = settings.memory + 1
    length(state.x_history) > keep && popfirst!(state.x_history)
    length(state.f_history) > keep && popfirst!(state.f_history)

    method_used = settings.method
    next_values = if settings.method == :linear || length(state.x_history) < 2
        method_used = :linear
        x .+ damping .* residual
    else
        residuals = hcat((state.f_history[i] .- state.x_history[i] for i in eachindex(state.x_history))...)
        try
            coefficients = _anderson_coefficients(residuals, settings.regularization)
            mixed_x = sum(coefficients[i] .* state.x_history[i] for i in eachindex(coefficients))
            mixed_f = sum(coefficients[i] .* state.f_history[i] for i in eachindex(coefficients))
            (1 - damping) .* mixed_x .+ damping .* mixed_f
        catch err
            @warn "Anderson solve failed; falling back to linear mixing" exception=(err, catch_backtrace())
            method_used = :linear_fallback
            x .+ damping .* residual
        end
    end
    return fields_from_vector(applied, next_values), (; method=method_used, damping, residual_norm=norm(residual))
end
