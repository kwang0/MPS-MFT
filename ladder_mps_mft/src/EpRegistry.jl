const DEFAULT_EP_REGISTRY = normpath(joinpath(@__DIR__, "..", "data", "E_p_values.csv"))

function load_ep_registry(path::AbstractString=DEFAULT_EP_REGISTRY)
    isfile(path) || throw(ArgumentError("E_p registry not found: $path"))
    records = EpRecord[]
    for row in CSV.File(path)
        push!(records, EpRecord(
            Int(row.L),
            Float64(row.U),
            Float64(row.V),
            Float64(row.t0),
            Float64(row.density),
            Int(row.chi),
            Float64(row.E_N),
            Float64(row.E_p),
            Float64(row.rel_diff),
        ))
    end
    isempty(records) && throw(ArgumentError("E_p registry is empty: $path"))
    return records
end

_parameter_match(left::Real, right::Real; atol::Real=1e-10) = isapprox(left, right; atol, rtol=0)

function _best_ep_record(records::AbstractVector{EpRecord})
    isempty(records) && throw(ArgumentError("cannot select from an empty E_p record set"))
    ordered = sort(collect(records); by=record -> (-record.chi, record.rel_diff))
    return first(ordered)
end

function _validate_ep_record(record::EpRecord, require_bound::Bool)
    require_bound && record.E_p >= 0 && throw(ArgumentError(
        "E_p=$(record.E_p) at t0=$(record.t0) does not describe a bound pair; " *
        "refusing to use |E_p| as a perturbative denominator",
    ))
    iszero(record.E_p) && throw(ArgumentError("E_p is zero at t0=$(record.t0)"))
    return record
end

function lookup_ep(
    records::AbstractVector{EpRecord};
    L::Integer,
    U::Real,
    V::Real,
    t0::Real,
    density::Real,
    tp::Real,
    source_path::AbstractString=DEFAULT_EP_REGISTRY,
    require_bound::Bool=true,
    allow_interpolation::Bool=false,
    atol::Real=1e-10,
)
    matches = filter(records) do record
        record.L == L &&
            _parameter_match(record.U, U; atol) &&
            _parameter_match(record.V, V; atol) &&
            _parameter_match(record.t0, t0; atol) &&
            _parameter_match(record.density, density; atol)
    end
    if !isempty(matches)
        selected = _validate_ep_record(_best_ep_record(matches), require_bound)
        denominator = abs(selected.E_p)
        return EpSelection(;
            record=selected,
            denominator,
            source_path=abspath(source_path),
            bound_pair=selected.E_p < 0,
            tp_below_pair_binding=Float64(tp) < denominator,
            mode=:exact,
            lower_record=selected,
            upper_record=selected,
            interpolation_weight=0.0,
        )
    end

    allow_interpolation || throw(ArgumentError(
        "no exact E_p entry for L=$L U=$U V=$V t0=$t0 density=$density; " *
        "set pair_binding.allow_interpolation=true to interpolate only within a bracket in t0",
    ))
    family = filter(records) do record
        record.L == L &&
            _parameter_match(record.U, U; atol) &&
            _parameter_match(record.V, V; atol) &&
            _parameter_match(record.density, density; atol)
    end
    isempty(family) && throw(ArgumentError(
        "no E_p family for L=$L U=$U V=$V density=$density; cannot interpolate",
    ))
    best_by_t0 = Dict{Float64,EpRecord}()
    for abscissa in unique(record.t0 for record in family)
        at_abscissa = filter(record -> _parameter_match(record.t0, abscissa; atol), family)
        best_by_t0[abscissa] = _best_ep_record(at_abscissa)
    end
    lower_values = filter(value -> value < Float64(t0) && !_parameter_match(value, t0; atol), collect(keys(best_by_t0)))
    upper_values = filter(value -> value > Float64(t0) && !_parameter_match(value, t0; atol), collect(keys(best_by_t0)))
    (isempty(lower_values) || isempty(upper_values)) && throw(ArgumentError(
        "t0=$t0 is not bracketed by E_p data for L=$L U=$U V=$V density=$density; extrapolation is forbidden",
    ))
    lower = _validate_ep_record(best_by_t0[maximum(lower_values)], require_bound)
    upper = _validate_ep_record(best_by_t0[minimum(upper_values)], require_bound)
    signbit(lower.E_p) == signbit(upper.E_p) || throw(ArgumentError(
        "E_p changes sign between t0=$(lower.t0) and t0=$(upper.t0); interpolation is not physically controlled",
    ))
    weight = (Float64(t0) - lower.t0) / (upper.t0 - lower.t0)
    ep_signed = muladd(weight, upper.E_p - lower.E_p, lower.E_p)
    iszero(ep_signed) && throw(ArgumentError("interpolated E_p is zero at t0=$t0"))
    require_bound && ep_signed >= 0 && throw(ArgumentError("interpolated E_p=$ep_signed is not bound"))
    selected = EpRecord(
        Int(L), Float64(U), Float64(V), Float64(t0), Float64(density),
        min(lower.chi, upper.chi),
        muladd(weight, upper.E_N - lower.E_N, lower.E_N),
        ep_signed,
        max(lower.rel_diff, upper.rel_diff),
    )
    denominator = abs(ep_signed)
    return EpSelection(;
        record=selected,
        denominator,
        source_path=abspath(source_path),
        bound_pair=ep_signed < 0,
        tp_below_pair_binding=Float64(tp) < denominator,
        mode=:linear_t0,
        lower_record=lower,
        upper_record=upper,
        interpolation_weight=weight,
    )
end

function lookup_ep(path::AbstractString=DEFAULT_EP_REGISTRY; kwargs...)
    return lookup_ep(load_ep_registry(path); source_path=path, kwargs...)
end

function validate_weak_coupling(selection::EpSelection, tp::Real; spin_gap=nothing, charge_gap=nothing)
    pair_ok = Float64(tp) < selection.denominator
    spin_ok = spin_gap === nothing ? missing : Float64(tp) < Float64(spin_gap)
    charge_ok = charge_gap === nothing ? missing : Float64(tp) < Float64(charge_gap)
    known = [value for value in (pair_ok, spin_ok, charge_ok) if !ismissing(value)]
    return (
        pair_binding_ok=pair_ok,
        spin_gap_ok=spin_ok,
        charge_gap_ok=charge_ok,
        all_known_scales_ok=all(known),
        fully_checked=spin_gap !== nothing && charge_gap !== nothing,
    )
end
