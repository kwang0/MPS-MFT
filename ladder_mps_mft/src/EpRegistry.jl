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
    atol::Real=1e-10,
)
    matches = filter(records) do record
        record.L == L &&
            _parameter_match(record.U, U; atol) &&
            _parameter_match(record.V, V; atol) &&
            _parameter_match(record.t0, t0; atol) &&
            _parameter_match(record.density, density; atol)
    end
    isempty(matches) && throw(ArgumentError(
        "no exact E_p entry for L=$L U=$U V=$V t0=$t0 density=$density; " *
        "interpolation is deliberately disabled",
    ))
    sort!(matches; by=record -> (-record.chi, record.rel_diff))
    selected = first(matches)
    bound_pair = selected.E_p < 0
    require_bound && !bound_pair && throw(ArgumentError(
        "E_p=$(selected.E_p) does not describe a bound pair; refusing to use |E_p| as a perturbative denominator",
    ))
    denominator = abs(selected.E_p)
    denominator > 0 || throw(ArgumentError("E_p is zero for the selected registry entry"))
    return EpSelection(;
        record=selected,
        denominator=denominator,
        source_path=abspath(source_path),
        bound_pair=bound_pair,
        tp_below_pair_binding=Float64(tp) < denominator,
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
