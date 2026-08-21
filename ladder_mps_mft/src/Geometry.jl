function normalize_geometry(raw)::Symbol
    value = Symbol(replace(lowercase(strip(String(raw))), "-" => "_", " " => "_"))
    value in SUPPORTED_GEOMETRIES || throw(ArgumentError(
        "unknown transverse geometry '$raw'; expected one of $(join(string.(SUPPORTED_GEOMETRIES), ", "))",
    ))
    return value
end

rung_leg_to_site(rung::Integer, leg::Integer) = 2 * (Int(rung) - 1) + Int(leg) + 1

function site_to_rung_leg(site::Integer)
    site >= 1 || throw(ArgumentError("site index must be positive"))
    return (div(Int(site) - 1, 2) + 1, mod(Int(site) - 1, 2))
end

"""
Return the two-leg density kernel K such that mu = K * (n - 1/2).
The kernel is for one rung and one spin species.
"""
function density_kernel(geometry, tp::Real, ep::Real)
    geom = normalize_geometry(geometry)
    ep > 0 || throw(ArgumentError("the perturbative E_p denominator must be positive"))
    g = Float64(tp)^2 / Float64(ep)
    if geom == :cubic_frustrated
        return 2g .* [2.0 1.0; 1.0 2.0]
    elseif geom == :cubic_unfrustrated
        return 6g .* [0.0 1.0; 1.0 0.0]
    else
        return 2g .* [0.0 1.0; 1.0 0.0]
    end
end

function transverse_coordination(geometry)::Int
    geom = normalize_geometry(geometry)
    return geom == :square ? 2 : 6
end
