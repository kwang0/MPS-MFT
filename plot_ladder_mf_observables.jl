using HDF5
import PyPlot
import PyCall

const _MatplotlibWidgets = PyCall.pyimport("matplotlib.widgets")

# Usage:
#   include("plot_ladder_mf_observables.jl")
#   plot_mf_profiles_from_file("stateless_data/results_...h5")
#   plot_middle_histories_from_file("stateless_data/results_...h5")
#   plot_mf_profiles_from_file("stateless_data/results_...h5"; source=:correlations)
#   plot_middle_histories_from_file("stateless_data/results_...h5"; source=:correlations)
#   plot_order_fourier_heatmaps_from_file("stateless_data/results_...h5"; source=:correlations)
#   plot_mf_change_slider_from_file("stateless_data/results_...h5"; field=:alpha)
#   plot_mf_change_slider_from_file("stateless_data/results_...h5"; field=:beta, spin=:up)
# The file-based wrappers put the HDF5 basename in the PyPlot figure title.
#
# Array convention:
#   alpha[i, ip, leg, legp] and alpha_list[i, ip, leg, legp, iter]
#   beta[spin, i, ip, leg, legp] and beta_list[spin, i, ip, leg, legp, iter]
#   spin=1 is down, spin=2 is up. leg=1/2 are Julia array leg indices.
#   C_pair_list[site_up, site_dn, iter] = <c_up c_dn>, C_exc_*_list[site_from, site_to, iter]
#   use the ladder site map site = 2 * (rung - 1) + leg for leg=1/2.

function load_mf_data(filename::AbstractString;
    alpha_key::AbstractString="alpha",
    beta_key::AbstractString="beta",
    alpha_list_key::AbstractString="alpha_list",
    beta_list_key::AbstractString="beta_list",
    C_pair_key::AbstractString="C_pair",
    C_exc_dn_key::AbstractString="C_exc_dn",
    C_exc_up_key::AbstractString="C_exc_up",
    C_pair_list_key::AbstractString="C_pair_list",
    C_exc_dn_list_key::AbstractString="C_exc_dn_list",
    C_exc_up_list_key::AbstractString="C_exc_up_list")

    h5open(filename, "r") do f
        names = Set(String.(keys(f)))
        return (
            alpha = alpha_key in names ? read(f, alpha_key) : nothing,
            beta = beta_key in names ? read(f, beta_key) : nothing,
            alpha_list = alpha_list_key in names ? read(f, alpha_list_key) : nothing,
            beta_list = beta_list_key in names ? read(f, beta_list_key) : nothing,
            C_pair = C_pair_key in names ? read(f, C_pair_key) : nothing,
            C_exc_dn = C_exc_dn_key in names ? read(f, C_exc_dn_key) : nothing,
            C_exc_up = C_exc_up_key in names ? read(f, C_exc_up_key) : nothing,
            C_pair_list = C_pair_list_key in names ? read(f, C_pair_list_key) : nothing,
            C_exc_dn_list = C_exc_dn_list_key in names ? read(f, C_exc_dn_list_key) : nothing,
            C_exc_up_list = C_exc_up_list_key in names ? read(f, C_exc_up_list_key) : nothing,
        )
    end
end

middle_rung(L::Integer) = cld(L, 2)
_plot_rung_leg_to_site(i::Integer, leg::Integer) = 2 * (i - 1) + leg

function _spin_index(spin)
    if spin isa Integer
        spin in (1, 2) || throw(ArgumentError("spin must be 1/:down or 2/:up"))
        return Int(spin)
    end

    s = spin isa Symbol ? String(spin) : lowercase(String(spin))
    s = lowercase(s)
    s in ("down", "dn", "d") && return 1
    s in ("up", "u") && return 2
    throw(ArgumentError("spin must be 1/:down or 2/:up"))
end

_spin_label(spin) = _spin_index(spin) == 1 ? "down" : "up"

function _check_leg(leg::Integer)
    leg in (1, 2) || throw(ArgumentError("leg must be 1 or 2"))
    return Int(leg)
end

function _check_index(i::Integer, n::Integer, name::AbstractString)
    1 <= i <= n || throw(ArgumentError("$name must be between 1 and $n; got $i"))
    return Int(i)
end

function _final_array(A, final_ndims::Int, iteration, name::AbstractString)
    A === nothing && throw(ArgumentError("$name was not found"))

    if ndims(A) == final_ndims
        iteration === nothing || throw(ArgumentError("$name has no MF-iteration dimension"))
        return A
    elseif ndims(A) == final_ndims + 1
        niter = size(A, final_ndims + 1)
        it = iteration === nothing ? niter : _check_index(iteration, niter, "iteration")
        return selectdim(A, final_ndims + 1, it)
    end

    throw(ArgumentError("$name must have $final_ndims dimensions, or $(final_ndims + 1) with MF iteration last; got $(ndims(A))"))
end

function _history_array(A, history_ndims::Int, name::AbstractString)
    A === nothing && throw(ArgumentError("$name was not found"))
    ndims(A) == history_ndims || throw(ArgumentError("$name must have $history_ndims dimensions with MF iteration last; got $(ndims(A))"))
    return A
end

_plot_values(vals; use_abs::Bool=false) = use_abs ? abs.(vals) : real.(vals)

function _source(source::Symbol, use_correlations::Bool)
    use_correlations && return :correlations
    source in (:mf, :alpha_beta) && return :mf
    source in (:correlations, :correlation, :raw, :corr) && return :correlations
    throw(ArgumentError("source must be :mf or :correlations"))
end

function _normal_corr(C_exc_dn, C_exc_up, spin, iteration)
    sigma = _spin_index(spin)
    if sigma == 1
        return _final_array(C_exc_dn, 2, iteration, "C_exc_dn_list")
    end
    return _final_array(C_exc_up, 2, iteration, "C_exc_up_list")
end

function _normal_corr_history(C_exc_dn_list, C_exc_up_list, spin)
    sigma = _spin_index(spin)
    if sigma == 1
        return _history_array(C_exc_dn_list, 3, "C_exc_dn_list")
    end
    return _history_array(C_exc_up_list, 3, "C_exc_up_list")
end

function _rung_count_from_sites(nsites::Integer)
    iseven(nsites) || throw(ArgumentError("raw correlation matrices must have an even number of sites; got $nsites"))
    return div(nsites, 2)
end

_math(expr::AbstractString) = "\$" * expr * "\$"
_spin_tex(spin) = _spin_index(spin) == 1 ? "\\downarrow" : "\\uparrow"

function _corr_value_label(expr::AbstractString, use_abs::Bool)
    if use_abs
        return _math("\\left|" * expr * "\\right|")
    end
    return _math("\\mathrm{Re}\\left[" * expr * "\\right]")
end

function _density_corr_expr(spin; rung_tex::AbstractString="i", leg_tex::AbstractString="\\ell")
    sigma = _spin_tex(spin)
    return "\\langle c^\\dagger_{" * rung_tex * "," * leg_tex * "," * sigma * "} c_{" * rung_tex * "," * leg_tex * "," * sigma * "} \\rangle"
end

function _density_corr_title(spin, leg::Integer; rung_tex::AbstractString="i")
    sigma = _spin_tex(spin)
    leg_tex = string(leg)
    expr = _density_corr_expr(spin; rung_tex=rung_tex, leg_tex=leg_tex)
    return _math("C^{" * sigma * "}_{\\mathrm{exc}}[(" * rung_tex * "," * leg_tex * "),(" * rung_tex * "," * leg_tex * ")] = " * expr)
end

function _cdw_corr_expr(; rung_tex::AbstractString="i", leg_tex::AbstractString="\\ell")
    return "\\langle n_{" * rung_tex * "," * leg_tex * ",\\uparrow} + n_{" * rung_tex * "," * leg_tex * ",\\downarrow} \\rangle"
end

function _sdw_corr_expr(; rung_tex::AbstractString="i", leg_tex::AbstractString="\\ell")
    return "\\langle n_{" * rung_tex * "," * leg_tex * ",\\uparrow} - n_{" * rung_tex * "," * leg_tex * ",\\downarrow} \\rangle"
end

function _cdw_corr_title(leg::Integer; rung_tex::AbstractString="i")
    leg_tex = string(leg)
    lhs = "C^{\\uparrow}_{\\mathrm{exc}}[(" * rung_tex * "," * leg_tex * "),(" * rung_tex * "," * leg_tex * ")] + C^{\\downarrow}_{\\mathrm{exc}}[(" * rung_tex * "," * leg_tex * "),(" * rung_tex * "," * leg_tex * ")]"
    return _math(lhs * " = " * _cdw_corr_expr(; rung_tex=rung_tex, leg_tex=leg_tex))
end

function _sdw_corr_title(leg::Integer; rung_tex::AbstractString="i")
    leg_tex = string(leg)
    lhs = "C^{\\uparrow}_{\\mathrm{exc}}[(" * rung_tex * "," * leg_tex * "),(" * rung_tex * "," * leg_tex * ")] - C^{\\downarrow}_{\\mathrm{exc}}[(" * rung_tex * "," * leg_tex * "),(" * rung_tex * "," * leg_tex * ")]"
    return _math(lhs * " = " * _sdw_corr_expr(; rung_tex=rung_tex, leg_tex=leg_tex))
end

function _onsite_pair_corr_expr(; rung_tex::AbstractString="i", leg_tex::AbstractString="\\ell")
    return "\\langle c_{" * rung_tex * "," * leg_tex * ",\\uparrow} c_{" * rung_tex * "," * leg_tex * ",\\downarrow} \\rangle"
end

function _onsite_pair_corr_title(leg::Integer; rung_tex::AbstractString="i")
    leg_tex = string(leg)
    expr = _onsite_pair_corr_expr(; rung_tex=rung_tex, leg_tex=leg_tex)
    return _math("C_{\\mathrm{pair}}[(" * rung_tex * "," * leg_tex * "),(" * rung_tex * "," * leg_tex * ")] = " * expr)
end

function _rung_pair_corr_expr(; rung_tex::AbstractString="i", leg1_tex::AbstractString="\\ell_1", leg2_tex::AbstractString="\\ell_2", symmetrize::Bool=false)
    expr12 = "\\langle c_{" * rung_tex * "," * leg2_tex * ",\\uparrow} c_{" * rung_tex * "," * leg1_tex * ",\\downarrow} \\rangle"
    if !symmetrize
        return expr12
    end
    expr21 = "\\langle c_{" * rung_tex * "," * leg1_tex * ",\\uparrow} c_{" * rung_tex * "," * leg2_tex * ",\\downarrow} \\rangle"
    return "\\frac{1}{2}\\left(" * expr12 * " + " * expr21 * "\\right)"
end

function _rung_pair_corr_title(leg1::Integer, leg2::Integer; rung_tex::AbstractString="i", symmetrize::Bool=false)
    leg1_tex = string(leg1)
    leg2_tex = string(leg2)
    if symmetrize
        lhs = "\\frac{1}{2}\\left(C_{\\mathrm{pair}}[(" * rung_tex * "," * leg2_tex * "),(" * rung_tex * "," * leg1_tex * ")] + C_{\\mathrm{pair}}[(" * rung_tex * "," * leg1_tex * "),(" * rung_tex * "," * leg2_tex * ")]\\right)"
    else
        lhs = "C_{\\mathrm{pair}}[(" * rung_tex * "," * leg2_tex * "),(" * rung_tex * "," * leg1_tex * ")]"
    end
    expr = _rung_pair_corr_expr(; rung_tex=rung_tex, leg1_tex=leg1_tex, leg2_tex=leg2_tex, symmetrize=symmetrize)
    return _math(lhs * " = " * expr)
end

_site_tex(rung_tex::AbstractString, leg_tex::AbstractString) = "(" * rung_tex * "," * leg_tex * ")"
_next_rung_tex(rung_tex::AbstractString) = all(isdigit, rung_tex) ? string(parse(Int, rung_tex) + 1) : rung_tex * "+1"
_singlet_pair_corr_def() = "\\Delta^s_{a,b}=\\frac{1}{2}\\left(\\langle c_{a,\\uparrow}c_{b,\\downarrow}\\rangle-\\langle c_{a,\\downarrow}c_{b,\\uparrow}\\rangle\\right)"
_dwave_corr_symbol(rung_tex::AbstractString="i") = "\\Delta^d_{" * rung_tex * "}"

function _dwave_corr_expr(; rung_tex::AbstractString="i", leg1_tex::AbstractString="1", leg2_tex::AbstractString="2")
    next_tex = _next_rung_tex(rung_tex)
    site_i1 = _site_tex(rung_tex, leg1_tex)
    site_i2 = _site_tex(rung_tex, leg2_tex)
    site_ip1 = _site_tex(next_tex, leg1_tex)
    site_ip2 = _site_tex(next_tex, leg2_tex)
    return _dwave_corr_symbol(rung_tex) * "=\\frac{1}{2}\\left(\\Delta^s_{" * site_i1 * "," * site_ip1 * "}+\\Delta^s_{" * site_i2 * "," * site_ip2 * "}\\right)-\\Delta^s_{" * site_i1 * "," * site_i2 * "}"
end

function _dwave_corr_title(leg1::Integer, leg2::Integer; rung_tex::AbstractString="i")
    return _math(_singlet_pair_corr_def()) * "\n" * _math(_dwave_corr_expr(; rung_tex=rung_tex, leg1_tex=string(leg1), leg2_tex=string(leg2)))
end

_dwave_alpha_symbol(rung_tex::AbstractString="i") = "\\Delta^d_{\\alpha," * rung_tex * "}"

function _dwave_alpha_expr(; rung_tex::AbstractString="i", leg1_tex::AbstractString="1", leg2_tex::AbstractString="2", symmetrize::Bool=false)
    next_tex = _next_rung_tex(rung_tex)
    leg_term = "\\frac{1}{2}\\left(\\alpha_{" * rung_tex * "," * next_tex * "," * leg1_tex * "," * leg1_tex * "}+\\alpha_{" * rung_tex * "," * next_tex * "," * leg2_tex * "," * leg2_tex * "}\\right)"
    rung_term = if symmetrize
        "\\frac{1}{2}\\left(\\alpha_{" * rung_tex * "," * rung_tex * "," * leg1_tex * "," * leg2_tex * "}+\\alpha_{" * rung_tex * "," * rung_tex * "," * leg2_tex * "," * leg1_tex * "}\\right)"
    else
        "\\alpha_{" * rung_tex * "," * rung_tex * "," * leg1_tex * "," * leg2_tex * "}"
    end
    return _dwave_alpha_symbol(rung_tex) * "=" * leg_term * "-" * rung_term
end

function _dwave_alpha_title(leg1::Integer, leg2::Integer; rung_tex::AbstractString="i", symmetrize::Bool=false)
    return _math(_dwave_alpha_expr(; rung_tex=rung_tex, leg1_tex=string(leg1), leg2_tex=string(leg2), symmetrize=symmetrize))
end

function density_from_beta(beta; spin=:up, leg::Integer=1, iteration=nothing, use_abs::Bool=false)
    b = _final_array(beta, 5, iteration, "beta")
    sigma = _spin_index(spin)
    leg = _check_leg(leg)
    L = size(b, 2)
    rungs = collect(1:L)
    vals = [b[sigma, i, i, leg, leg] for i in rungs]
    return rungs, _plot_values(vals; use_abs=use_abs)
end

function cdw_from_beta(beta; leg::Integer=1, iteration=nothing, use_abs::Bool=false)
    b = _final_array(beta, 5, iteration, "beta")
    leg = _check_leg(leg)
    L = size(b, 2)
    rungs = collect(1:L)
    vals = [b[2, i, i, leg, leg] + b[1, i, i, leg, leg] for i in rungs]
    return rungs, _plot_values(vals; use_abs=use_abs)
end

function sdw_from_beta(beta; leg::Integer=1, iteration=nothing, use_abs::Bool=false)
    b = _final_array(beta, 5, iteration, "beta")
    leg = _check_leg(leg)
    L = size(b, 2)
    rungs = collect(1:L)
    vals = [b[2, i, i, leg, leg] - b[1, i, i, leg, leg] for i in rungs]
    return rungs, _plot_values(vals; use_abs=use_abs)
end

function density_from_correlations(C_exc_dn, C_exc_up; spin=:up, leg::Integer=1, iteration=nothing, use_abs::Bool=false)
    c = _normal_corr(C_exc_dn, C_exc_up, spin, iteration)
    leg = _check_leg(leg)
    L = _rung_count_from_sites(size(c, 1))
    size(c, 2) == 2 * L || throw(ArgumentError("raw normal correlation matrix must be square"))
    rungs = collect(1:L)
    vals = [c[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] for i in rungs]
    return rungs, _plot_values(vals; use_abs=use_abs)
end

function _normal_corr_pair(C_exc_dn, C_exc_up, iteration)
    cdn = _final_array(C_exc_dn, 2, iteration, "C_exc_dn_list")
    cup = _final_array(C_exc_up, 2, iteration, "C_exc_up_list")
    size(cdn) == size(cup) || throw(ArgumentError("C_exc_dn and C_exc_up must have the same shape"))
    return cdn, cup
end

function cdw_from_correlations(C_exc_dn, C_exc_up; leg::Integer=1, iteration=nothing, use_abs::Bool=false)
    cdn, cup = _normal_corr_pair(C_exc_dn, C_exc_up, iteration)
    leg = _check_leg(leg)
    L = _rung_count_from_sites(size(cup, 1))
    size(cup, 2) == 2 * L || throw(ArgumentError("raw normal correlation matrices must be square"))
    rungs = collect(1:L)
    vals = [cup[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] +
            cdn[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] for i in rungs]
    return rungs, _plot_values(vals; use_abs=use_abs)
end

function sdw_from_correlations(C_exc_dn, C_exc_up; leg::Integer=1, iteration=nothing, use_abs::Bool=false)
    cdn, cup = _normal_corr_pair(C_exc_dn, C_exc_up, iteration)
    leg = _check_leg(leg)
    L = _rung_count_from_sites(size(cup, 1))
    size(cup, 2) == 2 * L || throw(ArgumentError("raw normal correlation matrices must be square"))
    rungs = collect(1:L)
    vals = [cup[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] -
            cdn[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] for i in rungs]
    return rungs, _plot_values(vals; use_abs=use_abs)
end

function onsite_pairing_from_alpha(alpha; leg::Integer=1, iteration=nothing, use_abs::Bool=false)
    a = _final_array(alpha, 4, iteration, "alpha")
    leg = _check_leg(leg)
    L = size(a, 1)
    rungs = collect(1:L)
    vals = [a[i, i, leg, leg] for i in rungs]
    return rungs, _plot_values(vals; use_abs=use_abs)
end

function onsite_pairing_from_correlations(C_pair; leg::Integer=1, iteration=nothing, use_abs::Bool=false)
    c = _final_array(C_pair, 2, iteration, "C_pair_list")
    leg = _check_leg(leg)
    L = _rung_count_from_sites(size(c, 1))
    size(c, 2) == 2 * L || throw(ArgumentError("raw pairing correlation matrix must be square"))
    rungs = collect(1:L)
    vals = [c[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] for i in rungs]
    return rungs, _plot_values(vals; use_abs=use_abs)
end

function rung_pairing_from_alpha(alpha;
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    symmetrize::Bool=false,
    use_abs::Bool=false)

    a = _final_array(alpha, 4, iteration, "alpha")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = size(a, 1)
    rungs = collect(1:L)

    vals = if symmetrize
        [0.5 * (a[i, i, leg1, leg2] + a[i, i, leg2, leg1]) for i in rungs]
    else
        [a[i, i, leg1, leg2] for i in rungs]
    end

    return rungs, _plot_values(vals; use_abs=use_abs)
end

function rung_pairing_from_correlations(C_pair;
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    symmetrize::Bool=false,
    use_abs::Bool=false)

    c = _final_array(C_pair, 2, iteration, "C_pair_list")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = _rung_count_from_sites(size(c, 1))
    size(c, 2) == 2 * L || throw(ArgumentError("raw pairing correlation matrix must be square"))
    rungs = collect(1:L)

    vals = if symmetrize
        [0.5 * (
            c[_plot_rung_leg_to_site(i, leg2), _plot_rung_leg_to_site(i, leg1)] +
            c[_plot_rung_leg_to_site(i, leg1), _plot_rung_leg_to_site(i, leg2)]
        ) for i in rungs]
    else
        [c[_plot_rung_leg_to_site(i, leg2), _plot_rung_leg_to_site(i, leg1)] for i in rungs]
    end

    return rungs, _plot_values(vals; use_abs=use_abs)
end

_alpha_entry(a, i::Integer, ip::Integer, leg::Integer, legp::Integer, it) =
    it === nothing ? a[i, ip, leg, legp] : a[i, ip, leg, legp, it]

function _local_dwave_alpha_value(a, i::Integer, leg1::Integer, leg2::Integer, symmetrize::Bool, it=nothing)
    leg_pair = 0.5 * (
        _alpha_entry(a, i, i + 1, leg1, leg1, it) +
        _alpha_entry(a, i, i + 1, leg2, leg2, it)
    )
    rung_pair = if symmetrize
        0.5 * (
            _alpha_entry(a, i, i, leg1, leg2, it) +
            _alpha_entry(a, i, i, leg2, leg1, it)
        )
    else
        _alpha_entry(a, i, i, leg1, leg2, it)
    end
    return leg_pair - rung_pair
end

function dwave_profile_from_alpha(alpha;
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    symmetrize::Bool=false,
    use_abs::Bool=false)

    a = _final_array(alpha, 4, iteration, "alpha")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = size(a, 1)
    L <= 1 && return Int[], Float64[]
    rungs = collect(1:(L - 1))
    vals = [_local_dwave_alpha_value(a, i, leg1, leg2, symmetrize) for i in rungs]
    return rungs, _plot_values(vals; use_abs=use_abs)
end

_singlet_bond_from_pair_corr(c, site_a::Integer, site_b::Integer) =
    0.5 * (c[site_a, site_b] + c[site_b, site_a])

function _local_dwave_corr_value(c, i::Integer, leg1::Integer, leg2::Integer)
    site_i1 = _plot_rung_leg_to_site(i, leg1)
    site_i2 = _plot_rung_leg_to_site(i, leg2)
    site_ip1 = _plot_rung_leg_to_site(i + 1, leg1)
    site_ip2 = _plot_rung_leg_to_site(i + 1, leg2)
    leg_pair = 0.5 * (
        _singlet_bond_from_pair_corr(c, site_i1, site_ip1) +
        _singlet_bond_from_pair_corr(c, site_i2, site_ip2)
    )
    rung_pair = _singlet_bond_from_pair_corr(c, site_i1, site_i2)
    return leg_pair - rung_pair
end

function dwave_profile_from_correlations(C_pair;
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    use_abs::Bool=false)

    c = _final_array(C_pair, 2, iteration, "C_pair_list")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = _rung_count_from_sites(size(c, 1))
    size(c, 2) == 2 * L || throw(ArgumentError("raw pairing correlation matrix must be square"))
    L <= 1 && return Int[], Float64[]
    rungs = collect(1:(L - 1))
    vals = [_local_dwave_corr_value(c, i, leg1, leg2) for i in rungs]
    return rungs, _plot_values(vals; use_abs=use_abs)
end

function middle_density_history(beta_list; spin=:up, leg::Integer=1, rung=nothing, use_abs::Bool=false)
    b = _history_array(beta_list, 6, "beta_list")
    sigma = _spin_index(spin)
    leg = _check_leg(leg)
    L = size(b, 2)
    rung = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")
    iters = collect(1:size(b, 6))
    vals = [b[sigma, rung, rung, leg, leg, it] for it in iters]
    return iters, _plot_values(vals; use_abs=use_abs)
end

function middle_cdw_history(beta_list; leg::Integer=1, rung=nothing, use_abs::Bool=false)
    b = _history_array(beta_list, 6, "beta_list")
    leg = _check_leg(leg)
    L = size(b, 2)
    rung = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")
    iters = collect(1:size(b, 6))
    vals = [b[2, rung, rung, leg, leg, it] + b[1, rung, rung, leg, leg, it] for it in iters]
    return iters, _plot_values(vals; use_abs=use_abs)
end

function middle_sdw_history(beta_list; leg::Integer=1, rung=nothing, use_abs::Bool=false)
    b = _history_array(beta_list, 6, "beta_list")
    leg = _check_leg(leg)
    L = size(b, 2)
    rung = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")
    iters = collect(1:size(b, 6))
    vals = [b[2, rung, rung, leg, leg, it] - b[1, rung, rung, leg, leg, it] for it in iters]
    return iters, _plot_values(vals; use_abs=use_abs)
end

function middle_density_history_from_correlations(C_exc_dn_list, C_exc_up_list; spin=:up, leg::Integer=1, rung=nothing, use_abs::Bool=false)
    c = _normal_corr_history(C_exc_dn_list, C_exc_up_list, spin)
    leg = _check_leg(leg)
    L = _rung_count_from_sites(size(c, 1))
    size(c, 2) == 2 * L || throw(ArgumentError("raw normal correlation matrices must be square"))
    rung = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")
    site = _plot_rung_leg_to_site(rung, leg)
    iters = collect(1:size(c, 3))
    vals = [c[site, site, it] for it in iters]
    return iters, _plot_values(vals; use_abs=use_abs)
end

function _normal_corr_history_pair(C_exc_dn_list, C_exc_up_list)
    cdn = _history_array(C_exc_dn_list, 3, "C_exc_dn_list")
    cup = _history_array(C_exc_up_list, 3, "C_exc_up_list")
    size(cdn) == size(cup) || throw(ArgumentError("C_exc_dn_list and C_exc_up_list must have the same shape"))
    return cdn, cup
end

function middle_cdw_history_from_correlations(C_exc_dn_list, C_exc_up_list; leg::Integer=1, rung=nothing, use_abs::Bool=false)
    cdn, cup = _normal_corr_history_pair(C_exc_dn_list, C_exc_up_list)
    leg = _check_leg(leg)
    L = _rung_count_from_sites(size(cup, 1))
    size(cup, 2) == 2 * L || throw(ArgumentError("raw normal correlation matrices must be square"))
    rung = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")
    site = _plot_rung_leg_to_site(rung, leg)
    iters = collect(1:size(cup, 3))
    vals = [cup[site, site, it] + cdn[site, site, it] for it in iters]
    return iters, _plot_values(vals; use_abs=use_abs)
end

function middle_sdw_history_from_correlations(C_exc_dn_list, C_exc_up_list; leg::Integer=1, rung=nothing, use_abs::Bool=false)
    cdn, cup = _normal_corr_history_pair(C_exc_dn_list, C_exc_up_list)
    leg = _check_leg(leg)
    L = _rung_count_from_sites(size(cup, 1))
    size(cup, 2) == 2 * L || throw(ArgumentError("raw normal correlation matrices must be square"))
    rung = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")
    site = _plot_rung_leg_to_site(rung, leg)
    iters = collect(1:size(cup, 3))
    vals = [cup[site, site, it] - cdn[site, site, it] for it in iters]
    return iters, _plot_values(vals; use_abs=use_abs)
end

function middle_onsite_pairing_history(alpha_list; leg::Integer=1, rung=nothing, use_abs::Bool=false)
    a = _history_array(alpha_list, 5, "alpha_list")
    leg = _check_leg(leg)
    L = size(a, 1)
    rung = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")
    iters = collect(1:size(a, 5))
    vals = [a[rung, rung, leg, leg, it] for it in iters]
    return iters, _plot_values(vals; use_abs=use_abs)
end

function middle_onsite_pairing_history_from_correlations(C_pair_list; leg::Integer=1, rung=nothing, use_abs::Bool=false)
    c = _history_array(C_pair_list, 3, "C_pair_list")
    leg = _check_leg(leg)
    L = _rung_count_from_sites(size(c, 1))
    size(c, 2) == 2 * L || throw(ArgumentError("raw pairing correlation matrices must be square"))
    rung = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")
    site = _plot_rung_leg_to_site(rung, leg)
    iters = collect(1:size(c, 3))
    vals = [c[site, site, it] for it in iters]
    return iters, _plot_values(vals; use_abs=use_abs)
end

function middle_rung_pairing_history(alpha_list;
    rung=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize::Bool=false,
    use_abs::Bool=false)

    a = _history_array(alpha_list, 5, "alpha_list")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = size(a, 1)
    rung = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")
    iters = collect(1:size(a, 5))

    vals = if symmetrize
        [0.5 * (a[rung, rung, leg1, leg2, it] + a[rung, rung, leg2, leg1, it]) for it in iters]
    else
        [a[rung, rung, leg1, leg2, it] for it in iters]
    end

    return iters, _plot_values(vals; use_abs=use_abs)
end

function middle_rung_pairing_history_from_correlations(C_pair_list;
    rung=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize::Bool=false,
    use_abs::Bool=false)

    c = _history_array(C_pair_list, 3, "C_pair_list")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = _rung_count_from_sites(size(c, 1))
    size(c, 2) == 2 * L || throw(ArgumentError("raw pairing correlation matrices must be square"))
    rung = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")
    site1 = _plot_rung_leg_to_site(rung, leg1)
    site2 = _plot_rung_leg_to_site(rung, leg2)
    iters = collect(1:size(c, 3))

    vals = if symmetrize
        [0.5 * (c[site2, site1, it] + c[site1, site2, it]) for it in iters]
    else
        [c[site2, site1, it] for it in iters]
    end

    return iters, _plot_values(vals; use_abs=use_abs)
end

function middle_dwave_history(alpha_list;
    rung=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize::Bool=false,
    use_abs::Bool=false)

    a = _history_array(alpha_list, 5, "alpha_list")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = size(a, 1)
    L > 1 || throw(ArgumentError("need at least two rungs to compute local d-wave profile"))
    rung = rung === nothing ? min(middle_rung(L), L - 1) : _check_index(rung, L - 1, "d-wave rung")
    iters = collect(1:size(a, 5))
    vals = [_local_dwave_alpha_value(a, rung, leg1, leg2, symmetrize, it) for it in iters]
    return iters, _plot_values(vals; use_abs=use_abs)
end

function middle_dwave_history_from_correlations(C_pair_list;
    rung=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    use_abs::Bool=false)

    c = _history_array(C_pair_list, 3, "C_pair_list")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = _rung_count_from_sites(size(c, 1))
    size(c, 2) == 2 * L || throw(ArgumentError("raw pairing correlation matrices must be square"))
    L > 1 || throw(ArgumentError("need at least two rungs to compute local d-wave profile"))
    rung = rung === nothing ? min(middle_rung(L), L - 1) : _check_index(rung, L - 1, "d-wave rung")
    iters = collect(1:size(c, 3))
    vals = [_local_dwave_corr_value(selectdim(c, 3, it), rung, leg1, leg2) for it in iters]
    return iters, _plot_values(vals; use_abs=use_abs)
end

function _momentum_shift_indices(n::Integer)
    n > 0 || throw(ArgumentError("Fourier transform needs at least one point"))
    half = div(n, 2)
    if iseven(n)
        return vcat((half + 1):n, 1:half)
    end
    return vcat((half + 2):n, 1:(half + 1))
end

function _momentum_axis(n::Integer)
    shifted = _momentum_shift_indices(n)
    half = n / 2
    return [((idx - 1) < half ? (idx - 1) : (idx - 1 - n)) * 2π / n for idx in shifted]
end

function _dft_1d(vals; normalize::Bool=true)
    n = length(vals)
    n > 0 || throw(ArgumentError("Fourier transform needs at least one point"))
    scale = normalize ? n : 1
    raw = [sum(vals[x] * exp(-2im * π * (k - 1) * (x - 1) / n) for x in 1:n) / scale for k in 1:n]
    shifted = _momentum_shift_indices(n)
    return _momentum_axis(n), raw[shifted]
end

function _fourier_display_values(vals, value::Symbol)
    if value in (:abs, :amplitude, :magnitude)
        return abs.(vals)
    elseif value in (:power, :intensity)
        return abs2.(vals)
    elseif value == :real
        return real.(vals)
    elseif value == :imag
        return imag.(vals)
    end
    throw(ArgumentError("value must be :abs, :power, :real, or :imag"))
end

function _fourier_colorbar_label(value::Symbol)
    value in (:abs, :amplitude, :magnitude) && return "Fourier amplitude"
    value in (:power, :intensity) && return "Fourier power"
    value == :real && return "Re Fourier component"
    value == :imag && return "Im Fourier component"
    return "Fourier value"
end

function _ladder_fourier_map(field; subtract_average::Bool=false, normalize::Bool=true, value::Symbol=:abs)
    size(field, 2) == 2 || throw(ArgumentError("ladder Fourier map expects an L x 2 order field"))
    f = Array(field)
    if subtract_average
        f = f .- sum(f) / length(f)
    end

    L = size(f, 1)
    scale = normalize ? 2L : 1
    qx = _momentum_axis(L)
    shifted = _momentum_shift_indices(L)
    vals = zeros(ComplexF64, 2, L)
    for k in 1:L
        phase_sum_0 = zero(ComplexF64)
        phase_sum_pi = zero(ComplexF64)
        for i in 1:L
            phase = exp(-2im * π * (k - 1) * (i - 1) / L)
            phase_sum_0 += (f[i, 1] + f[i, 2]) * phase
            phase_sum_pi += (f[i, 1] - f[i, 2]) * phase
        end
        vals[1, k] = phase_sum_0 / scale
        vals[2, k] = phase_sum_pi / scale
    end
    return qx, _fourier_display_values(vals[:, shifted], value)
end

function _chain_fourier_map(field; normalize::Bool=true, value::Symbol=:abs)
    qx, vals = _dft_1d(field; normalize=normalize)
    return qx, reshape(_fourier_display_values(vals, value), 1, :)
end

function _mf_order_fields(alpha, beta;
    iteration=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize_rung::Bool=false)

    a = _final_array(alpha, 4, iteration, "alpha")
    b = _final_array(beta, 5, iteration, "beta")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = size(b, 2)
    size(a, 1) == L || throw(ArgumentError("alpha and beta must have the same rung count"))
    L > 1 || throw(ArgumentError("need at least two rungs to compute local d-wave profile"))

    cdw = [b[2, i, i, leg, leg] + b[1, i, i, leg, leg] for i in 1:L, leg in 1:2]
    sdw = [b[2, i, i, leg, leg] - b[1, i, i, leg, leg] for i in 1:L, leg in 1:2]
    swave = [a[i, i, leg, leg] for i in 1:L, leg in 1:2]
    dwave = [_local_dwave_alpha_value(a, i, leg1, leg2, symmetrize_rung) for i in 1:(L - 1)]
    return (cdw=cdw, sdw=sdw, swave=swave, dwave=dwave)
end

function _correlation_order_fields(C_pair, C_exc_dn, C_exc_up;
    iteration=nothing,
    leg1::Integer=1,
    leg2::Integer=2)

    cpair = _final_array(C_pair, 2, iteration, "C_pair_list")
    cdn, cup = _normal_corr_pair(C_exc_dn, C_exc_up, iteration)
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = _rung_count_from_sites(size(cpair, 1))
    size(cpair, 2) == 2 * L || throw(ArgumentError("raw pairing correlation matrix must be square"))
    size(cdn) == size(cup) == size(cpair) || throw(ArgumentError("raw correlation matrices must have the same shape"))
    L > 1 || throw(ArgumentError("need at least two rungs to compute local d-wave profile"))

    cdw = [cup[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] +
           cdn[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] for i in 1:L, leg in 1:2]
    sdw = [cup[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] -
           cdn[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] for i in 1:L, leg in 1:2]
    swave = [cpair[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] for i in 1:L, leg in 1:2]
    dwave = [_local_dwave_corr_value(cpair, i, leg1, leg2) for i in 1:(L - 1)]
    return (cdw=cdw, sdw=sdw, swave=swave, dwave=dwave)
end

function _plot_fourier_panel!(fig, ax, heatmap;
    title::AbstractString,
    ladder::Bool,
    cmap::AbstractString,
    colorbar_label::AbstractString,
    kwargs...)

    ncols = size(heatmap, 2)
    zero_tick = div(ncols, 2)
    img = ax.imshow(heatmap; origin="lower", aspect="equal", interpolation="nearest", cmap=cmap, kwargs...)
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(-0.5, size(heatmap, 1) - 0.5)
    ax.set_xlabel(_math("k_x"))
    ax.set_ylabel(ladder ? _math("k_y") : "")
    ax.set_xticks([0, zero_tick, ncols - 1])
    ax.set_xticklabels([_math("-\\pi"), "0", _math("\\pi")])
    ax.set_yticks(ladder ? [0, 1] : [0])
    ax.set_yticklabels(ladder ? ["0", _math("\\pi")] : ["chain"])
    ax.set_title(title)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(colorbar_label)
    return img
end

function _plot_order_fourier_heatmaps(fields;
    value::Symbol=:abs,
    normalize::Bool=true,
    cmap::AbstractString="viridis",
    savepath=nothing,
    figure_title=nothing,
    source_label::AbstractString="",
    kwargs...)

    _, cdw_map = _ladder_fourier_map(fields.cdw; subtract_average=true, normalize=normalize, value=value)
    _, sdw_map = _ladder_fourier_map(fields.sdw; normalize=normalize, value=value)
    _, swave_map = _ladder_fourier_map(fields.swave; normalize=normalize, value=value)
    _, dwave_map = _chain_fourier_map(fields.dwave; normalize=normalize, value=value)

    fig, axes = PyPlot.subplots(4, 1, figsize=(10.5, 9.0))
    axs = vec(axes)
    cbar_label = _fourier_colorbar_label(value)
    label_suffix = isempty(source_label) ? "" : " ($source_label)"
    _plot_fourier_panel!(fig, axs[1], cdw_map; title="CDW Fourier heatmap, mean subtracted" * label_suffix, ladder=true, cmap=cmap, colorbar_label=cbar_label, kwargs...)
    _plot_fourier_panel!(fig, axs[2], sdw_map; title="SDW Fourier heatmap" * label_suffix, ladder=true, cmap=cmap, colorbar_label=cbar_label, kwargs...)
    _plot_fourier_panel!(fig, axs[3], swave_map; title="s-wave Fourier heatmap" * label_suffix, ladder=true, cmap=cmap, colorbar_label=cbar_label, kwargs...)
    _plot_fourier_panel!(fig, axs[4], dwave_map; title="d-wave local profile Fourier heatmap" * label_suffix, ladder=false, cmap=cmap, colorbar_label=cbar_label, kwargs...)

    tight_rect = figure_title === nothing ? nothing : (0, 0, 1, 0.95)
    _finish_figure(fig; figure_title=figure_title, tight_rect=tight_rect)
    fig.subplots_adjust(hspace=0.65)
    return _save_if_requested(fig, savepath)
end

function _save_if_requested(fig, savepath)
    savepath !== nothing && fig.savefig(savepath, bbox_inches="tight")
    return fig
end

function _plot_series!(ax, x, y; xlabel, ylabel, title, kwargs...)
    ax.plot(x, y; marker="o", linewidth=2, kwargs...)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(true, alpha=0.3)
    return ax
end

function _finish_figure(fig; figure_title=nothing, tight_rect=nothing)
    if figure_title !== nothing
        fig.suptitle(figure_title)
    end
    if tight_rect === nothing
        fig.tight_layout()
    else
        fig.tight_layout(rect=tight_rect)
    end
    return fig
end

_filename_title(filename::AbstractString) = basename(filename)

function _figure(figsize)
    fig, ax = PyPlot.subplots(figsize=figsize)
    return fig, ax
end

function _mf_change_field(field, spin)
    f = field isa Symbol ? field : Symbol(lowercase(String(field)))
    if f in (:alpha, :alphas)
        return :alpha, spin
    elseif f in (:beta, :betas)
        return :beta, spin
    elseif f in (:beta_up, :betaup, :up)
        return :beta, :up
    elseif f in (:beta_down, :betadown, :beta_dn, :betadn, :down, :dn)
        return :beta, :down
    end
    throw(ArgumentError("field must be :alpha, :beta, :beta_up, or :beta_down"))
end

function _mf_change_history(alpha_list, beta_list; field=:alpha, spin=:up, leg1::Integer=1, leg2::Integer=1, zero_threshold::Real=1e-4)
    quantity, spin = _mf_change_field(field, spin)
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)

    if quantity == :alpha
        a = _history_array(alpha_list, 5, "alpha_list")
        hist = Array(@view a[:, :, leg1, leg2, :])
        label = "\\alpha[i,i'," * ",$leg1,$leg2]"
    else
        b = _history_array(beta_list, 6, "beta_list")
        sigma = _spin_index(spin)
        hist = Array(@view b[sigma, :, :, leg1, leg2, :])
        label = "\\beta_{$(_spin_tex(spin))}[i,i'," * ",$leg1,$leg2]"
    end

    hist[abs.(hist) .< zero_threshold] .= zero(eltype(hist))
    return hist, label
end

function _mf_change_frame(hist, it::Integer; eps::Real=1e-12, threshold::Real=1e-3, mode::Symbol=:mask)
    niter = size(hist, 3)
    2 <= it <= niter || throw(ArgumentError("it must be between 2 and $niter; got $it"))

    cur = @view hist[:, :, it]
    prev = @view hist[:, :, it - 1]
    if mode in (:mask, :threshold, :bool)
        rel = abs.(cur .- prev) ./ (abs.(prev) .+ eps)
        return rel .> threshold
    elseif mode in (:relative, :rel)
        return abs.(cur .- prev) ./ (abs.(prev) .+ eps)
    elseif mode in (:absolute, :absdiff, :diff)
        return abs.(cur .- prev)
    end
    throw(ArgumentError("mode must be :mask, :relative, or :absolute"))
end

function _mf_change_colorbar_label(mode::Symbol, threshold::Real)
    if mode in (:mask, :threshold, :bool)
        return "relative change > $threshold"
    elseif mode in (:relative, :rel)
        return "relative change"
    elseif mode in (:absolute, :absdiff, :diff)
        return "absolute change"
    end
    return "change"
end

function _mf_change_title(label::AbstractString, it::Integer, niter::Integer; mode::Symbol=:mask, threshold::Real=1e-3)
    mode_text = mode in (:mask, :threshold, :bool) ? "relative-change mask, threshold=$threshold" :
        mode in (:relative, :rel) ? "relative change" : "absolute change"
    return _math(label * "\\quad " * "\\mathrm{" * mode_text * "}\\quad m=" * string(it) * "/" * string(niter))
end

function plot_mf_change_slider(alpha_list, beta_list;
    field=:alpha,
    spin=:up,
    leg1::Integer=1,
    leg2::Integer=1,
    idx0=nothing,
    eps::Real=1e-12,
    threshold::Real=1e-3,
    zero_threshold::Real=1e-4,
    mode::Symbol=:mask,
    figsize=(7.0, 6.0),
    cmap::AbstractString="viridis",
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    hist, label = _mf_change_history(alpha_list, beta_list; field=field, spin=spin, leg1=leg1, leg2=leg2, zero_threshold=zero_threshold)
    niter = size(hist, 3)
    niter >= 2 || throw(ArgumentError("need at least two MF iterations to plot changes; got $niter"))
    it0 = idx0 === nothing ? 2 : _check_index(idx0, niter, "idx0")
    it0 >= 2 || throw(ArgumentError("idx0 must be at least 2 because changes compare idx0 to idx0 - 1"))

    frame = _mf_change_frame(hist, it0; eps=eps, threshold=threshold, mode=mode)
    fig, ax = PyPlot.subplots(figsize=figsize)
    imshow_kwargs = mode in (:mask, :threshold, :bool) ? (vmin=0, vmax=1) : NamedTuple()
    img = ax.imshow(frame; origin="lower", aspect="auto", cmap=cmap, imshow_kwargs..., kwargs...)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(_mf_change_colorbar_label(mode, threshold))
    ax.set_xlabel("Rung index \$i'\$")
    ax.set_ylabel("Rung index \$i\$")
    ax.set_title(_mf_change_title(label, it0, niter; mode=mode, threshold=threshold))

    slider_ax = fig.add_axes([0.25, 0.05, 0.50, 0.03])
    slider = _MatplotlibWidgets.Slider(slider_ax, "MF iteration", 2, niter; valinit=it0, valstep=1, valfmt="%0.0f")

    function update(val)
        it = Int(round(val))
        img.set_data(_mf_change_frame(hist, it; eps=eps, threshold=threshold, mode=mode))
        ax.set_title(_mf_change_title(label, it, niter; mode=mode, threshold=threshold))
        fig.canvas.draw_idle()
        return nothing
    end

    callback_id = slider.on_changed(update)
    if figure_title !== nothing
        fig.suptitle(figure_title)
        fig.subplots_adjust(bottom=0.18, top=0.88)
    else
        fig.subplots_adjust(bottom=0.18)
    end
    _save_if_requested(fig, savepath)

    return (
        fig=fig,
        ax=ax,
        image=img,
        colorbar=cbar,
        slider=slider,
        callback=update,
        callback_id=callback_id,
        data=hist,
    )
end

function plot_mf_change_slider_from_file(filename::AbstractString;
    field=:alpha,
    spin=:up,
    leg1::Integer=1,
    leg2::Integer=1,
    figure_title=_filename_title(filename),
    kwargs...)

    data = load_mf_data(filename)
    return plot_mf_change_slider(data.alpha_list, data.beta_list; field=field, spin=spin, leg1=leg1, leg2=leg2, figure_title=figure_title, kwargs...)
end

function plot_density_from_beta(beta; spin=:up, leg::Integer=1, iteration=nothing, use_abs::Bool=false, savepath=nothing, figure_title=nothing, kwargs...)
    rungs, vals = density_from_beta(beta; spin=spin, leg=leg, iteration=iteration, use_abs=use_abs)
    ylabel = use_abs ? "|onsite beta $( _spin_label(spin) )|" : "onsite beta $(_spin_label(spin))"
    fig, ax = _figure((8.5, 4.0))
    _plot_series!(ax, rungs, vals;
        xlabel="Rung index",
        ylabel=ylabel,
        title="Density proxy from beta, leg $leg",
        kwargs...)
    _finish_figure(fig; figure_title=figure_title)
    return _save_if_requested(fig, savepath)
end

function plot_density_from_correlations(C_exc_dn, C_exc_up; spin=:up, leg::Integer=1, iteration=nothing, use_abs::Bool=false, savepath=nothing, figure_title=nothing, kwargs...)
    rungs, vals = density_from_correlations(C_exc_dn, C_exc_up; spin=spin, leg=leg, iteration=iteration, use_abs=use_abs)
    expr = _density_corr_expr(spin; leg_tex=string(leg))
    fig, ax = _figure((8.5, 4.0))
    _plot_series!(ax, rungs, vals;
        xlabel="Rung index \$i\$",
        ylabel=_corr_value_label(expr, use_abs),
        title=_density_corr_title(spin, leg),
        kwargs...)
    _finish_figure(fig; figure_title=figure_title)
    return _save_if_requested(fig, savepath)
end

function plot_onsite_pairing_from_alpha(alpha; leg::Integer=1, iteration=nothing, use_abs::Bool=false, savepath=nothing, figure_title=nothing, kwargs...)
    rungs, vals = onsite_pairing_from_alpha(alpha; leg=leg, iteration=iteration, use_abs=use_abs)
    ylabel = use_abs ? "|s-wave alpha|" : "s-wave alpha"
    fig, ax = _figure((8.5, 4.0))
    _plot_series!(ax, rungs, vals;
        xlabel="Rung index",
        ylabel=ylabel,
        title="s-wave pairing from alpha, leg $leg",
        kwargs...)
    _finish_figure(fig; figure_title=figure_title)
    return _save_if_requested(fig, savepath)
end

function plot_onsite_pairing_from_correlations(C_pair; leg::Integer=1, iteration=nothing, use_abs::Bool=false, savepath=nothing, figure_title=nothing, kwargs...)
    rungs, vals = onsite_pairing_from_correlations(C_pair; leg=leg, iteration=iteration, use_abs=use_abs)
    expr = _onsite_pair_corr_expr(; leg_tex=string(leg))
    fig, ax = _figure((8.5, 4.0))
    _plot_series!(ax, rungs, vals;
        xlabel="Rung index \$i\$",
        ylabel=_corr_value_label(expr, use_abs),
        title="s-wave: " * _onsite_pair_corr_title(leg),
        kwargs...)
    _finish_figure(fig; figure_title=figure_title)
    return _save_if_requested(fig, savepath)
end

function plot_rung_pairing_from_alpha(alpha;
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    symmetrize::Bool=false,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    rungs, vals = dwave_profile_from_alpha(alpha; leg1=leg1, leg2=leg2, iteration=iteration, symmetrize=symmetrize, use_abs=use_abs)
    ylabel = use_abs ? "|d-wave alpha|" : "d-wave alpha"
    fig, ax = _figure((8.5, 4.0))
    _plot_series!(ax, rungs, vals;
        xlabel="Rung index",
        ylabel=ylabel,
        title="d-wave alpha proxy: " * _dwave_alpha_title(leg1, leg2; symmetrize=symmetrize),
        kwargs...)
    _finish_figure(fig; figure_title=figure_title)
    return _save_if_requested(fig, savepath)
end

function plot_rung_pairing_from_correlations(C_pair;
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    symmetrize::Bool=false,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    rungs, vals = dwave_profile_from_correlations(C_pair; leg1=leg1, leg2=leg2, iteration=iteration, use_abs=use_abs)
    fig, ax = _figure((8.5, 4.0))
    _plot_series!(ax, rungs, vals;
        xlabel="Rung index \$i\$",
        ylabel=_corr_value_label(_dwave_corr_symbol(), use_abs),
        title="d-wave: " * _dwave_corr_title(leg1, leg2),
        kwargs...)
    _finish_figure(fig; figure_title=figure_title)
    return _save_if_requested(fig, savepath)
end

function plot_order_fourier_heatmaps(alpha, beta;
    iteration=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize_rung::Bool=false,
    value::Symbol=:abs,
    normalize::Bool=true,
    cmap::AbstractString="viridis",
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    fields = _mf_order_fields(alpha, beta; iteration=iteration, leg1=leg1, leg2=leg2, symmetrize_rung=symmetrize_rung)
    return _plot_order_fourier_heatmaps(fields; value=value, normalize=normalize, cmap=cmap, savepath=savepath, figure_title=figure_title, source_label="alpha/beta", kwargs...)
end

function plot_order_fourier_heatmaps_from_correlations(C_pair, C_exc_dn, C_exc_up;
    iteration=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    value::Symbol=:abs,
    normalize::Bool=true,
    cmap::AbstractString="viridis",
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    fields = _correlation_order_fields(C_pair, C_exc_dn, C_exc_up; iteration=iteration, leg1=leg1, leg2=leg2)
    return _plot_order_fourier_heatmaps(fields; value=value, normalize=normalize, cmap=cmap, savepath=savepath, figure_title=figure_title, source_label="correlations", kwargs...)
end

function plot_mf_profiles(alpha, beta;
    spin=:up,
    leg::Integer=1,
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    symmetrize_rung::Bool=false,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    r1, v1 = cdw_from_beta(beta; leg=leg, iteration=iteration, use_abs=use_abs)
    r2, v2 = sdw_from_beta(beta; leg=leg, iteration=iteration, use_abs=use_abs)
    r3, v3 = onsite_pairing_from_alpha(alpha; leg=leg, iteration=iteration, use_abs=use_abs)
    r4, v4 = dwave_profile_from_alpha(alpha; leg1=leg1, leg2=leg2, iteration=iteration, symmetrize=symmetrize_rung, use_abs=use_abs)

    fig, axes = PyPlot.subplots(4, 1, figsize=(8.5, 10.5))
    cdw_ylabel = use_abs ? "|CDW beta proxy|" : "CDW beta proxy"
    sdw_ylabel = use_abs ? "|SDW beta proxy|" : "SDW beta proxy"
    onsite_ylabel = use_abs ? "|s-wave alpha|" : "s-wave alpha"
    rung_ylabel = use_abs ? "|d-wave alpha|" : "d-wave alpha"

    _plot_series!(axes[1], r1, v1; xlabel="Rung index", ylabel=cdw_ylabel, title="CDW beta proxy: up + down, leg $leg", kwargs...)
    _plot_series!(axes[2], r2, v2; xlabel="Rung index", ylabel=sdw_ylabel, title="SDW beta proxy: up - down, leg $leg", kwargs...)
    _plot_series!(axes[3], r3, v3; xlabel="Rung index", ylabel=onsite_ylabel, title="s-wave pairing from alpha, leg $leg", kwargs...)
    _plot_series!(axes[4], r4, v4; xlabel="Rung index", ylabel=rung_ylabel, title="d-wave alpha proxy: " * _dwave_alpha_title(leg1, leg2; symmetrize=symmetrize_rung), kwargs...)

    tight_rect = figure_title === nothing ? nothing : (0, 0, 1, 0.95)
    _finish_figure(fig; figure_title=figure_title, tight_rect=tight_rect)
    return _save_if_requested(fig, savepath)
end

function plot_correlation_profiles(C_pair, C_exc_dn, C_exc_up;
    spin=:up,
    leg::Integer=1,
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    symmetrize_rung::Bool=false,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    r1, v1 = cdw_from_correlations(C_exc_dn, C_exc_up; leg=leg, iteration=iteration, use_abs=use_abs)
    r2, v2 = sdw_from_correlations(C_exc_dn, C_exc_up; leg=leg, iteration=iteration, use_abs=use_abs)
    r3, v3 = onsite_pairing_from_correlations(C_pair; leg=leg, iteration=iteration, use_abs=use_abs)
    r4, v4 = dwave_profile_from_correlations(C_pair; leg1=leg1, leg2=leg2, iteration=iteration, use_abs=use_abs)

    fig, axes = PyPlot.subplots(4, 1, figsize=(8.5, 10.5))
    cdw_expr = _cdw_corr_expr(; leg_tex=string(leg))
    sdw_expr = _sdw_corr_expr(; leg_tex=string(leg))
    onsite_expr = _onsite_pair_corr_expr(; leg_tex=string(leg))
    dwave_expr = _dwave_corr_symbol()

    _plot_series!(axes[1], r1, v1; xlabel="Rung index \$i\$", ylabel=_corr_value_label(cdw_expr, use_abs), title="CDW: " * _cdw_corr_title(leg), kwargs...)
    _plot_series!(axes[2], r2, v2; xlabel="Rung index \$i\$", ylabel=_corr_value_label(sdw_expr, use_abs), title="SDW: " * _sdw_corr_title(leg), kwargs...)
    _plot_series!(axes[3], r3, v3; xlabel="Rung index \$i\$", ylabel=_corr_value_label(onsite_expr, use_abs), title="s-wave: " * _onsite_pair_corr_title(leg), kwargs...)
    _plot_series!(axes[4], r4, v4; xlabel="Rung index \$i\$", ylabel=_corr_value_label(dwave_expr, use_abs), title="d-wave: " * _dwave_corr_title(leg1, leg2), kwargs...)

    tight_rect = figure_title === nothing ? nothing : (0, 0, 1, 0.95)
    _finish_figure(fig; figure_title=figure_title, tight_rect=tight_rect)
    return _save_if_requested(fig, savepath)
end

function plot_middle_histories(alpha_list, beta_list;
    spin=:up,
    leg::Integer=1,
    rung=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize_rung::Bool=false,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    b = _history_array(beta_list, 6, "beta_list")
    L = size(b, 2)
    L > 1 || throw(ArgumentError("need at least two rungs to compute local d-wave profile"))
    rung_to_plot = rung === nothing ? min(middle_rung(L), L - 1) : _check_index(rung, L - 1, "d-wave rung")

    it1, v1 = middle_cdw_history(beta_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it2, v2 = middle_sdw_history(beta_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it3, v3 = middle_onsite_pairing_history(alpha_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it4, v4 = middle_dwave_history(alpha_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, symmetrize=symmetrize_rung, use_abs=use_abs)

    fig, axes = PyPlot.subplots(4, 1, figsize=(8.5, 10.5))
    cdw_ylabel = use_abs ? "|CDW beta proxy|" : "CDW beta proxy"
    sdw_ylabel = use_abs ? "|SDW beta proxy|" : "SDW beta proxy"
    onsite_ylabel = use_abs ? "|s-wave alpha|" : "s-wave alpha"
    rung_ylabel = use_abs ? "|d-wave alpha|" : "d-wave alpha"

    _plot_series!(axes[1], it1, v1; xlabel="MF iteration", ylabel=cdw_ylabel, title="Middle rung $rung_to_plot CDW beta proxy: up + down", kwargs...)
    _plot_series!(axes[2], it2, v2; xlabel="MF iteration", ylabel=sdw_ylabel, title="Middle rung $rung_to_plot SDW beta proxy: up - down", kwargs...)
    _plot_series!(axes[3], it3, v3; xlabel="MF iteration", ylabel=onsite_ylabel, title="Middle rung $rung_to_plot s-wave pairing", kwargs...)
    _plot_series!(axes[4], it4, v4; xlabel="MF iteration", ylabel=rung_ylabel, title="Middle rung $rung_to_plot d-wave alpha proxy: " * _dwave_alpha_title(leg1, leg2; rung_tex=string(rung_to_plot), symmetrize=symmetrize_rung), kwargs...)

    tight_rect = figure_title === nothing ? nothing : (0, 0, 1, 0.95)
    _finish_figure(fig; figure_title=figure_title, tight_rect=tight_rect)
    return _save_if_requested(fig, savepath)
end

function plot_middle_histories_from_correlations(C_pair_list, C_exc_dn_list, C_exc_up_list;
    spin=:up,
    leg::Integer=1,
    rung=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize_rung::Bool=false,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    c = _history_array(C_pair_list, 3, "C_pair_list")
    L = _rung_count_from_sites(size(c, 1))
    L > 1 || throw(ArgumentError("need at least two rungs to compute local d-wave profile"))
    rung_to_plot = rung === nothing ? min(middle_rung(L), L - 1) : _check_index(rung, L - 1, "d-wave rung")

    it1, v1 = middle_cdw_history_from_correlations(C_exc_dn_list, C_exc_up_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it2, v2 = middle_sdw_history_from_correlations(C_exc_dn_list, C_exc_up_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it3, v3 = middle_onsite_pairing_history_from_correlations(C_pair_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it4, v4 = middle_dwave_history_from_correlations(C_pair_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, use_abs=use_abs)

    fig, axes = PyPlot.subplots(4, 1, figsize=(8.5, 10.5))
    rung_tex = string(rung_to_plot)
    cdw_expr = _cdw_corr_expr(; rung_tex=rung_tex, leg_tex=string(leg))
    sdw_expr = _sdw_corr_expr(; rung_tex=rung_tex, leg_tex=string(leg))
    onsite_expr = _onsite_pair_corr_expr(; rung_tex=rung_tex, leg_tex=string(leg))
    dwave_expr = _dwave_corr_symbol(rung_tex)

    _plot_series!(axes[1], it1, v1; xlabel="MF iteration \$m\$", ylabel=_corr_value_label(cdw_expr, use_abs), title="CDW: " * _cdw_corr_title(leg; rung_tex=rung_tex), kwargs...)
    _plot_series!(axes[2], it2, v2; xlabel="MF iteration \$m\$", ylabel=_corr_value_label(sdw_expr, use_abs), title="SDW: " * _sdw_corr_title(leg; rung_tex=rung_tex), kwargs...)
    _plot_series!(axes[3], it3, v3; xlabel="MF iteration \$m\$", ylabel=_corr_value_label(onsite_expr, use_abs), title="s-wave: " * _onsite_pair_corr_title(leg; rung_tex=rung_tex), kwargs...)
    _plot_series!(axes[4], it4, v4; xlabel="MF iteration \$m\$", ylabel=_corr_value_label(dwave_expr, use_abs), title="d-wave: " * _dwave_corr_title(leg1, leg2; rung_tex=rung_tex), kwargs...)

    tight_rect = figure_title === nothing ? nothing : (0, 0, 1, 0.95)
    _finish_figure(fig; figure_title=figure_title, tight_rect=tight_rect)
    return _save_if_requested(fig, savepath)
end

function plot_mf_profiles_from_file(filename::AbstractString; iteration=nothing, source::Symbol=:mf, use_correlations::Bool=false, savepath=nothing, kwargs...)
    data = load_mf_data(filename)
    if _source(source, use_correlations) == :correlations
        C_pair_source = (iteration === nothing && data.C_pair !== nothing) ? data.C_pair : data.C_pair_list
        C_exc_dn_source = (iteration === nothing && data.C_exc_dn !== nothing) ? data.C_exc_dn : data.C_exc_dn_list
        C_exc_up_source = (iteration === nothing && data.C_exc_up !== nothing) ? data.C_exc_up : data.C_exc_up_list
        return plot_correlation_profiles(C_pair_source, C_exc_dn_source, C_exc_up_source; iteration=iteration, figure_title=_filename_title(filename), savepath=savepath, kwargs...)
    else
        alpha_source = (iteration === nothing && data.alpha !== nothing) ? data.alpha : data.alpha_list
        beta_source = (iteration === nothing && data.beta !== nothing) ? data.beta : data.beta_list
        return plot_mf_profiles(alpha_source, beta_source; iteration=iteration, figure_title=_filename_title(filename), savepath=savepath, kwargs...)
    end
end

function plot_middle_histories_from_file(filename::AbstractString; source::Symbol=:mf, use_correlations::Bool=false, savepath=nothing, kwargs...)
    data = load_mf_data(filename)
    if _source(source, use_correlations) == :correlations
        return plot_middle_histories_from_correlations(data.C_pair_list, data.C_exc_dn_list, data.C_exc_up_list; figure_title=_filename_title(filename), savepath=savepath, kwargs...)
    else
        return plot_middle_histories(data.alpha_list, data.beta_list; figure_title=_filename_title(filename), savepath=savepath, kwargs...)
    end
end

function plot_order_fourier_heatmaps_from_file(filename::AbstractString; iteration=nothing, source::Symbol=:mf, use_correlations::Bool=false, savepath=nothing, kwargs...)
    data = load_mf_data(filename)
    if _source(source, use_correlations) == :correlations
        C_pair_source = (iteration === nothing && data.C_pair !== nothing) ? data.C_pair : data.C_pair_list
        C_exc_dn_source = (iteration === nothing && data.C_exc_dn !== nothing) ? data.C_exc_dn : data.C_exc_dn_list
        C_exc_up_source = (iteration === nothing && data.C_exc_up !== nothing) ? data.C_exc_up : data.C_exc_up_list
        return plot_order_fourier_heatmaps_from_correlations(C_pair_source, C_exc_dn_source, C_exc_up_source; iteration=iteration, figure_title=_filename_title(filename), savepath=savepath, kwargs...)
    else
        alpha_source = (iteration === nothing && data.alpha !== nothing) ? data.alpha : data.alpha_list
        beta_source = (iteration === nothing && data.beta !== nothing) ? data.beta : data.beta_list
        return plot_order_fourier_heatmaps(alpha_source, beta_source; iteration=iteration, figure_title=_filename_title(filename), savepath=savepath, kwargs...)
    end
end
