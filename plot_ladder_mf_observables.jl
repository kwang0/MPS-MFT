using HDF5
import PyPlot
import PyCall

const _MatplotlibWidgets = PyCall.pyimport("matplotlib.widgets")
const _MatplotlibColors = PyCall.pyimport("matplotlib.colors")
const _MatplotlibPatches = PyCall.pyimport("matplotlib.patches")
const _MatplotlibCm = PyCall.pyimport("matplotlib.cm")
const _MatplotlibLegendHandler = PyCall.pyimport("matplotlib.legend_handler")
const _MatplotlibLines = PyCall.pyimport("matplotlib.lines")
const _MatplotlibPath = PyCall.pyimport("matplotlib.path")
const _CDW_LEG_COLORS = ("#cb181d", "#fcae91")
const _SDW_LEG_COLORS = ("#000000", "#969696")
const _SWAVE_LEG_COLORS = ("#006d2c", "#a1d99b")
const _EXT_SWAVE_COLOR = "#006d2c"
const _DWAVE_COLOR = "purple"
const _ORDER_GRID_LEGEND_MARKER_SIZE = 10
const _MF_STACK_H_PAD = 1.6
const _FOURIER_STACK_HSPACE = 0.1
const _SCREEN_RESOLUTION_PIXELS = Ref{Any}(missing)
const _FIGURE_CALLBACK_REFS = IdDict{Any, Any}()
const _MACOS_COREGRAPHICS = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"

struct _CoreGraphicsPoint
    x::Cdouble
    y::Cdouble
end

struct _CoreGraphicsSize
    width::Cdouble
    height::Cdouble
end

struct _CoreGraphicsRect
    origin::_CoreGraphicsPoint
    size::_CoreGraphicsSize
end

# Usage:
#   include("plot_ladder_mf_observables.jl")
#   plot_mf_profiles_from_file("stateless_data/results_...h5")
#   plot_middle_histories_from_file("stateless_data/results_...h5")
#   plot_mf_profiles_from_file("stateless_data/results_...h5"; source=:correlations)
#   plot_middle_histories_from_file("stateless_data/results_...h5"; source=:correlations)
#   plot_order_fourier_heatmaps_from_file("stateless_data/results_...h5"; source=:correlations)
#   plot_order_fourier_max_grid()
#   plot_mf_change_slider_from_file("stateless_data/results_...h5"; field=:alpha)
#   plot_mf_change_slider_from_file("stateless_data/results_...h5"; field=:beta, spin=:up)
# The file-based wrappers put the HDF5 basename in the PyPlot figure title.
# Fourier heatmap/grid routines trim 5 boundary rungs from each end by default;
# pass trim_boundary_rungs=0 to include the full ladder.
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

function _trimmed_rung_range(L::Integer, trim_boundary_rungs::Integer; min_rungs::Integer=2)
    trim = Int(trim_boundary_rungs)
    trim >= 0 || throw(ArgumentError("trim_boundary_rungs must be nonnegative; got $trim"))
    remaining = L - 2 * trim
    remaining >= min_rungs ||
        throw(ArgumentError("trim_boundary_rungs=$trim removes $(2 * trim) rungs from L=$L, leaving $(max(remaining, 0)); need at least $min_rungs rungs"))
    return (trim + 1):(L - trim)
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
_extended_swave_corr_symbol(rung_tex::AbstractString="i") = "\\Delta^{s_{\\mathrm{ext}}}_{" * rung_tex * "}"

function _dwave_corr_expr(; rung_tex::AbstractString="i", leg1_tex::AbstractString="1", leg2_tex::AbstractString="2")
    next_tex = _next_rung_tex(rung_tex)
    site_i1 = _site_tex(rung_tex, leg1_tex)
    site_i2 = _site_tex(rung_tex, leg2_tex)
    site_ip1 = _site_tex(next_tex, leg1_tex)
    site_ip2 = _site_tex(next_tex, leg2_tex)
    return _dwave_corr_symbol(rung_tex) * "=\\frac{1}{2}\\left(\\Delta^s_{" * site_i1 * "," * site_ip1 * "}+\\Delta^s_{" * site_i2 * "," * site_ip2 * "}\\right)-\\Delta^s_{" * site_i1 * "," * site_i2 * "}"
end

function _dwave_corr_title(leg1::Integer, leg2::Integer; rung_tex::AbstractString="i", include_singlet_def::Bool=true)
    expr = _math(_dwave_corr_expr(; rung_tex=rung_tex, leg1_tex=string(leg1), leg2_tex=string(leg2)))
    return include_singlet_def ? _math(_singlet_pair_corr_def()) * "\n" * expr : expr
end

function _extended_swave_corr_expr(; rung_tex::AbstractString="i", leg1_tex::AbstractString="1", leg2_tex::AbstractString="2")
    next_tex = _next_rung_tex(rung_tex)
    site_i1 = _site_tex(rung_tex, leg1_tex)
    site_i2 = _site_tex(rung_tex, leg2_tex)
    site_ip1 = _site_tex(next_tex, leg1_tex)
    site_ip2 = _site_tex(next_tex, leg2_tex)
    return _extended_swave_corr_symbol(rung_tex) * "=\\frac{1}{2}\\left(\\Delta^s_{" * site_i1 * "," * site_ip1 * "}+\\Delta^s_{" * site_i2 * "," * site_ip2 * "}\\right)+\\Delta^s_{" * site_i1 * "," * site_i2 * "}"
end

function _extended_swave_corr_title(leg1::Integer, leg2::Integer; rung_tex::AbstractString="i", include_singlet_def::Bool=true)
    expr = _math(_extended_swave_corr_expr(; rung_tex=rung_tex, leg1_tex=string(leg1), leg2_tex=string(leg2)))
    return include_singlet_def ? _math(_singlet_pair_corr_def()) * "\n" * expr : expr
end

_dwave_alpha_symbol(rung_tex::AbstractString="i") = "\\Delta^d_{\\alpha," * rung_tex * "}"
_extended_swave_alpha_symbol(rung_tex::AbstractString="i") = "\\Delta^{s_{\\mathrm{ext}}}_{\\alpha," * rung_tex * "}"

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

function _extended_swave_alpha_expr(; rung_tex::AbstractString="i", leg1_tex::AbstractString="1", leg2_tex::AbstractString="2", symmetrize::Bool=false)
    next_tex = _next_rung_tex(rung_tex)
    leg_term = "\\frac{1}{2}\\left(\\alpha_{" * rung_tex * "," * next_tex * "," * leg1_tex * "," * leg1_tex * "}+\\alpha_{" * rung_tex * "," * next_tex * "," * leg2_tex * "," * leg2_tex * "}\\right)"
    rung_term = if symmetrize
        "\\frac{1}{2}\\left(\\alpha_{" * rung_tex * "," * rung_tex * "," * leg1_tex * "," * leg2_tex * "}+\\alpha_{" * rung_tex * "," * rung_tex * "," * leg2_tex * "," * leg1_tex * "}\\right)"
    else
        "\\alpha_{" * rung_tex * "," * rung_tex * "," * leg1_tex * "," * leg2_tex * "}"
    end
    return _extended_swave_alpha_symbol(rung_tex) * "=" * leg_term * "+" * rung_term
end

function _extended_swave_alpha_title(leg1::Integer, leg2::Integer; rung_tex::AbstractString="i", symmetrize::Bool=false)
    return _math(_extended_swave_alpha_expr(; rung_tex=rung_tex, leg1_tex=string(leg1), leg2_tex=string(leg2), symmetrize=symmetrize))
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

function _local_extended_swave_alpha_value(a, i::Integer, leg1::Integer, leg2::Integer, symmetrize::Bool, it=nothing)
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
    return leg_pair + rung_pair
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

function extended_swave_profile_from_alpha(alpha;
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
    vals = [_local_extended_swave_alpha_value(a, i, leg1, leg2, symmetrize) for i in rungs]
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

function _local_extended_swave_corr_value(c, i::Integer, leg1::Integer, leg2::Integer)
    site_i1 = _plot_rung_leg_to_site(i, leg1)
    site_i2 = _plot_rung_leg_to_site(i, leg2)
    site_ip1 = _plot_rung_leg_to_site(i + 1, leg1)
    site_ip2 = _plot_rung_leg_to_site(i + 1, leg2)
    leg_pair = 0.5 * (
        _singlet_bond_from_pair_corr(c, site_i1, site_ip1) +
        _singlet_bond_from_pair_corr(c, site_i2, site_ip2)
    )
    rung_pair = _singlet_bond_from_pair_corr(c, site_i1, site_i2)
    return leg_pair + rung_pair
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

function extended_swave_profile_from_correlations(C_pair;
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
    vals = [_local_extended_swave_corr_value(c, i, leg1, leg2) for i in rungs]
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

function middle_extended_swave_history(alpha_list;
    rung=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize::Bool=false,
    use_abs::Bool=false)

    a = _history_array(alpha_list, 5, "alpha_list")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = size(a, 1)
    L > 1 || throw(ArgumentError("need at least two rungs to compute local extended s-wave profile"))
    rung = rung === nothing ? min(middle_rung(L), L - 1) : _check_index(rung, L - 1, "extended s-wave rung")
    iters = collect(1:size(a, 5))
    vals = [_local_extended_swave_alpha_value(a, rung, leg1, leg2, symmetrize, it) for it in iters]
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

function middle_extended_swave_history_from_correlations(C_pair_list;
    rung=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    use_abs::Bool=false)

    c = _history_array(C_pair_list, 3, "C_pair_list")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = _rung_count_from_sites(size(c, 1))
    size(c, 2) == 2 * L || throw(ArgumentError("raw pairing correlation matrices must be square"))
    L > 1 || throw(ArgumentError("need at least two rungs to compute local extended s-wave profile"))
    rung = rung === nothing ? min(middle_rung(L), L - 1) : _check_index(rung, L - 1, "extended s-wave rung")
    iters = collect(1:size(c, 3))
    vals = [_local_extended_swave_corr_value(selectdim(c, 3, it), rung, leg1, leg2) for it in iters]
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

function _momentum_integer_axis(n::Integer)
    shifted = _momentum_shift_indices(n)
    half = n / 2
    return [Int((idx - 1) < half ? (idx - 1) : (idx - 1 - n)) for idx in shifted]
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

_rung_index_xlabel() = "Rung index " * _math("i")
_mf_iteration_xlabel() = "MF iteration " * _math("m")
_kx_xlabel() = _math("k_x")

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

function _shared_log_norm(maps, value::Symbol; vmin::Real=1e-7)
    value in (:abs, :amplitude, :magnitude, :power, :intensity) ||
        throw(ArgumentError("logscale Fourier heatmaps require value=:abs or :power; use logscale=false for :real or :imag"))
    vmin > 0 || throw(ArgumentError("vmin must be positive for logscale Fourier heatmaps"))

    positive_vals = Float64[]
    for heatmap in maps
        append!(positive_vals, [Float64(x) for x in vec(heatmap) if isfinite(x) && x > 0])
    end

    vmin = Float64(vmin)
    vmax = isempty(positive_vals) ? one(Float64) : maximum(positive_vals)
    vmax > vmin || (vmax = 10 * vmin)
    return _MatplotlibColors.LogNorm(vmin=vmin, vmax=vmax), vmin
end

_floor_for_lognorm(heatmap, floor::Real) = map(x -> isfinite(x) && x > floor ? x : floor, heatmap)

const _DEFAULT_ORDER_GRID_CMAPS = (cdw="Reds", sdw="Greys", swave="Greens", dwave="Purples")

function _result_filename_parameters(filename::AbstractString)
    m = match(r"_L_([0-9]+)_U_([-+0-9.eE]+)_V_([-+0-9.eE]+)_t0_([-+0-9.eE]+)_t_p_([-+0-9.eE]+).*_density_([-+0-9.eE]+)_", basename(filename))
    m === nothing && return nothing
    return (
        filename=filename,
        L=parse(Int, m.captures[1]),
        U=parse(Float64, m.captures[2]),
        V0=parse(Float64, m.captures[3]),
        t0=parse(Float64, m.captures[4]),
        tp=parse(Float64, m.captures[5]),
        density=parse(Float64, m.captures[6]),
    )
end

function _sorted_unique_floats(vals)
    return sort(unique(Float64.(collect(vals))))
end

function _nearest_grid_value(x::Real, vals; atol::Real=1e-8)
    for val in vals
        isapprox(Float64(x), Float64(val); atol=atol, rtol=0) && return Float64(val)
    end
    return nothing
end

function _grid_edges(vals; default_step::Real=0.2)
    xs = _sorted_unique_floats(vals)
    isempty(xs) && throw(ArgumentError("grid needs at least one coordinate"))
    if length(xs) == 1
        step = Float64(default_step)
        step > 0 || throw(ArgumentError("default grid step must be positive"))
        return [xs[1] - step / 2, xs[1] + step / 2]
    end

    mids = [(xs[i] + xs[i + 1]) / 2 for i in 1:(length(xs) - 1)]
    edges = Vector{Float64}(undef, length(xs) + 1)
    edges[1] = xs[1] - (mids[1] - xs[1])
    for i in 2:length(xs)
        edges[i] = mids[i - 1]
    end
    edges[end] = xs[end] + (xs[end] - mids[end])
    return edges
end

function _order_fields_from_file(filename::AbstractString;
    source::Symbol=:mf,
    use_correlations::Bool=false,
    iteration=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize_rung::Bool=false,
    trim_boundary_rungs::Integer=0)

    data = load_mf_data(filename)
    if _source(source, use_correlations) == :correlations
        C_pair_source = (iteration === nothing && data.C_pair !== nothing) ? data.C_pair : data.C_pair_list
        C_exc_dn_source = (iteration === nothing && data.C_exc_dn !== nothing) ? data.C_exc_dn : data.C_exc_dn_list
        C_exc_up_source = (iteration === nothing && data.C_exc_up !== nothing) ? data.C_exc_up : data.C_exc_up_list
        return _correlation_order_fields(C_pair_source, C_exc_dn_source, C_exc_up_source; iteration=iteration, leg1=leg1, leg2=leg2, trim_boundary_rungs=trim_boundary_rungs)
    end

    alpha_source = (iteration === nothing && data.alpha !== nothing) ? data.alpha : data.alpha_list
    beta_source = (iteration === nothing && data.beta !== nothing) ? data.beta : data.beta_list
    return _mf_order_fields(alpha_source, beta_source; iteration=iteration, leg1=leg1, leg2=leg2, symmetrize_rung=symmetrize_rung, trim_boundary_rungs=trim_boundary_rungs)
end

function _max_fourier_component(heatmap, qx; ladder::Bool)
    best_value = -Inf
    best_index = nothing
    for I in CartesianIndices(heatmap)
        val = Float64(heatmap[I])
        if isfinite(val) && val > best_value
            best_value = val
            best_index = I
        end
    end

    best_index === nothing && return (
        value=NaN,
        kx=NaN,
        ky=ladder ? NaN : nothing,
        kx_multiple=0,
        kx_period=0,
        ky_multiple=ladder ? 0 : nothing,
        ky_period=ladder ? 2 : nothing,
        row=0,
        col=0,
    )
    row = best_index[1]
    col = best_index[2]
    kx_period = length(qx)
    kx_multiple = _momentum_integer_axis(kx_period)[col]
    return (
        value=best_value,
        kx=Float64(qx[col]),
        ky=ladder ? (row == 1 ? 0.0 : π) : nothing,
        kx_multiple=kx_multiple,
        kx_period=kx_period,
        ky_multiple=ladder ? row - 1 : nothing,
        ky_period=ladder ? 2 : nothing,
        row=row,
        col=col,
    )
end

function _order_fourier_maxima(fields; value::Symbol=:abs, normalize::Bool=true)
    qx_cdw, cdw_map = _ladder_fourier_map(fields.cdw; subtract_average=true, normalize=normalize, value=value)
    qx_sdw, sdw_map = _ladder_fourier_map(fields.sdw; normalize=normalize, value=value)
    qx_swave, swave_map = _ladder_fourier_map(fields.swave; normalize=normalize, value=value)
    qx_extended_swave, extended_swave_map = _chain_fourier_map(fields.extended_swave; normalize=normalize, value=value)
    qx_dwave, dwave_map = _chain_fourier_map(fields.dwave; normalize=normalize, value=value)

    return (
        cdw=_max_fourier_component(cdw_map, qx_cdw; ladder=true),
        sdw=_max_fourier_component(sdw_map, qx_sdw; ladder=true),
        swave=_max_fourier_component(swave_map, qx_swave; ladder=true),
        extended_swave=_max_fourier_component(extended_swave_map, qx_extended_swave; ladder=false),
        dwave=_max_fourier_component(dwave_map, qx_dwave; ladder=false),
    )
end

function _fourier_grid_scale(vals, value::Symbol; logscale::Bool=true, vmin::Real=1e-7)
    finite_vals = [Float64(x) for x in vals if isfinite(Float64(x))]
    if logscale
        value in (:abs, :amplitude, :magnitude, :power, :intensity) ||
            throw(ArgumentError("logscale Fourier grids require value=:abs or :power; use logscale=false for :real or :imag"))
        floor = Float64(vmin)
        floor > 0 || throw(ArgumentError("vmin must be positive for logscale Fourier grids"))
        positive_vals = [x for x in finite_vals if x > 0]
        vmax = isempty(positive_vals) ? one(Float64) : maximum(positive_vals)
        vmax > floor || (vmax = 10 * floor)
        norm = _MatplotlibColors.LogNorm(vmin=floor, vmax=vmax)
        return (norm=norm, logscale=true, vmin=floor, vmax=vmax, floor=floor)
    end

    lo = isempty(finite_vals) ? 0.0 : minimum(finite_vals)
    hi = isempty(finite_vals) ? 1.0 : maximum(finite_vals)
    hi > lo || (hi = lo + one(Float64))
    norm = _MatplotlibColors.Normalize(vmin=lo, vmax=hi)
    return (norm=norm, logscale=false, vmin=lo, vmax=hi, floor=lo)
end

function _order_grid_cmap(cmap)
    return cmap isa AbstractString ? _MatplotlibCm.get_cmap(cmap) : cmap
end

function _order_grid_facecolor(cmap, val::Real, scale)
    x = Float64(val)
    isfinite(x) || return (0.90, 0.90, 0.90, 1.0)
    scale.logscale && (x = max(x, scale.floor))
    mapped = clamp(Float64(scale.norm(x)), 0.0, 1.0)
    return Tuple(cmap(mapped))
end

function _format_grid_number(x::Real)
    y = Float64(x)
    abs(y) < 5e-13 && (y = 0.0)
    return string(round(y; digits=6))
end

function _format_discrete_momentum(m::Integer, period::Integer)
    period > 0 || return "NaN"
    numerator = 2 * m
    denominator = period
    numerator == 0 && return "0"
    factor = gcd(abs(numerator), denominator)
    numerator = div(numerator, factor)
    denominator = div(denominator, factor)

    if denominator == 1
        numerator == 1 && return "π"
        numerator == -1 && return "-π"
        return string(numerator, "π")
    end

    numerator == 1 && return "π/$denominator"
    numerator == -1 && return "-π/$denominator"
    return string(numerator, "π/", denominator)
end

function _order_grid_legend_label(key::Symbol, label::AbstractString; leg1::Integer=1, leg2::Integer=2)
    if key == :cdw
        return label * "  " * _math("O^{\\mathrm{CDW}}_{i,\\ell}=n_{i,\\ell,\\uparrow}+n_{i,\\ell,\\downarrow}")
    elseif key == :sdw
        return label * "  " * _math("O^{\\mathrm{SDW}}_{i,\\ell}=n_{i,\\ell,\\uparrow}-n_{i,\\ell,\\downarrow}")
    elseif key == :swave
        return label * "  " * _math("O^{s}_{i,\\ell}=\\langle c_{i,\\ell,\\uparrow}c_{i,\\ell,\\downarrow}\\rangle")
    elseif key == :extended_swave
        l1 = string(_check_leg(leg1))
        l2 = string(_check_leg(leg2))
        return label * "  " * _math("O^{s_{\\mathrm{ext}}}_i=\\frac{1}{2}(\\Delta^s_{i" * l1 * ",i+1\\," * l1 * "}+\\Delta^s_{i" * l2 * ",i+1\\," * l2 * "})+\\Delta^s_{i" * l1 * ",i" * l2 * "}")
    elseif key == :dwave
        l1 = string(_check_leg(leg1))
        l2 = string(_check_leg(leg2))
        return label * "  " * _math("O^d_i=\\frac{1}{2}(\\Delta^s_{i" * l1 * ",i+1\\," * l1 * "}+\\Delta^s_{i" * l2 * ",i+1\\," * l2 * "})-\\Delta^s_{i" * l1 * ",i" * l2 * "}")
    end
    return label
end

function _order_grid_legend_marker(marker; facecolor, edgecolor="black", linewidth::Real=1.4)
    return _MatplotlibLines.Line2D([0], [0];
        linestyle="None",
        marker=marker,
        markersize=_ORDER_GRID_LEGEND_MARKER_SIZE,
        markerfacecolor=facecolor,
        markeredgecolor=edgecolor,
        markeredgewidth=linewidth)
end

_order_grid_lower_left_triangle_marker() = _MatplotlibPath.Path([
    (-0.5, -0.5),
    (0.5, -0.5),
    (-0.5, 0.5),
    (-0.5, -0.5),
])

_order_grid_upper_right_triangle_marker() = _MatplotlibPath.Path([
    (0.5, 0.5),
    (0.5, -0.5),
    (-0.5, 0.5),
    (0.5, 0.5),
])

function _order_grid_legend_handle(spec)
    facecolor = Tuple(spec.cmap(0.75))
    if spec.shape == :onsite_swave_triangle
        return (
            _order_grid_legend_marker(_order_grid_lower_left_triangle_marker(); facecolor=facecolor),
            _order_grid_legend_marker("s"; facecolor="none"),
        )
    elseif spec.shape == :extended_swave_triangle
        return (
            _order_grid_legend_marker(_order_grid_upper_right_triangle_marker(); facecolor=facecolor),
            _order_grid_legend_marker("s"; facecolor="none"),
        )
    end
    return _order_grid_legend_marker("s"; facecolor=facecolor)
end

function _point_in_triangle(x::Real, y::Real, vertices)
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    x3, y3 = vertices[3]
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    abs(denom) <= eps(Float64) && return false
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1 - a - b
    tol = 1e-12
    return a >= -tol && b >= -tol && c >= -tol
end

function _order_grid_region_contains(region, x::Real, y::Real)
    region.x0 <= x <= region.x1 && region.y0 <= y <= region.y1 || return false
    if :vertices in propertynames(region)
        return _point_in_triangle(x, y, region.vertices)
    end
    return true
end

function _order_grid_hover_text(label::AbstractString, rec, summary)
    momentum = summary.ky === nothing ?
        _math("k_x=" * _format_discrete_momentum(summary.kx_multiple, summary.kx_period)) :
        _math("k_x=" * _format_discrete_momentum(summary.kx_multiple, summary.kx_period)) *
        ", " * _math("k_y=" * _format_discrete_momentum(summary.ky_multiple, summary.ky_period))
    return label * "\n" *
        _math("V_0=" * _format_grid_number(rec.V0)) * ", " * _math("t_0=" * _format_grid_number(rec.t0)) * "\n" *
        "max=" * string(round(summary.value; sigdigits=5)) * "\n" *
        momentum
end

function _align_order_grid_hover_annotation!(fig, annotation, right_artist, bottom_artist)
    (right_artist === nothing || bottom_artist === nothing) && return nothing

    try
        renderer = fig.canvas.get_renderer()
        target_right = Float64(right_artist.get_window_extent(renderer).x1)
        target_bottom = Float64(bottom_artist.get_window_extent(renderer).y0)
        annotation.update_bbox_position_size(renderer)
        annotation_box = annotation.get_bbox_patch().get_window_extent(renderer)
        fig_box = fig.bbox
        current_position = collect(annotation.get_position())
        dx = (target_right - Float64(annotation_box.x1)) / Float64(fig_box.width)
        dy = (target_bottom - Float64(annotation_box.y0)) / Float64(fig_box.height)
        annotation.set_position((Float64(current_position[1]) + dx, Float64(current_position[2]) + dy))
    catch
        return nothing
    end

    return nothing
end

function _install_order_grid_hover!(fig, ax, regions; position=(0.97, 0.04), right_artist=nothing, bottom_artist=nothing)
    annotation = fig.text(position[1], position[2], "";
        transform=fig.transFigure,
        ha="right",
        va="bottom",
        visible=false,
        clip_on=false,
        zorder=20,
        bbox=Dict("boxstyle" => "round,pad=0.35", "fc" => "white", "ec" => "0.25", "alpha" => 0.95))
    active_region = Ref(0)

    function clear_hover()
        if active_region[] != 0 || annotation.get_visible()
            active_region[] = 0
            annotation.set_visible(false)
            fig.canvas.draw_idle()
        end
        return nothing
    end

    function hover(event)
        if event.inaxes != ax || event.xdata === nothing || event.ydata === nothing
            return clear_hover()
        end

        x = Float64(event.xdata)
        y = Float64(event.ydata)
        hit_index = 0
        hit_text = ""
        for (idx, region) in enumerate(regions)
            if _order_grid_region_contains(region, x, y)
                hit_index = idx
                hit_text = region.text
                break
            end
        end

        if hit_index == 0
            return clear_hover()
        elseif hit_index == active_region[] && annotation.get_visible()
            return nothing
        end

        active_region[] = hit_index
        annotation.set_text(hit_text)
        annotation.set_visible(true)
        _align_order_grid_hover_annotation!(fig, annotation, right_artist, bottom_artist)
        fig.canvas.draw_idle()
        return nothing
    end

    callback_id = fig.canvas.mpl_connect("motion_notify_event", hover)
    return annotation, hover, callback_id
end

function _install_order_grid_click!(fig, ax, regions; iteration=nothing)
    spawned_heatmap_figures = []
    spawned_detail_figures = []

    function click(event)
        if event.inaxes != ax || event.xdata === nothing || event.ydata === nothing
            return nothing
        end

        x = Float64(event.xdata)
        y = Float64(event.ydata)
        for region in regions
            if region.x0 <= x <= region.x1 && region.y0 <= y <= region.y1
                heatmap_fig = plot_order_fourier_heatmaps_from_file(region.filename; use_correlations=true, iteration=iteration, trim_boundary_rungs=region.trim_boundary_rungs)
                detail_fig = plot_mf_profiles_and_middle_histories_from_file(region.filename; use_correlations=true, iteration=iteration)
                push!(spawned_heatmap_figures, heatmap_fig)
                push!(spawned_detail_figures, detail_fig)
                heatmap_fig.canvas.draw_idle()
                detail_fig.canvas.draw_idle()
                return nothing
            end
        end

        return nothing
    end

    callback_id = fig.canvas.mpl_connect("button_press_event", click)
    return click, callback_id, spawned_heatmap_figures, spawned_detail_figures
end

function _position_fixed_colorbar_axes!(fig, ax, colorbar_axes;
    pad_inches::Real=0.35,
    width_inches::Real=0.17,
    gap_inches::Real=0.29,
    vertical_shift_inches::Real=0.40)

    fig_size = collect(fig.get_size_inches())
    fig_width = Float64(fig_size[1])
    fig_height = Float64(fig_size[2])
    fig_width > 0 && fig_height > 0 || return nothing

    bbox = ax.get_position()
    pad = Float64(pad_inches) / fig_width
    width = Float64(width_inches) / fig_width
    gap = Float64(gap_inches) / fig_width
    vertical_shift = Float64(vertical_shift_inches) / fig_height

    left = Float64(bbox.x1) + pad
    bottom = Float64(bbox.y0) - vertical_shift
    height = Float64(bbox.y1) - Float64(bbox.y0)
    for (idx, cax) in enumerate(colorbar_axes)
        cax.set_position([left + (idx - 1) * (width + gap), bottom, width, height])
    end
    return nothing
end

function _install_fixed_colorbar_layout!(fig, ax, colorbar_axes; kwargs...)
    _position_fixed_colorbar_axes!(fig, ax, colorbar_axes; kwargs...)

    function relayout(_event)
        _position_fixed_colorbar_axes!(fig, ax, colorbar_axes; kwargs...)
        fig.canvas.draw_idle()
        return nothing
    end

    callback_id = fig.canvas.mpl_connect("resize_event", relayout)
    return relayout, callback_id
end

function _mf_order_fields(alpha, beta;
    iteration=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize_rung::Bool=false,
    trim_boundary_rungs::Integer=0)

    a = _final_array(alpha, 4, iteration, "alpha")
    b = _final_array(beta, 5, iteration, "beta")
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = size(b, 2)
    size(a, 1) == L || throw(ArgumentError("alpha and beta must have the same rung count"))
    size(a, 2) == L && size(b, 3) == L || throw(ArgumentError("alpha and beta rung dimensions must be square"))
    rungs = _trimmed_rung_range(L, trim_boundary_rungs)
    if first(rungs) != 1 || last(rungs) != L
        a = a[rungs, rungs, :, :]
        b = b[:, rungs, rungs, :, :]
        L = length(rungs)
    end
    L > 1 || throw(ArgumentError("need at least two rungs to compute local d-wave profile"))

    cdw = [b[2, i, i, leg, leg] + b[1, i, i, leg, leg] for i in 1:L, leg in 1:2]
    sdw = [b[2, i, i, leg, leg] - b[1, i, i, leg, leg] for i in 1:L, leg in 1:2]
    swave = [a[i, i, leg, leg] for i in 1:L, leg in 1:2]
    extended_swave = [_local_extended_swave_alpha_value(a, i, leg1, leg2, symmetrize_rung) for i in 1:(L - 1)]
    dwave = [_local_dwave_alpha_value(a, i, leg1, leg2, symmetrize_rung) for i in 1:(L - 1)]
    return (cdw=cdw, sdw=sdw, swave=swave, extended_swave=extended_swave, dwave=dwave)
end

function _correlation_order_fields(C_pair, C_exc_dn, C_exc_up;
    iteration=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    trim_boundary_rungs::Integer=0)

    cpair = _final_array(C_pair, 2, iteration, "C_pair_list")
    cdn, cup = _normal_corr_pair(C_exc_dn, C_exc_up, iteration)
    leg1 = _check_leg(leg1)
    leg2 = _check_leg(leg2)
    L = _rung_count_from_sites(size(cpair, 1))
    size(cpair, 2) == 2 * L || throw(ArgumentError("raw pairing correlation matrix must be square"))
    size(cdn) == size(cup) == size(cpair) || throw(ArgumentError("raw correlation matrices must have the same shape"))
    rungs = _trimmed_rung_range(L, trim_boundary_rungs)
    if first(rungs) != 1 || last(rungs) != L
        sites = [_plot_rung_leg_to_site(i, leg) for i in rungs for leg in 1:2]
        cpair = cpair[sites, sites]
        cdn = cdn[sites, sites]
        cup = cup[sites, sites]
        L = length(rungs)
    end
    L > 1 || throw(ArgumentError("need at least two rungs to compute local d-wave profile"))

    cdw = [cup[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] +
           cdn[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] for i in 1:L, leg in 1:2]
    sdw = [cup[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] -
           cdn[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] for i in 1:L, leg in 1:2]
    swave = [cpair[_plot_rung_leg_to_site(i, leg), _plot_rung_leg_to_site(i, leg)] for i in 1:L, leg in 1:2]
    extended_swave = [_local_extended_swave_corr_value(cpair, i, leg1, leg2) for i in 1:(L - 1)]
    dwave = [_local_dwave_corr_value(cpair, i, leg1, leg2) for i in 1:(L - 1)]
    return (cdw=cdw, sdw=sdw, swave=swave, extended_swave=extended_swave, dwave=dwave)
end

function _plot_fourier_panel!(fig, ax, heatmap;
    title::AbstractString,
    ladder::Bool,
    cmap::AbstractString,
    norm=nothing,
    xlabel::AbstractString=_kx_xlabel(),
    kwargs...)

    ncols = size(heatmap, 2)
    zero_tick = div(ncols, 2)
    img = if norm === nothing
        ax.imshow(heatmap; origin="lower", aspect="equal", interpolation="nearest", cmap=cmap, kwargs...)
    else
        ax.imshow(heatmap; origin="lower", aspect="equal", interpolation="nearest", cmap=cmap, norm=norm, kwargs...)
    end
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(-0.5, size(heatmap, 1) - 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ladder ? _math("k_y") : "")
    ax.set_xticks([0, zero_tick, ncols - 1])
    ax.set_xticklabels([_math("-\\pi"), "0", _math("\\pi")])
    ax.set_yticks(ladder ? [0, 1] : [0])
    ax.set_yticklabels(ladder ? ["0", _math("\\pi")] : ["chain"])
    ax.set_title(title)
    return img
end

function _plot_order_fourier_heatmaps(fields;
    value::Symbol=:abs,
    normalize::Bool=true,
    cmap::AbstractString="inferno",
    logscale::Bool=true,
    vmin::Real=1e-7,
    figsize=(9.0, 8.8),
    savepath=nothing,
    figure_title=nothing,
    source_label::AbstractString="",
    kwargs...)

    _, cdw_map = _ladder_fourier_map(fields.cdw; subtract_average=true, normalize=normalize, value=value)
    _, sdw_map = _ladder_fourier_map(fields.sdw; normalize=normalize, value=value)
    _, swave_map = _ladder_fourier_map(fields.swave; normalize=normalize, value=value)
    _, extended_swave_map = _chain_fourier_map(fields.extended_swave; normalize=normalize, value=value)
    _, dwave_map = _chain_fourier_map(fields.dwave; normalize=normalize, value=value)
    norm = nothing
    if logscale
        norm, log_floor = _shared_log_norm((cdw_map, sdw_map, swave_map, extended_swave_map, dwave_map), value; vmin=vmin)
        cdw_map = _floor_for_lognorm(cdw_map, log_floor)
        sdw_map = _floor_for_lognorm(sdw_map, log_floor)
        swave_map = _floor_for_lognorm(swave_map, log_floor)
        extended_swave_map = _floor_for_lognorm(extended_swave_map, log_floor)
        dwave_map = _floor_for_lognorm(dwave_map, log_floor)
    end

    fig, axes = PyPlot.subplots(5, 1, figsize=_screen_figsize(figsize))
    axs = vec(axes)
    cbar_label = _fourier_colorbar_label(value)
    imgs = [
        _plot_fourier_panel!(fig, axs[1], cdw_map; title="CDW, mean subtracted", ladder=true, cmap=cmap, norm=norm, xlabel="", kwargs...),
        _plot_fourier_panel!(fig, axs[2], sdw_map; title="SDW", ladder=true, cmap=cmap, norm=norm, xlabel="", kwargs...),
        _plot_fourier_panel!(fig, axs[3], swave_map; title="on-site s-wave", ladder=true, cmap=cmap, norm=norm, xlabel="", kwargs...),
        _plot_fourier_panel!(fig, axs[4], extended_swave_map; title="extended s-wave local profile", ladder=false, cmap=cmap, norm=norm, xlabel="", kwargs...),
        _plot_fourier_panel!(fig, axs[5], dwave_map; title="d-wave local profile", ladder=false, cmap=cmap, norm=norm, kwargs...),
    ]
    if figure_title !== nothing
        fig.suptitle(figure_title)
    end
    top = figure_title === nothing ? 0.96 : 0.92
    fig.subplots_adjust(left=0.11, right=0.84, bottom=0.08, top=top, hspace=_FOURIER_STACK_HSPACE)
    cax = fig.add_axes([0.88, 0.12, 0.025, top - 0.16])
    cbar = fig.colorbar(imgs[1], cax=cax)
    cbar.set_label(cbar_label)
    return _save_if_requested(fig, savepath)
end

function _screen_resolution_pixels()
    cached = _SCREEN_RESOLUTION_PIXELS[]
    cached !== missing && return cached

    resolution = nothing
    if Sys.isapple()
        try
            display = ccall((:CGMainDisplayID, _MACOS_COREGRAPHICS), UInt32, ())
            bounds = ccall((:CGDisplayBounds, _MACOS_COREGRAPHICS), _CoreGraphicsRect, (UInt32,), display)
            width = Int(round(bounds.size.width))
            height = Int(round(bounds.size.height))
            width > 0 && height > 0 && (resolution = (width, height))
        catch
        end
    end

    _SCREEN_RESOLUTION_PIXELS[] = resolution
    return resolution
end

function _matplotlib_figure_dpi()
    try
        dpi = Float64(PyPlot.matplotlib.rcParams["figure.dpi"])
        dpi > 0 && return dpi
    catch
    end
    return 100.0
end

function _screen_figsize(fallback_figsize)
    resolution = _screen_resolution_pixels()
    resolution === nothing && return fallback_figsize
    dpi = _matplotlib_figure_dpi()
    return (Float64(resolution[1]) / dpi, Float64(resolution[2]) / dpi)
end

function _save_if_requested(fig, savepath)
    savepath !== nothing && fig.savefig(savepath, bbox_inches="tight")
    return fig
end

function _plot_series!(ax, x, y; xlabel, ylabel, title, kwargs...)
    ax.plot(x, y; marker="o", markersize=4, linewidth=1.5, kwargs...)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(true, alpha=0.3)
    return ax
end

_other_leg(leg::Integer) = _check_leg(leg) == 1 ? 2 : 1

function _leg_color(colors, leg::Integer)
    leg = _check_leg(leg)
    return colors[leg]
end

_cdw_leg_color(leg::Integer) = _leg_color(_CDW_LEG_COLORS, leg)
_sdw_leg_color(leg::Integer) = _leg_color(_SDW_LEG_COLORS, leg)
_swave_leg_color(leg::Integer) = _leg_color(_SWAVE_LEG_COLORS, leg)

function _place_right_legend!(ax)
    return ax.legend(loc="center right", borderaxespad=0.4)
end

function _plot_other_leg!(ax, x, y; leg::Integer, color)
    ax.plot(x, y;
        marker="o",
        markersize=4,
        linewidth=1.5,
        color=color,
        label="leg $leg")
    _place_right_legend!(ax)
    return ax
end

function _plot_leg_pair!(ax, x, y, x_other, y_other;
    leg::Integer,
    other_leg::Integer,
    show_other_leg::Bool,
    color_fn,
    xlabel,
    ylabel,
    title,
    kwargs...)

    _plot_series!(ax, x, y;
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        label="leg $leg",
        color=color_fn(leg),
        kwargs...)
    main_line = ax.lines[end]
    other_line = nothing
    if show_other_leg
        _plot_other_leg!(ax, x_other, y_other; leg=other_leg, color=color_fn(other_leg))
        other_line = ax.lines[end]
    end
    return (main=main_line, other=other_line)
end

function _profile_iteration_count(specs...)
    counts = Int[]
    for spec in specs
        A, final_ndims, name = spec
        A === nothing && return nothing
        if ndims(A) == final_ndims + 1
            push!(counts, size(A, final_ndims + 1))
        elseif ndims(A) == final_ndims
            return nothing
        else
            throw(ArgumentError("$name must have $final_ndims dimensions, or $(final_ndims + 1) with MF iteration last; got $(ndims(A))"))
        end
    end
    isempty(counts) && return nothing
    all(count -> count == first(counts), counts) ||
        throw(ArgumentError("profile history arrays must have the same MF iteration count; got $counts"))
    return first(counts)
end

_initial_profile_iteration(iteration, niter) =
    niter === nothing ? iteration : (iteration === nothing ? niter : _check_index(iteration, niter, "iteration"))

function _set_line_data!(line, x, y)
    line.set_data(x, y)
    return nothing
end

function _add_iteration_marker!(ax, iteration::Integer)
    return ax.axvline(iteration; color="0.25", linestyle=":", linewidth=1.3, alpha=0.85)
end

function _set_iteration_marker!(line, iteration::Integer)
    line.set_xdata([iteration, iteration])
    return nothing
end

function _include_finite_bounds(lo::Float64, hi::Float64, vals)
    for val in vals
        x = Float64(val)
        if isfinite(x)
            lo = min(lo, x)
            hi = max(hi, x)
        end
    end
    return lo, hi
end

function _padded_bounds(lo::Float64, hi::Float64; fraction::Real=0.05, fallback_span::Real=1.0)
    if !isfinite(lo) || !isfinite(hi)
        return (0.0, Float64(fallback_span))
    end

    span = hi - lo
    if span <= 0
        center = lo
        span = max(abs(center), Float64(fallback_span))
    end

    pad = max(Float64(fraction) * span, eps(Float64))
    return lo - pad, hi + pad
end

function _set_profile_axis_limits!(ax, profile_history, xfield::Symbol, yfield::Symbol;
    other_xfield=nothing,
    other_yfield=nothing,
    show_other_leg::Bool=false)

    xmin, xmax = Inf, -Inf
    ymin, ymax = Inf, -Inf
    for data in profile_history
        xmin, xmax = _include_finite_bounds(xmin, xmax, getproperty(data, xfield))
        ymin, ymax = _include_finite_bounds(ymin, ymax, getproperty(data, yfield))
        if show_other_leg && other_xfield !== nothing && other_yfield !== nothing
            xmin, xmax = _include_finite_bounds(xmin, xmax, getproperty(data, other_xfield))
            ymin, ymax = _include_finite_bounds(ymin, ymax, getproperty(data, other_yfield))
        end
    end

    xlo, xhi = _padded_bounds(xmin, xmax; fraction=0.02, fallback_span=1.0)
    ylo, yhi = _padded_bounds(ymin, ymax; fraction=0.08, fallback_span=1.0)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    return nothing
end

function _profile_tight_rect(figure_title, has_slider::Bool)
    if has_slider
        return figure_title === nothing ? (0, 0.12, 1, 1) : (0, 0.12, 1, 0.95)
    end
    return figure_title === nothing ? nothing : (0, 0, 1, 0.95)
end

function _axes_union_bbox(target_axes)
    axes = collect(target_axes)
    isempty(axes) && return nothing

    bbox = axes[1].get_position()
    x0 = Float64(bbox.x0)
    y0 = Float64(bbox.y0)
    x1 = Float64(bbox.x1)
    y1 = Float64(bbox.y1)
    for ax in axes[2:end]
        bbox = ax.get_position()
        x0 = min(x0, Float64(bbox.x0))
        y0 = min(y0, Float64(bbox.y0))
        x1 = max(x1, Float64(bbox.x1))
        y1 = max(y1, Float64(bbox.y1))
    end
    return (x0=x0, y0=y0, x1=x1, y1=y1, width=x1 - x0, height=y1 - y0)
end

function _bottom_left_slider_rect(fig, target_axes)
    bbox = target_axes === nothing ? nothing : _axes_union_bbox(target_axes)
    bbox === nothing && return [0.12, 0.04, 0.42, 0.025]

    y = max(0.035, bbox.y0 - 0.075)
    return [bbox.x0, y, bbox.width, 0.025]
end

function _install_iteration_slider!(fig, niter::Integer, initial_iteration::Integer, update; figure_title=nothing, target_axes=nothing)
    slider_ax = fig.add_axes(_bottom_left_slider_rect(fig, target_axes))
    slider = _MatplotlibWidgets.Slider(slider_ax, "MF iteration", 1, niter; valinit=initial_iteration, valstep=1, valfmt="%0.0f")
    last_iteration = Ref(initial_iteration)

    function on_changed(val)
        it = _check_index(Int(round(val)), niter, "iteration")
        it == last_iteration[] && return nothing
        last_iteration[] = it
        update(it)
        fig.canvas.draw_idle()
        return nothing
    end

    callback_id = slider.on_changed(on_changed)
    refs = (slider=slider, callback=on_changed, callback_id=callback_id, last_iteration=last_iteration)
    _FIGURE_CALLBACK_REFS[fig] = refs
    return refs
end

function _finish_figure(fig; figure_title=nothing, tight_rect=nothing, h_pad=nothing)
    if figure_title !== nothing
        fig.suptitle(figure_title)
    end
    if tight_rect === nothing && h_pad === nothing
        fig.tight_layout()
    elseif tight_rect === nothing
        fig.tight_layout(h_pad=h_pad)
    elseif h_pad === nothing
        fig.tight_layout(rect=tight_rect)
    else
        fig.tight_layout(rect=tight_rect, h_pad=h_pad)
    end
    return fig
end

_filename_title(filename::AbstractString) = basename(filename)

function _figure(figsize)
    fig, ax = PyPlot.subplots(figsize=_screen_figsize(figsize))
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
    fig, ax = PyPlot.subplots(figsize=_screen_figsize(figsize))
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

function plot_extended_swave_from_alpha(alpha;
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    symmetrize::Bool=false,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    rungs, vals = extended_swave_profile_from_alpha(alpha; leg1=leg1, leg2=leg2, iteration=iteration, symmetrize=symmetrize, use_abs=use_abs)
    ylabel = use_abs ? "|extended s-wave alpha|" : "extended s-wave alpha"
    fig, ax = _figure((8.5, 4.0))
    _plot_series!(ax, rungs, vals;
        xlabel="Rung index",
        ylabel=ylabel,
        title="extended s-wave alpha proxy: " * _extended_swave_alpha_title(leg1, leg2; symmetrize=symmetrize),
        color=_EXT_SWAVE_COLOR,
        kwargs...)
    _finish_figure(fig; figure_title=figure_title)
    return _save_if_requested(fig, savepath)
end

function plot_extended_swave_from_correlations(C_pair;
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    rungs, vals = extended_swave_profile_from_correlations(C_pair; leg1=leg1, leg2=leg2, iteration=iteration, use_abs=use_abs)
    fig, ax = _figure((8.5, 4.0))
    _plot_series!(ax, rungs, vals;
        xlabel="Rung index \$i\$",
        ylabel=_corr_value_label(_extended_swave_corr_symbol(), use_abs),
        title="extended s-wave: " * _extended_swave_corr_title(leg1, leg2),
        color=_EXT_SWAVE_COLOR,
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
        color=_DWAVE_COLOR,
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
        color=_DWAVE_COLOR,
        kwargs...)
    _finish_figure(fig; figure_title=figure_title)
    return _save_if_requested(fig, savepath)
end

function plot_order_fourier_heatmaps(alpha, beta;
    iteration=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize_rung::Bool=false,
    trim_boundary_rungs::Integer=5,
    value::Symbol=:abs,
    normalize::Bool=true,
    cmap::AbstractString="inferno",
    logscale::Bool=true,
    vmin::Real=1e-7,
    figsize=(9.0, 8.8),
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    fields = _mf_order_fields(alpha, beta; iteration=iteration, leg1=leg1, leg2=leg2, symmetrize_rung=symmetrize_rung, trim_boundary_rungs=trim_boundary_rungs)
    return _plot_order_fourier_heatmaps(fields; value=value, normalize=normalize, cmap=cmap, logscale=logscale, vmin=vmin, figsize=figsize, savepath=savepath, figure_title=figure_title, source_label="alpha/beta", kwargs...)
end

function plot_order_fourier_heatmaps_from_correlations(C_pair, C_exc_dn, C_exc_up;
    iteration=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    trim_boundary_rungs::Integer=5,
    value::Symbol=:abs,
    normalize::Bool=true,
    cmap::AbstractString="inferno",
    logscale::Bool=true,
    vmin::Real=1e-7,
    figsize=(9.0, 7.2),
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    fields = _correlation_order_fields(C_pair, C_exc_dn, C_exc_up; iteration=iteration, leg1=leg1, leg2=leg2, trim_boundary_rungs=trim_boundary_rungs)
    return _plot_order_fourier_heatmaps(fields; value=value, normalize=normalize, cmap=cmap, logscale=logscale, vmin=vmin, figsize=figsize, savepath=savepath, figure_title=figure_title, source_label="correlations", kwargs...)
end

function plot_order_fourier_max_grid(;
    data_dir::AbstractString="stateless_data",
    suffix::AbstractString="_nodamping.h5",
    t0_values=0.8:0.2:1.6,
    t0_min::Real=0.8,
    t0_max::Real=1.6,
    t0_atol::Real=1e-8,
    source::Symbol=:mf,
    use_correlations::Bool=false,
    iteration=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize_rung::Bool=false,
    trim_boundary_rungs::Integer=5,
    value::Symbol=:abs,
    normalize::Bool=true,
    logscale::Bool=true,
    vmin::Real=1e-4,
    order_cmaps=_DEFAULT_ORDER_GRID_CMAPS,
    figsize=(9.5, 6.8),
    savepath=nothing,
    figure_title=nothing,
    hover::Bool=true,
    click_detail_plots::Bool=true,
    colorbar_pad_inches::Real=0.35,
    colorbar_width_inches::Real=0.17,
    colorbar_gap_inches::Real=0.29,
    default_t0_step::Real=0.2,
    default_V0_step::Real=0.2)

    isdir(data_dir) || throw(ArgumentError("data_dir does not exist: $data_dir"))
    t0_min <= t0_max || throw(ArgumentError("t0_min must be <= t0_max"))
    t0_grid_values = t0_values === nothing ? nothing :
        [x for x in _sorted_unique_floats(t0_values) if t0_min - t0_atol <= x <= t0_max + t0_atol]
    t0_values !== nothing && isempty(t0_grid_values) &&
        throw(ArgumentError("t0_values has no entries in [$t0_min, $t0_max]"))

    filenames = sort(filter(fn -> endswith(basename(fn), suffix), readdir(data_dir; join=true)))
    records = []
    for filename in filenames
        params = _result_filename_parameters(filename)
        params === nothing && continue
        t0_min - t0_atol <= params.t0 <= t0_max + t0_atol || continue
        t0_grid = t0_grid_values === nothing ? params.t0 : _nearest_grid_value(params.t0, t0_grid_values; atol=t0_atol)
        t0_grid === nothing && continue

        fields = _order_fields_from_file(filename;
            source=source,
            use_correlations=use_correlations,
            iteration=iteration,
            leg1=leg1,
            leg2=leg2,
            symmetrize_rung=symmetrize_rung,
            trim_boundary_rungs=trim_boundary_rungs)
        maxima = _order_fourier_maxima(fields; value=value, normalize=normalize)
        push!(records, (filename=filename, params=params, V0=params.V0, t0=t0_grid, maxima=maxima))
    end

    isempty(records) && throw(ArgumentError("no $suffix files with t0 in [$t0_min, $t0_max] were found in $data_dir"))

    t_values = t0_grid_values === nothing ? _sorted_unique_floats([rec.t0 for rec in records]) : t0_grid_values
    v_values = _sorted_unique_floats([rec.V0 for rec in records])
    x_edges = _grid_edges(t_values; default_step=default_t0_step)
    y_edges = _grid_edges(v_values; default_step=default_V0_step)

    grid = Dict{Tuple{Float64, Float64}, Any}()
    for rec in records
        key = (rec.V0, rec.t0)
        haskey(grid, key) && throw(ArgumentError("multiple files map to V0=$(rec.V0), t0=$(rec.t0)"))
        grid[key] = rec
    end

    swave_cmap = _order_grid_cmap(getfield(order_cmaps, :swave))
    order_specs = (
        (key=:cdw, label="CDW", cmap=_order_grid_cmap(getfield(order_cmaps, :cdw)), xslot=1, yslot=2, shape=:rect),
        (key=:sdw, label="SDW", cmap=_order_grid_cmap(getfield(order_cmaps, :sdw)), xslot=2, yslot=2, shape=:rect),
        (key=:swave, label="on-site s-wave", cmap=swave_cmap, xslot=1, yslot=1, shape=:onsite_swave_triangle),
        (key=:extended_swave, label="extended s-wave", cmap=swave_cmap, xslot=1, yslot=1, shape=:extended_swave_triangle),
        (key=:dwave, label="d-wave", cmap=_order_grid_cmap(getfield(order_cmaps, :dwave)), xslot=2, yslot=1, shape=:rect),
    )
    colorbar_specs = (
        (label="CDW", cmap=getfield(order_specs, 1).cmap),
        (label="SDW", cmap=getfield(order_specs, 2).cmap),
        (label="s-wave", cmap=swave_cmap),
        (label="d-wave", cmap=getfield(order_specs, 5).cmap),
    )

    max_values = Float64[]
    for rec in records, spec in order_specs
        push!(max_values, getfield(rec.maxima, spec.key).value)
    end
    scale = _fourier_grid_scale(max_values, value; logscale=logscale, vmin=vmin)

    fig, ax = PyPlot.subplots(figsize=_screen_figsize(figsize))
    hover_regions = []
    click_regions = []
    for (iy, V0) in enumerate(v_values), (ix, t0) in enumerate(t_values)
        x0 = x_edges[ix]
        y0 = y_edges[iy]
        w = x_edges[ix + 1] - x0
        h = y_edges[iy + 1] - y0

        rec = get(grid, (V0, t0), nothing)
        if rec === nothing
            ax.add_patch(_MatplotlibPatches.Rectangle((x0, y0), w, h;
                facecolor="0.90",
                edgecolor="black",
                linewidth=1.2))
            continue
        end

        click_detail_plots && push!(click_regions, (
            x0=x0,
            x1=x0 + w,
            y0=y0,
            y1=y0 + h,
            filename=rec.filename,
            trim_boundary_rungs=trim_boundary_rungs,
        ))

        subw = w / 2
        subh = h / 2
        strongest_value = -Inf
        strongest_shape = nothing
        for spec in order_specs
            summary = getfield(rec.maxima, spec.key)
            sx = x0 + (spec.xslot - 1) * subw
            sy = y0 + (spec.yslot - 1) * subh
            facecolor = _order_grid_facecolor(spec.cmap, summary.value, scale)
            vertices = nothing
            shape_ref = if spec.shape == :onsite_swave_triangle
                vertices = [(sx, sy), (sx + subw, sy), (sx, sy + subh)]
                ax.add_patch(_MatplotlibPatches.Polygon(vertices;
                    closed=true,
                    facecolor=facecolor,
                    edgecolor="none",
                    linewidth=0,
                    zorder=1))
                (kind=:triangle, vertices=vertices)
            elseif spec.shape == :extended_swave_triangle
                vertices = [(sx + subw, sy + subh), (sx + subw, sy), (sx, sy + subh)]
                ax.add_patch(_MatplotlibPatches.Polygon(vertices;
                    closed=true,
                    facecolor=facecolor,
                    edgecolor="none",
                    linewidth=0,
                    zorder=1))
                (kind=:triangle, vertices=vertices)
            else
                ax.add_patch(_MatplotlibPatches.Rectangle((sx, sy), subw, subh;
                    facecolor=facecolor,
                    edgecolor="none",
                    linewidth=0,
                    zorder=1))
                (kind=:rect, x=sx, y=sy, w=subw, h=subh)
            end
            summary_value = Float64(summary.value)
            if isfinite(summary_value) && summary_value > strongest_value
                strongest_value = summary_value
                strongest_shape = shape_ref
            end
            if hover
                region = vertices === nothing ?
                    (x0=sx, x1=sx + subw, y0=sy, y1=sy + subh, text=_order_grid_hover_text(spec.label, rec, summary)) :
                    (x0=sx, x1=sx + subw, y0=sy, y1=sy + subh, vertices=vertices, text=_order_grid_hover_text(spec.label, rec, summary))
                push!(hover_regions, region)
            end
        end
        ax.plot([x0 + subw, x0 + subw], [y0, y0 + h]; color="black", linewidth=1.2, zorder=3)
        ax.plot([x0, x0 + w], [y0 + subh, y0 + subh]; color="black", linewidth=1.2, zorder=3)
        ax.plot([x0 + subw, x0], [y0, y0 + subh]; color="black", linewidth=1.2, zorder=3)
        ax.add_patch(_MatplotlibPatches.Rectangle((x0, y0), w, h;
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            zorder=3))
        if strongest_shape !== nothing
            if strongest_shape.kind == :triangle
                ax.add_patch(_MatplotlibPatches.Polygon(strongest_shape.vertices;
                    closed=true,
                    facecolor="none",
                    edgecolor="yellow",
                    linewidth=2.0,
                    zorder=5))
            else
                ax.add_patch(_MatplotlibPatches.Rectangle((strongest_shape.x, strongest_shape.y), strongest_shape.w, strongest_shape.h;
                    facecolor="none",
                    edgecolor="yellow",
                    linewidth=2.0,
                    zorder=5))
            end
        end
    end

    ax.set_xlim(first(x_edges), last(x_edges))
    ax.set_ylim(first(y_edges), last(y_edges))
    ax.set_aspect("equal")
    ax.set_xlabel(_math("t_0"))
    ax.set_ylabel(_math("V_0"))
    ax.set_xticks(t_values)
    ax.set_xticklabels([_format_grid_number(x) for x in t_values])
    ax.set_yticks(v_values)
    ax.set_yticklabels([_format_grid_number(y) for y in v_values])

    if figure_title === nothing
        Ls = sort(unique([rec.params.L for rec in records]))
        Us = sort(unique([rec.params.U for rec in records]))
        tps = sort(unique([rec.params.tp for rec in records]))
        densities = sort(unique([rec.params.density for rec in records]))
        pieces = ["Fourier order maxima"]
        length(Ls) == 1 && push!(pieces, _math("L=" * string(Ls[1])))
        length(Us) == 1 && push!(pieces, _math("U=" * _format_grid_number(Us[1])))
        length(tps) == 1 && push!(pieces, _math("t_\\perp=" * _format_grid_number(tps[1])))
        length(densities) == 1 && push!(pieces, _math("n=" * _format_grid_number(densities[1])))
        ax.set_title(join(pieces, ", "))
    else
        ax.set_title(figure_title)
    end

    legend_handles = [_order_grid_legend_handle(spec) for spec in order_specs]
    legend_labels = [_order_grid_legend_label(spec.key, spec.label; leg1=leg1, leg2=leg2) for spec in order_specs]
    legend_handler_map = Dict(
        PyCall.pybuiltin("tuple") => _MatplotlibLegendHandler.HandlerTuple(ndivide=1, pad=0.0)
    )
    legend = ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=false,
        handler_map=legend_handler_map,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.7,
        labelspacing=0.65)

    top = 0.92
    fig.subplots_adjust(left=0.11, right=0.76, bottom=0.42, top=top)
    colorbars = []
    for (idx, spec) in enumerate(colorbar_specs)
        sm = _MatplotlibCm.ScalarMappable(norm=scale.norm, cmap=spec.cmap)
        sm.set_array(max_values)
        cax = fig.add_axes([0.0, 0.0, 0.01, 0.01])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.text(0.5, 1.02, spec.label;
            transform=cbar.ax.transAxes,
            rotation=90,
            rotation_mode="anchor",
            ha="left",
            va="center",
            clip_on=false)
        if idx < length(colorbar_specs)
            cbar.ax.tick_params(labelleft=false, labelright=false)
        end
        push!(colorbars, cbar)
    end
    colorbar_layout_callback, colorbar_layout_callback_id = _install_fixed_colorbar_layout!(fig, ax, [cbar.ax for cbar in colorbars];
        pad_inches=colorbar_pad_inches,
        width_inches=colorbar_width_inches,
        gap_inches=colorbar_gap_inches)
    scale_suffix = logscale ? " (shared log scale)" : " (shared linear scale)"
    colorbars[end].set_label(_fourier_colorbar_label(value) * " max" * scale_suffix)

    annotation = nothing
    hover_callback = nothing
    callback_id = nothing
    if hover
        annotation, hover_callback, callback_id = _install_order_grid_hover!(fig, ax, hover_regions;
            right_artist=colorbars[end].ax.yaxis.label,
            bottom_artist=legend)
    end
    click_callback = nothing
    click_callback_id = nothing
    spawned_heatmap_figures = []
    spawned_detail_figures = []
    if click_detail_plots
        click_callback, click_callback_id, spawned_heatmap_figures, spawned_detail_figures = _install_order_grid_click!(fig, ax, click_regions; iteration=iteration)
    end

    _save_if_requested(fig, savepath)
    return (
        fig=fig,
        ax=ax,
        colorbars=colorbars,
        legend=legend,
        annotation=annotation,
        callback=hover_callback,
        callback_id=callback_id,
        click_callback=click_callback,
        click_callback_id=click_callback_id,
        clicked_heatmap_figures=spawned_heatmap_figures,
        clicked_detail_figures=spawned_detail_figures,
        clicked_profile_figures=spawned_detail_figures,
        colorbar_layout_callback=colorbar_layout_callback,
        colorbar_layout_callback_id=colorbar_layout_callback_id,
        data=records,
        scale=scale,
    )
end

function plot_mf_profiles(alpha, beta;
    spin=:up,
    leg::Integer=1,
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    symmetrize_rung::Bool=false,
    show_other_leg::Bool=true,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    niter = _profile_iteration_count((alpha, 4, "alpha"), (beta, 5, "beta"))
    profile_iteration = _initial_profile_iteration(iteration, niter)
    other_leg = _other_leg(leg)

    function profile_data(it)
        r1, v1 = cdw_from_beta(beta; leg=leg, iteration=it, use_abs=use_abs)
        r2, v2 = sdw_from_beta(beta; leg=leg, iteration=it, use_abs=use_abs)
        r3, v3 = onsite_pairing_from_alpha(alpha; leg=leg, iteration=it, use_abs=use_abs)
        r4, v4 = extended_swave_profile_from_alpha(alpha; leg1=leg1, leg2=leg2, iteration=it, symmetrize=symmetrize_rung, use_abs=use_abs)
        r5, v5 = dwave_profile_from_alpha(alpha; leg1=leg1, leg2=leg2, iteration=it, symmetrize=symmetrize_rung, use_abs=use_abs)
        r1o, v1o = cdw_from_beta(beta; leg=other_leg, iteration=it, use_abs=use_abs)
        r2o, v2o = sdw_from_beta(beta; leg=other_leg, iteration=it, use_abs=use_abs)
        r3o, v3o = onsite_pairing_from_alpha(alpha; leg=other_leg, iteration=it, use_abs=use_abs)
        return (r1=r1, v1=v1, r2=r2, v2=v2, r3=r3, v3=v3, r4=r4, v4=v4, r5=r5, v5=v5,
            r1o=r1o, v1o=v1o, r2o=r2o, v2o=v2o, r3o=r3o, v3o=v3o)
    end

    has_slider = niter !== nothing && niter > 1
    profile_history = has_slider ? [profile_data(it) for it in 1:niter] : nothing
    data = has_slider ? profile_history[profile_iteration] : profile_data(profile_iteration)

    fig, axes = PyPlot.subplots(5, 1, figsize=_screen_figsize((8.5, 12.5)))
    cdw_ylabel = use_abs ? "|CDW beta proxy|" : "CDW beta proxy"
    sdw_ylabel = use_abs ? "|SDW beta proxy|" : "SDW beta proxy"
    onsite_ylabel = use_abs ? "|s-wave alpha|" : "s-wave alpha"
    extended_swave_ylabel = use_abs ? "|extended s-wave alpha|" : "extended s-wave alpha"
    rung_ylabel = use_abs ? "|d-wave alpha|" : "d-wave alpha"

    leg_title = show_other_leg ? "" : ", leg $leg"
    cdw_lines = _plot_leg_pair!(axes[1], data.r1, data.v1, data.r1o, data.v1o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_cdw_leg_color, xlabel="", ylabel=cdw_ylabel, title="CDW beta proxy: up + down$leg_title", kwargs...)
    sdw_lines = _plot_leg_pair!(axes[2], data.r2, data.v2, data.r2o, data.v2o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_sdw_leg_color, xlabel="", ylabel=sdw_ylabel, title="SDW beta proxy: up - down$leg_title", kwargs...)
    swave_lines = _plot_leg_pair!(axes[3], data.r3, data.v3, data.r3o, data.v3o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_swave_leg_color, xlabel="", ylabel=onsite_ylabel, title="s-wave pairing from alpha$leg_title", kwargs...)
    _plot_series!(axes[4], data.r4, data.v4; xlabel="", ylabel=extended_swave_ylabel, title="extended s-wave alpha proxy: " * _extended_swave_alpha_title(leg1, leg2; symmetrize=symmetrize_rung), color=_EXT_SWAVE_COLOR, kwargs...)
    extended_swave_line = axes[4].lines[end]
    _plot_series!(axes[5], data.r5, data.v5; xlabel=_rung_index_xlabel(), ylabel=rung_ylabel, title="d-wave alpha proxy: " * _dwave_alpha_title(leg1, leg2; symmetrize=symmetrize_rung), color=_DWAVE_COLOR, kwargs...)
    dwave_line = axes[5].lines[end]

    if has_slider
        _set_profile_axis_limits!(axes[1], profile_history, :r1, :v1; other_xfield=:r1o, other_yfield=:v1o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[2], profile_history, :r2, :v2; other_xfield=:r2o, other_yfield=:v2o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[3], profile_history, :r3, :v3; other_xfield=:r3o, other_yfield=:v3o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[4], profile_history, :r4, :v4)
        _set_profile_axis_limits!(axes[5], profile_history, :r5, :v5)
    end

    tight_rect = _profile_tight_rect(figure_title, has_slider)
    _finish_figure(fig; figure_title=figure_title, tight_rect=tight_rect, h_pad=_MF_STACK_H_PAD)
    if has_slider
        function update_iteration(it)
            updated = profile_history[it]
            _set_line_data!(cdw_lines.main, updated.r1, updated.v1)
            show_other_leg && _set_line_data!(cdw_lines.other, updated.r1o, updated.v1o)
            _set_line_data!(sdw_lines.main, updated.r2, updated.v2)
            show_other_leg && _set_line_data!(sdw_lines.other, updated.r2o, updated.v2o)
            _set_line_data!(swave_lines.main, updated.r3, updated.v3)
            show_other_leg && _set_line_data!(swave_lines.other, updated.r3o, updated.v3o)
            _set_line_data!(extended_swave_line, updated.r4, updated.v4)
            _set_line_data!(dwave_line, updated.r5, updated.v5)
            return nothing
        end
        _install_iteration_slider!(fig, niter, profile_iteration, update_iteration; figure_title=figure_title, target_axes=axes)
    end
    return _save_if_requested(fig, savepath)
end

function plot_correlation_profiles(C_pair, C_exc_dn, C_exc_up;
    spin=:up,
    leg::Integer=1,
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    symmetrize_rung::Bool=false,
    show_other_leg::Bool=true,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    niter = _profile_iteration_count((C_pair, 2, "C_pair_list"), (C_exc_dn, 2, "C_exc_dn_list"), (C_exc_up, 2, "C_exc_up_list"))
    profile_iteration = _initial_profile_iteration(iteration, niter)
    other_leg = _other_leg(leg)

    function profile_data(it)
        r1, v1 = cdw_from_correlations(C_exc_dn, C_exc_up; leg=leg, iteration=it, use_abs=use_abs)
        r2, v2 = sdw_from_correlations(C_exc_dn, C_exc_up; leg=leg, iteration=it, use_abs=use_abs)
        r3, v3 = onsite_pairing_from_correlations(C_pair; leg=leg, iteration=it, use_abs=use_abs)
        r4, v4 = extended_swave_profile_from_correlations(C_pair; leg1=leg1, leg2=leg2, iteration=it, use_abs=use_abs)
        r5, v5 = dwave_profile_from_correlations(C_pair; leg1=leg1, leg2=leg2, iteration=it, use_abs=use_abs)
        r1o, v1o = cdw_from_correlations(C_exc_dn, C_exc_up; leg=other_leg, iteration=it, use_abs=use_abs)
        r2o, v2o = sdw_from_correlations(C_exc_dn, C_exc_up; leg=other_leg, iteration=it, use_abs=use_abs)
        r3o, v3o = onsite_pairing_from_correlations(C_pair; leg=other_leg, iteration=it, use_abs=use_abs)
        return (r1=r1, v1=v1, r2=r2, v2=v2, r3=r3, v3=v3, r4=r4, v4=v4, r5=r5, v5=v5,
            r1o=r1o, v1o=v1o, r2o=r2o, v2o=v2o, r3o=r3o, v3o=v3o)
    end

    has_slider = niter !== nothing && niter > 1
    profile_history = has_slider ? [profile_data(it) for it in 1:niter] : nothing
    data = has_slider ? profile_history[profile_iteration] : profile_data(profile_iteration)

    fig, axes = PyPlot.subplots(5, 1, figsize=_screen_figsize((8.5, 12.5)))
    leg_tex = show_other_leg ? "\\ell" : string(leg)
    leg_title = show_other_leg ? "legs $leg and $other_leg" : "leg $leg"
    cdw_expr = _cdw_corr_expr(; leg_tex=leg_tex)
    sdw_expr = _sdw_corr_expr(; leg_tex=leg_tex)
    onsite_expr = _onsite_pair_corr_expr(; leg_tex=leg_tex)
    extended_swave_expr = _extended_swave_corr_symbol()
    dwave_expr = _dwave_corr_symbol()
    cdw_title = show_other_leg ? "CDW" : "CDW: " * _cdw_corr_title(leg)
    sdw_title = show_other_leg ? "SDW" : "SDW: " * _sdw_corr_title(leg)
    onsite_title = show_other_leg ? "s-wave" : "s-wave: " * _onsite_pair_corr_title(leg)

    cdw_lines = _plot_leg_pair!(axes[1], data.r1, data.v1, data.r1o, data.v1o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_cdw_leg_color, xlabel="", ylabel=_corr_value_label(cdw_expr, use_abs), title=cdw_title, kwargs...)
    sdw_lines = _plot_leg_pair!(axes[2], data.r2, data.v2, data.r2o, data.v2o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_sdw_leg_color, xlabel="", ylabel=_corr_value_label(sdw_expr, use_abs), title=sdw_title, kwargs...)
    swave_lines = _plot_leg_pair!(axes[3], data.r3, data.v3, data.r3o, data.v3o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_swave_leg_color, xlabel="", ylabel=_corr_value_label(onsite_expr, use_abs), title=onsite_title, kwargs...)
    _plot_series!(axes[4], data.r4, data.v4; xlabel="", ylabel=_corr_value_label(extended_swave_expr, use_abs), title="extended s-wave: " * _extended_swave_corr_title(leg1, leg2; include_singlet_def=false), color=_EXT_SWAVE_COLOR, kwargs...)
    extended_swave_line = axes[4].lines[end]
    _plot_series!(axes[5], data.r5, data.v5; xlabel=_rung_index_xlabel(), ylabel=_corr_value_label(dwave_expr, use_abs), title="d-wave: " * _dwave_corr_title(leg1, leg2; include_singlet_def=false), color=_DWAVE_COLOR, kwargs...)
    dwave_line = axes[5].lines[end]

    if has_slider
        _set_profile_axis_limits!(axes[1], profile_history, :r1, :v1; other_xfield=:r1o, other_yfield=:v1o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[2], profile_history, :r2, :v2; other_xfield=:r2o, other_yfield=:v2o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[3], profile_history, :r3, :v3; other_xfield=:r3o, other_yfield=:v3o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[4], profile_history, :r4, :v4)
        _set_profile_axis_limits!(axes[5], profile_history, :r5, :v5)
    end

    tight_rect = _profile_tight_rect(figure_title, has_slider)
    _finish_figure(fig; figure_title=figure_title, tight_rect=tight_rect, h_pad=_MF_STACK_H_PAD)
    if has_slider
        function update_iteration(it)
            updated = profile_history[it]
            _set_line_data!(cdw_lines.main, updated.r1, updated.v1)
            show_other_leg && _set_line_data!(cdw_lines.other, updated.r1o, updated.v1o)
            _set_line_data!(sdw_lines.main, updated.r2, updated.v2)
            show_other_leg && _set_line_data!(sdw_lines.other, updated.r2o, updated.v2o)
            _set_line_data!(swave_lines.main, updated.r3, updated.v3)
            show_other_leg && _set_line_data!(swave_lines.other, updated.r3o, updated.v3o)
            _set_line_data!(extended_swave_line, updated.r4, updated.v4)
            _set_line_data!(dwave_line, updated.r5, updated.v5)
            return nothing
        end
        _install_iteration_slider!(fig, niter, profile_iteration, update_iteration; figure_title=figure_title, target_axes=axes)
    end
    return _save_if_requested(fig, savepath)
end

function plot_middle_histories(alpha_list, beta_list;
    spin=:up,
    leg::Integer=1,
    rung=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize_rung::Bool=false,
    show_other_leg::Bool=true,
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
    it4, v4 = middle_extended_swave_history(alpha_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, symmetrize=symmetrize_rung, use_abs=use_abs)
    it5, v5 = middle_dwave_history(alpha_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, symmetrize=symmetrize_rung, use_abs=use_abs)
    other_leg = _other_leg(leg)
    it1o, v1o = middle_cdw_history(beta_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)
    it2o, v2o = middle_sdw_history(beta_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)
    it3o, v3o = middle_onsite_pairing_history(alpha_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)

    fig, axes = PyPlot.subplots(5, 1, figsize=_screen_figsize((8.5, 12.5)))
    cdw_ylabel = use_abs ? "|CDW beta proxy|" : "CDW beta proxy"
    sdw_ylabel = use_abs ? "|SDW beta proxy|" : "SDW beta proxy"
    onsite_ylabel = use_abs ? "|s-wave alpha|" : "s-wave alpha"
    extended_swave_ylabel = use_abs ? "|extended s-wave alpha|" : "extended s-wave alpha"
    rung_ylabel = use_abs ? "|d-wave alpha|" : "d-wave alpha"

    leg_title = show_other_leg ? "" : ", leg $leg"
    _plot_series!(axes[1], it1, v1; xlabel="", ylabel=cdw_ylabel, title="Middle rung $rung_to_plot CDW beta proxy: up + down$leg_title", label="leg $leg", color=_cdw_leg_color(leg), kwargs...)
    show_other_leg && _plot_other_leg!(axes[1], it1o, v1o; leg=other_leg, color=_cdw_leg_color(other_leg))
    _plot_series!(axes[2], it2, v2; xlabel="", ylabel=sdw_ylabel, title="Middle rung $rung_to_plot SDW beta proxy: up - down$leg_title", label="leg $leg", color=_sdw_leg_color(leg), kwargs...)
    show_other_leg && _plot_other_leg!(axes[2], it2o, v2o; leg=other_leg, color=_sdw_leg_color(other_leg))
    _plot_series!(axes[3], it3, v3; xlabel="", ylabel=onsite_ylabel, title="Middle rung $rung_to_plot s-wave pairing$leg_title", label="leg $leg", color=_swave_leg_color(leg), kwargs...)
    show_other_leg && _plot_other_leg!(axes[3], it3o, v3o; leg=other_leg, color=_swave_leg_color(other_leg))
    _plot_series!(axes[4], it4, v4; xlabel="", ylabel=extended_swave_ylabel, title="Middle rung $rung_to_plot extended s-wave alpha proxy: " * _extended_swave_alpha_title(leg1, leg2; rung_tex=string(rung_to_plot), symmetrize=symmetrize_rung), color=_EXT_SWAVE_COLOR, kwargs...)
    _plot_series!(axes[5], it5, v5; xlabel=_mf_iteration_xlabel(), ylabel=rung_ylabel, title="Middle rung $rung_to_plot d-wave alpha proxy: " * _dwave_alpha_title(leg1, leg2; rung_tex=string(rung_to_plot), symmetrize=symmetrize_rung), color=_DWAVE_COLOR, kwargs...)

    tight_rect = figure_title === nothing ? nothing : (0, 0, 1, 0.95)
    _finish_figure(fig; figure_title=figure_title, tight_rect=tight_rect, h_pad=_MF_STACK_H_PAD)
    return _save_if_requested(fig, savepath)
end

function plot_middle_histories_from_correlations(C_pair_list, C_exc_dn_list, C_exc_up_list;
    spin=:up,
    leg::Integer=1,
    rung=nothing,
    leg1::Integer=1,
    leg2::Integer=2,
    symmetrize_rung::Bool=false,
    show_other_leg::Bool=true,
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
    it4, v4 = middle_extended_swave_history_from_correlations(C_pair_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, use_abs=use_abs)
    it5, v5 = middle_dwave_history_from_correlations(C_pair_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, use_abs=use_abs)
    other_leg = _other_leg(leg)
    it1o, v1o = middle_cdw_history_from_correlations(C_exc_dn_list, C_exc_up_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)
    it2o, v2o = middle_sdw_history_from_correlations(C_exc_dn_list, C_exc_up_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)
    it3o, v3o = middle_onsite_pairing_history_from_correlations(C_pair_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)

    fig, axes = PyPlot.subplots(5, 1, figsize=_screen_figsize((8.5, 12.5)))
    rung_tex = string(rung_to_plot)
    leg_tex = show_other_leg ? "\\ell" : string(leg)
    leg_title = show_other_leg ? "" : ", leg $leg"
    cdw_expr = _cdw_corr_expr(; rung_tex=rung_tex, leg_tex=leg_tex)
    sdw_expr = _sdw_corr_expr(; rung_tex=rung_tex, leg_tex=leg_tex)
    onsite_expr = _onsite_pair_corr_expr(; rung_tex=rung_tex, leg_tex=leg_tex)
    extended_swave_expr = _extended_swave_corr_symbol(rung_tex)
    dwave_expr = _dwave_corr_symbol(rung_tex)
    cdw_title = show_other_leg ? "Middle rung $rung_to_plot CDW" : "CDW: " * _cdw_corr_title(leg; rung_tex=rung_tex)
    sdw_title = show_other_leg ? "Middle rung $rung_to_plot SDW" : "SDW: " * _sdw_corr_title(leg; rung_tex=rung_tex)
    onsite_title = show_other_leg ? "Middle rung $rung_to_plot s-wave" : "s-wave: " * _onsite_pair_corr_title(leg; rung_tex=rung_tex)

    _plot_series!(axes[1], it1, v1; xlabel="", ylabel=_corr_value_label(cdw_expr, use_abs), title=cdw_title, label="leg $leg", color=_cdw_leg_color(leg), kwargs...)
    show_other_leg && _plot_other_leg!(axes[1], it1o, v1o; leg=other_leg, color=_cdw_leg_color(other_leg))
    _plot_series!(axes[2], it2, v2; xlabel="", ylabel=_corr_value_label(sdw_expr, use_abs), title=sdw_title, label="leg $leg", color=_sdw_leg_color(leg), kwargs...)
    show_other_leg && _plot_other_leg!(axes[2], it2o, v2o; leg=other_leg, color=_sdw_leg_color(other_leg))
    _plot_series!(axes[3], it3, v3; xlabel="", ylabel=_corr_value_label(onsite_expr, use_abs), title=onsite_title, label="leg $leg", color=_swave_leg_color(leg), kwargs...)
    show_other_leg && _plot_other_leg!(axes[3], it3o, v3o; leg=other_leg, color=_swave_leg_color(other_leg))
    _plot_series!(axes[4], it4, v4; xlabel="", ylabel=_corr_value_label(extended_swave_expr, use_abs), title="extended s-wave: " * _extended_swave_corr_title(leg1, leg2; rung_tex=rung_tex, include_singlet_def=false), color=_EXT_SWAVE_COLOR, kwargs...)
    _plot_series!(axes[5], it5, v5; xlabel=_mf_iteration_xlabel(), ylabel=_corr_value_label(dwave_expr, use_abs), title="d-wave: " * _dwave_corr_title(leg1, leg2; rung_tex=rung_tex, include_singlet_def=false), color=_DWAVE_COLOR, kwargs...)

    tight_rect = figure_title === nothing ? nothing : (0, 0, 1, 0.95)
    _finish_figure(fig; figure_title=figure_title, tight_rect=tight_rect, h_pad=_MF_STACK_H_PAD)
    return _save_if_requested(fig, savepath)
end

function plot_mf_profiles_and_middle_histories(alpha, beta, alpha_list, beta_list;
    spin=:up,
    leg::Integer=1,
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    rung=nothing,
    symmetrize_rung::Bool=false,
    show_other_leg::Bool=true,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    profile_alpha = (ndims(alpha) == 4 && alpha_list !== nothing) ? alpha_list : alpha
    profile_beta = (ndims(beta) == 5 && beta_list !== nothing) ? beta_list : beta
    niter = _profile_iteration_count((profile_alpha, 4, "alpha"), (profile_beta, 5, "beta"))
    profile_iteration = _initial_profile_iteration(iteration, niter)
    other_leg = _other_leg(leg)

    function profile_data(it)
        r1, v1 = cdw_from_beta(profile_beta; leg=leg, iteration=it, use_abs=use_abs)
        r2, v2 = sdw_from_beta(profile_beta; leg=leg, iteration=it, use_abs=use_abs)
        r3, v3 = onsite_pairing_from_alpha(profile_alpha; leg=leg, iteration=it, use_abs=use_abs)
        r4, v4 = extended_swave_profile_from_alpha(profile_alpha; leg1=leg1, leg2=leg2, iteration=it, symmetrize=symmetrize_rung, use_abs=use_abs)
        r5, v5 = dwave_profile_from_alpha(profile_alpha; leg1=leg1, leg2=leg2, iteration=it, symmetrize=symmetrize_rung, use_abs=use_abs)
        r1o, v1o = cdw_from_beta(profile_beta; leg=other_leg, iteration=it, use_abs=use_abs)
        r2o, v2o = sdw_from_beta(profile_beta; leg=other_leg, iteration=it, use_abs=use_abs)
        r3o, v3o = onsite_pairing_from_alpha(profile_alpha; leg=other_leg, iteration=it, use_abs=use_abs)
        return (r1=r1, v1=v1, r2=r2, v2=v2, r3=r3, v3=v3, r4=r4, v4=v4, r5=r5, v5=v5,
            r1o=r1o, v1o=v1o, r2o=r2o, v2o=v2o, r3o=r3o, v3o=v3o)
    end

    has_slider = niter !== nothing && niter > 1
    profile_history = has_slider ? [profile_data(it) for it in 1:niter] : nothing
    data = has_slider ? profile_history[profile_iteration] : profile_data(profile_iteration)

    b = _history_array(beta_list, 6, "beta_list")
    L = size(b, 2)
    L > 1 || throw(ArgumentError("need at least two rungs to compute local d-wave profile"))
    rung_to_plot = rung === nothing ? min(middle_rung(L), L - 1) : _check_index(rung, L - 1, "d-wave rung")
    it1, hv1 = middle_cdw_history(beta_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it2, hv2 = middle_sdw_history(beta_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it3, hv3 = middle_onsite_pairing_history(alpha_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it4, hv4 = middle_extended_swave_history(alpha_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, symmetrize=symmetrize_rung, use_abs=use_abs)
    it5, hv5 = middle_dwave_history(alpha_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, symmetrize=symmetrize_rung, use_abs=use_abs)
    it1o, hv1o = middle_cdw_history(beta_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)
    it2o, hv2o = middle_sdw_history(beta_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)
    it3o, hv3o = middle_onsite_pairing_history(alpha_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)

    fig, axes = PyPlot.subplots(5, 2, figsize=_screen_figsize((15.0, 12.5)))
    cdw_ylabel = use_abs ? "|CDW beta proxy|" : "CDW beta proxy"
    sdw_ylabel = use_abs ? "|SDW beta proxy|" : "SDW beta proxy"
    onsite_ylabel = use_abs ? "|s-wave alpha|" : "s-wave alpha"
    extended_swave_ylabel = use_abs ? "|extended s-wave alpha|" : "extended s-wave alpha"
    rung_ylabel = use_abs ? "|d-wave alpha|" : "d-wave alpha"
    leg_title = show_other_leg ? "" : ", leg $leg"

    cdw_profile_lines = _plot_leg_pair!(axes[1, 1], data.r1, data.v1, data.r1o, data.v1o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_cdw_leg_color, xlabel="", ylabel=cdw_ylabel, title="CDW beta proxy: up + down$leg_title", kwargs...)
    _plot_leg_pair!(axes[1, 2], it1, hv1, it1o, hv1o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_cdw_leg_color, xlabel="", ylabel=cdw_ylabel, title="Middle rung $rung_to_plot CDW beta proxy: up + down$leg_title", kwargs...)
    sdw_profile_lines = _plot_leg_pair!(axes[2, 1], data.r2, data.v2, data.r2o, data.v2o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_sdw_leg_color, xlabel="", ylabel=sdw_ylabel, title="SDW beta proxy: up - down$leg_title", kwargs...)
    _plot_leg_pair!(axes[2, 2], it2, hv2, it2o, hv2o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_sdw_leg_color, xlabel="", ylabel=sdw_ylabel, title="Middle rung $rung_to_plot SDW beta proxy: up - down$leg_title", kwargs...)
    swave_profile_lines = _plot_leg_pair!(axes[3, 1], data.r3, data.v3, data.r3o, data.v3o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_swave_leg_color, xlabel="", ylabel=onsite_ylabel, title="s-wave pairing from alpha$leg_title", kwargs...)
    _plot_leg_pair!(axes[3, 2], it3, hv3, it3o, hv3o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_swave_leg_color, xlabel="", ylabel=onsite_ylabel, title="Middle rung $rung_to_plot s-wave pairing$leg_title", kwargs...)
    _plot_series!(axes[4, 1], data.r4, data.v4; xlabel="", ylabel=extended_swave_ylabel, title="extended s-wave alpha proxy: " * _extended_swave_alpha_title(leg1, leg2; symmetrize=symmetrize_rung), color=_EXT_SWAVE_COLOR, kwargs...)
    extended_swave_profile_line = axes[4, 1].lines[end]
    _plot_series!(axes[4, 2], it4, hv4; xlabel="", ylabel=extended_swave_ylabel, title="Middle rung $rung_to_plot extended s-wave alpha proxy: " * _extended_swave_alpha_title(leg1, leg2; rung_tex=string(rung_to_plot), symmetrize=symmetrize_rung), color=_EXT_SWAVE_COLOR, kwargs...)
    _plot_series!(axes[5, 1], data.r5, data.v5; xlabel=_rung_index_xlabel(), ylabel=rung_ylabel, title="d-wave alpha proxy: " * _dwave_alpha_title(leg1, leg2; symmetrize=symmetrize_rung), color=_DWAVE_COLOR, kwargs...)
    dwave_profile_line = axes[5, 1].lines[end]
    _plot_series!(axes[5, 2], it5, hv5; xlabel=_mf_iteration_xlabel(), ylabel=rung_ylabel, title="Middle rung $rung_to_plot d-wave alpha proxy: " * _dwave_alpha_title(leg1, leg2; rung_tex=string(rung_to_plot), symmetrize=symmetrize_rung), color=_DWAVE_COLOR, kwargs...)
    markers = [_add_iteration_marker!(axes[row, 2], profile_iteration) for row in 1:5]

    if has_slider
        _set_profile_axis_limits!(axes[1, 1], profile_history, :r1, :v1; other_xfield=:r1o, other_yfield=:v1o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[2, 1], profile_history, :r2, :v2; other_xfield=:r2o, other_yfield=:v2o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[3, 1], profile_history, :r3, :v3; other_xfield=:r3o, other_yfield=:v3o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[4, 1], profile_history, :r4, :v4)
        _set_profile_axis_limits!(axes[5, 1], profile_history, :r5, :v5)
    end

    tight_rect = _profile_tight_rect(figure_title, has_slider)
    _finish_figure(fig; figure_title=figure_title, tight_rect=tight_rect, h_pad=_MF_STACK_H_PAD)
    if has_slider
        function update_iteration(it)
            updated = profile_history[it]
            _set_line_data!(cdw_profile_lines.main, updated.r1, updated.v1)
            show_other_leg && _set_line_data!(cdw_profile_lines.other, updated.r1o, updated.v1o)
            _set_line_data!(sdw_profile_lines.main, updated.r2, updated.v2)
            show_other_leg && _set_line_data!(sdw_profile_lines.other, updated.r2o, updated.v2o)
            _set_line_data!(swave_profile_lines.main, updated.r3, updated.v3)
            show_other_leg && _set_line_data!(swave_profile_lines.other, updated.r3o, updated.v3o)
            _set_line_data!(extended_swave_profile_line, updated.r4, updated.v4)
            _set_line_data!(dwave_profile_line, updated.r5, updated.v5)
            for marker in markers
                _set_iteration_marker!(marker, it)
            end
            return nothing
        end
        _install_iteration_slider!(fig, niter, profile_iteration, update_iteration; figure_title=figure_title, target_axes=[axes[row, 1] for row in 1:5])
    end
    return _save_if_requested(fig, savepath)
end

function plot_correlation_profiles_and_middle_histories(C_pair, C_exc_dn, C_exc_up, C_pair_list, C_exc_dn_list, C_exc_up_list;
    spin=:up,
    leg::Integer=1,
    leg1::Integer=1,
    leg2::Integer=2,
    iteration=nothing,
    rung=nothing,
    symmetrize_rung::Bool=false,
    show_other_leg::Bool=true,
    use_abs::Bool=false,
    savepath=nothing,
    figure_title=nothing,
    kwargs...)

    profile_C_pair = (ndims(C_pair) == 2 && C_pair_list !== nothing) ? C_pair_list : C_pair
    profile_C_exc_dn = (ndims(C_exc_dn) == 2 && C_exc_dn_list !== nothing) ? C_exc_dn_list : C_exc_dn
    profile_C_exc_up = (ndims(C_exc_up) == 2 && C_exc_up_list !== nothing) ? C_exc_up_list : C_exc_up
    niter = _profile_iteration_count((profile_C_pair, 2, "C_pair_list"), (profile_C_exc_dn, 2, "C_exc_dn_list"), (profile_C_exc_up, 2, "C_exc_up_list"))
    profile_iteration = _initial_profile_iteration(iteration, niter)
    other_leg = _other_leg(leg)

    function profile_data(it)
        r1, v1 = cdw_from_correlations(profile_C_exc_dn, profile_C_exc_up; leg=leg, iteration=it, use_abs=use_abs)
        r2, v2 = sdw_from_correlations(profile_C_exc_dn, profile_C_exc_up; leg=leg, iteration=it, use_abs=use_abs)
        r3, v3 = onsite_pairing_from_correlations(profile_C_pair; leg=leg, iteration=it, use_abs=use_abs)
        r4, v4 = extended_swave_profile_from_correlations(profile_C_pair; leg1=leg1, leg2=leg2, iteration=it, use_abs=use_abs)
        r5, v5 = dwave_profile_from_correlations(profile_C_pair; leg1=leg1, leg2=leg2, iteration=it, use_abs=use_abs)
        r1o, v1o = cdw_from_correlations(profile_C_exc_dn, profile_C_exc_up; leg=other_leg, iteration=it, use_abs=use_abs)
        r2o, v2o = sdw_from_correlations(profile_C_exc_dn, profile_C_exc_up; leg=other_leg, iteration=it, use_abs=use_abs)
        r3o, v3o = onsite_pairing_from_correlations(profile_C_pair; leg=other_leg, iteration=it, use_abs=use_abs)
        return (r1=r1, v1=v1, r2=r2, v2=v2, r3=r3, v3=v3, r4=r4, v4=v4, r5=r5, v5=v5,
            r1o=r1o, v1o=v1o, r2o=r2o, v2o=v2o, r3o=r3o, v3o=v3o)
    end

    has_slider = niter !== nothing && niter > 1
    profile_history = has_slider ? [profile_data(it) for it in 1:niter] : nothing
    data = has_slider ? profile_history[profile_iteration] : profile_data(profile_iteration)

    c = _history_array(C_pair_list, 3, "C_pair_list")
    L = _rung_count_from_sites(size(c, 1))
    L > 1 || throw(ArgumentError("need at least two rungs to compute local d-wave profile"))
    rung_to_plot = rung === nothing ? min(middle_rung(L), L - 1) : _check_index(rung, L - 1, "d-wave rung")
    it1, hv1 = middle_cdw_history_from_correlations(C_exc_dn_list, C_exc_up_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it2, hv2 = middle_sdw_history_from_correlations(C_exc_dn_list, C_exc_up_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it3, hv3 = middle_onsite_pairing_history_from_correlations(C_pair_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it4, hv4 = middle_extended_swave_history_from_correlations(C_pair_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, use_abs=use_abs)
    it5, hv5 = middle_dwave_history_from_correlations(C_pair_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, use_abs=use_abs)
    it1o, hv1o = middle_cdw_history_from_correlations(C_exc_dn_list, C_exc_up_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)
    it2o, hv2o = middle_sdw_history_from_correlations(C_exc_dn_list, C_exc_up_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)
    it3o, hv3o = middle_onsite_pairing_history_from_correlations(C_pair_list; leg=other_leg, rung=rung_to_plot, use_abs=use_abs)

    fig, axes = PyPlot.subplots(5, 2, figsize=_screen_figsize((15.0, 12.5)))
    profile_leg_tex = show_other_leg ? "\\ell" : string(leg)
    history_rung_tex = string(rung_to_plot)
    history_leg_tex = show_other_leg ? "\\ell" : string(leg)
    profile_cdw_expr = _cdw_corr_expr(; leg_tex=profile_leg_tex)
    profile_sdw_expr = _sdw_corr_expr(; leg_tex=profile_leg_tex)
    profile_onsite_expr = _onsite_pair_corr_expr(; leg_tex=profile_leg_tex)
    history_cdw_expr = _cdw_corr_expr(; rung_tex=history_rung_tex, leg_tex=history_leg_tex)
    history_sdw_expr = _sdw_corr_expr(; rung_tex=history_rung_tex, leg_tex=history_leg_tex)
    history_onsite_expr = _onsite_pair_corr_expr(; rung_tex=history_rung_tex, leg_tex=history_leg_tex)
    profile_extended_swave_expr = _extended_swave_corr_symbol()
    history_extended_swave_expr = _extended_swave_corr_symbol(history_rung_tex)
    profile_dwave_expr = _dwave_corr_symbol()
    history_dwave_expr = _dwave_corr_symbol(history_rung_tex)
    leg_title = show_other_leg ? "" : ", leg $leg"
    profile_cdw_title = show_other_leg ? "CDW" : "CDW: " * _cdw_corr_title(leg)
    profile_sdw_title = show_other_leg ? "SDW" : "SDW: " * _sdw_corr_title(leg)
    profile_onsite_title = show_other_leg ? "s-wave" : "s-wave: " * _onsite_pair_corr_title(leg)
    history_cdw_title = show_other_leg ? "Middle rung $rung_to_plot CDW" : "CDW: " * _cdw_corr_title(leg; rung_tex=history_rung_tex)
    history_sdw_title = show_other_leg ? "Middle rung $rung_to_plot SDW" : "SDW: " * _sdw_corr_title(leg; rung_tex=history_rung_tex)
    history_onsite_title = show_other_leg ? "Middle rung $rung_to_plot s-wave" : "s-wave: " * _onsite_pair_corr_title(leg; rung_tex=history_rung_tex)

    cdw_profile_lines = _plot_leg_pair!(axes[1, 1], data.r1, data.v1, data.r1o, data.v1o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_cdw_leg_color, xlabel="", ylabel=_corr_value_label(profile_cdw_expr, use_abs), title=profile_cdw_title, kwargs...)
    _plot_leg_pair!(axes[1, 2], it1, hv1, it1o, hv1o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_cdw_leg_color, xlabel="", ylabel=_corr_value_label(history_cdw_expr, use_abs), title=history_cdw_title, kwargs...)
    sdw_profile_lines = _plot_leg_pair!(axes[2, 1], data.r2, data.v2, data.r2o, data.v2o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_sdw_leg_color, xlabel="", ylabel=_corr_value_label(profile_sdw_expr, use_abs), title=profile_sdw_title, kwargs...)
    _plot_leg_pair!(axes[2, 2], it2, hv2, it2o, hv2o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_sdw_leg_color, xlabel="", ylabel=_corr_value_label(history_sdw_expr, use_abs), title=history_sdw_title, kwargs...)
    swave_profile_lines = _plot_leg_pair!(axes[3, 1], data.r3, data.v3, data.r3o, data.v3o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_swave_leg_color, xlabel="", ylabel=_corr_value_label(profile_onsite_expr, use_abs), title=profile_onsite_title, kwargs...)
    _plot_leg_pair!(axes[3, 2], it3, hv3, it3o, hv3o; leg=leg, other_leg=other_leg, show_other_leg=show_other_leg, color_fn=_swave_leg_color, xlabel="", ylabel=_corr_value_label(history_onsite_expr, use_abs), title=history_onsite_title, kwargs...)
    _plot_series!(axes[4, 1], data.r4, data.v4; xlabel="", ylabel=_corr_value_label(profile_extended_swave_expr, use_abs), title="extended s-wave: " * _extended_swave_corr_title(leg1, leg2; include_singlet_def=false), color=_EXT_SWAVE_COLOR, kwargs...)
    extended_swave_profile_line = axes[4, 1].lines[end]
    _plot_series!(axes[4, 2], it4, hv4; xlabel="", ylabel=_corr_value_label(history_extended_swave_expr, use_abs), title="extended s-wave: " * _extended_swave_corr_title(leg1, leg2; rung_tex=history_rung_tex, include_singlet_def=false), color=_EXT_SWAVE_COLOR, kwargs...)
    _plot_series!(axes[5, 1], data.r5, data.v5; xlabel=_rung_index_xlabel(), ylabel=_corr_value_label(profile_dwave_expr, use_abs), title="d-wave: " * _dwave_corr_title(leg1, leg2; include_singlet_def=false), color=_DWAVE_COLOR, kwargs...)
    dwave_profile_line = axes[5, 1].lines[end]
    _plot_series!(axes[5, 2], it5, hv5; xlabel=_mf_iteration_xlabel(), ylabel=_corr_value_label(history_dwave_expr, use_abs), title="d-wave: " * _dwave_corr_title(leg1, leg2; rung_tex=history_rung_tex, include_singlet_def=false), color=_DWAVE_COLOR, kwargs...)
    markers = [_add_iteration_marker!(axes[row, 2], profile_iteration) for row in 1:5]

    if has_slider
        _set_profile_axis_limits!(axes[1, 1], profile_history, :r1, :v1; other_xfield=:r1o, other_yfield=:v1o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[2, 1], profile_history, :r2, :v2; other_xfield=:r2o, other_yfield=:v2o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[3, 1], profile_history, :r3, :v3; other_xfield=:r3o, other_yfield=:v3o, show_other_leg=show_other_leg)
        _set_profile_axis_limits!(axes[4, 1], profile_history, :r4, :v4)
        _set_profile_axis_limits!(axes[5, 1], profile_history, :r5, :v5)
    end

    tight_rect = _profile_tight_rect(figure_title, has_slider)
    _finish_figure(fig; figure_title=figure_title, tight_rect=tight_rect, h_pad=_MF_STACK_H_PAD)
    if has_slider
        function update_iteration(it)
            updated = profile_history[it]
            _set_line_data!(cdw_profile_lines.main, updated.r1, updated.v1)
            show_other_leg && _set_line_data!(cdw_profile_lines.other, updated.r1o, updated.v1o)
            _set_line_data!(sdw_profile_lines.main, updated.r2, updated.v2)
            show_other_leg && _set_line_data!(sdw_profile_lines.other, updated.r2o, updated.v2o)
            _set_line_data!(swave_profile_lines.main, updated.r3, updated.v3)
            show_other_leg && _set_line_data!(swave_profile_lines.other, updated.r3o, updated.v3o)
            _set_line_data!(extended_swave_profile_line, updated.r4, updated.v4)
            _set_line_data!(dwave_profile_line, updated.r5, updated.v5)
            for marker in markers
                _set_iteration_marker!(marker, it)
            end
            return nothing
        end
        _install_iteration_slider!(fig, niter, profile_iteration, update_iteration; figure_title=figure_title, target_axes=[axes[row, 1] for row in 1:5])
    end
    return _save_if_requested(fig, savepath)
end

function plot_mf_profiles_from_file(filename::AbstractString; iteration=nothing, source::Symbol=:mf, use_correlations::Bool=false, savepath=nothing, kwargs...)
    data = load_mf_data(filename)
    if _source(source, use_correlations) == :correlations
        C_pair_source = iteration === nothing && data.C_pair_list !== nothing ? data.C_pair_list :
            ((iteration === nothing && data.C_pair !== nothing) ? data.C_pair : data.C_pair_list)
        C_exc_dn_source = iteration === nothing && data.C_exc_dn_list !== nothing ? data.C_exc_dn_list :
            ((iteration === nothing && data.C_exc_dn !== nothing) ? data.C_exc_dn : data.C_exc_dn_list)
        C_exc_up_source = iteration === nothing && data.C_exc_up_list !== nothing ? data.C_exc_up_list :
            ((iteration === nothing && data.C_exc_up !== nothing) ? data.C_exc_up : data.C_exc_up_list)
        return plot_correlation_profiles(C_pair_source, C_exc_dn_source, C_exc_up_source; iteration=iteration, figure_title=_filename_title(filename), savepath=savepath, kwargs...)
    else
        alpha_source = iteration === nothing && data.alpha_list !== nothing ? data.alpha_list :
            ((iteration === nothing && data.alpha !== nothing) ? data.alpha : data.alpha_list)
        beta_source = iteration === nothing && data.beta_list !== nothing ? data.beta_list :
            ((iteration === nothing && data.beta !== nothing) ? data.beta : data.beta_list)
        return plot_mf_profiles(alpha_source, beta_source; iteration=iteration, figure_title=_filename_title(filename), savepath=savepath, kwargs...)
    end
end

function plot_mf_profiles_and_middle_histories_from_file(filename::AbstractString; iteration=nothing, source::Symbol=:mf, use_correlations::Bool=false, savepath=nothing, kwargs...)
    data = load_mf_data(filename)
    if _source(source, use_correlations) == :correlations
        C_pair_source = iteration === nothing && data.C_pair_list !== nothing ? data.C_pair_list :
            ((iteration === nothing && data.C_pair !== nothing) ? data.C_pair : data.C_pair_list)
        C_exc_dn_source = iteration === nothing && data.C_exc_dn_list !== nothing ? data.C_exc_dn_list :
            ((iteration === nothing && data.C_exc_dn !== nothing) ? data.C_exc_dn : data.C_exc_dn_list)
        C_exc_up_source = iteration === nothing && data.C_exc_up_list !== nothing ? data.C_exc_up_list :
            ((iteration === nothing && data.C_exc_up !== nothing) ? data.C_exc_up : data.C_exc_up_list)
        return plot_correlation_profiles_and_middle_histories(
            C_pair_source,
            C_exc_dn_source,
            C_exc_up_source,
            data.C_pair_list,
            data.C_exc_dn_list,
            data.C_exc_up_list;
            iteration=iteration,
            figure_title=_filename_title(filename),
            savepath=savepath,
            kwargs...)
    else
        alpha_source = iteration === nothing && data.alpha_list !== nothing ? data.alpha_list :
            ((iteration === nothing && data.alpha !== nothing) ? data.alpha : data.alpha_list)
        beta_source = iteration === nothing && data.beta_list !== nothing ? data.beta_list :
            ((iteration === nothing && data.beta !== nothing) ? data.beta : data.beta_list)
        return plot_mf_profiles_and_middle_histories(
            alpha_source,
            beta_source,
            data.alpha_list,
            data.beta_list;
            iteration=iteration,
            figure_title=_filename_title(filename),
            savepath=savepath,
            kwargs...)
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

function plot_order_fourier_heatmaps_from_file(filename::AbstractString; iteration=nothing, source::Symbol=:mf, use_correlations::Bool=false, trim_boundary_rungs::Integer=5, savepath=nothing, kwargs...)
    data = load_mf_data(filename)
    if _source(source, use_correlations) == :correlations
        C_pair_source = (iteration === nothing && data.C_pair !== nothing) ? data.C_pair : data.C_pair_list
        C_exc_dn_source = (iteration === nothing && data.C_exc_dn !== nothing) ? data.C_exc_dn : data.C_exc_dn_list
        C_exc_up_source = (iteration === nothing && data.C_exc_up !== nothing) ? data.C_exc_up : data.C_exc_up_list
        return plot_order_fourier_heatmaps_from_correlations(C_pair_source, C_exc_dn_source, C_exc_up_source; iteration=iteration, trim_boundary_rungs=trim_boundary_rungs, figure_title=_filename_title(filename), savepath=savepath, kwargs...)
    else
        alpha_source = (iteration === nothing && data.alpha !== nothing) ? data.alpha : data.alpha_list
        beta_source = (iteration === nothing && data.beta !== nothing) ? data.beta : data.beta_list
        return plot_order_fourier_heatmaps(alpha_source, beta_source; iteration=iteration, trim_boundary_rungs=trim_boundary_rungs, figure_title=_filename_title(filename), savepath=savepath, kwargs...)
    end
end
