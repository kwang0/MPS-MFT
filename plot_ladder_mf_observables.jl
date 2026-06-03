using HDF5
import PyPlot

# Usage:
#   include("plot_ladder_mf_observables.jl")
#   plot_mf_profiles_from_file("stateless_data/results_...h5")
#   plot_middle_histories_from_file("stateless_data/results_...h5")
#   plot_mf_profiles_from_file("stateless_data/results_...h5"; source=:correlations)
#   plot_middle_histories_from_file("stateless_data/results_...h5"; source=:correlations)
# The file-based wrappers put the HDF5 basename in the PyPlot figure title.
#
# Array convention:
#   alpha[i, ip, leg, legp] and alpha_list[i, ip, leg, legp, iter]
#   beta[spin, i, ip, leg, legp] and beta_list[spin, i, ip, leg, legp, iter]
#   spin=1 is down, spin=2 is up. leg=1/2 are Julia array leg indices.
#   C_pair_list[site_up, site_dn, iter], C_exc_*_list[site_from, site_to, iter]
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

function density_from_beta(beta; spin=:up, leg::Integer=1, iteration=nothing, use_abs::Bool=false)
    b = _final_array(beta, 5, iteration, "beta")
    sigma = _spin_index(spin)
    leg = _check_leg(leg)
    L = size(b, 2)
    rungs = collect(1:L)
    vals = [b[sigma, i, i, leg, leg] for i in rungs]
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
    ylabel = use_abs ? "|onsite alpha|" : "onsite alpha"
    fig, ax = _figure((8.5, 4.0))
    _plot_series!(ax, rungs, vals;
        xlabel="Rung index",
        ylabel=ylabel,
        title="On-site pairing from alpha, leg $leg",
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
        title=_onsite_pair_corr_title(leg),
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

    rungs, vals = rung_pairing_from_alpha(alpha; leg1=leg1, leg2=leg2, iteration=iteration, symmetrize=symmetrize, use_abs=use_abs)
    ylabel = use_abs ? "|rung alpha|" : "rung alpha"
    title_suffix = symmetrize ? "avg legs $leg1,$leg2" : "legs $leg1,$leg2"
    fig, ax = _figure((8.5, 4.0))
    _plot_series!(ax, rungs, vals;
        xlabel="Rung index",
        ylabel=ylabel,
        title="Rung pairing from alpha, $title_suffix",
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

    rungs, vals = rung_pairing_from_correlations(C_pair; leg1=leg1, leg2=leg2, iteration=iteration, symmetrize=symmetrize, use_abs=use_abs)
    expr = _rung_pair_corr_expr(; leg1_tex=string(leg1), leg2_tex=string(leg2), symmetrize=symmetrize)
    fig, ax = _figure((8.5, 4.0))
    _plot_series!(ax, rungs, vals;
        xlabel="Rung index \$i\$",
        ylabel=_corr_value_label(expr, use_abs),
        title=_rung_pair_corr_title(leg1, leg2; symmetrize=symmetrize),
        kwargs...)
    _finish_figure(fig; figure_title=figure_title)
    return _save_if_requested(fig, savepath)
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

    r1, v1 = density_from_beta(beta; spin=spin, leg=leg, iteration=iteration, use_abs=use_abs)
    r2, v2 = onsite_pairing_from_alpha(alpha; leg=leg, iteration=iteration, use_abs=use_abs)
    r3, v3 = rung_pairing_from_alpha(alpha; leg1=leg1, leg2=leg2, iteration=iteration, symmetrize=symmetrize_rung, use_abs=use_abs)

    fig, axes = PyPlot.subplots(3, 1, figsize=(8.5, 8.5))
    density_ylabel = use_abs ? "|onsite beta $( _spin_label(spin) )|" : "onsite beta $(_spin_label(spin))"
    onsite_ylabel = use_abs ? "|onsite alpha|" : "onsite alpha"
    rung_ylabel = use_abs ? "|rung alpha|" : "rung alpha"
    rung_suffix = symmetrize_rung ? "avg legs $leg1,$leg2" : "legs $leg1,$leg2"

    _plot_series!(axes[1], r1, v1; xlabel="Rung index", ylabel=density_ylabel, title="Density proxy from beta, leg $leg", kwargs...)
    _plot_series!(axes[2], r2, v2; xlabel="Rung index", ylabel=onsite_ylabel, title="On-site pairing from alpha, leg $leg", kwargs...)
    _plot_series!(axes[3], r3, v3; xlabel="Rung index", ylabel=rung_ylabel, title="Rung pairing from alpha, $rung_suffix", kwargs...)

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

    r1, v1 = density_from_correlations(C_exc_dn, C_exc_up; spin=spin, leg=leg, iteration=iteration, use_abs=use_abs)
    r2, v2 = onsite_pairing_from_correlations(C_pair; leg=leg, iteration=iteration, use_abs=use_abs)
    r3, v3 = rung_pairing_from_correlations(C_pair; leg1=leg1, leg2=leg2, iteration=iteration, symmetrize=symmetrize_rung, use_abs=use_abs)

    fig, axes = PyPlot.subplots(3, 1, figsize=(8.5, 8.5))
    density_expr = _density_corr_expr(spin; leg_tex=string(leg))
    onsite_expr = _onsite_pair_corr_expr(; leg_tex=string(leg))
    rung_expr = _rung_pair_corr_expr(; leg1_tex=string(leg1), leg2_tex=string(leg2), symmetrize=symmetrize_rung)

    _plot_series!(axes[1], r1, v1; xlabel="Rung index \$i\$", ylabel=_corr_value_label(density_expr, use_abs), title=_density_corr_title(spin, leg), kwargs...)
    _plot_series!(axes[2], r2, v2; xlabel="Rung index \$i\$", ylabel=_corr_value_label(onsite_expr, use_abs), title=_onsite_pair_corr_title(leg), kwargs...)
    _plot_series!(axes[3], r3, v3; xlabel="Rung index \$i\$", ylabel=_corr_value_label(rung_expr, use_abs), title=_rung_pair_corr_title(leg1, leg2; symmetrize=symmetrize_rung), kwargs...)

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
    rung_to_plot = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")

    it1, v1 = middle_density_history(beta_list; spin=spin, leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it2, v2 = middle_onsite_pairing_history(alpha_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it3, v3 = middle_rung_pairing_history(alpha_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, symmetrize=symmetrize_rung, use_abs=use_abs)

    fig, axes = PyPlot.subplots(3, 1, figsize=(8.5, 8.5))
    density_ylabel = use_abs ? "|onsite beta $( _spin_label(spin) )|" : "onsite beta $(_spin_label(spin))"
    onsite_ylabel = use_abs ? "|onsite alpha|" : "onsite alpha"
    rung_ylabel = use_abs ? "|rung alpha|" : "rung alpha"

    _plot_series!(axes[1], it1, v1; xlabel="MF iteration", ylabel=density_ylabel, title="Middle rung $rung_to_plot density proxy", kwargs...)
    _plot_series!(axes[2], it2, v2; xlabel="MF iteration", ylabel=onsite_ylabel, title="Middle rung $rung_to_plot on-site pairing", kwargs...)
    _plot_series!(axes[3], it3, v3; xlabel="MF iteration", ylabel=rung_ylabel, title="Middle rung $rung_to_plot rung pairing", kwargs...)

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
    rung_to_plot = rung === nothing ? middle_rung(L) : _check_index(rung, L, "rung")

    it1, v1 = middle_density_history_from_correlations(C_exc_dn_list, C_exc_up_list; spin=spin, leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it2, v2 = middle_onsite_pairing_history_from_correlations(C_pair_list; leg=leg, rung=rung_to_plot, use_abs=use_abs)
    it3, v3 = middle_rung_pairing_history_from_correlations(C_pair_list; rung=rung_to_plot, leg1=leg1, leg2=leg2, symmetrize=symmetrize_rung, use_abs=use_abs)

    fig, axes = PyPlot.subplots(3, 1, figsize=(8.5, 8.5))
    rung_tex = string(rung_to_plot)
    density_expr = _density_corr_expr(spin; rung_tex=rung_tex, leg_tex=string(leg))
    onsite_expr = _onsite_pair_corr_expr(; rung_tex=rung_tex, leg_tex=string(leg))
    rung_expr = _rung_pair_corr_expr(; rung_tex=rung_tex, leg1_tex=string(leg1), leg2_tex=string(leg2), symmetrize=symmetrize_rung)

    _plot_series!(axes[1], it1, v1; xlabel="MF iteration \$m\$", ylabel=_corr_value_label(density_expr, use_abs), title=_density_corr_title(spin, leg; rung_tex=rung_tex), kwargs...)
    _plot_series!(axes[2], it2, v2; xlabel="MF iteration \$m\$", ylabel=_corr_value_label(onsite_expr, use_abs), title=_onsite_pair_corr_title(leg; rung_tex=rung_tex), kwargs...)
    _plot_series!(axes[3], it3, v3; xlabel="MF iteration \$m\$", ylabel=_corr_value_label(rung_expr, use_abs), title=_rung_pair_corr_title(leg1, leg2; rung_tex=rung_tex, symmetrize=symmetrize_rung), kwargs...)

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
