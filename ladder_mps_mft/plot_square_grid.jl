#=
Plot the compiled square chi=200 HDF5 with the original Fourier-grid renderer.

    include("ladder_mps_mft/plot_square_grid.jl")
    grid = plot_square_grid()                     # physical correlations
    grid = plot_square_grid(; source=:mf)         # measured MF field proxies

Pass a bundle as the first positional argument. Hover reports Fourier maxima;
click opens the Fourier maps and full MF profiles with middle histories and
the iteration slider. Temporary plotting
copies live until Julia exits, so the returned figure remains interactive.

Local batch render (uses the Julia environment with HDF5, PyCall, PyPlot):
    julia --startup-file=no ladder_mps_mft/plot_square_grid.jl [BUNDLE] [OUTDIR]
=#
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    ENV["MPLBACKEND"] = "Agg"
end
if !isdefined(@__MODULE__, :plot_phase1_mf_profiles_and_middle_histories)
    include(joinpath(@__DIR__, "plot_phase1_mf_observables.jl"))
end

const DEFAULT_SQUARE_GRID_BUNDLE = joinpath(@__DIR__, "output", "square_grid_chi200_20260908", "square_grid_chi200.h5")

function _square_grid_history_file(bundle, row)
    filename = joinpath(dirname(row.filename), "history_$(row.point_id).h5")
    isfile(filename) && return filename
    h5open(bundle, "r") do f
        snapshot = f["points/$(row.point_id)/source_snapshot"]
        h5open(filename, "w") do out
            for key in keys(snapshot)
                HDF5.copy_object(snapshot, key, out, key)
            end
        end
    end
    return filename
end

function _square_grid_legacy_histories(filename; source, figure_title)
    h5open(filename, "r") do f
        if _source(source, false) == :correlations
            pair, down, up = (read(f, k) for k in ("C_pair_list", "C_exc_dn_list", "C_exc_up_list"))
            return plot_correlation_profiles_and_middle_histories(pair, down, up, pair, down, up; figure_title)
        end
        alpha = read(f, "alpha_list")
        beta = _p1_legacy_beta_history(read(f, "beta_list"), read(f, "mu_cdw_list"))
        fig = plot_mf_profiles_and_middle_histories(alpha, beta, alpha, beta; figure_title)
        _p1_relabel_profile_axes!(fig; profile_and_history=true)
        return fig
    end
end

function _square_grid_mf_history_titles!(fig)
    # Keep the two-column view readable at the legacy figure size.
    for (index, title) in ((7, "Extended s-wave MF profile"), (8, "Middle rung extended s-wave MF history"),
        (9, "d-wave MF profile"), (10, "Middle rung d-wave MF history"))
        fig.axes[index].set_title(title)
    end
    return fig
end

function plot_square_grid(bundle::AbstractString=DEFAULT_SQUARE_GRID_BUNDLE;
    source::Symbol=:correlations, savepath=nothing, click_detail_plots::Bool=true,
    figure_title=nothing, trim_boundary_rungs::Integer=5, kwargs...)
    bundle = abspath(bundle)
    cache = mktempdir(; prefix="square-grid-")
    metadata = []
    h5open(bundle, "r") do f
        read(f, "artifact_kind") in ("square_grid_bundle", "square_grid_terminal_snapshot_bundle") || error("Not a square-grid bundle")
        read(f, "schema_version") >= 2 || error("This older bundle omits MF histories. Rebuild it with scripts/compile_square_grid.py.")
        for name in sort(collect(keys(f["points"])))
            p = f["points/$name"]
            filename = joinpath(cache, read(p, "plot_filename"))
            h5open(filename, "w") do out
                for key in keys(p["plot_data"])
                    out[key] = read(p["plot_data"], key)
                end
            end
            push!(metadata, (; filename, point_id=name, t0=read(p, "t0"), V=read(p, "V"),
                quality=read(p, "quality"), status=read(p, "status"), source=read(p, "source")))
        end
    end
    detail = source == :correlations ? "physical correlations" : "measured MF field proxies"
    title = something(figure_title, "Square Fourier order maxima ($detail)\nL=64, U=8, t_perp=0.1, n=0.9375, chi=200; trim=$trim_boundary_rungs rungs/end")
    result = plot_order_fourier_max_grid(; data_dir=cache, transverse_geometry=:square,
        t0_values=[1., 1.2, 1.4], t0_min=1., t0_max=1.4, source,
        figure_title=title, trim_boundary_rungs, savepath=nothing, click_detail_plots=false, kwargs...)
    if source != :correlations
        for (text, label) in zip(result.legend.get_texts(),
            ("CDW Hartree proxy", "SDW Hartree proxy", "On-site s-wave alpha proxy", "Extended s-wave alpha proxy", "d-wave alpha proxy"))
            text.set_text(label)
        end
    end
    for row in metadata
        if row.quality != "accepted"
            label = row.quality == "diverging" ? "DIVERGING" : "LEGACY"
            result.ax.text(row.t0, row.V + 0.087, label; ha="center", va="top", fontsize=8,
                zorder=10, bbox=Dict("facecolor"=>"white", "edgecolor"=>"none", "alpha"=>0.9, "pad"=>1))
            if row.quality == "diverging"
                result.ax.add_patch(_MatplotlibPatches.Rectangle((row.t0-.1, row.V-.1), .2, .2;
                    fill=false, hatch="///", edgecolor="0.45", linewidth=1.2, zorder=6))
            end
        end
    end
    result.fig.text(0.11, 0.012, "LEGACY: historical completion; DIVERGING: terminal diagnostic, unconverged. Snapshot: 2026-09-08.";
        fontsize=8, ha="left")
    # Load complete native histories lazily, keeping the grid itself lightweight.
    callback = function(event)
        (event.inaxes != result.ax || event.xdata === nothing || event.ydata === nothing) && return
        for row in metadata
            if abs(Float64(event.xdata)-row.t0) <= .1 && abs(Float64(event.ydata)-row.V) <= .1
                label = "Square chi=200: t0=$(row.t0), V=$(row.V) [$(row.status)]\n$detail; terminal snapshot"
                heat = plot_order_fourier_heatmaps_from_file(row.filename; source, trim_boundary_rungs, figure_title=label)
                history_file = _square_grid_history_file(bundle, row)
                history_label = "Square chi=200: t0=$(row.t0), V=$(row.V) [$(row.status)]"
                profiles = if row.quality == "legacy"
                    _square_grid_legacy_histories(history_file; source,
                        figure_title=history_label * "\nComplete saved legacy histories ($detail)")
                else
                    plot_phase1_mf_profiles_and_middle_histories(history_file;
                        stitch_parent_history=false,
                        figure_title=history_label * "\nComplete measured MF history; initial seed at iteration 1, then all updates")
                end
                (row.quality != "legacy" || _source(source, false) == :mf) && _square_grid_mf_history_titles!(profiles)
                push!(result.clicked_heatmap_figures, heat)
                push!(result.clicked_detail_figures, profiles)
                return
            end
        end
    end
    callback_id = click_detail_plots ? result.fig.canvas.mpl_connect("button_press_event", callback) : nothing
    _FIGURE_CALLBACK_REFS[result.fig] = (callback, result, metadata, cache)
    _save_if_requested(result.fig, savepath)
    return merge(result, (; bundle=abspath(bundle), metadata, cache_directory=cache,
        click_callback=callback, click_callback_id=callback_id))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    bundle = isempty(ARGS) ? DEFAULT_SQUARE_GRID_BUNDLE : ARGS[1]
    outdir = length(ARGS) < 2 ? dirname(bundle) : ARGS[2]
    mkpath(outdir)
    for source in (:correlations, :mf)
        result = plot_square_grid(bundle; source, hover=false, click_detail_plots=false)
        for extension in ("png", "pdf")
            result.fig.savefig(joinpath(outdir, "fourier_grid_$(source).$extension"); dpi=200, bbox_inches="tight")
        end
        open(joinpath(outdir, "fourier_maxima_$(source).csv"), "w") do io
            println(io, "t0,V,channel,amplitude,kx,ky")
            for rec in result.data, channel in propertynames(rec.maxima)
                m = getproperty(rec.maxima, channel)
                println(io, join((rec.t0, rec.V0, channel, m.value, m.kx, Float64(something(m.ky, NaN))), ","))
            end
        end
        PyPlot.close(result.fig)
    end
end
