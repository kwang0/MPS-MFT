"""
Plot the mean-field profiles saved by the refactored Phase 1 campaign.

This is an adapter for `../plot_ladder_mf_observables.jl`; it does not change
the legacy plotting code. The refactored states store density Hartree fields in
`mu_cdw`, rather than on the diagonal of `beta`, so this adapter maps
`mu_cdw[down/up]` onto the legacy CDW/SDW profile slots before plotting.

Interactive use from the MPS-MFT checkout:

    include("ladder_mps_mft/plot_phase1_mf_observables.jl")
    states = phase1_campaign_states()
    state = states["frustrated__pairing_s1"]

    plot_phase1_mf_profiles_and_middle_histories(state)
    plot_phase1_seed_profiles(state)
    plot_phase1_mf_profiles_from_file(state; source=:applied)

Schema-v7 files additionally store the exact time-zero seed inside the field
history; schema-v5/v6 files use their equivalent `fields/initial` record.
In profile/history figures that seed is displayed as MF iteration 1, followed
by the fields measured after each stored MF update.
When a locally synced `parent_checkpoint` has a complete measured-field
history, the profile/history plot prepends that history, removes the duplicated
handoff field, and marks the start of the continuation segment.
For older v2 files, the adapter falls back to the saved seed, best
checkpoint, and final/orbit snapshots and explicitly labels them as sparse
saved snapshots rather than a continuous MF history.

Run this file as a script to render all nine states:

    julia ladder_mps_mft/plot_phase1_mf_observables.jl [RUN_DIR] [OUTPUT_DIR]
"""

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && !haskey(ENV, "MPLBACKEND")
    ENV["MPLBACKEND"] = "Agg"
end

using HDF5
using Random

include(joinpath(@__DIR__, "..", "plot_ladder_mf_observables.jl"))

const DEFAULT_PHASE1_RUN_DIRECTORY = joinpath(
    @__DIR__,
    "output",
    "phase1_gpu",
    "20260823_phase1_gpu_v2",
)

_p1_site(rung::Integer, leg::Integer) = 2 * (rung - 1) + leg

function _p1_string(file, path::AbstractString; default::AbstractString="")
    haskey(file, path) || return String(default)
    return String(read(file, path))
end

function _p1_int(file, path::AbstractString; default::Integer=0)
    haskey(file, path) || return Int(default)
    return Int(read(file, path))
end

function _p1_float(file, path::AbstractString; default::Real=NaN)
    haskey(file, path) || return Float64(default)
    return Float64(read(file, path))
end

function _p1_bool(file, path::AbstractString; default::Bool=false)
    haskey(file, path) || return default
    return Bool(read(file, path))
end

function _p1_read_fields(group)
    return (
        alpha=Float64.(read(group, "alpha")),
        beta=Float64.(read(group, "beta")),
        mu_cdw=Float64.(read(group, "mu_cdw")),
    )
end

function _p1_read_fields(file, path::AbstractString)
    haskey(file, path) || throw(ArgumentError("state has no field group '$path'"))
    return _p1_read_fields(file[path])
end

function _p1_metadata(file)
    return (
        L=_p1_int(file, "model/L"),
        r_range=_p1_int(file, "model/r_range"),
        geometry=_p1_string(file, "model/transverse_geometry"),
        target_density=_p1_float(file, "model/density"),
        seed=_p1_string(file, "provenance/initial_seed"),
        seed_label=_p1_string(file, "provenance/seed_label"),
        random_seed=_p1_int(file, "provenance/random_seed"),
        initial_amplitude=_p1_float(file, "provenance/initial_amplitude"),
        initial_state_source=_p1_string(file, "provenance/initial_state_source"),
        inherit_from=_p1_string(file, "provenance/inherit_from"),
        inherit_format=_p1_string(file, "provenance/inherit_format"),
        parent_checkpoint=_p1_string(file, "provenance/parent_checkpoint"),
        resume_checkpoint=_p1_string(file, "provenance/resume_checkpoint"),
        status=_p1_string(file, "status"),
        accepted=_p1_bool(file, "accepted"),
        period=_p1_int(file, "fundamental_period"),
    )
end

"""Return the newest `state.h5` for each campaign label."""
function phase1_campaign_states(run_directory::AbstractString=DEFAULT_PHASE1_RUN_DIRECTORY)
    absolute_run_directory = abspath(run_directory)
    migrated_stateless_directory = joinpath(absolute_run_directory, "stateless_results")
    results_directory = isdir(migrated_stateless_directory) ? migrated_stateless_directory :
        joinpath(absolute_run_directory, "results")
    isdir(results_directory) || throw(ArgumentError(
        "Phase 1 results directory does not exist: $results_directory",
    ))

    states = Dict{String,String}()
    for (root, _, files) in walkdir(results_directory)
        "state.h5" in files || continue
        path = joinpath(root, "state.h5")
        components = splitpath(relpath(path, results_directory))
        isempty(components) && continue
        label = first(components)
        if !haskey(states, label) || path > states[label]
            states[label] = path
        end
    end
    isempty(states) && throw(ArgumentError("no state.h5 files found below $results_directory"))
    return states
end

function phase1_state_file(
    label::AbstractString;
    run_directory::AbstractString=DEFAULT_PHASE1_RUN_DIRECTORY,
)
    states = phase1_campaign_states(run_directory)
    haskey(states, label) || throw(ArgumentError(
        "unknown Phase 1 label '$label'; available labels are $(join(sort!(collect(keys(states))), ", "))",
    ))
    return states[String(label)]
end

function _p1_independent_seed_fields(metadata)
    L = metadata.L
    r_range = metadata.r_range
    amplitude = metadata.initial_amplitude
    rng = MersenneTwister(metadata.random_seed)
    alpha = zeros(Float64, L, L, 2, 2)
    beta = zeros(Float64, 2, L, L, 2, 2)
    mu_cdw = zeros(Float64, 2, 2 * L)
    seed = Symbol(lowercase(metadata.seed))

    if seed == :pairing
        for rung in 1:L, offset in 0:r_range
            target = rung + offset
            target <= L || continue
            for leg in 1:2, other_leg in 1:2
                offset == 0 && other_leg < leg && continue
                value = amplitude * (2rand(rng) - 1)
                alpha[rung, target, leg, other_leg] = value
                alpha[target, rung, other_leg, leg] = value
            end
        end
    elseif seed == :sdw
        for rung in 1:L, leg0 in 0:1
            site = _p1_site(rung, leg0 + 1)
            phase = isodd(rung + leg0) ? -1.0 : 1.0
            mu_cdw[1, site] = amplitude * phase
            mu_cdw[2, site] = -amplitude * phase
        end
    elseif seed == :cdw
        for site in 1:(2 * L)
            phase = isodd(div(site - 1, 2)) ? 1.0 : -1.0
            mu_cdw[:, site] .= amplitude * phase
        end
    elseif seed != :zero
        throw(ArgumentError("unsupported initial seed '$seed'"))
    end

    return (; alpha, beta, mu_cdw)
end

function _p1_local_artifact_path(recorded_path::AbstractString)
    isempty(recorded_path) && return ""
    isfile(recorded_path) && return abspath(recorded_path)

    normalized = replace(String(recorded_path), '\\' => '/')
    marker = "output/"
    position = findfirst(marker, normalized)
    if position !== nothing
        relative = normalized[first(position):end]
        candidate = joinpath(@__DIR__, split(relative, '/')...)
        isfile(candidate) && return candidate
    end

    marker = "phase1_gpu/"
    position = findfirst(marker, normalized)
    if position !== nothing
        relative = normalized[first(position):end]
        candidate = joinpath(@__DIR__, "output", split(relative, '/')...)
        isfile(candidate) && return candidate
    end
    return ""
end

function phase1_seed_fields(state_file::AbstractString; parent_path=nothing)
    isfile(state_file) || throw(ArgumentError("state file does not exist: $state_file"))
    metadata, saved_initial = h5open(state_file, "r") do file
        return (
            _p1_metadata(file),
            haskey(file, "fields/initial") ? _p1_read_fields(file, "fields/initial") : nothing,
        )
    end
    saved_initial !== nothing && return saved_initial

    metadata.initial_state_source == "independent" &&
        return _p1_independent_seed_fields(metadata)

    recorded = if parent_path !== nothing
        String(parent_path)
    elseif metadata.initial_state_source == "field_inherit"
        metadata.inherit_from
    elseif metadata.initial_state_source == "resume"
        metadata.resume_checkpoint
    else
        metadata.parent_checkpoint
    end
    local_path = _p1_local_artifact_path(recorded)
    isempty(local_path) && throw(ArgumentError(
        "could not locate the recorded $(metadata.initial_state_source) seed '$recorded' locally; " *
        "pass parent_path=... explicitly",
    ))
    return h5open(local_path, "r") do file
        if haskey(file, "fields/restart")
            return _p1_read_fields(file, "fields/restart")
        end
        haskey(file, "alpha") && haskey(file, "beta") || throw(ArgumentError(
            "recorded field-inherit source has neither fields/restart nor legacy alpha/beta",
        ))
        alpha = Float64.(read(file, "alpha"))
        L = size(alpha, 1)
        return (
            alpha,
            beta=Float64.(read(file, "beta")),
            mu_cdw=haskey(file, "mu_cdw") ? Float64.(read(file, "mu_cdw")) :
                zeros(Float64, 2, 2 * L),
        )
    end
end

# The legacy plotter reads CDW/SDW profiles from the diagonal of beta. In the
# refactored model those local spin-resolved Hartree fields live in mu_cdw and
# the true beta diagonal is intentionally zero. Only the plotting copy is
# changed here; off-diagonal exchange fields remain untouched.
function _p1_legacy_beta(fields)
    beta = copy(fields.beta)
    L = size(fields.alpha, 1)
    size(fields.mu_cdw) == (2, 2 * L) || throw(ArgumentError(
        "mu_cdw must have shape (2, $(2 * L)); got $(size(fields.mu_cdw))",
    ))
    for rung in 1:L, leg in 1:2
        site = _p1_site(rung, leg)
        beta[1, rung, rung, leg, leg] = fields.mu_cdw[1, site]
        beta[2, rung, rung, leg, leg] = fields.mu_cdw[2, site]
    end
    return beta
end

function _p1_legacy_beta_history(beta, mu_cdw)
    ndims(beta) == 6 || throw(ArgumentError("beta history must be rank 6"))
    ndims(mu_cdw) == 3 || throw(ArgumentError("mu_cdw history must be rank 3"))
    L = size(beta, 2)
    niter = size(beta, 6)
    size(mu_cdw) == (2, 2 * L, niter) || throw(ArgumentError(
        "mu_cdw history must have shape (2, $(2 * L), $niter); got $(size(mu_cdw))",
    ))
    mapped = copy(beta)
    for iteration in 1:niter, rung in 1:L, leg in 1:2
        site = _p1_site(rung, leg)
        mapped[1, rung, rung, leg, leg, iteration] = mu_cdw[1, site, iteration]
        mapped[2, rung, rung, leg, leg, iteration] = mu_cdw[2, site, iteration]
    end
    return mapped
end

function _p1_prepend_history_sample(sample, history)
    size(sample) == size(history)[1:(end - 1)] || throw(DimensionMismatch(
        "time-zero MF seed shape does not match the stored field history",
    ))
    return cat(sample, history; dims=ndims(history))
end

function _p1_complete_history(
    state_file::AbstractString,
    source::Symbol;
    include_seed::Bool=true,
)
    source in (:applied, :measured) || throw(ArgumentError(
        "history_source must be :applied or :measured",
    ))
    return h5open(state_file, "r") do file
        base = "history/fields/$(String(source))"
        haskey(file, base) || return nothing
        iterations = Int.(read(file, "history/iteration"))
        alpha = Float64.(read(file, "$base/alpha"))
        beta = Float64.(read(file, "$base/beta"))
        mu_cdw = Float64.(read(file, "$base/mu_cdw"))
        if include_seed
            seed_base = haskey(file, "history/fields/seed") ?
                "history/fields/seed" : "fields/initial"
            haskey(file, seed_base) || throw(ArgumentError(
                "complete MF history has no saved time-zero seed: $state_file",
            ))
            seed_iteration = haskey(file, "history/fields/seed_iteration") ?
                Int(read(file, "history/fields/seed_iteration")) : 0
            seed_iteration == 0 || throw(ArgumentError(
                "stored MF seed iteration must be zero; got $seed_iteration",
            ))
            seed = _p1_read_fields(file, seed_base)
            alpha = _p1_prepend_history_sample(seed.alpha, alpha)
            beta = _p1_prepend_history_sample(seed.beta, beta)
            mu_cdw = _p1_prepend_history_sample(seed.mu_cdw, mu_cdw)
            iterations = [seed_iteration; iterations]
        end
        return (; iterations, alpha, beta, mu_cdw)
    end
end

function _p1_final_iteration(file)
    haskey(file, "history/iteration") || return 0
    iterations = Int.(read(file, "history/iteration"))
    return isempty(iterations) ? 0 : last(iterations)
end

function _p1_saved_snapshots(state_file::AbstractString; include_seed::Bool=true, parent_path=nothing)
    snapshots = NamedTuple[]
    if include_seed
        push!(snapshots, (
            label="seed",
            iteration=0,
            role=:seed,
            fields=phase1_seed_fields(state_file; parent_path),
        ))
    end

    best_path = joinpath(dirname(state_file), "checkpoint_best.h5")
    best_snapshot = if isfile(best_path)
        h5open(best_path, "r") do file
            (
                label="best",
                iteration=_p1_final_iteration(file),
                role=:best_measured,
                fields=_p1_read_fields(file, "fields/measured"),
            )
        end
    else
        nothing
    end

    terminal_snapshots = h5open(state_file, "r") do file
        values = NamedTuple[]
        if haskey(file, "cycle_members")
            names = sort!(String.(collect(keys(file["cycle_members"]))))
            for (phase, name) in enumerate(names)
                group = file["cycle_members/$name"]
                push!(values, (
                    label="p$(phase)",
                    iteration=_p1_int(group, "iteration"),
                    role=:orbit_measured,
                    fields=_p1_read_fields(group, "measured"),
                ))
            end
        else
            iteration = _p1_final_iteration(file)
            push!(values, (
                label="final",
                iteration,
                role=:final_measured,
                fields=_p1_read_fields(file, "fields/measured"),
            ))
        end
        return values
    end

    terminal_iterations = Set(snapshot.iteration for snapshot in terminal_snapshots)
    if best_snapshot !== nothing && !(best_snapshot.iteration in terminal_iterations)
        push!(snapshots, best_snapshot)
    end
    append!(snapshots, terminal_snapshots)
    sort!(snapshots; by=snapshot -> (snapshot.iteration, String(snapshot.label)))
    return snapshots
end

"""
Return the saved field snapshots used by the profile/history adapter.

For v2 these are the reconstructed seed, the best measured checkpoint when it
is distinct, and the measured final state or measured period-two phases.
"""
phase1_saved_mf_snapshots(state_file::AbstractString; kwargs...) =
    _p1_saved_snapshots(abspath(state_file); kwargs...)

function _p1_snapshot_fields(
    state_file::AbstractString,
    source::Symbol;
    cycle_phase::Integer=1,
    parent_path=nothing,
)
    source == :seed && return phase1_seed_fields(state_file; parent_path)
    return h5open(state_file, "r") do file
        if source in (:applied, :measured, :restart)
            return _p1_read_fields(file, "fields/$(String(source))")
        elseif source in (:cycle_applied, :cycle_measured)
            haskey(file, "cycle_members") || throw(ArgumentError(
                "state has no saved cycle members",
            ))
            names = sort!(String.(collect(keys(file["cycle_members"]))))
            1 <= cycle_phase <= length(names) || throw(ArgumentError(
                "cycle_phase must be between 1 and $(length(names)); got $cycle_phase",
            ))
            member = names[cycle_phase]
            field_kind = source == :cycle_applied ? "applied" : "measured"
            return _p1_read_fields(file, "cycle_members/$member/$field_kind")
        end
        throw(ArgumentError(
            "source must be :seed, :applied, :measured, :restart, :cycle_applied, or :cycle_measured",
        ))
    end
end

function _p1_figure_title(state_file::AbstractString, detail::AbstractString)
    metadata = h5open(state_file, "r") do file
        _p1_metadata(file)
    end
    return join((
        "$(metadata.geometry) | $(metadata.seed_label) | status=$(metadata.status) | accepted=$(metadata.accepted) | period=$(metadata.period)",
        detail,
    ), "\n")
end

function _p1_relabel_profile_axes!(fig; profile_and_history::Bool)
    axes = collect(fig.axes)
    main_axes = profile_and_history ? axes[1:min(10, length(axes))] : axes[1:min(5, length(axes))]
    row_axes = if profile_and_history
        ([main_axes[1], main_axes[2]], [main_axes[3], main_axes[4]])
    else
        ([main_axes[1]], [main_axes[2]])
    end

    for ax in row_axes[1]
        ax.set_ylabel("CDW Hartree field")
        ax.set_title("CDW Hartree field: mu_up + mu_down")
    end
    for ax in row_axes[2]
        ax.set_ylabel("SDW Hartree field")
        ax.set_title("SDW Hartree field: mu_up - mu_down")
    end
    return main_axes
end

function _p1_save_after_relabel(fig, savepath; dpi=nothing)
    if savepath !== nothing
        mkpath(dirname(abspath(String(savepath))))
        if dpi === nothing
            fig.savefig(savepath, bbox_inches="tight")
        else
            fig.savefig(savepath, bbox_inches="tight", dpi=dpi)
        end
    end
    return fig
end

"""Plot one saved Phase 1 field profile using the legacy five-row layout."""
function plot_phase1_mf_profiles_from_file(
    state_file::AbstractString;
    source::Symbol=:measured,
    cycle_phase::Integer=1,
    parent_path=nothing,
    savepath=nothing,
    dpi=nothing,
    figure_title=nothing,
    kwargs...,
)
    state_file = abspath(state_file)
    fields = _p1_snapshot_fields(
        state_file,
        source;
        cycle_phase,
        parent_path,
    )
    detail = source in (:cycle_applied, :cycle_measured) ?
        "profile=$(source), phase=$cycle_phase" : "profile=$(source)"
    title = something(figure_title, _p1_figure_title(state_file, detail))
    fig = plot_mf_profiles(
        fields.alpha,
        _p1_legacy_beta(fields);
        figure_title=title,
        savepath=nothing,
        kwargs...,
    )
    _p1_relabel_profile_axes!(fig; profile_and_history=false)
    return _p1_save_after_relabel(fig, savepath; dpi)
end

"""Plot the deterministic independent seed (or recorded parent seed) profile."""
function plot_phase1_seed_profiles(state_file::AbstractString; kwargs...)
    return plot_phase1_mf_profiles_from_file(state_file; source=:seed, kwargs...)
end

function _p1_snapshot_index(snapshot, snapshots)
    snapshot == :latest && return length(snapshots)
    snapshot isa Integer && begin
        1 <= snapshot <= length(snapshots) || throw(ArgumentError(
            "snapshot index must be between 1 and $(length(snapshots)); got $snapshot",
        ))
        return Int(snapshot)
    end
    requested = lowercase(String(snapshot))
    index = findfirst(item -> lowercase(String(item.label)) == requested, snapshots)
    index === nothing && throw(ArgumentError(
        "unknown snapshot '$snapshot'; available snapshots are $(join(getfield.(snapshots, :label), ", "))",
    ))
    return index
end

function _p1_relabel_snapshot_histories!(fig, snapshots)
    axes = _p1_relabel_profile_axes!(fig; profile_and_history=true)
    history_axes = [axes[2], axes[4], axes[6], axes[8], axes[10]]
    tick_labels = [
        snapshot.iteration == 0 ? snapshot.label : "$(snapshot.label) i$(snapshot.iteration)"
        for snapshot in snapshots
    ]
    positions = collect(1:length(snapshots))
    for ax in history_axes
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels; rotation=25, ha="right")
        for line in ax.lines
            line.set_linestyle("--")
        end
    end
    history_axes[end].set_xlabel("Saved MF snapshot (not a complete iteration history)")

    refs = get(_FIGURE_CALLBACK_REFS, fig, nothing)
    if refs !== nothing && hasproperty(refs, :slider)
        refs.slider.label.set_text("Saved snapshot")
    end
    return fig
end

function _p1_history_index(requested, iterations)
    isempty(iterations) && throw(ArgumentError("complete MF history is empty"))
    requested == :latest && return length(iterations)
    requested isa Integer || throw(ArgumentError(
        "snapshot must be :latest or an MF iteration number when complete history is available",
    ))
    index = findfirst(==(Int(requested)), iterations)
    index === nothing && throw(ArgumentError(
        "MF iteration $requested is not present; available range is $(first(iterations)) to $(last(iterations))",
    ))
    return index
end

function _p1_plot_iterations(iterations, include_seed::Bool)
    include_seed || return iterations
    isempty(iterations) && throw(ArgumentError("complete MF history is empty"))
    first(iterations) == 0 || throw(ArgumentError(
        "complete MF history with a seed must begin at stored iteration zero",
    ))
    return collect(1:length(iterations))
end

function _p1_history_endpoint(history, index::Integer)
    return (
        alpha=selectdim(history.alpha, ndims(history.alpha), index),
        beta=selectdim(history.beta, ndims(history.beta), index),
        mu_cdw=selectdim(history.mu_cdw, ndims(history.mu_cdw), index),
    )
end

function _p1_assert_same_fields(left, right, context::AbstractString)
    for component in (:alpha, :beta, :mu_cdw)
        left_values = getproperty(left, component)
        right_values = getproperty(right, component)
        size(left_values) == size(right_values) || throw(DimensionMismatch(
            "$context $(String(component)) shapes differ: $(size(left_values)) versus $(size(right_values))",
        ))
        left_values == right_values || throw(ArgumentError(
            "$context $(String(component)) values differ",
        ))
    end
    return nothing
end

function _p1_parent_history_path(state_file::AbstractString; parent_path=nothing)
    recorded = if parent_path !== nothing
        String(parent_path)
    else
        h5open(state_file, "r") do file
            _p1_string(file, "provenance/parent_checkpoint")
        end
    end
    isempty(recorded) && return nothing
    local_path = _p1_local_artifact_path(recorded)
    if isempty(local_path) && parent_path !== nothing
        throw(ArgumentError(
            "could not locate the requested parent history '$recorded' locally",
        ))
    end
    return isempty(local_path) ? nothing : local_path
end

function _p1_stitch_parent_measured_history(
    state_file::AbstractString,
    continuation_history;
    include_seed::Bool,
    parent_path=nothing,
)
    local_parent = _p1_parent_history_path(state_file; parent_path)
    local_parent === nothing && return nothing
    parent_history = _p1_complete_history(local_parent, :measured; include_seed)
    parent_history === nothing && return nothing

    parent_final = _p1_history_endpoint(parent_history, length(parent_history.iterations))
    continuation_seed = phase1_seed_fields(state_file)
    _p1_assert_same_fields(
        parent_final,
        continuation_seed,
        "parent terminal measured field and continuation seed",
    )

    continuation_first = include_seed ? 2 : 1
    continuation_count = length(continuation_history.iterations)
    continuation_first <= continuation_count || throw(ArgumentError(
        "continuation history contains no MF updates after its initial seed",
    ))
    function append_component(component::Symbol)
        parent_values = getproperty(parent_history, component)
        continuation_values = getproperty(continuation_history, component)
        trailing = selectdim(
            continuation_values,
            ndims(continuation_values),
            continuation_first:continuation_count,
        )
        return cat(parent_values, trailing; dims=ndims(parent_values))
    end

    alpha = append_component(:alpha)
    beta = append_component(:beta)
    mu_cdw = append_component(:mu_cdw)
    parent_samples = length(parent_history.iterations)
    iterations = collect(1:size(alpha, ndims(alpha)))
    parent_updates = parent_samples - (include_seed ? 1 : 0)
    continuation_updates = continuation_count - (include_seed ? 1 : 0)
    return (
        history=(; iterations, alpha, beta, mu_cdw),
        parent_path=local_parent,
        parent_samples,
        parent_updates,
        continuation_updates,
    )
end

function _p1_relabel_complete_histories!(fig, iterations, source::Symbol)
    axes = _p1_relabel_profile_axes!(fig; profile_and_history=true)
    history_axes = [axes[2], axes[4], axes[6], axes[8], axes[10]]
    for ax in history_axes
        ax.set_xlabel("")
        for line in ax.lines
            length(line.get_xdata()) == length(iterations) && line.set_xdata(iterations)
        end
        ax.relim()
        ax.autoscale_view()
    end
    history_axes[end].set_xlabel("MF iteration ($(String(source)) fields)")
    return fig
end

function _p1_mark_continuation_boundary!(fig, parent_samples::Integer)
    axes = collect(fig.axes)
    history_axes = [axes[2], axes[4], axes[6], axes[8], axes[10]]
    boundary = parent_samples + 0.5
    for ax in history_axes
        ax.axvline(boundary; color="0.45", linestyle="--", linewidth=1.1, alpha=0.8)
    end
    history_axes[1].text(
        boundary,
        0.98,
        "continuation",
        transform=history_axes[1].get_xaxis_transform(),
        rotation=90,
        va="top",
        ha="right",
        color="0.35",
        fontsize=8,
    )
    return fig
end

"""
Plot Phase 1 profiles and middle-rung values. Complete measured histories are
stitched to a locally available parent history by default. Set
`stitch_parent_history=false` to show only the requested state. For v2, the
left column is controlled by a saved-snapshot slider and the right column
connects only the explicitly labelled retained snapshots.
"""
function plot_phase1_mf_profiles_and_middle_histories(
    state_file::AbstractString;
    snapshot=:latest,
    history_source::Symbol=:measured,
    include_seed::Bool=true,
    stitch_parent_history::Bool=true,
    parent_path=nothing,
    savepath=nothing,
    dpi=nothing,
    figure_title=nothing,
    kwargs...,
)
    state_file = abspath(state_file)
    complete_history = _p1_complete_history(state_file, history_source; include_seed)
    if complete_history !== nothing
        stitched = if stitch_parent_history && history_source == :measured
            _p1_stitch_parent_measured_history(
                state_file,
                complete_history;
                include_seed,
                parent_path,
            )
        else
            nothing
        end
        plotted_history = stitched === nothing ? complete_history : stitched.history
        plot_iterations = stitched === nothing ?
            _p1_plot_iterations(plotted_history.iterations, include_seed) :
            plotted_history.iterations
        selected = _p1_history_index(snapshot, plot_iterations)
        beta_list = _p1_legacy_beta_history(plotted_history.beta, plotted_history.mu_cdw)
        update_count = length(plotted_history.iterations) - (include_seed ? 1 : 0)
        detail = if stitched !== nothing
            seed_detail = include_seed ? "initial seed at MF iteration 1; " : ""
            continuation_start = stitched.parent_samples + 1
            "stitched $(history_source) MF history: $(seed_detail)" *
            "$(stitched.parent_updates) parent updates + $(stitched.continuation_updates) continuation updates; " *
            "continuation begins at MF iteration $continuation_start"
        elseif include_seed
            "complete $(history_source) MF history: initial seed at MF iteration 1 plus $update_count updates"
        else
            "complete $(history_source) MF history: $update_count updates"
        end
        title = something(figure_title, _p1_figure_title(state_file, detail))
        fig = plot_mf_profiles_and_middle_histories(
            plotted_history.alpha,
            beta_list,
            plotted_history.alpha,
            beta_list;
            iteration=selected,
            figure_title=title,
            savepath=nothing,
            kwargs...,
        )
        _p1_relabel_complete_histories!(fig, plot_iterations, history_source)
        stitched !== nothing && _p1_mark_continuation_boundary!(fig, stitched.parent_samples)
        return _p1_save_after_relabel(fig, savepath; dpi)
    end

    history_source == :measured || throw(ArgumentError(
        "this pre-schema-v5 state has no complete $history_source field history; only measured saved-snapshot fallback is available",
    ))
    snapshots = _p1_saved_snapshots(state_file; include_seed, parent_path)
    isempty(snapshots) && throw(ArgumentError("no saved field snapshots found in $state_file"))
    alpha_list = cat((item.fields.alpha for item in snapshots)...; dims=5)
    beta_list = cat((_p1_legacy_beta(item.fields) for item in snapshots)...; dims=6)
    selected = _p1_snapshot_index(snapshot, snapshots)
    labels = join(
        (item.iteration == 0 ? item.label : "$(item.label)=i$(item.iteration)" for item in snapshots),
        ", ",
    )
    detail = "saved MF snapshots: $labels; v2 did not store the full field history"
    title = something(figure_title, _p1_figure_title(state_file, detail))

    fig = plot_mf_profiles_and_middle_histories(
        alpha_list,
        beta_list,
        alpha_list,
        beta_list;
        iteration=selected,
        figure_title=title,
        savepath=nothing,
        kwargs...,
    )
    _p1_relabel_snapshot_histories!(fig, snapshots)
    return _p1_save_after_relabel(fig, savepath; dpi)
end

"""Render profile/history and seed figures for every state in one campaign."""
function save_phase1_campaign_mf_observables(
    run_directory::AbstractString=DEFAULT_PHASE1_RUN_DIRECTORY;
    output_directory::AbstractString=joinpath(abspath(run_directory), "plots", "mf_profiles"),
    close_figures::Bool=true,
    dpi::Real=140,
)
    states = phase1_campaign_states(run_directory)
    profile_directory = joinpath(output_directory, "profiles_and_saved_histories")
    seed_directory = joinpath(output_directory, "seeds")
    mkpath(profile_directory)
    mkpath(seed_directory)
    outputs = NamedTuple[]

    for label in sort!(collect(keys(states)))
        state_file = states[label]
        profile_path = joinpath(profile_directory, "$label.png")
        seed_path = joinpath(seed_directory, "$label.png")
        profile_fig = plot_phase1_mf_profiles_and_middle_histories(
            state_file;
            savepath=profile_path,
            dpi,
        )
        seed_fig = plot_phase1_seed_profiles(state_file; savepath=seed_path, dpi)
        push!(outputs, (; label, state_file, profile_path, seed_path))
        if close_figures
            PyPlot.close(profile_fig)
            PyPlot.close(seed_fig)
        end
        println("rendered $label")
    end
    return outputs
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_directory = isempty(ARGS) ? DEFAULT_PHASE1_RUN_DIRECTORY : abspath(ARGS[1])
    output_directory = length(ARGS) >= 2 ? abspath(ARGS[2]) :
        joinpath(run_directory, "plots", "mf_profiles")
    outputs = save_phase1_campaign_mf_observables(
        run_directory;
        output_directory,
    )
    println("output_directory=$(abspath(output_directory))")
    println("figures=$(2 * length(outputs))")
end
