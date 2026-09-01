function _density_corrected_energy(file, canonical::Real, period::Integer)
    if haskey(file, "solution_target_density_corrected_variational_energy")
        stored = Float64(read(file, "solution_target_density_corrected_variational_energy"))
        isfinite(stored) && return stored, "stored_target_density_corrected"
    end
    haskey(file, "model/L") && haskey(file, "model/density") || return Float64(canonical),
        "legacy_uncorrected"
    target_particles = 2 * Int(read(file, "model/L")) * Float64(read(file, "model/density"))
    if period > 1 && haskey(file, "cycle_members")
        corrected = Float64[]
        members = file["cycle_members"]
        for name in sort!(String.(collect(keys(members))))
            group = members[name]
            required = (
                "energy/canonical_variational_energy",
                "chemical_potential",
                "correlations/density_down",
                "correlations/density_up",
            )
            all(path -> haskey(group, path), required) || return Float64(canonical),
                "legacy_uncorrected"
            energy = Float64(read(group, "energy/canonical_variational_energy"))
            mu = Float64(read(group, "chemical_potential"))
            particles = sum(read(group, "correlations/density_down")) +
                sum(read(group, "correlations/density_up"))
            push!(corrected, energy + mu * (target_particles - particles))
        end
        !isempty(corrected) && return mean(corrected), "reconstructed_target_density_corrected"
    elseif haskey(file, "energy/target_density_corrected_variational_energy")
        stored = Float64(read(file, "energy/target_density_corrected_variational_energy"))
        isfinite(stored) && return stored, "stored_target_density_corrected"
    elseif all(path -> haskey(file, path), (
        "chemical_potential",
        "correlations/density_down",
        "correlations/density_up",
    ))
        mu = Float64(read(file, "chemical_potential"))
        particles = sum(read(file, "correlations/density_down")) +
            sum(read(file, "correlations/density_up"))
        return Float64(canonical) + mu * (target_particles - particles),
            "reconstructed_target_density_corrected"
    end
    return Float64(canonical), "legacy_uncorrected"
end

function _result_row(path::AbstractString)
    return h5open(path, "r") do file
        completed = haskey(file, "completed") && Bool(read(file, "completed"))
        accepted = haskey(file, "accepted") && Bool(read(file, "accepted"))
        status = haskey(file, "status") ? String(read(file, "status")) : "unknown"
        solution_kind = haskey(file, "solution_kind") ? String(read(file, "solution_kind")) : "unknown"
        period = haskey(file, "fundamental_period") ? Int(read(file, "fundamental_period")) : 0
        orbit_validated = haskey(file, "orbit_validated") && Bool(read(file, "orbit_validated"))
        geometry = haskey(file, "model/transverse_geometry") ? String(read(file, "model/transverse_geometry")) : "unknown"
        fingerprint = haskey(file, "provenance/model_fingerprint") ? String(read(file, "provenance/model_fingerprint")) : ""
        numerical_fingerprint = haskey(file, "provenance/numerical_fingerprint") ? String(read(file, "provenance/numerical_fingerprint")) : ""
        implementation_sha256 = haskey(file, "provenance/implementation_sha256") ? String(read(file, "provenance/implementation_sha256")) : ""
        ep_source_sha256 = haskey(file, "provenance/ep_source_sha256") ? String(read(file, "provenance/ep_source_sha256")) : ""
        canonical_energy = if haskey(file, "solution_canonical_variational_energy")
            Float64(read(file, "solution_canonical_variational_energy"))
        elseif haskey(file, "energy/canonical_variational_energy")
            Float64(read(file, "energy/canonical_variational_energy"))
        else
            NaN
        end
        energy, energy_kind = _density_corrected_energy(file, canonical_energy, period)
        branch = haskey(file, "provenance/branch_label") ? String(read(file, "provenance/branch_label")) : "unknown"
        return (; path=abspath(path), completed, accepted, status, solution_kind, period, orbit_validated,
            geometry, fingerprint,
            numerical_fingerprint, implementation_sha256, ep_source_sha256,
            energy, canonical_energy, energy_kind, branch)
    end
end

function select_completed_runs(directory::AbstractString; include_incomplete::Bool=false, geometry=nothing)
    isdir(directory) || throw(ArgumentError("result directory not found: $directory"))
    requested_geometry = geometry === nothing ? nothing : String(normalize_geometry(geometry))
    rows = NamedTuple[]
    paths = String[]
    for (root, _, names) in walkdir(directory)
        for name in names
            name == "state.h5" && push!(paths, joinpath(root, name))
        end
    end
    for path in sort(paths)
        row = try
            _result_row(path)
        catch
            continue
        end
        requested_geometry !== nothing && row.geometry != requested_geometry && continue
        valid_solution = row.status == "fixed_point" ||
            (row.status == "periodic_solution" && row.orbit_validated)
        accepted_solution = row.completed && row.accepted && valid_solution
        !include_incomplete && !accepted_solution && continue
        push!(rows, merge(row, (; plot_style=accepted_solution ? "solid" : "hatched")))
    end
    return rows
end

function compare_variational_branches(paths::AbstractVector{<:AbstractString})
    rows = [_result_row(path) for path in paths]
    isempty(rows) && return rows
    fingerprints = unique(row.fingerprint for row in rows)
    length(fingerprints) == 1 || throw(ArgumentError(
        "variational branches must have the same model fingerprint; cross-geometry ranking is not meaningful",
    ))
    length(unique(row.numerical_fingerprint for row in rows)) == 1 || throw(ArgumentError(
        "variational branches must use the same numerical settings fingerprint",
    ))
    length(unique(row.implementation_sha256 for row in rows)) == 1 || throw(ArgumentError(
        "variational branches must use the same implementation fingerprint",
    ))
    length(unique(row.ep_source_sha256 for row in rows)) == 1 || throw(ArgumentError(
        "variational branches must use the same E_p registry content",
    ))
    all(row -> row.completed && row.accepted && row.status in ("fixed_point", "periodic_solution"), rows) ||
        throw(ArgumentError("only accepted fixed points or validated periodic solutions may enter the variational ranking"))
    all(row -> row.period == 1 || row.orbit_validated, rows) || throw(ArgumentError(
        "periodic branches require an unmixed validated orbit",
    ))
    all(row -> row.energy_kind != "legacy_uncorrected", rows) || throw(ArgumentError(
        "variational ranking requires a stored or reconstructable target-density-corrected energy",
    ))
    return sort(rows; by=row -> row.energy)
end
