function _write_dict(group, values::AbstractDict)
    for (raw_key, value) in values
        key = String(raw_key)
        if value isa AbstractDict
            child = create_group(group, key)
            _write_dict(child, value)
        elseif value === nothing || value === missing
            group[key] = ""
        elseif value isa Symbol
            group[key] = String(value)
        else
            group[key] = value
        end
    end
end

function _write_fields(group, fields::FieldState)
    group["alpha"] = fields.alpha
    group["beta"] = fields.beta
    group["mu_cdw"] = fields.mu_cdw
end

function _read_fields(group)
    return FieldState(read(group, "alpha"), read(group, "beta"), read(group, "mu_cdw"))
end

function _stack_field_component(
    records::AbstractVector{IterationRecord},
    source::Symbol,
    component::Symbol,
)
    isempty(records) && throw(ArgumentError("cannot stack an empty MF field history"))
    sample = getfield(getfield(first(records), source), component)
    history = Array{Float64}(undef, (size(sample)..., length(records)))
    final_dimension = ndims(history)
    for (index, record) in enumerate(records)
        values = getfield(getfield(record, source), component)
        size(values) == size(sample) || throw(DimensionMismatch(
            "$source.$component changed shape within one MF history",
        ))
        selectdim(history, final_dimension, index) .= values
    end
    return history
end

function _write_field_history(group, records::AbstractVector{IterationRecord})
    isempty(records) && throw(ArgumentError("cannot write an empty MF field history"))
    group["seed_iteration"] = 0
    _write_fields(create_group(group, "seed"), first(records).applied)
    for source in (:applied, :measured)
        source_group = create_group(group, String(source))
        for component in (:alpha, :beta, :mu_cdw)
            source_group[String(component)] = _stack_field_component(records, source, component)
        end
    end
end

function _write_correlations(group, correlations::CorrelationState)
    group["pair"] = correlations.pair
    group["exchange_down"] = correlations.exchange_down
    group["exchange_up"] = correlations.exchange_up
    group["density_down"] = correlations.density_down
    group["density_up"] = correlations.density_up
end

function _write_energy(group, energy::EnergyBreakdown)
    for field in fieldnames(EnergyBreakdown)
        group[String(field)] = getfield(energy, field)
    end
end

function _write_dmrg_history(group, records::AbstractVector{IterationRecord})
    for record in records
        record_group = create_group(group, lpad(string(record.iteration), 4, '0'))
        record_group["sweep_energy"] = record.dmrg_sweep_energies
        record_group["sweep_max_discarded_weight"] = record.dmrg_sweep_max_discarded_weights
        record_group["sweep_maxlinkdim"] = record.dmrg_sweep_maxlinkdims
        record_group["max_discarded_weight"] = record.dmrg_max_discarded_weight
        record_group["maximum_link_dimension"] = record.dmrg_maxlinkdim
    end
    return group
end

function write_checkpoint(
    path::AbstractString;
    settings::ProjectSettings,
    psi::MPS,
    records::AbstractVector{IterationRecord},
    diagnostic::ConvergenceDiagnostic,
    restart_fields::Union{Nothing,FieldState}=nothing,
    chemical_potential::Union{Nothing,Real}=nothing,
    provenance=collect_provenance(settings),
    immutable::Bool=false,
    phase_psis=Dict{Int,MPS}(),
)
    immutable && ispath(path) && throw(ArgumentError("refusing to overwrite immutable artifact: $path"))
    mkpath(dirname(path))
    temporary = tempname(dirname(path))
    storage_psi = move_to_cpu(psi)
    h5open(temporary, "w") do file
        file["schema_version"] = 7
        file["artifact_kind"] = "ladder_mps_mft_state"
        file["process_completed"] = diagnostic.status != :iterating
        file["accepted"] = diagnostic.accepted
        file["completed"] = diagnostic.accepted
        file["status"] = String(diagnostic.status)
        file["convergence_reason"] = diagnostic.reason
        file["solution_kind"] = String(diagnostic.solution_kind)
        file["fundamental_period"] = diagnostic.fundamental_period
        file["orbit_validated"] = diagnostic.orbit_validated
        file["unmixed_cycle_probe"] = diagnostic.unmixed_probe
        file["solution_canonical_variational_energy"] = diagnostic.solution_canonical_variational_energy
        file["solution_target_density_corrected_variational_energy"] =
            diagnostic.solution_target_density_corrected_variational_energy
        file["orbit_energy_spread"] = diagnostic.orbit_energy_spread
        file["orbit_target_density_corrected_energy_spread"] =
            diagnostic.orbit_target_density_corrected_energy_spread
        file["orbit_density_contrast"] = diagnostic.orbit_density_contrast
        file["fixed_point_abs_residual"] = diagnostic.fixed_point_abs_residual
        file["fixed_point_rel_residual"] = diagnostic.fixed_point_rel_residual
        file["fixed_point_residual_cosine"] = diagnostic.fixed_point_residual_cosine
        file["fixed_point_contraction_estimate"] = diagnostic.fixed_point_contraction_estimate
        file["fixed_point_extrapolation_factor"] = diagnostic.fixed_point_extrapolation_factor
        file["fixed_point_extrapolated_abs_residual"] =
            diagnostic.fixed_point_extrapolated_abs_residual
        file["fixed_point_extrapolated_rel_residual"] =
            diagnostic.fixed_point_extrapolated_rel_residual
        file["cycle_abs_residual"] = diagnostic.cycle_abs_residual
        file["cycle_rel_residual"] = diagnostic.cycle_rel_residual
        file["cycle_oscillation_cosine"] = diagnostic.cycle_oscillation_cosine
        file["cycle_two_step_ratio"] = diagnostic.cycle_two_step_ratio
        file["density_error"] = diagnostic.density_error
        file["variational_energy_change"] = diagnostic.variational_energy_change
        file["hamiltonian_identity_error_per_site"] = diagnostic.hamiltonian_identity_error_per_site
        file["effective_eigenvalue_error_per_site"] = diagnostic.effective_eigenvalue_error_per_site
        file["best_iteration"] = diagnostic.best_iteration
        file["psi"] = storage_psi
        model_group = create_group(file, "model")
        _write_dict(model_group, Dict(
            "L" => settings.model.L, "t" => settings.model.t, "U" => settings.model.U,
            "V" => settings.model.V, "t0" => settings.model.t0, "tp" => settings.model.tp,
            "density" => settings.model.density, "mu_initial" => settings.model.mu_initial,
            "r_range" => settings.model.r_range,
            "transverse_geometry" => String(settings.model.geometry), "E_p" => settings.model.ep,
            "E_p_signed" => settings.model.ep_signed, "E_p_source" => settings.model.ep_source,
            "E_p_mode" => String(settings.model.ep_mode),
            "E_p_t0_lower" => settings.model.ep_t0_lower,
            "E_p_t0_upper" => settings.model.ep_t0_upper,
            "E_p_lower_signed" => settings.model.ep_lower_signed,
            "E_p_upper_signed" => settings.model.ep_upper_signed,
            "E_p_interpolation_weight" => settings.model.ep_interpolation_weight,
            "E_p_lower_chi" => settings.model.ep_lower_chi,
            "E_p_upper_chi" => settings.model.ep_upper_chi,
            "effective_mf_coupling_tp2_over_ep" => settings.model.tp^2 / settings.model.ep,
            "runtime_backend" => String(settings.runtime.backend),
            "tensor_scalar_type" => String(settings.runtime.tensor_scalar_type),
            "conserve_sz" => settings.runtime.conserve_sz,
            "conserve_nfparity" => settings.runtime.conserve_nfparity,
        ))
        if !isempty(records)
            last_record = last(records)
            fields_group = create_group(file, "fields")
            initial_group = create_group(fields_group, "initial")
            applied_group = create_group(fields_group, "applied")
            measured_group = create_group(fields_group, "measured")
            _write_fields(initial_group, first(records).applied)
            _write_fields(applied_group, last_record.applied)
            _write_fields(measured_group, last_record.measured)
            restart_group = create_group(fields_group, "restart")
            _write_fields(restart_group, something(restart_fields, last_record.measured))
            file["chemical_potential"] = Float64(something(chemical_potential, last_record.chemical_potential))
            correlations_group = create_group(file, "correlations")
            _write_correlations(correlations_group, last_record.correlations)
            energy_group = create_group(file, "energy")
            _write_energy(energy_group, last_record.variational)
            history_group = create_group(file, "history")
            history_group["iteration"] = [record.iteration for record in records]
            history_group["update_mode"] = String.(getfield.(records, :update_mode))
            history_group["density"] = [record.density for record in records]
            history_group["chemical_potential"] = [record.chemical_potential for record in records]
            history_group["mu_search_status"] = String.(getfield.(records, :mu_search_status))
            history_group["mu_evaluations"] = [record.mu_evaluations for record in records]
            history_group["mu_density_converged"] = [record.mu_density_converged for record in records]
            history_group["mu_density_slope"] = [record.mu_density_slope for record in records]
            history_group["effective_energy"] = [record.effective_energy for record in records]
            history_group["variational_energy"] = [record.variational.canonical_variational_energy for record in records]
            history_group["target_density_corrected_variational_energy"] = [
                record.variational.target_density_corrected_variational_energy for record in records
            ]
            history_group["field_abs_residual"] = [record.field_abs_residual for record in records]
            history_group["field_rel_residual"] = [record.field_rel_residual for record in records]
            history_group["wall_seconds"] = [record.wall_seconds for record in records]
            history_group["dmrg_max_discarded_weight"] = [
                record.dmrg_max_discarded_weight for record in records
            ]
            history_group["dmrg_maxlinkdim"] = [record.dmrg_maxlinkdim for record in records]
            _write_dmrg_history(create_group(history_group, "dmrg"), records)
            _write_field_history(create_group(history_group, "fields"), records)
            period = diagnostic.fundamental_period
            if period > 1 && length(records) >= period
                cycle_group = create_group(file, "cycle_members")
                for (member, record) in enumerate(records[(end - period + 1):end])
                    member_group = create_group(cycle_group, lpad(string(member), 3, '0'))
                    _write_fields(create_group(member_group, "applied"), record.applied)
                    _write_fields(create_group(member_group, "measured"), record.measured)
                    _write_correlations(create_group(member_group, "correlations"), record.correlations)
                    _write_energy(create_group(member_group, "energy"), record.variational)
                    member_group["iteration"] = record.iteration
                    member_group["update_mode"] = String(record.update_mode)
                    member_group["density"] = record.density
                    member_group["chemical_potential"] = record.chemical_potential
                    member_group["variational_energy"] = record.variational.canonical_variational_energy
                    member_group["target_density_corrected_variational_energy"] =
                        record.variational.target_density_corrected_variational_energy
                    member_group["dmrg_max_discarded_weight"] = record.dmrg_max_discarded_weight
                    member_group["dmrg_maxlinkdim"] = record.dmrg_maxlinkdim
                    if haskey(phase_psis, record.iteration)
                        member_group["psi"] = move_to_cpu(phase_psis[record.iteration])
                    end
                end
            end
        end
        _write_dict(create_group(file, "provenance"), provenance)
    end
    mv(temporary, path; force=!immutable)
    return path
end

const _STATELESS_MPS_BASENAME = r"^psi(?:_N_[0-9]+)?$"

_omit_from_stateless_copy(name::AbstractString) = occursin(_STATELESS_MPS_BASENAME, name)

function _copy_hdf5_attributes!(destination, source)
    source_attributes = attributes(source)
    destination_attributes = attributes(destination)
    for name in keys(source_attributes)
        destination_attributes[name] = read(source_attributes, name)
    end
    return destination
end

function _copy_hdf5_without_mps!(destination, source, prefix::AbstractString, omitted::Vector{String})
    _copy_hdf5_attributes!(destination, source)
    for name in sort!(String.(collect(keys(source))))
        path = isempty(prefix) ? name : "$prefix/$name"
        if _omit_from_stateless_copy(name)
            push!(omitted, path)
            continue
        end
        object = source[name]
        try
            if object isa HDF5.Group
                child = create_group(destination, name)
                try
                    _copy_hdf5_without_mps!(child, object, path, omitted)
                finally
                    close(child)
                end
            else
                HDF5.copy_object(source, name, destination, name)
            end
        finally
            close(object)
        end
    end
    return destination
end

"""
    write_stateless_copy(source, destination; force=false)

Create an analysis-ready HDF5 copy while recursively omitting every MPS-bearing
`psi` or `psi_N_<sector>` object. The copy preserves all other groups,
datasets, attributes, mean-field histories, observables, and provenance. It
also records the absolute full-artifact location and SHA-256 under
`analysis_storage`; it is intentionally not a restart checkpoint.
"""
function write_stateless_copy(
    source::AbstractString,
    destination::AbstractString;
    force::Bool=false,
)
    source_path = abspath(source)
    destination_path = abspath(destination)
    isfile(source_path) || throw(ArgumentError("source HDF5 artifact not found: $source_path"))
    source_path == destination_path && throw(ArgumentError("source and stateless destination must differ"))
    !force && ispath(destination_path) && throw(ArgumentError(
        "refusing to overwrite stateless artifact: $destination_path",
    ))
    mkpath(dirname(destination_path))
    source_before = stat(source_path)
    full_sha256 = sha256_file(source_path)
    temporary = tempname(dirname(destination_path))
    omitted = String[]
    try
        h5open(source_path, "r") do input
            haskey(input, "analysis_storage/is_stateless_copy") &&
                Bool(read(input, "analysis_storage/is_stateless_copy")) &&
                throw(ArgumentError("source is already a stateless analysis copy: $source_path"))
            h5open(temporary, "w") do output
                _copy_hdf5_without_mps!(output, input, "", omitted)
                storage = create_group(output, "analysis_storage")
                storage["schema_version"] = 1
                storage["is_stateless_copy"] = true
                storage["restartable"] = false
                storage["full_artifact_path"] = source_path
                storage["full_artifact_sha256"] = full_sha256
                storage["full_artifact_size_bytes"] = Int64(source_before.size)
                storage["generated_utc"] = string(now(UTC))
                storage["omitted_paths"] = join(omitted, "\n")
                close(storage)
            end
        end
        source_after = stat(source_path)
        (source_after.size == source_before.size && source_after.mtime == source_before.mtime) ||
            throw(ArgumentError("source changed while its stateless copy was being made: $source_path"))
        mv(temporary, destination_path; force=force)
    catch
        ispath(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return (
        source_path,
        destination_path,
        full_sha256,
        compact_sha256=sha256_file(destination_path),
        source_bytes=source_before.size,
        compact_bytes=stat(destination_path).size,
        omitted_paths=copy(omitted),
    )
end

const _STATELESS_TEXT_EXTENSIONS = Set((
    ".csv", ".json", ".md", ".sha256", ".toml", ".tsv", ".txt",
))

"""
    mirror_stateless_tree(source_root, destination_root; force=true)

Mirror one result tree for analysis. HDF5 files are copied with all MPS objects
removed; small text metadata are copied verbatim. A `stateless_manifest.tsv`
binds every lightweight artifact to the path and SHA-256 of its full source.
"""
function mirror_stateless_tree(
    source_root::AbstractString,
    destination_root::AbstractString;
    force::Bool=true,
)
    source = abspath(source_root)
    destination = abspath(destination_root)
    isdir(source) || throw(ArgumentError("full result directory not found: $source"))
    source == destination && throw(ArgumentError("full and stateless result directories must differ"))
    destination_relative_to_source = relpath(destination, source)
    destination_relative_to_source != ".." &&
        !startswith(destination_relative_to_source, joinpath("..", "")) && throw(ArgumentError(
        "stateless destination must not be nested below the full result directory",
    ))
    mkpath(destination)
    records = NamedTuple[]
    for (directory, subdirectories, names) in walkdir(source)
        filter!(name -> name != ".snapshots", subdirectories)
        for name in sort!(names)
            source_path = joinpath(directory, name)
            relative_path = relpath(source_path, source)
            relative_path == "stateless_manifest.tsv" && continue
            destination_path = joinpath(destination, relative_path)
            extension = lowercase(splitext(name)[2])
            if extension in (".h5", ".hdf5")
                result = write_stateless_copy(source_path, destination_path; force)
                push!(records, (
                    relative_path,
                    kind="stateless_hdf5",
                    source_path=result.source_path,
                    source_sha256=result.full_sha256,
                    source_bytes=result.source_bytes,
                    compact_path=result.destination_path,
                    compact_sha256=result.compact_sha256,
                    compact_bytes=result.compact_bytes,
                    omitted_paths=join(result.omitted_paths, ","),
                ))
            elseif extension in _STATELESS_TEXT_EXTENSIONS
                mkpath(dirname(destination_path))
                cp(source_path, destination_path; force)
                source_hash = sha256_file(source_path)
                push!(records, (
                    relative_path,
                    kind="verbatim_metadata",
                    source_path=abspath(source_path),
                    source_sha256=source_hash,
                    source_bytes=stat(source_path).size,
                    compact_path=abspath(destination_path),
                    compact_sha256=sha256_file(destination_path),
                    compact_bytes=stat(destination_path).size,
                    omitted_paths="",
                ))
            end
        end
    end
    manifest_path = joinpath(destination, "stateless_manifest.tsv")
    temporary = tempname(destination)
    open(temporary, "w") do io
        println(io, join((
            "relative_path", "kind", "full_path", "full_sha256", "full_bytes",
            "compact_path", "compact_sha256", "compact_bytes", "omitted_paths",
        ), '\t'))
        for record in sort!(records; by=record -> record.relative_path)
            println(io, join((
                record.relative_path,
                record.kind,
                record.source_path,
                record.source_sha256,
                record.source_bytes,
                record.compact_path,
                record.compact_sha256,
                record.compact_bytes,
                record.omitted_paths,
            ), '\t'))
        end
    end
    mv(temporary, manifest_path; force=true)
    return (; source, destination, records, manifest_path)
end

"""
Read the complete per-iteration mean-field history saved by schema-v5-and-newer states.

`source=:applied` returns the fields used to build each effective Hamiltonian;
`source=:measured` returns the corresponding raw DMRG mean-field map outputs.
With `include_seed=true`, prepend the exact initial field at iteration zero.
Schema-v7 states store that seed under `history/fields/seed`; schema-v5/v6
states fall back to their equivalent `fields/initial` record. The final array
dimension is the MF-history index and matches `iterations`.
"""
function read_field_history(
    path::AbstractString;
    source::Symbol=:measured,
    include_seed::Bool=false,
)
    source in (:applied, :measured) || throw(ArgumentError(
        "field-history source must be :applied or :measured",
    ))
    isfile(path) || throw(ArgumentError("state not found: $path"))
    return h5open(path, "r") do file
        base = "history/fields/$(String(source))"
        haskey(file, base) || throw(ArgumentError(
            "state does not contain a complete $source MF history (requires schema version 5 or newer): $path",
        ))
        iterations = Int.(read(file, "history/iteration"))
        alpha = Float64.(read(file, "$base/alpha"))
        beta = Float64.(read(file, "$base/beta"))
        mu_cdw = Float64.(read(file, "$base/mu_cdw"))
        count = length(iterations)
        all(array -> size(array, ndims(array)) == count, (alpha, beta, mu_cdw)) ||
            throw(DimensionMismatch("stored $source MF arrays do not match history/iteration"))
        if include_seed
            seed_base = haskey(file, "history/fields/seed") ?
                "history/fields/seed" : "fields/initial"
            haskey(file, seed_base) || throw(ArgumentError(
                "state has no saved time-zero MF seed: $path",
            ))
            seed_iteration = haskey(file, "history/fields/seed_iteration") ?
                Int(read(file, "history/fields/seed_iteration")) : 0
            seed_iteration == 0 || throw(ArgumentError(
                "stored MF seed iteration must be zero; got $seed_iteration in $path",
            ))
            seed = _read_fields(file[seed_base])
            alpha = cat(seed.alpha, alpha; dims=ndims(alpha))
            beta = cat(seed.beta, beta; dims=ndims(beta))
            mu_cdw = cat(seed.mu_cdw, mu_cdw; dims=ndims(mu_cdw))
            iterations = [seed_iteration; iterations]
        end
        return (; iterations, alpha, beta, mu_cdw)
    end
end

"""
Read a field-only warm start from either a refactored state or a legacy ladder
HDF5 file. No MPS is loaded. Legacy files without `mu_cdw` receive the same
zero Hartree-field fallback used by the legacy GPU driver.
"""
function read_inherited_fields(path::AbstractString)
    isfile(path) || throw(ArgumentError("inherited field state not found: $path"))
    return h5open(path, "r") do file
        if haskey(file, "fields/restart")
            haskey(file, "chemical_potential") || throw(ArgumentError(
                "refactored inherit source has no chemical_potential: $path",
            ))
            return (
                fields=_read_fields(file["fields/restart"]),
                chemical_potential=Float64(read(file, "chemical_potential")),
                format=:refactored,
                source_geometry=haskey(file, "model/transverse_geometry") ?
                    String(read(file, "model/transverse_geometry")) : nothing,
            )
        end
        haskey(file, "alpha") && haskey(file, "beta") || throw(ArgumentError(
            "inherit_from must be a refactored state with fields/restart or a legacy state with top-level alpha and beta: $path",
        ))
        alpha = Float64.(read(file, "alpha"))
        beta = Float64.(read(file, "beta"))
        ndims(alpha) == 4 || throw(DimensionMismatch("legacy inherited alpha must be rank 4"))
        L = size(alpha, 1)
        mu_cdw = haskey(file, "mu_cdw") ? Float64.(read(file, "mu_cdw")) : zeros(Float64, 2, 2 * L)
        haskey(file, "mu") || throw(ArgumentError("legacy inherit source has no top-level mu: $path"))
        return (
            fields=FieldState(alpha, beta, mu_cdw),
            chemical_potential=Float64(read(file, "mu")),
            format=:legacy,
            source_geometry=haskey(file, "transverse_geometry") ?
                String(read(file, "transverse_geometry")) : nothing,
        )
    end
end

function read_checkpoint(
    path::AbstractString;
    orbit_phase::Union{Nothing,Integer}=nothing,
)
    isfile(path) || throw(ArgumentError("checkpoint not found: $path"))
    return h5open(path, "r") do file
        if !haskey(file, "psi") && haskey(file, "analysis_storage/is_stateless_copy")
            full_path = String(read(file, "analysis_storage/full_artifact_path"))
            throw(ArgumentError(
                "stateless analysis copy is not restartable; use its full artifact: $full_path",
            ))
        end
        if orbit_phase !== nothing
            orbit_phase >= 1 || throw(ArgumentError("orbit phase must be positive"))
            phase_name = lpad(string(orbit_phase), 3, '0')
            phase_path = "cycle_members/$phase_name"
            haskey(file, phase_path) || throw(ArgumentError(
                "checkpoint has no stored orbit phase $orbit_phase: $path",
            ))
            phase = file[phase_path]
            for required in ("psi", "applied", "measured", "chemical_potential")
                haskey(phase, required) || throw(ArgumentError(
                    "stored orbit phase $orbit_phase has no $required: $path",
                ))
            end
            measured = _read_fields(phase["measured"])
            return (
                psi=read(phase, "psi", MPS),
                applied=_read_fields(phase["applied"]),
                measured,
                restart=measured,
                chemical_potential=Float64(read(phase, "chemical_potential")),
                accepted=Bool(read(file, "accepted")),
                status=Symbol(read(file, "status")),
                solution_kind=haskey(file, "solution_kind") ? Symbol(read(file, "solution_kind")) : :unknown,
                fundamental_period=haskey(file, "fundamental_period") ? Int(read(file, "fundamental_period")) : 0,
                orbit_validated=haskey(file, "orbit_validated") && Bool(read(file, "orbit_validated")),
                model_fingerprint=read(file, "provenance/model_fingerprint"),
            )
        end
        return (
            psi=read(file, "psi", MPS),
            applied=_read_fields(file["fields/applied"]),
            measured=_read_fields(file["fields/measured"]),
            restart=haskey(file, "fields/restart") ? _read_fields(file["fields/restart"]) : _read_fields(file["fields/measured"]),
            chemical_potential=haskey(file, "chemical_potential") ? Float64(read(file, "chemical_potential")) : 0.0,
            accepted=Bool(read(file, "accepted")),
            status=Symbol(read(file, "status")),
            solution_kind=haskey(file, "solution_kind") ? Symbol(read(file, "solution_kind")) : :unknown,
            fundamental_period=haskey(file, "fundamental_period") ? Int(read(file, "fundamental_period")) : 0,
            orbit_validated=haskey(file, "orbit_validated") && Bool(read(file, "orbit_validated")),
            model_fingerprint=read(file, "provenance/model_fingerprint"),
        )
    end
end

function read_orbit_phase_states(path::AbstractString)
    isfile(path) || throw(ArgumentError("state not found: $path"))
    return h5open(path, "r") do file
        if haskey(file, "analysis_storage/is_stateless_copy")
            full_path = String(read(file, "analysis_storage/full_artifact_path"))
            throw(ArgumentError(
                "stateless analysis copy has no orbit MPSs; use its full artifact: $full_path",
            ))
        end
        haskey(file, "cycle_members") || return NamedTuple[]
        states = NamedTuple[]
        for name in sort(collect(keys(file["cycle_members"])))
            group = file["cycle_members/$name"]
            haskey(group, "psi") || throw(ArgumentError("orbit phase $name has no saved MPS"))
            push!(states, (
                phase=parse(Int, name),
                iteration=Int(read(group, "iteration")),
                psi=read(group, "psi", MPS),
            ))
        end
        return states
    end
end

function write_run_summary_markdown(path::AbstractString, settings::ProjectSettings, diagnostic::ConvergenceDiagnostic, records)
    mkpath(dirname(path))
    final = isempty(records) ? nothing : last(records)
    open(path, "w") do io
        println(io, "# MPS+MF run summary")
        println(io)
        println(io, "Generated: `$(now(UTC))` UTC")
        println(io)
        println(io, "## Identity")
        println(io)
        println(io, "- Branch: `$(settings.run.branch_label)`")
        println(io, "- Preparation: `$(settings.run.preparation)`")
        println(io, "- Direction: `$(settings.run.direction)`")
        println(io, "- Seed: `$(settings.run.seed_label)` (`$(settings.run.random_seed)`)" )
        seed_metadata = initial_seed_metadata(settings.model, settings.run)
        println(io, "- Initial seed channel / protocol: `$(settings.run.initial_seed)` / `$(seed_metadata.protocol)`")
        println(io, "- Initial seed fingerprint: `$(initial_seed_fingerprint(settings))`")
        println(io, "- Initial mode number / q over pi / phase over pi: `$(seed_metadata.mode_number)` / `$(seed_metadata.mode_wavevector_pi)` / `$(seed_metadata.mode_phase_pi)`")
        println(io, "- Pairing form factor / leg parity: `$(seed_metadata.pairing_form_factor)` / `$(seed_metadata.resolved_leg_parity)`")
        if settings.run.initial_seed in (:stripe, :stripe_pairing)
            println(io, "- Stripe envelope / spin / charge modes: `$(seed_metadata.stripe_envelope_mode_number)` / `$(seed_metadata.stripe_spin_mode_number)` / `$(seed_metadata.stripe_charge_mode_number)`")
            println(io, "- Stripe spin / charge q over pi: `$(seed_metadata.stripe_spin_wavevector_pi)` / `$(seed_metadata.stripe_charge_wavevector_pi)`")
            println(io, "- Stripe charge:spin / pairing:spin source ratios: `$(seed_metadata.stripe_charge_to_spin_ratio)` / `$(seed_metadata.stripe_pairing_to_spin_ratio)`")
        end
        if settings.run.initial_seed == :legacy_pairing
            println(io, "- Legacy-like pairing field RNG seed: `$(seed_metadata.legacy_pairing_random_seed)`")
            println(io, "- Legacy-like pairing center-of-mass structure: `$(seed_metadata.legacy_pairing_center_of_mass_structure)`")
            println(io, "- Legacy-like beta / mu_cdw initialization: `$(seed_metadata.legacy_pairing_beta_initialization)` / `$(seed_metadata.legacy_pairing_mu_cdw_initialization)`")
        end
        println(io, "- Initial seed normalization: `$(seed_metadata.normalization)`")
        println(io, "- Geometry: `$(settings.model.geometry)`")
        println(io, "- Runtime backend: `$(settings.runtime.backend)`")
        println(io, "- Tensor scalar type: `$(settings.runtime.tensor_scalar_type)`")
        println(io, "- Conserved S_z / fermion parity: `$(settings.runtime.conserve_sz)` / `$(settings.runtime.conserve_nfparity)`")
        println(io, "- Model fingerprint: `$(model_fingerprint(settings.model))`")
        println(io, "- Numerical fingerprint: `$(numerical_fingerprint(settings))`")
        println(io, "- Solver implementation SHA-256: `$(implementation_fingerprint(settings))`")
        println(io, "- Full source/config/launcher/test tree SHA-256: `$(tree_fingerprint())`")
        println(io, "- Configuration SHA-256: `$(isfile(settings.config_path) ? sha256_file(settings.config_path) : "not-file-backed")`")
        println(io, "- Field-only inherit source: `$(something(settings.run.inherit_from, "none"))`")
        println(io, "- Parent checkpoint: `$(something(settings.run.parent_checkpoint, "none"))`")
        println(io, "- Parent orbit phase: `$(something(settings.run.parent_orbit_phase, "none"))`")
        println(io, "- Resume checkpoint: `$(something(settings.run.resume_checkpoint, "none"))`")
        println(io)
        println(io, "## Numerical outcome")
        println(io)
        println(io, "- Status: `$(diagnostic.status)`")
        println(io, "- Accepted physical solution: `$(diagnostic.accepted)`")
        println(io, "- Reason: $(diagnostic.reason)")
        println(io, "- Solution kind: `$(diagnostic.solution_kind)`")
        println(io, "- Fundamental period: `$(diagnostic.fundamental_period)`")
        println(io, "- Orbit validated from unmixed map: `$(diagnostic.orbit_validated && diagnostic.unmixed_probe)`")
        println(io, "- Solution canonical variational energy: `$(diagnostic.solution_canonical_variational_energy)`")
        println(io, "- Solution target-density-corrected energy: `$(diagnostic.solution_target_density_corrected_variational_energy)`")
        println(io, "- Orbit phase-energy spread: `$(diagnostic.orbit_energy_spread)`")
        println(io, "- Orbit target-density-corrected spread: `$(diagnostic.orbit_target_density_corrected_energy_spread)`")
        println(io, "- Orbit density contrast: `$(diagnostic.orbit_density_contrast)`")
        println(io, "- Fixed-point residual (relative): `$(diagnostic.fixed_point_rel_residual)`")
        println(io, "- Fixed-point residual cosine / lambda: `$(diagnostic.fixed_point_residual_cosine)` / `$(diagnostic.fixed_point_contraction_estimate)`")
        println(io, "- Fixed-point extrapolation factor / relative residual: `$(diagnostic.fixed_point_extrapolation_factor)` / `$(diagnostic.fixed_point_extrapolated_rel_residual)`")
        println(io, "- Period-two step cosine / two-step ratio: `$(diagnostic.cycle_oscillation_cosine)` / `$(diagnostic.cycle_two_step_ratio)`")
        println(io, "- Hamiltonian identity error/site: `$(diagnostic.hamiltonian_identity_error_per_site)`")
        println(io, "- Effective eigenvalue error/site: `$(diagnostic.effective_eigenvalue_error_per_site)`")
        if final !== nothing
            println(io, "- Iterations: `$(length(records))`")
            println(io, "- Density: `$(final.density)`")
            println(io, "- Final mu-search status/evaluations: `$(final.mu_search_status)` / `$(final.mu_evaluations)`")
            println(io, "- Final density slope dn/dmu: `$(final.mu_density_slope)`")
            println(io, "- Canonical variational energy: `$(final.variational.canonical_variational_energy)`")
            println(io, "- Target-density correction / corrected energy: `$(final.variational.target_density_correction)` / `$(final.variational.target_density_corrected_variational_energy)`")
            println(io, "- Direct / reconstructed variational energy: `$(final.variational.direct_variational_energy)` / `$(final.variational.reconstructed_variational_energy)`")
            println(io, "- Variational consistency error: `$(final.variational.variational_consistency_error)`")
            println(io, "- Effective-H eigenvalue: `$(final.effective_energy)`")
            println(io, "- Effective-H expectation: `$(final.variational.effective_expectation)`")
            println(io, "- Double-counting correction: `$(final.variational.double_counting_correction)`")
            println(io, "- Final DMRG maximum discarded weight / link dimension: `$(final.dmrg_max_discarded_weight)` / `$(final.dmrg_maxlinkdim)`")
        end
        println(io)
        println(io, "## Control scales")
        println(io)
        println(io, "- Signed registry E_p: `$(settings.model.ep_signed)`")
        println(io, "- Denominator |E_p|: `$(settings.model.ep)`")
        println(io, "- E_p mode: `$(settings.model.ep_mode)`")
        println(io, "- E_p t0 bracket: `$(settings.model.ep_t0_lower)` to `$(settings.model.ep_t0_upper)` (weight `$(settings.model.ep_interpolation_weight)`)")
        println(io, "- E_p endpoint values: `$(settings.model.ep_lower_signed)` to `$(settings.model.ep_upper_signed)`")
        println(io, "- Effective MF coupling t_perp^2 / |E_p|: `$(settings.model.tp^2 / settings.model.ep)`")
        println(io, "- E_p registry: `$(settings.model.ep_source)`")
        println(io, "- Warm-start mu-resolve noise: `$(settings.dmrg.mu_warm_start_noise)`")
        println(io, "- Period-two cosine / two-step-ratio gates: `$(settings.convergence.period2_oscillation_cosine_max)` / `$(settings.convergence.period2_two_step_ratio_max)`")
        println(io, "- Slow-mode cosine gate: `$(settings.convergence.slow_mode_cosine_min)`")
        println(io)
        println(io, "This summary is generated evidence. Add collaborator interpretation to `docs/RUN_LOG.md`; do not edit an immutable HDF5 artifact.")
    end
    return path
end
