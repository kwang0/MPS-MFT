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
        file["schema_version"] = 4
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
        file["orbit_energy_spread"] = diagnostic.orbit_energy_spread
        file["orbit_density_contrast"] = diagnostic.orbit_density_contrast
        file["fixed_point_abs_residual"] = diagnostic.fixed_point_abs_residual
        file["fixed_point_rel_residual"] = diagnostic.fixed_point_rel_residual
        file["cycle_abs_residual"] = diagnostic.cycle_abs_residual
        file["cycle_rel_residual"] = diagnostic.cycle_rel_residual
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
            "conserve_sz" => settings.runtime.conserve_sz,
            "conserve_nfparity" => settings.runtime.conserve_nfparity,
        ))
        if !isempty(records)
            last_record = last(records)
            fields_group = create_group(file, "fields")
            applied_group = create_group(fields_group, "applied")
            measured_group = create_group(fields_group, "measured")
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
            history_group["effective_energy"] = [record.effective_energy for record in records]
            history_group["variational_energy"] = [record.variational.canonical_variational_energy for record in records]
            history_group["field_abs_residual"] = [record.field_abs_residual for record in records]
            history_group["field_rel_residual"] = [record.field_rel_residual for record in records]
            history_group["wall_seconds"] = [record.wall_seconds for record in records]
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

function read_checkpoint(path::AbstractString)
    isfile(path) || throw(ArgumentError("checkpoint not found: $path"))
    return h5open(path, "r") do file
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
        println(io, "- Geometry: `$(settings.model.geometry)`")
        println(io, "- Runtime backend: `$(settings.runtime.backend)`")
        println(io, "- Conserved S_z / fermion parity: `$(settings.runtime.conserve_sz)` / `$(settings.runtime.conserve_nfparity)`")
        println(io, "- Model fingerprint: `$(model_fingerprint(settings.model))`")
        println(io, "- Numerical fingerprint: `$(numerical_fingerprint(settings))`")
        println(io, "- Implementation SHA-256: `$(implementation_fingerprint())`")
        println(io, "- Configuration SHA-256: `$(isfile(settings.config_path) ? sha256_file(settings.config_path) : "not-file-backed")`")
        println(io, "- Parent checkpoint: `$(something(settings.run.parent_checkpoint, "none"))`")
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
        println(io, "- Orbit phase-energy spread: `$(diagnostic.orbit_energy_spread)`")
        println(io, "- Orbit density contrast: `$(diagnostic.orbit_density_contrast)`")
        println(io, "- Fixed-point residual (relative): `$(diagnostic.fixed_point_rel_residual)`")
        println(io, "- Hamiltonian identity error/site: `$(diagnostic.hamiltonian_identity_error_per_site)`")
        println(io, "- Effective eigenvalue error/site: `$(diagnostic.effective_eigenvalue_error_per_site)`")
        if final !== nothing
            println(io, "- Iterations: `$(length(records))`")
            println(io, "- Density: `$(final.density)`")
            println(io, "- Final mu-search status/evaluations: `$(final.mu_search_status)` / `$(final.mu_evaluations)`")
            println(io, "- Canonical variational energy: `$(final.variational.canonical_variational_energy)`")
            println(io, "- Direct / reconstructed variational energy: `$(final.variational.direct_variational_energy)` / `$(final.variational.reconstructed_variational_energy)`")
            println(io, "- Variational consistency error: `$(final.variational.variational_consistency_error)`")
            println(io, "- Effective-H eigenvalue: `$(final.effective_energy)`")
            println(io, "- Effective-H expectation: `$(final.variational.effective_expectation)`")
            println(io, "- Double-counting correction: `$(final.variational.double_counting_correction)`")
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
        println(io)
        println(io, "This summary is generated evidence. Add collaborator interpretation to `docs/RUN_LOG.md`; do not edit an immutable HDF5 artifact.")
    end
    return path
end
