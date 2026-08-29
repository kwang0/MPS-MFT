sha256_file(path::AbstractString) = bytes2hex(open(SHA.sha256, path))

function implementation_fingerprint(root::AbstractString=PROJECT_ROOT)
    files = String[]
    for (directory, subdirectories, names) in walkdir(root)
        filter!(name -> name != "output" && !startswith(name, "."), subdirectories)
        for name in names
            extension = lowercase(splitext(name)[2])
            extension in (".jl", ".toml", ".csv", ".sh") || continue
            push!(files, joinpath(directory, name))
        end
    end
    sort!(files; by=path -> relpath(path, root))
    context = SHA.SHA256_CTX()
    for path in files
        SHA.update!(context, codeunits(relpath(path, root)))
        SHA.update!(context, UInt8[0x00])
        SHA.update!(context, read(path))
        SHA.update!(context, UInt8[0x00])
    end
    return bytes2hex(SHA.digest!(context))
end

function _read_git(args...; default="unknown")
    try
        return strip(read(`git -C $PROJECT_ROOT $(args)`, String))
    catch
        return default
    end
end

function model_fingerprint(model::ModelSettings)
    payload = join((
        model.L, model.t, model.U, model.V, model.t0, model.tp, model.density, model.mu_initial,
        model.r_range, model.geometry, model.ep, model.ep_signed, model.ep_mode,
        model.ep_t0_lower, model.ep_t0_upper, model.ep_lower_signed, model.ep_upper_signed,
        model.ep_interpolation_weight, model.ep_lower_chi, model.ep_upper_chi,
    ), "|")
    return bytes2hex(SHA.sha256(payload))
end

function numerical_fingerprint(settings::ProjectSettings)
    values = Any[]
    for block in (settings.dmrg, settings.mixing, settings.convergence)
        for field in fieldnames(typeof(block))
            push!(values, field, getfield(block, field))
        end
    end
    # Device and symmetry representation can change the numerical trajectory;
    # CPU thread topology is performance-only and remains ordinary provenance.
    push!(
        values,
        :backend,
        settings.runtime.backend,
        :tensor_scalar_type,
        settings.runtime.tensor_scalar_type,
        :conserve_sz,
        settings.runtime.conserve_sz,
        :conserve_nfparity,
        settings.runtime.conserve_nfparity,
    )
    return bytes2hex(SHA.sha256(join(values, "|")))
end

function initial_seed_fingerprint(settings::ProjectSettings)
    run = settings.run
    values = Any[
        :L,
        settings.model.L,
        :density,
        settings.model.density,
        :r_range,
        settings.model.r_range,
        :protocol,
        run.initial_seed_protocol,
        :channel,
        run.initial_seed,
        :amplitude,
        run.initial_amplitude,
        :random_seed,
        run.random_seed,
    ]
    if run.initial_seed_protocol == :matched_mode
        append!(values, Any[
            :mode_number,
            run.initial_mode_number,
            :mode_phase_pi,
            run.initial_mode_phase_pi,
            :normalization,
            :full_field_l2_per_sqrt_physical_site,
        ])
        if run.initial_seed == :pairing
            append!(values, Any[
                :pairing_form_factor,
                run.initial_pairing_form_factor,
            ])
        elseif run.initial_seed in (:sdw, :cdw)
            append!(values, Any[
                :resolved_leg_parity,
                resolved_initial_leg_parity(run.initial_seed, run.initial_leg_parity),
            ])
        end
    end
    return bytes2hex(SHA.sha256(join(values, "|")))
end

function collect_provenance(settings::ProjectSettings)
    config_hash = isfile(settings.config_path) ? sha256_file(settings.config_path) : ""
    inherit_hash = settings.run.inherit_from === nothing ? "" :
        (settings.run.inherit_sha256 === nothing ? sha256_file(settings.run.inherit_from) : settings.run.inherit_sha256)
    parent_hash = settings.run.parent_checkpoint === nothing ? "" :
        (settings.run.parent_sha256 === nothing ? sha256_file(settings.run.parent_checkpoint) : settings.run.parent_sha256)
    resume_hash = settings.run.resume_checkpoint === nothing ? "" :
        (settings.run.resume_sha256 === nothing ? sha256_file(settings.run.resume_checkpoint) : settings.run.resume_sha256)
    gpu_manifest = joinpath(PROJECT_ROOT, "gpu", "Manifest.toml")
    seed_metadata = initial_seed_metadata(settings.model, settings.run)
    return Dict{String,Any}(
        "generated_utc" => string(now(UTC)),
        "schema_version" => 3,
        "git_commit" => _read_git("rev-parse", "HEAD"),
        "git_branch" => _read_git("branch", "--show-current"),
        "git_dirty" => !isempty(_read_git("status", "--porcelain"; default="")),
        "implementation_sha256" => implementation_fingerprint(),
        "model_fingerprint" => model_fingerprint(settings.model),
        "numerical_fingerprint" => numerical_fingerprint(settings),
        "config_path" => settings.config_path,
        "config_sha256" => config_hash,
        "ep_source" => settings.model.ep_source,
        "ep_source_sha256" => isfile(settings.model.ep_source) ? sha256_file(settings.model.ep_source) : "",
        "ep_mode" => String(settings.model.ep_mode),
        "ep_signed" => settings.model.ep_signed,
        "ep_denominator" => settings.model.ep,
        "ep_t0_lower" => settings.model.ep_t0_lower,
        "ep_t0_upper" => settings.model.ep_t0_upper,
        "ep_lower_signed" => settings.model.ep_lower_signed,
        "ep_upper_signed" => settings.model.ep_upper_signed,
        "ep_interpolation_weight" => settings.model.ep_interpolation_weight,
        "ep_lower_chi" => settings.model.ep_lower_chi,
        "ep_upper_chi" => settings.model.ep_upper_chi,
        "effective_mf_coupling_tp2_over_ep" => settings.model.tp^2 / settings.model.ep,
        "runtime_backend" => String(settings.runtime.backend),
        "tensor_scalar_type" => String(settings.runtime.tensor_scalar_type),
        "conserve_sz" => settings.runtime.conserve_sz,
        "conserve_nfparity" => settings.runtime.conserve_nfparity,
        "inherit_from" => something(settings.run.inherit_from, ""),
        "inherit_sha256" => inherit_hash,
        "parent_checkpoint" => something(settings.run.parent_checkpoint, ""),
        "parent_sha256" => parent_hash,
        "parent_orbit_phase" => something(settings.run.parent_orbit_phase, 0),
        "resume_checkpoint" => something(settings.run.resume_checkpoint, ""),
        "resume_sha256" => resume_hash,
        "branch_label" => settings.run.branch_label,
        "preparation" => settings.run.preparation,
        "direction" => settings.run.direction,
        "seed_label" => settings.run.seed_label,
        "initial_seed" => String(settings.run.initial_seed),
        "initial_amplitude" => settings.run.initial_amplitude,
        "random_seed" => settings.run.random_seed,
        "initial_seed_protocol" => String(seed_metadata.protocol),
        "initial_seed_fingerprint" => initial_seed_fingerprint(settings),
        "initial_mode_number" => seed_metadata.mode_number,
        "initial_mode_wavevector_pi" => seed_metadata.mode_wavevector_pi,
        "initial_mode_phase_pi" => seed_metadata.mode_phase_pi,
        "initial_mode_basis" => seed_metadata.mode_basis,
        "initial_pairing_form_factor" => String(seed_metadata.pairing_form_factor),
        "initial_leg_parity_requested" => String(seed_metadata.requested_leg_parity),
        "initial_leg_parity_resolved" => String(seed_metadata.resolved_leg_parity),
        "initial_seed_normalization" => seed_metadata.normalization,
        "initial_seed_target_l2_per_physical_site" => seed_metadata.target_field_l2_per_physical_site,
        "julia_version" => string(VERSION),
        "itensors_version" => string(Base.pkgversion(ITensors)),
        "itensormps_version" => string(Base.pkgversion(ITensorMPS)),
        "gpu_manifest_path" => isfile(gpu_manifest) ? gpu_manifest : "",
        "gpu_manifest_sha256" => isfile(gpu_manifest) ? sha256_file(gpu_manifest) : "",
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "hostname" => get(ENV, "HOSTNAME", "unknown"),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
        "slurm_cpus_per_task" => get(ENV, "SLURM_CPUS_PER_TASK", ""),
    )
end

function verify_inherit!(settings::ProjectSettings)
    source = settings.run.inherit_from
    source === nothing && return nothing
    isfile(source) || throw(ArgumentError("inherited field state does not exist: $source"))
    expected = settings.run.inherit_sha256
    expected === nothing && throw(ArgumentError("inherit_sha256 is required when inherit_from is set"))
    actual = sha256_file(source)
    actual == lowercase(expected) || throw(ArgumentError("inherited field-state SHA-256 mismatch"))
    return actual
end

function verify_parent!(settings::ProjectSettings)
    parent = settings.run.parent_checkpoint
    parent === nothing && return nothing
    isfile(parent) || throw(ArgumentError("parent checkpoint does not exist: $parent"))
    expected = settings.run.parent_sha256
    expected === nothing && throw(ArgumentError("parent_sha256 is required when parent_checkpoint is set"))
    actual = sha256_file(parent)
    actual == lowercase(expected) || throw(ArgumentError("parent checkpoint SHA-256 mismatch"))
    return actual
end

function verify_resume!(settings::ProjectSettings)
    checkpoint = settings.run.resume_checkpoint
    checkpoint === nothing && return nothing
    isfile(checkpoint) || throw(ArgumentError("resume checkpoint does not exist: $checkpoint"))
    expected = settings.run.resume_sha256
    expected === nothing && throw(ArgumentError("resume_sha256 is required when resume_checkpoint is set"))
    actual = sha256_file(checkpoint)
    actual == lowercase(expected) || throw(ArgumentError("resume checkpoint SHA-256 mismatch"))
    return actual
end
