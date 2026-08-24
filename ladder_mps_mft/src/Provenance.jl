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

function collect_provenance(settings::ProjectSettings)
    config_hash = isfile(settings.config_path) ? sha256_file(settings.config_path) : ""
    parent_hash = settings.run.parent_checkpoint === nothing ? "" :
        (settings.run.parent_sha256 === nothing ? sha256_file(settings.run.parent_checkpoint) : settings.run.parent_sha256)
    resume_hash = settings.run.resume_checkpoint === nothing ? "" :
        (settings.run.resume_sha256 === nothing ? sha256_file(settings.run.resume_checkpoint) : settings.run.resume_sha256)
    gpu_manifest = joinpath(PROJECT_ROOT, "gpu", "Manifest.toml")
    return Dict{String,Any}(
        "generated_utc" => string(now(UTC)),
        "schema_version" => 1,
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
        "parent_checkpoint" => something(settings.run.parent_checkpoint, ""),
        "parent_sha256" => parent_hash,
        "resume_checkpoint" => something(settings.run.resume_checkpoint, ""),
        "resume_sha256" => resume_hash,
        "branch_label" => settings.run.branch_label,
        "preparation" => settings.run.preparation,
        "direction" => settings.run.direction,
        "seed_label" => settings.run.seed_label,
        "initial_seed" => String(settings.run.initial_seed),
        "initial_amplitude" => settings.run.initial_amplitude,
        "random_seed" => settings.run.random_seed,
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
