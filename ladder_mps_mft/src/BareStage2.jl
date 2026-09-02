const STAGE2_PAIR_FORM_FACTORS = (:onsite_s, :rung_s, :leg_s, :extended_s, :d_wave)

function _stage2_artifact_sha256(source::AbstractString)
    sidecar = abspath(source) * ".sha256"
    if isfile(sidecar)
        fields = split(strip(read(sidecar, String)))
        !isempty(fields) && occursin(r"^[0-9a-fA-F]{64}$", fields[1]) &&
            return lowercase(fields[1])
    end
    return sha256_file(abspath(source))
end

function load_bare_stage2_settings(path::AbstractString, model::ModelSettings)
    raw = TOML.parsefile(abspath(path))
    table = _table(raw, "stage2")
    settings = BareStage2Settings(;
        field_strength=Float64(_value(table, "field_strength", 1e-4)),
        validation_field_strength=Float64(_value(table, "validation_field_strength", 5e-5)),
        charge_even_modes=Int.(_value(table, "charge_even_modes", [7, 8, 9])),
        spin_odd_modes=Int.(_value(table, "spin_odd_modes", [58, 59, 63])),
        pair_form_factors=Symbol.(lowercase.(String.(_value(
            table,
            "pair_form_factors",
            String.(STAGE2_PAIR_FORM_FACTORS),
        )))),
        covariance_candidates=String.(_value(
            table,
            "covariance_candidates",
            ["charge:even:1", "charge:odd:1", "spin:even:1"],
        )),
        geometries=normalize_geometry.(_value(
            table,
            "geometries",
            String.(SUPPORTED_GEOMETRIES),
        )),
        maxdim=Int(_value(table, "maxdim", 1200)),
        normal_reference_sweeps=Int(_value(table, "normal_reference_sweeps", 20)),
        pair_reference_sweeps=Int(_value(table, "pair_reference_sweeps", 20)),
        probe_sweeps=Int(_value(table, "probe_sweeps", 20)),
        validation_sweeps=Int(_value(table, "validation_sweeps", 20)),
        minimum_convergence_sweep=Int(_value(table, "minimum_convergence_sweep", 4)),
        cutoff=Float64(_value(table, "cutoff", 1e-10)),
        energy_tol=Float64(_value(table, "energy_tol", 1e-9)),
        noise_floor=Float64(_value(table, "noise_floor", 1e-8)),
        last_five_energy_tol=Float64(_value(table, "last_five_energy_tol", 1e-6)),
        density_tol=Float64(_value(table, "density_tol", 2e-4)),
        orthogonalization_tol=Float64(_value(table, "orthogonalization_tol", 1e-10)),
        reciprocity_relative_tol=Float64(_value(table, "reciprocity_relative_tol", 5e-2)),
        cross_block_relative_tol=Float64(_value(table, "cross_block_relative_tol", 5e-2)),
        linearity_relative_tol=Float64(_value(table, "linearity_relative_tol", 5e-2)),
        top_validation_modes=Int(_value(table, "top_validation_modes", 3)),
        eigsolve_krylovdim=Int(_value(table, "eigsolve_krylovdim", 8)),
        output_level=Int(_value(table, "output_level", 1)),
        maximum_job_seconds=Float64(_value(table, "maximum_job_seconds", 11.5 * 3600)),
    )
    settings.field_strength > 0 || throw(ArgumentError("stage2 field_strength must be positive"))
    0 < settings.validation_field_strength < settings.field_strength || throw(ArgumentError(
        "stage2 validation_field_strength must lie strictly between zero and field_strength",
    ))
    isempty(settings.charge_even_modes) && throw(ArgumentError("charge_even_modes cannot be empty"))
    isempty(settings.spin_odd_modes) && throw(ArgumentError("spin_odd_modes cannot be empty"))
    all(mode -> 1 <= mode <= model.L - 1, settings.charge_even_modes) || throw(ArgumentError(
        "charge_even_modes must lie between 1 and L-1",
    ))
    all(mode -> 0 <= mode <= model.L - 1, settings.spin_odd_modes) || throw(ArgumentError(
        "spin_odd_modes must lie between 0 and L-1",
    ))
    length(unique(settings.charge_even_modes)) == length(settings.charge_even_modes) ||
        throw(ArgumentError("charge_even_modes contains duplicates"))
    length(unique(settings.spin_odd_modes)) == length(settings.spin_odd_modes) ||
        throw(ArgumentError("spin_odd_modes contains duplicates"))
    all(in(STAGE2_PAIR_FORM_FACTORS), settings.pair_form_factors) || throw(ArgumentError(
        "unknown Stage 2 pairing form factor",
    ))
    length(unique(settings.pair_form_factors)) == length(settings.pair_form_factors) ||
        throw(ArgumentError("pair_form_factors contains duplicates"))
    all(in(SUPPORTED_GEOMETRIES), settings.geometries) || throw(ArgumentError(
        "unknown Stage 2 geometry",
    ))
    length(unique(settings.geometries)) == length(settings.geometries) ||
        throw(ArgumentError("stage2 geometries contains duplicates"))
    settings.maxdim > 0 || throw(ArgumentError("stage2 maxdim must be positive"))
    minimum((settings.normal_reference_sweeps, settings.pair_reference_sweeps,
        settings.probe_sweeps, settings.validation_sweeps)) >=
        settings.minimum_convergence_sweep || throw(ArgumentError(
            "every Stage 2 sweep budget must reach minimum_convergence_sweep",
        ))
    settings.minimum_convergence_sweep >= 2 || throw(ArgumentError(
        "stage2 minimum_convergence_sweep must be at least two",
    ))
    settings.cutoff > 0 || throw(ArgumentError("stage2 cutoff must be positive"))
    settings.energy_tol >= 0 || throw(ArgumentError("stage2 energy_tol must be nonnegative"))
    settings.noise_floor >= 0 || throw(ArgumentError("stage2 noise_floor must be nonnegative"))
    settings.last_five_energy_tol > 0 || throw(ArgumentError(
        "stage2 last_five_energy_tol must be positive",
    ))
    settings.density_tol > 0 || throw(ArgumentError("stage2 density_tol must be positive"))
    settings.orthogonalization_tol > 0 || throw(ArgumentError(
        "stage2 orthogonalization_tol must be positive",
    ))
    settings.reciprocity_relative_tol > 0 || throw(ArgumentError(
        "stage2 reciprocity_relative_tol must be positive",
    ))
    settings.cross_block_relative_tol > 0 || throw(ArgumentError(
        "stage2 cross_block_relative_tol must be positive",
    ))
    settings.linearity_relative_tol > 0 || throw(ArgumentError(
        "stage2 linearity_relative_tol must be positive",
    ))
    settings.top_validation_modes >= 2 || throw(ArgumentError(
        "stage2 top_validation_modes must be at least two so normal and pairing are both checked",
    ))
    settings.maximum_job_seconds > 0 || throw(ArgumentError(
        "stage2 maximum_job_seconds must be positive",
    ))
    for specification in settings.covariance_candidates
        occursin(r"^(charge|spin):(even|odd):[1-9][0-9]*$", specification) ||
            throw(ArgumentError("invalid covariance candidate '$specification'"))
    end
    return settings
end

function _write_stage2_zero_field_reference(
    destination::AbstractString,
    artifact_kind::AbstractString,
    result,
    convergence,
    density::Real,
    correlations::CorrelationState,
    particle_number::Integer,
    bank,
    config_path::AbstractString;
    parent_reference_path::AbstractString="",
    parent_reference_sha256::AbstractString="",
)
    output = abspath(destination)
    ispath(output) && throw(ArgumentError("refusing to overwrite immutable reference: $output"))
    mkpath(dirname(output))
    temporary = tempname(dirname(output))
    h5open(temporary, "w") do file
        file["schema_version"] = 1
        file["artifact_kind"] = String(artifact_kind)
        file["complete"] = true
        file["scientifically_accepted"] = convergence.accepted
        file["particle_number_target"] = Int(particle_number)
        file["density"] = Float64(density)
        file["chemical_potential"] = bank.chemical_potential
        file["psi"] = move_to_cpu(result.psi)
        _write_correlations(create_group(file, "zero_field_correlations"), correlations)
        _write_stage2_dmrg(create_group(file, "dmrg"), result, convergence)
        source = create_group(file, "sources")
        source["candidate_bank_path"] = bank.path
        source["candidate_bank_sha256"] = bank.sha256
        source["candidate_bank_fingerprint"] = bank.bank_fingerprint
        source["backbone_path"] = bank.backbone_path
        source["backbone_sha256"] = bank.backbone_sha256
        source["parent_reference_path"] = String(parent_reference_path)
        source["parent_reference_sha256"] = String(parent_reference_sha256)
        provenance = create_group(file, "provenance")
        provenance["config_path"] = abspath(config_path)
        provenance["config_sha256"] = sha256_file(abspath(config_path))
        provenance["implementation_sha256"] = implementation_fingerprint()
        provenance["git_commit"] = _read_git("rev-parse", "HEAD")
        provenance["julia_threads"] = Threads.nthreads()
    end
    mv(temporary, output)
    return (;
        path=output,
        sha256=sha256_file(output),
        accepted=convergence.accepted,
        density=Float64(density),
        energy=result.energy,
        last_five_energy_change=convergence.last_five_energy_change,
    )
end

function run_stage2_normal_reference(
    candidate_bank_path::AbstractString,
    config_path::AbstractString,
    output_path::AbstractString,
)
    project = load_settings(config_path)
    project.runtime.backend == :cpu || throw(ArgumentError("Stage 2 requires runtime.backend=cpu"))
    stage2 = load_bare_stage2_settings(config_path, project.model)
    bank = read_stage2_candidate_bank(candidate_bank_path)
    bank.config_sha256 == sha256_file(abspath(config_path)) || throw(ArgumentError(
        "Stage 2 candidate bank configuration hash mismatch",
    ))
    ground = _read_stage2_ground_state(bank)
    zero = zero_field_state(project.model)
    sites = siteinds(ground.psi)
    hamiltonian = build_mf_mpo(
        sites,
        project.model,
        zero,
        bank.chemical_potential;
        backend=project.runtime,
    )
    dmrg_settings = _stage2_dmrg_settings(stage2, stage2.normal_reference_sweeps)
    result = run_dmrg_ground(
        sites,
        hamiltonian,
        project.model.density,
        dmrg_settings;
        psi_init=ground.psi,
        rng=MersenneTwister(project.run.random_seed + 1900),
        deadline=time() + stage2.maximum_job_seconds,
        backend=project.runtime,
        noise_schedule=_stage2_noise_schedule(stage2, stage2.normal_reference_sweeps),
        minimum_convergence_sweep=stage2.minimum_convergence_sweep,
    )
    _, correlations = calculate_mean_fields(result.psi, project.model; threshold=0.0)
    density = (sum(correlations.density_down) + sum(correlations.density_up)) /
        (2 * project.model.L)
    convergence = _stage2_scientifically_converged(
        result,
        density,
        project.model,
        stage2,
    )
    return _write_stage2_zero_field_reference(
        output_path,
        "bare_ladder_stage2_normal_reference",
        result,
        convergence,
        density,
        correlations,
        ground.particle_number,
        bank,
        config_path,
    )
end

function _read_stage2_zero_field_reference(
    path::AbstractString,
    expected_kind::AbstractString,
    bank;
    load_state::Bool=true,
    calculate_hash::Bool=true,
)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("Stage 2 zero-field reference not found: $source"))
    payload = h5open(source, "r") do file
        String(read(file, "artifact_kind")) == expected_kind || throw(ArgumentError(
            "unexpected Stage 2 reference kind at $source",
        ))
        Bool(read(file, "complete")) || throw(ArgumentError("incomplete Stage 2 reference: $source"))
        Bool(read(file, "scientifically_accepted")) || throw(ArgumentError(
            "Stage 2 zero-field reference failed its convergence gates: $source",
        ))
        String(read(file, "sources/candidate_bank_sha256")) == bank.sha256 ||
            throw(ArgumentError("zero-field reference candidate-bank hash mismatch"))
        group = file["zero_field_correlations"]
        correlations = CorrelationState(
            Float64.(read(group, "pair")),
            Float64.(read(group, "exchange_down")),
            Float64.(read(group, "exchange_up")),
            Float64.(read(group, "density_down")),
            Float64.(read(group, "density_up")),
        )
        (;
            path=source,
            psi=load_state ? read(file, "psi", MPS) : nothing,
            correlations,
            density=Float64(read(file, "density")),
            energy=Float64(read(file, "dmrg/energy")),
            particle_number=Int(read(file, "particle_number_target")),
        )
    end
    return calculate_hash ? merge(payload, (; sha256=_stage2_artifact_sha256(source))) : payload
end

read_stage2_normal_reference(path::AbstractString, bank; kwargs...) =
    _read_stage2_zero_field_reference(
        path,
        "bare_ladder_stage2_normal_reference",
        bank;
        kwargs...,
    )

function zero_field_state(model::ModelSettings)
    return FieldState(
        zeros(Float64, model.L, model.L, 2, 2),
        zeros(Float64, 2, model.L, model.L, 2, 2),
        zeros(Float64, 2, 2 * model.L),
    )
end

function field_metric_dot(left::FieldState, right::FieldState, model::ModelSettings)
    size(left.alpha) == size(right.alpha) || throw(DimensionMismatch("alpha field shapes differ"))
    size(left.beta) == size(right.beta) || throw(DimensionMismatch("beta field shapes differ"))
    size(left.mu_cdw) == size(right.mu_cdw) || throw(DimensionMismatch("Hartree field shapes differ"))
    return (
        sum(left.alpha .* right.alpha) +
        sum(left.beta .* right.beta) +
        sum(left.mu_cdw .* right.mu_cdw)
    ) / (2 * model.L)
end

field_metric_norm(fields::FieldState, model::ModelSettings) =
    sqrt(max(field_metric_dot(fields, fields, model), 0.0))

function scale_fields(fields::FieldState, scale::Real)
    value = Float64(scale)
    return FieldState(value .* fields.alpha, value .* fields.beta, value .* fields.mu_cdw)
end

function subtract_fields(left::FieldState, right::FieldState)
    return FieldState(
        left.alpha .- right.alpha,
        left.beta .- right.beta,
        left.mu_cdw .- right.mu_cdw,
    )
end

function _field_axpy(fields::FieldState, scale::Real, direction::FieldState)
    value = Float64(scale)
    return FieldState(
        fields.alpha .+ value .* direction.alpha,
        fields.beta .+ value .* direction.beta,
        fields.mu_cdw .+ value .* direction.mu_cdw,
    )
end

function normalize_fields(fields::FieldState, model::ModelSettings)
    magnitude = field_metric_norm(fields, model)
    magnitude > eps(Float64) || throw(ArgumentError("cannot normalize a zero field direction"))
    return scale_fields(fields, inv(magnitude))
end

function _model_with_geometry(
    model::ModelSettings,
    geometry::Symbol;
    ep_signed::Real=model.ep_signed,
)
    kernel_ep_signed = Float64(ep_signed)
    isfinite(kernel_ep_signed) && kernel_ep_signed != 0 || throw(ArgumentError(
        "Stage 2 geometry kernel requires a finite nonzero pair binding",
    ))
    return ModelSettings(;
        L=model.L,
        t=model.t,
        U=model.U,
        V=model.V,
        t0=model.t0,
        tp=model.tp,
        density=model.density,
        mu_initial=model.mu_initial,
        r_range=model.r_range,
        geometry,
        ep=abs(kernel_ep_signed),
        ep_signed=kernel_ep_signed,
        ep_source=model.ep_source,
        ep_mode=model.ep_mode,
        ep_t0_lower=model.ep_t0_lower,
        ep_t0_upper=model.ep_t0_upper,
        ep_lower_signed=model.ep_lower_signed,
        ep_upper_signed=model.ep_upper_signed,
        ep_interpolation_weight=model.ep_interpolation_weight,
        ep_lower_chi=model.ep_lower_chi,
        ep_upper_chi=model.ep_upper_chi,
    )
end

function _normal_covariance_fields(
    vector::AbstractVector,
    model::ModelSettings,
    channel::Symbol,
    parity::Symbol,
)
    length(vector) == model.L || throw(DimensionMismatch(
        "Stage 1 covariance vector length differs from model.L",
    ))
    channel in (:charge, :spin) || throw(ArgumentError("unknown normal channel '$channel'"))
    parity in (:even, :odd) || throw(ArgumentError("unknown leg parity '$parity'"))
    fields = zero_field_state(model)
    for rung in 1:model.L, leg in 0:1
        site = rung_leg_to_site(rung, leg)
        leg_sign = parity == :odd && leg == 1 ? -1.0 : 1.0
        value = Float64(vector[rung]) * leg_sign
        if channel == :charge
            fields.mu_cdw[:, site] .= value
        else
            fields.mu_cdw[1, site] = value
            fields.mu_cdw[2, site] = -value
        end
    end
    return normalize_fields(fields, model)
end

function _canonicalize_mode_sign(vector::AbstractVector)
    values = Float64.(vector)
    pivot = argmax(abs.(values))
    values[pivot] < 0 && (values .*= -1)
    return values
end

function _candidate_metadata(;
    label,
    block,
    channel,
    parity,
    origin,
    mode_number=0,
    q_over_pi=NaN,
    form_factor=:none,
    covariance_rank=0,
    covariance_eigenvalue=NaN,
    covariance_edge_weight=NaN,
    fields,
)
    return (;
        label=String(label),
        block=Symbol(block),
        channel=Symbol(channel),
        parity=Symbol(parity),
        origin=Symbol(origin),
        mode_number=Int(mode_number),
        q_over_pi=Float64(q_over_pi),
        form_factor=Symbol(form_factor),
        covariance_rank=Int(covariance_rank),
        covariance_eigenvalue=Float64(covariance_eigenvalue),
        covariance_edge_weight=Float64(covariance_edge_weight),
        fields,
    )
end

function _read_stage1_covariance_candidate(
    file,
    specification::AbstractString,
    model::ModelSettings,
)
    channel_raw, parity_raw, rank_raw = split(specification, ':')
    channel = Symbol(channel_raw)
    parity = Symbol(parity_raw)
    rank = parse(Int, rank_raw)
    base = "normal/$channel_raw/$parity_raw"
    vectors = read(file, "$base/eigenvectors")
    values = Float64.(read(file, "$base/eigenvalues"))
    modes = Int.(read(file, "$base/mode_number"))
    q_values = Float64.(read(file, "$base/q_over_pi"))
    edge = Float64.(read(file, "$base/edge_weight"))
    1 <= rank <= size(vectors, 2) || throw(ArgumentError(
        "Stage 1 covariance candidate '$specification' is unavailable; stored rank is $(size(vectors, 2))",
    ))
    vector = _canonicalize_mode_sign(vectors[:, rank])
    fields = _normal_covariance_fields(vector, model, channel, parity)
    return _candidate_metadata(;
        label="cov_$(channel_raw)_$(parity_raw)_rank$(rank)",
        block=:normal,
        channel,
        parity,
        origin=:stage1_covariance,
        mode_number=modes[rank],
        q_over_pi=q_values[rank],
        covariance_rank=rank,
        covariance_eigenvalue=values[rank],
        covariance_edge_weight=edge[rank],
        fields,
    )
end

function build_stage2_candidates(
    stage1_path::AbstractString,
    model::ModelSettings,
    settings::BareStage2Settings,
)
    candidates = NamedTuple[]
    for mode in settings.charge_even_modes
        fields = initial_fields(
            model;
            seed=:cdw,
            amplitude=1.0,
            protocol=:matched_mode,
            mode_number=mode,
            pairing_form_factor=:onsite_s,
            leg_parity=:even,
        )
        push!(candidates, _candidate_metadata(;
            label="charge_even_m$(mode)",
            block=:normal,
            channel=:charge,
            parity=:even,
            origin=:motivated,
            mode_number=mode,
            q_over_pi=initial_mode_wavevector_pi(model, mode),
            fields=normalize_fields(fields, model),
        ))
    end
    for mode in settings.spin_odd_modes
        fields = initial_fields(
            model;
            seed=:sdw,
            amplitude=1.0,
            protocol=:matched_mode,
            mode_number=mode,
            pairing_form_factor=:onsite_s,
            leg_parity=:odd,
        )
        push!(candidates, _candidate_metadata(;
            label="spin_odd_m$(mode)",
            block=:normal,
            channel=:spin,
            parity=:odd,
            origin=:motivated,
            mode_number=mode,
            q_over_pi=initial_mode_wavevector_pi(model, mode),
            fields=normalize_fields(fields, model),
        ))
    end
    h5open(abspath(stage1_path), "r") do file
        String(read(file, "artifact_kind")) == "bare_ladder_stage1_covariance_screen" ||
            throw(ArgumentError("not a Stage 1 covariance artifact: $stage1_path"))
        Bool(read(file, "complete")) || throw(ArgumentError("incomplete Stage 1 artifact"))
        Bool(read(file, "covariance_psd_pass")) || throw(ArgumentError(
            "Stage 1 covariance artifact failed its PSD gate",
        ))
        for specification in settings.covariance_candidates
            push!(candidates, _read_stage1_covariance_candidate(
                file,
                specification,
                model,
            ))
        end
    end
    for form_factor in settings.pair_form_factors
        fields = initial_fields(
            model;
            seed=:pairing,
            amplitude=1.0,
            protocol=:matched_mode,
            mode_number=0,
            pairing_form_factor=form_factor,
            leg_parity=:auto,
        )
        push!(candidates, _candidate_metadata(;
            label="pair_$(form_factor)_q0",
            block=:pair,
            channel=:pair,
            parity=:even,
            origin=:motivated,
            mode_number=0,
            q_over_pi=0.0,
            form_factor,
            fields=normalize_fields(fields, model),
        ))
    end
    return candidates
end

function orthonormalize_stage2_candidates(
    candidates::AbstractVector,
    model::ModelSettings;
    tolerance::Real=1e-10,
)
    tolerance > 0 || throw(ArgumentError("orthonormalization tolerance must be positive"))
    basis = NamedTuple[]
    residual_norms = Float64[]
    retained_basis_index = Int[]
    for (candidate_index, candidate) in enumerate(candidates)
        residual = copy(candidate.fields)
        for _ in 1:2
            for basis_item in basis
                coefficient = field_metric_dot(basis_item.fields, residual, model)
                residual = _field_axpy(residual, -coefficient, basis_item.fields)
            end
        end
        magnitude = field_metric_norm(residual, model)
        push!(residual_norms, magnitude)
        if magnitude > tolerance
            fields = scale_fields(residual, inv(magnitude))
            push!(basis, merge(candidate, (;
                fields,
                source_candidate_index=candidate_index,
                basis_index=length(basis) + 1,
            )))
            push!(retained_basis_index, length(basis))
        else
            push!(retained_basis_index, 0)
        end
    end
    projections = [
        field_metric_dot(basis_item.fields, candidate.fields, model)
        for basis_item in basis, candidate in candidates
    ]
    gram = [
        field_metric_dot(left.fields, right.fields, model)
        for left in basis, right in basis
    ]
    maximum_orthogonality_error = isempty(basis) ? 0.0 : maximum(abs.(gram - I))
    maximum_orthogonality_error <= 100 * tolerance || throw(ArgumentError(
        "Stage 2 basis orthogonality error $maximum_orthogonality_error exceeds tolerance",
    ))
    return (;
        candidates=collect(candidates),
        basis,
        candidate_projection=projections,
        candidate_residual_norm=residual_norms,
        candidate_retained_basis_index=retained_basis_index,
        gram,
        maximum_orthogonality_error,
    )
end

function _stage2_bank_fingerprint(bank, model::ModelSettings)
    io = IOBuffer()
    write(io, "stage2_candidate_bank_v1\n")
    write(io, string(model.L, '|', model.U, '|', model.V, '|', model.t0, '|', model.density, '\n'))
    for item in bank.basis
        write(io, item.label)
        write(io, UInt8(0))
        for component in (item.fields.alpha, item.fields.beta, item.fields.mu_cdw)
            write(io, reinterpret(UInt8, vec(component)))
        end
    end
    return bytes2hex(sha256(take!(io)))
end

function _validate_stage2_model(file, model::ModelSettings)
    for name in (:L, :r_range)
        Int(read(file, "model/$(String(name))")) == getproperty(model, name) ||
            throw(ArgumentError("Stage 2 model mismatch in $name"))
    end
    for name in (:t, :U, :V, :t0, :tp, :density, :ep, :ep_signed)
        isapprox(
            Float64(read(file, "model/$(String(name))")),
            getproperty(model, name);
            rtol=1e-13,
            atol=1e-14,
        ) || throw(ArgumentError("Stage 2 model mismatch in $name"))
    end
    Symbol(String(read(file, "model/geometry"))) == model.geometry || throw(ArgumentError(
        "Stage 2 model geometry differs from the source artifact",
    ))
    return true
end

function _read_stage1_zero_field_correlations(file)
    return CorrelationState(
        Float64.(read(file, "zero_field_raw_map/pair")),
        Float64.(read(file, "zero_field_raw_map/exchange_down")),
        Float64.(read(file, "zero_field_raw_map/exchange_up")),
        Float64.(read(file, "zero_field_raw_map/density_down")),
        Float64.(read(file, "zero_field_raw_map/density_up")),
    )
end

function _write_stage2_item(group, item)
    group["label"] = item.label
    group["block"] = String(item.block)
    group["channel"] = String(item.channel)
    group["parity"] = String(item.parity)
    group["origin"] = String(item.origin)
    group["mode_number"] = item.mode_number
    group["q_over_pi"] = item.q_over_pi
    group["form_factor"] = String(item.form_factor)
    group["covariance_rank"] = item.covariance_rank
    group["covariance_eigenvalue"] = item.covariance_eigenvalue
    group["covariance_edge_weight"] = item.covariance_edge_weight
    hasproperty(item, :source_candidate_index) &&
        (group["source_candidate_index"] = item.source_candidate_index)
    hasproperty(item, :basis_index) && (group["basis_index"] = item.basis_index)
    _write_fields(create_group(group, "fields"), item.fields)
    return group
end

function _read_stage2_item(group)
    return (;
        label=String(read(group, "label")),
        block=Symbol(String(read(group, "block"))),
        channel=Symbol(String(read(group, "channel"))),
        parity=Symbol(String(read(group, "parity"))),
        origin=Symbol(String(read(group, "origin"))),
        mode_number=Int(read(group, "mode_number")),
        q_over_pi=Float64(read(group, "q_over_pi")),
        form_factor=Symbol(String(read(group, "form_factor"))),
        covariance_rank=Int(read(group, "covariance_rank")),
        covariance_eigenvalue=Float64(read(group, "covariance_eigenvalue")),
        covariance_edge_weight=Float64(read(group, "covariance_edge_weight")),
        source_candidate_index=haskey(group, "source_candidate_index") ?
            Int(read(group, "source_candidate_index")) : 0,
        basis_index=haskey(group, "basis_index") ? Int(read(group, "basis_index")) : 0,
        fields=_read_fields(group["fields"]),
    )
end

function write_stage2_candidate_bank(
    destination::AbstractString,
    stage1_path::AbstractString,
    backbone_path::AbstractString,
    model::ModelSettings,
    settings::BareStage2Settings,
    config_path::AbstractString;
    immutable::Bool=true,
)
    output = abspath(destination)
    immutable && ispath(output) && throw(ArgumentError(
        "refusing to overwrite immutable Stage 2 candidate bank: $output",
    ))
    stage1_source = abspath(stage1_path)
    backbone_source = abspath(backbone_path)
    isfile(stage1_source) || throw(ArgumentError("Stage 1 artifact not found: $stage1_source"))
    isfile(backbone_source) || throw(ArgumentError("backbone artifact not found: $backbone_source"))
    backbone_sha256 = sha256_file(backbone_source)
    stage1_sha256 = sha256_file(stage1_source)
    baseline = h5open(stage1_source, "r") do file
        _validate_stage2_model(file, model)
        recorded_backbone = String(read(file, "provenance/backbone_sha256"))
        recorded_backbone == backbone_sha256 || throw(ArgumentError(
            "Stage 1 artifact does not reference the supplied backbone hash",
        ))
        _read_stage1_zero_field_correlations(file)
    end
    ground_energy, chemical_potential, kernel_pair_binding = h5open(backbone_source, "r") do file
        String(read(file, "artifact_kind")) == "isolated_ladder_backbone" ||
            throw(ArgumentError("not an isolated-ladder backbone: $backbone_source"))
        Bool(read(file, "scientifically_accepted")) || throw(ArgumentError(
            "backbone failed its scientific acceptance gate",
        ))
        _validate_stage2_model(file, model)
        target = Int(read(file, "energies/particle_number"))
        label = backbone_sector_label(target, 0)
        (
            Float64(read(file, "sectors/$label/energy")),
            Float64(read(file, "energies/chemical_potential")),
            Float64(read(file, "energies/hole_pair_binding")),
        )
    end
    isfinite(kernel_pair_binding) && kernel_pair_binding != 0 || throw(ArgumentError(
        "backbone hole pair binding is not finite and nonzero",
    ))
    candidates = build_stage2_candidates(stage1_source, model, settings)
    bank = orthonormalize_stage2_candidates(
        candidates,
        model;
        tolerance=settings.orthogonalization_tol,
    )
    fingerprint = _stage2_bank_fingerprint(bank, model)
    normal_indices = [item.basis_index for item in bank.basis if item.block == :normal]
    pair_indices = [item.basis_index for item in bank.basis if item.block == :pair]
    isempty(normal_indices) && throw(ArgumentError("Stage 2 bank has no normal directions"))
    isempty(pair_indices) && throw(ArgumentError("Stage 2 bank has no pairing directions"))
    mkpath(dirname(output))
    temporary = tempname(dirname(output))
    h5open(temporary, "w") do file
        file["schema_version"] = 1
        file["artifact_kind"] = "bare_ladder_stage2_candidate_bank"
        file["complete"] = true
        file["candidate_count"] = length(bank.candidates)
        file["basis_count"] = length(bank.basis)
        file["normal_basis_indices"] = normal_indices
        file["pair_basis_indices"] = pair_indices
        file["bank_fingerprint"] = fingerprint
        file["candidate_projection"] = bank.candidate_projection
        file["candidate_residual_norm"] = bank.candidate_residual_norm
        file["candidate_retained_basis_index"] = bank.candidate_retained_basis_index
        file["basis_gram"] = bank.gram
        file["maximum_orthogonality_error"] = bank.maximum_orthogonality_error
        file["field_metric"] = "Euclidean FieldState norm divided by 2L; every basis vector has unit norm"
        file["ground_energy"] = ground_energy
        file["chemical_potential"] = chemical_potential
        file["kernel_pair_binding"] = kernel_pair_binding
        file["kernel_pair_binding_source"] = "backbone/energies/hole_pair_binding"
        file["kernel_prefactor"] = 2 * model.tp^2 / abs(kernel_pair_binding)
        _write_backbone_model(create_group(file, "model"), model)
        candidate_group = create_group(file, "candidates")
        for (index, item) in enumerate(bank.candidates)
            _write_stage2_item(create_group(candidate_group, lpad(string(index), 3, '0')), item)
        end
        basis_group = create_group(file, "basis")
        for item in bank.basis
            _write_stage2_item(
                create_group(basis_group, lpad(string(item.basis_index), 3, '0')),
                item,
            )
        end
        _write_correlations(create_group(file, "zero_field_correlations"), baseline)
        source = create_group(file, "sources")
        source["stage1_path"] = stage1_source
        source["stage1_sha256"] = stage1_sha256
        source["stage1_size_bytes"] = Int64(stat(stage1_source).size)
        source["backbone_path"] = backbone_source
        source["backbone_sha256"] = backbone_sha256
        source["backbone_size_bytes"] = Int64(stat(backbone_source).size)
        provenance = create_group(file, "provenance")
        provenance["config_path"] = abspath(config_path)
        provenance["config_sha256"] = sha256_file(abspath(config_path))
        provenance["implementation_sha256"] = implementation_fingerprint()
        provenance["git_commit"] = _read_git("rev-parse", "HEAD")
        provenance["julia_threads"] = Threads.nthreads()
    end
    mv(temporary, output; force=!immutable)
    return (;
        path=output,
        sha256=sha256_file(output),
        candidate_count=length(bank.candidates),
        basis_count=length(bank.basis),
        normal_basis_count=length(normal_indices),
        pair_basis_count=length(pair_indices),
        fingerprint,
        maximum_orthogonality_error=bank.maximum_orthogonality_error,
    )
end

function read_stage2_candidate_bank(path::AbstractString)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("Stage 2 candidate bank not found: $source"))
    payload = h5open(source, "r") do file
        String(read(file, "artifact_kind")) == "bare_ladder_stage2_candidate_bank" ||
            throw(ArgumentError("not a Stage 2 candidate bank: $source"))
        Bool(read(file, "complete")) || throw(ArgumentError("incomplete Stage 2 candidate bank"))
        basis_parent = file["basis"]
        basis = [_read_stage2_item(basis_parent[name]) for name in sort(String.(collect(keys(basis_parent))))]
        length(basis) == Int(read(file, "basis_count")) || throw(ArgumentError(
            "Stage 2 candidate bank basis count mismatch",
        ))
        correlations_group = file["zero_field_correlations"]
        baseline = CorrelationState(
            Float64.(read(correlations_group, "pair")),
            Float64.(read(correlations_group, "exchange_down")),
            Float64.(read(correlations_group, "exchange_up")),
            Float64.(read(correlations_group, "density_down")),
            Float64.(read(correlations_group, "density_up")),
        )
        (;
            basis,
            normal_indices=Int.(read(file, "normal_basis_indices")),
            pair_indices=Int.(read(file, "pair_basis_indices")),
            bank_fingerprint=String(read(file, "bank_fingerprint")),
            candidate_count=Int(read(file, "candidate_count")),
            candidate_projection=Float64.(read(file, "candidate_projection")),
            candidate_residual_norm=Float64.(read(file, "candidate_residual_norm")),
            candidate_retained_basis_index=Int.(read(file, "candidate_retained_basis_index")),
            maximum_orthogonality_error=Float64(read(file, "maximum_orthogonality_error")),
            ground_energy=Float64(read(file, "ground_energy")),
            chemical_potential=Float64(read(file, "chemical_potential")),
            kernel_pair_binding=Float64(read(file, "kernel_pair_binding")),
            kernel_pair_binding_source=String(read(file, "kernel_pair_binding_source")),
            kernel_prefactor=Float64(read(file, "kernel_prefactor")),
            baseline,
            backbone_path=String(read(file, "sources/backbone_path")),
            backbone_sha256=String(read(file, "sources/backbone_sha256")),
            backbone_size_bytes=Int64(read(file, "sources/backbone_size_bytes")),
            stage1_path=String(read(file, "sources/stage1_path")),
            stage1_sha256=String(read(file, "sources/stage1_sha256")),
            config_sha256=String(read(file, "provenance/config_sha256")),
        )
    end
    return merge(payload, (; path=source, sha256=sha256_file(source)))
end

function _stage2_dmrg_settings(
    settings::BareStage2Settings,
    sweeps::Integer,
)
    return DMRGSettings(;
        nsweeps=Int(sweeps),
        maxdim=settings.maxdim,
        cutoff=settings.cutoff,
        energy_tol=settings.energy_tol,
        eigsolve_krylovdim=settings.eigsolve_krylovdim,
        max_time_seconds=settings.maximum_job_seconds,
        output_level=settings.output_level,
    )
end

function _stage2_noise_schedule(settings::BareStage2Settings, sweeps::Integer)
    count = Int(sweeps)
    count >= 1 || throw(ArgumentError("Stage 2 sweep count must be positive"))
    seed = settings.noise_floor
    pattern = seed == 0 ? [0.0] : [seed, seed, seed / 10, 0.0]
    return _extend_schedule(pattern, count)
end

function _stage2_scientifically_converged(
    result,
    density::Real,
    model::ModelSettings,
    settings::BareStage2Settings,
)
    last_five = last_five_sweep_change(result.sweep_energies)
    energy_pass = settings.energy_tol == 0 || result.energy_converged
    density_pass = abs(Float64(density) - model.density) <= settings.density_tol
    return (;
        accepted=!result.timed_out && energy_pass && density_pass &&
            last_five <= settings.last_five_energy_tol,
        energy_pass,
        density_pass,
        last_five_energy_change=last_five,
    )
end

function _read_stage2_ground_state(bank)
    stat(bank.backbone_path).size == bank.backbone_size_bytes || throw(ArgumentError(
        "backbone size changed after Stage 2 candidate-bank verification",
    ))
    return h5open(bank.backbone_path, "r") do file
        String(read(file, "artifact_kind")) == "isolated_ladder_backbone" ||
            throw(ArgumentError("not an isolated-ladder backbone: $(bank.backbone_path)"))
        Bool(read(file, "scientifically_accepted")) || throw(ArgumentError(
            "backbone failed its scientific acceptance gate",
        ))
        target = Int(read(file, "energies/particle_number"))
        label = backbone_sector_label(target, 0)
        (;
            psi=read(file, "sectors/$label/psi", MPS),
            particle_number=target,
            energy=Float64(read(file, "sectors/$label/energy")),
        )
    end
end

function _write_stage2_dmrg(group, result, convergence)
    group["energy"] = result.energy
    group["timed_out"] = result.timed_out
    group["energy_converged"] = result.energy_converged
    group["sweep_energy"] = result.sweep_energies
    group["sweep_max_discarded_weight"] = result.sweep_max_discarded_weights
    group["sweep_maxlinkdim"] = result.sweep_maxlinkdims
    group["max_discarded_weight"] = result.max_discarded_weight
    group["maximum_link_dimension"] = result.maximum_link_dimension
    group["last_five_energy_change"] = convergence.last_five_energy_change
    group["energy_pass"] = convergence.energy_pass
    group["density_pass"] = convergence.density_pass
    return group
end

function run_stage2_pair_reference(
    candidate_bank_path::AbstractString,
    normal_reference_path::AbstractString,
    config_path::AbstractString,
    output_path::AbstractString,
)
    project = load_settings(config_path)
    project.runtime.backend == :cpu || throw(ArgumentError("Stage 2 requires runtime.backend=cpu"))
    stage2 = load_bare_stage2_settings(config_path, project.model)
    bank = read_stage2_candidate_bank(candidate_bank_path)
    bank.config_sha256 == sha256_file(abspath(config_path)) || throw(ArgumentError(
        "Stage 2 candidate bank configuration hash mismatch",
    ))
    normal_reference = read_stage2_normal_reference(normal_reference_path, bank)
    pair_psi = ITensorMPS.removeqn(normal_reference.psi, "Nf")
    site_space = sprint(show, ITensorMPS.ITensors.space(siteind(pair_psi, 1)))
    occursin("NfParity", site_space) || throw(ArgumentError(
        "pairing reference lost fermion parity while removing Nf",
    ))
    !occursin("Nf,", site_space) || throw(ArgumentError(
        "pairing reference still carries Nf after removeqn",
    ))
    zero = zero_field_state(project.model)
    sites = siteinds(pair_psi)
    hamiltonian = build_mf_mpo(
        sites,
        project.model,
        zero,
        bank.chemical_potential;
        backend=project.runtime,
    )
    dmrg_settings = _stage2_dmrg_settings(stage2, stage2.pair_reference_sweeps)
    result = run_dmrg_ground(
        sites,
        hamiltonian,
        project.model.density,
        dmrg_settings;
        psi_init=pair_psi,
        rng=MersenneTwister(project.run.random_seed + 2000),
        deadline=time() + stage2.maximum_job_seconds,
        backend=project.runtime,
        noise_schedule=_stage2_noise_schedule(stage2, stage2.pair_reference_sweeps),
        minimum_convergence_sweep=stage2.minimum_convergence_sweep,
    )
    _, correlations = calculate_mean_fields(result.psi, project.model; threshold=0.0)
    density = (sum(correlations.density_down) + sum(correlations.density_up)) /
        (2 * project.model.L)
    convergence = _stage2_scientifically_converged(
        result,
        density,
        project.model,
        stage2,
    )
    return _write_stage2_zero_field_reference(
        output_path,
        "bare_ladder_stage2_pair_reference",
        result,
        convergence,
        density,
        correlations,
        normal_reference.particle_number,
        bank,
        config_path;
        parent_reference_path=normal_reference.path,
        parent_reference_sha256=normal_reference.sha256,
    )
end

function read_stage2_pair_reference(path::AbstractString, bank)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("Stage 2 pairing reference not found: $source"))
    payload = _read_stage2_pair_reference_unhashed(source, bank)
    return merge(payload, (; sha256=_stage2_artifact_sha256(source)))
end

function _read_stage2_pair_reference_unhashed(source::AbstractString, bank)
    return _read_stage2_zero_field_reference(
        source,
        "bare_ladder_stage2_pair_reference",
        bank;
        load_state=true,
        calculate_hash=false,
    )
end

function field_conjugate_expectation(
    direction::FieldState,
    correlations::CorrelationState,
    model::ModelSettings,
)
    value = 0.0
    for site in 1:(2 * model.L)
        value += direction.mu_cdw[1, site] * correlations.density_down[site]
        value += direction.mu_cdw[2, site] * correlations.density_up[site]
    end
    for i in 1:model.L, ip in 1:model.L
        abs(i - ip) <= model.r_range || continue
        for leg in 0:1, other_leg in 0:1
            site_i = rung_leg_to_site(i, leg)
            site_ip = rung_leg_to_site(ip, other_leg)
            alpha = direction.alpha[i, ip, leg + 1, other_leg + 1]
            value -= 2 * alpha * correlations.pair[site_ip, site_i]
            site_i == site_ip && continue
            value += direction.beta[1, i, ip, leg + 1, other_leg + 1] *
                correlations.exchange_down[site_i, site_ip]
            value += direction.beta[2, i, ip, leg + 1, other_leg + 1] *
                correlations.exchange_up[site_i, site_ip]
        end
    end
    return value
end

function _stage2_project_response(
    response::FieldState,
    basis::AbstractVector,
    model::ModelSettings,
)
    return [field_metric_dot(item.fields, response, model) for item in basis]
end

function _stage2_component_norms(fields::FieldState, model::ModelSettings)
    denominator = 2 * model.L
    return (;
        alpha=sqrt(sum(abs2, fields.alpha) / denominator),
        beta=sqrt(sum(abs2, fields.beta) / denominator),
        mu_cdw=sqrt(sum(abs2, fields.mu_cdw) / denominator),
        total=field_metric_norm(fields, model),
    )
end

function _stage2_map_responses(
    baseline::CorrelationState,
    measured::CorrelationState,
    bank,
    model::ModelSettings,
    settings::BareStage2Settings,
    amplitude::Real,
)
    responses = Dict{Symbol,NamedTuple}()
    for geometry in settings.geometries
        geometry_model = _model_with_geometry(
            model,
            geometry;
            ep_signed=bank.kernel_pair_binding,
        )
        baseline_map = mean_fields_from_correlations(baseline, geometry_model; threshold=0.0)
        measured_map = mean_fields_from_correlations(measured, geometry_model; threshold=0.0)
        response = scale_fields(subtract_fields(measured_map, baseline_map), inv(Float64(amplitude)))
        coordinates = _stage2_project_response(response, bank.basis, model)
        norms = _stage2_component_norms(response, model)
        projected_norm = norm(coordinates)
        leakage_relative = norms.total > eps(Float64) ?
            sqrt(max(norms.total^2 - projected_norm^2, 0.0)) / norms.total : 0.0
        responses[geometry] = (;
            baseline=baseline_map,
            measured=measured_map,
            response,
            coordinates,
            norms,
            projected_norm,
            leakage_relative,
        )
    end
    return responses
end

function _write_stage2_probe_result(
    destination::AbstractString,
    result,
    convergence,
    density::Real,
    direction,
    amplitude::Real,
    correlations::CorrelationState,
    raw_baseline,
    raw_measured,
    raw_response,
    map_responses,
    bank,
    config_path::AbstractString,
    normal_reference,
    pair_reference,
)
    output = abspath(destination)
    ispath(output) && throw(ArgumentError("refusing to overwrite immutable Stage 2 probe: $output"))
    mkpath(dirname(output))
    temporary = tempname(dirname(output))
    h5open(temporary, "w") do file
        file["schema_version"] = 1
        file["artifact_kind"] = "bare_ladder_stage2_probe"
        file["complete"] = true
        file["scientifically_accepted"] = convergence.accepted
        file["basis_index"] = direction.basis_index
        file["label"] = direction.label
        file["block"] = String(direction.block)
        file["channel"] = String(direction.channel)
        file["parity"] = String(direction.parity)
        file["field_strength"] = Float64(amplitude)
        file["density"] = Float64(density)
        file["raw_conjugate_baseline"] = raw_baseline
        file["raw_conjugate_measured"] = raw_measured
        file["raw_susceptibility_column"] = raw_response
        file["psi"] = move_to_cpu(result.psi)
        _write_fields(create_group(file, "input_direction"), direction.fields)
        _write_correlations(create_group(file, "measured_correlations"), correlations)
        _write_stage2_dmrg(create_group(file, "dmrg"), result, convergence)
        geometry_group = create_group(file, "geometry_maps")
        for geometry in sort(collect(keys(map_responses)); by=string)
            response = map_responses[geometry]
            child = create_group(geometry_group, String(geometry))
            _write_fields(create_group(child, "baseline"), response.baseline)
            _write_fields(create_group(child, "measured"), response.measured)
            _write_fields(create_group(child, "response"), response.response)
            child["projected_coordinates"] = response.coordinates
            child["response_norm"] = response.norms.total
            child["response_alpha_norm"] = response.norms.alpha
            child["response_beta_norm"] = response.norms.beta
            child["response_mu_cdw_norm"] = response.norms.mu_cdw
            child["projected_norm"] = response.projected_norm
            child["leakage_relative"] = response.leakage_relative
        end
        source = create_group(file, "sources")
        source["candidate_bank_path"] = bank.path
        source["candidate_bank_sha256"] = bank.sha256
        source["candidate_bank_fingerprint"] = bank.bank_fingerprint
        source["backbone_path"] = bank.backbone_path
        source["backbone_sha256"] = bank.backbone_sha256
        source["stage1_path"] = bank.stage1_path
        source["stage1_sha256"] = bank.stage1_sha256
        source["normal_reference_path"] = normal_reference.path
        source["normal_reference_sha256"] = normal_reference.sha256
        source["pair_reference_path"] = pair_reference === nothing ? "" : pair_reference.path
        source["pair_reference_sha256"] = pair_reference === nothing ? "" : pair_reference.sha256
        provenance = create_group(file, "provenance")
        provenance["config_path"] = abspath(config_path)
        provenance["config_sha256"] = sha256_file(abspath(config_path))
        provenance["implementation_sha256"] = implementation_fingerprint()
        provenance["git_commit"] = _read_git("rev-parse", "HEAD")
        provenance["julia_threads"] = Threads.nthreads()
    end
    mv(temporary, output)
    return (;
        path=output,
        sha256=sha256_file(output),
        accepted=convergence.accepted,
        density=Float64(density),
        energy=result.energy,
        last_five_energy_change=convergence.last_five_energy_change,
    )
end

function run_stage2_discovery_probe(
    candidate_bank_path::AbstractString,
    normal_reference_path::AbstractString,
    pair_reference_path::Union{Nothing,AbstractString},
    config_path::AbstractString,
    block::Symbol,
    local_index::Integer,
    output_path::AbstractString;
    amplitude::Union{Nothing,Real}=nothing,
)
    block in (:normal, :pair) || throw(ArgumentError("Stage 2 block must be normal or pair"))
    project = load_settings(config_path)
    project.runtime.backend == :cpu || throw(ArgumentError("Stage 2 requires runtime.backend=cpu"))
    stage2 = load_bare_stage2_settings(config_path, project.model)
    bank = read_stage2_candidate_bank(candidate_bank_path)
    bank.config_sha256 == sha256_file(abspath(config_path)) || throw(ArgumentError(
        "Stage 2 candidate bank configuration hash mismatch",
    ))
    indices = block == :normal ? bank.normal_indices : bank.pair_indices
    1 <= local_index <= length(indices) || throw(ArgumentError(
        "Stage 2 $block probe index must lie in 1:$(length(indices))",
    ))
    direction = bank.basis[indices[Int(local_index)]]
    direction.block == block || error("Stage 2 candidate bank block index is inconsistent")
    resolved_amplitude = amplitude === nothing ? stage2.field_strength : Float64(amplitude)
    resolved_amplitude > 0 || throw(ArgumentError("Stage 2 probe amplitude must be positive"))
    normal_reference = read_stage2_normal_reference(
        normal_reference_path,
        bank;
        load_state=block == :normal,
    )
    pair_reference = nothing
    reference = if block == :normal
        normal_reference
    else
        pair_reference_path === nothing && throw(ArgumentError(
            "pair probes require the parity-only zero-field reference",
        ))
        pair_reference = read_stage2_pair_reference(pair_reference_path, bank)
        pair_reference
    end
    if block == :normal
        field_metric_norm(FieldState(direction.fields.alpha, direction.fields.beta,
            zeros(size(direction.fields.mu_cdw))), project.model) <= 100eps(Float64) ||
            throw(ArgumentError("normal Stage 2 direction contains a pairing or exchange field"))
    else
        maximum(abs, direction.fields.mu_cdw) <= 100eps(Float64) || throw(ArgumentError(
            "pair Stage 2 direction contains a Hartree field",
        ))
    end
    applied = scale_fields(direction.fields, resolved_amplitude)
    sites = siteinds(reference.psi)
    hamiltonian = build_mf_mpo(
        sites,
        project.model,
        applied,
        bank.chemical_potential;
        backend=project.runtime,
    )
    dmrg_settings = _stage2_dmrg_settings(stage2, stage2.probe_sweeps)
    result = run_dmrg_ground(
        sites,
        hamiltonian,
        project.model.density,
        dmrg_settings;
        psi_init=reference.psi,
        rng=MersenneTwister(project.run.random_seed + 3000 + direction.basis_index),
        deadline=time() + stage2.maximum_job_seconds,
        backend=project.runtime,
        noise_schedule=_stage2_noise_schedule(stage2, stage2.probe_sweeps),
        minimum_convergence_sweep=stage2.minimum_convergence_sweep,
    )
    _, correlations = calculate_mean_fields(result.psi, project.model; threshold=0.0)
    density = (sum(correlations.density_down) + sum(correlations.density_up)) /
        (2 * project.model.L)
    convergence = _stage2_scientifically_converged(
        result,
        density,
        project.model,
        stage2,
    )
    raw_baseline = [
        field_conjugate_expectation(item.fields, reference.correlations, project.model)
        for item in bank.basis
    ]
    raw_measured = [
        field_conjugate_expectation(item.fields, correlations, project.model)
        for item in bank.basis
    ]
    raw_response = .-(raw_measured .- raw_baseline) ./
        (resolved_amplitude * 2 * project.model.L)
    map_responses = _stage2_map_responses(
        reference.correlations,
        correlations,
        bank,
        project.model,
        stage2,
        resolved_amplitude,
    )
    return _write_stage2_probe_result(
        output_path,
        result,
        convergence,
        density,
        direction,
        resolved_amplitude,
        correlations,
        raw_baseline,
        raw_measured,
        raw_response,
        map_responses,
        bank,
        config_path,
        normal_reference,
        pair_reference,
    )
end

function _read_stage2_probe_column(path::AbstractString, bank, settings::BareStage2Settings)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("Stage 2 probe artifact not found: $source"))
    return h5open(source, "r") do file
        String(read(file, "artifact_kind")) == "bare_ladder_stage2_probe" ||
            throw(ArgumentError("not a Stage 2 probe artifact: $source"))
        Bool(read(file, "complete")) || throw(ArgumentError("incomplete Stage 2 probe: $source"))
        String(read(file, "sources/candidate_bank_sha256")) == bank.sha256 ||
            throw(ArgumentError("Stage 2 probe candidate-bank hash mismatch: $source"))
        amplitude = Float64(read(file, "field_strength"))
        isapprox(amplitude, settings.field_strength; rtol=0, atol=10eps(settings.field_strength)) ||
            throw(ArgumentError("Stage 2 discovery probe has the wrong field strength: $source"))
        geometry_coordinates = Dict(
            geometry => Float64.(read(file, "geometry_maps/$(String(geometry))/projected_coordinates"))
            for geometry in settings.geometries
        )
        geometry_leakage = Dict(
            geometry => Float64(read(file, "geometry_maps/$(String(geometry))/leakage_relative"))
            for geometry in settings.geometries
        )
        geometry_beta_fraction = Dict(
            geometry => let
                total = Float64(read(file, "geometry_maps/$(String(geometry))/response_norm"))
                beta = Float64(read(file, "geometry_maps/$(String(geometry))/response_beta_norm"))
                total > eps(Float64) ? beta / total : 0.0
            end
            for geometry in settings.geometries
        )
        (;
            path=source,
            sha256=sha256_file(source),
            basis_index=Int(read(file, "basis_index")),
            label=String(read(file, "label")),
            block=Symbol(String(read(file, "block"))),
            accepted=Bool(read(file, "scientifically_accepted")),
            amplitude,
            density=Float64(read(file, "density")),
            energy=Float64(read(file, "dmrg/energy")),
            last_five_energy_change=Float64(read(file, "dmrg/last_five_energy_change")),
            max_discarded_weight=Float64(read(file, "dmrg/max_discarded_weight")),
            maximum_link_dimension=Int(read(file, "dmrg/maximum_link_dimension")),
            raw_column=Float64.(read(file, "raw_susceptibility_column")),
            geometry_coordinates,
            geometry_leakage,
            geometry_beta_fraction,
            normal_reference_path=String(read(file, "sources/normal_reference_path")),
            normal_reference_sha256=String(read(file, "sources/normal_reference_sha256")),
            pair_reference_path=String(read(file, "sources/pair_reference_path")),
            pair_reference_sha256=String(read(file, "sources/pair_reference_sha256")),
        )
    end
end

function _relative_antisymmetry(matrix::AbstractMatrix)
    denominator = max(norm(matrix), eps(Float64))
    return norm(matrix - transpose(matrix)) / denominator
end

function _stage2_block_eigensystem(matrix::AbstractMatrix, indices::AbstractVector{<:Integer})
    block = Matrix{Float64}(matrix[indices, indices])
    decomposition = eigen(block)
    order = sortperm(abs.(decomposition.values); rev=true)
    values = ComplexF64.(decomposition.values[order])
    vectors = ComplexF64.(decomposition.vectors[:, order])
    residuals = [
        norm(block * vectors[:, index] - values[index] * vectors[:, index])
        for index in eachindex(values)
    ]
    return (; matrix=block, eigenvalues=values, eigenvectors=vectors, residuals)
end

function _real_eigenvector(vector::AbstractVector{<:Complex}, value::Complex)
    abs(imag(value)) <= 1e-8 * max(1.0, abs(value)) || return nothing
    pivot = argmax(abs.(vector))
    phase = abs(vector[pivot]) > eps(Float64) ? exp(-im * angle(vector[pivot])) : 1.0 + 0im
    rotated = vector .* phase
    norm(imag.(rotated)) <= 1e-8 * max(norm(real.(rotated)), eps(Float64)) || return nothing
    values = real.(rotated)
    magnitude = norm(values)
    magnitude > eps(Float64) || return nothing
    values ./= magnitude
    pivot = argmax(abs.(values))
    values[pivot] < 0 && (values .*= -1)
    return values
end

function _stage2_validation_directions(
    eigensystems,
    bank,
    settings::BareStage2Settings,
)
    pool = NamedTuple[]
    for geometry in settings.geometries, block in (:normal, :pair)
        spectrum = eigensystems[(geometry, block)]
        indices = block == :normal ? bank.normal_indices : bank.pair_indices
        for rank in eachindex(spectrum.eigenvalues)
            local_vector = _real_eigenvector(
                spectrum.eigenvectors[:, rank],
                spectrum.eigenvalues[rank],
            )
            local_vector === nothing && continue
            coefficients = zeros(Float64, length(bank.basis))
            coefficients[indices] .= local_vector
            push!(pool, (;
                geometry,
                block,
                rank,
                eigenvalue=real(spectrum.eigenvalues[rank]),
                score=abs(spectrum.eigenvalues[rank]),
                coefficients,
            ))
        end
    end
    sort!(pool; by=item -> item.score, rev=true)
    selected = NamedTuple[]
    for required_block in (:normal, :pair)
        index = findfirst(item -> item.block == required_block, pool)
        index === nothing && throw(ArgumentError(
            "no real Stage 2 $required_block eigenmode is available for validation",
        ))
        push!(selected, pool[index])
    end
    for item in pool
        length(selected) >= settings.top_validation_modes && break
        duplicate = any(
            prior -> prior.block == item.block &&
                abs(dot(prior.coefficients, item.coefficients)) >= 0.98,
            selected,
        )
        duplicate || push!(selected, item)
    end
    length(selected) >= settings.top_validation_modes || throw(ArgumentError(
        "only $(length(selected)) independent real response modes were available for validation",
    ))
    selected = selected[1:settings.top_validation_modes]
    return [
        let
            template = first(bank.basis).fields
            fields = FieldState(
                zeros(Float64, size(template.alpha)),
                zeros(Float64, size(template.beta)),
                zeros(Float64, size(template.mu_cdw)),
            )
            for (coefficient, basis_item) in zip(item.coefficients, bank.basis)
                fields = _field_axpy(fields, coefficient, basis_item.fields)
            end
            merge(item, (;
                validation_index=index,
                fields,
                label="$(String(item.geometry))_$(String(item.block))_mode$(item.rank)",
            ))
        end
        for (index, item) in enumerate(selected)
    ]
end

function _write_stage2_eigensystem(group, spectrum, model::ModelSettings)
    group["matrix"] = spectrum.matrix
    group["eigenvalue_real"] = real.(spectrum.eigenvalues)
    group["eigenvalue_imag"] = imag.(spectrum.eigenvalues)
    group["eigenvector_real"] = real.(spectrum.eigenvectors)
    group["eigenvector_imag"] = imag.(spectrum.eigenvectors)
    group["residual_norm"] = spectrum.residuals
    group["absolute_eigenvalue"] = abs.(spectrum.eigenvalues)
    group["critical_tp"] = [
        abs(value) > eps(Float64) ? model.tp / sqrt(abs(value)) : Inf
        for value in spectrum.eigenvalues
    ]
    return group
end

function assemble_stage2_discovery(
    candidate_bank_path::AbstractString,
    probe_paths::AbstractVector{<:AbstractString},
    config_path::AbstractString,
    output_path::AbstractString,
    summary_path::AbstractString,
)
    destination = abspath(output_path)
    summary_destination = abspath(summary_path)
    gate_destination = splitext(destination)[1] * "_gates.tsv"
    validation_path = splitext(destination)[1] * "_validation_plan.tsv"
    any(ispath, (destination, summary_destination, gate_destination, validation_path)) &&
        throw(ArgumentError(
        "refusing to overwrite immutable Stage 2 discovery output",
    ))
    project = load_settings(config_path)
    stage2 = load_bare_stage2_settings(config_path, project.model)
    bank = read_stage2_candidate_bank(candidate_bank_path)
    bank.config_sha256 == sha256_file(abspath(config_path)) || throw(ArgumentError(
        "Stage 2 candidate bank configuration hash mismatch",
    ))
    length(probe_paths) == length(bank.basis) || throw(ArgumentError(
        "expected $(length(bank.basis)) Stage 2 probes, got $(length(probe_paths))",
    ))
    probes = [_read_stage2_probe_column(path, bank, stage2) for path in probe_paths]
    indices = [probe.basis_index for probe in probes]
    sort(indices) == collect(1:length(bank.basis)) || throw(ArgumentError(
        "Stage 2 probe artifacts do not cover each basis direction exactly once",
    ))
    sort!(probes; by=probe -> probe.basis_index)
    length(unique(probe.normal_reference_sha256 for probe in probes)) == 1 ||
        throw(ArgumentError("Stage 2 probes used different normal zero-field references"))
    normal_reference_sha256 = first(probes).normal_reference_sha256
    pair_probe_records = filter(probe -> probe.block == :pair, probes)
    length(unique(probe.pair_reference_sha256 for probe in pair_probe_records)) == 1 ||
        throw(ArgumentError("Stage 2 pair probes used different parity-only references"))
    pair_reference_sha256 = first(pair_probe_records).pair_reference_sha256
    count = length(probes)
    raw_susceptibility = zeros(Float64, count, count)
    maps = Dict(geometry => zeros(Float64, count, count) for geometry in stage2.geometries)
    leakage = Dict(geometry => zeros(Float64, count) for geometry in stage2.geometries)
    beta_fraction = Dict(geometry => zeros(Float64, count) for geometry in stage2.geometries)
    for probe in probes
        raw_susceptibility[:, probe.basis_index] .= probe.raw_column
        for geometry in stage2.geometries
            maps[geometry][:, probe.basis_index] .= probe.geometry_coordinates[geometry]
            leakage[geometry][probe.basis_index] = probe.geometry_leakage[geometry]
            beta_fraction[geometry][probe.basis_index] = probe.geometry_beta_fraction[geometry]
        end
    end
    normal_reciprocity_relative_error = _relative_antisymmetry(
        raw_susceptibility[bank.normal_indices, bank.normal_indices],
    )
    pair_reciprocity_relative_error = _relative_antisymmetry(
        raw_susceptibility[bank.pair_indices, bank.pair_indices],
    )
    reciprocity_relative_error = max(
        normal_reciprocity_relative_error,
        pair_reciprocity_relative_error,
    )
    reciprocity_pass = reciprocity_relative_error <= stage2.reciprocity_relative_tol
    raw_cross_block_relative_norm = (
        norm(raw_susceptibility[bank.pair_indices, bank.normal_indices]) +
        norm(raw_susceptibility[bank.normal_indices, bank.pair_indices])
    ) / max(
        norm(raw_susceptibility[bank.normal_indices, bank.normal_indices]) +
        norm(raw_susceptibility[bank.pair_indices, bank.pair_indices]),
        eps(Float64),
    )
    cross_block_pass = raw_cross_block_relative_norm <= stage2.cross_block_relative_tol
    all_probes_accepted = all(probe.accepted for probe in probes)
    maximum_projected_leakage_relative = maximum(
        maximum(leakage[geometry]) for geometry in stage2.geometries
    )
    maximum_beta_fraction = maximum(
        maximum(beta_fraction[geometry]) for geometry in stage2.geometries
    )
    eigensystems = Dict{Tuple{Symbol,Symbol},NamedTuple}()
    for geometry in stage2.geometries
        eigensystems[(geometry, :normal)] = _stage2_block_eigensystem(
            maps[geometry],
            bank.normal_indices,
        )
        eigensystems[(geometry, :pair)] = _stage2_block_eigensystem(
            maps[geometry],
            bank.pair_indices,
        )
    end
    validation = _stage2_validation_directions(eigensystems, bank, stage2)
    scientific_acceptance = all_probes_accepted && reciprocity_pass && cross_block_pass
    mkpath(dirname(destination))
    temporary = tempname(dirname(destination))
    h5open(temporary, "w") do file
        file["schema_version"] = 1
        file["artifact_kind"] = "bare_ladder_stage2_discovery"
        file["complete"] = true
        file["scientifically_accepted"] = scientific_acceptance
        file["all_probes_accepted"] = all_probes_accepted
        file["reciprocity_pass"] = reciprocity_pass
        file["reciprocity_relative_error"] = reciprocity_relative_error
        file["normal_reciprocity_relative_error"] = normal_reciprocity_relative_error
        file["pair_reciprocity_relative_error"] = pair_reciprocity_relative_error
        file["reciprocity_relative_tol"] = stage2.reciprocity_relative_tol
        file["cross_block_pass"] = cross_block_pass
        file["raw_cross_block_relative_norm"] = raw_cross_block_relative_norm
        file["cross_block_relative_tol"] = stage2.cross_block_relative_tol
        file["field_strength"] = stage2.field_strength
        file["raw_susceptibility"] = raw_susceptibility
        if reciprocity_pass
            file["raw_susceptibility_symmetric"] =
                (raw_susceptibility + transpose(raw_susceptibility)) / 2
        end
        file["maximum_projected_leakage_relative"] = maximum_projected_leakage_relative
        file["maximum_beta_fraction"] = maximum_beta_fraction
        file["kernel_pair_binding"] = bank.kernel_pair_binding
        file["kernel_pair_binding_source"] = bank.kernel_pair_binding_source
        file["kernel_prefactor"] = bank.kernel_prefactor
        file["normal_basis_indices"] = bank.normal_indices
        file["pair_basis_indices"] = bank.pair_indices
        _write_backbone_model(create_group(file, "model"), project.model)
        basis_group = create_group(file, "basis")
        for item in bank.basis
            _write_stage2_item(
                create_group(basis_group, lpad(string(item.basis_index), 3, '0')),
                item,
            )
        end
        probes_group = create_group(file, "probes")
        for probe in probes
            child = create_group(probes_group, lpad(string(probe.basis_index), 3, '0'))
            child["path"] = probe.path
            child["sha256"] = probe.sha256
            child["label"] = probe.label
            child["block"] = String(probe.block)
            child["accepted"] = probe.accepted
            child["density"] = probe.density
            child["energy"] = probe.energy
            child["last_five_energy_change"] = probe.last_five_energy_change
            child["max_discarded_weight"] = probe.max_discarded_weight
            child["maximum_link_dimension"] = probe.maximum_link_dimension
        end
        geometry_group = create_group(file, "geometry_maps")
        for geometry in stage2.geometries
            child = create_group(geometry_group, String(geometry))
            child["projected_jacobian"] = maps[geometry]
            child["column_leakage_relative"] = leakage[geometry]
            child["column_beta_fraction"] = beta_fraction[geometry]
            normal_to_pair = maps[geometry][bank.pair_indices, bank.normal_indices]
            pair_to_normal = maps[geometry][bank.normal_indices, bank.pair_indices]
            child["normal_pair_cross_relative_norm"] = (
                norm(normal_to_pair) + norm(pair_to_normal)
            ) / max(norm(maps[geometry]), eps(Float64))
            _write_stage2_eigensystem(
                create_group(child, "normal"),
                eigensystems[(geometry, :normal)],
                project.model,
            )
            _write_stage2_eigensystem(
                create_group(child, "pair"),
                eigensystems[(geometry, :pair)],
                project.model,
            )
        end
        validation_group = create_group(file, "validation_directions")
        for item in validation
            child = create_group(
                validation_group,
                lpad(string(item.validation_index), 3, '0'),
            )
            child["label"] = item.label
            child["geometry"] = String(item.geometry)
            child["block"] = String(item.block)
            child["source_eigen_rank"] = item.rank
            child["source_eigenvalue"] = item.eigenvalue
            child["basis_coefficients"] = item.coefficients
            _write_fields(create_group(child, "fields"), item.fields)
        end
        source = create_group(file, "sources")
        source["candidate_bank_path"] = bank.path
        source["candidate_bank_sha256"] = bank.sha256
        source["candidate_bank_fingerprint"] = bank.bank_fingerprint
        source["backbone_path"] = bank.backbone_path
        source["backbone_sha256"] = bank.backbone_sha256
        source["stage1_path"] = bank.stage1_path
        source["stage1_sha256"] = bank.stage1_sha256
        source["normal_reference_path"] = first(probes).normal_reference_path
        source["normal_reference_sha256"] = normal_reference_sha256
        source["pair_reference_path"] = first(pair_probe_records).pair_reference_path
        source["pair_reference_sha256"] = pair_reference_sha256
        provenance = create_group(file, "provenance")
        provenance["config_path"] = abspath(config_path)
        provenance["config_sha256"] = sha256_file(abspath(config_path))
        provenance["implementation_sha256"] = implementation_fingerprint()
        provenance["git_commit"] = _read_git("rev-parse", "HEAD")
        provenance["julia_threads"] = Threads.nthreads()
    end
    mv(temporary, destination)

    mkpath(dirname(summary_destination))
    open(summary_destination, "w") do io
        println(io, "geometry\tblock\trank\teigenvalue_real\teigenvalue_imag\tabs_eigenvalue\tcritical_tp\tresidual_norm")
        for geometry in stage2.geometries, block in (:normal, :pair)
            spectrum = eigensystems[(geometry, block)]
            for rank in eachindex(spectrum.eigenvalues)
                value = spectrum.eigenvalues[rank]
                critical_tp = abs(value) > eps(Float64) ?
                    project.model.tp / sqrt(abs(value)) : Inf
                @printf(
                    io,
                    "%s\t%s\t%d\t%.16g\t%.16g\t%.16g\t%.16g\t%.16g\n",
                    String(geometry),
                    String(block),
                    rank,
                    real(value),
                    imag(value),
                    abs(value),
                    critical_tp,
                    spectrum.residuals[rank],
                )
            end
        end
    end
    open(validation_path, "w") do io
        println(io, "validation_index\tlabel\tgeometry\tblock\tsource_rank\tsource_eigenvalue")
        for item in validation
            @printf(
                io,
                "%d\t%s\t%s\t%s\t%d\t%.16g\n",
                item.validation_index,
                item.label,
                String(item.geometry),
                String(item.block),
                item.rank,
                item.eigenvalue,
            )
        end
    end
    open(gate_destination, "w") do io
        println(io, "scientifically_accepted\tall_probes_accepted\treciprocity_pass\tnormal_reciprocity_relative_error\tpair_reciprocity_relative_error\traw_cross_block_relative_norm\tmaximum_projected_leakage_relative\tmaximum_beta_fraction\tkernel_pair_binding\tkernel_prefactor")
        @printf(
            io,
            "%s\t%s\t%s\t%.16g\t%.16g\t%.16g\t%.16g\t%.16g\t%.16g\t%.16g\n",
            scientific_acceptance,
            all_probes_accepted,
            reciprocity_pass,
            normal_reciprocity_relative_error,
            pair_reciprocity_relative_error,
            raw_cross_block_relative_norm,
            maximum_projected_leakage_relative,
            maximum_beta_fraction,
            bank.kernel_pair_binding,
            bank.kernel_prefactor,
        )
    end
    return (;
        path=destination,
        sha256=sha256_file(destination),
        summary_path=summary_destination,
        gate_path=gate_destination,
        validation_path,
        scientifically_accepted=scientific_acceptance,
        all_probes_accepted,
        reciprocity_relative_error,
        raw_cross_block_relative_norm,
        maximum_projected_leakage_relative,
        maximum_beta_fraction,
        validation_count=length(validation),
    )
end

function read_stage2_validation_direction(
    discovery_path::AbstractString,
    validation_index::Integer,
    bank,
)
    source = abspath(discovery_path)
    isfile(source) || throw(ArgumentError("Stage 2 discovery artifact not found: $source"))
    return h5open(source, "r") do file
        String(read(file, "artifact_kind")) == "bare_ladder_stage2_discovery" ||
            throw(ArgumentError("not a Stage 2 discovery artifact: $source"))
        Bool(read(file, "complete")) || throw(ArgumentError("incomplete Stage 2 discovery artifact"))
        Bool(read(file, "scientifically_accepted")) || throw(ArgumentError(
            "Stage 2 discovery failed its scientific gates; validation is not authorized",
        ))
        String(read(file, "sources/candidate_bank_sha256")) == bank.sha256 ||
            throw(ArgumentError("Stage 2 discovery candidate-bank hash mismatch"))
        key = lpad(string(Int(validation_index)), 3, '0')
        haskey(file, "validation_directions/$key") || throw(ArgumentError(
            "Stage 2 validation direction $validation_index is unavailable",
        ))
        group = file["validation_directions/$key"]
        coefficients = Float64.(read(group, "basis_coefficients"))
        length(coefficients) == length(bank.basis) || throw(ArgumentError(
            "Stage 2 validation coefficient dimension mismatch",
        ))
        (;
            validation_index=Int(validation_index),
            label=String(read(group, "label")),
            geometry=Symbol(String(read(group, "geometry"))),
            block=Symbol(String(read(group, "block"))),
            source_eigen_rank=Int(read(group, "source_eigen_rank")),
            source_eigenvalue=Float64(read(group, "source_eigenvalue")),
            coefficients,
            fields=_read_fields(group["fields"]),
            discovery_path=source,
            discovery_sha256=sha256_file(source),
        )
    end
end

function _validation_reference(
    block::Symbol,
    bank,
    normal_reference_path::AbstractString,
    pair_reference_path,
)
    if block == :normal
        return read_stage2_normal_reference(normal_reference_path, bank)
    end
    pair_reference_path === nothing && throw(ArgumentError(
        "pair validation requires the parity-only zero-field reference",
    ))
    return _read_stage2_pair_reference_unhashed(abspath(pair_reference_path), bank)
end

function _write_stage2_validation_amplitude(
    temporary::AbstractString,
    group_name::AbstractString,
    project::ProjectSettings,
    stage2::BareStage2Settings,
    bank,
    direction,
    normal_reference_path::AbstractString,
    pair_reference_path,
    amplitude::Real,
)
    reference = _validation_reference(
        direction.block,
        bank,
        normal_reference_path,
        pair_reference_path,
    )
    sites = siteinds(reference.psi)
    applied = scale_fields(direction.fields, amplitude)
    hamiltonian = build_mf_mpo(
        sites,
        project.model,
        applied,
        bank.chemical_potential;
        backend=project.runtime,
    )
    dmrg_settings = _stage2_dmrg_settings(stage2, stage2.validation_sweeps)
    result = run_dmrg_ground(
        sites,
        hamiltonian,
        project.model.density,
        dmrg_settings;
        psi_init=reference.psi,
        rng=MersenneTwister(project.run.random_seed + 4000 + direction.validation_index),
        deadline=time() + stage2.maximum_job_seconds,
        backend=project.runtime,
        noise_schedule=_stage2_noise_schedule(stage2, stage2.validation_sweeps),
        minimum_convergence_sweep=stage2.minimum_convergence_sweep,
    )
    _, correlations = calculate_mean_fields(result.psi, project.model; threshold=0.0)
    density = (sum(correlations.density_down) + sum(correlations.density_up)) /
        (2 * project.model.L)
    convergence = _stage2_scientifically_converged(
        result,
        density,
        project.model,
        stage2,
    )
    raw_baseline = [
        field_conjugate_expectation(item.fields, reference.correlations, project.model)
        for item in bank.basis
    ]
    raw_measured = [
        field_conjugate_expectation(item.fields, correlations, project.model)
        for item in bank.basis
    ]
    raw_response = .-(raw_measured .- raw_baseline) ./
        (Float64(amplitude) * 2 * project.model.L)
    map_responses = _stage2_map_responses(
        reference.correlations,
        correlations,
        bank,
        project.model,
        stage2,
        amplitude,
    )
    h5open(temporary, "r+") do file
        group = create_group(file, group_name)
        group["field_strength"] = Float64(amplitude)
        group["scientifically_accepted"] = convergence.accepted
        group["density"] = density
        group["raw_susceptibility_action"] = raw_response
        group["psi"] = move_to_cpu(result.psi)
        _write_correlations(create_group(group, "measured_correlations"), correlations)
        _write_stage2_dmrg(create_group(group, "dmrg"), result, convergence)
        geometry_group = create_group(group, "geometry_maps")
        for geometry in stage2.geometries
            response = map_responses[geometry]
            child = create_group(geometry_group, String(geometry))
            _write_fields(create_group(child, "response"), response.response)
            child["projected_coordinates"] = response.coordinates
            child["response_norm"] = response.norms.total
            child["response_beta_norm"] = response.norms.beta
            child["projected_norm"] = response.projected_norm
            child["leakage_relative"] = response.leakage_relative
        end
    end
    coordinates = Dict(
        geometry => map_responses[geometry].coordinates
        for geometry in stage2.geometries
    )
    return (;
        accepted=convergence.accepted,
        density,
        energy=result.energy,
        last_five_energy_change=convergence.last_five_energy_change,
        coordinates,
    )
end

function run_stage2_validation_probe(
    discovery_path::AbstractString,
    candidate_bank_path::AbstractString,
    normal_reference_path::AbstractString,
    pair_reference_path::Union{Nothing,AbstractString},
    config_path::AbstractString,
    validation_index::Integer,
    output_path::AbstractString,
)
    destination = abspath(output_path)
    ispath(destination) && throw(ArgumentError(
        "refusing to overwrite immutable Stage 2 validation probe: $destination",
    ))
    project = load_settings(config_path)
    project.runtime.backend == :cpu || throw(ArgumentError("Stage 2 requires runtime.backend=cpu"))
    stage2 = load_bare_stage2_settings(config_path, project.model)
    bank = read_stage2_candidate_bank(candidate_bank_path)
    direction = read_stage2_validation_direction(discovery_path, validation_index, bank)
    pair_reference_sha256 = if direction.block == :pair
        pair_reference_path === nothing && throw(ArgumentError(
            "pair validation requires a pairing reference path",
        ))
        _stage2_artifact_sha256(abspath(pair_reference_path))
    else
        ""
    end
    mkpath(dirname(destination))
    temporary = tempname(dirname(destination))
    h5open(temporary, "w") do file
        file["schema_version"] = 1
        file["artifact_kind"] = "bare_ladder_stage2_validation_probe"
        file["validation_index"] = direction.validation_index
        file["label"] = direction.label
        file["geometry"] = String(direction.geometry)
        file["block"] = String(direction.block)
        file["source_eigen_rank"] = direction.source_eigen_rank
        file["source_eigenvalue"] = direction.source_eigenvalue
        file["basis_coefficients"] = direction.coefficients
        _write_fields(create_group(file, "input_direction"), direction.fields)
        source = create_group(file, "sources")
        source["discovery_path"] = direction.discovery_path
        source["discovery_sha256"] = direction.discovery_sha256
        source["candidate_bank_path"] = bank.path
        source["candidate_bank_sha256"] = bank.sha256
        source["normal_reference_path"] = abspath(normal_reference_path)
        source["normal_reference_sha256"] = _stage2_artifact_sha256(normal_reference_path)
        source["pair_reference_path"] = pair_reference_path === nothing ? "" : abspath(pair_reference_path)
        source["pair_reference_sha256"] = pair_reference_sha256
        provenance = create_group(file, "provenance")
        provenance["config_path"] = abspath(config_path)
        provenance["config_sha256"] = sha256_file(abspath(config_path))
        provenance["implementation_sha256"] = implementation_fingerprint()
        provenance["git_commit"] = _read_git("rev-parse", "HEAD")
        provenance["julia_threads"] = Threads.nthreads()
    end
    full = _write_stage2_validation_amplitude(
        temporary,
        "full_amplitude",
        project,
        stage2,
        bank,
        direction,
        normal_reference_path,
        pair_reference_path,
        stage2.field_strength,
    )
    GC.gc()
    half = _write_stage2_validation_amplitude(
        temporary,
        "half_amplitude",
        project,
        stage2,
        bank,
        direction,
        normal_reference_path,
        pair_reference_path,
        stage2.validation_field_strength,
    )
    primary_full = full.coordinates[direction.geometry]
    primary_half = half.coordinates[direction.geometry]
    linearity_relative_error = norm(primary_full - primary_half) /
        max(norm(primary_half), eps(Float64))
    linearity_pass = linearity_relative_error <= stage2.linearity_relative_tol
    h5open(temporary, "r+") do file
        comparison = create_group(file, "comparison")
        comparison["linearity_relative_error"] = linearity_relative_error
        comparison["linearity_relative_tol"] = stage2.linearity_relative_tol
        comparison["linearity_pass"] = linearity_pass
        geometry_group = create_group(comparison, "geometry_maps")
        for geometry in stage2.geometries
            full_coordinates = full.coordinates[geometry]
            half_coordinates = half.coordinates[geometry]
            richardson = 2 .* half_coordinates .- full_coordinates
            child = create_group(geometry_group, String(geometry))
            child["full_coordinates"] = full_coordinates
            child["half_coordinates"] = half_coordinates
            child["richardson_coordinates"] = richardson
            child["linearity_relative_error"] = norm(full_coordinates - half_coordinates) /
                max(norm(half_coordinates), eps(Float64))
            child["rayleigh_full"] = dot(direction.coefficients, full_coordinates)
            child["rayleigh_half"] = dot(direction.coefficients, half_coordinates)
            child["rayleigh_richardson"] = dot(direction.coefficients, richardson)
        end
        file["scientifically_accepted"] = full.accepted && half.accepted && linearity_pass
        file["complete"] = true
    end
    mv(temporary, destination)
    accepted = full.accepted && half.accepted && linearity_pass
    return (;
        path=destination,
        sha256=sha256_file(destination),
        accepted,
        label=direction.label,
        geometry=direction.geometry,
        block=direction.block,
        linearity_relative_error,
        full_energy=full.energy,
        half_energy=half.energy,
    )
end

function assemble_stage2_validation(
    discovery_path::AbstractString,
    validation_paths::AbstractVector{<:AbstractString},
    config_path::AbstractString,
    output_path::AbstractString,
    summary_path::AbstractString,
)
    destination = abspath(output_path)
    summary_destination = abspath(summary_path)
    (ispath(destination) || ispath(summary_destination)) && throw(ArgumentError(
        "refusing to overwrite immutable Stage 2 validation output",
    ))
    project = load_settings(config_path)
    stage2 = load_bare_stage2_settings(config_path, project.model)
    discovery_source = abspath(discovery_path)
    discovery_sha256 = sha256_file(discovery_source)
    records = NamedTuple[]
    for path in validation_paths
        source = abspath(path)
        isfile(source) || throw(ArgumentError("validation artifact not found: $source"))
        record = h5open(source, "r") do file
            String(read(file, "artifact_kind")) == "bare_ladder_stage2_validation_probe" ||
                throw(ArgumentError("not a Stage 2 validation probe: $source"))
            Bool(read(file, "complete")) || throw(ArgumentError("incomplete validation probe: $source"))
            String(read(file, "sources/discovery_sha256")) == discovery_sha256 ||
                throw(ArgumentError("validation discovery hash mismatch: $source"))
            geometry = Symbol(String(read(file, "geometry")))
            comparison = "comparison/geometry_maps/$(String(geometry))"
            (;
                path=source,
                sha256=sha256_file(source),
                validation_index=Int(read(file, "validation_index")),
                label=String(read(file, "label")),
                geometry,
                block=Symbol(String(read(file, "block"))),
                source_eigenvalue=Float64(read(file, "source_eigenvalue")),
                accepted=Bool(read(file, "scientifically_accepted")),
                linearity_relative_error=Float64(read(file, "comparison/linearity_relative_error")),
                rayleigh_full=Float64(read(file, "$comparison/rayleigh_full")),
                rayleigh_half=Float64(read(file, "$comparison/rayleigh_half")),
                rayleigh_richardson=Float64(read(file, "$comparison/rayleigh_richardson")),
            )
        end
        push!(records, record)
    end
    sort!(records; by=record -> record.validation_index)
    [record.validation_index for record in records] == collect(1:stage2.top_validation_modes) ||
        throw(ArgumentError("validation artifacts do not cover every selected mode exactly once"))
    all_accepted = all(record.accepted for record in records)
    mkpath(dirname(destination))
    temporary = tempname(dirname(destination))
    h5open(temporary, "w") do file
        file["schema_version"] = 1
        file["artifact_kind"] = "bare_ladder_stage2_validation"
        file["complete"] = true
        file["scientifically_accepted"] = all_accepted
        file["validation_count"] = length(records)
        file["linearity_relative_tol"] = stage2.linearity_relative_tol
        source = create_group(file, "sources")
        source["discovery_path"] = discovery_source
        source["discovery_sha256"] = discovery_sha256
        record_group = create_group(file, "validation")
        for record in records
            child = create_group(record_group, lpad(string(record.validation_index), 3, '0'))
            child["path"] = record.path
            child["sha256"] = record.sha256
            child["label"] = record.label
            child["geometry"] = String(record.geometry)
            child["block"] = String(record.block)
            child["source_eigenvalue"] = record.source_eigenvalue
            child["accepted"] = record.accepted
            child["linearity_relative_error"] = record.linearity_relative_error
            child["rayleigh_full"] = record.rayleigh_full
            child["rayleigh_half"] = record.rayleigh_half
            child["rayleigh_richardson"] = record.rayleigh_richardson
        end
        provenance = create_group(file, "provenance")
        provenance["config_path"] = abspath(config_path)
        provenance["config_sha256"] = sha256_file(abspath(config_path))
        provenance["implementation_sha256"] = implementation_fingerprint()
        provenance["git_commit"] = _read_git("rev-parse", "HEAD")
    end
    mv(temporary, destination)
    open(summary_destination, "w") do io
        println(io, "validation_index\tlabel\tgeometry\tblock\tsource_eigenvalue\trayleigh_h\trayleigh_h_over_2\trayleigh_richardson\tlinearity_relative_error\taccepted")
        for record in records
            @printf(
                io,
                "%d\t%s\t%s\t%s\t%.16g\t%.16g\t%.16g\t%.16g\t%.16g\t%s\n",
                record.validation_index,
                record.label,
                String(record.geometry),
                String(record.block),
                record.source_eigenvalue,
                record.rayleigh_full,
                record.rayleigh_half,
                record.rayleigh_richardson,
                record.linearity_relative_error,
                record.accepted,
            )
        end
    end
    return (;
        path=destination,
        sha256=sha256_file(destination),
        summary_path=summary_destination,
        scientifically_accepted=all_accepted,
        validation_count=length(records),
    )
end
