const BACKBONE_SECTOR_OFFSETS = (
    (0, 0),
    (0, 2),
    (-1, 1),
    (-2, 0),
    (1, 1),
    (2, 0),
)

function load_ladder_backbone_settings(path::AbstractString)
    raw = TOML.parsefile(abspath(path))
    table = _table(raw, "backbone")
    settings = LadderBackboneSettings(;
        chi_ladder=Int.(_value(table, "chi_ladder", [400, 800, 1200])),
        pre_relax_maxdim=Int(_value(table, "pre_relax_maxdim", 200)),
        pre_relax_sweeps=Int(_value(table, "pre_relax_sweeps", 15)),
        stage_sweeps=Int(_value(table, "stage_sweeps", 40)),
        minimum_sweeps_at_maxdim=Int(_value(table, "minimum_sweeps_at_maxdim", 4)),
        cutoff=Float64(_value(table, "cutoff", 1e-10)),
        energy_tol=Float64(_value(table, "energy_tol", 1e-7)),
        noise_floor=Float64(_value(table, "noise_floor", 1e-8)),
        eigsolve_krylovdim=Int(_value(table, "eigsolve_krylovdim", 8)),
        output_level=Int(_value(table, "output_level", 1)),
        maximum_sector_seconds=Float64(_value(table, "maximum_sector_seconds", 23.5 * 3600)),
        last_five_energy_tol=Float64(_value(table, "last_five_energy_tol", 1e-4)),
        random_seed=Int(_value(table, "random_seed", 20260901)),
    )
    isempty(settings.chi_ladder) && throw(ArgumentError("backbone chi_ladder cannot be empty"))
    issorted(settings.chi_ladder) || throw(ArgumentError("backbone chi_ladder must be sorted"))
    all(>(0), settings.chi_ladder) || throw(ArgumentError("backbone chi values must be positive"))
    settings.pre_relax_maxdim > 0 || throw(ArgumentError("pre_relax_maxdim must be positive"))
    settings.pre_relax_sweeps >= 4 || throw(ArgumentError("pre_relax_sweeps must be at least four"))
    settings.stage_sweeps >= settings.minimum_sweeps_at_maxdim || throw(ArgumentError(
        "stage_sweeps must be at least minimum_sweeps_at_maxdim",
    ))
    settings.minimum_sweeps_at_maxdim >= 2 || throw(ArgumentError(
        "minimum_sweeps_at_maxdim must be at least two",
    ))
    settings.cutoff > 0 || throw(ArgumentError("backbone cutoff must be positive"))
    settings.energy_tol >= 0 || throw(ArgumentError("backbone energy_tol must be nonnegative"))
    settings.noise_floor > 0 || throw(ArgumentError("backbone noise_floor must be positive"))
    settings.last_five_energy_tol > 0 || throw(ArgumentError(
        "last_five_energy_tol must be positive",
    ))
    return settings
end

function load_bare_stage1_settings(path::AbstractString)
    raw = TOML.parsefile(abspath(path))
    table = _table(raw, "stage1")
    settings = BareStage1Settings(;
        top_modes=Int(_value(table, "top_modes", 6)),
        covariance_psd_tol=Float64(_value(table, "covariance_psd_tol", 1e-8)),
        bulk_edge_fractions=Float64.(_value(table, "bulk_edge_fractions", [0.15, 0.20, 0.25])),
        pair_classes=Symbol.(lowercase.(String.(_value(
            table,
            "pair_classes",
            ["onsite0", "onsite1", "rung", "leg0", "leg1"],
        )))),
    )
    settings.top_modes >= 1 || throw(ArgumentError("stage1 top_modes must be positive"))
    settings.covariance_psd_tol > 0 || throw(ArgumentError("covariance_psd_tol must be positive"))
    all(f -> 0 <= f < 0.5, settings.bulk_edge_fractions) || throw(ArgumentError(
        "bulk_edge_fractions must lie in [0, 0.5)",
    ))
    allowed = Set((:onsite0, :onsite1, :rung, :leg0, :leg1))
    all(in(allowed), settings.pair_classes) || throw(ArgumentError(
        "unknown Stage 1 pairing class",
    ))
    return settings
end

"""
Electron site indices carrying redundant `NfParity` together with `Nf` and
`Sz`. ITensor's built-in Electron `space` selects the `(Nf,Sz)` branch before
the parity branch, so an explicit space is required if the saved fixed-number
MPS must later support `removeqn(psi, "Nf")` without also losing parity.
"""
function backbone_siteinds(model::ModelSettings)
    space = [
        QN(("Nf", 0, -1), ("NfParity", 0, -2), ("Sz", 0)) => 1,
        QN(("Nf", 1, -1), ("NfParity", 1, -2), ("Sz", 1)) => 1,
        QN(("Nf", 1, -1), ("NfParity", 1, -2), ("Sz", -1)) => 1,
        QN(("Nf", 2, -1), ("NfParity", 0, -2), ("Sz", 0)) => 1,
    ]
    return [Index(space; tags="Site,Electron,n=$site") for site in 1:(2 * model.L)]
end

function backbone_sector_keys(model::ModelSettings)
    particle_number = round(Int, 2 * model.L * model.density)
    iseven(particle_number) || throw(ArgumentError(
        "the isolated-ladder backbone currently requires an even target particle number",
    ))
    return [(particle_number + delta_n, twice_sz) for (delta_n, twice_sz) in BACKBONE_SECTOR_OFFSETS]
end

backbone_sector_label(particle_number::Integer, twice_sz::Integer) =
    "N$(Int(particle_number))_twoSz$(Int(twice_sz))"

function _uniform_positions(total::Integer, selected::Integer)
    0 <= selected <= total || throw(ArgumentError("selected positions must lie in 0:total"))
    selected == 0 && return Int[]
    return [floor(Int, (index - 0.5) * total / selected) + 1 for index in 1:selected]
end

"""
Create a fixed-sector product state whose holes (or doublons above half filling)
are spread over the full open ladder. This avoids the legacy domain-wall state
that placed every hole at the right boundary.
"""
function spread_fixed_sector_product_state(
    sites,
    particle_number::Integer,
    twice_sz::Integer;
    phase::Integer=0,
)
    number_sites = length(sites)
    iseven(particle_number + twice_sz) || throw(ArgumentError("N + 2Sz must be even"))
    n_up = div(particle_number + twice_sz, 2)
    n_down = particle_number - n_up
    0 <= n_up <= number_sites || throw(ArgumentError("invalid up-spin count"))
    0 <= n_down <= number_sites || throw(ArgumentError("invalid down-spin count"))
    0 <= particle_number <= 2 * number_sites || throw(ArgumentError("invalid particle number"))

    states = fill("Emp", number_sites)
    doublons = max(0, particle_number - number_sites)
    for position in _uniform_positions(number_sites, doublons)
        states[position] = "UpDn"
    end
    available = findall(==("Emp"), states)
    up_only = n_up - doublons
    down_only = n_down - doublons
    up_slots = Set(_uniform_positions(length(available), up_only))
    remaining_up = up_only
    remaining_down = down_only
    for (slot, position) in enumerate(available)
        choose_up = slot in up_slots
        if isodd(phase) && remaining_up == remaining_down
            choose_up = !choose_up
        end
        if choose_up && remaining_up > 0
            states[position] = "Up"
            remaining_up -= 1
        elseif remaining_down > 0
            states[position] = "Dn"
            remaining_down -= 1
        elseif remaining_up > 0
            states[position] = "Up"
            remaining_up -= 1
        end
    end
    remaining_up == 0 && remaining_down == 0 || error("failed to construct fixed-sector state")
    return states
end

function _backbone_dmrg_settings(
    backbone::LadderBackboneSettings,
    nsweeps::Integer,
    maxdim::Integer,
)
    return DMRGSettings(;
        nsweeps=Int(nsweeps),
        maxdim=Int(maxdim),
        cutoff=backbone.cutoff,
        energy_tol=backbone.energy_tol,
        eigsolve_krylovdim=backbone.eigsolve_krylovdim,
        max_time_seconds=backbone.maximum_sector_seconds,
        output_level=backbone.output_level,
    )
end

function _pre_relax_schedules(backbone::LadderBackboneSettings)
    maximum = backbone.pre_relax_maxdim
    maxdims = _extend_schedule(
        unique([min(20, maximum), min(50, maximum), min(100, maximum), maximum]),
        backbone.pre_relax_sweeps,
    )
    noises = _extend_schedule(
        [max(1e-5, backbone.noise_floor), max(1e-6, backbone.noise_floor),
         max(1e-7, backbone.noise_floor), backbone.noise_floor],
        backbone.pre_relax_sweeps,
    )
    first_final = findfirst(==(maximum), maxdims)
    minimum_sweep = min(
        backbone.pre_relax_sweeps,
        something(first_final, backbone.pre_relax_sweeps) +
            backbone.minimum_sweeps_at_maxdim - 1,
    )
    return (; maxdims, noises, minimum_sweep)
end

function _chi_stage_schedules(backbone::LadderBackboneSettings, maximum::Integer)
    maxdims = fill(Int(maximum), backbone.stage_sweeps)
    noises = _extend_schedule(
        [max(1e-7, backbone.noise_floor), backbone.noise_floor],
        backbone.stage_sweeps,
    )
    return (;
        maxdims,
        noises,
        minimum_sweep=min(backbone.stage_sweeps, backbone.minimum_sweeps_at_maxdim),
    )
end

function last_five_sweep_change(energies::AbstractVector)
    finite = filter(isfinite, Float64.(energies))
    length(finite) < 2 && return Inf
    tail = finite[max(1, end - 4):end]
    return maximum(tail) - minimum(tail)
end

function run_backbone_sector(
    model::ModelSettings,
    backbone::LadderBackboneSettings,
    particle_number::Integer,
    twice_sz::Integer;
    runtime::RuntimeSettings=RuntimeSettings(),
    resume=nothing,
    stage_callback::Function=(psi, stages) -> nothing,
)
    runtime.backend == :cpu || throw(ArgumentError("isolated-ladder sectors require the CPU backend"))
    sites, psi, stages = if resume === nothing
        fresh_sites = backbone_siteinds(model)
        states = spread_fixed_sector_product_state(
            fresh_sites,
            particle_number,
            twice_sz;
            phase=backbone.random_seed,
        )
        (fresh_sites, productMPS(fresh_sites, states), NamedTuple[])
    else
        resume.particle_number == particle_number || throw(ArgumentError(
            "backbone checkpoint particle-number sector mismatch",
        ))
        resume.twice_sz == twice_sz || throw(ArgumentError(
            "backbone checkpoint spin sector mismatch",
        ))
        (siteinds(resume.psi), resume.psi, copy(resume.stages))
    end
    plan = vcat(
        [(kind=:pre_relax, maxdim=backbone.pre_relax_maxdim)],
        [(kind=:chi, maxdim=chi) for chi in backbone.chi_ladder],
    )
    length(stages) <= length(plan) || throw(ArgumentError(
        "backbone checkpoint contains more stages than the configured schedule",
    ))
    for index in eachindex(stages)
        stages[index].kind == plan[index].kind && stages[index].maxdim == plan[index].maxdim ||
            throw(ArgumentError("backbone checkpoint stage schedule differs from the configuration"))
    end
    hamiltonian = build_bare_ladder_mpo(sites, model; backend=runtime)
    deadline = time() + backbone.maximum_sector_seconds

    if isempty(stages)
        pre = _pre_relax_schedules(backbone)
        pre_settings = _backbone_dmrg_settings(
            backbone,
            backbone.pre_relax_sweeps,
            backbone.pre_relax_maxdim,
        )
        result = run_dmrg_ground(
            sites,
            hamiltonian,
            particle_number / length(sites),
            pre_settings;
            psi_init=psi,
            rng=MersenneTwister(backbone.random_seed + particle_number + twice_sz),
            deadline,
            backend=runtime,
            maxdim_schedule=pre.maxdims,
            noise_schedule=pre.noises,
            minimum_convergence_sweep=pre.minimum_sweep,
        )
        result.timed_out && error("backbone pre-relaxation timed out for $(backbone_sector_label(particle_number, twice_sz))")
        psi = result.psi
        push!(stages, _backbone_stage_record(
            :pre_relax,
            backbone.pre_relax_maxdim,
            result,
            backbone,
        ))
        stage_callback(psi, copy(stages))
    end

    for (chi_index, chi) in enumerate(backbone.chi_ladder)
        stage_index = chi_index + 1
        length(stages) >= stage_index && continue
        schedule = _chi_stage_schedules(backbone, chi)
        settings = _backbone_dmrg_settings(backbone, backbone.stage_sweeps, chi)
        result = run_dmrg_ground(
            sites,
            hamiltonian,
            particle_number / length(sites),
            settings;
            psi_init=psi,
            rng=MersenneTwister(backbone.random_seed + 17chi + particle_number + twice_sz),
            deadline,
            backend=runtime,
            maxdim_schedule=schedule.maxdims,
            noise_schedule=schedule.noises,
            minimum_convergence_sweep=schedule.minimum_sweep,
        )
        result.timed_out && error("backbone DMRG timed out at chi=$chi for $(backbone_sector_label(particle_number, twice_sz))")
        psi = result.psi
        push!(stages, _backbone_stage_record(:chi, chi, result, backbone))
        stage_callback(psi, copy(stages))
    end

    final = last(stages)
    return (;
        particle_number=Int(particle_number),
        twice_sz=Int(twice_sz),
        energy=final.energy,
        psi,
        stages,
        last_five_energy_change=final.last_five_energy_change,
        converged=final.kind == :chi && final.scientifically_converged,
    )
end

function write_backbone_stage_checkpoint(
    path::AbstractString,
    psi::MPS,
    stages,
    particle_number::Integer,
    twice_sz::Integer,
    model::ModelSettings,
    config_path::AbstractString;
    immutable::Bool=true,
)
    isempty(stages) && throw(ArgumentError("cannot checkpoint an empty backbone stage list"))
    destination = abspath(path)
    immutable && ispath(destination) && throw(ArgumentError(
        "refusing to overwrite immutable backbone stage checkpoint: $destination",
    ))
    mkpath(dirname(destination))
    temporary = tempname(dirname(destination))
    h5open(temporary, "w") do file
        file["schema_version"] = 1
        file["artifact_kind"] = "ladder_backbone_stage_checkpoint"
        file["complete"] = true
        file["particle_number"] = Int(particle_number)
        file["twice_sz"] = Int(twice_sz)
        file["completed_stage_count"] = length(stages)
        file["psi"] = psi
        _write_backbone_model(create_group(file, "model"), model)
        _write_backbone_stages(create_group(file, "dmrg_stages"), stages)
        provenance = create_group(file, "provenance")
        provenance["config_path"] = abspath(config_path)
        provenance["config_sha256"] = sha256_file(abspath(config_path))
        provenance["implementation_sha256"] = implementation_fingerprint()
        provenance["git_commit"] = _read_git("rev-parse", "HEAD")
        provenance["julia_threads"] = Threads.nthreads()
    end
    mv(temporary, destination; force=!immutable)
    return destination
end

function read_backbone_stage_checkpoint(path::AbstractString)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("backbone stage checkpoint not found: $source"))
    return h5open(source, "r") do file
        String(read(file, "artifact_kind")) == "ladder_backbone_stage_checkpoint" ||
            throw(ArgumentError("not a backbone stage checkpoint: $source"))
        Bool(read(file, "complete")) || throw(ArgumentError(
            "incomplete backbone stage checkpoint: $source",
        ))
        stages = _read_backbone_stages(file)
        Int(read(file, "completed_stage_count")) == length(stages) || throw(ArgumentError(
            "backbone stage checkpoint count does not match its evidence",
        ))
        return (;
            path=source,
            sha256=sha256_file(source),
            particle_number=Int(read(file, "particle_number")),
            twice_sz=Int(read(file, "twice_sz")),
            config_sha256=String(read(file, "provenance/config_sha256")),
            implementation_sha256=String(read(file, "provenance/implementation_sha256")),
            stages,
            psi=read(file, "psi", MPS),
        )
    end
end

function _backbone_stage_record(
    kind::Symbol,
    maxdim::Integer,
    result,
    backbone::LadderBackboneSettings,
)
    last_five = last_five_sweep_change(result.sweep_energies)
    return (;
        kind,
        maxdim=Int(maxdim),
        energy=result.energy,
        timed_out=result.timed_out,
        energy_converged=result.energy_converged,
        sweep_energies=result.sweep_energies,
        sweep_max_discarded_weights=result.sweep_max_discarded_weights,
        sweep_maxlinkdims=result.sweep_maxlinkdims,
        max_discarded_weight=result.max_discarded_weight,
        maximum_link_dimension=result.maximum_link_dimension,
        last_five_energy_change=last_five,
        scientifically_converged=!result.timed_out &&
            (backbone.energy_tol == 0 || result.energy_converged) &&
            last_five <= backbone.last_five_energy_tol,
    )
end

function _write_backbone_model(group, model::ModelSettings)
    for name in (:L, :t, :U, :V, :t0, :tp, :density, :r_range)
        group[String(name)] = getproperty(model, name)
    end
    group["geometry"] = String(model.geometry)
    group["ep"] = model.ep
    group["ep_signed"] = model.ep_signed
    return group
end

function _write_backbone_stages(group, stages)
    for (index, stage) in enumerate(stages)
        child = create_group(group, lpad(string(index), 3, '0'))
        child["kind"] = String(stage.kind)
        child["maxdim"] = stage.maxdim
        child["energy"] = stage.energy
        child["timed_out"] = stage.timed_out
        child["energy_converged"] = stage.energy_converged
        child["sweep_energy"] = stage.sweep_energies
        child["sweep_max_discarded_weight"] = stage.sweep_max_discarded_weights
        child["sweep_maxlinkdim"] = stage.sweep_maxlinkdims
        child["max_discarded_weight"] = stage.max_discarded_weight
        child["maximum_link_dimension"] = stage.maximum_link_dimension
        child["last_five_energy_change"] = stage.last_five_energy_change
        child["scientifically_converged"] = stage.scientifically_converged
    end
    return group
end

function _read_backbone_stages(file)
    parent = file["dmrg_stages"]
    names = sort(String.(collect(keys(parent))))
    return [
        let child = parent[name]
            (;
                kind=Symbol(String(read(child, "kind"))),
                maxdim=Int(read(child, "maxdim")),
                energy=Float64(read(child, "energy")),
                timed_out=Bool(read(child, "timed_out")),
                energy_converged=Bool(read(child, "energy_converged")),
                sweep_energies=Float64.(read(child, "sweep_energy")),
                sweep_max_discarded_weights=Float64.(read(child, "sweep_max_discarded_weight")),
                sweep_maxlinkdims=Int.(read(child, "sweep_maxlinkdim")),
                max_discarded_weight=Float64(read(child, "max_discarded_weight")),
                maximum_link_dimension=Int(read(child, "maximum_link_dimension")),
                last_five_energy_change=Float64(read(child, "last_five_energy_change")),
                scientifically_converged=Bool(read(child, "scientifically_converged")),
            )
        end
        for name in names
    ]
end

function write_backbone_sector(
    path::AbstractString,
    result,
    model::ModelSettings,
    config_path::AbstractString;
    immutable::Bool=true,
)
    destination = abspath(path)
    immutable && ispath(destination) && throw(ArgumentError(
        "refusing to overwrite immutable backbone sector: $destination",
    ))
    mkpath(dirname(destination))
    temporary = tempname(dirname(destination))
    h5open(temporary, "w") do file
        file["schema_version"] = 2
        file["artifact_kind"] = "ladder_backbone_sector"
        file["complete"] = true
        file["particle_number"] = result.particle_number
        file["twice_sz"] = result.twice_sz
        file["energy"] = result.energy
        file["last_five_energy_change"] = result.last_five_energy_change
        file["converged"] = result.converged
        file["psi"] = result.psi
        _write_backbone_model(create_group(file, "model"), model)
        _write_backbone_stages(create_group(file, "dmrg_stages"), result.stages)
        provenance = create_group(file, "provenance")
        provenance["config_path"] = abspath(config_path)
        provenance["config_sha256"] = sha256_file(abspath(config_path))
        provenance["implementation_sha256"] = implementation_fingerprint()
        provenance["git_commit"] = _read_git("rev-parse", "HEAD")
        provenance["julia_threads"] = Threads.nthreads()
    end
    mv(temporary, destination; force=!immutable)
    return destination
end

function read_backbone_sector(path::AbstractString; load_state::Bool=true)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("backbone sector not found: $source"))
    return h5open(source, "r") do file
        String(read(file, "artifact_kind")) == "ladder_backbone_sector" || throw(ArgumentError(
            "not a backbone sector artifact: $source",
        ))
        Bool(read(file, "complete")) || throw(ArgumentError("incomplete backbone sector: $source"))
        return (;
            path=source,
            sha256=sha256_file(source),
            particle_number=Int(read(file, "particle_number")),
            twice_sz=Int(read(file, "twice_sz")),
            energy=Float64(read(file, "energy")),
            last_five_energy_change=Float64(read(file, "last_five_energy_change")),
            converged=Bool(read(file, "converged")),
            config_sha256=String(read(file, "provenance/config_sha256")),
            implementation_sha256=String(read(file, "provenance/implementation_sha256")),
            stages=_read_backbone_stages(file),
            psi=load_state ? read(file, "psi", MPS) : nothing,
        )
    end
end

function backbone_energy_summary(sectors, model::ModelSettings)
    energies = Dict((sector.particle_number, sector.twice_sz) => sector.energy for sector in sectors)
    target = round(Int, 2 * model.L * model.density)
    required = backbone_sector_keys(model)
    all(haskey(energies, key) for key in required) || throw(ArgumentError(
        "sector collection does not contain every required backbone sector",
    ))
    e0 = energies[(target, 0)]
    return (;
        particle_number=target,
        energies,
        spin_gap=energies[(target, 2)] - e0,
        charge_gap=0.5 * (energies[(target + 2, 0)] + energies[(target - 2, 0)] - 2 * e0),
        hole_pair_binding=energies[(target - 2, 0)] + e0 - 2 * energies[(target - 1, 1)],
        particle_pair_binding=energies[(target + 2, 0)] + e0 - 2 * energies[(target + 1, 1)],
        chemical_potential=(energies[(target + 2, 0)] - energies[(target - 2, 0)]) / 4,
    )
end

function backbone_validity(summary, model::ModelSettings)
    ratio(scale) = scale > 0 ? model.tp / scale : Inf
    return (;
        tp_over_pair_binding=ratio(abs(summary.hole_pair_binding)),
        tp_over_spin_gap=ratio(summary.spin_gap),
        tp_over_charge_gap=ratio(summary.charge_gap),
        pair_binding_near_zero=abs(summary.hole_pair_binding) <= 10eps(Float64),
    )
end

function backbone_chi_dependence(sectors, model::ModelSettings)
    isempty(sectors) && return NamedTuple[]
    stage_count = length(first(sectors).stages)
    all(length(sector.stages) == stage_count for sector in sectors) || throw(ArgumentError(
        "backbone sectors do not have the same number of DMRG stages",
    ))
    dependence = NamedTuple[]
    for index in 1:stage_count
        reference = first(sectors).stages[index]
        all(
            sector.stages[index].kind == reference.kind &&
                sector.stages[index].maxdim == reference.maxdim
            for sector in sectors
        ) || throw(ArgumentError("backbone sector DMRG stage schedules differ"))
        stage_sectors = [
            (;
                particle_number=sector.particle_number,
                twice_sz=sector.twice_sz,
                energy=sector.stages[index].energy,
            )
            for sector in sectors
        ]
        push!(dependence, (;
            index,
            kind=reference.kind,
            maxdim=reference.maxdim,
            summary=backbone_energy_summary(stage_sectors, model),
            all_sectors_converged=all(
                sector.stages[index].scientifically_converged for sector in sectors
            ),
        ))
    end
    return dependence
end

function _write_energy_summary(group, summary)
    group["particle_number"] = summary.particle_number
    group["spin_gap"] = summary.spin_gap
    group["charge_gap"] = summary.charge_gap
    group["hole_pair_binding"] = summary.hole_pair_binding
    group["particle_pair_binding"] = summary.particle_pair_binding
    group["chemical_potential"] = summary.chemical_potential
    for (key, energy) in sort(collect(summary.energies); by=first)
        group[backbone_sector_label(key...)] = energy
    end
    return group
end

function write_backbone_artifact(
    path::AbstractString,
    sector_paths::AbstractVector{<:AbstractString},
    model::ModelSettings,
    config_path::AbstractString;
    immutable::Bool=true,
)
    destination = abspath(path)
    immutable && ispath(destination) && throw(ArgumentError(
        "refusing to overwrite immutable backbone artifact: $destination",
    ))
    sectors = [read_backbone_sector(path; load_state=false) for path in sector_paths]
    sort!(sectors; by=sector -> (sector.particle_number, sector.twice_sz))
    length(sectors) == length(BACKBONE_SECTOR_OFFSETS) || throw(ArgumentError(
        "expected $(length(BACKBONE_SECTOR_OFFSETS)) sector artifacts, got $(length(sectors))",
    ))
    length(Set((sector.particle_number, sector.twice_sz) for sector in sectors)) == length(sectors) ||
        throw(ArgumentError("duplicate backbone sector artifacts"))
    config_sha256 = sha256_file(abspath(config_path))
    all(sector.config_sha256 == config_sha256 for sector in sectors) || throw(ArgumentError(
        "backbone sectors were not produced from this exact configuration",
    ))
    length(Set(sector.implementation_sha256 for sector in sectors)) == 1 || throw(ArgumentError(
        "backbone sectors were not produced by the same implementation",
    ))
    summary = backbone_energy_summary(sectors, model)
    validity = backbone_validity(summary, model)
    chi_dependence = backbone_chi_dependence(sectors, model)
    all_sectors_converged = all(sector.converged for sector in sectors)

    mkpath(dirname(destination))
    temporary = tempname(dirname(destination))
    h5open(temporary, "w") do file
        file["schema_version"] = 2
        file["artifact_kind"] = "isolated_ladder_backbone"
        file["complete"] = true
        file["all_sectors_converged"] = all_sectors_converged
        file["scientifically_accepted"] = all_sectors_converged
        _write_backbone_model(create_group(file, "model"), model)
        energy_group = create_group(file, "energies")
        _write_energy_summary(energy_group, summary)
        dependence_group = create_group(file, "chi_dependence")
        for stage in chi_dependence
            label = stage.kind == :pre_relax ?
                "$(lpad(stage.index, 3, '0'))_pre_relax_chi$(stage.maxdim)" :
                "$(lpad(stage.index, 3, '0'))_chi$(stage.maxdim)"
            child = create_group(dependence_group, label)
            child["kind"] = String(stage.kind)
            child["maxdim"] = stage.maxdim
            child["all_sectors_converged"] = stage.all_sectors_converged
            _write_energy_summary(child, stage.summary)
        end
        validity_group = create_group(file, "validity")
        for key in keys(validity)
            validity_group[String(key)] = getproperty(validity, key)
        end
        sector_group = create_group(file, "sectors")
        for sector in sectors
            child = create_group(
                sector_group,
                backbone_sector_label(sector.particle_number, sector.twice_sz),
            )
            child["particle_number"] = sector.particle_number
            child["twice_sz"] = sector.twice_sz
            child["energy"] = sector.energy
            child["last_five_energy_change"] = sector.last_five_energy_change
            child["converged"] = sector.converged
            child["source_path"] = sector.path
            child["source_sha256"] = sector.sha256
            h5open(sector.path, "r") do source
                child["psi"] = read(source, "psi", MPS)
            end
        end
        provenance = create_group(file, "provenance")
        provenance["config_path"] = abspath(config_path)
        provenance["config_sha256"] = config_sha256
        provenance["implementation_sha256"] = implementation_fingerprint()
        provenance["git_commit"] = _read_git("rev-parse", "HEAD")
    end
    mv(temporary, destination; force=!immutable)
    return (;
        path=destination,
        sha256=sha256_file(destination),
        summary,
        validity,
        chi_dependence,
        all_sectors_converged,
    )
end

function read_backbone_ground_state(path::AbstractString)
    source = abspath(path)
    isfile(source) || throw(ArgumentError("backbone artifact not found: $source"))
    return h5open(source, "r") do file
        String(read(file, "artifact_kind")) == "isolated_ladder_backbone" || throw(ArgumentError(
            "not an isolated-ladder backbone artifact: $source",
        ))
        Bool(read(file, "complete")) || throw(ArgumentError("incomplete backbone artifact: $source"))
        Bool(read(file, "scientifically_accepted")) || throw(ArgumentError(
            "backbone artifact failed its sector-convergence gates: $source",
        ))
        target = Int(read(file, "energies/particle_number"))
        label = backbone_sector_label(target, 0)
        return (;
            psi=read(file, "sectors/$label/psi", MPS),
            particle_number=target,
            energy=Float64(read(file, "sectors/$label/energy")),
            chemical_potential=Float64(read(file, "energies/chemical_potential")),
            config_sha256=String(read(file, "provenance/config_sha256")),
            implementation_sha256=String(read(file, "provenance/implementation_sha256")),
            backbone_sha256=sha256_file(source),
        )
    end
end
