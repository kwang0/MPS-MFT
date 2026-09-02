#!/usr/bin/env julia

using LadderMPSMFT

function usage()
    error(
        "usage: julia --project=. scripts/run_ladder_backbone.jl " *
        "CONFIG.toml RUN_DIRECTORY {all|assemble|SECTOR_INDEX}",
    )
end

length(ARGS) == 3 || usage()
config_path = abspath(ARGS[1])
run_directory = abspath(ARGS[2])
action = lowercase(ARGS[3])
settings = load_settings(config_path)
settings.runtime.backend == :cpu || error("the isolated-ladder backbone requires runtime.backend=cpu")
backbone = load_ladder_backbone_settings(config_path)
threading = configure_threading!(settings.runtime)
sector_directory = joinpath(run_directory, "sectors")
mkpath(sector_directory)
keys = backbone_sector_keys(settings.model)
stage_plan = vcat(
    [(kind=:pre_relax, maxdim=backbone.pre_relax_maxdim)],
    [(kind=:chi, maxdim=chi) for chi in backbone.chi_ladder],
)

function sector_path(key)
    return joinpath(sector_directory, backbone_sector_label(key...) * ".h5")
end

function run_one(index::Int)
    1 <= index <= length(keys) || error("sector index must lie in 1:$(length(keys))")
    key = keys[index]
    path = sector_path(key)
    ispath(path) && error("refusing to overwrite immutable sector artifact: $path")
    println("backbone_sector_index=$index")
    println("backbone_sector=$(backbone_sector_label(key...))")
    println("julia_threads=$(threading.julia)")
    println("threaded_blocksparse=$(threading.blocksparse)")
    checkpoint_directory = joinpath(run_directory, "stage_checkpoints", backbone_sector_label(key...))
    checkpoint_path(stage_index) = let stage = stage_plan[stage_index]
        label = stage.kind == :pre_relax ? "pre_relax_chi$(stage.maxdim)" : "chi$(stage.maxdim)"
        joinpath(
            checkpoint_directory,
            "stage_$(lpad(stage_index, 3, '0'))_$(label).h5",
        )
    end
    resume = nothing
    for stage_index in reverse(eachindex(stage_plan))
        candidate = checkpoint_path(stage_index)
        isfile(candidate) || continue
        resume = read_backbone_stage_checkpoint(candidate)
        resume.config_sha256 == LadderMPSMFT.sha256_file(config_path) || error(
            "backbone checkpoint configuration hash mismatch: $candidate",
        )
        resume.implementation_sha256 == LadderMPSMFT.implementation_fingerprint() || error(
            "backbone checkpoint implementation hash mismatch: $candidate",
        )
        println("resuming_backbone_checkpoint=$candidate")
        println("resuming_completed_stages=$(length(resume.stages))")
        break
    end
    function save_stage(psi, stages)
        stage_index = length(stages)
        checkpoint = checkpoint_path(stage_index)
        write_backbone_stage_checkpoint(
            checkpoint,
            psi,
            stages,
            key...,
            settings.model,
            config_path;
            immutable=true,
        )
        println("stage_checkpoint=$checkpoint")
        println("stage_checkpoint_sha256=$(LadderMPSMFT.sha256_file(checkpoint))")
    end
    result = run_backbone_sector(
        settings.model,
        backbone,
        key...;
        runtime=settings.runtime,
        resume,
        stage_callback=save_stage,
    )
    write_backbone_sector(path, result, settings.model, config_path; immutable=true)
    println("sector_path=$path")
    println("sector_sha256=$(LadderMPSMFT.sha256_file(path))")
    println("sector_energy=$(result.energy)")
    println("last_five_energy_change=$(result.last_five_energy_change)")
    println("sector_converged=$(result.converged)")
    return path
end

function assemble()
    paths = [sector_path(key) for key in keys]
    missing = filter(path -> !isfile(path), paths)
    isempty(missing) || error("cannot assemble; missing sector artifacts: $(join(missing, ", "))")
    output_path = joinpath(run_directory, "backbone.h5")
    result = write_backbone_artifact(
        output_path,
        paths,
        settings.model,
        config_path;
        immutable=true,
    )
    println("backbone_path=$(result.path)")
    println("backbone_sha256=$(result.sha256)")
    println("all_sectors_converged=$(result.all_sectors_converged)")
    println("spin_gap=$(result.summary.spin_gap)")
    println("charge_gap=$(result.summary.charge_gap)")
    println("hole_pair_binding=$(result.summary.hole_pair_binding)")
    println("particle_pair_binding=$(result.summary.particle_pair_binding)")
    println("chemical_potential=$(result.summary.chemical_potential)")
    println("tp_over_pair_binding=$(result.validity.tp_over_pair_binding)")
    println("tp_over_spin_gap=$(result.validity.tp_over_spin_gap)")
    println("tp_over_charge_gap=$(result.validity.tp_over_charge_gap)")
    for stage in result.chi_dependence
        println(
            "chi_dependence_stage=$(stage.index) kind=$(stage.kind) chi=$(stage.maxdim) " *
            "converged=$(stage.all_sectors_converged) spin_gap=$(stage.summary.spin_gap) " *
            "charge_gap=$(stage.summary.charge_gap) " *
            "hole_pair_binding=$(stage.summary.hole_pair_binding) " *
            "particle_pair_binding=$(stage.summary.particle_pair_binding)",
        )
    end
end

if action == "all"
    for index in eachindex(keys)
        run_one(index)
    end
    assemble()
elseif action == "assemble"
    assemble()
else
    index = tryparse(Int, action)
    index === nothing && usage()
    run_one(index)
end
