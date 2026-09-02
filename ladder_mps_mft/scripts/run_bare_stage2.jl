#!/usr/bin/env julia

using HDF5
using LadderMPSMFT
using Printf

function usage()
    error(
        "usage:\n" *
        "  julia --project=. scripts/run_bare_stage2.jl prepare CONFIG.toml BACKBONE.h5 STAGE1.h5 CANDIDATE_BANK.h5\n" *
        "  julia --project=. scripts/run_bare_stage2.jl normal-reference CONFIG.toml CANDIDATE_BANK.h5 OUTPUT.h5\n" *
        "  julia --project=. scripts/run_bare_stage2.jl pair-reference CONFIG.toml CANDIDATE_BANK.h5 NORMAL_REFERENCE.h5 OUTPUT.h5\n" *
        "  julia --project=. scripts/run_bare_stage2.jl probe CONFIG.toml CANDIDATE_BANK.h5 NORMAL_REFERENCE.h5 PAIR_REFERENCE.h5|- {normal|pair} LOCAL_INDEX OUTPUT.h5\n" *
        "  julia --project=. scripts/run_bare_stage2.jl assemble CONFIG.toml CANDIDATE_BANK.h5 PROBE_DIRECTORY OUTPUT.h5\n" *
        "  julia --project=. scripts/run_bare_stage2.jl validate CONFIG.toml CANDIDATE_BANK.h5 DISCOVERY.h5 NORMAL_REFERENCE.h5 PAIR_REFERENCE.h5 VALIDATION_INDEX OUTPUT.h5\n" *
        "  julia --project=. scripts/run_bare_stage2.jl assemble-validation CONFIG.toml DISCOVERY.h5 VALIDATION_DIRECTORY OUTPUT.h5 COUNT",
    )
end

isempty(ARGS) && usage()
action = lowercase(ARGS[1])

function write_bank_summary(bank_path::AbstractString)
    destination = splitext(abspath(bank_path))[1] * "_summary.tsv"
    ispath(destination) && error("refusing to overwrite candidate-bank summary: $destination")
    h5open(abspath(bank_path), "r") do file
        candidate_projection = read(file, "candidate_projection")
        residual = read(file, "candidate_residual_norm")
        retained = Int.(read(file, "candidate_retained_basis_index"))
        candidate_parent = file["candidates"]
        open(destination, "w") do io
            println(io, "candidate_index\tlabel\tblock\tchannel\tparity\torigin\tmode_number\tq_over_pi\tform_factor\tresidual_norm\tretained_basis_index\tbasis_coefficients")
            for (index, name) in enumerate(sort(String.(collect(keys(candidate_parent)))))
                child = candidate_parent[name]
                coefficients = join((@sprintf("%.16g", value) for value in candidate_projection[:, index]), ",")
                @printf(
                    io,
                    "%d\t%s\t%s\t%s\t%s\t%s\t%d\t%.16g\t%s\t%.16g\t%d\t%s\n",
                    index,
                    String(read(child, "label")),
                    String(read(child, "block")),
                    String(read(child, "channel")),
                    String(read(child, "parity")),
                    String(read(child, "origin")),
                    Int(read(child, "mode_number")),
                    Float64(read(child, "q_over_pi")),
                    String(read(child, "form_factor")),
                    Float64(residual[index]),
                    retained[index],
                    coefficients,
                )
            end
        end
    end
    return destination
end

if action == "prepare"
    length(ARGS) == 5 || usage()
    config_path, backbone_path, stage1_path, output_path = abspath.(ARGS[2:5])
    project = load_settings(config_path)
    stage2 = load_bare_stage2_settings(config_path, project.model)
    threading = configure_threading!(project.runtime)
    println("stage2_action=prepare")
    println("julia_threads=$(threading.julia)")
    println("threaded_blocksparse=$(threading.blocksparse)")
    result = write_stage2_candidate_bank(
        output_path,
        stage1_path,
        backbone_path,
        project.model,
        stage2,
        config_path;
        immutable=true,
    )
    summary_path = write_bank_summary(result.path)
    println("candidate_bank=$(result.path)")
    println("candidate_bank_sha256=$(result.sha256)")
    println("candidate_count=$(result.candidate_count)")
    println("basis_count=$(result.basis_count)")
    println("normal_basis_count=$(result.normal_basis_count)")
    println("pair_basis_count=$(result.pair_basis_count)")
    println("bank_fingerprint=$(result.fingerprint)")
    println("maximum_orthogonality_error=$(result.maximum_orthogonality_error)")
    println("candidate_bank_summary=$summary_path")
elseif action == "normal-reference"
    length(ARGS) == 4 || usage()
    config_path, bank_path, output_path = abspath.(ARGS[2:4])
    project = load_settings(config_path)
    threading = configure_threading!(project.runtime)
    println("stage2_action=normal_reference")
    println("julia_threads=$(threading.julia)")
    println("threaded_blocksparse=$(threading.blocksparse)")
    result = run_stage2_normal_reference(bank_path, config_path, output_path)
    open(result.path * ".sha256", "w") do io
        println(io, result.sha256, "  ", basename(result.path))
    end
    println("normal_reference=$(result.path)")
    println("normal_reference_sha256=$(result.sha256)")
    println("scientifically_accepted=$(result.accepted)")
    println("density=$(result.density)")
    println("energy=$(result.energy)")
    println("last_five_energy_change=$(result.last_five_energy_change)")
elseif action == "pair-reference"
    length(ARGS) == 5 || usage()
    config_path, bank_path, normal_reference_path, output_path = abspath.(ARGS[2:5])
    project = load_settings(config_path)
    threading = configure_threading!(project.runtime)
    println("stage2_action=pair_reference")
    println("julia_threads=$(threading.julia)")
    println("threaded_blocksparse=$(threading.blocksparse)")
    result = run_stage2_pair_reference(
        bank_path,
        normal_reference_path,
        config_path,
        output_path,
    )
    open(result.path * ".sha256", "w") do io
        println(io, result.sha256, "  ", basename(result.path))
    end
    println("pair_reference=$(result.path)")
    println("pair_reference_sha256=$(result.sha256)")
    println("scientifically_accepted=$(result.accepted)")
    println("density=$(result.density)")
    println("energy=$(result.energy)")
    println("last_five_energy_change=$(result.last_five_energy_change)")
elseif action == "probe"
    length(ARGS) == 8 || usage()
    config_path = abspath(ARGS[2])
    bank_path = abspath(ARGS[3])
    normal_reference_path = abspath(ARGS[4])
    pair_reference_path = ARGS[5] == "-" ? nothing : abspath(ARGS[5])
    block = Symbol(lowercase(ARGS[6]))
    local_index = something(tryparse(Int, ARGS[7]), 0)
    output_path = abspath(ARGS[8])
    project = load_settings(config_path)
    threading = configure_threading!(project.runtime)
    println("stage2_action=probe")
    println("stage2_block=$block")
    println("stage2_local_index=$local_index")
    println("julia_threads=$(threading.julia)")
    println("threaded_blocksparse=$(threading.blocksparse)")
    result = run_stage2_discovery_probe(
        bank_path,
        normal_reference_path,
        pair_reference_path,
        config_path,
        block,
        local_index,
        output_path,
    )
    println("probe_path=$(result.path)")
    println("probe_sha256=$(result.sha256)")
    println("scientifically_accepted=$(result.accepted)")
    println("density=$(result.density)")
    println("energy=$(result.energy)")
    println("last_five_energy_change=$(result.last_five_energy_change)")
elseif action == "assemble"
    length(ARGS) == 5 || usage()
    config_path, bank_path, probe_directory, output_path = abspath.(ARGS[2:5])
    bank = read_stage2_candidate_bank(bank_path)
    paths = String[]
    for local_index in eachindex(bank.normal_indices)
        push!(paths, joinpath(probe_directory, @sprintf("normal_%03d.h5", local_index)))
    end
    for local_index in eachindex(bank.pair_indices)
        push!(paths, joinpath(probe_directory, @sprintf("pair_%03d.h5", local_index)))
    end
    summary_path = splitext(output_path)[1] * "_summary.tsv"
    println("stage2_action=assemble")
    println("probe_count=$(length(paths))")
    result = assemble_stage2_discovery(
        bank_path,
        paths,
        config_path,
        output_path,
        summary_path,
    )
    println("stage2_discovery=$(result.path)")
    println("stage2_discovery_sha256=$(result.sha256)")
    println("stage2_summary=$(result.summary_path)")
    println("stage2_gates=$(result.gate_path)")
    println("stage2_validation_plan=$(result.validation_path)")
    println("scientifically_accepted=$(result.scientifically_accepted)")
    println("all_probes_accepted=$(result.all_probes_accepted)")
    println("reciprocity_relative_error=$(result.reciprocity_relative_error)")
    println("raw_cross_block_relative_norm=$(result.raw_cross_block_relative_norm)")
    println("maximum_projected_leakage_relative=$(result.maximum_projected_leakage_relative)")
    println("maximum_beta_fraction=$(result.maximum_beta_fraction)")
    println("validation_count=$(result.validation_count)")
elseif action == "validate"
    length(ARGS) == 8 || usage()
    config_path, bank_path, discovery_path, normal_reference_path, pair_reference_path =
        abspath.(ARGS[2:6])
    validation_index = something(tryparse(Int, ARGS[7]), 0)
    output_path = abspath(ARGS[8])
    project = load_settings(config_path)
    threading = configure_threading!(project.runtime)
    println("stage2_action=validate")
    println("validation_index=$validation_index")
    println("julia_threads=$(threading.julia)")
    println("threaded_blocksparse=$(threading.blocksparse)")
    result = run_stage2_validation_probe(
        discovery_path,
        bank_path,
        normal_reference_path,
        pair_reference_path,
        config_path,
        validation_index,
        output_path,
    )
    println("validation_path=$(result.path)")
    println("validation_sha256=$(result.sha256)")
    println("label=$(result.label)")
    println("geometry=$(result.geometry)")
    println("block=$(result.block)")
    println("linearity_relative_error=$(result.linearity_relative_error)")
    println("scientifically_accepted=$(result.accepted)")
elseif action == "assemble-validation"
    length(ARGS) == 6 || usage()
    config_path, discovery_path, validation_directory, output_path = abspath.(ARGS[2:5])
    count = something(tryparse(Int, ARGS[6]), 0)
    count >= 1 || error("validation count must be positive")
    paths = [
        joinpath(validation_directory, @sprintf("validation_%03d.h5", index))
        for index in 1:count
    ]
    summary_path = splitext(output_path)[1] * "_summary.tsv"
    println("stage2_action=assemble_validation")
    result = assemble_stage2_validation(
        discovery_path,
        paths,
        config_path,
        output_path,
        summary_path,
    )
    println("stage2_validation=$(result.path)")
    println("stage2_validation_sha256=$(result.sha256)")
    println("stage2_validation_summary=$(result.summary_path)")
    println("validation_count=$(result.validation_count)")
    println("scientifically_accepted=$(result.scientifically_accepted)")
else
    usage()
end
