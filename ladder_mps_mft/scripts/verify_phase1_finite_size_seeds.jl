#!/usr/bin/env julia
# Local HDF5/config integration check. No MPS construction or DMRG.
using HDF5
using LadderMPSMFT
using Test

project = normpath(joinpath(@__DIR__, ".."))
config_dir = isempty(ARGS) ? joinpath(project, "configs", "phase1_gpu_square_size_compare_chi200") : abspath(only(ARGS))
configs = sort(filter(path -> endswith(path, ".toml"), readdir(config_dir; join=true)))
@assert length(configs) == 4
numerical_fingerprints = String[]
model_fingerprints = Dict{Int,Vector{String}}()

@testset "four field-only length seeds through the solver reader" begin
    for path in configs
        settings = load_settings(path)
        model = settings.model
        @test model.L in (96, 128)
        @test settings.dmrg.maxdim == 200
        @test settings.convergence.hamiltonian_identity_tol == 1e-8
        @test model.ep_reference_L == 64
        @test model.ep_mode == :fixed_reference_length
        @test model.ep_signed == -0.14653773091916378
        @test settings.run.parent_checkpoint === nothing
        @test settings.run.resume_checkpoint === nothing
        LadderMPSMFT.verify_inherit!(settings)
        seed = read_inherited_fields(settings.run.inherit_from)
        @test seed.format == :refactored
        @test seed.source_geometry == "square"
        @test LadderMPSMFT._field_shapes_match(seed.fields, model)
        h5open(settings.run.inherit_from, "r") do file
            @test !haskey(file, "psi")
            source_path = joinpath(project, String(read(file, "seed_provenance/source_path")))
            @test LadderMPSMFT.sha256_file(source_path) == String(read(file, "seed_provenance/source_compact_sha256"))
            @test seed.chemical_potential == Float64(read(file, "chemical_potential"))
            h5open(source_path, "r") do source
                for name in (:alpha, :beta, :mu_cdw)
                    old = read(source, "fields/measured/$name")
                    new = getproperty(seed.fields, name)
                    if name == :alpha
                        @test new[1:32,1:32,:,:] == old[1:32,1:32,:,:]
                        @test new[end-31:end,end-31:end,:,:] == old[end-31:end,end-31:end,:,:]
                    elseif name == :beta
                        @test new[:,1:32,1:32,:,:] == old[:,1:32,1:32,:,:]
                        @test new[:,end-31:end,end-31:end,:,:] == old[:,end-31:end,end-31:end,:,:]
                    else
                        @test new[:,1:64] == old[:,1:64]
                        @test new[:,end-63:end] == old[:,end-63:end]
                    end
                end
            end
        end
        push!(numerical_fingerprints, LadderMPSMFT.numerical_fingerprint(settings))
        push!(get!(model_fingerprints, model.L, String[]), LadderMPSMFT.model_fingerprint(model))
        println("config=$(basename(path)) L=$(model.L) E_p=$(model.ep_signed) reference_L=$(model.ep_reference_L) seed_sha256=$(settings.run.inherit_sha256)")
    end
    @test length(unique(numerical_fingerprints)) == 1
    @test all(length(unique(values)) == 1 for values in Base.values(model_fingerprints))
    @test length(unique(first(values) for values in Base.values(model_fingerprints))) == 2
end
