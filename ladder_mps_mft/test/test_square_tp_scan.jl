module SquareTpScanTests
using Test, LadderMPSMFT, HDF5, TOML, LinearAlgebra
include("../scripts/prepare_phase1_square_tp_scan.jl")

const ROOT = dirname(@__DIR__)

@testset "square t_perp campaign preparation" begin
    mktempdir() do directory
        base = joinpath(ROOT,"configs/phase1_gpu_square_tp_scan_chi200_raw60.toml")
        reference = joinpath(ROOT,"data/two_basin_references.h5")
        control, full = joinpath(directory,"control"), joinpath(directory,"full")
        rows = prepare_square_tp_scan(base,reference,control,full,"tp_scan_test")
        expected = Set((v,tp,f) for (v,tp) in ((0.,.06),(0.,.08),(-.2,.12),(-.2,.14))
                       for f in ("stripe","pairing"))
        @test length(rows) == 8
        @test Set((r.V,r.tp,r.family) for r in rows) == expected
        @test length(unique(r.label for r in rows)) == 8
        @test length(unique(r.model_fingerprint for r in rows)) == 4
        @test length(unique(r.numerical_fingerprint for r in rows)) == 1
        @test_throws ErrorException prepare_square_tp_scan(base,reference,control,full,"tp_scan_test")
        stripe, pairing = two_basin_references(reference)
        for r in rows
            s = load_settings(r.config)
            @test (s.model.L,s.model.t,s.model.U,s.model.t0,s.model.density) == (64,1.,8.,1.4,.9375)
            @test s.model.geometry == :square && s.model.spatial_cell == :one_ladder
            @test (s.model.V,s.model.tp) == (r.V,r.tp)
            @test s.model.ep_mode == :exact
            @test s.model.ep_signed == (r.V == 0 ? -.14653773091916378 : -.2068002629740704)
            @test (s.dmrg.maxdim,s.run.max_iterations,s.convergence.minimum_iterations,s.convergence.stable_iterations) == (200,60,40,10)
            @test s.convergence.probe_iterations == 60 && s.mixing.damping == 1
            @test !s.mixing.adaptive && s.mixing.method == :linear
            @test s.run.parent_checkpoint === nothing && s.run.resume_checkpoint === nothing
            @test s.run.random_seed == 1404 && s.run.full_pair_correlations
            @test s.run.inherit_sha256 == LadderMPSMFT.sha256_file(r.seed) == r.seed_sha256
            @test r.config_sha256 == LadderMPSMFT.sha256_file(r.config)
            @test_throws ErrorException validate_raw_basin_contract(s) # legacy campaigns still pin tp=.1
            # Independent expected mixture and quadratic scaling against the old tp=.1 map.
            primary, other = r.family == "stripe" ? (stripe,pairing) : (pairing,stripe)
            mixed = CorrelationState((.95 .* getfield(primary,k) .+ .05 .* getfield(other,k)
                                      for k in fieldnames(CorrelationState))...)
            raw = TOML.parsefile(r.config)
            raw["model"]["tp"] = .1
            reference_config = joinpath(directory,"tp01.toml")
            open(io -> TOML.print(io,raw),reference_config,"w")
            old_fields = LadderMPSMFT.mean_fields_from_correlations(mixed,load_settings(reference_config).model)
            saved = read_inherited_fields(r.seed).fields
            for component in (:alpha,:beta,:mu_cdw)
                @test getfield(saved,component) ≈ (r.tp/.1)^2 .* getfield(old_fields,component) rtol=1e-13
            end
            @test norm(saved.alpha) > 1e-8
            @test norm(saved.mu_cdw[2,:] .- saved.mu_cdw[1,:]) > 1e-6
            h5open(r.seed,"r") do f
                @test read(f,"seed_provenance/target_tp") == r.tp
                @test read(f,"seed_provenance/fresh_mps")
                @test read(f,"seed_provenance/reference_bundle_sha256") == TWO_BASIN_REFERENCE_SHA
            end
        end
        contract = TOML.parsefile(joinpath(control,"seed_contract.toml"))
        @test contract["branches"] == 8 && contract["stage"] == "tp_scan"
        @test !contract["interpolated_ep"] && contract["fresh_mps"]

        # The reused preparer still emits the historical four anchors at tp=.1.
        anchors = prepare_two_basin_grid(base,reference,joinpath(directory,"anchors"),
            joinpath(directory,"anchors_full"),"anchors_regression")
        @test length(anchors) == 4 && all(r.tp == .1 for r in anchors)
        @test Set((r.t0,r.V,r.family) for r in anchors) ==
            Set((1.4,v,f) for v in (-.4,0.) for f in ("stripe","pairing"))
    end
end
end
