#!/usr/bin/env julia
# Read-only history qualification and temporary preparation; no DMRG or scheduler.
using Test, CSV, HDF5, LadderMPSMFT, TOML
include("prepare_phase1_two_basin_grid.jl")
include("prepare_phase1_square_two_basin_finish.jl")
include("replay_two_basin_convergence.jl")

function qualify_next_campaigns()
    root = dirname(@__DIR__)
    cubic_base = joinpath(root,"configs/phase1_gpu_cubic_unfrustrated_two_basin_chi200_raw60.toml")
    square_base = joinpath(root,"configs/phase1_gpu_square_two_basin_finish20.toml")
    anchor = joinpath(root,"output/phase1_gpu/20260908_square_two_basin_95_5_80_anchors")
    cubic = load_settings(cubic_base); square = load_settings(square_base)
    @testset "new campaign preparation" begin
        mktempdir() do tmp
            rows = prepare_two_basin_grid(cubic_base,joinpath(root,"data/two_basin_references.h5"),
                joinpath(tmp,"cubic_control"),joinpath(tmp,"cubic_full"),"cubic_test";
                stage="grid",geometry=:cubic_unfrustrated)
            @test length(rows) == 18
            @test Set((r.t0,r.V,r.family) for r in rows) ==
                Set((t,V,f) for t in (1.,1.2,1.4), V in (-.4,-.2,0.), f in ("stripe","pairing"))
            for row in rows
                s = load_settings(row.config)
                @test s.model.geometry == :cubic_unfrustrated
                @test (s.run.max_iterations,s.convergence.minimum_iterations,s.convergence.stable_iterations) == (60,40,10)
                @test s.convergence.probe_iterations >= s.run.max_iterations
                @test s.convergence.field_rel_tol == 1e-4 && s.convergence.channel_noise_floor == 1.5e-6
                @test s.run.save_every == 1
                h5open(row.seed,"r") do f
                    @test read(f,"seed_provenance/target_geometry") == "cubic_unfrustrated"
                    @test read(f,"seed_provenance/epsilon") == .05
                    correlations = CorrelationState((read(f,"template_correlations/$(key)")
                        for key in fieldnames(CorrelationState))...)
                    fields = LadderMPSMFT.mean_fields_from_correlations(correlations,s.model)
                    @test read(f,"fields/restart/alpha") == fields.alpha
                    @test read(f,"fields/restart/mu_cdw") == fields.mu_cdw
                end
            end
            preview = joinpath(tmp,"square_control")
            pairs = prepare_square_two_basin_finish(square_base,anchor,preview,
                joinpath(tmp,"square_full"),"square_test";compact_preview=true)
            @test length(pairs) == 2
            @test !TOML.parsefile(joinpath(preview,"continuation_contract.toml"))["full_sources_verified"]
            @test Set(r.parent_iterations for r in pairs) == Set([80,62])
            for row in pairs
                s = load_settings(row.config)
                @test TOML.parsefile(row.config)["run"]["parent_checkpoint"] == row.parent_checkpoint
                @test s.run.parent_sha256 == row.parent_sha256
                @test s.run.resume_checkpoint === nothing && s.run.inherit_from === nothing
                @test s.run.max_iterations == 20 && s.convergence.minimum_iterations == 10
                @test s.convergence.channel_noise_floor == 5e-7 && s.convergence.field_rel_tol == 1e-4
                @test s.model.mu_initial == 1.65 # parent supplies actual initial mu
            end
            @test_throws ErrorException prepare_square_two_basin_finish(square_base,anchor,preview,
                joinpath(tmp,"square_full"),"square_test";compact_preview=true)
            if !isfile(first(pairs).parent_checkpoint)
                blocked = joinpath(tmp,"no_full_mps")
                @test_throws ErrorException prepare_square_two_basin_finish(square_base,anchor,blocked,
                    joinpath(tmp,"unverified_full"),"must_fail")
                @test !isdir(joinpath(blocked,"configs"))
            end
        end
    end
    output = NamedTuple[]
    for source in CSV.File(joinpath(root,"docs/reports/two_basin_grid_20260915/sources.csv"))
        path = joinpath(root,split(replace(source.source,'\\'=>'/'),'/')...)
        @test LadderMPSMFT.sha256_file(path) == source.source_sha256
        records, terminal = replay_records(path)
        for (name,settings) in (("square_finish",square.convergence),("cubic_unscaled_square_stress_test",cubic.convergence))
            passes = Int[]
            for i in settings.minimum_iterations:length(records)
                history_gates(@view(records[1:i]),settings).all_available_gates && push!(passes,i)
            end
            last = history_gates(records,settings)
            push!(output,merge((controls=name,t0=source.t0,V=source.V,family=String(source.family),
                records=length(records),first_history_gate_pass=isempty(passes) ? 0 : first(passes),
                source_sha256=String(source.source_sha256)),last))
            # Both known resolved instabilities must remain blocked at every prefix.
            if (source.t0 == 1.4 && source.V == 0) || (source.t0 == 1.2 && source.V == -.4)
                @test isempty(passes)
            end
            if source.t0 == 1.4 && source.V == -.4
                @test !isempty(passes)
            end
        end
        @test LadderMPSMFT.sha256_file(path) == source.source_sha256
    end
    out = joinpath(root,"docs/reports/two_basin_next_campaigns_20260915")
    mkpath(out)
    CSV.write(joinpath(out,"history_gate_qualification.csv"),output)
    open(joinpath(out,"qualification_contract.toml"),"w") do io
        TOML.print(io,Dict("cubic_config_sha256"=>LadderMPSMFT.sha256_file(cubic_base),
            "square_config_sha256"=>LadderMPSMFT.sha256_file(square_base),
            "implementation_sha256"=>implementation_fingerprint(),
            "boundary"=>"Saved square histories only; cubic thresholds tested on unscaled square fields as a more permissive stress test, not a cubic convergence forecast. Missing per-iteration identity errors prevent acceptance certification. Original sources and flags unchanged. Square continuation uses a fresh ten-record window, not old acceptance history."))
    end
    println("Wrote history qualification: ",out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    qualify_next_campaigns()
end
