module TrellisIntertwinedPreparationTests
using Test
include("../scripts/prepare_phase1_trellis_intertwined.jl")
const ROOT = dirname(@__DIR__)

@testset "four fresh two-ladder intertwined starts" begin
    mktempdir() do directory
        base = joinpath(ROOT,"configs/phase1_gpu_trellis_intertwined_chi200_raw60.toml")
        recipe_path = joinpath(ROOT,"data/positive_v_intertwined_recipe.toml")
        control,full = joinpath(directory,"control"),joinpath(directory,"full")
        rows = prepare_trellis_intertwined(base,recipe_path,control,full,"intertwined_test")
        expected_ep = Dict((1.2,0.)=>-.17989619749147323,(1.2,.2)=>-.15307266912955697,
                           (1.4,0.)=>-.14653773091916378,(1.4,.2)=>-.11678278200975001)
        @test length(rows)==4 && Set((r.t0,r.V) for r in rows)==Set(keys(expected_ep))
        @test length(unique(r.label for r in rows))==4
        @test length(unique(r.model_fingerprint for r in rows))==4
        @test length(unique(r.numerical_fingerprint for r in rows))==1
        @test_throws ErrorException prepare_trellis_intertwined(base,recipe_path,control,full,"duplicate")
        @test_throws ErrorException prepare_trellis_intertwined(base,recipe_path,control,full,"../unsafe")
        templates = CorrelationState[]
        for row in rows
            s = load_settings(row.config)
            @test (s.model.t0,s.model.V,s.model.ep_signed)==(row.t0,row.V,expected_ep[(row.t0,row.V)])
            @test s.model.ep_mode==:exact && s.model.tau0==s.model.tau1==.1
            @test s.model.geometry==:trellis && s.model.trellis_cell==:two_ladder
            @test (s.model.L,s.model.U,s.model.density,s.dmrg.maxdim)==(64,8.,.9375,200)
            @test (s.run.max_iterations,s.convergence.probe_iterations,s.convergence.minimum_iterations,s.convergence.stable_iterations)==(60,60,40,10)
            @test s.run.parent_checkpoint===nothing && s.run.resume_checkpoint===nothing
            @test s.run.full_pair_correlations && s.run.quick_diagnostics && s.run.random_seed==1404
            @test s.mixing.method==:linear && s.mixing.damping==1 && !s.mixing.adaptive
            @test s.run.inherit_sha256==row.seed_sha256==LadderMPSMFT.sha256_file(row.seed)
            @test row.config_sha256==LadderMPSMFT.sha256_file(row.config)
            @test s.run.branch_label==s.run.seed_label=="intertwined_lambda16"
            h5open(row.seed,"r") do f
                @test read(f,"artifact_kind")=="trellis_correlation_seed"
                @test read(f,"spatial_ladders")==2 && !haskey(f,"psi")
                @test read(f,"model_fingerprint")==row.model_fingerprint
                @test read(f,"seed_provenance/fresh_mps") && !read(f,"seed_provenance/source_completed")
                @test read(f,"seed_provenance/recipe_sha256")==POSITIVE_V_RECIPE_SHA
                @test read(f,"seed_provenance/charge_pairing_wavelength_rungs")==16
                @test read(f,"seed_provenance/spin_envelope_wavelength_rungs")==32
                c = LadderMPSMFT._trellis_read_correlations(f["template_correlations"])
                push!(templates,c)
                n = (c.density_up+c.density_down)[1:2:end]
                spin = (c.density_up-c.density_down)[1:2:end]/2 .* (-1.).^(0:63)
                pair = [c.pair[2i-1,2i] for i in 1:64]
                @test mean(n) ≈ .9375 atol=1e-14
                @test n[1:48] ≈ n[17:64] atol=1e-14
                @test spin[1:48] ≈ -spin[17:64] atol=1e-14
                @test all(pair .> 0) && all([c.pair[2i-1,2i+1] for i in 1:63] .< 0)
                @test cor(pair,1 .- n) ≈ 1 atol=1e-14
                @test cor(pair,spin.^2) ≈ -1 atol=1e-14
                rebuilt = trellis_mean_fields([c,c],s.model)
                for (i,name) in enumerate(("A","B"))
                    @test !haskey(f,"ladders/$name/psi")
                    fields = LadderMPSMFT._read_fields(f["ladders/$name/fields"])
                    for key in fieldnames(FieldState)
                        @test getfield(fields,key)==getfield(rebuilt[i],key)
                    end
                    @test norm(fields.alpha)>1e-8
                    @test norm(fields.mu_cdw[2,:]-fields.mu_cdw[1,:])>1e-6
                end
            end
        end
        # Only target couplings change; all four points receive the identical shape.
        @test all(getfield(c,k)==getfield(first(templates),k) for c in templates for k in fieldnames(CorrelationState))
        contract = TOML.parsefile(joinpath(control,"seed_contract.toml"))
        @test contract["branches"]==4 && contract["trellis_cell"]=="two_ladder"
        @test contract["maximum_ladder_solves"]==480 && !contract["interpolated_ep"]
        # Production seed loader: CPU product-state construction only, no DMRG.
        raw = TOML.parsefile(first(rows).config)
        raw["runtime"]["backend"] = "cpu"
        cpu_config = joinpath(directory,"cpu_readback.toml")
        open(io -> TOML.print(io,raw),cpu_config,"w")
        starts = LadderMPSMFT._trellis_initial_states(load_settings(cpu_config))
        @test length(starts)==2 && all(length(x.psi)==128 for x in starts)
        @test starts[1].psi !== starts[2].psi
        # Accidental damping and wrong cells are rejected before creating a campaign.
        raw = TOML.parsefile(base)
        raw["mixing"]["damping"] = .5
        invalid = joinpath(directory,"invalid.toml")
        open(io -> TOML.print(io,raw),invalid,"w")
        @test_throws ArgumentError prepare_trellis_intertwined(invalid,recipe_path,joinpath(directory,"bad"),full,"bad")
        raw = TOML.parsefile(base); raw["model"]["trellis_cell"]="one_ladder"
        open(io -> TOML.print(io,raw),invalid,"w")
        @test_throws ErrorException prepare_trellis_intertwined(invalid,recipe_path,joinpath(directory,"bad"),full,"bad")
    end
end
end
