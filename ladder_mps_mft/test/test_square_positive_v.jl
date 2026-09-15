using Test
include("../scripts/prepare_phase1_square_positive_v.jl")

const POS_ROOT = dirname(@__DIR__)
const POS_BASE = joinpath(POS_ROOT,"configs/phase1_gpu_square_positive_v_chi200_raw60.toml")
const POS_RECIPE = joinpath(POS_ROOT,"data/positive_v_intertwined_recipe.toml")

@testset "intertwined shape and square kernel" begin
    model = load_settings(POS_BASE).model
    recipe = positive_v_recipe(POS_RECIPE)
    @test model.ep_mode==:exact && model.ep_signed ≈ -.15307266912955697
    for wavelength in (8,16)
        c = intertwined_correlations(recipe,model,wavelength)
        @test c.pair==transpose(c.pair)
        @test c.exchange_down==transpose(c.exchange_down)
        @test c.exchange_up==transpose(c.exchange_up)
        @test c.density_up==diag(c.exchange_up) && c.density_down==diag(c.exchange_down)
        @test mean(c.density_down+c.density_up) ≈ .9375 atol=1e-14
        @test sum(c.density_up-c.density_down) ≈ 0 atol=1e-14
        n = (c.density_up+c.density_down)[1:2:end]
        sz = (c.density_up-c.density_down)[1:2:end]/2 .* (-1.).^(0:63)
        rung = [c.pair[2i-1,2i] for i in 1:64]
        leg = [c.pair[2i-1,2i+1] for i in 1:63]
        @test all(rung .> 0) && all(leg .< 0)
        @test cor(rung,1 .- n) ≈ 1 atol=1e-14
        @test cor(rung,sz.^2) ≈ -1 atol=1e-14
        @test n[1:64-wavelength] ≈ n[1+wavelength:64] atol=1e-14
        @test sz[1:64-wavelength] ≈ -sz[1+wavelength:64] atol=1e-14
        @test 2sum(1 .- n[1:wavelength]) ≈ wavelength/8 atol=1e-14
        fields = LadderMPSMFT.mean_fields_from_correlations(c,model)
        g = model.tp^2/model.ep
        @test fields.mu_cdw[2,1] ≈ 2g*(c.density_up[2]-.5)
        @test fields.alpha[1,2,1,1] ≈ 2g*c.pair[4,2]
        @test all(iszero,fields.alpha[:,:,1,2]) && all(iszero,fields.alpha[:,:,2,1])
        @test all(iszero,fields.beta[:,:,:,1,2]) && all(iszero,fields.beta[:,:,:,2,1])
    end
end

@testset "four fresh comparable positive-V preparations" begin
  mktempdir() do preview
    rows=prepare_square_positive_v(POS_BASE,joinpath(POS_ROOT,"data/two_basin_references.h5"),POS_RECIPE,
        joinpath(preview,"control"),joinpath(preview,"full"),"20260915_square_t012_vp02_four_seeds_60")
    @test length(rows)==4 && length(unique(r.label for r in rows))==4
    stripe,pairing=two_basin_references(joinpath(POS_ROOT,"data/two_basin_references.h5"))
    for row in rows
        s=load_settings(row.config)
        @test (s.run.max_iterations,s.convergence.minimum_iterations,s.convergence.stable_iterations)==(60,40,10)
        @test s.dmrg.maxdim==200 && s.model.geometry==:square && (s.model.t0,s.model.V)==(1.2,.2)
        @test s.convergence.channel_noise_floor==5e-7 && s.convergence.field_rel_tol==1e-4
        @test s.convergence.variational_energy_tol==s.dmrg.energy_tol==s.convergence.dmrg_sweep_energy_tol==1e-7
        @test s.mixing.method==:linear && s.mixing.damping==1 && s.convergence.probe_iterations==60
        @test s.run.parent_checkpoint===nothing && s.run.resume_checkpoint===nothing
        @test LadderMPSMFT.sha256_file(row.seed)==s.run.inherit_sha256
        h5open(row.seed,"r") do f
            c=CorrelationState((read(f,"template_correlations/$key") for key in fieldnames(CorrelationState))...)
            fields=LadderMPSMFT.mean_fields_from_correlations(c,s.model)
            readback=read_inherited_fields(row.seed)
            @test all(getfield(readback.fields,key)==getfield(fields,key) for key in fieldnames(FieldState))
            @test !read(f,"seed_provenance/is_scf_solution") && read(f,"seed_provenance/fresh_mps")
            if row.charge_wavelength==0
                expected=row.family=="stripe_weak_other" ? blend_correlations(stripe,pairing) : blend_correlations(pairing,stripe)
                @test all(getfield(expected,key)==getfield(c,key) for key in fieldnames(CorrelationState))
            else
                @test !read(f,"seed_provenance/source_completed")
                @test read(f,"seed_provenance/legacy_source_sha256")==POSITIVE_V_LEGACY_SHA
            end
        end
    end
    for key in (:model_fingerprint,:numerical_fingerprint,:implementation_sha256,:ep_source_sha256)
        @test length(unique(getproperty(r,key) for r in rows))==1
    end
    contract=TOML.parsefile(joinpath(preview,"control/seed_contract.toml"))
    @test contract["branches"]==4 && contract["intertwined_charge_wavelengths"]==[8,16]
    @test_throws ErrorException prepare_square_positive_v(POS_BASE,joinpath(POS_ROOT,"data/two_basin_references.h5"),
        POS_RECIPE,joinpath(preview,"control"),joinpath(preview,"full"),"duplicate")
  end
end
