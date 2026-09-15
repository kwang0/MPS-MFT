using Test, LadderMPSMFT, HDF5, TOML, CSV
include("../scripts/prepare_phase1_square_two_basin_fine_cuts.jl")
include("test_fixed_reference_length.jl")

const FINE_ROOT = dirname(@__DIR__)

@testset "axis-specific pair-binding interpolation" begin
    registry = joinpath(FINE_ROOT,"data/E_p_values.csv")
    kwargs = (L=64,U=8,density=.9375,tp=.1)
    # Keep exact lookup and the existing default t0 interpolation unchanged.
    exact = lookup_ep(registry;kwargs...,t0=1.4,V=0.,interpolation_axis=:V)
    @test exact.mode == :exact && exact.record.E_p == -.14653773091916378
    old = lookup_ep(registry;kwargs...,t0=1.1,V=-.2,allow_interpolation=true)
    @test old.mode == :linear_t0 && old.record.E_p ≈ -.18452309659153343
    @test_throws ArgumentError lookup_ep(registry;kwargs...,t0=1.4,V=-.1)
    @test_throws ArgumentError lookup_ep(registry;kwargs...,t0=1.4,V=-.1,allow_interpolation=true)
    @test_throws ArgumentError lookup_ep(registry;kwargs...,t0=1.4,V=.6,allow_interpolation=true,interpolation_axis=:V)
    @test_throws ArgumentError lookup_ep(registry;kwargs...,t0=.9,V=-.4,allow_interpolation=true)
    @test_throws ArgumentError lookup_ep(registry;kwargs...,t0=1.4,V=0.,interpolation_axis=:U)
    crossing = [EpRecord(4,2.,v,1.,1.,20,-1.,ep,1e-6) for (v,ep) in [(-.2,-.1),(0.,.1)]]
    @test_throws ArgumentError lookup_ep(crossing;L=4,U=2,V=-.1,t0=1.,density=1.,tp=.01,
        allow_interpolation=true,interpolation_axis=:V,require_bound=false)
    before = load_settings(joinpath(FINE_ROOT,"configs/phase1_gpu_square_two_basin_finish20.toml"))
    @test LadderMPSMFT.model_fingerprint(before.model) == "879c440a56028e19184315d58f12f1ee47189b0fadafbccfee60facb63c905a8"
    for spec in TWO_BASIN_FINE_CUTS
        selection=lookup_ep(registry;kwargs...,V=spec.V,t0=spec.t0,allow_interpolation=true,interpolation_axis=spec.axis)
        lo,hi = spec.axis == :V ? (-.2068002629740704,-.14653773091916378) : (-.25124588461187614,-.24962435880865996)
        w=((spec.axis == :V ? spec.V : spec.t0)-spec.lower)/(spec.upper-spec.lower)
        @test selection.record.E_p ≈ (1-w)*lo+w*hi atol=1e-15
        @test selection.denominator == -selection.record.E_p
        @test selection.mode == (spec.axis == :V ? :linear_V : :linear_t0)
        @test selection.lower_record.E_p == lo && selection.upper_record.E_p == hi
        @test selection.interpolation_weight ≈ w
        @test selection.bound_pair && selection.tp_below_pair_binding
    end
end

@testset "six square points and both seeded branches" begin
    mktempdir() do directory
        rows=prepare_square_two_basin_fine_cuts(
            joinpath(FINE_ROOT,"configs/phase1_gpu_square_two_basin_fine_cuts_chi200_raw60.toml"),
            joinpath(FINE_ROOT,"data/two_basin_references.h5"),joinpath(directory,"control"),
            joinpath(directory,"full"),"fine_cuts_test")
        @test length(rows)==12 && length(unique(r.label for r in rows))==12
        @test Set((r.t0,r.V,r.family) for r in rows)==Set((p.t0,p.V,f) for p in TWO_BASIN_FINE_CUTS for f in ("stripe","pairing"))
        for row in rows
            s=load_settings(row.config)
            @test (s.dmrg.maxdim,s.run.max_iterations,s.convergence.minimum_iterations,s.convergence.stable_iterations)==(200,60,40,10)
            @test s.convergence.channel_noise_floor==5e-7 && s.convergence.field_rel_tol==1e-4
            @test s.dmrg.energy_tol==s.convergence.dmrg_sweep_energy_tol==1e-7
            @test s.convergence.variational_energy_tol==1e-7
            @test s.mixing.method==:linear && s.mixing.damping==1 && s.convergence.probe_iterations>=60
            @test s.run.parent_checkpoint===nothing && s.run.resume_checkpoint===nothing
            @test row.ep_mode==String(s.model.ep_mode)
            @test_throws ErrorException validate_raw_basin_contract(s) # old exact-only campaigns remain protected
            provenance=collect_provenance(s)
            @test provenance["ep_mode"]==row.ep_mode
            @test provenance["ep_V_lower"]==s.model.ep_V_lower && provenance["ep_V_upper"]==s.model.ep_V_upper
            h5open(row.seed,"r") do f
                @test read(f,"seed_provenance/ep_mode")==row.ep_mode
                @test read(f,"seed_provenance/ep_interpolation_weight")==row.ep_interpolation_weight
                correlations=CorrelationState((read(f,"template_correlations/$key") for key in fieldnames(CorrelationState))...)
                fields=LadderMPSMFT.mean_fields_from_correlations(correlations,s.model)
                @test read(f,"fields/restart/alpha")==fields.alpha
                @test read(f,"fields/restart/mu_cdw")==fields.mu_cdw
            end
        end
        contract=TOML.parsefile(joinpath(directory,"control/seed_contract.toml"))
        @test contract["interpolated_ep"] && contract["branches"]==12
        output=joinpath(FINE_ROOT,"docs/reports/two_basin_fine_cuts_20260915")
        mkpath(output)
        CSV.write(joinpath(output,"prepared_points.csv"),[
            (t0=r.t0,V=r.V,family=r.family,ep_mode=r.ep_mode,ep_signed=r.ep_signed,
             ep_t0_lower=r.ep_t0_lower,ep_t0_upper=r.ep_t0_upper,
             ep_V_lower=r.ep_V_lower,ep_V_upper=r.ep_V_upper,weight=r.ep_interpolation_weight,
             model_fingerprint=r.model_fingerprint,numerical_fingerprint=r.numerical_fingerprint,
             implementation_sha256=r.implementation_sha256,ep_source_sha256=r.ep_source_sha256) for r in rows])
    end
end
