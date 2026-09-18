using Test, HDF5, ITensors, ITensorMPS, LadderMPSMFT, LinearAlgebra, Random, TOML
include("../scripts/prepare_phase1_trellis_comparison.jl")
const TRELLIS_ROOT = normpath(joinpath(@__DIR__, ".."))

function trellis_test_model(; L=5, r_range=2, cell=:one_ladder, tau0=.13, tau1=.08)
    ModelSettings(; L, U=2., t0=1., tau0, tau1, density=1., mu_initial=1., r_range,
        geometry=:trellis, trellis_cell=cell, ep=.4, ep_signed=-.4, ep_source="synthetic")
end

function trellis_test_correlations(L; seed=1, amplitude=.1)
    rng = MersenneTwister(seed)
    matrices = [amplitude * randn(rng, 2L, 2L) for _ in 1:3]
    matrices = [(M + transpose(M))/2 for M in matrices]
    CorrelationState(matrices..., diag(matrices[2]), diag(matrices[3]))
end

trellis_shift(c, d, step) = CorrelationState((getfield(c,k) + step * getfield(d,k) for k in fieldnames(CorrelationState))...)
function trellis_test_energy(corr, model)
    fields = trellis_mean_fields(corr, model)
    sum(variational_energy(0., 0., fields[i], corr[i], model; bare_ladder_energy=0.).canonical_variational_energy for i in eachindex(corr))
end

@testset "trellis paths, projection, square limit and OBC" begin
    model = trellis_test_model(; r_range=4, tau0=.1, tau1=0.)
    corr = trellis_test_correlations(model.L)
    square = ModelSettings(; L=model.L, tp=model.tau0, ep=model.ep, r_range=model.r_range, geometry=:square)
    a, b = only(trellis_mean_fields([corr],model)), mean_fields_from_correlations(corr,square)
    for key in fieldnames(FieldState); @test getfield(a,key) ≈ getfield(b,key); end
    @test_throws ArgumentError density_kernel(:trellis,.1,.4)
    @test_throws DimensionMismatch trellis_mean_fields([corr],trellis_test_model(;cell=:two_ladder))
    model = trellis_test_model(; tau0=.1, tau1=.1)
    out = only(trellis_mean_fields([corr],model))
    for leg in 0:1, i in 1:model.L, j in 1:model.L
        abs(i-j) <= model.r_range || continue
        offsets = leg == 1 ? (0,1) : (0,-1)
        expected_pair, expected_down = 0., 0.
        for di in offsets, dj in offsets
            p,q = i+di,j+dj
            1 <= p <= model.L && 1 <= q <= model.L && abs(p-q) <= model.r_range || continue
            a,b = rung_leg_to_site(p,1-leg),rung_leg_to_site(q,1-leg)
            expected_pair += 2*.1^2/model.ep*corr.pair[b,a]
            expected_down += 2*.1^2/model.ep*(corr.exchange_down[a,b] - .5*(a==b))
        end
        @test out.alpha[i,j,leg+1,leg+1] ≈ expected_pair atol=1e-14
        actual = i==j ? out.mu_cdw[1,rung_leg_to_site(i,leg)] : out.beta[1,i,j,leg+1,leg+1]
        @test actual ≈ expected_down atol=1e-14
    end
    @test all(iszero,out.alpha[:,:,1,2]) && all(iszero,out.beta[:,:,:,2,1])
    zero = CorrelationState(zeros(10,10),zeros(10,10),zeros(10,10),zeros(10),zeros(10))
    out = only(trellis_mean_fields([zero],model))
    @test out.beta[1,2,3,1,1] ≈ -.1^2/model.ep
    @test out.mu_cdw[1,1] ≈ -.1^2/model.ep
    @test out.mu_cdw[1,3] ≈ -2*.1^2/model.ep
end

@testset "reciprocity and variational derivative, including affine normal fields" begin
    for cell in (:one_ladder,:two_ladder), range in (0,1,2,4)
        model = trellis_test_model(; cell, r_range=range)
        count = LadderMPSMFT.trellis_ladders(model)
        c = [trellis_test_correlations(model.L;seed=i) for i in 1:count]
        d = [trellis_test_correlations(model.L;seed=i+20) for i in 1:count]
        epsilon = 1e-5
        plus = trellis_test_energy([trellis_shift(c[i],d[i],epsilon) for i in 1:count],model)
        minus = trellis_test_energy([trellis_shift(c[i],d[i],-epsilon) for i in 1:count],model)
        fields = trellis_mean_fields(c,model)
        derivative = sum(sum(values(field_energy_components(fields[i],d[i],model))) for i in 1:count)
        @test (plus-minus)/(2epsilon) ≈ derivative atol=1e-9 rtol=1e-8
        # Off-range input must not leak into the retained fields.
        far = zeros(2model.L,2model.L)
        for a in 1:2model.L, b in 1:2model.L
            abs(first(site_to_rung_leg(a))-first(site_to_rung_leg(b))) > range && (far[a,b]=1.)
        end
        delta = CorrelationState(far,far,far,zeros(2model.L),zeros(2model.L))
        other = trellis_mean_fields([trellis_shift(x,delta,1.) for x in c],model)
        for i in 1:count, key in fieldnames(FieldState)
            @test getfield(other[i],key) ≈ getfield(fields[i],key) atol=1e-14
        end
    end
    # In the symmetric rectangular cell A uses forward paths on both legs,
    # B backward paths; they are spatially distinct even for equal templates.
    model = trellis_test_model(;cell=:two_ladder,tau0=.1,tau1=.1)
    c = trellis_test_correlations(model.L)
    f = trellis_mean_fields([c,c],model)
    single = only(trellis_mean_fields([c],trellis_test_model(;tau0=.1,tau1=.1)))
    @test f[1].alpha[:,:,2,2] == single.alpha[:,:,2,2]
    @test f[2].alpha[:,:,1,1] == single.alpha[:,:,1,1]
    @test f[1].alpha[:,:,1,1] != f[2].alpha[:,:,1,1]
end

@testset "four immutable branches, matched reference seeds and fingerprints" begin
    mktempdir() do dir
        args = (joinpath(TRELLIS_ROOT,"configs/phase1_gpu_trellis_chi200_raw60.toml"),
            joinpath(TRELLIS_ROOT,"data/two_basin_references.h5"),joinpath(dir,"control"),joinpath(dir,"full"),"test_trellis")
        rows = prepare_trellis_comparison(args...)
        @test length(rows)==4 && sum(row.spatial_ladders for row in rows)==6
        @test rows[1].model_fingerprint==rows[2].model_fingerprint != rows[3].model_fingerprint==rows[4].model_fingerprint
        @test length(unique(row.numerical_fingerprint for row in rows))==1
        @test_throws ErrorException prepare_trellis_comparison(args...)
        for (one,two) in ((rows[1],rows[3]),(rows[2],rows[4]))
            h5open(one.seed,"r") do a
                h5open(two.seed,"r") do b
                    for key in fieldnames(CorrelationState)
                        @test read(a,"template_correlations/$key")==read(b,"template_correlations/$key")
                    end
                end
            end
        end
        old = load_settings(joinpath(TRELLIS_ROOT,"configs/phase1_gpu_square_two_basin_fine_cuts_chi200_raw60.toml"))
        @test old.model.geometry==:square && old.convergence.accepted_periods==[1,2]
    end
end

@testset "spatial A/B stationarity is distinct from temporal period two" begin
    model = trellis_test_model(;cell=:two_ladder)
    zero = LadderMPSMFT.zero_field_state(model)
    other = copy(zero)
    other.mu_cdw[1,1] = .03
    corr = trellis_test_correlations(model.L)
    energy = variational_energy(0.,0.,zero,corr,model;bare_ladder_energy=0.)
    record(i,applied,measured) = begin
        absolute,relative = LadderMPSMFT.hybrid_distance(measured,applied)
        IterationRecord(;iteration=i,update_mode=:unmixed_probe,applied,measured,correlations=corr,
            density=1.,chemical_potential=0.,mu_search_status=:density_tolerance,mu_evaluations=1,
            mu_density_converged=true,effective_energy=0.,variational=energy,
            field_abs_residual=absolute,field_rel_residual=relative,wall_seconds=.1)
    end
    convergence = ConvergenceSettings(;accepted_periods=[1],stable_iterations=2,minimum_iterations=2)
    settings = ProjectSettings(;model,convergence)
    stationary = [[record(i,zero,zero) for i in 1:8],[record(i,other,other) for i in 1:8]]
    diagnostic,_ = LadderMPSMFT._trellis_cell_diagnostic(stationary,settings)
    @test diagnostic.accepted && diagnostic.fundamental_period==1
    alternating = [record(i,isodd(i) ? zero : other,isodd(i) ? other : zero) for i in 1:8]
    diagnostic,members = LadderMPSMFT._trellis_cell_diagnostic([stationary[1],alternating],settings)
    @test !diagnostic.accepted && diagnostic.fundamental_period==0
    @test members[1].accepted && !members[2].accepted
end

if get(ENV,"TRELLIS_DMRG_SMOKE","0") == "1"
    @testset "tiny CPU spatial-cell driver, storage, compaction and complete-cell resume" begin
        mktempdir() do dir
            for cell in (:one_ladder,:two_ladder)
                model = trellis_test_model(;L=2,r_range=1,cell,tau0=.04,tau1=.04)
                count = LadderMPSMFT.trellis_ladders(model)
                corr = trellis_test_correlations(2;amplitude=.02)
                seed = joinpath(dir,"$(cell)_seed.h5")
                fields = trellis_mean_fields(fill(corr,count),model)
                h5open(seed,"w") do file
                    file["artifact_kind"]="trellis_correlation_seed"
                    file["model_fingerprint"]=LadderMPSMFT.model_fingerprint(model)
                    LadderMPSMFT._write_correlations(create_group(file,"template_correlations"),corr)
                    for i in 1:count
                        LadderMPSMFT._write_fields(create_group(file,"ladders/$(i==1 ? "A" : "B")/fields"),fields[i])
                    end
                end
                dmrg = DMRGSettings(;nsweeps=8,maxdim=16,energy_tol=1e-9,output_level=0,mu_density_tol=1e-5,
                    mu_max_iterations=16,max_time_seconds=180.)
                runtime = RuntimeSettings(;conserve_sz=false,conserve_nfparity=false,threaded_blocksparse=false)
                mixing = MixingSettings(;method=:linear,damping=1.,minimum_damping=1.,maximum_damping=1.,adaptive=false)
                convergence = ConvergenceSettings(;minimum_iterations=3,stable_iterations=2,accepted_periods=[1],
                    channel_residuals=true,channel_noise_floor=5e-7,dmrg_sweep_energy_tol=1e-9)
                run = RunSettings(;output_directory=dir,inherit_from=seed,inherit_sha256=LadderMPSMFT.sha256_file(seed),
                    max_iterations=2,require_accepted_solution=false,quick_diagnostics=true)
                settings = ProjectSettings(;model,dmrg,runtime,mixing,convergence,run)
                result = run_scf(settings)
                @test result.diagnostic.status==:maximum_iterations && !result.diagnostic.accepted
                @test length(result.diagnostics_paths)==count
                @test all(h5read(p,"full_pair_correlations") for p in result.diagnostics_paths)
                @test all(!h5read(p,"accepted") for p in result.diagnostics_paths)
                @test length(result.records)==count
                for history in result.records
                    @test length(history)==2
                    for key in fieldnames(FieldState)
                        @test getfield(history[2].applied,key)==getfield(history[1].measured,key)
                    end
                    @test all(abs(r.variational.hamiltonian_identity_error)<1e-10 for r in history)
                end
                measured = trellis_mean_fields([last(h).correlations for h in result.records],model)
                for i in 1:count, key in fieldnames(FieldState)
                    @test getfield(last(result.records[i]).measured,key) == getfield(measured[i],key)
                end
                compact = joinpath(dir,"$(cell)_compact.h5")
                write_stateless_copy(result.state_path,compact)
                h5open(compact,"r") do file
                    @test read(file,"physical_sites")==4count
                    for i in 1:count
                        name = i==1 ? "A" : "B"
                        @test !haskey(file,"ladders/$name/psi")
                        @test size(read(file,"ladders/$name/history/correlations/density_down"))==(4,2)
                    end
                    @test read(file,"history/canonical_variational_energy_per_site")[end] ≈
                        sum(last(h).variational.canonical_variational_energy for h in result.records)/(4count)
                end
                resume = RunSettings(;output_directory=dir,resume_checkpoint=result.state_path,
                    resume_sha256=LadderMPSMFT.sha256_file(result.state_path),max_iterations=1,require_accepted_solution=false)
                resumed_settings = ProjectSettings(;model,dmrg,runtime,mixing,convergence,run=resume)
                starts = LadderMPSMFT._trellis_initial_states(resumed_settings)
                @test length(starts)==count
                for i in 1:count, key in fieldnames(FieldState)
                    @test getfield(starts[i].fields,key)==getfield(last(result.records[i]).measured,key)
                end
                @test_throws ArgumentError LadderMPSMFT._write_trellis_checkpoint(result.state_path;settings,
                    psis=getfield.(starts,:psi),histories=result.records,diagnostic=result.diagnostic,members=ConvergenceDiagnostic[],
                    restart_fields=measured,provenance=Dict(),immutable=true)
            end
        end
    end
end
