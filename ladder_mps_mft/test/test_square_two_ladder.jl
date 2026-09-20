using Test, HDF5, ITensors, ITensorMPS, LadderMPSMFT, LinearAlgebra, Random, TOML
include("../scripts/prepare_phase1_square_two_ladder.jl")
const SQUARE_CELL_ROOT = normpath(joinpath(@__DIR__,".."))

function square_cell_model(;L=5,r_range=2,spatial_cell=:two_ladder)
    ModelSettings(;L,U=2.,V=-.2,t0=1.4,tp=.04,density=1.,mu_initial=.5,r_range,
        geometry=:square,spatial_cell,ep=.4,ep_signed=-.4,ep_source="synthetic")
end
function square_cell_correlations(L;seed=1)
    rng = MersenneTwister(seed)
    matrices = [.02randn(rng,2L,2L) for _ in 1:3]
    matrices = [(x+x')/2 for x in matrices]
    dn,up = .5 .+ .05randn(rng,2L), .5 .+ .05randn(rng,2L)
    matrices[2][diagind(matrices[2])] .= dn
    matrices[3][diagind(matrices[3])] .= up
    CorrelationState(matrices...,dn,up)
end
square_corr_add(c,d,h) = CorrelationState((getfield(c,k)+h*getfield(d,k) for k in fieldnames(CorrelationState))...)
function square_cell_energy(cs,m)
    fs = spatial_cell_mean_fields(cs,m)
    sum(variational_energy(0.,0.,fs[i],cs[i],m;bare_ladder_energy=0.).canonical_variational_energy for i in 1:2)
end

@testset "unshifted square bonds, identical-cell reduction and variational derivative" begin
    for range in (0,1,2,4)
        m = square_cell_model(;r_range=range)
        one = square_cell_model(;r_range=range,spatial_cell=:one_ladder)
        cs = [square_cell_correlations(m.L;seed=i) for i in 1:2]
        fs = spatial_cell_mean_fields(cs,m)
        g = 2m.tp^2/m.ep
        for target in 1:2, leg in 0:1, i in 1:m.L
            other = cs[3-target]
            a,b = rung_leg_to_site(i,leg),rung_leg_to_site(i,1-leg)
            @test fs[target].mu_cdw[1,a] ≈ g*(other.density_down[b]-.5)
            @test fs[target].mu_cdw[2,a] ≈ g*(other.density_up[b]-.5)
            for j in 1:m.L
                bp = rung_leg_to_site(j,1-leg)
                @test fs[target].alpha[i,j,leg+1,leg+1] ≈ (abs(i-j)<=range ? g*other.pair[bp,b] : 0.)
                @test fs[target].beta[1,i,j,leg+1,leg+1] ≈ (0<abs(i-j)<=range ? g*other.exchange_down[b,bp] : 0.)
            end
        end
        equal = spatial_cell_mean_fields([cs[1],cs[1]],m)
        old = mean_fields_from_correlations(cs[1],one)
        for f in equal, k in fieldnames(FieldState)
            @test getfield(f,k) == getfield(old,k)
        end
        e1 = variational_energy(0.,0.,old,cs[1],one;bare_ladder_energy=0.).canonical_variational_energy
        @test square_cell_energy([cs[1],cs[1]],m)/(4m.L) ≈ e1/(2m.L)
        @test LadderMPSMFT.model_fingerprint(m) != LadderMPSMFT.model_fingerprint(one)
        ds = [square_cell_correlations(m.L;seed=i+30) for i in 1:2]
        h = 1e-5
        finite = (square_cell_energy([square_corr_add(cs[i],ds[i],h) for i in 1:2],m)-
            square_cell_energy([square_corr_add(cs[i],ds[i],-h) for i in 1:2],m))/(2h)
        derivative = sum(sum(values(field_energy_components(fs[i],ds[i],m))) for i in 1:2)
        @test finite ≈ derivative atol=1e-10 rtol=1e-8
    end
    m = square_cell_model()
    @test_throws DimensionMismatch spatial_cell_mean_fields([square_cell_correlations(m.L)],m)
    # Localized density perturbations do not shift along x or create OBC wrap links.
    base = CorrelationState(zeros(10,10),.5Matrix{Float64}(I,10,10),.5Matrix{Float64}(I,10,10),fill(.5,10),fill(.5,10))
    perturbed = deepcopy(base)
    perturbed.density_up[1] += .1
    perturbed.exchange_up[1,1] += .1
    f = spatial_cell_mean_fields([base,perturbed],m)
    @test findall(!iszero,f[1].mu_cdw) == [CartesianIndex(2,2)]
    @test all(iszero,f[2].mu_cdw)
end

@testset "four immutable square A/B preparations and legacy compatibility" begin
    mktempdir() do dir
        args = (joinpath(SQUARE_CELL_ROOT,"configs/phase1_gpu_square_two_ladder_chi200_raw60.toml"),
            joinpath(SQUARE_CELL_ROOT,"data/two_basin_references.h5"),joinpath(dir,"control"),joinpath(dir,"full"),"test_square_AB")
        rows = prepare_square_two_ladder(args...)
        @test length(rows)==4 && sum(r.spatial_ladders for r in rows)==8
        @test Set(r.V for r in rows)==Set([-.4,-.2])
        @test_throws ErrorException prepare_square_two_ladder(args...)
        stripe,pairing = two_basin_references(args[2])
        shifted = square_shift_seed(stripe,8)
        for k in fieldnames(CorrelationState)
            @test getfield(square_shift_seed(shifted,-8),k)==getfield(stripe,k)
        end
        @test sum(shifted.density_up) ≈ sum(stripe.density_up)
        @test sum(shifted.density_down) ≈ sum(stripe.density_down)
        for row in rows
            s = load_settings(row.config)
            @test s.model.geometry==:square && s.model.spatial_cell==:two_ladder
            @test s.run.quick_diagnostics && s.run.full_pair_correlations
            @test s.run.inherit_sha256==LadderMPSMFT.sha256_file(row.seed)
            @test s.convergence.accepted_periods==[1] && s.mixing.damping==1
            h5open(row.seed,"r") do file
                cs = [LadderMPSMFT._trellis_read_correlations(file["ladders/$name/template_correlations"]) for name in ("A","B")]
                fs = spatial_cell_mean_fields(cs,s.model)
                @test norm(fs[1].mu_cdw-fs[2].mu_cdw)>1e-6
                for (i,name) in enumerate(("A","B")),k in fieldnames(FieldState)
                    @test getfield(fs[i],k)==read(file,"ladders/$name/fields/$k")
                end
                @test cs[1].density_up ≈ (row.family=="stripe" ? .95stripe.density_up+.05pairing.density_up : .95pairing.density_up+.05stripe.density_up)
                @test cs[2].density_up ≈ (row.family=="stripe" ? .95shifted.density_up+.05pairing.density_up : .95pairing.density_up+.05shifted.density_up)
            end
        end
        old = load_settings(joinpath(SQUARE_CELL_ROOT,"configs/phase1_gpu_square_two_basin_chi200_raw.toml"))
        @test old.model.spatial_cell==:one_ladder && old.convergence.accepted_periods==[1,2]
        # Archived trellis models lack the new setting but retain their fingerprint.
        audit = TOML.parsefile(joinpath(SQUARE_CELL_ROOT,"docs/reports/trellis_progress_20260918/same_cell_energy_audit_20260920.toml"))
        for row in audit["runs"]
            path = joinpath(SQUARE_CELL_ROOT,row["source"])
            isfile(path) || continue
            h5open(path,"r") do file
                @test LadderMPSMFT._diagnostic_model(file).spatial_cell==:one_ladder
            end
        end
    end
end

if get(ENV,"SQUARE_CELL_DMRG_SMOKE","0")=="1"
    @testset "tiny square A/B CPU driver, full measurements, storage and resume" begin
        mktempdir() do dir
            model = square_cell_model(;L=2,r_range=1)
            cs = [square_cell_correlations(2;seed=i) for i in 1:2]
            fs = spatial_cell_mean_fields(cs,model)
            seed = joinpath(dir,"seed.h5")
            h5open(seed,"w") do file
                file["artifact_kind"]="spatial_cell_correlation_seed"
                file["model_fingerprint"]=LadderMPSMFT.model_fingerprint(model)
                for (i,name) in enumerate(("A","B"))
                    LadderMPSMFT._write_correlations(create_group(file,"ladders/$name/template_correlations"),cs[i])
                    LadderMPSMFT._write_fields(create_group(file,"ladders/$name/fields"),fs[i])
                end
            end
            dmrg = DMRGSettings(;nsweeps=8,maxdim=16,energy_tol=1e-9,output_level=0,mu_density_tol=1e-5,
                mu_max_iterations=16,max_time_seconds=180.)
            runtime = RuntimeSettings(;conserve_sz=false,conserve_nfparity=false,threaded_blocksparse=false)
            mixing = MixingSettings(;method=:linear,damping=1.,minimum_damping=1.,maximum_damping=1.,adaptive=false)
            convergence = ConvergenceSettings(;minimum_iterations=3,stable_iterations=2,accepted_periods=[1],
                channel_residuals=true,channel_noise_floor=5e-7,dmrg_sweep_energy_tol=1e-9)
            run = RunSettings(;output_directory=dir,inherit_from=seed,inherit_sha256=LadderMPSMFT.sha256_file(seed),
                max_iterations=2,require_accepted_solution=false,quick_diagnostics=true,full_pair_correlations=true)
            settings = ProjectSettings(;model,dmrg,runtime,mixing,convergence,run)
            result = run_scf(settings)
            @test result.diagnostic.status==:maximum_iterations && !result.diagnostic.accepted
            @test length(result.diagnostics_paths)==2
            for path in result.diagnostics_paths
                @test h5read(path,"full_pair_correlations") && h5read(path,"measurement_complete")
                @test h5read(path,"spatial_ladders")==2 && h5read(path,"geometry")=="square"
                @test !h5read(path,"accepted")
            end
            for h in result.records
                @test length(h)==2
                for k in fieldnames(FieldState)
                    @test getfield(h[2].applied,k)==getfield(h[1].measured,k)
                end
                @test all(abs(r.variational.hamiltonian_identity_error)<1e-10 for r in h)
            end
            measured = spatial_cell_mean_fields([last(h).correlations for h in result.records],model)
            compact = joinpath(dir,"compact.h5")
            write_stateless_copy(result.state_path,compact)
            h5open(compact,"r") do f
                @test read(f,"artifact_kind")=="spatial_cell_mps_mft_state"
                @test read(f,"physical_sites")==8 && read(f,"model/spatial_cell")=="two_ladder"
                @test read(f,"model/transverse_geometry")=="square"
                for (i,name) in enumerate(("A","B"))
                    @test !haskey(f,"ladders/$name/psi")
                    @test read(f,"ladders/$name/longitudinal_origin")==0
                    @test size(read(f,"ladders/$name/history/correlations/density_up"))==(4,2)
                    for k in fieldnames(FieldState)
                        @test read(f,"ladders/$name/fields/measured/$k")==getfield(measured[i],k)
                    end
                end
                @test last(read(f,"history/canonical_variational_energy_per_site")) ≈ sum(last(h).variational.canonical_variational_energy for h in result.records)/8
            end
            resume = RunSettings(;output_directory=dir,resume_checkpoint=result.state_path,
                resume_sha256=LadderMPSMFT.sha256_file(result.state_path),max_iterations=1,require_accepted_solution=false)
            resumed = ProjectSettings(;model,dmrg,runtime,mixing,convergence,run=resume)
            starts = LadderMPSMFT._trellis_initial_states(resumed)
            @test length(starts)==2
            for i in 1:2,k in fieldnames(FieldState)
                @test getfield(starts[i].fields,k)==getfield(measured[i],k)
            end
        end
    end
end
