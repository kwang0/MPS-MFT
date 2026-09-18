using Test, LadderMPSMFT, ITensors, ITensorMPS, HDF5, LinearAlgebra, Random
const D = LadderMPSMFT

@testset "complete equal-time measurements" begin
    Random.seed!(81)
    model = ModelSettings(L=2, r_range=1, ep=.1, ep_signed=-.1, density=1.)
    sites = siteinds("Electron", 4; conserve_qns=false)
    psi = random_mps(ComplexF64, sites; linkdims=4)
    normalize!(psi)
    diagnostics = compute_ladder_diagnostics(psi, model)
    pairs = diagnostics.pair_correlations
    @test length(pairs.basis_class) == 8
    @test pairs.removal ≈ pairs.removal' atol=1e-11
    @test minimum(eigvals(Hermitian(pairs.removal_connected))) > -1e-10
    @test minimum(eigvals(Hermitian(pairs.addition_connected))) > -1e-10
    for i in eachindex(pairs.basis_class), j in eachindex(pairs.basis_class)
        # Independent OpSum contractions test mixed bond classes, overlapping
        # supports, onsite normalization and complex conjugation.
        a, b = pairs.basis_site1[i], pairs.basis_site2[i]
        c, d = pairs.basis_site1[j], pairs.basis_site2[j]
        terms(a,b) = a == b ? [(1., "Cup", a, "Cdn", b)] :
            [(1., "Cup", a, "Cdn", b), (-1., "Cdn", a, "Cup", b)]
        adjoint_op = Dict("Cup"=>"Cdagup", "Cdn"=>"Cdagdn")
        os = OpSum()
        for l in terms(a,b), r in terms(c,d)
            add!(os, l[1]*r[1], l[2],l[3],l[4],l[5],adjoint_op[r[4]],r[5],adjoint_op[r[2]],r[3])
        end
        @test pairs.addition[i,j] ≈ inner(psi', MPO(os,sites), psi) atol=1e-10
    end
    @test diagnostics.charge_connected ≈ diagnostics.charge_correlation - diagnostics.density * diagnostics.density'
    @test diagnostics.spin_connected ≈ diagnostics.spin_correlation - diagnostics.spin * diagnostics.spin'
    @test all(0 .<= diagnostics.double_occupancy .<= 1)
    # A number-conserving state must still support measurements of the
    # symmetry-forbidden one-point anomalous/transverse expectations (zero).
    qsites = siteinds("Electron",4;conserve_qns=true)
    qpsi = MPS(qsites,["Up","Dn","Up","Dn"])
    qd = compute_ladder_diagnostics(qpsi, model)
    @test all(iszero, qd.pair_correlations.expectation)
    @test all(iszero, qd.spin_plus)

    mktempdir() do dir
        function fixture(path; trellis=false, accepted=false, status="maximum_iterations", period=0)
            m = trellis ? ModelSettings(L=2,r_range=1,ep=.1,ep_signed=-.1,density=1.,geometry=:trellis,trellis_cell=:two_ladder) : model
            h5open(path,"w") do f
                f["artifact_kind"] = trellis ? "trellis_mps_mft_state" : "ladder_mps_mft_state"
                f["accepted"]=accepted; f["status"]=status
                f["solution_kind"]=accepted ? "fixed_point" : "none"
                f["fundamental_period"]=period
                D._write_dict(create_group(f,"model"), Dict(String(k)=>getfield(m,k) for k in fieldnames(ModelSettings)))
                f["provenance/model_fingerprint"]=D.model_fingerprint(m)
                if trellis
                    f["spatial_ladders"]=2
                    for name in ("A","B")
                        f["ladders/$name/psi"]=psi
                        f["ladders/$name/history/iteration"]=[1,2]
                    end
                else
                    f["psi"]=psi; f["history/iteration"]=[1,2]
                end
            end
        end
        source=joinpath(dir,"state.h5"); fixture(source)
        sha=D.sha256_file(source)
        @test_throws ErrorException measure_state_diagnostics(source)
        paths=measure_state_diagnostics(source;allow_unaccepted=true,expected_sha256=sha)
        @test D.sha256_file(source)==sha
        h5open(only(paths),"r") do f
            @test !read(f,"accepted")
            @test read(f,"status")=="maximum_iterations"
            @test read(f,"period")==0
            @test read(f,"sample_kind")=="terminal_snapshot"
            @test read(f,"state_sha256")==sha
            @test haskey(f,"pair_correlations/removal_connected")
            @test haskey(f,"spin_correlation")
            @test read(f,"measurement_complete")
        end
        @test_throws ErrorException measure_state_diagnostics(source;allow_unaccepted=true)
        @test measure_state_diagnostics(source;allow_unaccepted=true,reuse=true)==paths
        @test_throws ErrorException measure_state_diagnostics(source;allow_unaccepted=true,expected_sha256="wrong")
        compact=joinpath(dir,"compact.h5"); write_stateless_copy(source,compact)
        @test_throws ErrorException measure_state_diagnostics(compact;allow_unaccepted=true)
        tsource=joinpath(dir,"trellis.h5"); fixture(tsource;trellis=true)
        tpaths=measure_state_diagnostics(tsource;output_directory=joinpath(dir,"trellis"),allow_unaccepted=true)
        @test length(tpaths)==2
        @test [h5read(p,"spatial_ladder") for p in tpaths]==["A","B"]
        @test all(h5read(p,"period")==0 for p in tpaths)
    end
    @test RunSettings().full_pair_correlations
    @test D.terminal_diagnostics_enabled(ConvergenceDiagnostic(status=:maximum_iterations),RunSettings())
    @test !D.terminal_diagnostics_enabled(ConvergenceDiagnostic(status=:time_limit),RunSettings())
end

@testset "automatic max-iteration diagnostics" begin
    mktempdir() do directory
        model=ModelSettings(L=2,U=2.,density=1.,mu_initial=1.,r_range=1,ep=.4,ep_signed=-.4,geometry=:square)
        settings=ProjectSettings(;model,
            dmrg=DMRGSettings(nsweeps=4,maxdim=16,output_level=0,mu_max_iterations=8,max_time_seconds=180.),
            mixing=MixingSettings(method=:linear,damping=1.,minimum_damping=1.,maximum_damping=1.,adaptive=false),
            convergence=ConvergenceSettings(minimum_iterations=3,accepted_periods=[1]),
            runtime=RuntimeSettings(conserve_sz=false,conserve_nfparity=false,threaded_blocksparse=false),
            run=RunSettings(output_directory=directory,max_iterations=1,require_accepted_solution=false))
        result=run_scf(settings)
        @test result.diagnostic.status==:maximum_iterations
        @test !result.diagnostic.accepted
        @test length(result.diagnostics_paths)==1
        @test h5read(only(result.diagnostics_paths),"full_pair_correlations")
        @test !h5read(result.state_path,"accepted")
    end
end
