module RawBasinTests
using Test, HDF5, ITensorMPS, LadderMPSMFT, LinearAlgebra
include("../scripts/compare_two_basin_grid.jl")
include("../scripts/prepare_phase1_two_basin_grid.jl")

function fields(pairing, spin)
    mu = vcat(fill(-spin, 1, 4), fill(spin, 1, 4))
    FieldState(fill(pairing, 2, 2, 2, 2), zeros(2, 2, 2, 2, 2), mu)
end

function record(i, x=fields(1., 0.), y=x; energy=0., sweeps=[-1., -1. - 5e-9])
    energies = (; (key => Float64(energy) for key in fieldnames(EnergyBreakdown))...)
    energy_record = EnergyBreakdown(; merge(energies, (hamiltonian_identity_error=0.,
        effective_eigenvalue_error=0., variational_consistency_error=0.))...)
    absolute, relative = LadderMPSMFT.hybrid_distance(x, y)
    IterationRecord(iteration=i, applied=x, measured=y, update_mode=:unmixed_probe,
        correlations=CorrelationState(zeros(4,4), zeros(4,4), zeros(4,4), fill(.5,4), fill(.5,4)),
        density=1., chemical_potential=0., mu_search_status=:density_tolerance, mu_evaluations=1,
        mu_density_converged=true, effective_energy=energy, variational=energy_record,
        field_abs_residual=absolute, field_rel_residual=relative, wall_seconds=.1,
        dmrg_sweep_energies=sweeps, dmrg_sweep_max_discarded_weights=fill(1e-9,length(sweeps)),
        dmrg_sweep_maxlinkdims=fill(2,length(sweeps)))
end

@testset "raw basin convergence and histories" begin
    settings = ConvergenceSettings(channel_residuals=true, minimum_iterations=50,
        stable_iterations=5, field_abs_tol=1e-7, field_rel_tol=1e-4,
        variational_energy_tol=1e-8, dmrg_sweep_energy_tol=1e-8, probe_iterations=80,
        period_abs_tol=2e-7, period_rel_tol=2e-4)
    stable = [record(i) for i in 1:50]
    @test !assess_convergence(stable[1:49], settings, 1.).accepted
    @test assess_convergence(stable, settings, 1.).accepted
    # A weak growing spin field can hide behind the contracting pairing residual.
    growing = [record(i, fields(1., 1e-5 * 1.1^i),
                      fields(1. + .001 * .5^i, 1e-5 * 1.1^(i+1))) for i in 1:5]
    @test assess_convergence(growing, ConvergenceSettings(), 1.).accepted
    @test !assess_convergence(growing, ConvergenceSettings(channel_residuals=true), 1.).accepted
    spin = only(filter(row -> row.name == :spin, LadderMPSMFT.channel_diagnostics(growing, settings)))
    @test spin.contraction ≈ 1.1
    @test isinf(spin.factor) && !spin.passes
    below_floor = [record(i, fields(1., 1e-9 * 1.1^i), fields(1., 1e-9 * 1.1^(i+1))) for i in 1:5]
    @test all(row -> row.passes, LadderMPSMFT.channel_diagnostics(below_floor, settings))
    @test all(row -> row.passes && isfinite(row.relative), LadderMPSMFT.channel_diagnostics(stable, settings))
    bad_dmrg = copy(stable); bad_dmrg[end] = record(50; sweeps=[-1., -1.01])
    @test !assess_convergence(bad_dmrg, settings, 1.).accepted
    missing_dmrg = copy(stable); missing_dmrg[end] = record(50; sweeps=Float64[])
    @test !assess_convergence(missing_dmrg, settings, 1.).accepted
    drifting_energy = copy(stable); drifting_energy[end-3] = record(47; energy=1e-5)
    @test !assess_convergence(drifting_energy, settings, 1.).accepted
    @test assess_convergence(drifting_energy, ConvergenceSettings(), 1.).accepted

    # Channel recurrence catches drift that a large pairing background hides.
    orbit = [record(i, fields(1., isodd(i) ? .01 : -.01),
                       fields(1., isodd(i) ? -.01 : .01)) for i in 1:50]
    @test assess_convergence(orbit, settings, 1.).accepted
    @test assess_convergence(orbit, settings, 1.).fundamental_period == 2
    @test !assess_convergence(orbit[1:20], settings, 1.).accepted
    bad_orbit = copy(orbit)
    bad_orbit[end] = record(50, fields(1., -.01), fields(1., .01001))
    @test !LadderMPSMFT._channel_orbit_pass(bad_orbit, settings, 2)

    base = load_settings(joinpath(@__DIR__, "../configs/phase1_gpu_square_two_basin_chi200_raw.toml"))
    @test base.dmrg.maxdim == 200
    @test base.convergence.probe_iterations >= base.run.max_iterations
    @test base.mixing.method == :linear && !base.mixing.adaptive
    @test base.mixing.damping == base.mixing.minimum_damping == base.mixing.maximum_damping == 1
    state = LadderMPSMFT.MixingState(base.mixing)
    for i in 1:5
        y, metadata = mix_fields!(state, fields(1., i*.01), fields(.5, i*.02), base.mixing)
        @test y.alpha == fields(.5, i*.02).alpha
        @test y.mu_cdw == fields(.5, i*.02).mu_cdw
        @test metadata.method == :linear
    end
    candidate = ConvergenceDiagnostic(status=:periodic_candidate, unmixed_probe=true)
    for i in (20, 50, 79)
        @test LadderMPSMFT._recurrence_action(candidate, base.convergence;
            probe_steps=i, probe_origin=:initial, mixer_probe_completed=false) == :continue_probe
    end

    # Check the actual storage path without running DMRG.
    model = ModelSettings(L=2, U=2., density=1., r_range=1, ep=.2, ep_signed=-.2)
    project = ProjectSettings(model=model, convergence=settings,
        runtime=RuntimeSettings(conserve_sz=false, conserve_nfparity=false))
    psi = MPS(siteinds("Electron", 4), ["Up", "Dn", "Up", "Dn"])
    mktempdir() do directory
        path = joinpath(directory, "state.h5")
        write_checkpoint(path; settings=project, psi, records=stable,
            diagnostic=assess_convergence(stable, settings, 1.), provenance=Dict())
        h5open(path, "r") do f
            @test read(f, "history/target_density_corrected_variational_energy") == zeros(50)
            @test length(keys(f["history/channels"])) == 6
            @test all(read(f, "history/channels/spin/passes"))
            @test all(read(f, "history/dmrg_sweep_gate_pass"))
            @test size(read(f, "history/fields/measured/alpha"), 5) == 50
        end
    end

    # Affine density kernels require weights summing to one. Check all target
    # field components rather than transplanting coupling-weighted sources.
    primary = CorrelationState(fill(.1,4,4), fill(.2,4,4), fill(.3,4,4), fill(.4,4), fill(.5,4))
    competing = CorrelationState(fill(.5,4,4), fill(.4,4,4), fill(.3,4,4), fill(.2,4), fill(.1,4))
    combined = mean_fields_from_correlations(blend_correlations(primary, competing), model)
    a = mean_fields_from_correlations(primary, model)
    b = mean_fields_from_correlations(competing, model)
    for key in (:alpha, :beta, :mu_cdw)
        @test getfield(combined, key) ≈ .95 .* getfield(a, key) .+ .05 .* getfield(b, key)
    end

    mktempdir() do directory
        provenance = Dict("model_fingerprint" => "fixture_model", "numerical_fingerprint" => "fixture_numerics",
            "implementation_sha256" => "fixture_implementation", "ep_source_sha256" => "fixture_registry")
        entries = [(label=family, family=family, t0=1.4, V=0.,
                    model_fingerprint="fixture_model", numerical_fingerprint="fixture_numerics",
                    implementation_sha256="fixture_implementation", ep_source_sha256="fixture_registry")
                   for family in ("stripe", "pairing")]
        CSV.write(joinpath(directory, "manifest.tsv"), entries; delim='\t')
        output = joinpath(directory, "comparison.csv")
        @test only(compare_two_basin_grid(directory, output)).status == "awaiting_results"
        for family in ("stripe", "pairing")
            records = [record(i; energy=family=="stripe" ? -.004 : 0.) for i in 1:50]
            path = joinpath(directory, "results", family, "20300101T000000_test", "state.h5")
            write_checkpoint(path; settings=project, psi, records,
                diagnostic=assess_convergence(records, settings, 1.),
                provenance=merge(provenance, Dict("branch_label" => family)))
        end
        result = only(compare_two_basin_grid(directory, output))
        @test result.lower_energy_seed_family == "stripe"
        @test result.delta_stripe_minus_pairing_per_site ≈ -.001
        # A later unaccepted endpoint must not inherit an earlier winner.
        later = joinpath(directory, "results", "stripe", "20300102T000000_test", "state.h5")
        write_checkpoint(later; settings=project, psi, records=stable,
            diagnostic=ConvergenceDiagnostic(status=:maximum_iterations),
            provenance=merge(provenance, Dict("branch_label" => "stripe")))
        @test only(compare_two_basin_grid(directory, output)).status == "not_rankable"
        # Reject missing/mismatched provenance even if acceptance is true.
        write_checkpoint(later; settings=project, psi, records=stable,
            diagnostic=assess_convergence(stable, settings, 1.),
            provenance=merge(provenance, Dict("model_fingerprint" => "wrong", "branch_label" => "stripe")))
        @test only(compare_two_basin_grid(directory, output)).status == "not_rankable"
        write_checkpoint(later; settings=project, psi, records=stable,
            diagnostic=assess_convergence(stable, settings, 1.),
            provenance=merge(provenance, Dict("branch_label" => "stripe")))
        @test only(compare_two_basin_grid(directory, output)).status == "energy_unresolved"
    end
end
end
