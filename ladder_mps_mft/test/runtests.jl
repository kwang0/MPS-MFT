using Test
using HDF5
using ITensorMPS
using LadderMPSMFT
using LinearAlgebra
using Random
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))

function test_model(; geometry=:cubic_frustrated)
    return ModelSettings(;
        L=2,
        U=2.0,
        t0=1.0,
        tp=0.1,
        density=1.0,
        r_range=1,
        geometry,
        ep=0.2,
        ep_signed=-0.2,
        ep_source="synthetic",
    )
end

function test_fields(value=0.0)
    alpha = zeros(2, 2, 2, 2)
    beta = zeros(2, 2, 2, 2, 2)
    mu = zeros(2, 4)
    mu[1, 1] = value
    return FieldState(alpha, beta, mu)
end

function test_correlations()
    return CorrelationState(zeros(4, 4), zeros(4, 4), zeros(4, 4), fill(0.5, 4), fill(0.5, 4))
end

function test_energy(value=0.0)
    return EnergyBreakdown(
        effective_eigenvalue=value,
        effective_expectation=value,
        effective_eigenvalue_error=0.0,
        bare_ladder_energy=value,
        reconstructed_bare_ladder_energy=value,
        hamiltonian_identity_error=0.0,
        chemical_potential_term=0.0,
        pair_field_energy=0.0,
        exchange_field_energy=0.0,
        density_field_energy=0.0,
        pair_transverse_energy=0.0,
        exchange_transverse_energy=0.0,
        density_transverse_energy=0.0,
        double_counting_correction=0.0,
        reconstructed_variational_energy=value,
        direct_variational_energy=value,
        variational_consistency_error=0.0,
        canonical_variational_energy=value,
        grand_potential=value,
    )
end

function test_record(iteration, applied, measured; energy=0.0, density=1.0, update_mode=:unknown, correlations=test_correlations())
    absolute, relative = LadderMPSMFT.hybrid_distance(measured, applied)
    return IterationRecord(;
        iteration=iteration,
        update_mode,
        applied,
        measured,
        correlations,
        density,
        chemical_potential=0.0,
        mu_search_status=:density_tolerance,
        mu_evaluations=1,
        mu_density_converged=true,
        effective_energy=energy,
        variational=test_energy(energy),
        field_abs_residual=absolute,
        field_rel_residual=relative,
        wall_seconds=0.1,
    )
end

@testset "geometry and E_p registry" begin
    @test normalize_geometry("cubic-frustrated") == :cubic_frustrated
    @test normalize_geometry(:square) == :square
    @test_throws ArgumentError normalize_geometry(:triangular)
    @test density_kernel(:cubic_frustrated, 0.1, 0.2) ≈ [0.2 0.1; 0.1 0.2]
    @test density_kernel(:cubic_unfrustrated, 0.1, 0.2) ≈ [0.0 0.3; 0.3 0.0]
    @test density_kernel(:square, 0.1, 0.2) ≈ [0.0 0.1; 0.1 0.0]

    registry = joinpath(ROOT, "data", "E_p_values.csv")
    selection = lookup_ep(registry; L=64, U=8, V=0, t0=1, density=0.9375, tp=0.1)
    @test selection.record.chi == 1000
    @test selection.record.E_p ≈ -0.13251724
    @test selection.denominator ≈ 0.13251724
    @test selection.bound_pair
    @test selection.tp_below_pair_binding
    @test_throws ArgumentError lookup_ep(registry; L=64, U=8, V=0, t0=2, density=0.9375, tp=0.1)
end

@testset "configuration and deterministic seeds" begin
    settings = load_settings(joinpath(ROOT, "configs", "phase0_timing.toml"))
    @test settings.model.geometry == :cubic_frustrated
    @test settings.model.ep_signed < 0
    @test settings.run.initial_seed == :pairing
    @test settings.runtime.backend == :cpu
    @test settings.convergence.unmixed_cycle_probe
    @test settings.convergence.accepted_periods == [1, 2]
    @test settings.convergence.orbit_bulk_fraction == 0.5
    @test settings.model.mu_initial == 0.0
    @test settings.dmrg.mu_density_tol == 5e-4
    @test settings.dmrg.mu_max_iterations == 16
    @test settings.dmrg.mu_bracket_step == 0.05
    @test !settings.run.require_accepted_solution
    first_seed = initial_fields(test_model(); seed=:pairing, rng=MersenneTwister(7))
    second_seed = initial_fields(test_model(); seed=:pairing, rng=MersenneTwister(7))
    @test first_seed.alpha == second_seed.alpha
    @test first_seed.alpha == permutedims(first_seed.alpha, (2, 1, 4, 3))
    @test all(iszero, first_seed.beta)
end

@testset "Phase 0 run environment round trip" begin
    script = joinpath(ROOT, "slurm", "phase0_calibrate_cpu.sh")
    mktempdir() do directory
        command = `bash -c 'source "$1" plan >/dev/null; write_environment "$2/run.env"; load_environment "$2"; printf "%s\n" "$PHASE0_RUN_SCRIPT_VERSION"' bash $script $directory`
        loaded_version = read(command, String)
        environment = read(joinpath(directory, "run.env"), String)
        @test occursin(r"(?m)^PHASE0_RUN_SCRIPT_VERSION=1\.3\.1$", environment)
        @test !occursin(r"(?m)^PHASE0_SCRIPT_VERSION=", environment)
        @test strip(loaded_version) == "1.3.1"

        compatible_environment = replace(environment,
            "PHASE0_RUN_SCRIPT_VERSION=1.3.1" => "PHASE0_RUN_SCRIPT_VERSION=1.3.0",
        )
        write(joinpath(directory, "run.env"), compatible_environment)
        compatible_version = read(
            `bash -c 'source "$1" plan >/dev/null; load_environment "$2"; printf "%s\n" "$PHASE0_RUN_SCRIPT_VERSION"' bash $script $directory`,
            String,
        )
        @test strip(compatible_version) == "1.3.0"

        script_source = read(script, String)
        @test occursin("submit_matrix_jobs \"\$run_dir\" \"\$seed_id\" pending", script_source)
        @test occursin("submit_matrix_jobs \"\$run_dir\" \"\$seed_id\" completed", script_source)
    end
end

@testset "fixed-mu Phase 0 payload and report" begin
    config = joinpath(ROOT, "test", "fixtures", "phase0_tiny.toml")
    seed_script = joinpath(ROOT, "scripts", "phase0_prepare_seed.jl")
    payload_script = joinpath(ROOT, "scripts", "phase0_payload.jl")
    report_script = joinpath(ROOT, "scripts", "phase0_report.jl")
    julia = Base.julia_cmd()
    mktempdir() do directory
        seed_path = joinpath(directory, "seed_state.h5")
        run(pipeline(
            `$julia --startup-file=no --project=$ROOT $seed_script $config $seed_path`,
            stdout=devnull,
        ))
        seed = h5open(seed_path, "r") do file
            return (
                schema=Int(read(file, "schema_version")),
                benchmark_kind=String(read(file, "benchmark_kind")),
                density=Float64(read(file, "density")),
                target=Float64(read(file, "target_density")),
                chemical_potential=Float64(read(file, "chemical_potential")),
                maximum_bond_dimension=Int(read(file, "maximum_bond_dimension")),
            )
        end
        @test seed.schema == 3
        @test seed.benchmark_kind == "fixed_mu_dmrg"
        @test 0.0 <= seed.density <= 2.0
        @test isfinite(seed.chemical_potential)
        @test 1 <= seed.maximum_bond_dimension <= 4

        metrics_directory = joinpath(directory, "metrics")
        mkpath(metrics_directory)
        metric_path = joinpath(metrics_directory, "serial-t1.toml")
        payload_command = addenv(
            `$julia --startup-file=no --project=$ROOT $payload_script $config $metric_path serial-t1 serial`,
            "PHASE0_SEED_STATE" => seed_path,
            "PHASE0_REPETITIONS" => "1",
            "PHASE0_COMPILE_WARMUP" => "0",
        )
        run(pipeline(payload_command, stdout=devnull))
        metric = TOML.parsefile(metric_path)
        @test metric["schema_version"] == 4
        @test metric["benchmark_kind"] == "fixed_mu_dmrg"
        @test metric["timed_region"] == "run_dmrg_ground_only"
        @test metric["dmrg_solves"] == [1]
        @test !metric["mpo_construction_timed"]
        @test !metric["initial_mps_copy_timed"]
        @test !metric["garbage_collection_timed"]
        @test !metric["compile_warmup_timed"]
        @test metric["seed_chemical_potential"] ≈ seed.chemical_potential
        @test metric["seed_config_sha256"] == metric["config_sha256"]
        @test metric["seed_git_commit"] == metric["git_commit"]
        @test metric["seed_implementation_sha256"] == metric["implementation_sha256"]
        @test metric["benchmark_chemical_potential"] ≈ seed.chemical_potential
        @test length(metric["seconds"]) == 1

        write(joinpath(directory, "candidates.tsv"),
            "label\tjulia_threads\tbackend\tslurm_logical_cpus\nserial-t1\t1\tserial\t2\n")
        write(joinpath(metrics_directory, "serial-t1.time"),
            "Maximum resident set size (kbytes): 1024\nExit status: 0\n")
        run(pipeline(
            `$julia --startup-file=no --project=$ROOT $report_script $directory`,
            stdout=devnull,
        ))
        recommendation = read(joinpath(directory, "recommendation.md"), String)
        @test occursin("Median fixed-mu DMRG time", recommendation)
        @test occursin("run_dmrg_ground", recommendation)
        @test occursin("Estimated comparison with the legacy GPU path", recommendation)

        bad_directory = joinpath(directory, "bad-workload")
        bad_metrics = joinpath(bad_directory, "metrics")
        mkpath(bad_metrics)
        write(joinpath(bad_directory, "candidates.tsv"),
            "label\tjulia_threads\tbackend\tslurm_logical_cpus\nserial-t1\t1\tserial\t2\n")
        bad_metric = deepcopy(metric)
        bad_metric["timed_region"] = "density_search"
        open(joinpath(bad_metrics, "serial-t1.toml"), "w") do io
            TOML.print(io, bad_metric; sorted=true)
        end
        write(joinpath(bad_metrics, "serial-t1.time"),
            "Maximum resident set size (kbytes): 1024\nExit status: 0\n")
        bad_report = run(pipeline(
            ignorestatus(`$julia --startup-file=no --project=$ROOT $report_script $bad_directory`),
            stdout=devnull,
            stderr=devnull,
        ))
        @test !success(bad_report)
    end
end

@testset "variational double-counting functional" begin
    model = test_model()
    fields = test_fields()
    fields.alpha[1, 1, 1, 1] = 0.3
    correlations = test_correlations()
    correlations.pair[1, 1] = 0.4
    energy = variational_energy(-10.0, 2.0, fields, correlations, model)
    @test energy.pair_field_energy ≈ -0.24
    @test energy.pair_transverse_energy ≈ -0.12
    @test energy.double_counting_correction ≈ 0.12
    @test energy.chemical_potential_term ≈ 8.0
    @test energy.canonical_variational_energy ≈ -1.88

    direct = variational_energy(
        -10.0,
        2.0,
        fields,
        correlations,
        model;
        effective_expectation=-10.0,
        bare_ladder_energy=-1.76,
    )
    @test direct.hamiltonian_identity_error ≈ 0.0
    @test direct.variational_consistency_error ≈ 0.0
    @test direct.direct_variational_energy ≈ -1.88

    measured_fields = copy(fields)
    measured_fields.alpha[1, 1, 1, 1] = 0.6
    off_fixed_point = variational_energy(
        -10.0,
        2.0,
        fields,
        correlations,
        model;
        interaction_fields=measured_fields,
    )
    @test off_fixed_point.pair_field_energy ≈ -0.24
    @test off_fixed_point.pair_transverse_energy ≈ -0.24

    density_fields = test_fields()
    density_fields.mu_cdw[1, 1] = 0.2
    density_correlations = test_correlations()
    density_correlations.density_down[1] = 0.7
    density_energy = variational_energy(-3.0, 0.0, density_fields, density_correlations, model)
    @test density_energy.density_field_energy ≈ 0.14
    @test density_energy.density_transverse_energy ≈ 0.02
    @test density_energy.double_counting_correction ≈ -0.12
    @test density_energy.canonical_variational_energy ≈ -3.12
end

@testset "period-resolved convergence" begin
    settings = ConvergenceSettings(
        field_abs_tol=1e-8,
        field_rel_tol=1e-8,
        density_tol=1e-8,
        variational_energy_tol=1e-8,
        stable_iterations=2,
        max_period=4,
        period_repeats=2,
        period_abs_tol=1e-10,
        period_rel_tol=1e-10,
        stagnation_window=20,
    )
    @test detect_period(test_fields.([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]), settings).period == 2
    @test detect_period(test_fields.([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]), settings).period == 3
    @test detect_period(test_fields.([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]), settings).period == 4
    # Repeating only one phase is insufficient: every phase must recur.
    @test detect_period(test_fields.([0.0, 1.0, 0.0, 2.0, 0.0, 3.0]), settings).period == 0

    fixed_records = [
        test_record(1, test_fields(1.0), test_fields(1.0 + 1e-12)),
        test_record(2, test_fields(1.0), test_fields(1.0 + 1e-12)),
    ]
    fixed = assess_convergence(fixed_records, settings, 1.0)
    @test fixed.status == :fixed_point
    @test fixed.accepted

    values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    applied_values = [1.0, values[1:end - 1]...]
    cycle_records = IterationRecord[]
    for (index, value) in enumerate(values)
        correlations = test_correlations()
        profile = value == 0.0 ? [0.6, 0.4, 0.6, 0.4] : [0.4, 0.6, 0.4, 0.6]
        correlations.density_down .= profile
        correlations.density_up .= profile
        push!(cycle_records, test_record(
            index,
            test_fields(applied_values[index]),
            test_fields(value);
            energy=isodd(index) ? 0.0 : 2.0,
            update_mode=:unmixed_probe,
            correlations,
        ))
    end
    cycle = assess_convergence(cycle_records, settings, 1.0)
    @test cycle.status == :periodic_solution
    @test cycle.fundamental_period == 2
    @test cycle.accepted
    @test cycle.orbit_validated
    @test cycle.solution_canonical_variational_energy ≈ 1.0
    @test cycle.orbit_energy_spread ≈ 2.0
    @test cycle.orbit_density_contrast ≈ 0.4

    mixed_cycle_records = [
        test_record(index, test_fields(applied_values[index]), test_fields(value); update_mode=:anderson)
        for (index, value) in enumerate(values)
    ]
    mixed_cycle = assess_convergence(mixed_cycle_records, settings, 1.0)
    @test mixed_cycle.status == :periodic_candidate
    @test !mixed_cycle.accepted
    @test !mixed_cycle.orbit_validated
end

@testset "mixing" begin
    applied = test_fields(0.0)
    measured = test_fields(1.0)
    settings = MixingSettings(method=:linear, damping=0.25, minimum_damping=0.1, maximum_damping=0.8, adaptive=false)
    state = LadderMPSMFT.MixingState(settings)
    mixed, metadata = mix_fields!(state, applied, measured, settings)
    @test mixed.mu_cdw[1, 1] ≈ 0.25
    @test metadata.method == :linear

    anderson = MixingSettings(method=:anderson, damping=0.5, minimum_damping=0.1,
        maximum_damping=0.8, adaptive=false, regularization=1e-12)
    anderson_state = LadderMPSMFT.MixingState(anderson)
    mix_fields!(anderson_state, test_fields(0.0), test_fields(1.0), anderson)
    midpoint, metadata = mix_fields!(anderson_state, test_fields(1.0), test_fields(0.0), anderson)
    @test metadata.method == :anderson
    @test midpoint.mu_cdw[1, 1] ≈ 0.5 atol=1e-10
end

@testset "ladder diagnostics primitives" begin
    correlation = Matrix{Float64}(I, 4, 4)
    expectation = zeros(4)
    grid = LadderMPSMFT.structure_factor_grid(correlation, expectation, 2)
    @test size(grid.values) == (2, 2)
    @test all(grid.values .≈ 1.0)
    states = LadderMPSMFT.fixed_sector_product_state(1:8, 6, 0; rng=MersenneTwister(4))
    @test length(states) == 8
    @test count(==("Up"), states) + count(==("UpDn"), states) == 3
    @test count(==("Dn"), states) + count(==("UpDn"), states) == 3
end

@testset "checkpoint schema and strict selection" begin
    mktempdir() do directory
        model = test_model()
        run = RunSettings(output_directory=directory, quick_diagnostics=false)
        settings = ProjectSettings(model=model, run=run)
        sites = LadderMPSMFT.make_sites(model)
        psi = productMPS(sites, density_product_state(4, 1.0; rng=MersenneTwister(2)))
        record = test_record(1, test_fields(0.0), test_fields(0.0))
        diagnostic = ConvergenceDiagnostic(
            status=:fixed_point,
            accepted=true,
            reason="test fixed point",
            solution_kind=:fixed_point,
            fundamental_period=1,
            solution_canonical_variational_energy=0.0,
            orbit_energy_spread=0.0,
            orbit_density_contrast=0.0,
            fixed_point_abs_residual=0.0,
            fixed_point_rel_residual=0.0,
            cycle_abs_residual=0.0,
            cycle_rel_residual=0.0,
            density_error=0.0,
            variational_energy_change=0.0,
            hamiltonian_identity_error_per_site=0.0,
            effective_eigenvalue_error_per_site=0.0,
            best_iteration=1,
        )
        run_directory = joinpath(directory, "nested", "run")
        path = joinpath(run_directory, "state.h5")
        write_checkpoint(path; settings, psi, records=[record], diagnostic,
            restart_fields=test_fields(0.25), chemical_potential=0.4, immutable=true)
        checkpoint = read_checkpoint(path)
        @test checkpoint.accepted
        @test checkpoint.chemical_potential ≈ 0.4
        @test checkpoint.restart.mu_cdw[1, 1] ≈ 0.25
        @test_throws ArgumentError write_checkpoint(path; settings, psi, records=[record], diagnostic, immutable=true)
        selected = select_completed_runs(directory)
        @test length(selected) == 1
        @test only(selected).path == path
        @test only(selected).plot_style == "solid"

        second_settings = ProjectSettings(
            model=model,
            run=RunSettings(output_directory=directory, branch_label="sdw", quick_diagnostics=false),
        )
        second_path = joinpath(directory, "nested", "second", "state.h5")
        second_record = test_record(1, test_fields(0.0), test_fields(0.0); energy=-0.1)
        second_diagnostic = ConvergenceDiagnostic(
            status=:fixed_point,
            accepted=true,
            reason="test fixed point",
            solution_kind=:fixed_point,
            fundamental_period=1,
            solution_canonical_variational_energy=-0.1,
            orbit_energy_spread=0.0,
            orbit_density_contrast=0.0,
            fixed_point_abs_residual=0.0,
            fixed_point_rel_residual=0.0,
            cycle_abs_residual=0.0,
            cycle_rel_residual=0.0,
            density_error=0.0,
            variational_energy_change=0.0,
            hamiltonian_identity_error_per_site=0.0,
            effective_eigenvalue_error_per_site=0.0,
            best_iteration=1,
        )
        write_checkpoint(second_path; settings=second_settings, psi, records=[second_record], diagnostic=second_diagnostic,
            restart_fields=test_fields(0.0), chemical_potential=0.0, immutable=true)
        ranking = compare_variational_branches([path, second_path])
        @test first(ranking).path == second_path
        @test first(ranking).energy ≈ -0.1

        periodic_path = joinpath(directory, "nested", "periodic", "state.h5")
        periodic_records = [
            test_record(1, test_fields(1.0), test_fields(0.0); energy=-0.2, update_mode=:unmixed_probe),
            test_record(2, test_fields(0.0), test_fields(1.0); energy=-0.4, update_mode=:unmixed_probe),
        ]
        periodic_diagnostic = ConvergenceDiagnostic(
            status=:periodic_solution,
            accepted=true,
            reason="test periodic solution",
            solution_kind=:periodic_orbit,
            fundamental_period=2,
            orbit_validated=true,
            unmixed_probe=true,
            solution_canonical_variational_energy=-0.3,
            orbit_energy_spread=0.2,
            orbit_density_contrast=0.5,
            cycle_abs_residual=0.0,
            cycle_rel_residual=0.0,
            density_error=0.0,
            variational_energy_change=0.0,
            hamiltonian_identity_error_per_site=0.0,
            effective_eigenvalue_error_per_site=0.0,
            best_iteration=2,
        )
        phase_psis = Dict(1 => deepcopy(psi), 2 => deepcopy(psi))
        write_checkpoint(
            periodic_path;
            settings,
            psi,
            records=periodic_records,
            diagnostic=periodic_diagnostic,
            restart_fields=test_fields(0.0),
            chemical_potential=0.0,
            immutable=true,
            phase_psis,
        )
        periodic_checkpoint = read_checkpoint(periodic_path)
        @test periodic_checkpoint.accepted
        @test periodic_checkpoint.fundamental_period == 2
        @test periodic_checkpoint.orbit_validated
        @test length(read_orbit_phase_states(periodic_path)) == 2
        ranking = compare_variational_branches([path, second_path, periodic_path])
        @test first(ranking).path == periodic_path
        @test first(ranking).energy ≈ -0.3

        invalid_periodic_path = joinpath(directory, "nested", "invalid-periodic", "state.h5")
        invalid_periodic_diagnostic = ConvergenceDiagnostic(
            status=:periodic_solution,
            accepted=true,
            reason="deliberately inconsistent test artifact",
            solution_kind=:periodic_orbit,
            fundamental_period=2,
            orbit_validated=false,
            unmixed_probe=false,
            solution_canonical_variational_energy=-1.0,
            best_iteration=2,
        )
        write_checkpoint(
            invalid_periodic_path;
            settings,
            psi,
            records=periodic_records,
            diagnostic=invalid_periodic_diagnostic,
            restart_fields=test_fields(0.0),
            chemical_potential=0.0,
            immutable=true,
        )
        @test length(select_completed_runs(directory)) == 3
        included = select_completed_runs(directory; include_incomplete=true)
        invalid_row = only(filter(row -> row.path == invalid_periodic_path, included))
        @test invalid_row.plot_style == "hatched"
        @test_throws ArgumentError compare_variational_branches([path, invalid_periodic_path])
    end
end
