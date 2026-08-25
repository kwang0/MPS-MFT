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
    @test selection.mode == :exact
    @test selection.lower_record === selection.upper_record
    @test_throws ArgumentError lookup_ep(registry; L=64, U=8, V=0, t0=2, density=0.9375, tp=0.1)

    interpolated = lookup_ep(
        registry;
        L=64,
        U=8,
        V=-0.2,
        t0=1.1,
        density=0.9375,
        tp=0.1,
        allow_interpolation=true,
    )
    @test interpolated.mode == :linear_t0
    @test interpolated.interpolation_weight ≈ 0.5
    @test interpolated.lower_record.t0 ≈ 1.0
    @test interpolated.upper_record.t0 ≈ 1.2
    @test interpolated.record.E_p ≈ -0.18452309659153343
    @test interpolated.denominator ≈ 0.18452309659153343
    @test_throws ArgumentError lookup_ep(
        registry;
        L=64, U=8, V=-0.2, t0=0.9, density=0.9375, tp=0.1,
        allow_interpolation=true,
    )

    sign_changing = [
        EpRecord(4, 2.0, 0.0, 1.0, 1.0, 10, -1.0, -0.1, 1e-6),
        EpRecord(4, 2.0, 0.0, 1.2, 1.0, 10, -1.1, 0.1, 1e-6),
    ]
    @test_throws ArgumentError lookup_ep(
        sign_changing;
        L=4, U=2, V=0, t0=1.1, density=1.0, tp=0.01,
        source_path="synthetic", require_bound=false, allow_interpolation=true,
    )
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
    gpu_settings = load_settings(joinpath(ROOT, "configs", "phase1_gpu_base.toml"))
    @test gpu_settings.runtime.backend == :gpu
    @test gpu_settings.runtime.tensor_scalar_type == :float64
    @test !gpu_settings.runtime.conserve_sz
    @test !gpu_settings.runtime.conserve_nfparity
    @test gpu_settings.convergence.cycle_action == :continue
    @test gpu_settings.model.ep_mode == :linear_t0
    @test gpu_settings.model.ep_signed ≈ -0.18452309659153343
    @test gpu_settings.model.tp^2 / gpu_settings.model.ep ≈ 0.05419375777188662
    recurrence_settings = load_settings(joinpath(ROOT, "configs", "phase1_gpu_recurrence_chi400.toml"))
    @test recurrence_settings.model.geometry == :cubic_unfrustrated
    @test recurrence_settings.dmrg.maxdim == 400
    @test recurrence_settings.dmrg.nsweeps == 16
    @test recurrence_settings.convergence.cycle_action == :stop
    @test recurrence_settings.run.max_iterations == recurrence_settings.convergence.probe_iterations + 1
    conflicting_lineage = ProjectSettings(
        model=test_model(),
        run=RunSettings(
            inherit_from="legacy.h5",
            inherit_sha256=repeat("0", 64),
            parent_checkpoint="parent.h5",
            parent_sha256=repeat("1", 64),
        ),
    )
    @test_throws ArgumentError validate_settings(conflicting_lineage)
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=test_model(),
        run=RunSettings(parent_orbit_phase=1),
    ))
    @test_throws ArgumentError validate_settings(ProjectSettings(
        model=test_model(),
        run=RunSettings(
            parent_checkpoint="parent.h5",
            parent_sha256=repeat("0", 64),
            parent_orbit_phase=0,
        ),
    ))
    invalid_gpu = ProjectSettings(
        model=test_model(),
        runtime=RuntimeSettings(backend=:gpu, threaded_blocksparse=false),
    )
    @test_throws ArgumentError validate_settings(invalid_gpu)
    cpu_fingerprint = LadderMPSMFT.numerical_fingerprint(ProjectSettings(model=test_model()))
    dense_cpu_fingerprint = LadderMPSMFT.numerical_fingerprint(ProjectSettings(
        model=test_model(),
        runtime=RuntimeSettings(threaded_blocksparse=false, conserve_sz=false, conserve_nfparity=false),
    ))
    @test cpu_fingerprint != dense_cpu_fingerprint
    float32_fingerprint = LadderMPSMFT.numerical_fingerprint(ProjectSettings(
        model=test_model(),
        runtime=RuntimeSettings(tensor_scalar_type=:float32),
    ))
    @test cpu_fingerprint != float32_fingerprint
    first_seed = initial_fields(test_model(); seed=:pairing, rng=MersenneTwister(7))
    second_seed = initial_fields(test_model(); seed=:pairing, rng=MersenneTwister(7))
    @test first_seed.alpha == second_seed.alpha
    @test first_seed.alpha == permutedims(first_seed.alpha, (2, 1, 4, 3))
    @test all(iszero, first_seed.beta)
    sdw_seed = initial_fields(test_model(); seed=:sdw, amplitude=1.0)
    @test sdw_seed.mu_cdw[1, :] == [-1.0, 1.0, 1.0, -1.0]
    @test sdw_seed.mu_cdw[2, :] == -sdw_seed.mu_cdw[1, :]
    cdw_seed = initial_fields(test_model(); seed=:cdw, amplitude=1.0)
    @test cdw_seed.mu_cdw[1, :] == [-1.0, -1.0, 1.0, 1.0]
    @test cdw_seed.mu_cdw[2, :] == cdw_seed.mu_cdw[1, :]
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

@testset "Phase 1 guarded GPU launcher" begin
    script = joinpath(ROOT, "slurm", "phase1_gpu.sh")
    migration_script = joinpath(ROOT, "slurm", "migrate_phase1_to_scratch.sh")
    corrupt_cleaner = joinpath(ROOT, "scripts", "prune_corrupt_auxiliary_hdf5.jl")
    script_source = read(script, String)
    migration_source = read(migration_script, String)
    cleaner_source = read(corrupt_cleaner, String)
    @test !occursin(r"(?m)^\s*module load cudatoolkit\s*$", script_source)
    @test occursin("module unload cudatoolkit", script_source)
    @test occursin("sanitize_cuda_runtime_environment", script_source)
    @test occursin("require_current_run_version", script_source)
    @test occursin("require_worker_compatible_run_version", script_source)
    @test occursin("PHASE1_SCRIPT_VERSION=\"1.4.0\"", script_source)
    @test occursin("prepare-recovery)", script_source)
    @test occursin("prepare-recurrence)", script_source)
    @test occursin("--licenses=scratch,cfs", script_source)
    @test occursin("compact_results.jl", script_source)
    @test !occursin("transfer_files.py", migration_source)
    @test occursin("cp -a --", migration_source)
    @test occursin("sha256sum -c", migration_source)
    @test occursin("direct_scratch_copy_sha256_verified=true", migration_source)
    @test occursin("--prune-corrupt-auxiliary", migration_source)
    @test occursin("critical_corrupt_hdf5", cleaner_source)
    @test occursin("lowercase(name) == \"state.h5\"", cleaner_source)
    @test occursin("verify_stateless_results.jl", migration_source)
    @test occursin("--prune-cfs", migration_source)
    @test occursin("gpu_linalg_preflight!", read(joinpath(ROOT, "scripts", "gpu_smoke.jl"), String))
    @test occursin("gpu_linalg_preflight!", read(joinpath(ROOT, "scripts", "run_scf_gpu.jl"), String))
    bash_executable = something(Sys.which("bash"), "bash")
    sanitized_environment = read(
        `$bash_executable -c 'source "$1" plan >/dev/null; export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2; export LD_LIBRARY_PATH=/safe/one:/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/math_libs/13.2/lib64:/safe/two; sanitize_cuda_runtime_environment; printf "%s|%s|%s\n" "${LD_LIBRARY_PATH-unset}" "${CUDA_HOME-unset}" "${CUDA_PATH-unset}"' bash $script`,
        String,
    )
    @test strip(sanitized_environment) == "/safe/one:/safe/two|unset|unset"
    function bash_path(path::AbstractString)
        normalized = replace(abspath(path), '\\' => '/')
        match_result = match(r"^([A-Za-z]):/(.*)$", normalized)
        return match_result === nothing ? normalized :
            "/$(lowercase(match_result.captures[1]))/$(match_result.captures[2])"
    end
    bash_environment_path = strip(read(
        `$bash_executable -lc 'printf "%s" "${PATH}"'`,
        String,
    ))
    mock_bin = joinpath(ROOT, "test", "fixtures", "mock_slurm")
    julia_executable = Base.julia_cmd().exec[1]
    mktempdir() do directory
        run_root = joinpath(directory, "runs")
        scratch_root = joinpath(directory, "scratch")
        budget_root = joinpath(directory, "budget")
        ledger = joinpath(budget_root, "ledger.tsv")
        environment = (
            "PATH" => string(bash_path(mock_bin), ":", bash_environment_path),
            "MOCK_SLURM_STATE_DIR" => bash_path(joinpath(directory, "slurm")),
            "PHASE1_RUN_ROOT" => bash_path(run_root),
            "PHASE1_SCRATCH_ROOT" => bash_path(scratch_root),
            "PHASE1_BUDGET_ROOT" => bash_path(budget_root),
            "PHASE1_LEDGER_PATH" => bash_path(ledger),
            "PHASE1_JULIA" => bash_path(julia_executable),
        )
        run(pipeline(
            addenv(`$bash_executable $script submit mock_phase1`, environment...),
            stdout=devnull,
        ))
        smoke_path = joinpath(run_root, "mock_phase1", "gpu_smoke.h5")
        h5open(smoke_path, "w") do file
            file["completed"] = true
            file["energy"] = -1.0
            file["density"] = 1.0
            device = create_group(file, "device")
            device["cuda_runtime_library_isolation"] = "passed"
            device["tensor_scalar_type"] = "float64"
            linalg = create_group(file, "linalg_preflight")
            linalg["dimension"] = 256
            linalg["scalar_type"] = "Float64"
            psi_group = create_group(file, "psi")
            tensor_group = create_group(psi_group, "MPS[1]")
            storage_group = create_group(tensor_group, "storage")
            storage_group["data"] = Float64[1.0]
        end
        run(pipeline(
            addenv(`$bash_executable $script submit-matrix mock_phase1`, environment...),
            stdout=devnull,
        ))
        ledger_rows = readlines(ledger)[2:end]
        @test length(ledger_rows) == 10
        @test sum(parse(Float64, split(row, '\t')[7]) for row in ledger_rows) ≈ 27.125
        @test length(readlines(joinpath(run_root, "mock_phase1", "manifest.tsv"))) == 10
        @test length(readlines(joinpath(run_root, "mock_phase1", "jobs.tsv"))) == 11
        @test isdir(joinpath(scratch_root, "mock_phase1", "results"))
        prepared_config = TOML.parsefile(joinpath(
            run_root,
            "mock_phase1",
            "configs",
            "frustrated__pairing_s1.segment-001.toml",
        ))
        @test startswith(
            prepared_config["run"]["output_directory"],
            joinpath(scratch_root, "mock_phase1", "results"),
        )
        @test isempty(readdir(joinpath(run_root, "mock_phase1", "results")))

        source_run = joinpath(run_root, "source_recurrence")
        source_full_root = joinpath(scratch_root, "source_recurrence")
        source_full_state = joinpath(source_full_root, "results", "candidate", "state.h5")
        recurrence_base = load_settings(joinpath(
            ROOT,
            "configs",
            "phase1_gpu_recurrence_chi400.toml",
        ))
        mkpath(dirname(source_full_state))
        h5open(source_full_state, "w") do file
            file["status"] = "periodic_candidate"
            file["accepted"] = false
            file["fundamental_period"] = 2
            provenance = create_group(file, "provenance")
            provenance["model_fingerprint"] = LadderMPSMFT.model_fingerprint(recurrence_base.model)
            provenance["numerical_fingerprint"] = "source-numerical-fingerprint"
            provenance["tensor_scalar_type"] = "float64"
            cycles = create_group(file, "cycle_members")
            for (index, phase_name) in enumerate(("001", "002"))
                phase = create_group(cycles, phase_name)
                phase["psi"] = Float64[parse(Int, phase_name)]
                create_group(phase, "applied")
                create_group(phase, "measured")
                phase["chemical_potential"] = 1.0
                phase["iteration"] = 47 + index
                phase["update_mode"] = "unmixed_probe"
            end
        end
        source_compact_state = joinpath(
            source_run,
            "stateless_results",
            "unfrustrated__pairing_s1",
            "candidate",
            "state.h5",
        )
        mkpath(dirname(source_compact_state))
        h5open(source_compact_state, "w") do file
            analysis = create_group(file, "analysis_storage")
            analysis["is_stateless_copy"] = true
            analysis["full_artifact_path"] = source_full_state
            analysis["full_artifact_sha256"] = LadderMPSMFT.sha256_file(source_full_state)
            file["status"] = "periodic_candidate"
            file["accepted"] = false
            file["fundamental_period"] = 2
            provenance = create_group(file, "provenance")
            provenance["model_fingerprint"] = LadderMPSMFT.model_fingerprint(recurrence_base.model)
            provenance["numerical_fingerprint"] = "source-numerical-fingerprint"
            provenance["tensor_scalar_type"] = "float64"
            cycles = create_group(file, "cycle_members")
            for (index, phase_name) in enumerate(("001", "002"))
                phase = create_group(cycles, phase_name)
                create_group(phase, "applied")
                create_group(phase, "measured")
                phase["iteration"] = 47 + index
                phase["update_mode"] = "unmixed_probe"
            end
        end
        mkpath(source_run)
        write(joinpath(source_run, "full_storage_path.txt"), source_full_root * "\n")
        ledger_before_recurrence = read(ledger, String)
        recurrence_plan = read(
            addenv(`$bash_executable $script plan-recurrence`, environment...),
            String,
        )
        @test occursin("9.125000000 node-hours", recurrence_plan)
        run(pipeline(
            addenv(
                `$bash_executable $script prepare-recurrence source_recurrence mock_recurrence`,
                environment...,
            ),
            stdout=devnull,
        ))
        @test read(ledger, String) == ledger_before_recurrence
        recurrence_run = joinpath(run_root, "mock_recurrence")
        @test strip(read(joinpath(recurrence_run, "branch_count.txt"), String)) == "3"
        @test length(readlines(joinpath(recurrence_run, "manifest.tsv"))) == 4
        @test length(filter(
            name -> endswith(name, ".segment-001.toml"),
            readdir(joinpath(recurrence_run, "configs")),
        )) == 3
        phase_config = TOML.parsefile(joinpath(
            recurrence_run,
            "configs",
            "unfrustrated__pairing_s1_phase001_chi400.segment-001.toml",
        ))
        @test phase_config["run"]["parent_orbit_phase"] == 1
        @test phase_config["run"]["parent_sha256"] == LadderMPSMFT.sha256_file(source_full_state)
        @test phase_config["dmrg"]["maxdim"] == 400
        @test phase_config["convergence"]["cycle_action"] == "stop"

        rejected = run(pipeline(
            ignorestatus(addenv(
                `$bash_executable $script submit cap_rejection`,
                environment...,
                "PHASE1_ADDITIONAL_NODE_HOUR_CAP" => "27.1",
            )),
            stdout=devnull,
            stderr=devnull,
        ))
        @test !success(rejected)
        @test !ispath(joinpath(run_root, "cap_rejection"))
        @test strip(read(joinpath(directory, "slurm", "next_job_id"), String)) == "700010"
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

    raw_candidate = ConvergenceDiagnostic(
        status=:periodic_candidate,
        accepted=false,
        solution_kind=:periodic_orbit,
        fundamental_period=2,
        unmixed_probe=true,
    )
    continue_settings = ConvergenceSettings(probe_iterations=20, cycle_action=:continue)
    @test LadderMPSMFT._recurrence_action(
        raw_candidate,
        continue_settings;
        probe_steps=8,
        probe_origin=:initial,
        mixer_probe_completed=false,
    ) == :continue_probe
    @test LadderMPSMFT._recurrence_action(
        raw_candidate,
        continue_settings;
        probe_steps=20,
        probe_origin=:initial,
        mixer_probe_completed=false,
    ) == :enter_mixing
    @test LadderMPSMFT._recurrence_action(
        raw_candidate,
        continue_settings;
        probe_steps=20,
        probe_origin=:mixer_recurrence,
        mixer_probe_completed=true,
    ) == :stop_candidate
    mixed_candidate = ConvergenceDiagnostic(
        status=:periodic_candidate,
        accepted=false,
        solution_kind=:periodic_orbit,
        fundamental_period=2,
        unmixed_probe=false,
    )
    @test LadderMPSMFT._recurrence_action(
        mixed_candidate,
        continue_settings;
        probe_steps=0,
        probe_origin=:none,
        mixer_probe_completed=false,
    ) == :start_mixer_probe
    @test LadderMPSMFT._recurrence_action(
        mixed_candidate,
        continue_settings;
        probe_steps=0,
        probe_origin=:none,
        mixer_probe_completed=true,
    ) == :stop_candidate
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
        psi_float32 = LadderMPSMFT.convert_tensor_scalar_type(psi, :float32)
        psi_float64 = move_to_backend(
            psi_float32,
            RuntimeSettings(backend=:cpu, tensor_scalar_type=:float64),
        )
        @test all(tensor -> eltype(tensor) == Float64, psi_float64)
        record = test_record(1, test_fields(0.0), test_fields(0.5))
        history_record = test_record(2, test_fields(0.25), test_fields(0.75))
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
        write_checkpoint(path; settings, psi, records=[record, history_record], diagnostic,
            restart_fields=test_fields(0.25), chemical_potential=0.4, immutable=true)
        checkpoint = read_checkpoint(path)
        @test checkpoint.accepted
        @test checkpoint.chemical_potential ≈ 0.4
        @test checkpoint.restart.mu_cdw[1, 1] ≈ 0.25
        measured_history = read_field_history(path; source=:measured)
        applied_history = read_field_history(path; source=:applied)
        @test measured_history.iterations == [1, 2]
        @test size(measured_history.alpha) == (2, 2, 2, 2, 2)
        @test size(measured_history.beta) == (2, 2, 2, 2, 2, 2)
        @test size(measured_history.mu_cdw) == (2, 4, 2)
        @test measured_history.mu_cdw[1, 1, :] == [0.5, 0.75]
        @test applied_history.mu_cdw[1, 1, :] == [0.0, 0.25]
        refactored_inherit = read_inherited_fields(path)
        @test refactored_inherit.format == :refactored
        @test refactored_inherit.fields.mu_cdw[1, 1] == 0.25
        @test refactored_inherit.chemical_potential == 0.4
        h5open(path, "r") do file
            @test Int(read(file, "schema_version")) == 5
            @test read(file, "fields/initial/mu_cdw")[1, 1] == 0.0
        end
        @test_throws ArgumentError write_checkpoint(path; settings, psi, records=[record], diagnostic, immutable=true)

        legacy_path = joinpath(directory, "legacy_fields.h5")
        h5open(legacy_path, "w") do file
            file["alpha"] = test_fields(0.0).alpha
            file["beta"] = test_fields(0.0).beta
            file["mu"] = 1.25
        end
        inherited = read_inherited_fields(legacy_path)
        @test inherited.format == :legacy
        @test inherited.chemical_potential == 1.25
        @test size(inherited.fields.mu_cdw) == (2, 4)
        @test all(iszero, inherited.fields.mu_cdw)
        inherit_settings = ProjectSettings(
            model=model,
            run=RunSettings(
                output_directory=directory,
                inherit_from=legacy_path,
                inherit_sha256=LadderMPSMFT.sha256_file(legacy_path),
                quick_diagnostics=false,
            ),
        )
        inherited_start = LadderMPSMFT._initial_state(inherit_settings)
        @test inherited_start.source == "field_inherit"
        @test inherited_start.inherit_format == "legacy"
        @test inherited_start.chemical_potential == 1.25
        @test length(inherited_start.psi) == 4
        bad_hash_settings = ProjectSettings(
            model=model,
            run=RunSettings(
                inherit_from=legacy_path,
                inherit_sha256=repeat("0", 64),
                quick_diagnostics=false,
            ),
        )
        @test_throws ArgumentError LadderMPSMFT._initial_state(bad_hash_settings)
        gpu_resume_settings = ProjectSettings(
            model=model,
            runtime=RuntimeSettings(
                backend=:gpu,
                threaded_blocksparse=false,
                conserve_sz=false,
                conserve_nfparity=false,
            ),
            run=RunSettings(
                output_directory=directory,
                resume_checkpoint=path,
                resume_sha256=LadderMPSMFT.sha256_file(path),
                quick_diagnostics=false,
            ),
        )
        @test_throws ArgumentError LadderMPSMFT._initial_state(gpu_resume_settings)
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
        first_phase = read_checkpoint(periodic_path; orbit_phase=1)
        second_phase = read_checkpoint(periodic_path; orbit_phase=2)
        @test first_phase.applied.alpha == test_fields(1.0).alpha
        @test first_phase.measured.alpha == test_fields(0.0).alpha
        @test first_phase.restart.alpha == first_phase.measured.alpha
        @test second_phase.applied.alpha == test_fields(0.0).alpha
        @test second_phase.measured.alpha == test_fields(1.0).alpha
        @test_throws ArgumentError read_checkpoint(periodic_path; orbit_phase=3)
        phase_parent_settings = ProjectSettings(
            model=model,
            run=RunSettings(
                output_directory=directory,
                parent_checkpoint=periodic_path,
                parent_sha256=LadderMPSMFT.sha256_file(periodic_path),
                parent_orbit_phase=1,
                quick_diagnostics=false,
            ),
        )
        phase_parent_start = LadderMPSMFT._initial_state(phase_parent_settings)
        @test phase_parent_start.source == "parent_orbit_phase_001"
        @test phase_parent_start.fields.alpha == test_fields(0.0).alpha
        @test length(read_orbit_phase_states(periodic_path)) == 2
        stateless_path = joinpath(directory, "stateless_periodic.h5")
        stateless = write_stateless_copy(periodic_path, stateless_path)
        @test stateless.source_bytes > stateless.compact_bytes
        h5open(stateless_path, "r") do file
            @test !haskey(file, "psi")
            @test !haskey(file, "cycle_members/001/psi")
            @test !haskey(file, "cycle_members/002/psi")
            @test haskey(file, "history/fields/applied/alpha")
            @test Bool(read(file, "analysis_storage/is_stateless_copy"))
            @test !Bool(read(file, "analysis_storage/restartable"))
            @test String(read(file, "analysis_storage/full_artifact_sha256")) ==
                LadderMPSMFT.sha256_file(periodic_path)
        end
        @test read_field_history(stateless_path; source=:measured).iterations == [1, 2]
        @test read_inherited_fields(stateless_path).format == :refactored
        @test_throws ArgumentError read_checkpoint(stateless_path)
        @test_throws ArgumentError read_checkpoint(stateless_path; orbit_phase=1)
        @test_throws ArgumentError read_orbit_phase_states(stateless_path)

        full_tree = joinpath(directory, "full_tree")
        compact_tree = joinpath(directory, "compact_tree")
        mkpath(joinpath(full_tree, "nested"))
        cp(periodic_path, joinpath(full_tree, "nested", "artifact.h5"))
        write(joinpath(full_tree, "nested", "run_summary.md"), "test summary\n")
        mirror = mirror_stateless_tree(full_tree, compact_tree)
        @test length(mirror.records) == 2
        @test isfile(joinpath(compact_tree, "stateless_manifest.tsv"))
        @test isfile(joinpath(compact_tree, "nested", "run_summary.md"))
        h5open(joinpath(compact_tree, "nested", "artifact.h5"), "r") do file
            @test !haskey(file, "psi")
            @test haskey(file, "history/fields/measured/beta")
        end

        pair_binding_path = joinpath(directory, "pair_binding.h5")
        h5open(pair_binding_path, "w") do file
            file["E_N_120"] = -10.0
            file["psi_N_120"] = Float64[1, 2, 3]
        end
        pair_binding_stateless = joinpath(directory, "pair_binding_stateless.h5")
        write_stateless_copy(pair_binding_path, pair_binding_stateless)
        h5open(pair_binding_stateless, "r") do file
            @test haskey(file, "E_N_120")
            @test !haskey(file, "psi_N_120")
        end
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
