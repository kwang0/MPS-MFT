#!/usr/bin/env julia

using HDF5
using LadderMPSMFT
using TOML

length(ARGS) == 4 || error(
    "usage: julia --project=. scripts/prepare_phase1_square_tight5.jl SOURCE_CONTROL_RUN CONTROL_RUN FULL_RUN RUN_ID",
)

const SOURCE_LABELS = (
    "square__pairing_dwave_m000_chi200_loose",
    "square__legacy_pairing_mixed_chi200_loose",
    "square__stripe_m004_chi200_loose",
    "square__stripe_m005_chi200_loose",
    "square__stripe_pairing_m004_chi200_loose",
    "square__stripe_pairing_m005_chi200_loose",
)
const ANALYSIS_FIELD_FLOOR_SCAN = "0,1e-6,1e-5,1e-4"
const EXPECTED_EP_SIGNED = -0.24962435880865996

source_run = abspath(ARGS[1])
control_run = abspath(ARGS[2])
full_run = abspath(ARGS[3])
run_id = ARGS[4]

isdir(source_run) || error("square tight-five source run not found: $source_run")
occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID: $run_id")

function only_compact_state(source_run::AbstractString, label::AbstractString)
    states = String[]
    for root_name in ("stateless_results", "results")
        root = joinpath(source_run, root_name, label)
        isdir(root) || continue
        for (directory, _, names) in walkdir(root)
            "state.h5" in names && push!(states, realpath(joinpath(directory, "state.h5")))
        end
    end
    unique!(states)
    length(states) == 1 || error(
        "expected exactly one compact state for $label, found $(length(states))",
    )
    return only(states)
end

function source_metadata(compact_path::AbstractString, full_root::AbstractString)
    compact = h5open(compact_path, "r") do file
        required = (
            "analysis_storage/is_stateless_copy",
            "analysis_storage/full_artifact_path",
            "analysis_storage/full_artifact_sha256",
            "status", "accepted", "solution_kind", "fundamental_period",
            "solution_canonical_variational_energy", "fixed_point_rel_residual",
            "density_error", "history/iteration", "provenance/model_fingerprint",
            "provenance/numerical_fingerprint", "provenance/implementation_sha256",
            "provenance/ep_source_sha256", "provenance/tensor_scalar_type",
        )
        for name in required
            haskey(file, name) || error("source compact state has no $name: $compact_path")
        end
        Bool(read(file, "analysis_storage/is_stateless_copy")) || error(
            "source is not a stateless analysis state: $compact_path",
        )
        status = Symbol(read(file, "status"))
        accepted = Bool(read(file, "accepted"))
        solution_kind = Symbol(read(file, "solution_kind"))
        period = Int(read(file, "fundamental_period"))
        accepted || error("source state is not accepted: $compact_path")
        status == :fixed_point || error("source state is $status rather than fixed_point")
        solution_kind == :fixed_point || error(
            "source solution kind is $solution_kind rather than fixed_point",
        )
        period == 1 || error("source fixed point has fundamental period $period")
        tensor_scalar_type = lowercase(String(read(file, "provenance/tensor_scalar_type")))
        tensor_scalar_type == "float64" || error(
            "source state tensor scalar type is $tensor_scalar_type rather than float64",
        )
        return (
            full_path=String(read(file, "analysis_storage/full_artifact_path")),
            full_sha256=lowercase(String(read(file, "analysis_storage/full_artifact_sha256"))),
            status,
            accepted,
            solution_kind,
            period,
            energy=Float64(read(file, "solution_canonical_variational_energy")),
            field_rel_residual=Float64(read(file, "fixed_point_rel_residual")),
            density_error=Float64(read(file, "density_error")),
            history_records=length(read(file, "history/iteration")),
            model_fingerprint=String(read(file, "provenance/model_fingerprint")),
            numerical_fingerprint=String(read(file, "provenance/numerical_fingerprint")),
            implementation_sha256=String(read(file, "provenance/implementation_sha256")),
            ep_source_sha256=String(read(file, "provenance/ep_source_sha256")),
            tensor_scalar_type,
            compact_sha256=LadderMPSMFT.sha256_file(compact_path),
        )
    end

    isfile(compact.full_path) || error(
        "source full parent is unavailable; run this preparation on Perlmutter: $(compact.full_path)",
    )
    resolved_root = realpath(full_root)
    resolved_full = realpath(compact.full_path)
    relative = relpath(resolved_full, resolved_root)
    (relative == ".." || startswith(relative, "..$(Base.Filesystem.path_separator)")) && error(
        "source full parent escapes the recorded scratch run: $resolved_full",
    )
    actual_sha256 = LadderMPSMFT.sha256_file(resolved_full)
    actual_sha256 == compact.full_sha256 || error(
        "source full parent SHA-256 differs from its compact link: $resolved_full",
    )
    full = h5open(resolved_full, "r") do file
        for name in (
            "psi", "fields/restart", "chemical_potential", "status", "accepted",
            "solution_kind", "fundamental_period", "provenance/model_fingerprint",
            "provenance/numerical_fingerprint", "provenance/implementation_sha256",
            "provenance/ep_source_sha256", "provenance/tensor_scalar_type",
        )
            haskey(file, name) || error("source full parent has no $name: $resolved_full")
        end
        return (
            status=Symbol(read(file, "status")),
            accepted=Bool(read(file, "accepted")),
            solution_kind=Symbol(read(file, "solution_kind")),
            period=Int(read(file, "fundamental_period")),
            model_fingerprint=String(read(file, "provenance/model_fingerprint")),
            numerical_fingerprint=String(read(file, "provenance/numerical_fingerprint")),
            implementation_sha256=String(read(file, "provenance/implementation_sha256")),
            ep_source_sha256=String(read(file, "provenance/ep_source_sha256")),
            tensor_scalar_type=lowercase(String(read(file, "provenance/tensor_scalar_type"))),
        )
    end
    for key in (
        :status, :accepted, :solution_kind, :period, :model_fingerprint,
        :numerical_fingerprint, :implementation_sha256, :ep_source_sha256,
        :tensor_scalar_type,
    )
        getfield(full, key) == getfield(compact, key) || error(
            "full and compact source metadata differ for $key: $compact_path",
        )
    end
    return merge(compact, (; full_path=resolved_full))
end

full_root_path = joinpath(source_run, "full_storage_path.txt")
isfile(full_root_path) || error("source run has no full_storage_path.txt")
full_root = strip(read(full_root_path, String))
isdir(full_root) || error(
    "source full storage is unavailable; run this preparation on Perlmutter: $full_root",
)

config_directory = joinpath(control_run, "configs")
ispath(config_directory) && !isempty(readdir(config_directory)) && error(
    "refusing to overwrite nonempty configuration directory: $config_directory",
)
mkpath(config_directory)
mkpath(joinpath(control_run, "results"))
mkpath(joinpath(full_run, "results"))

manifest_path = joinpath(control_run, "manifest.tsv")
ispath(manifest_path) && error("refusing to overwrite manifest: $manifest_path")
model_fingerprints = String[]
source_numerical_fingerprints = String[]
new_numerical_fingerprints = String[]

open(manifest_path, "w") do io
    println(io, join((
        "label", "geometry", "branch", "seed", "random_seed", "config", "config_sha256",
        "ep_mode", "ep_signed", "ep_abs", "tp2_over_ep",
        "source_run", "source_label", "source_compact_state", "source_compact_sha256",
        "parent_checkpoint", "parent_sha256", "parent_status", "parent_solution_kind",
        "parent_period", "parent_energy", "parent_field_rel_residual", "parent_density_error",
        "parent_history_records", "parent_numerical_fingerprint", "parent_implementation_sha256",
        "parent_ep_source_sha256", "parent_tensor_scalar_type",
        "full_output_directory", "stateless_output_directory",
        "map_field_threshold", "analysis_field_floor_scan", "preliminary_energy_only",
        "max_additional_mf_updates", "raw_map_only", "dmrg_sweeps", "dmrg_maxdim",
        "dmrg_energy_tol", "dmrg_cutoff", "dmrg_max_time_seconds", "mu_density_tol",
        "outer_density_tol", "field_abs_tol", "field_rel_tol", "variational_energy_tol",
        "period_abs_tol", "period_rel_tol",
    ), '\t'))

    for source_label in SOURCE_LABELS
        compact_path = only_compact_state(source_run, source_label)
        source = source_metadata(compact_path, full_root)
        source_config_path = joinpath(source_run, "configs", "$source_label.segment-001.toml")
        isfile(source_config_path) || error(
            "source configuration not found for $source_label: $source_config_path",
        )
        source_settings = load_settings(source_config_path)
        LadderMPSMFT.model_fingerprint(source_settings.model) == source.model_fingerprint || error(
            "source config and state model fingerprints differ for $source_label",
        )
        LadderMPSMFT.numerical_fingerprint(source_settings) == source.numerical_fingerprint || error(
            "source config and state numerical fingerprints differ for $source_label",
        )
        source_settings.model.geometry == :square || error("source geometry changed")
        source_settings.model.L == 64 || error("source L changed")
        source_settings.model.U == 8.0 || error("source U changed")
        source_settings.model.V == -0.4 || error("source V changed")
        source_settings.model.t0 == 1.4 || error("source t0 changed")
        source_settings.model.tp == 0.1 || error("source t_perp changed")
        source_settings.model.density == 0.9375 || error("source density changed")
        source_settings.model.ep_mode == :exact || error("source E_p is not exact")
        source_settings.model.ep_signed == EXPECTED_EP_SIGNED || error("source E_p changed")
        source_settings.runtime.backend == :gpu || error("source backend is not GPU")
        source_settings.runtime.tensor_scalar_type == :float64 || error(
            "source config does not request Float64 tensors",
        )

        raw = TOML.parsefile(source_config_path)
        dmrg = raw["dmrg"]
        convergence = raw["convergence"]
        run = raw["run"]
        dmrg["nsweeps"] = 16
        dmrg["maxdim"] = 200
        dmrg["cutoff"] = 1.0e-11
        dmrg["energy_tol"] = 1.0e-9
        dmrg["max_time_seconds"] = 9000.0
        dmrg["mu_density_tol"] = 1.0e-4
        convergence["field_abs_tol"] = 1.0e-7
        convergence["field_rel_tol"] = 1.0e-4
        convergence["density_tol"] = 1.0e-4
        convergence["variational_energy_tol"] = 1.0e-7
        convergence["hamiltonian_identity_tol"] = 1.0e-10
        convergence["effective_energy_consistency_tol"] = 1.0e-8
        convergence["period_abs_tol"] = 2.0e-7
        convergence["period_rel_tol"] = 2.0e-4
        convergence["unmixed_cycle_probe"] = true
        convergence["probe_max_period"] = 2
        convergence["probe_iterations"] = 9
        convergence["cycle_action"] = "stop"
        run["max_iterations"] = 5
        run["require_accepted_solution"] = false
        target_label = replace(source_label, "_loose" => "_tight5")
        full_output_directory = joinpath(full_run, "results", target_label)
        stateless_output_directory = joinpath(control_run, "results", target_label)
        run["output_directory"] = full_output_directory
        run["preparation"] = "square_accepted_parent_tight_five_update_probe"
        run["direction"] = "from_$(basename(source_run))"
        for key in (
            "inherit_from", "inherit_sha256",
            "parent_checkpoint", "parent_sha256", "parent_orbit_phase",
            "resume_checkpoint", "resume_sha256",
        )
            pop!(run, key, nothing)
        end
        run["parent_checkpoint"] = source.full_path
        run["parent_sha256"] = source.full_sha256

        config_path = joinpath(config_directory, "$target_label.segment-001.toml")
        ispath(config_path) && error("refusing to overwrite tight-five config: $config_path")
        open(config_path, "w") do config_io
            TOML.print(config_io, raw; sorted=true)
        end
        settings = load_settings(config_path)
        settings.run.parent_checkpoint == source.full_path || error(
            "prepared branch lost its full parent: $target_label",
        )
        settings.run.parent_sha256 == source.full_sha256 || error(
            "prepared branch lost its parent SHA-256: $target_label",
        )
        settings.run.resume_checkpoint === nothing || error(
            "tight-five branch must begin a fresh history rather than resume one",
        )
        settings.run.max_iterations == 5 || error("tight-five update ceiling changed")
        settings.convergence.probe_iterations == 9 || error("raw probe budget changed")
        settings.convergence.cycle_action == :stop || error("cycle policy changed")
        settings.dmrg.max_time_seconds == 9000.0 || error("DMRG deadline changed")

        model = settings.model
        push!(model_fingerprints, LadderMPSMFT.model_fingerprint(model))
        push!(source_numerical_fingerprints, source.numerical_fingerprint)
        push!(new_numerical_fingerprints, LadderMPSMFT.numerical_fingerprint(settings))
        println(io, join((
            target_label,
            String(model.geometry),
            String(settings.run.branch_label),
            String(settings.run.initial_seed),
            string(settings.run.random_seed),
            config_path,
            LadderMPSMFT.sha256_file(config_path),
            String(model.ep_mode),
            string(model.ep_signed),
            string(model.ep),
            string(model.tp^2 / model.ep),
            basename(source_run),
            source_label,
            compact_path,
            source.compact_sha256,
            source.full_path,
            source.full_sha256,
            String(source.status),
            String(source.solution_kind),
            string(source.period),
            string(source.energy),
            string(source.field_rel_residual),
            string(source.density_error),
            string(source.history_records),
            source.numerical_fingerprint,
            source.implementation_sha256,
            source.ep_source_sha256,
            source.tensor_scalar_type,
            full_output_directory,
            stateless_output_directory,
            "0",
            ANALYSIS_FIELD_FLOOR_SCAN,
            "true",
            "5",
            "true",
            string(settings.dmrg.nsweeps),
            string(settings.dmrg.maxdim),
            string(settings.dmrg.energy_tol),
            string(settings.dmrg.cutoff),
            string(settings.dmrg.max_time_seconds),
            string(settings.dmrg.mu_density_tol),
            string(settings.convergence.density_tol),
            string(settings.convergence.field_abs_tol),
            string(settings.convergence.field_rel_tol),
            string(settings.convergence.variational_energy_tol),
            string(settings.convergence.period_abs_tol),
            string(settings.convergence.period_rel_tol),
        ), '\t'))
    end
end

length(unique(model_fingerprints)) == 1 || error(
    "tight-five branches do not share one model fingerprint",
)
length(unique(source_numerical_fingerprints)) == 1 || error(
    "tight-five parents do not share one source numerical fingerprint",
)
length(unique(new_numerical_fingerprints)) == 1 || error(
    "tight-five branches do not share one new numerical fingerprint",
)

println("source_run=$(basename(source_run))")
println("branch_count=$(length(SOURCE_LABELS))")
println("source_numerical_fingerprint=$(only(unique(source_numerical_fingerprints)))")
println("numerical_fingerprint=$(only(unique(new_numerical_fingerprints)))")
println("map_field_threshold=0")
println("analysis_field_floor_scan=$ANALYSIS_FIELD_FLOOR_SCAN")
println("manifest_path=$manifest_path")
