#!/usr/bin/env julia

using HDF5
using LadderMPSMFT
using TOML

length(ARGS) == 6 || error(
    "usage: julia --project=. scripts/prepare_phase1_square_v0_chi400_compare.jl " *
    "BASE_CONFIG.toml PAIRING_SOURCE_RUN FROZEN_SOURCE_RUN CONTROL_RUN FULL_RUN RUN_ID",
)

const PAIRING_SOURCE_LABEL = "square__pairing_dwave_m000_chi200_loose"
const FROZEN_SOURCE_LABEL = "square__legacy_terminal_fields_frozen_dmrg_chi200"
const EXPECTED_EP_SIGNED = -0.14653773091916378
const INACTIVE_ONSITE_BETA_TOL = 1.0e-12

base_path = abspath(ARGS[1])
pairing_run = realpath(ARGS[2])
frozen_run = realpath(ARGS[3])
control_run = abspath(ARGS[4])
full_run = abspath(ARGS[5])
run_id = ARGS[6]

isfile(base_path) || error("chi=400 comparison base configuration not found: $base_path")
isdir(pairing_run) || error("pairing source run not found: $pairing_run")
isdir(frozen_run) || error("frozen legacy source run not found: $frozen_run")
isdir(full_run) || error("full scratch output directory not found: $full_run")
occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID: $run_id")

function only_analysis_state(source_run::AbstractString, label::AbstractString)
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

function state_metadata(path::AbstractString)
    return h5open(path, "r") do file
        required = (
            "analysis_storage/is_stateless_copy",
            "analysis_storage/full_artifact_path",
            "analysis_storage/full_artifact_sha256",
            "accepted", "completed", "status", "solution_kind", "fundamental_period",
            "orbit_validated", "density_error", "chemical_potential",
            "fixed_point_abs_residual", "fixed_point_rel_residual",
            "solution_canonical_variational_energy",
            "solution_target_density_corrected_variational_energy",
            "history/iteration", "history/update_mode",
            "model/transverse_geometry", "provenance/model_fingerprint",
            "provenance/numerical_fingerprint", "provenance/implementation_sha256",
            "provenance/ep_source_sha256", "provenance/tensor_scalar_type",
        )
        for name in required
            haskey(file, name) || error("source state has no $name: $path")
        end
        Bool(read(file, "analysis_storage/is_stateless_copy")) || error(
            "source is not a stateless analysis copy: $path",
        )
        return (
            path=path,
            compact_sha256=LadderMPSMFT.sha256_file(path),
            full_path=String(read(file, "analysis_storage/full_artifact_path")),
            full_sha256=lowercase(String(read(file, "analysis_storage/full_artifact_sha256"))),
            accepted=Bool(read(file, "accepted")),
            completed=Bool(read(file, "completed")),
            status=Symbol(read(file, "status")),
            solution_kind=Symbol(read(file, "solution_kind")),
            period=Int(read(file, "fundamental_period")),
            orbit_validated=Bool(read(file, "orbit_validated")),
            selection_eligible=haskey(file, "provenance/selection_eligible") ?
                Bool(read(file, "provenance/selection_eligible")) : true,
            density_error=Float64(read(file, "density_error")),
            chemical_potential=Float64(read(file, "chemical_potential")),
            field_abs_residual=Float64(read(file, "fixed_point_abs_residual")),
            field_rel_residual=Float64(read(file, "fixed_point_rel_residual")),
            canonical_energy=Float64(read(file, "solution_canonical_variational_energy")),
            target_energy=Float64(read(
                file,
                "solution_target_density_corrected_variational_energy",
            )),
            history_records=length(read(file, "history/iteration")),
            update_modes=String.(read(file, "history/update_mode")),
            geometry=String(read(file, "model/transverse_geometry")),
            model_fingerprint=String(read(file, "provenance/model_fingerprint")),
            numerical_fingerprint=String(read(file, "provenance/numerical_fingerprint")),
            implementation_sha256=String(read(file, "provenance/implementation_sha256")),
            ep_source_sha256=String(read(file, "provenance/ep_source_sha256")),
            tensor_scalar_type=lowercase(String(read(file, "provenance/tensor_scalar_type"))),
        )
    end
end

function full_parent_metadata(source_run::AbstractString, compact, model::ModelSettings)
    locator = joinpath(source_run, "full_storage_path.txt")
    isfile(locator) || error("source run has no full_storage_path.txt: $source_run")
    full_root = strip(read(locator, String))
    isdir(full_root) || error(
        "source full storage is unavailable; run preparation on Perlmutter: $full_root",
    )
    isfile(compact.full_path) || error(
        "source full parent is unavailable; run preparation on Perlmutter: $(compact.full_path)",
    )
    resolved_root = realpath(full_root)
    resolved_full = realpath(compact.full_path)
    relative = relpath(resolved_full, resolved_root)
    (relative == ".." || startswith(relative, "..$(Base.Filesystem.path_separator)")) && error(
        "source full parent escapes its recorded scratch run: $resolved_full",
    )
    actual_sha256 = LadderMPSMFT.sha256_file(resolved_full)
    actual_sha256 == compact.full_sha256 || error(
        "source full parent SHA-256 differs from its compact link: $resolved_full",
    )

    full = h5open(resolved_full, "r") do file
        for name in (
            "psi", "fields/applied", "fields/measured", "fields/restart",
            "chemical_potential", "accepted", "completed", "status", "solution_kind",
            "fundamental_period", "orbit_validated", "provenance/model_fingerprint",
            "provenance/numerical_fingerprint", "provenance/implementation_sha256",
            "provenance/ep_source_sha256", "provenance/tensor_scalar_type",
        )
            haskey(file, name) || error("source full parent has no $name: $resolved_full")
        end
        return (
            accepted=Bool(read(file, "accepted")),
            completed=Bool(read(file, "completed")),
            status=Symbol(read(file, "status")),
            solution_kind=Symbol(read(file, "solution_kind")),
            period=Int(read(file, "fundamental_period")),
            orbit_validated=Bool(read(file, "orbit_validated")),
            chemical_potential=Float64(read(file, "chemical_potential")),
            model_fingerprint=String(read(file, "provenance/model_fingerprint")),
            numerical_fingerprint=String(read(file, "provenance/numerical_fingerprint")),
            implementation_sha256=String(read(file, "provenance/implementation_sha256")),
            ep_source_sha256=String(read(file, "provenance/ep_source_sha256")),
            tensor_scalar_type=lowercase(String(read(file, "provenance/tensor_scalar_type"))),
        )
    end
    for key in (
        :accepted, :completed, :status, :solution_kind, :period, :orbit_validated,
        :chemical_potential, :model_fingerprint, :numerical_fingerprint,
        :implementation_sha256, :ep_source_sha256, :tensor_scalar_type,
    )
        getfield(full, key) == getfield(compact, key) || error(
            "full and compact source metadata differ for $key: $(compact.path)",
        )
    end

    inherited = read_inherited_fields(resolved_full)
    inherited.format == :refactored || error("full parent is not a refactored checkpoint")
    fields = inherited.fields
    all(isfinite, fields.alpha) || error("parent restart alpha contains nonfinite values")
    all(isfinite, fields.beta) || error("parent restart beta contains nonfinite values")
    all(isfinite, fields.mu_cdw) || error("parent restart mu_cdw contains nonfinite values")
    onsite_beta_max = maximum(
        abs(fields.beta[spin, rung, rung, leg, leg])
        for spin in axes(fields.beta, 1), rung in axes(fields.beta, 2), leg in axes(fields.beta, 4)
    )
    return (
        full_path=resolved_full,
        full_sha256=actual_sha256,
        restart_onsite_beta_max=Float64(onsite_beta_max),
        restart_field_l2_per_site=field_l2_per_physical_site(fields, model),
    )
end

function source_config(source_run::AbstractString, label::AbstractString)
    path = joinpath(source_run, "configs", "$label.segment-001.toml")
    isfile(path) || error("source configuration not found: $path")
    return path, load_settings(path)
end

base = load_settings(base_path)
base.model.geometry == :square || error("comparison base geometry must be square")
base.model.L == 64 || error("comparison base L must be 64")
base.model.U == 8.0 || error("comparison base U must be 8")
base.model.V == 0.0 || error("comparison base V must be 0")
base.model.t0 == 1.4 || error("comparison base t0 must be 1.4")
base.model.tp == 0.1 || error("comparison base t_perp must be 0.1")
base.model.density == 0.9375 || error("comparison base density must be 0.9375")
base.model.ep_mode == :exact || error("comparison requires an exact E_p registry row")
base.model.ep_signed == EXPECTED_EP_SIGNED || error("comparison resolved an unexpected E_p")
base.runtime.backend == :gpu || error("comparison must use the GPU backend")
base.runtime.tensor_scalar_type == :float64 || error("comparison must use Float64 tensors")
base.dmrg.nsweeps == 16 || error("comparison must use 16 DMRG sweeps")
base.dmrg.maxdim == 400 || error("comparison must use chi=400")
base.dmrg.cutoff == 1.0e-11 || error("comparison must use cutoff=1e-11")
base.dmrg.energy_tol == 1.0e-9 || error("comparison must use DMRG energy_tol=1e-9")
base.dmrg.max_time_seconds == 41400.0 || error("comparison DMRG deadline changed")
base.dmrg.mu_density_tol == 1.0e-4 || error("inner density tolerance must be 1e-4")
base.dmrg.mu_warm_start_noise == 1.0e-8 || error("warm mu re-solve noise must be 1e-8")
base.convergence.density_tol == 1.0e-4 || error("outer density tolerance must be 1e-4")
base.convergence.field_abs_tol == 1.0e-7 || error("field absolute tolerance must be 1e-7")
base.convergence.field_rel_tol == 1.0e-4 || error("field relative tolerance must be 1e-4")
base.convergence.variational_energy_tol == 1.0e-7 || error(
    "variational energy tolerance must be 1e-7 per physical site",
)
base.convergence.unmixed_cycle_probe || error("comparison must preserve a raw-map probe")
base.convergence.probe_iterations == 20 || error("raw-map probe must allow 20 updates")
base.convergence.cycle_action == :continue || error(
    "unaccepted raw recurrence must be archived before optional Anderson acceleration",
)
base.run.max_iterations == 80 || error("comparison must allow up to 80 MF updates")

pairing_compact = state_metadata(only_analysis_state(pairing_run, PAIRING_SOURCE_LABEL))
pairing_compact.accepted || error("pairing source is not accepted")
pairing_compact.completed || error("pairing source is not completed")
pairing_compact.status == :fixed_point || error("pairing source is not a fixed point")
pairing_compact.solution_kind == :fixed_point || error("pairing source kind is not fixed_point")
pairing_compact.period == 1 || error("pairing source does not have period one")
!pairing_compact.orbit_validated || error("period-one pairing source unexpectedly claims an orbit")

frozen_compact = state_metadata(only_analysis_state(frozen_run, FROZEN_SOURCE_LABEL))
!frozen_compact.accepted || error("frozen diagnostic unexpectedly claims acceptance")
frozen_compact.status == :frozen_field_evaluation || error(
    "legacy source is not the frozen-field diagnostic",
)
frozen_compact.solution_kind == :diagnostic || error("legacy source kind is not diagnostic")
frozen_compact.period == 0 || error("frozen diagnostic unexpectedly claims a period")
!frozen_compact.orbit_validated || error("frozen diagnostic unexpectedly claims an orbit")
!frozen_compact.selection_eligible || error("frozen diagnostic unexpectedly claims selection eligibility")
frozen_compact.history_records == 1 || error("frozen diagnostic must contain one DMRG record")
frozen_compact.update_modes == ["frozen_legacy_evaluation"] || error(
    "frozen diagnostic has an unexpected update history",
)

for source in (pairing_compact, frozen_compact)
    source.geometry == "square" || error("source geometry is not square")
    source.tensor_scalar_type == "float64" || error("source tensor type is not Float64")
    isfinite(source.canonical_energy) || error("source canonical energy is not finite")
    isfinite(source.target_energy) || error("source target-density energy is not finite")
end
for key in (
    :model_fingerprint, :numerical_fingerprint, :implementation_sha256,
    :ep_source_sha256, :tensor_scalar_type,
)
    getfield(pairing_compact, key) == getfield(frozen_compact, key) || error(
        "pairing and frozen source metadata differ for $key",
    )
end

pairing_config_path, pairing_settings = source_config(pairing_run, PAIRING_SOURCE_LABEL)
frozen_config_path, frozen_settings = source_config(frozen_run, FROZEN_SOURCE_LABEL)
for (path, settings, metadata) in (
    (pairing_config_path, pairing_settings, pairing_compact),
    (frozen_config_path, frozen_settings, frozen_compact),
)
    LadderMPSMFT.model_fingerprint(settings.model) == metadata.model_fingerprint || error(
        "source config and state model fingerprints differ: $path",
    )
    LadderMPSMFT.numerical_fingerprint(settings) == metadata.numerical_fingerprint || error(
        "source config and state numerical fingerprints differ: $path",
    )
end

new_model_fingerprint = LadderMPSMFT.model_fingerprint(base.model)
new_model_fingerprint == pairing_compact.model_fingerprint || error(
    "chi=400 comparison model differs from its chi=200 parents",
)
current_implementation = implementation_fingerprint(base)
current_implementation == pairing_compact.implementation_sha256 || error(
    "current src/ plus GPU Manifest differ from the parent solver implementation",
)
LadderMPSMFT.sha256_file(base.model.ep_source) == pairing_compact.ep_source_sha256 || error(
    "current E_p registry differs from the parent registry",
)

pairing_full = full_parent_metadata(pairing_run, pairing_compact, base.model)
frozen_full = full_parent_metadata(frozen_run, frozen_compact, base.model)
pairing_full.restart_onsite_beta_max <= INACTIVE_ONSITE_BETA_TOL || error(
    "pairing restart retains inactive onsite beta entries",
)
frozen_full.restart_onsite_beta_max <= INACTIVE_ONSITE_BETA_TOL || error(
    "frozen restart was not sanitized to its measured physical-field map",
)

branches = (
    (
        label="square__pairing_dwave_m000_chi400_tight",
        branch_label="pairing_bond_dimension_control",
        direction="from_chi200_pairing_dwave_m000",
        seed_label="accepted_chi200_pairing_dwave_m000_parent",
        role="representative_pairing_lineage",
        parent_run=pairing_run,
        parent_label=PAIRING_SOURCE_LABEL,
        parent=pairing_compact,
        full=pairing_full,
        restart_policy="accepted_period1_measured_restart",
    ),
    (
        label="square__legacy_like_continuation_chi400_tight",
        branch_label="legacy_like_basin_continuation",
        direction="from_chi200_frozen_legacy_measured_map",
        seed_label="frozen_legacy_chi200_measured_parent",
        role="legacy_like_high_amplitude_basin_lineage",
        parent_run=frozen_run,
        parent_label=FROZEN_SOURCE_LABEL,
        parent=frozen_compact,
        full=frozen_full,
        restart_policy="frozen_one_step_measured_restart_inactive_onsite_beta_zero",
    ),
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
new_numerical_fingerprints = String[]

open(manifest_path, "w") do io
    println(io, join((
        "label", "geometry", "L", "U", "V", "t0", "tp", "density", "chi",
        "comparison_role", "config", "config_sha256", "source_run", "source_label",
        "source_compact_state", "source_compact_sha256", "parent_checkpoint",
        "parent_sha256", "parent_status", "parent_accepted", "parent_selection_eligible",
        "parent_period", "parent_canonical_energy", "parent_target_corrected_energy",
        "parent_density_error", "parent_stored_field_abs_residual",
        "parent_stored_field_rel_residual", "parent_history_records",
        "parent_numerical_fingerprint", "parent_implementation_sha256",
        "parent_ep_source_sha256", "parent_tensor_scalar_type", "restart_policy",
        "restart_onsite_beta_max", "restart_field_l2_per_site",
        "full_output_directory", "stateless_output_directory", "dmrg_sweeps",
        "dmrg_maxdim", "dmrg_cutoff", "dmrg_energy_tol", "dmrg_max_time_seconds",
        "mu_density_tol", "outer_density_tol", "field_abs_tol", "field_rel_tol",
        "variational_energy_tol", "stable_iterations", "raw_probe_updates",
        "anderson_after_recurrence_free_probe", "max_mf_updates", "requested_time",
        "requested_node_hours", "bond_dimension_comparison", "thermodynamic_phase_claim",
    ), '\t'))

    for branch in branches
        raw = TOML.parsefile(base_path)
        run = raw["run"]
        for key in (
            "inherit_from", "inherit_sha256", "parent_checkpoint", "parent_sha256",
            "parent_orbit_phase", "resume_checkpoint", "resume_sha256",
        )
            pop!(run, key, nothing)
        end
        full_output_directory = joinpath(full_run, "results", branch.label)
        stateless_output_directory = joinpath(control_run, "results", branch.label)
        run["output_directory"] = full_output_directory
        run["branch_label"] = branch.branch_label
        run["preparation"] = "square_v0_chi200_parent_to_chi400_tight_basin_comparison"
        run["direction"] = branch.direction
        run["seed_label"] = branch.seed_label
        run["parent_checkpoint"] = branch.full.full_path
        run["parent_sha256"] = branch.full.full_sha256

        config_path = joinpath(config_directory, "$(branch.label).segment-001.toml")
        ispath(config_path) && error("refusing to overwrite comparison config: $config_path")
        open(config_path, "w") do config_io
            TOML.print(config_io, raw; sorted=true)
        end
        settings = load_settings(config_path)
        settings.run.parent_checkpoint == branch.full.full_path || error(
            "prepared branch lost its full parent: $(branch.label)",
        )
        settings.run.parent_sha256 == branch.full.full_sha256 || error(
            "prepared branch lost its parent SHA-256: $(branch.label)",
        )
        settings.run.resume_checkpoint === nothing || error(
            "chi=400 comparison must start a fresh history from a parent",
        )
        LadderMPSMFT.model_fingerprint(settings.model) == new_model_fingerprint || error(
            "prepared branch model changed: $(branch.label)",
        )
        implementation_fingerprint(settings) == current_implementation || error(
            "prepared branch implementation changed: $(branch.label)",
        )
        push!(new_numerical_fingerprints, LadderMPSMFT.numerical_fingerprint(settings))

        model = settings.model
        println(io, join((
            branch.label, String(model.geometry), string(model.L), string(model.U),
            string(model.V), string(model.t0), string(model.tp), string(model.density),
            string(settings.dmrg.maxdim), branch.role, config_path,
            LadderMPSMFT.sha256_file(config_path), basename(branch.parent_run),
            branch.parent_label, branch.parent.path, branch.parent.compact_sha256,
            branch.full.full_path, branch.full.full_sha256, String(branch.parent.status),
            string(branch.parent.accepted), string(branch.parent.selection_eligible),
            string(branch.parent.period), string(branch.parent.canonical_energy),
            string(branch.parent.target_energy), string(branch.parent.density_error),
            string(branch.parent.field_abs_residual), string(branch.parent.field_rel_residual),
            string(branch.parent.history_records), branch.parent.numerical_fingerprint,
            branch.parent.implementation_sha256, branch.parent.ep_source_sha256,
            branch.parent.tensor_scalar_type, branch.restart_policy,
            string(branch.full.restart_onsite_beta_max),
            string(branch.full.restart_field_l2_per_site), full_output_directory,
            stateless_output_directory, string(settings.dmrg.nsweeps),
            string(settings.dmrg.maxdim), string(settings.dmrg.cutoff),
            string(settings.dmrg.energy_tol), string(settings.dmrg.max_time_seconds),
            string(settings.dmrg.mu_density_tol), string(settings.convergence.density_tol),
            string(settings.convergence.field_abs_tol),
            string(settings.convergence.field_rel_tol),
            string(settings.convergence.variational_energy_tol),
            string(settings.convergence.stable_iterations),
            string(settings.convergence.probe_iterations), "true",
            string(settings.run.max_iterations), "12:00:00", "3.000000000", "true", "false",
        ), '\t'))
    end
end

length(unique(new_numerical_fingerprints)) == 1 || error(
    "chi=400 comparison branches do not share one numerical fingerprint",
)
new_numerical_fingerprint = only(unique(new_numerical_fingerprints))
new_numerical_fingerprint != pairing_compact.numerical_fingerprint || error(
    "chi=400 tight numerical fingerprint unexpectedly equals the chi=200 loose parent",
)

println("pairing_source_run=$(basename(pairing_run))")
println("frozen_source_run=$(basename(frozen_run))")
println("branch_count=$(length(branches))")
println("chi=400")
println("density_tolerance=1.0e-4")
println("field_relative_tolerance=1.0e-4")
println("parent_numerical_fingerprint=$(pairing_compact.numerical_fingerprint)")
println("numerical_fingerprint=$new_numerical_fingerprint")
println("implementation_sha256=$current_implementation")
println("pairing_parent=$(pairing_full.full_path)")
println("frozen_parent=$(frozen_full.full_path)")
println("frozen_restart_onsite_beta_max=$(frozen_full.restart_onsite_beta_max)")
println("manifest_path=$manifest_path")
