#!/usr/bin/env julia

using CUDA
using Dates
using HDF5
using ITensorMPS
using ITensors
using LadderMPSMFT
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

length(ARGS) == 1 || error(
    "usage: julia --project=gpu scripts/run_frozen_legacy_gpu.jl CONFIG.toml",
)

function require_scalar(file, name::AbstractString)
    haskey(file, name) || error("legacy file has no $name")
    return read(file, name)
end

function last_history_slice(values)
    ndims(values) >= 1 || error("legacy history must have at least one dimension")
    size(values, ndims(values)) >= 1 || error("legacy history is empty")
    return Array(selectdim(values, ndims(values), size(values, ndims(values))))
end

function legacy_snapshot(path::AbstractString, model::ModelSettings)
    inherited = read_inherited_fields(path)
    inherited.format == :legacy || error("frozen-field evaluator requires a legacy HDF5 source")
    inherited.source_geometry === nothing && error("legacy source has no transverse geometry")
    normalize_geometry(inherited.source_geometry) == model.geometry || error(
        "legacy geometry $(inherited.source_geometry) differs from $(model.geometry)",
    )
    size(inherited.fields.alpha) == (model.L, model.L, 2, 2) || error(
        "legacy alpha shape does not match model.L=$(model.L)",
    )
    size(inherited.fields.beta) == (2, model.L, model.L, 2, 2) || error(
        "legacy beta shape does not match model.L=$(model.L)",
    )
    size(inherited.fields.mu_cdw) == (2, 2 * model.L) || error(
        "legacy mu_cdw shape does not match model.L=$(model.L)",
    )
    return h5open(path, "r") do file
        for (name, expected) in (
            ("U", model.U), ("V", model.V), ("t0", model.t0), ("t_p", model.tp),
        )
            actual = Float64(require_scalar(file, name))
            isapprox(actual, expected; atol=0, rtol=1e-12) || error(
                "legacy $name=$actual differs from current-model $expected",
            )
        end
        pair_history = Float64.(require_scalar(file, "C_pair_list"))
        down_history = Float64.(require_scalar(file, "C_exc_dn_list"))
        up_history = Float64.(require_scalar(file, "C_exc_up_list"))
        pair = last_history_slice(pair_history)
        exchange_down = last_history_slice(down_history)
        exchange_up = last_history_slice(up_history)
        expected_matrix_shape = (2 * model.L, 2 * model.L)
        size(pair) == expected_matrix_shape || error("legacy pair-correlation shape changed")
        size(exchange_down) == expected_matrix_shape || error("legacy down-exchange shape changed")
        size(exchange_up) == expected_matrix_shape || error("legacy up-exchange shape changed")
        correlations = CorrelationState(
            pair,
            exchange_down,
            exchange_up,
            Float64.(diag(exchange_down)),
            Float64.(diag(exchange_up)),
        )
        measured = FieldState(
            last_history_slice(Float64.(require_scalar(file, "alpha_list"))),
            last_history_slice(Float64.(require_scalar(file, "beta_list"))),
            last_history_slice(Float64.(require_scalar(file, "mu_cdw_list"))),
        )
        legacy_abs_residual, legacy_rel_residual = LadderMPSMFT.hybrid_distance(
            measured,
            inherited.fields,
        )
        legacy_effective_energy = Float64(require_scalar(file, "E"))
        reconstructed = variational_energy(
            legacy_effective_energy,
            inherited.chemical_potential,
            inherited.fields,
            correlations,
            model;
            interaction_fields=inherited.fields,
            effective_expectation=legacy_effective_energy,
        )
        scalar(name, default=NaN) = haskey(file, name) ? Float64(read(file, name)) : default
        return (
            fields=inherited.fields,
            chemical_potential=inherited.chemical_potential,
            correlations,
            measured,
            completed=Bool(require_scalar(file, "completed")),
            period2_cycle_detected=Bool(require_scalar(file, "period2_cycle_detected")),
            effective_energy=legacy_effective_energy,
            density=(sum(correlations.density_down) + sum(correlations.density_up)) /
                (2 * model.L),
            field_abs_residual=legacy_abs_residual,
            field_rel_residual=legacy_rel_residual,
            reconstructed,
            order_param=scalar("order_param"),
            dwave_order_param=scalar("dwave_order_param"),
            cdw_order_param=scalar("cdw_order_param"),
            gap=scalar("gap"),
        )
    end
end

function unique_output_directory(settings::ProjectSettings)
    label = join((
        String(settings.model.geometry),
        settings.run.branch_label,
        settings.run.preparation,
        settings.run.direction,
        settings.run.seed_label,
    ), "__")
    safe_label = replace(label, r"[^A-Za-z0-9_.-]+" => "-")
    stamp = Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS")
    suffix = first(LadderMPSMFT.model_fingerprint(settings.model), 12)
    directory = joinpath(settings.run.output_directory, safe_label, "$(stamp)_$(getpid())_$suffix")
    ispath(directory) && error("refusing to overwrite frozen-field output: $directory")
    mkpath(directory)
    return directory
end

function write_key_values(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "quantity\tvalue")
        for (name, value) in rows
            println(io, name, '\t', value)
        end
    end
    return path
end

config_path = abspath(ARGS[1])
raw = TOML.parsefile(config_path)
haskey(raw, "frozen_legacy") || error("config has no [frozen_legacy] contract")
contract = raw["frozen_legacy"]
get(contract, "policy", "") ==
    "single_fresh_dmrg_at_saved_legacy_fields_and_mu_no_mf_update_no_mu_search" || error(
        "unsupported frozen-field policy",
    )
get(contract, "selection_eligible", true) == false || error(
    "frozen-field diagnostic must be selection-ineligible",
)

settings = load_settings(config_path)
validate_settings(settings)
target_label = basename(settings.run.output_directory)
settings.runtime.backend == :gpu || error("frozen-field entry point requires runtime.backend=gpu")
settings.runtime.tensor_scalar_type == :float64 || error("frozen-field entry point requires Float64")
settings.run.max_iterations == 1 || error("frozen-field diagnostic must permit exactly one evaluation")
settings.run.inherit_from === nothing && error("frozen-field config has no legacy source")
settings.run.parent_checkpoint === nothing || error("frozen-field diagnostic cannot load a parent MPS")
settings.run.resume_checkpoint === nothing || error("frozen-field diagnostic cannot resume an MPS")

legacy_path = realpath(String(contract["source_path"]))
legacy_sha256 = lowercase(String(contract["source_sha256"]))
LadderMPSMFT.sha256_file(legacy_path) == legacy_sha256 || error("legacy source SHA-256 mismatch")
settings.run.inherit_from == legacy_path || error("run inheritance path differs from frozen contract")
settings.run.inherit_sha256 == legacy_sha256 || error("run inheritance hash differs from frozen contract")

reference_labels = String.(contract["reference_labels"])
reference_paths = realpath.(String.(contract["reference_states"]))
reference_hashes = lowercase.(String.(contract["reference_state_sha256"]))
length(reference_labels) == length(reference_paths) == length(reference_hashes) || error(
    "reference label/path/hash counts differ",
)
length(reference_paths) == 6 || error("frozen comparison requires the six accepted reference states")
for (path, expected) in zip(reference_paths, reference_hashes)
    LadderMPSMFT.sha256_file(path) == expected || error("reference state SHA-256 mismatch: $path")
end
ranked_references = compare_variational_branches(reference_paths)

expected_model = String(contract["expected_model_fingerprint"])
expected_numerical = String(contract["expected_numerical_fingerprint"])
expected_implementation = String(contract["expected_implementation_sha256"])
expected_ep = String(contract["expected_ep_source_sha256"])
LadderMPSMFT.model_fingerprint(settings.model) == expected_model || error(
    "prepared model fingerprint differs from the reference campaign",
)
LadderMPSMFT.numerical_fingerprint(settings) == expected_numerical || error(
    "prepared numerical fingerprint differs from the reference campaign",
)
implementation_fingerprint(settings) == expected_implementation || error(
    "current src/ plus GPU Manifest differ from the reference campaign",
)
all(row -> row.fingerprint == expected_model, ranked_references) || error(
    "reference model fingerprint changed",
)
all(row -> row.numerical_fingerprint == expected_numerical, ranked_references) || error(
    "reference numerical fingerprint changed",
)
all(row -> row.implementation_sha256 == expected_implementation, ranked_references) || error(
    "reference implementation fingerprint changed",
)
all(row -> row.ep_source_sha256 == expected_ep, ranked_references) || error(
    "reference E_p registry fingerprint changed",
)

preflight = gpu_linalg_preflight!()
println("gpu_linalg_preflight_dimension=$(preflight.dimension)")
println("gpu_linalg_preflight_scalar_type=$(preflight.scalar_type)")
println("gpu_tensor_scalar_type=$(settings.runtime.tensor_scalar_type)")
println("gpu_cuda_runtime=$(CUDA.runtime_version())")
println("gpu_cuda_driver=$(CUDA.driver_version())")
println("gpu_cuda_toolkit_source=$(CUDA.local_toolkit ? "local" : "artifact")")
ensure_backend!(settings.runtime)
threading = configure_threading!(settings.runtime)

legacy = legacy_snapshot(legacy_path, settings.model)
output_directory = unique_output_directory(settings)
sites = LadderMPSMFT.make_sites(settings.model, settings.runtime)
effective_hamiltonian = build_mf_mpo(
    sites,
    settings.model,
    legacy.fields,
    legacy.chemical_potential;
    backend=settings.runtime,
)
bare_hamiltonian = LadderMPSMFT.build_bare_ladder_mpo(
    sites,
    settings.model;
    backend=settings.runtime,
)

start_time = time()
dmrg_result = run_dmrg_ground(
    sites,
    effective_hamiltonian,
    settings.model.density,
    settings.dmrg;
    rng=MersenneTwister(settings.run.random_seed),
    deadline=start_time + settings.dmrg.max_time_seconds,
    backend=settings.runtime,
)
psi = dmrg_result.psi
density = LadderMPSMFT.average_density(psi)
measured, correlations = calculate_mean_fields(psi, settings.model; threshold=0.0)
absolute_residual, relative_residual = LadderMPSMFT.hybrid_distance(measured, legacy.fields)
effective_expectation = Float64(real(inner(psi', effective_hamiltonian, psi)))
bare_energy = Float64(real(inner(psi', bare_hamiltonian, psi)))
energy = variational_energy(
    dmrg_result.energy,
    legacy.chemical_potential,
    legacy.fields,
    correlations,
    settings.model;
    interaction_fields=legacy.fields,
    effective_expectation,
    bare_ladder_energy=bare_energy,
)
record = IterationRecord(;
    iteration=1,
    update_mode=:frozen_legacy_evaluation,
    applied=copy(legacy.fields),
    measured,
    correlations,
    density,
    chemical_potential=legacy.chemical_potential,
    mu_search_status=:fixed_legacy_mu_no_search,
    mu_evaluations=1,
    mu_density_converged=abs(density - settings.model.density) <= settings.dmrg.mu_density_tol,
    effective_energy=dmrg_result.energy,
    variational=energy,
    field_abs_residual=absolute_residual,
    field_rel_residual=relative_residual,
    wall_seconds=time() - start_time,
    dmrg_max_discarded_weight=dmrg_result.max_discarded_weight,
    dmrg_maxlinkdim=dmrg_result.maximum_link_dimension,
    dmrg_sweep_energies=dmrg_result.sweep_energies,
    dmrg_sweep_max_discarded_weights=dmrg_result.sweep_max_discarded_weights,
    dmrg_sweep_maxlinkdims=dmrg_result.sweep_maxlinkdims,
)
number_sites = 2 * settings.model.L
diagnostic = ConvergenceDiagnostic(;
    status=dmrg_result.timed_out ? :time_limit : :frozen_field_evaluation,
    accepted=false,
    reason=dmrg_result.timed_out ?
        "single frozen-field DMRG reached its internal deadline; no MF update or mu search was run" :
        "single frozen-field DMRG completed; diagnostic is not an SCF acceptance test",
    solution_kind=:diagnostic,
    fundamental_period=0,
    orbit_validated=false,
    unmixed_probe=false,
    solution_canonical_variational_energy=energy.canonical_variational_energy,
    solution_target_density_corrected_variational_energy=
        energy.target_density_corrected_variational_energy,
    fixed_point_abs_residual=absolute_residual,
    fixed_point_rel_residual=relative_residual,
    density_error=abs(density - settings.model.density),
    hamiltonian_identity_error_per_site=abs(energy.hamiltonian_identity_error) / number_sites,
    effective_eigenvalue_error_per_site=abs(energy.effective_eigenvalue_error) / number_sites,
    best_iteration=1,
)

provenance = collect_provenance(settings)
provenance["analysis_role"] = "frozen_legacy_one_shot_energy_diagnostic"
provenance["selection_eligible"] = false
provenance["dmrg_solve_count"] = 1
provenance["mf_update_count"] = 0
provenance["mu_search_count"] = 0
provenance["legacy_source_sha256"] = legacy_sha256
provenance["legacy_saved_effective_energy"] = legacy.effective_energy
provenance["legacy_saved_mu"] = legacy.chemical_potential
provenance["legacy_saved_density"] = legacy.density
provenance["legacy_saved_field_abs_residual"] = legacy.field_abs_residual
provenance["legacy_saved_field_rel_residual"] = legacy.field_rel_residual
provenance["legacy_completed_flag"] = legacy.completed
provenance["legacy_period2_cycle_detected"] = legacy.period2_cycle_detected
provenance["reference_run"] = String(contract["reference_run"])
provenance["reference_labels"] = join(reference_labels, "\n")
provenance["reference_state_sha256"] = join(reference_hashes, "\n")
provenance["threading"] = Dict(string(key) => value for (key, value) in pairs(threading))
provenance["device"] = LadderMPSMFT.backend_metadata(settings.runtime)

state_path = joinpath(output_directory, "state.h5")
write_checkpoint(
    state_path;
    settings,
    psi,
    records=[record],
    diagnostic,
    restart_fields=measured,
    chemical_potential=legacy.chemical_potential,
    provenance,
    immutable=true,
)
state_sha256 = LadderMPSMFT.sha256_file(state_path)

diagnostics = compute_ladder_diagnostics(psi, settings.model; full_pair_correlations=false)
diagnostics_path = joinpath(output_directory, "diagnostics.h5")
write_diagnostics(
    diagnostics_path,
    diagnostics;
    state_sha256,
    metadata=Dict(
        "analysis_role" => "frozen_legacy_one_shot_energy_diagnostic",
        "selection_eligible" => false,
        "legacy_source_sha256" => legacy_sha256,
    ),
    immutable=true,
)

legacy_table = write_key_values(joinpath(output_directory, "legacy_saved_observables.tsv"), [
    "legacy_completed" => legacy.completed,
    "legacy_period2_cycle_detected" => legacy.period2_cycle_detected,
    "legacy_effective_hamiltonian_energy_not_rankable" => legacy.effective_energy,
    "legacy_mu" => legacy.chemical_potential,
    "legacy_density_from_saved_correlations" => legacy.density,
    "legacy_field_abs_residual_from_saved_last_map" => legacy.field_abs_residual,
    "legacy_field_rel_residual_from_saved_last_map" => legacy.field_rel_residual,
    "legacy_reconstructed_canonical_energy_non_authoritative" =>
        legacy.reconstructed.canonical_variational_energy,
    "legacy_reconstructed_target_density_energy_non_authoritative" =>
        legacy.reconstructed.target_density_corrected_variational_energy,
    "legacy_order_param" => legacy.order_param,
    "legacy_dwave_order_param" => legacy.dwave_order_param,
    "legacy_cdw_order_param" => legacy.cdw_order_param,
    "legacy_gap" => legacy.gap,
])

frozen_table = write_key_values(joinpath(output_directory, "frozen_dmrg_observables.tsv"), [
    "selection_eligible" => false,
    "dmrg_timed_out" => dmrg_result.timed_out,
    "dmrg_energy_converged" => dmrg_result.energy_converged,
    "dmrg_sweeps_recorded" => length(dmrg_result.sweep_energies),
    "dmrg_max_discarded_weight" => dmrg_result.max_discarded_weight,
    "dmrg_maxlinkdim" => dmrg_result.maximum_link_dimension,
    "saved_legacy_mu_used_without_search" => legacy.chemical_potential,
    "density" => density,
    "density_error" => diagnostic.density_error,
    "raw_map_field_abs_residual" => absolute_residual,
    "raw_map_field_rel_residual" => relative_residual,
    "effective_eigenvalue" => energy.effective_eigenvalue,
    "effective_expectation" => energy.effective_expectation,
    "bare_ladder_energy" => energy.bare_ladder_energy,
    "canonical_variational_energy" => energy.canonical_variational_energy,
    "target_density_correction" => energy.target_density_correction,
    "target_density_corrected_variational_energy" =>
        energy.target_density_corrected_variational_energy,
    "double_counting_correction" => energy.double_counting_correction,
    "hamiltonian_identity_error_per_site" => diagnostic.hamiltonian_identity_error_per_site,
    "effective_eigenvalue_error_per_site" => diagnostic.effective_eigenvalue_error_per_site,
    "charge_peak_qx_over_pi" => diagnostics.charge_peak.qx_over_pi,
    "charge_peak_ky_over_pi" => diagnostics.charge_peak.ky_over_pi,
    "spin_peak_qx_over_pi" => diagnostics.spin_peak.qx_over_pi,
    "spin_peak_ky_over_pi" => diagnostics.spin_peak.ky_over_pi,
    "spin_q_mismatch" => diagnostics.spin_q_mismatch,
    "K_rho_site_normalized" => diagnostics.K_rho.K_rho_site_normalized,
    "K_rho_rung_normalized" => diagnostics.K_rho.K_rho_rung_normalized,
    "central_charge" => diagnostics.central_charge.central_charge,
])

label_by_path = Dict(normpath(path) => label for (path, label) in zip(reference_paths, reference_labels))
comparison_path = joinpath(output_directory, "energy_comparison.tsv")
minimum_reference_energy = minimum(row.energy for row in ranked_references)
open(comparison_path, "w") do io
    println(io, join((
        "rank", "role", "selection_eligible", "accepted", "label", "energy_kind",
        "target_density_corrected_energy", "delta_to_accepted_min", "state_path",
    ), '\t'))
    for (rank, row) in enumerate(ranked_references)
        label = label_by_path[normpath(row.path)]
        println(io, join((
            string(rank), "accepted_reference", "true", "true", label, row.energy_kind,
            string(row.energy), string(row.energy - minimum_reference_energy), row.path,
        ), '\t'))
    end
    println(io, join((
        "NA", "frozen_legacy_diagnostic", "false", "false", target_label,
        "single_dmrg_target_density_corrected",
        string(energy.target_density_corrected_variational_energy),
        string(energy.target_density_corrected_variational_energy - minimum_reference_energy),
        state_path,
    ), '\t'))
end

summary_path = joinpath(output_directory, "run_summary.md")
open(summary_path, "w") do io
    println(io, "# Frozen legacy-field one-shot DMRG")
    println(io)
    println(io, "This job ran exactly one fresh DMRG at the terminal legacy fields and saved legacy chemical potential. It ran no chemical-potential search and no mean-field update.")
    println(io)
    println(io, "- Selection eligibility: **false** (diagnostic only; not an accepted SCF state)")
    println(io, "- Legacy source SHA-256: `$legacy_sha256`")
    println(io, "- DMRG timed out: `$(dmrg_result.timed_out)`")
    println(io, "- Density: `$(density)` (target `$(settings.model.density)`)")
    println(io, "- One-step raw-map residual: abs `$(absolute_residual)`, rel `$(relative_residual)`")
    println(io, "- Canonical energy: `$(energy.canonical_variational_energy)`")
    println(io, "- Target-density-corrected energy: `$(energy.target_density_corrected_variational_energy)`")
    println(io, "- Delta to the minimum of the six accepted references: `$(energy.target_density_corrected_variational_energy - minimum_reference_energy)`")
    println(io, "- Maximum discarded weight: `$(dmrg_result.max_discarded_weight)`")
    println(io, "- Realized maximum link dimension: `$(dmrg_result.maximum_link_dimension)`")
    println(io)
    println(io, "The six accepted references retain their formal within-fingerprint ranking. The frozen legacy row is reported beside them only as the requested conditional energetic diagnostic.")
end

println("state_path=$state_path")
println("state_sha256=$state_sha256")
println("diagnostics_path=$diagnostics_path")
println("legacy_observables_path=$legacy_table")
println("frozen_observables_path=$frozen_table")
println("energy_comparison_path=$comparison_path")
println("summary_path=$summary_path")
println("status=$(diagnostic.status)")
println("accepted=$(diagnostic.accepted)")
println("selection_eligible=false")
println("dmrg_solves=1")
println("mf_updates=0")
println("mu_searches=0")
println("target_density_corrected_energy=$(energy.target_density_corrected_variational_energy)")
println("delta_to_accepted_min=$(energy.target_density_corrected_variational_energy - minimum_reference_energy)")

dmrg_result.timed_out && exit(2)
