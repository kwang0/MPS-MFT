#!/usr/bin/env julia

using HDF5
using LadderMPSMFT
using TOML

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/finalize_frozen_legacy_result.jl CONFIG.toml STATE.h5",
)

function atomic_write(writer, path::AbstractString)
    mkpath(dirname(path))
    temporary = tempname(dirname(path))
    try
        open(writer, temporary, "w")
        mv(temporary, path; force=true)
    catch
        ispath(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return path
end

config_path = abspath(ARGS[1])
state_path = abspath(ARGS[2])
isfile(config_path) || error("frozen-field config not found: $config_path")
isfile(state_path) || error("frozen-field state not found: $state_path")
raw = TOML.parsefile(config_path)
haskey(raw, "frozen_legacy") || error("config has no [frozen_legacy] contract")
contract = raw["frozen_legacy"]
settings = load_settings(config_path)
target_label = basename(settings.run.output_directory)

reference_labels = String.(contract["reference_labels"])
reference_paths = realpath.(String.(contract["reference_states"]))
reference_hashes = lowercase.(String.(contract["reference_state_sha256"]))
length(reference_labels) == length(reference_paths) == length(reference_hashes) || error(
    "reference label/path/hash counts differ",
)
for (path, expected) in zip(reference_paths, reference_hashes)
    LadderMPSMFT.sha256_file(path) == expected || error("reference state SHA-256 mismatch: $path")
end
ranked_references = compare_variational_branches(reference_paths)

frozen = h5open(state_path, "r") do file
    required = (
        "status", "accepted", "solution_kind", "fundamental_period",
        "solution_canonical_variational_energy",
        "solution_target_density_corrected_variational_energy", "density_error",
        "fixed_point_abs_residual", "fixed_point_rel_residual", "chemical_potential",
        "hamiltonian_identity_error_per_site", "effective_eigenvalue_error_per_site",
        "history/density",
        "history/dmrg_max_discarded_weight", "history/dmrg_maxlinkdim",
        "history/dmrg/0001/sweep_energy", "provenance/model_fingerprint",
        "provenance/numerical_fingerprint", "provenance/implementation_sha256",
        "provenance/ep_source_sha256", "provenance/legacy_source_sha256",
        "energy/effective_eigenvalue", "energy/effective_expectation",
        "energy/bare_ladder_energy", "energy/canonical_variational_energy",
        "energy/target_density_correction",
        "energy/target_density_corrected_variational_energy",
        "energy/double_counting_correction",
    )
    for name in required
        haskey(file, name) || error("frozen state has no $name: $state_path")
    end
    status = String(read(file, "status"))
    status == "frozen_field_evaluation" || error(
        "state status is $status rather than frozen_field_evaluation",
    )
    !Bool(read(file, "accepted")) || error("frozen diagnostic must not be accepted")
    String(read(file, "solution_kind")) == "diagnostic" || error(
        "frozen state is not classified as a diagnostic",
    )
    Int(read(file, "fundamental_period")) == 0 || error(
        "frozen diagnostic must not claim a period",
    )
    sweep_energies = Float64.(read(file, "history/dmrg/0001/sweep_energy"))
    length(sweep_energies) >= 2 || error("frozen state has fewer than two DMRG sweep energies")
    dmrg_energy_change = abs(sweep_energies[end] - sweep_energies[end - 1])
    dmrg_energy_converged = dmrg_energy_change <= settings.dmrg.energy_tol
    dmrg_energy_converged || error(
        "saved DMRG did not meet its energy tolerance: change=$dmrg_energy_change, " *
        "tolerance=$(settings.dmrg.energy_tol)",
    )
    return (;
        status,
        canonical_energy=Float64(read(file, "solution_canonical_variational_energy")),
        target_energy=Float64(read(
            file,
            "solution_target_density_corrected_variational_energy",
        )),
        density=Float64(last(read(file, "history/density"))),
        density_error=Float64(read(file, "density_error")),
        chemical_potential=Float64(read(file, "chemical_potential")),
        field_abs_residual=Float64(read(file, "fixed_point_abs_residual")),
        field_rel_residual=Float64(read(file, "fixed_point_rel_residual")),
        effective_eigenvalue=Float64(read(file, "energy/effective_eigenvalue")),
        effective_expectation=Float64(read(file, "energy/effective_expectation")),
        bare_ladder_energy=Float64(read(file, "energy/bare_ladder_energy")),
        energy_canonical=Float64(read(file, "energy/canonical_variational_energy")),
        target_density_correction=Float64(read(file, "energy/target_density_correction")),
        energy_target_corrected=Float64(read(
            file,
            "energy/target_density_corrected_variational_energy",
        )),
        double_counting_correction=Float64(read(file, "energy/double_counting_correction")),
        hamiltonian_identity_error_per_site=Float64(read(
            file,
            "hamiltonian_identity_error_per_site",
        )),
        effective_eigenvalue_error_per_site=Float64(read(
            file,
            "effective_eigenvalue_error_per_site",
        )),
        max_discarded_weight=Float64(last(read(file, "history/dmrg_max_discarded_weight"))),
        maxlinkdim=Int(last(read(file, "history/dmrg_maxlinkdim"))),
        sweeps=length(sweep_energies),
        dmrg_energy_change,
        dmrg_energy_converged,
        model_fingerprint=String(read(file, "provenance/model_fingerprint")),
        numerical_fingerprint=String(read(file, "provenance/numerical_fingerprint")),
        implementation_sha256=String(read(file, "provenance/implementation_sha256")),
        ep_source_sha256=String(read(file, "provenance/ep_source_sha256")),
        legacy_source_sha256=String(read(file, "provenance/legacy_source_sha256")),
    )
end

isapprox(frozen.canonical_energy, frozen.energy_canonical; atol=1e-12, rtol=0.0) || error(
    "top-level and record-level canonical energies differ",
)
isapprox(frozen.target_energy, frozen.energy_target_corrected; atol=1e-12, rtol=0.0) || error(
    "top-level and record-level target-density-corrected energies differ",
)

diagnostics_path = joinpath(dirname(state_path), "diagnostics.h5")
isfile(diagnostics_path) || error("frozen-field diagnostics not found: $diagnostics_path")
diagnostics = h5open(diagnostics_path, "r") do file
    required = (
        "charge_peak/qx_over_pi", "charge_peak/ky_over_pi",
        "spin_peak/qx_over_pi", "spin_peak/ky_over_pi", "spin_q_mismatch",
        "K_rho/K_rho_site_normalized", "K_rho/K_rho_rung_normalized",
        "central_charge/central_charge",
    )
    for name in required
        haskey(file, name) || error("frozen diagnostics has no $name: $diagnostics_path")
    end
    return (;
        charge_peak_qx_over_pi=Float64(read(file, "charge_peak/qx_over_pi")),
        charge_peak_ky_over_pi=Float64(read(file, "charge_peak/ky_over_pi")),
        spin_peak_qx_over_pi=Float64(read(file, "spin_peak/qx_over_pi")),
        spin_peak_ky_over_pi=Float64(read(file, "spin_peak/ky_over_pi")),
        spin_q_mismatch=Float64(read(file, "spin_q_mismatch")),
        K_rho_site_normalized=Float64(read(file, "K_rho/K_rho_site_normalized")),
        K_rho_rung_normalized=Float64(read(file, "K_rho/K_rho_rung_normalized")),
        central_charge=Float64(read(file, "central_charge/central_charge")),
    )
end

for (field, expected_key) in (
    (:model_fingerprint, "expected_model_fingerprint"),
    (:numerical_fingerprint, "expected_numerical_fingerprint"),
    (:implementation_sha256, "expected_implementation_sha256"),
    (:ep_source_sha256, "expected_ep_source_sha256"),
)
    getfield(frozen, field) == String(contract[expected_key]) || error(
        "frozen state $field differs from its reference campaign",
    )
end
frozen.legacy_source_sha256 == lowercase(String(contract["source_sha256"])) || error(
    "frozen state legacy-source hash differs from its config",
)

frozen_table_path = joinpath(dirname(state_path), "frozen_dmrg_observables.tsv")
atomic_write(frozen_table_path) do io
    println(io, "quantity\tvalue")
    for (key, value) in (
        "selection_eligible" => false,
        "dmrg_timed_out" => false,
        "dmrg_energy_converged" => frozen.dmrg_energy_converged,
        "dmrg_sweeps_recorded" => frozen.sweeps,
        "dmrg_final_energy_change" => frozen.dmrg_energy_change,
        "dmrg_energy_tolerance" => settings.dmrg.energy_tol,
        "dmrg_max_discarded_weight" => frozen.max_discarded_weight,
        "dmrg_maxlinkdim" => frozen.maxlinkdim,
        "saved_legacy_mu_used_without_search" => frozen.chemical_potential,
        "density" => frozen.density,
        "density_error" => frozen.density_error,
        "raw_map_field_abs_residual" => frozen.field_abs_residual,
        "raw_map_field_rel_residual" => frozen.field_rel_residual,
        "effective_eigenvalue" => frozen.effective_eigenvalue,
        "effective_expectation" => frozen.effective_expectation,
        "bare_ladder_energy" => frozen.bare_ladder_energy,
        "canonical_variational_energy" => frozen.canonical_energy,
        "target_density_correction" => frozen.target_density_correction,
        "target_density_corrected_variational_energy" => frozen.target_energy,
        "double_counting_correction" => frozen.double_counting_correction,
        "hamiltonian_identity_error_per_site" => frozen.hamiltonian_identity_error_per_site,
        "effective_eigenvalue_error_per_site" => frozen.effective_eigenvalue_error_per_site,
        "charge_peak_qx_over_pi" => diagnostics.charge_peak_qx_over_pi,
        "charge_peak_ky_over_pi" => diagnostics.charge_peak_ky_over_pi,
        "spin_peak_qx_over_pi" => diagnostics.spin_peak_qx_over_pi,
        "spin_peak_ky_over_pi" => diagnostics.spin_peak_ky_over_pi,
        "spin_q_mismatch" => diagnostics.spin_q_mismatch,
        "K_rho_site_normalized" => diagnostics.K_rho_site_normalized,
        "K_rho_rung_normalized" => diagnostics.K_rho_rung_normalized,
        "central_charge" => diagnostics.central_charge,
    )
        println(io, key, '\t', value)
    end
end

label_by_path = Dict(normpath(path) => label for (path, label) in zip(reference_paths, reference_labels))
minimum_reference_energy = minimum(row.energy for row in ranked_references)
comparison_path = joinpath(dirname(state_path), "energy_comparison.tsv")
atomic_write(comparison_path) do io
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
        "single_dmrg_target_density_corrected", string(frozen.target_energy),
        string(frozen.target_energy - minimum_reference_energy), state_path,
    ), '\t'))
end

summary_path = joinpath(dirname(state_path), "run_summary.md")
atomic_write(summary_path) do io
    println(io, "# Frozen legacy-field one-shot DMRG")
    println(io)
    println(io, "The DMRG completed and its immutable HDF5 state was written before a reporting-only diagnostics-field error terminated the original job. This summary was regenerated from that saved state without rerunning DMRG or modifying any HDF5 file.")
    println(io)
    println(io, "- Selection eligibility: **false** (diagnostic only; not an accepted SCF state)")
    println(io, "- Target-density-corrected energy: `$(frozen.target_energy)`")
    println(io, "- Delta to the lowest accepted reference: `$(frozen.target_energy - minimum_reference_energy)`")
    println(io, "- Delta per physical site: `$((frozen.target_energy - minimum_reference_energy) / (2 * settings.model.L))`")
    println(io, "- Density: `$(frozen.density)` (error `$(frozen.density_error)`)")
    println(io, "- One-step raw-map residual: abs `$(frozen.field_abs_residual)`, rel `$(frozen.field_rel_residual)`")
    println(io, "- Recorded DMRG sweeps: `$(frozen.sweeps)`")
    println(io, "- Final DMRG energy change: `$(frozen.dmrg_energy_change)` (tolerance `$(settings.dmrg.energy_tol)`)")
    println(io, "- Maximum discarded weight: `$(frozen.max_discarded_weight)`")
    println(io, "- Realized maximum link dimension: `$(frozen.maxlinkdim)`")
    println(io)
    println(io, "The six accepted references retain the formal within-fingerprint ranking. The frozen legacy row is shown only as the requested conditional energetic diagnostic.")
end

println("frozen_observables_path=$frozen_table_path")
println("energy_comparison_path=$comparison_path")
println("summary_path=$summary_path")
println("target_density_corrected_energy=$(frozen.target_energy)")
println("delta_to_accepted_min=$(frozen.target_energy - minimum_reference_energy)")
println("selection_eligible=false")
