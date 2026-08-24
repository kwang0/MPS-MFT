#!/usr/bin/env julia

using Dates
using HDF5
using LadderMPSMFT
using Printf
using Statistics

(length(ARGS) == 1 || length(ARGS) == 2) || error(
    "usage: julia --project=. scripts/audit_phase1_campaign.jl RUN_DIRECTORY [OUTPUT_DIRECTORY]",
)
run_directory = abspath(ARGS[1])
output_directory = length(ARGS) == 2 ? abspath(ARGS[2]) : joinpath(run_directory, "audit")
isdir(run_directory) || error("Phase 1 run directory not found: $run_directory")
ispath(output_directory) && error("refusing to overwrite audit output: $output_directory")

function latest_file(root::AbstractString, name::AbstractString)
    paths = String[]
    for (directory, _, names) in walkdir(root)
        name in names && push!(paths, joinpath(directory, name))
    end
    isempty(paths) && error("no $name found below $root")
    sort!(paths; by=path -> (mtime(path), path))
    return last(paths)
end

function scalar(file, path)
    value = read(file, path)
    return value isa AbstractString ? String(value) : value
end

function mode_counts(values)
    counts = Dict{String,Int}()
    order = String[]
    for raw in values
        value = String(raw)
        haskey(counts, value) || push!(order, value)
        counts[value] = get(counts, value, 0) + 1
    end
    return join(("$value:$(counts[value])" for value in order), ",")
end

function state_row(label::AbstractString, path::AbstractString)
    config_candidates = filter(
        path -> startswith(basename(path), "$label.segment-") && endswith(path, ".toml"),
        readdir(joinpath(run_directory, "configs"); join=true),
    )
    isempty(config_candidates) && error("no configuration found for $label")
    sort!(config_candidates)
    settings = load_settings(last(config_candidates))
    return h5open(path, "r") do file
        history = file["history"]
        iterations = read(history, "iteration")
        modes = read(history, "update_mode")
        mu_field = read(file, "fields/measured/mu_cdw")
        size(mu_field, 1) == 2 || error("unexpected spin-resolved Hartree field shape in $path")
        sites = size(mu_field, 2)
        first_bulk = fld(sites, 4) + 1
        last_bulk = sites - fld(sites, 4)
        bulk = first_bulk:last_bulk
        charge_field = 0.5 .* (mu_field[1, :] .+ mu_field[2, :])
        spin_field = 0.5 .* (mu_field[2, :] .- mu_field[1, :])
        tensor_path = "psi/MPS[1]/storage/data"
        haskey(file, tensor_path) || error("MPS tensor storage missing in $path")
        status = Symbol(scalar(file, "status"))
        unmixed = Bool(scalar(file, "unmixed_cycle_probe"))
        class = if Bool(scalar(file, "accepted"))
            "accepted"
        elseif status == :periodic_candidate && unmixed
            "raw_map_candidate"
        elseif status == :periodic_candidate
            "mixer_dependent_candidate"
        else
            String(status)
        end
        return (
            label=String(label),
            geometry=String(scalar(file, "model/transverse_geometry")),
            seed=String(scalar(file, "provenance/initial_seed")),
            path=String(path),
            status=String(status),
            class,
            accepted=Bool(scalar(file, "accepted")),
            period=Int(scalar(file, "fundamental_period")),
            unmixed,
            iterations=length(iterations),
            mode_counts=mode_counts(modes),
            solution_energy=Float64(scalar(file, "solution_canonical_variational_energy")),
            final_energy=Float64(scalar(file, "energy/canonical_variational_energy")),
            orbit_energy_spread=Float64(scalar(file, "orbit_energy_spread")),
            residual_rel=Float64(scalar(file, "fixed_point_rel_residual")),
            cycle_rel=Float64(scalar(file, "cycle_rel_residual")),
            density_error=Float64(scalar(file, "density_error")),
            variational_energy_change=Float64(scalar(file, "variational_energy_change")),
            identity_error_per_site=Float64(scalar(file, "hamiltonian_identity_error_per_site")),
            effective_error_per_site=Float64(scalar(file, "effective_eigenvalue_error_per_site")),
            wall_hours=sum(Float64.(read(history, "wall_seconds"))) / 3600,
            alpha_max=maximum(abs, read(file, "fields/measured/alpha")),
            beta_max=maximum(abs, read(file, "fields/measured/beta")),
            charge_field_std=std(charge_field[bulk]),
            spin_field_std=std(spin_field[bulk]),
            tensor_scalar_type=string(eltype(read(file, tensor_path))),
            density_gate=Float64(scalar(file, "density_error")) <= settings.convergence.density_tol,
            energy_gate=Float64(scalar(file, "variational_energy_change")) <= settings.convergence.variational_energy_tol,
            identity_gate=Float64(scalar(file, "hamiltonian_identity_error_per_site")) <= settings.convergence.hamiltonian_identity_tol,
            effective_gate=Float64(scalar(file, "effective_eigenvalue_error_per_site")) <= settings.convergence.effective_energy_consistency_tol,
        )
    end
end

manifest_path = joinpath(run_directory, "manifest.tsv")
isfile(manifest_path) || error("manifest not found: $manifest_path")
labels = [split(line, '\t')[1] for line in readlines(manifest_path)[2:end]]
rows = [state_row(label, latest_file(joinpath(run_directory, "results", label), "state.h5")) for label in labels]

mkpath(output_directory)
tsv_path = joinpath(output_directory, "states.tsv")
columns = fieldnames(typeof(first(rows)))
open(tsv_path, "w") do io
    println(io, join(string.(columns), '\t'))
    for row in rows
        println(io, join((string(getfield(row, column)) for column in columns), '\t'))
    end
end

report_path = joinpath(output_directory, "report.md")
open(report_path, "w") do io
    accepted_count = count(row -> row.accepted, rows)
    raw_count = count(row -> row.class == "raw_map_candidate", rows)
    mixed_count = count(row -> row.class == "mixer_dependent_candidate", rows)
    identity_failures = count(row -> !row.identity_gate, rows)
    effective_failures = count(row -> !row.effective_gate, rows)
    tensor_types = sort!(unique(row.tensor_scalar_type for row in rows))
    wall_hours = sum(row.wall_hours for row in rows)
    println(io, "# Phase 1 campaign audit")
    println(io)
    println(io, "Generated: `$(now(UTC))` UTC")
    println(io)
    println(io, "- Run: `$(basename(run_directory))`")
    println(io, "- Accepted states: `$accepted_count / $(length(rows))`")
    println(io, "- Raw-map periodic candidates: `$raw_count`")
    println(io, "- Mixer-dependent periodic candidates: `$mixed_count`")
    println(io, "- Hamiltonian-identity gate failures: `$identity_failures / $(length(rows))`")
    println(io, "- Effective-energy gate failures: `$effective_failures / $(length(rows))`")
    println(io, "- Saved MPS scalar types: `$(join(tensor_types, ", "))`")
    println(io, @sprintf("- Stored MF-iteration wall time: `%.3f GPU-hours` (`%.3f` one-of-four-GPU node-hours before scheduler/compile overhead)", wall_hours, wall_hours / 4))
    println(io)
    println(io, "| Branch | Geometry | Seed | Class | Iter | E solution | r rel | dE/site | H identity/site | alpha max | charge std | spin std |")
    println(io, "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows
        println(io, @sprintf(
            "| `%s` | `%s` | `%s` | `%s` | %d | %.9f | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e |",
            row.label,
            row.geometry,
            row.seed,
            row.class,
            row.iterations,
            row.solution_energy,
            row.residual_rel,
            row.variational_energy_change,
            row.identity_error_per_site,
            row.alpha_max,
            row.charge_field_std,
            row.spin_field_std,
        ))
    end
    println(io)
    println(io, "No branch-energy ranking is authorized unless every compared state is accepted and shares the required fingerprints. `states.tsv` retains the exact gates and source paths.")
end

println("audit_tsv=$tsv_path")
println("audit_report=$report_path")
