#=
Evaluate the saved paired trellis MPS twice in the rectangular A/B functional.
Only stored bare-ladder expectation values and correlation matrices are needed.
No MPS is optimized, no field is mixed, and no acceptance flag is changed.
=#
using CSV, HDF5, LinearAlgebra, Printf, SHA, TOML

# Load the actual array kernels without loading or running the DMRG stack.
module TrellisEnergyAudit
using HDF5, LinearAlgebra, Printf
include("../src/Types.jl")
include("../src/Geometry.jl")
include("../src/Variational.jl")
include("../src/Trellis.jl")
end
const CoreAudit = TrellisEnergyAudit
const PROJECT = normpath(joinpath(@__DIR__, ".."))
const OUT = joinpath(PROJECT, "docs", "reports", "trellis_progress_20260918")
sha(path) = open(path, "r") do io; bytes2hex(sha256(io)); end

function model_from_file(file; cell=nothing)
    values = Dict{Symbol,Any}()
    for key in fieldnames(CoreAudit.ModelSettings)
        value = read(file["model"], string(key))
        values[key] = fieldtype(CoreAudit.ModelSettings, key) === Symbol ? Symbol(value) : value
    end
    cell !== nothing && (values[:trellis_cell] = cell)
    return CoreAudit.ModelSettings(; values...)
end

function evaluate(correlations, bare, mus, model)
    fields = CoreAudit.trellis_mean_fields(correlations, model)
    energies = map(eachindex(correlations)) do i
        # The Hamiltonian expectation here is a contraction in a FIXED saved
        # state, not a newly solved effective-Hamiltonian ground-state energy.
        c = correlations[i]
        number = sum(c.density_up) + sum(c.density_down)
        linear = CoreAudit.field_energy_components(fields[i], c, model)
        expectation = bare[i] - mus[i] * number + sum(values(linear))
        CoreAudit.variational_energy(expectation, mus[i], fields[i], c, model;
            bare_ladder_energy=bare[i])
    end
    sites = 2model.L * length(correlations)
    result = Dict{String,Any}(
        "physical_sites" => sites,
        "canonical_per_site" => sum(e.canonical_variational_energy for e in energies) / sites,
        "target_corrected_per_site" => sum(e.target_density_corrected_variational_energy for e in energies) / sites,
        "density_correction_per_site" => sum(e.target_density_correction for e in energies) / sites,
        "bare_ladder_per_site" => sum(bare) / sites,
        "pair_transverse_per_site" => sum(e.pair_transverse_energy for e in energies) / sites,
        "exchange_transverse_per_site" => sum(e.exchange_transverse_energy for e in energies) / sites,
        "density_transverse_per_site" => sum(e.density_transverse_energy for e in energies) / sites,
    )
    return result, fields
end

function main()
    BLAS.set_num_threads(1)
    rows = Dict{String,Any}[]
    common_physics = nothing
    for input in CSV.File(joinpath(OUT, "sources.csv"))
        path = joinpath(PROJECT, String(input.source))
        @assert sha(path) == input.compact_sha256
        result = h5open(path, "r") do file
            model = model_from_file(file)
            physics = Dict(string(k) => getfield(model, k) for k in
                (:L, :t, :U, :V, :t0, :tp, :tau0, :tau1, :density, :r_range, :ep, :ep_signed))
            if common_physics === nothing
                common_physics = physics
            else
                @assert common_physics == physics
            end
            @assert model.tau0 == model.tau1 == 0.1
            @assert !read(file, "accepted")
            labels = model.trellis_cell == :one_ladder ? ["A"] : ["A", "B"]
            cs = [CoreAudit._trellis_read_correlations(file["ladders/$label/correlations"]) for label in labels]
            bare = [read(file["ladders/$label/energy"], "bare_ladder_energy") for label in labels]
            mus = [read(file["ladders/$label"], "chemical_potential") for label in labels]
            densities = [(sum(c.density_up) + sum(c.density_down)) / (2model.L) for c in cs]
            @assert all(abs(n - model.density) <= 1e-5 for n in densities)
            original, fields = evaluate(cs, bare, mus, model)
            errors = Float64[]
            for (i, label) in enumerate(labels), key in fieldnames(CoreAudit.FieldState)
                saved = read(file["ladders/$label/fields/measured"], string(key))
                push!(errors, maximum(abs.(getfield(fields[i], key) - saved)))
            end
            @assert maximum(errors) < 1e-12
            canon = read(file["history"], "canonical_variational_energy_per_site")
            target = read(file["history"], "target_density_corrected_variational_energy_per_site")
            @assert isapprox(original["canonical_per_site"], last(canon); atol=1e-12, rtol=0)
            @assert isapprox(original["target_corrected_per_site"], last(target); atol=1e-12, rtol=0)
            out = Dict{String,Any}(
                "job_id" => string(input.job_id), "label" => String(input.label),
                "source" => String(input.source), "source_sha256" => String(input.compact_sha256),
                "cell" => string(model.trellis_cell), "accepted" => false,
                "density_per_ladder" => densities,
                "max_recomputed_field_error" => maximum(errors), "original" => original,
                "target_energy_last10_min" => minimum(target[end-9:end]),
                "target_energy_last10_max" => maximum(target[end-9:end]),
            )
            if model.trellis_cell == :one_ladder
                rectangle = model_from_file(file; cell=:two_ladder)
                embedded, _ = evaluate([cs[1], cs[1]], [bare[1], bare[1]], [mus[1], mus[1]], rectangle)
                out["paired_copies_in_rectangular_cell"] = embedded
                out["embedding_energy_shift_per_site"] = embedded["canonical_per_site"] - original["canonical_per_site"]
            end
            return out
        end
        @assert sha(path) == input.compact_sha256
        push!(rows, result)
    end
    comparisons = Dict{String,Any}[]
    for paired in filter(r -> r["cell"] == "one_ladder", rows), stripe in filter(r -> r["cell"] == "two_ladder", rows)
        p, s = paired["paired_copies_in_rectangular_cell"], stripe["original"]
        push!(comparisons, Dict{String,Any}(
            "paired_source_job" => paired["job_id"], "stripe_source_job" => stripe["job_id"],
            "stripe_minus_paired_canonical_per_site" => s["canonical_per_site"] - p["canonical_per_site"],
            "stripe_minus_paired_target_corrected_per_site" => s["target_corrected_per_site"] - p["target_corrected_per_site"],
        ))
    end
    output = Dict{String,Any}(
        "date" => "2026-09-20", "common_physics" => common_physics,
        "method" => "Product of two identical saved paired ladder MPSs, with original bare-ladder expectations; all interaction fields recomputed using the actual rectangular-cell kernel. No optimization.",
        "scope" => "Comparable trial-state energies of the implemented MF functional. Not accepted-branch ranking or a proof of the global minimum. Target corrections use stored chemical potentials.",
        "kernel_sha256" => sha(joinpath(PROJECT, "src", "Trellis.jl")),
        "variational_sha256" => sha(joinpath(PROJECT, "src", "Variational.jl")),
        "runs" => rows, "same_cell_comparisons" => comparisons,
    )
    open(joinpath(OUT, "same_cell_energy_audit_20260920.toml"), "w") do io
        TOML.print(io, output; sorted=true)
    end
    for r in rows
        @printf("%s original E/site=%.12f target=%.12f density_correction=%.3e\n", r["job_id"],
            r["original"]["canonical_per_site"], r["original"]["target_corrected_per_site"],
            r["original"]["density_correction_per_site"])
        if haskey(r, "paired_copies_in_rectangular_cell")
            @printf("  rectangular paired copies target=%.12f shift=%.6e\n",
                r["paired_copies_in_rectangular_cell"]["target_corrected_per_site"], r["embedding_energy_shift_per_site"])
        end
    end
    for c in comparisons
        @printf("stripe %s minus paired %s: canonical=%.9e target=%.9e t/site\n", c["stripe_source_job"],
            c["paired_source_job"], c["stripe_minus_paired_canonical_per_site"], c["stripe_minus_paired_target_corrected_per_site"])
    end
    println("Verified all source hashes, common couplings, recomputed fields and original endpoint energies.")
end

main()
