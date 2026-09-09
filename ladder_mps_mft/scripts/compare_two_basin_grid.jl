#!/usr/bin/env julia
using CSV, HDF5, LadderMPSMFT, Statistics

const BASIN_FINGERPRINTS = (:model_fingerprint, :numerical_fingerprint, :implementation_sha256, :ep_source_sha256)

function latest_basin_state(directory)
    isdir(directory) || return nothing
    paths = [joinpath(root, name) for (root, _, names) in walkdir(directory) for name in names if name == "state.h5"]
    isempty(paths) && return nothing
    # The solver uses UTC timestamp-prefixed leaf directories. Sync mtimes are
    # not used to infer which segment is the newest.
    sort!(paths; by=path -> basename(dirname(path)))
    return last(paths)
end

function basin_energy_window(path, period)
    h5open(path, "r") do f
        values = Float64.(read(f, "history/target_density_corrected_variational_energy"))
        sites = 2 * Int(read(f, "model/L"))
        length(values) >= 5period || return Inf
        means = [mean(values[end-(i+1)*period+1:end-i*period]) / sites for i in 0:4]
        return maximum(means) - minimum(means)
    end
end

function compare_two_basin_grid(run_directory, output_path)
    manifest = CSV.File(joinpath(run_directory, "manifest.tsv"); delim='\t', types=String)
    groups = Dict{Tuple{Float64,Float64},Vector{Any}}()
    for row in manifest
        key = (parse(Float64, row.t0), parse(Float64, row.V))
        push!(get!(groups, key, Any[]), row)
    end
    results = NamedTuple[]
    for ((t0, V), entries) in sort!(collect(groups); by=first)
        Set(String(row.family) for row in entries) == Set(["stripe", "pairing"]) && length(entries) == 2 || error("point must have exactly two seed families")
        paths = Dict(String(row.family) => latest_basin_state(joinpath(run_directory, "results", row.label)) for row in entries)
        status = "awaiting_results"; note = "Both terminal endpoints are required."
        stripe_energy = pairing_energy = delta = resolution_floor = NaN
        preferred = ""
        if all(path -> path !== nothing, values(paths))
            try
                for entry in entries
                    h5open(paths[entry.family], "r") do f
                        for key in BASIN_FINGERPRINTS
                            expected = String(getproperty(entry, key))
                            isempty(expected) && error("missing manifest fingerprint: $key")
                            String(read(f, "provenance/" * String(key))) == expected || error("state/manifest fingerprint mismatch: $key")
                        end
                    end
                end
                ranked = compare_variational_branches([paths["stripe"], paths["pairing"]])
                all(row -> isfinite(row.energy), ranked) || error("nonfinite canonical solution energy")
                sites = h5open(f -> 2Int(read(f, "model/L")), paths["stripe"], "r")
                stripe_row = only(filter(row -> row.path == abspath(paths["stripe"]), ranked))
                pairing_row = only(filter(row -> row.path == abspath(paths["pairing"]), ranked))
                stripe_energy, pairing_energy = stripe_row.energy / sites, pairing_row.energy / sites
                delta = stripe_energy - pairing_energy
                # This is a finite-run resolution screen, not a rigorous error
                # bound or a thermodynamic phase assignment.
                resolution_floor = 10max(1e-8, basin_energy_window(paths["stripe"], stripe_row.period),
                                              basin_energy_window(paths["pairing"], pairing_row.period))
                if abs(delta) > resolution_floor
                    status = "lower_recorded_energy"
                    preferred = delta < 0 ? "stripe" : "pairing"
                    note = "Matched accepted solutions; preferred label identifies seed ancestry, not the final order."
                else
                    status = "energy_unresolved"
                    note = "Energy gap is within ten times the larger five-solution energy span or 1e-8 t/site tolerance."
                end
            catch err
                status = "not_rankable"
                note = sprint(showerror, err)
            end
        end
        push!(results, (t0=t0, V=V, status=status, lower_energy_seed_family=preferred,
            stripe_corrected_energy_per_site=stripe_energy, pairing_corrected_energy_per_site=pairing_energy,
            delta_stripe_minus_pairing_per_site=delta, resolution_screen_per_site=resolution_floor,
            stripe_state=something(paths["stripe"], ""), pairing_state=something(paths["pairing"], ""), note=note))
    end
    mkpath(dirname(abspath(output_path)))
    CSV.write(output_path, results)
    println("Wrote $(length(results)) point comparisons to $output_path")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: julia --project=. scripts/compare_two_basin_grid.jl RUN_DIRECTORY OUTPUT.csv")
    compare_two_basin_grid(ARGS...)
end
