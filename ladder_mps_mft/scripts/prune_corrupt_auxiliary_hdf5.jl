#!/usr/bin/env julia

using HDF5

function usage()
    println(stderr, "usage: julia --project=. scripts/prune_corrupt_auxiliary_hdf5.jl --apply SOURCE_ROOT [PEER_ROOT ...]")
end

function is_hdf5_candidate(name::AbstractString)
    lower = lowercase(name)
    return endswith(lower, ".h5") || endswith(lower, ".hdf5") ||
        occursin(r"\.h(?:5|df5)\.corrupt-", lower)
end

function is_explicit_corrupt_backup(name::AbstractString)
    return occursin(r"\.h(?:5|df5)\.corrupt-", lowercase(name))
end

function hdf5_error(path::AbstractString)
    try
        h5open(path, "r") do file
            # Force the root group metadata to be read, while avoiding large datasets.
            keys(file)
        end
        return nothing
    catch error
        return replace(sprint(showerror, error), '\n' => ' ')
    end
end

function main(arguments)
    apply = false
    roots = String[]
    for argument in arguments
        if argument == "--apply"
            apply = true
        elseif startswith(argument, "--")
            error("unknown option: $argument")
        else
            push!(roots, abspath(argument))
        end
    end
    apply || error("refusing to delete without --apply")
    isempty(roots) && (usage(); error("SOURCE_ROOT is required"))

    source_root = first(roots)
    peer_roots = roots[2:end]
    isdir(source_root) || error("source result tree not found: $source_root")
    any(root -> root == source_root, peer_roots) && error("source and peer roots must differ")

    corrupt = NamedTuple[]
    for (directory, _, names) in walkdir(source_root)
        for name in sort!(names)
            is_hdf5_candidate(name) || continue
            path = joinpath(directory, name)
            isfile(path) || continue
            reason = if is_explicit_corrupt_backup(name)
                "explicit_corrupt_backup"
            else
                hdf5_error(path)
            end
            isnothing(reason) && continue
            relative_path = relpath(path, source_root)
            critical = lowercase(name) == "state.h5"
            push!(corrupt, (; path, relative_path, critical, reason))
        end
    end

    for record in corrupt
        label = record.critical ? "critical_corrupt_hdf5" : "auxiliary_corrupt_hdf5"
        println(label, "=", record.relative_path, "\treason=", record.reason)
    end
    critical = filter(record -> record.critical, corrupt)
    isempty(critical) || error(
        "refusing cleanup because $(length(critical)) final state.h5 artifact(s) are unreadable",
    )

    auxiliary = filter(record -> !record.critical, corrupt)
    for record in auxiliary
        rm(record.path; force=true)
        println("removed=", record.path)
        for peer_root in peer_roots
            peer_path = joinpath(peer_root, record.relative_path)
            if ispath(peer_path)
                rm(peer_path; force=true)
                println("removed=", peer_path)
            end
        end
    end
    println("corrupt_auxiliary_removed=", length(auxiliary))
end

main(ARGS)
