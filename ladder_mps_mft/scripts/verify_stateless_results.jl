#!/usr/bin/env julia

using HDF5
using LadderMPSMFT

(length(ARGS) == 1 || (length(ARGS) == 2 && ARGS[2] == "--full")) || error(
    "usage: julia --project=. scripts/verify_stateless_results.jl STATELESS_RESULTS_DIRECTORY [--full]",
)

root = abspath(ARGS[1])
verify_full = length(ARGS) == 2
manifest_path = joinpath(root, "stateless_manifest.tsv")
isfile(manifest_path) || error("stateless manifest not found: $manifest_path")

function verify_no_mps(group, prefix="")
    for name in String.(collect(keys(group)))
        path = isempty(prefix) ? name : "$prefix/$name"
        occursin(r"^psi(?:_N_[0-9]+)?$", name) && error("MPS object remains in stateless HDF5: $path")
        object = group[name]
        try
            object isa HDF5.Group && verify_no_mps(object, path)
        finally
            close(object)
        end
    end
end

lines = readlines(manifest_path)
isempty(lines) && error("empty stateless manifest: $manifest_path")
expected_header = [
    "relative_path", "kind", "full_path", "full_sha256", "full_bytes",
    "compact_path", "compact_sha256", "compact_bytes", "omitted_paths",
]
split(first(lines), '\t'; keepempty=true) == expected_header || error(
    "unexpected stateless manifest header: $manifest_path",
)

full_bytes_total = Ref(0)
compact_bytes_total = Ref(0)
for line in lines[2:end]
    columns = split(line, '\t'; keepempty=true)
    length(columns) == length(expected_header) || error("malformed stateless manifest row: $line")
    relative_path, kind, full_path, full_sha256, full_bytes_text,
        _, compact_sha256, compact_bytes_text, _ = columns
    compact_path = joinpath(root, relative_path)
    isfile(compact_path) || error("stateless artifact missing: $compact_path")
    LadderMPSMFT.sha256_file(compact_path) == compact_sha256 || error(
        "stateless artifact SHA-256 mismatch: $compact_path",
    )
    stat(compact_path).size == parse(Int, compact_bytes_text) || error(
        "stateless artifact size mismatch: $compact_path",
    )
    if kind == "stateless_hdf5"
        h5open(compact_path, "r") do file
            verify_no_mps(file)
            haskey(file, "analysis_storage/is_stateless_copy") || error(
                "HDF5 copy lacks analysis-storage metadata: $compact_path",
            )
            Bool(read(file, "analysis_storage/is_stateless_copy")) || error(
                "HDF5 artifact is not marked stateless: $compact_path",
            )
            String(read(file, "analysis_storage/full_artifact_sha256")) == full_sha256 || error(
                "full-artifact hash metadata mismatch: $compact_path",
            )
        end
    end
    if verify_full
        isfile(full_path) || error("full artifact missing: $full_path")
        LadderMPSMFT.sha256_file(full_path) == full_sha256 || error(
            "full artifact SHA-256 mismatch: $full_path",
        )
        stat(full_path).size == parse(Int, full_bytes_text) || error(
            "full artifact size mismatch: $full_path",
        )
    end
    full_bytes_total[] += parse(Int, full_bytes_text)
    compact_bytes_total[] += parse(Int, compact_bytes_text)
end

println("stateless_manifest=$manifest_path")
println("artifact_count=$(length(lines) - 1)")
println("full_bytes=$(full_bytes_total[])")
println("compact_bytes=$(compact_bytes_total[])")
println("full_artifacts_verified=$verify_full")
