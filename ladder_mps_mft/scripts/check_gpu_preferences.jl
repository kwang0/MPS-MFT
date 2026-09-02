#!/usr/bin/env julia

# This check deliberately avoids importing CUDA, HDF5, or MPI. It is safe to
# run on a Perlmutter login node and reports the effective preferences after
# Julia has merged the active GPU project with the rest of LOAD_PATH.
const CUDA_RUNTIME_UUID = Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2")
const MPI_PREFERENCES_UUID = Base.UUID("3da0fdf6-3ccc-4f1b-acd9-58baa6c99267")

function bool_preference(preferences::AbstractDict, key::String)
    haskey(preferences, key) || error("missing required preference: $key")
    value = preferences[key]
    value isa Bool && return value
    value isa AbstractString || error("preference $key must be Boolean-compatible, got $(repr(value))")
    parsed = tryparse(Bool, value)
    parsed === nothing && error("preference $key must be true or false, got $(repr(value))")
    return parsed
end

cuda_preferences = Base.get_preferences(CUDA_RUNTIME_UUID)
mpi_preferences = Base.get_preferences(MPI_PREFERENCES_UUID)

cuda_local = bool_preference(cuda_preferences, "local")
cuda_local && error("CUDA_Runtime_jll.local must be false for artifact-toolkit jobs")

raw_cuda_version = get(cuda_preferences, "version", nothing)
raw_cuda_version === nothing && error("CUDA_Runtime_jll.version must be explicitly pinned to 13.0")
cuda_version = tryparse(VersionNumber, string(raw_cuda_version))
cuda_version === nothing && error("invalid CUDA_Runtime_jll.version: $(repr(raw_cuda_version))")
(cuda_version.major, cuda_version.minor) == (13, 0) || error(
    "effective CUDA_Runtime_jll.version must be 13.0, got $(raw_cuda_version)",
)

mpi_binary = get(mpi_preferences, "binary", nothing)
mpi_binary == "MPICH_jll" || error(
    "effective MPIPreferences.binary must be MPICH_jll, got $(repr(mpi_binary))",
)
mpi_preloads = get(mpi_preferences, "preloads", nothing)
mpi_preloads isa AbstractVector || error(
    "effective MPIPreferences.preloads must be an empty array, got $(repr(mpi_preloads))",
)
isempty(mpi_preloads) || error(
    "effective MPIPreferences.preloads must be empty, got $(repr(mpi_preloads))",
)

println("gpu_preference_check=passed")
println("cuda_runtime_preference_local=false")
println("cuda_runtime_preference_version=13.0")
println("mpi_preference_binary=MPICH_jll")
println("mpi_preference_preloads=0")
