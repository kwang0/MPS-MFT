#!/usr/bin/env julia

using HDF5

length(ARGS) == 1 || error("usage: julia --project=. scripts/validate_gpu_smoke.jl GPU_SMOKE.h5")
path = abspath(ARGS[1])
isfile(path) || error("GPU smoke artifact not found: $path")

summary = h5open(path, "r") do file
    Bool(read(file, "completed")) || error("GPU smoke completion marker is false")
    Int(read(file, "linalg_preflight/dimension")) == 256 || error(
        "GPU smoke did not run the required 256-dimensional linear-algebra preflight",
    )
    String(read(file, "linalg_preflight/scalar_type")) == "Float64" || error(
        "GPU smoke did not run the required Float64 linear-algebra preflight",
    )
    String(read(file, "device/cuda_runtime_library_isolation")) == "passed" || error(
        "GPU smoke did not certify CUDA runtime-library isolation",
    )
    configured = Symbol(read(file, "device/tensor_scalar_type"))
    configured == :float64 || error(
        "GPU smoke requested $configured tensors; Phase 1 production requires float64",
    )
    tensor_data_path = "psi/MPS[1]/storage/data"
    haskey(file, tensor_data_path) || error("GPU smoke MPS tensor storage is missing")
    stored_type = eltype(read(file, tensor_data_path))
    stored_type == Float64 || error(
        "GPU smoke saved $stored_type MPS tensors; Phase 1 production requires Float64",
    )
    return (
        energy=Float64(read(file, "energy")),
        density=Float64(read(file, "density")),
        tensor_scalar_type=stored_type,
    )
end

println("gpu_smoke_validated=true")
println("tensor_scalar_type=$(summary.tensor_scalar_type)")
println("energy=$(summary.energy)")
println("density=$(summary.density)")
