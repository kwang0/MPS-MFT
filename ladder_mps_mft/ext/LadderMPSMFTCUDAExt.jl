module LadderMPSMFTCUDAExt

using CUDA
using LadderMPSMFT
using Libdl
using LinearAlgebra

const CUDA_RUNTIME_LIBRARY_MARKERS = (
    "cudart",
    "nvperf",
    "nvvm",
    "nvrtc",
    "nvJitLink",
    "cublas",
    "cupti",
    "cusparse",
    "cufft",
    "curand",
    "cusolver",
)

function nonartifact_cuda_runtime_libraries()
    CUDA.local_toolkit && error(
        "Phase 1 requires CUDA.jl's pinned artifact toolkit, but CUDA.jl is configured " *
        "to use a local CUDA toolkit",
    )
    offenders = String[]
    for library in Libdl.dllist()
        occursin("artifacts", library) && continue
        any(marker -> occursin(marker, library), CUDA_RUNTIME_LIBRARY_MARKERS) || continue
        push!(offenders, library)
    end
    return sort!(unique!(offenders))
end

function assert_artifact_runtime_isolation!()
    offenders = nonartifact_cuda_runtime_libraries()
    isempty(offenders) || error(
        "CUDA runtime-library isolation failed: CUDA.jl is using its artifact toolkit, " *
        "but system CUDA libraries are already loaded: $(join(offenders, ", ")). " *
        "Unload the cudatoolkit/NVIDIA-HPC-SDK modules and remove their library paths.",
    )
    return true
end

function ensure_cuda!()
    CUDA.functional(true) || error("CUDA.jl is installed but no functional CUDA device is available")
    assert_artifact_runtime_isolation!()
    return true
end

function linalg_preflight!(dimension::Integer)
    dimension >= 2 || throw(ArgumentError("GPU linear-algebra preflight dimension must be at least 2"))
    ensure_cuda!()
    matrix = CUDA.rand(Float32, dimension, dimension)
    gram = Symmetric(matrix * transpose(matrix))
    values = eigen(gram).values
    CUDA.synchronize()
    values_cpu = Array(values)
    all(isfinite, values_cpu) || error("GPU linear-algebra preflight produced nonfinite eigenvalues")
    assert_artifact_runtime_isolation!()
    return (
        dimension=Int(dimension),
        minimum_eigenvalue=Float64(minimum(values_cpu)),
        maximum_eigenvalue=Float64(maximum(values_cpu)),
    )
end

to_gpu(value) = CUDA.cu(value)

function cuda_metadata()
    ensure_cuda!()
    device = CUDA.device()
    return Dict{String,Any}(
        "cuda_jl_version" => string(Base.pkgversion(CUDA)),
        "cuda_device" => string(CUDA.name(device)),
        "cuda_capability" => string(CUDA.capability(device)),
        "cuda_runtime" => string(CUDA.runtime_version()),
        "cuda_driver" => string(CUDA.driver_version()),
        "cuda_toolkit_source" => "artifact",
        "cuda_runtime_library_isolation" => "passed",
    )
end

end
