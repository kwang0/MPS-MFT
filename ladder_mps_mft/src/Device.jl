function _cuda_extension()
    extension = Base.get_extension(@__MODULE__, :LadderMPSMFTCUDAExt)
    extension === nothing && throw(ArgumentError(
        "backend=gpu requires CUDA.jl to be loaded in the current process; " *
        "run with scripts/run_scf_gpu.jl and the gpu environment",
    ))
    return extension
end

function ensure_backend!(runtime::RuntimeSettings)
    runtime.backend == :cpu && return true
    runtime.backend == :gpu || throw(ArgumentError("unknown runtime backend $(runtime.backend)"))
    return _cuda_extension().ensure_cuda!()
end

function gpu_linalg_preflight!(; dimension::Integer=256)
    return _cuda_extension().linalg_preflight!(dimension)
end

function move_to_backend(value, backend::Symbol)
    backend == :cpu && return value
    backend == :gpu || throw(ArgumentError("unknown runtime backend $backend"))
    return _cuda_extension().to_gpu(value)
end

move_to_backend(value, runtime::RuntimeSettings) = move_to_backend(value, runtime.backend)

move_to_cpu(value) = ITensors.cpu(value)

function backend_metadata(runtime::RuntimeSettings)
    base = Dict{String,Any}(
        "backend" => String(runtime.backend),
        "conserve_sz" => runtime.conserve_sz,
        "conserve_nfparity" => runtime.conserve_nfparity,
        "qn_conservation" => runtime.conserve_sz || runtime.conserve_nfparity,
    )
    runtime.backend == :cpu && return base
    merge!(base, _cuda_extension().cuda_metadata())
    return base
end
