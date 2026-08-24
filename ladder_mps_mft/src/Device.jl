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

function convert_tensor_scalar_type(value, tensor_scalar_type::Symbol)
    requested_type = tensor_scalar_type == :float64 ? Float64 :
        tensor_scalar_type == :float32 ? Float32 : throw(ArgumentError(
            "unsupported tensor scalar type: $tensor_scalar_type",
        ))
    convert_one = if tensor_scalar_type == :float64
        ITensors.NDTensors.double_precision
    else
        ITensors.NDTensors.single_precision
    end
    # NDTensors' generic precision walker recurses through MPS/MPO container
    # type metadata on this pinned stack. Mapping their ITensors individually
    # preserves the MPS/MPO container and avoids that recursion.
    if value isa Union{MPS,MPO}
        return all(tensor -> eltype(tensor) == requested_type, value) ? value : map(convert_one, value)
    end
    return eltype(value) == requested_type ? value : convert_one(value)
end

function move_to_backend(value, backend::Symbol)
    backend == :cpu && return value
    backend == :gpu || throw(ArgumentError("unknown runtime backend $backend"))
    return _cuda_extension().to_gpu(value, :float64)
end

function move_to_backend(value, runtime::RuntimeSettings)
    runtime.backend == :cpu && return convert_tensor_scalar_type(value, runtime.tensor_scalar_type)
    runtime.backend == :gpu || throw(ArgumentError("unknown runtime backend $(runtime.backend)"))
    return _cuda_extension().to_gpu(value, runtime.tensor_scalar_type)
end

move_to_cpu(value) = ITensors.cpu(value)

function backend_metadata(runtime::RuntimeSettings)
    base = Dict{String,Any}(
        "backend" => String(runtime.backend),
        "tensor_scalar_type" => String(runtime.tensor_scalar_type),
        "conserve_sz" => runtime.conserve_sz,
        "conserve_nfparity" => runtime.conserve_nfparity,
        "qn_conservation" => runtime.conserve_sz || runtime.conserve_nfparity,
    )
    runtime.backend == :cpu && return base
    merge!(base, _cuda_extension().cuda_metadata())
    return base
end
