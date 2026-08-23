module LadderMPSMFTCUDAExt

using CUDA
using LadderMPSMFT

function ensure_cuda!()
    CUDA.functional(true) || error("CUDA.jl is installed but no functional CUDA device is available")
    return true
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
    )
end

end
