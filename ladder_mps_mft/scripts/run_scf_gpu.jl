#!/usr/bin/env julia

using LadderMPSMFT
using CUDA

length(ARGS) == 1 || error("usage: julia --project=gpu scripts/run_scf_gpu.jl CONFIG.toml")
settings = load_settings(ARGS[1])
settings.runtime.backend == :gpu || error("GPU entry point requires runtime.backend=gpu")
preflight = gpu_linalg_preflight!()
println("gpu_linalg_preflight_dimension=$(preflight.dimension)")
println("gpu_linalg_preflight_scalar_type=$(preflight.scalar_type)")
println("gpu_tensor_scalar_type=$(settings.runtime.tensor_scalar_type)")
result = run_scf(settings)
println("state_path=$(result.state_path)")
println("summary_path=$(result.summary_path)")
println("status=$(result.diagnostic.status)")
println("accepted=$(result.diagnostic.accepted)")
if settings.run.require_accepted_solution && !result.diagnostic.accepted
    exit(2)
end
