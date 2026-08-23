#!/usr/bin/env julia

using HDF5
using ITensorMPS
using ITensors
using LadderMPSMFT
using CUDA
using Random

length(ARGS) == 1 || error("usage: julia --project=gpu scripts/gpu_smoke.jl OUTPUT.h5")
output_path = abspath(ARGS[1])
ispath(output_path) && error("refusing to overwrite GPU smoke artifact: $output_path")
runtime = RuntimeSettings(
    backend=:gpu,
    blas_threads=1,
    strided_threads=1,
    threaded_blocksparse=false,
    conserve_sz=false,
    conserve_nfparity=false,
)
ensure_backend!(runtime)
configure_threading!(runtime)
preflight = gpu_linalg_preflight!()
model = ModelSettings(
    L=2,
    U=2.0,
    t0=1.0,
    tp=0.01,
    density=1.0,
    r_range=1,
    geometry=:square,
    ep=0.2,
    ep_signed=-0.2,
    ep_source="gpu_smoke",
    ep_mode=:exact,
    ep_t0_lower=1.0,
    ep_t0_upper=1.0,
    ep_lower_signed=-0.2,
    ep_upper_signed=-0.2,
    ep_lower_chi=4,
    ep_upper_chi=4,
)
sites = LadderMPSMFT.make_sites(model, runtime)
any(ITensors.hasqns, sites) && error("GPU smoke unexpectedly constructed QN site indices")
fields = initial_fields(model; seed=:pairing, amplitude=1e-4, rng=MersenneTwister(11))
hamiltonian = build_mf_mpo(sites, model, fields, 0.0; backend=:gpu)
settings = DMRGSettings(
    nsweeps=1,
    maxdim=4,
    cutoff=1e-6,
    energy_tol=0.0,
    eigsolve_krylovdim=3,
    output_level=0,
    max_time_seconds=300.0,
)
result = run_dmrg_ground(
    sites,
    hamiltonian,
    model.density,
    settings;
    rng=MersenneTwister(11),
    deadline=time() + 300,
    backend=:gpu,
)
result.timed_out && error("tiny GPU DMRG timed out")
density = LadderMPSMFT.average_density(result.psi)
measured, correlations = calculate_mean_fields(result.psi, model)
all(isfinite, measured.alpha) || error("nonfinite GPU mean field")
all(isfinite, correlations.density_up) || error("nonfinite GPU observable")
psi_cpu = move_to_cpu(result.psi)
mkpath(dirname(output_path))
h5open(output_path, "w") do file
    file["schema_version"] = 1
    file["artifact_kind"] = "ladder_mps_mft_gpu_smoke"
    file["completed"] = true
    file["energy"] = result.energy
    file["density"] = density
    file["psi"] = psi_cpu
    device = create_group(file, "device")
    for (key, value) in backend_metadata(runtime)
        device[key] = value
    end
    linalg = create_group(file, "linalg_preflight")
    linalg["dimension"] = preflight.dimension
    linalg["minimum_eigenvalue"] = preflight.minimum_eigenvalue
    linalg["maximum_eigenvalue"] = preflight.maximum_eigenvalue
end
loaded = h5open(output_path, "r") do file
    Bool(read(file, "completed")) || error("GPU smoke completion marker missing")
    read(file, "psi", MPS)
end
length(loaded) == 2 * model.L || error("GPU smoke checkpoint round trip changed MPS length")
println("gpu_smoke_path=$output_path")
println("energy=$(result.energy)")
println("density=$density")
println("linalg_preflight_dimension=$(preflight.dimension)")
