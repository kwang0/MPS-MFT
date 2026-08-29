#!/usr/bin/env julia

using HDF5
using ITensorMPS
using LadderMPSMFT
using Random

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/phase0_prepare_seed.jl CONFIG.toml SEED_STATE.h5",
)
config_path, output_path = abspath.(ARGS)
ispath(output_path) && error("refusing to overwrite immutable Phase 0 seed: $output_path")
settings = load_settings(config_path)

# Phase 0 compares DMRG implementations, not chemical-potential search
# strategies. Prepare one modest chi=64 fixed-mu state so the timed chi=200
# solves begin from the same realistic warm start. This setup solve is not part
# of any candidate timing.
runtime = RuntimeSettings(blas_threads=1, strided_threads=1, threaded_blocksparse=true)
threading = configure_threading!(runtime)
rng = MersenneTwister(settings.run.random_seed)
sites = LadderMPSMFT.make_sites(settings.model)
fields = initial_fields(
    settings.model;
    seed=settings.run.initial_seed,
    amplitude=settings.run.initial_amplitude,
    rng,
    protocol=settings.run.initial_seed_protocol,
    mode_number=settings.run.initial_mode_number,
    mode_phase_pi=settings.run.initial_mode_phase_pi,
    pairing_form_factor=settings.run.initial_pairing_form_factor,
    leg_parity=settings.run.initial_leg_parity,
)
psi_product = productMPS(
    sites,
    density_product_state(
        2 * settings.model.L,
        settings.model.density;
        rng=MersenneTwister(settings.run.random_seed),
    ),
)
hamiltonian = build_mf_mpo(sites, settings.model, fields, settings.model.mu_initial)
seed_dmrg = DMRGSettings(
    nsweeps=2,
    maxdim=min(64, settings.dmrg.maxdim),
    cutoff=settings.dmrg.cutoff,
    energy_tol=0.0,
    eigsolve_krylovdim=settings.dmrg.eigsolve_krylovdim,
    max_time_seconds=settings.dmrg.max_time_seconds,
    output_level=settings.dmrg.output_level,
)
result = run_dmrg_ground(
    sites,
    hamiltonian,
    settings.model.density,
    seed_dmrg;
    psi_init=psi_product,
    rng=MersenneTwister(settings.run.random_seed),
    deadline=time() + settings.dmrg.max_time_seconds,
)
result.timed_out && error("fixed-mu Phase 0 warm-start preparation reached its time limit")
psi = result.psi
seed_density = LadderMPSMFT.average_density(psi)

mkpath(dirname(output_path))
temporary = tempname(dirname(output_path))
h5open(temporary, "w") do file
    file["schema_version"] = 3
    file["artifact_kind"] = "phase0_fixed_mu_dmrg_seed"
    file["benchmark_kind"] = "fixed_mu_dmrg"
    file["scientific_state"] = false
    file["psi"] = psi
    fields_group = create_group(file, "fields")
    fields_group["alpha"] = fields.alpha
    fields_group["beta"] = fields.beta
    fields_group["mu_cdw"] = fields.mu_cdw
    file["energy"] = result.energy
    file["chemical_potential"] = settings.model.mu_initial
    file["density"] = seed_density
    file["target_density"] = settings.model.density
    file["maximum_bond_dimension"] = maxlinkdim(psi)
    file["preparation_nsweeps"] = seed_dmrg.nsweeps
    file["preparation_maxdim"] = seed_dmrg.maxdim
    file["model_fingerprint"] = LadderMPSMFT.model_fingerprint(settings.model)
    file["config_sha256"] = LadderMPSMFT.sha256_file(config_path)
    file["ep_source_sha256"] = LadderMPSMFT.sha256_file(settings.model.ep_source)
    file["git_commit"] = LadderMPSMFT._read_git("rev-parse", "HEAD")
    file["implementation_sha256"] = LadderMPSMFT.implementation_fingerprint()
    file["julia_threads"] = threading.julia
    file["threaded_blocksparse"] = threading.blocksparse
end
mv(temporary, output_path)

println("seed_path=$output_path")
println("seed_sha256=$(LadderMPSMFT.sha256_file(output_path))")
println("benchmark_kind=fixed_mu_dmrg")
println("benchmark_mu=$(settings.model.mu_initial)")
println("seed_density=$seed_density")
println("seed_energy=$(result.energy)")
println("seed_maximum_bond_dimension=$(maxlinkdim(psi))")
