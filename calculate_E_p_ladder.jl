using ITensors
using ITensorMPS
using HDF5
using LinearAlgebra
using Printf

# -----------------------------------------------------------------------------
# Basic ladder construction (mirror main_loop_script_ladder.jl)
# -----------------------------------------------------------------------------

function configure_threading()
    BLAS.set_num_threads(1)
    ITensors.Strided.disable_threads()

    if Threads.nthreads() > 1
        ITensors.enable_threaded_blocksparse()
        blocksparse_threads = "enabled"
    else
        blocksparse_threads = "disabled because Julia has only 1 thread"
        @warn "Julia is running with one thread. On Perlmutter, use `julia -t 32` or the submit wrapper for threaded block-sparse DMRG."
    end

    println("Threading setup: Julia threads=$(Threads.nthreads()), BLAS threads=$(BLAS.get_num_threads()), ITensors block-sparse threading=$(blocksparse_threads), Strided threading=disabled")
    flush(stdout)
end

function enforce_single_slurm_task()
    ntasks_text = get(ENV, "SLURM_STEP_NUM_TASKS", get(ENV, "SLURM_NTASKS", "1"))
    ntasks = tryparse(Int, ntasks_text)
    ntasks === nothing && return
    ntasks <= 1 && return

    procid = tryparse(Int, get(ENV, "SLURM_PROCID", "0"))
    procid = procid === nothing ? 0 : procid
    if procid == 0
        error("calculate_E_p_ladder.jl is a single-process script, but Slurm launched $ntasks tasks. Use `srun -n 1 ... julia -t <threads>` so multiple ranks do not race on the same checkpoint.")
    else
        println("Duplicate Slurm task $procid/$ntasks exiting before DMRG; this script should run with srun -n 1.")
        exit(0)
    end
end

mutable struct DemoObserver <: AbstractObserver
   energy_tol::Float64
   last_energy::Float64

   DemoObserver(energy_tol=0.0) = new(energy_tol,1000.0)
end

function ITensorMPS.checkdone!(o::DemoObserver;kwargs...)
  sw = kwargs[:sweep]
  energy = kwargs[:energy]
  if abs(energy-o.last_energy)/abs(energy) < o.energy_tol
    println("Stopping DMRG after sweep $sw")
    return true
  end
  # Otherwise, update last_energy and keep going
  o.last_energy = energy
  return false
end

function make_sites_conserveN(L::Int)
    return siteinds("Electron", L; conserve_sz=true, conserve_nf=true)
end

function rung_leg_to_site(rung::Int, leg::Int)
    return 2 * (rung - 1) + leg + 1
end

function build_hubbard_ladder(; L::Int, t::Float64=1.0, U::Float64=0.0, V::Float64=0.0, mu::Float64=0.0, t0::Float64=0.0, sites=nothing)
    if sites === nothing
        sites = make_sites_conserveN(2 * L)
    elseif length(sites) != 2 * L
        error("Inherited state has $(length(sites)) sites, expected $(2 * L) for L=$L")
    end
    os = OpSum()

    for i in 1:(2 * L)
        add!(os, U, "Nupdn", i)
        add!(os, -mu, "Ntot", i)
    end

    for rung in 1:(L - 1)
        for leg in 0:1
            s = rung_leg_to_site(rung, leg)
            n = rung_leg_to_site(rung + 1, leg)
            add!(os, -t, "Cdagup", s, "Cup", n)
            add!(os, -t, "Cdagup", n, "Cup", s)
            add!(os, -t, "Cdagdn", s, "Cdn", n)
            add!(os, -t, "Cdagdn", n, "Cdn", s)
            if V != 0.0
                add!(os, V, "Ntot", s, "Ntot", n)
            end
        end
    end


    for rung in 1:L
        s0 = rung_leg_to_site(rung, 0)
        s1 = rung_leg_to_site(rung, 1)
        if t0 != 0.0
            add!(os, -t0, "Cdagup", s0, "Cup", s1)
            add!(os, -t0, "Cdagup", s1, "Cup", s0)
            add!(os, -t0, "Cdagdn", s0, "Cdn", s1)
            add!(os, -t0, "Cdagdn", s1, "Cdn", s0)
        end
        if V != 0.0
            add!(os, V, "Ntot", s0, "Ntot", s1)
        end
    end

    return sites, MPO(os, sites)
end

# -----------------------------------------------------------------------------
# Initial states and counting helpers
# -----------------------------------------------------------------------------

function ladder_initial_state(L::Int, N_particles::Int)
    total_sites = 2 * L
    states = fill("Emp", total_sites)
    N_particles <= 0 && return states
    max_particles = 2 * total_sites
    N_particles > max_particles && error("N_particles=$N_particles exceeds max=$max_particles for $(total_sites) sites")

    placed = 0
    for rung in 1:L
        placed >= N_particles && break
        s0 = rung_leg_to_site(rung, 0)
        s1 = rung_leg_to_site(rung, 1)
        if placed + 2 <= N_particles
            states[s0] = "Up"
            states[s1] = "Dn"
            placed += 2
        elseif placed + 1 <= N_particles
            states[s0] = "Up"
            placed += 1
        end
    end

    idx = 1
    while placed < N_particles && idx <= total_sites
        if states[idx] == "Emp"
            states[idx] = "Up"
            placed += 1
        elseif states[idx] == "Up" || states[idx] == "Dn"
            states[idx] = "UpDn"
            placed += 1
        end
        idx += 1
    end

    return states
end

function count_particles(states::Vector{String})
    total = 0
    n_up = 0
    n_dn = 0
    for s in states
        if s == "Up"
            total += 1
            n_up += 1
        elseif s == "Dn"
            total += 1
            n_dn += 1
        elseif s == "UpDn"
            total += 2
            n_up += 1
            n_dn += 1
        end
    end
    return total, n_up, n_dn
end

# -----------------------------------------------------------------------------
# DMRG driver
# -----------------------------------------------------------------------------

const DEFAULT_E_TOL = 1e-5

function make_sweeps(maxsweeps::Int, maxdim::Int; cutoff::Float64)
    sweeps = Sweeps(maxsweeps)
    maxdim!(sweeps,
            min(20, maxdim),
            min(50, maxdim),
            min(100, maxdim),
            min(200, maxdim),
            maxdim)
    cutoff!(sweeps, cutoff)
    noise!(sweeps, 1e-4, 1e-5, 1e-6, 1e-7, 0.0)
    return sweeps
end

function run_dmrg_energy(L::Int, N_particles::Int; t::Float64=1.0, U::Float64=0.0, V::Float64=0.0, t0::Float64=0.0,
                         mu::Float64=0.0, maxdim::Int=1000, maxsweeps::Int=60, cutoff::Float64=1e-10,
                         E_tol::Float64=DEFAULT_E_TOL, psi_init=nothing)
    N_particles < 0 && error("Cannot target negative particle number: $N_particles")

    local sites
    local psi0
    if psi_init !== nothing
        length(psi_init) == 2 * L || error("Inherited MPS has $(length(psi_init)) sites, expected $(2 * L) for L=$L")
        sites = siteinds(psi_init)
        psi0 = psi_init
        println("  Initial state: inherited MPS for N=$N_particles")
    else
        states = ladder_initial_state(L, N_particles)
        total, n_up, n_dn = count_particles(states)
        total == N_particles || error("Initial state mismatch: expected $N_particles, got $total")
        println("  Initial state: $total particles ($n_up up, $n_dn dn)")
        sites = make_sites_conserveN(2 * L)
        psi0 = productMPS(sites, states)
    end

    _, H = build_hubbard_ladder(L=L, t=t, U=U, V=V, mu=mu, t0=t0, sites=sites)
    sweeps = make_sweeps(maxsweeps, maxdim; cutoff=cutoff)
    energy, psi = dmrg(H, psi0, sweeps; observer = DemoObserver(E_tol), outputlevel=1)
    return energy, psi
end

# -----------------------------------------------------------------------------
# Pair-binding checkpoints
# -----------------------------------------------------------------------------

energy_key(N_particles::Int) = "E_N_$(N_particles)"
state_key(N_particles::Int) = "psi_N_$(N_particles)"

function default_pair_binding_file(L::Int, U::Float64, V::Float64, t0::Float64, density::Float64, maxdim::Int)
    return "E_p_ladder_L_$(L)_U_$(U)_V_$(V)_t0_$(t0)_density_$(density)_chi_$(maxdim).h5"
end

# Reads energies and status only. MPS states stay on disk; load a single
# sector on demand with load_inherited_state to keep memory bounded.
function read_pair_binding_checkpoint(filename::AbstractString)
    energies = Dict{Int,Float64}()
    E_p = nothing
    completed = false

    isfile(filename) || return energies, E_p, completed

    h5open(filename, "r") do F
        names = Set(String.(keys(F)))
        N_values = Int[]
        if "N_values" in names
            N_values = Int.(read(F, "N_values"))
        else
            for name in names
                m = match(r"^E_N_(\d+)$", name)
                m === nothing && continue
                push!(N_values, parse(Int, m.captures[1]))
            end
        end

        for N in N_values
            ekey = energy_key(N)
            if ekey in names
                energies[N] = Float64(read(F, ekey))
            end
        end

        E_p = "E_p" in names ? Float64(read(F, "E_p")) : nothing
        completed = "completed" in names ? Bool(read(F, "completed")) : false
    end

    return energies, E_p, completed
end

function has_saved_state(filename::AbstractString, N_particles::Int)
    isfile(filename) || return false
    return h5open(F -> haskey(F, state_key(N_particles)), filename, "r")
end

function load_inherited_state(filename::AbstractString, N_particles::Int)
    isfile(filename) || return nothing
    psi = nothing
    h5open(filename, "r") do F
        key = state_key(N_particles)
        if haskey(F, key)
            psi = read(F, key, MPS)
        end
    end
    return psi
end

function write_or_replace(F, name::AbstractString, value)
    haskey(F, name) && HDF5.delete_object(F, name)
    F[name] = value
end

function write_or_replace(F, name::AbstractString, value::MPS)
    haskey(F, name) && HDF5.delete_object(F, name)
    write(F, name, value)
end

# Updates the checkpoint in place ("cw" mode) instead of rewriting it, so only
# the sectors passed in `states` need to be in memory; earlier sectors stay on
# disk untouched.
function write_pair_binding_checkpoint(filename::AbstractString; L::Int, U::Float64, V::Float64, density::Float64,
                                       t::Float64, t0::Float64, maxdim::Int, maxsweeps::Int, cutoff::Float64,
                                       E_tol::Float64, energies::Dict{Int,Float64}, states::Dict{Int,MPS},
                                       E_p=nothing, completed::Bool=false, inherit_from=nothing)
    outdir = dirname(filename)
    !isempty(outdir) && mkpath(outdir)

    N_values = sort(collect(keys(energies)))
    h5open(filename, "cw") do F
        write_or_replace(F, "L", L)
        write_or_replace(F, "U", U)
        write_or_replace(F, "V", V)
        write_or_replace(F, "density", density)
        write_or_replace(F, "t", t)
        write_or_replace(F, "t0", t0)
        write_or_replace(F, "maxdim", maxdim)
        write_or_replace(F, "maxsweeps", maxsweeps)
        write_or_replace(F, "cutoff", cutoff)
        write_or_replace(F, "E_tol", E_tol)
        write_or_replace(F, "N_values", N_values)
        write_or_replace(F, "completed", completed)
        inherit_from !== nothing && write_or_replace(F, "inherit_from", inherit_from)

        for N in N_values
            write_or_replace(F, energy_key(N), energies[N])
            if haskey(states, N)
                write_or_replace(F, state_key(N), ITensors.cpu(states[N]))
            end
        end

        E_p !== nothing && write_or_replace(F, "E_p", E_p)
    end
end

# -----------------------------------------------------------------------------
# Pair binding computation
# -----------------------------------------------------------------------------

function calculate_pair_binding_energy(L::Int, U::Float64, V::Float64, density::Float64; t::Float64=1.0,
                                       t0::Float64=1.0, maxdim::Int=1000, maxsweeps::Int=200,
                                       cutoff::Float64=1e-10, E_tol::Float64=DEFAULT_E_TOL,
                                       outfile::Union{Nothing,String}=nothing,
                                       inherit_from::Union{Nothing,String}=nothing, force::Bool=false)
    total_sites = 2 * L
    N_base = round(Int, density * total_sites)
    outfile = outfile === nothing ? default_pair_binding_file(L, U, V, t0, density, maxdim) : outfile
    println("="^60)
    println("Pair binding energy for Hubbard ladder")
    println("L = $L rungs, $(total_sites) sites")
    println("U = $U, V = $V, t = $t, t0 = $t0, density = $density")
    println("N_base = $N_base")
    println("Checkpoint = $outfile")
    inherit_from !== nothing && println("Inheriting states from $inherit_from")
    println("="^60)

    energies, saved_E_p, completed = read_pair_binding_checkpoint(outfile)
    if completed && saved_E_p !== nothing && !force
        println("Completed checkpoint found. Reusing E_p = $saved_E_p")
        return saved_E_p
    elseif !isempty(energies)
        println("Resuming from checkpoint with saved particle sectors: $(sort(collect(keys(energies))))")
    end

    targets = [(N_base - 2, "N-2"), (N_base - 1, "N-1"), (N_base, "N")]
    for (idx, (N_particles, label)) in enumerate(targets)
        println("\n[$idx/3] Computing E($label=$N_particles)")
        if !force && haskey(energies, N_particles) && has_saved_state(outfile, N_particles)
            println("  Reusing saved E($N_particles) = $(energies[N_particles])")
            continue
        end

        psi_init = load_inherited_state(outfile, N_particles)
        psi_init !== nothing && println("  Resuming from saved state $(state_key(N_particles))")
        if psi_init === nothing && inherit_from !== nothing
            psi_init = load_inherited_state(inherit_from, N_particles)
            psi_init !== nothing && println("  Loaded inherited state $(state_key(N_particles))")
        end

        energy, psi = run_dmrg_energy(L, N_particles; t=t, U=U, V=V, t0=t0, maxdim=maxdim,
            maxsweeps=maxsweeps, cutoff=cutoff, E_tol=E_tol, psi_init=psi_init)
        energies[N_particles] = energy
        println("  E($label) = $energy")

        write_pair_binding_checkpoint(outfile; L=L, U=U, V=V, density=density, t=t, t0=t0,
            maxdim=maxdim, maxsweeps=maxsweeps, cutoff=cutoff, E_tol=E_tol,
            energies=energies, states=Dict(N_particles => psi), completed=false,
            inherit_from=inherit_from)

        # Release this sector's MPS before starting the next one; only the
        # scalar energies are needed from here on.
        psi = nothing
        psi_init = nothing
        GC.gc()
    end

    ENm2 = energies[N_base - 2]
    ENm1 = energies[N_base - 1]
    EN = energies[N_base]


    E_p = ENm2 + EN - 2 * ENm1
    println("\n" * "="^60)
    println("RESULTS:")
    println("  E($(N_base-2)) = $ENm2")
    println("  E($(N_base-1)) = $ENm1")
    println("  E($N_base) = $EN")
    @printf("  E_p = %.8f  (%s hole pairing)\n", E_p, E_p < 0 ? "attractive" : "repulsive")
    println("="^60)

    # States are already on disk from the per-sector writes; this final update
    # only records E_p and flips completed=true.
    write_pair_binding_checkpoint(outfile; L=L, U=U, V=V, density=density, t=t, t0=t0,
        maxdim=maxdim, maxsweeps=maxsweeps, cutoff=cutoff, E_tol=E_tol,
        energies=energies, states=Dict{Int,MPS}(), E_p=E_p, completed=true, inherit_from=inherit_from)
    return E_p
end

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

function parse_cli_options(args)
    positional = String[]
    inherit_from = nothing
    outfile = nothing
    force = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--inherit-from"
            i == length(args) && error("--inherit-from requires a path")
            inherit_from = args[i + 1]
            i += 2
        elseif startswith(arg, "--inherit-from=")
            inherit_from = split(arg, "=", limit=2)[2]
            i += 1
        elseif arg == "--outfile"
            i == length(args) && error("--outfile requires a path")
            outfile = args[i + 1]
            i += 2
        elseif startswith(arg, "--outfile=")
            outfile = split(arg, "=", limit=2)[2]
            i += 1
        elseif arg == "--force"
            force = true
            i += 1
        else
            push!(positional, arg)
            i += 1
        end
    end

    # Backward-compatible positional extras:
    #   arg 6 = inherit_from, arg 7 = outfile
    if length(positional) >= 6 && inherit_from === nothing
        inherit_from = positional[6]
    end
    if length(positional) >= 7 && outfile === nothing
        outfile = positional[7]
    end
    length(positional) > 5 && (positional = positional[1:5])

    return positional, inherit_from, outfile, force
end

function main()
    enforce_single_slurm_task()

    positional, inherit_from, outfile, force = parse_cli_options(ARGS)
    if length(positional) != 5
        println("Usage: julia calculate_E_p_ladder.jl <L> <U> <V> <t0> <density> [--inherit-from previous.h5] [--outfile output.h5] [--force]")
        println("  L: number of rungs (total sites = 2L)")
        println("  U: onsite interaction strength")
        println("  V: nearest-neighbor interaction strength")
        println("  t0: rung hopping")
        println("  density: particle density (0-2)")
        println("  --inherit-from: optional HDF5 checkpoint to warm-start matching N sectors")
        println("  --outfile: optional HDF5 checkpoint path for this run")
        println("  --force: recompute sectors even if they are present in the checkpoint")
        exit(1)
    end

    L = parse(Int, positional[1])
    U = parse(Float64, positional[2])
    V = parse(Float64, positional[3])
    t0 = parse(Float64, positional[4])
    density = parse(Float64, positional[5])

    println("Calculating pair binding energy for ladder")
    println("L=$L, U=$U, V=$V, t0=$t0, density=$density")
    flush(stdout)

    configure_threading()

    E_p = calculate_pair_binding_energy(L, U, V, density; t=1.0, t0=t0,
        outfile=outfile, inherit_from=inherit_from, force=force)
    println("RESULT: E_p = $E_p")
end

main()
