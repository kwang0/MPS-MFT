using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

# -----------------------------------------------------------------------------
# Basic ladder construction (mirror main_loop_script_ladder.jl)
# -----------------------------------------------------------------------------

function make_sites_conserveN(L::Int)
    return siteinds("Electron", L; conserve_sz=true, conserve_nf=true)
end

function rung_leg_to_site(rung::Int, leg::Int)
    return 2 * (rung - 1) + leg + 1
end

function build_hubbard_ladder(; L::Int, t::Float64=1.0, U::Float64=0.0, mu::Float64=0.0, t0::Float64=0.0)
    sites = make_sites_conserveN(2 * L)
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
        end
    end

    if t0 != 0
        for rung in 1:L
            s0 = rung_leg_to_site(rung, 0)
            s1 = rung_leg_to_site(rung, 1)
            add!(os, -t0, "Cdagup", s0, "Cup", s1)
            add!(os, -t0, "Cdagup", s1, "Cup", s0)
            add!(os, -t0, "Cdagdn", s0, "Cdn", s1)
            add!(os, -t0, "Cdagdn", s1, "Cdn", s0)
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

function run_dmrg_energy(L::Int, N_particles::Int; t::Float64=1.0, U::Float64=0.0, t0::Float64=0.0,
                         mu::Float64=0.0, maxdim::Int=1000, maxsweeps::Int=60, cutoff::Float64=1e-10)
    N_particles < 0 && error("Cannot target negative particle number: $N_particles")
    states = ladder_initial_state(L, N_particles)
    total, n_up, n_dn = count_particles(states)
    total == N_particles || error("Initial state mismatch: expected $N_particles, got $total")
    println("  Initial state: $total particles ($n_up up, $n_dn dn)")

    sites, H = build_hubbard_ladder(L=L, t=t, U=U, mu=mu, t0=t0)
    psi0 = productMPS(sites, states)
    sweeps = make_sweeps(maxsweeps, maxdim; cutoff=cutoff)
    energy, _ = dmrg(H, psi0, sweeps; outputlevel=1)
    return energy
end

# -----------------------------------------------------------------------------
# Pair binding computation
# -----------------------------------------------------------------------------

function calculate_pair_binding_energy(L::Int, U::Float64, density::Float64; t::Float64=1.0,
                                       t0::Float64=1.0, maxdim::Int=1000, maxsweeps::Int=100, offset::Int=0)
    total_sites = 2 * L
    N_base = round(Int, density * total_sites)
    println("="^60)
    println("Pair binding energy for Hubbard ladder")
    println("L = $L rungs, $(total_sites) sites")
    println("U = $U, t = $t, t0 = $t0, density = $density")
    println("N_base = $N_base")
    println("="^60)

    if offset == 2
        println("\n[1/3] Computing E(N-2=$(N_base-2))")
        ENm2 = run_dmrg_energy(L, N_base - 2; t=t, U=U, t0=t0, maxdim=maxdim, maxsweeps=maxsweeps)
        println("  E(N-2) = $ENm2")
        return
    elseif offset == 1
        println("\n[2/3] Computing E(N-1=$(N_base-1))")
        ENm1 = run_dmrg_energy(L, N_base - 1; t=t, U=U, t0=t0, maxdim=maxdim, maxsweeps=maxsweeps)
        println("  E(N-1) = $ENm1")
        return
    elseif offset == 0
        println("\n[3/3] Computing E(N=$N_base)")
        EN = run_dmrg_energy(L, N_base; t=t, U=U, t0=t0, maxdim=maxdim, maxsweeps=maxsweeps)
        println("  E(N) = $EN")
        return
    else
        return
    end

    # E_p = ENm2 + EN - 2 * ENm1
    # println("\n" * "="^60)
    # println("RESULTS:")
    # println("  E($(N_base-2)) = $ENm2")
    # println("  E($(N_base-1)) = $ENm1")
    # println("  E($N_base)     = $EN")
    # @printf("  E_p = %.8f  (%s hole pairing)\n", E_p, E_p < 0 ? "attractive" : "repulsive")
    # println("="^60)
    # return E_p
end

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

function main()
    # if length(ARGS) != 4
    #     println("Usage: julia calculate_E_p_ladder.jl <L> <U> <t0> <density>")
    #     println("  L: number of rungs (total sites = 2L)")
    #     println("  U: onsite interaction strength")
    #     println("  t0: rung hopping")
    #     println("  density: particle density (0-2)")
    #     exit(1)
    # end

    L = parse(Int, ARGS[1])
    U = parse(Float64, ARGS[2])
    t0 = parse(Float64, ARGS[3])
    density = parse(Float64, ARGS[4])
    offset = parse(Int, ARGS[5])  # 0: N, 1: N-1, 2: N-2

    # println("Calculating pair binding energy for ladder")
    println("L=$L, U=$U, t0=$t0, density=$density")
    flush(stdout)

    BLAS.set_num_threads(1)
    ITensors.Strided.disable_threads()
    ITensors.enable_threaded_blocksparse()

    E_p = calculate_pair_binding_energy(L, U, density; t=1.0, t0=t0, offset=offset)
    # println("RESULT: E_p = $E_p")
end

main()
