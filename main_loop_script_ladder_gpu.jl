using ITensors
using ITensorMPS
using CUDA
using HDF5
using LinearAlgebra
using Printf
using Serialization
using TickTock
using Random

# Global flag to signal time limit exceeded
const TIME_LIMIT_EXCEEDED = Ref(false)
const TIME_LIMIT_SECONDS = 47.5 * 60 * 60  # 47.5 hours

# Observer that checks timer after each sweep
mutable struct TimerObserver <: AbstractObserver
end

function ITensorMPS.checkdone!(o::TimerObserver; kwargs...)
    if peektimer() > TIME_LIMIT_SECONDS
        println("Time limit approaching, stopping DMRG early...")
        TIME_LIMIT_EXCEEDED[] = true
        return true  # Signal DMRG to stop
    end
    return false
end

# Build Electron site indices with no conservation for GPU runs
function make_sites(L::Int)
    return siteinds("Electron", L; conserve_sz=false, conserve_nfparity=false)
end

# Optionally build Electron sites that DO conserve particle number (used for E_p calculation)
function make_sites_conserveN(L::Int)
    return siteinds("Electron", L; conserve_sz=true, conserve_nf=true)
end

# Compute average density
function average_density(psi::MPS)
    return sum(expect(psi, "Ntot")) / length(psi)
end

# Construct a product state with specified density
function density_product_state(L::Int, density::Float64)
    N_particles = round(Int, density * L)
    states = fill("Emp", L)
    
    # Fill with alternating up/down spins up to N_particles
    n_up = div(N_particles, 2)
    n_dn = N_particles - n_up
    
    # Fill first N_particles sites
    for i in 1:n_up
        states[i] = "Up"
    end
    for i in 1:n_dn
        states[n_up + i] = "Dn"
    end
    
    # Shuffle to distribute particles and spins uniformly
    rng = MersenneTwister(1234)
    shuffle!(rng, states)
    
    return states
end

# -----------------------------
# Ladder geometry helpers
# -----------------------------
# Ladder has L rungs (i=1..L) and 2 legs (j=0,1)
# MPS site indexing: site s = 2*(i-1) + j + 1
# So rung 1: sites 1,2; rung 2: sites 3,4; etc.

function rung_leg_to_site(i::Int, j::Int)
    return 2 * (i - 1) + j + 1
end

function site_to_rung_leg(s::Int)
    i = div(s - 1, 2) + 1  # rung index
    j = (s - 1) % 2        # leg index (0 or 1)
    return i, j
end

# -----------------------------
# Pair-binding energy E_p at half-filling n=0.5 (conserving N)
# -----------------------------

# Standard ladder Hubbard MPO (no MF terms), with chemical potential mu, t=1 by default, and rung hopping t0
function build_MPO_Hubbard_standard(; L::Int, t::Float64=1.0, U::Float64=0.0, mu::Float64=0.0, t0::Float64=0.0)
    # L is number of rungs, total sites = 2L
    s = make_sites_conserveN(2 * L)
    os = OpSum()
    # Onsite
    for i in 1:(2*L)
        add!(os, U, "Nupdn", i)
        add!(os, -mu, "Ntot", i)
    end
    # NN hopping along legs (open bc): L-1 bonds connect L rungs
    for i_rung in 1:(L-1)
        for j_leg in 0:1
            site = rung_leg_to_site(i_rung, j_leg)
            next_site = rung_leg_to_site(i_rung + 1, j_leg)
            add!(os, -t, "Cdagup", site, "Cup", next_site)
            add!(os, -t, "Cdagup", next_site, "Cup", site)
            add!(os, -t, "Cdagdn", site, "Cdn", next_site)
            add!(os, -t, "Cdagdn", next_site, "Cdn", site)
        end
    end
    # Rung hopping (between legs of same rung)
    if t0 != 0
        for i_rung in 1:L
            site0 = rung_leg_to_site(i_rung, 0)
            site1 = rung_leg_to_site(i_rung, 1)
            add!(os, -t0, "Cdagup", site0, "Cup", site1)
            add!(os, -t0, "Cdagup", site1, "Cup", site0)
            add!(os, -t0, "Cdagdn", site0, "Cdn", site1)
            add!(os, -t0, "Cdagdn", site1, "Cdn", site0)
        end
    end
    H = cu(MPO(os, s))
    return s, H
end

# DMRG with a specific product state (conserving N sector)
function run_dmrg_with_prodstate(s, H, prod_states::Vector{String}; nsweeps, maxdim, cutoff)
    psi0 = productMPS(s, prod_states)
    E, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    return E
end


# Pair binding energy: E_pair = 2 E(N+1,S=1/2) - E(N,S=0) - E(N+2,S=0)
function calculate_pair_binding_energy(L::Int, U::Float64; t::Float64=1.0, t0::Float64=1.0)
    nsweeps = 20
    maxdim = 200
    cutoff = 1e-10

    s, H = build_MPO_Hubbard_standard(L=L, t=t, U=U, mu=0.0, t0=t0)

    # Base product state at specified density
    base0 = density_product_state(2 * L, 0.5)  # Default to half-filling for E_p calculation

    # E(N)
    EN = run_dmrg_with_prodstate(s, H, copy(base0); nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)

    # E(N+1): flip one Emp -> Up (if available)
    baseN1 = copy(base0)
    for i in 1:(2*L)
        if baseN1[i] == "Emp"
            baseN1[i] = "Up"
            break
        end
    end
    ENp1 = run_dmrg_with_prodstate(s, H, baseN1; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)

    # E(N+2): add one more Down similarly
    baseN2 = copy(baseN1)
    for i in 1:(2*L)
        if baseN2[i] == "Emp"
            baseN2[i] = "Dn"
            break
        end
    end
    ENp2 = run_dmrg_with_prodstate(s, H, baseN2; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)

    pairing_energy = 2 * ENp1 - EN - ENp2
    return pairing_energy
end

# Build mean-field-augmented ladder Hubbard MPO
# Parameters:
#  - L (number of rungs), t, U, mu, V (nn density-density), r_range
#  - alpha[L,L,2,2]: pairing between rungs i,i' and legs j,j'
#  - beta[2,L,L,2,2]: spin-resolved hopping between rungs i,i' and legs j,j'
#  - t0: rung hopping
#  - optional phi_ext (Peierls flux per hop)
function build_MPO_MF(; L::Int, t::Float64, U::Float64, mu::Float64, V::Float64,
    r_range::Int, alpha::Array{Float64,4}, beta::Array{Float64,5}, t0::Float64,
    phi_ext::Union{Nothing,Float64}=nothing)
    # L is number of rungs, total sites = 2L
    s = make_sites(2 * L)
    os = OpSum()

    # Onsite terms: -mu*Ntot + U*Nup*Ndn
    for i in 1:(2*L)
        add!(os, -mu, "Ntot", i)
        add!(os, U, "Nupdn", i)
    end

    # Nearest-neighbor hopping along legs (open bc): L-1 bonds connect L rungs
    for i_rung in 1:(L-1)
        for j_leg in 0:1
            site = rung_leg_to_site(i_rung, j_leg)
            next_site = rung_leg_to_site(i_rung + 1, j_leg)
            hop = -t
            if phi_ext !== nothing
                # Simple Peierls phase e^{i*phi} for +x hops; use symmetric h.c.
                ϕ = 2π * phi_ext
                hop_fwd = hop * cos(ϕ)
                hop_im = hop * sin(ϕ)
                # Up spin
                add!(os, hop_fwd, "Cdagup", site, "Cup", next_site)
                add!(os, hop_fwd, "Cdagup", next_site, "Cup", site)
                add!(os, hop_im, "iCdagup", site, "Cup", next_site)
                add!(os, -hop_im, "iCdagup", next_site, "Cup", site)
                # Down spin
                add!(os, hop_fwd, "Cdagdn", site, "Cdn", next_site)
                add!(os, hop_fwd, "Cdagdn", next_site, "Cdn", site)
                add!(os, hop_im, "iCdagdn", site, "Cdn", next_site)
                add!(os, -hop_im, "iCdagdn", next_site, "Cdn", site)
            else
                # Standard hermitian hopping
                add!(os, hop, "Cdagup", site, "Cup", next_site)
                add!(os, hop, "Cdagup", next_site, "Cup", site)
                add!(os, hop, "Cdagdn", site, "Cdn", next_site)
                add!(os, hop, "Cdagdn", next_site, "Cdn", site)
            end
        end
    end

    # Rung hopping (between legs of same rung)
    if t0 != 0
        for i_rung in 1:L
            site0 = rung_leg_to_site(i_rung, 0)
            site1 = rung_leg_to_site(i_rung, 1)
            add!(os, -t0, "Cdagup", site0, "Cup", site1)
            add!(os, -t0, "Cdagup", site1, "Cup", site0)
            add!(os, -t0, "Cdagdn", site0, "Cdn", site1)
            add!(os, -t0, "Cdagdn", site1, "Cdn", site0)
        end
    end

    # Optional nearest-neighbor density-density V N_i N_{i+1} along legs
    if V != 0
        for i_rung in 1:(L-1)
            for j_leg in 0:1
                site = rung_leg_to_site(i_rung, j_leg)
                next_site = rung_leg_to_site(i_rung + 1, j_leg)
                add!(os, V, "Nup", site, "Nup", next_site)
                add!(os, V, "Nup", site, "Ndn", next_site)
                add!(os, V, "Ndn", site, "Nup", next_site)
                add!(os, V, "Ndn", site, "Ndn", next_site)
            end
        end
    end

    # Pairing alpha terms: alpha[i, i', j, j'] for rungs i,i' and legs j,j'
    for i in 1:L, i_p in 1:L
        if abs(i - i_p) <= r_range
            for j in 0:1, j_p in 0:1
                a = alpha[i, i_p, j+1, j_p+1]  # Julia 1-indexed
                if a != 0
                    site_i = rung_leg_to_site(i, j)
                    site_ip = rung_leg_to_site(i_p, j_p)
                    add!(os, -a, "Cup", site_ip, "Cdn", site_i)
                    add!(os, -a, "Cdagdn", site_i, "Cdagup", site_ip)  # h.c.
                end
            end
        end
    end

    # Normal beta terms: spin-resolved hopping beta[σ, i, i', j, j']
    for i in 1:L, i_p in 1:L
        if abs(i - i_p) <= r_range
            for j in 0:1, j_p in 0:1
                site_i = rung_leg_to_site(i, j)
                site_ip = rung_leg_to_site(i_p, j_p)
                
                if site_i != site_ip # No onsite terms for now (don't consider CDW)
                    bdn = beta[1, i, i_p, j+1, j_p+1]  # Julia 1-indexed
                    if bdn != 0
                        add!(os, bdn, "Cdagdn", site_i, "Cdn", site_ip)
                        # add!(os, bdn, "Cdagdn", site_ip, "Cdn", site_i) # Double counted
                    end
                    
                    bup = beta[2, i, i_p, j+1, j_p+1]
                    if bup != 0
                        add!(os, bup, "Cdagup", site_i, "Cup", site_ip)
                        # add!(os, bup, "Cdagup", site_ip, "Cup", site_i)
                    end
                end
            end
        end
    end

    H = cu(MPO(os, s))
    return s, H
end

# DMRG ground state with product-state initialization
function run_dmrg_ground(s, H, density::Float64; nsweeps=10, maxdim=200, cutoff=1e-10)
    L_sites = length(s)  # Total MPS sites (2L for ladder)
    psi0 = cu(productMPS(s, density_product_state(L_sites, density)))

    sweeps = Sweeps(nsweeps)
    maxdim!(sweeps, min(10, maxdim), min(20, maxdim), 100, maxdim)
    cutoff!(sweeps, cutoff)
    noise!(sweeps, 1e-5, 1e-6, 1e-7, 1e-8, 0.0)

    obs = TimerObserver()
    E0, psi0 = dmrg(H, psi0, sweeps; observer=obs)
    return E0, psi0
end

# First excited state energy via DMRG orthogonalization to ground state
function run_dmrg_excited(s, H, psi0, density::Float64; nsweeps=10, maxdim=200, cutoff=1e-10)
    L_sites = length(s)  # Total MPS sites (2L for ladder)
    psi1 = cu(productMPS(s, density_product_state(L_sites, density)))

    sweeps = Sweeps(nsweeps)
    maxdim!(sweeps, min(10, maxdim), min(20, maxdim), 100, maxdim)
    cutoff!(sweeps, cutoff)
    noise!(sweeps, 1e-5, 1e-6, 1e-7, 1e-8, 0.0)

    obs = TimerObserver()
    _, psi1 = dmrg(H, [psi0], psi1, sweeps; observer=obs)
    E1 = inner(psi1', H, psi1)
    return E1, psi1
end

# Solve one Hamiltonian instance and return (density, energy, psi)
function solve_Ham(mu, model_params, alpha, beta, density; nsweeps::Int=10, maxdim=200, cutoff=1e-10)
    L = model_params[:L]
    t = model_params[:t]
    U = model_params[:U]
    V = get(model_params, :V, 0.0)
    r_range = model_params[:r_range]
    t0 = model_params[:t0]
    φ = get(model_params, :phi_ext, nothing)

    s, H = build_MPO_MF(L=L, t=t, U=U, mu=mu, V=V, r_range=r_range, alpha=alpha, beta=beta, t0=t0, phi_ext=φ)
    E, psi = run_dmrg_ground(s, H, density; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    n = average_density(psi)
    return n, E, psi, s, H
end

# Check convergence of alpha/beta by comparing diagonals up to r_range with relative tolerance
# function close_ab(alpha, alpha_meas, beta, beta_meas, r_range; thresh=1e-4)
#     eps = 1e-12
#     for r in 0:r_range
#         # alpha
#         a = diag(alpha, r)
#         am = diag(alpha_meas, r)
#         if any(abs.(am .- a) ./ (abs.(a) .+ eps) .> thresh)
#             println("Alpha not converged. Triggered by relative error of $(maximum(abs.(am .- a) ./ (abs.(a) .+ eps)))")
#             return false
#         end
#         # beta down
#         b = diag(view(beta, 1, :, :), r)
#         bm = diag(view(beta_meas, 1, :, :), r)
#         if any(abs.(bm .- b) ./ (abs.(b) .+ eps) .> thresh)
#             println("Beta not converged. Triggered by relative error of $(maximum(abs.(bm .- b) ./ (abs.(b) .+ eps)))")
#             return false
#         end
#         # beta up
#         b = diag(view(beta, 2, :, :), r)
#         bm = diag(view(beta_meas, 2, :, :), r)
#         if any(abs.(bm .- b) ./ (abs.(b) .+ eps) .> thresh)
#             println("Beta not converged. Triggered by relative error of $(maximum(abs.(bm .- b) ./ (abs.(b) .+ eps)))")
#             return false
#         end
#     end
#     return true
# end

# Check convergence of alpha/beta by checking rms of relative errors along diagonals up to r_range
function close_ab(alpha, alpha_meas, beta, beta_meas, r_range; thresh=1e-4)
    eps = 1e-12
    for j in 1:2, j_p in 1:2
        for r in 0:r_range
            n_diag = size(alpha, 1) - r
            n_diag <= 0 && continue
            # alpha
            a = [alpha[i, i+r, j, j_p] for i in 1:n_diag]
            am = [alpha_meas[i, i+r, j, j_p] for i in 1:n_diag]
            errs = (am .- a) ./ (abs.(a) .+ eps)
            rms_err = sqrt(sum(errs .^ 2) / length(errs))
            if rms_err > thresh
                println("Alpha not converged.")
                return false
            end
        end
    end

    for σ in 1:2, j in 1:2, j_p in 1:2
        for r in 0:r_range
            n_diag = size(beta, 2) - r
            n_diag <= 0 && continue
            # beta
            b = [beta[σ, i, i+r, j, j_p] for i in 1:n_diag]
            bm = [beta_meas[σ, i, i+r, j, j_p] for i in 1:n_diag]
            errs = (bm .- b) ./ (abs.(b) .+ eps)
            rms_err = sqrt(sum(errs .^ 2) / length(errs))
            if rms_err > thresh
                println("Beta not converged.")
                return false
            end
        end
    end
    return true
end

# Find mu to hit target density using secant with bisection fallback
function find_mu_for_target_density(model_params, alpha, beta, mu_init, n_target;
    tol=1e-3, delta_mu=0.01, max_iter=100, dmrg_kw)
    mu0 = mu_init
    n0, E0, psi0, s0, H0 = solve_Ham(mu0, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
    
    if abs(n0 - n_target) / max(n_target, 1e-12) <= tol
        println("Same mu: $mu0 and continuing")
        return mu0, n0, E0, psi0, s0, H0
    end
    println("Not same mu, searching again")

    # 0 and 1 are fixed to be left and right. "new" keeps track of latest
    if n0 < n_target
        mu_new = mu0 + delta_mu
        n_new, E_new, psi_new, s_new, H_new = solve_Ham(mu_new, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
        mu1, n1 = mu_new, n_new
    else
        mu1, n1 = mu0, n0
        mu_new = mu0 - delta_mu
        n_new, E_new, psi_new, s_new, H_new = solve_Ham(mu_new, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
        mu0, n0 = mu_new, n_new
    end

    factor = 1.0
    multiplier = 3.0
    for it in 1:max_iter
        if TIME_LIMIT_EXCEEDED[] || peektimer() > TIME_LIMIT_SECONDS
            println("Time limit exceeded, exiting loop")
            return mu_new, n_new, E_new, psi_new, s_new, H_new
        end

        println("n0: $n0; n1: $n1; mu0: $mu0; mu1: $mu1")
        if abs(n_new - n_target) / max(n_target, 1e-12) <= tol
            println("FOUND MU; mu: $mu_new  n: $n_new")
            return mu_new, n_new, E_new, psi_new, s_new, H_new
        end

        if (n_target >= n1 && n_target >= n0)
            println("n_target > n1 and n_target > n0 triggered")
            mu0, n0 = mu1, n1
            mu_new = mu0 + factor * delta_mu
            n_new, E_new, psi_new, s_new, H_new = solve_Ham(mu_new, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
            mu1, n1 = mu_new, n_new
            factor *= multiplier # increase step size exponentially until range found
        elseif (n_target <= n1 && n_target <= n0)
            println("n_target < n1 and n_target < n0 triggered")
            mu1, n1 = mu0, n0
            mu_new = mu0 - factor * delta_mu
            n_new, E_new, psi_new, s_new, H_new = solve_Ham(mu_new, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
            mu0, n0 = mu_new, n_new
            factor *= multiplier # increase step size exponentially until range found
        else
            # refinement loop
            println("\nRange found between mu0: $mu0 (n0: $n0) and mu1: $mu1 (n1: $n1); refining...")
            factor = 1.0  # reset factor
            for j in 1:max_iter
                if TIME_LIMIT_EXCEEDED[] || peektimer() > TIME_LIMIT_SECONDS
                    println("Time limit exceeded, exiting loop")
                    return mu_new, n_new, E_new, psi_new, s_new, H_new
                end
                println("n0: $n0; n1: $n1; mu0: $mu0; mu1: $mu1")
                
                # Check if either existing endpoint already satisfies the tolerance
                if abs(n0 - n_target) / max(n_target, 1e-12) <= tol
                    println("FOUND MU (n0 already at target); mu: $mu0  n: $n0")
                    # Need to solve again to get full return values if not already available
                    n0, E0, psi0, s0, H0 = solve_Ham(mu0, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
                    return mu0, n0, E0, psi0, s0, H0
                end
                if abs(n1 - n_target) / max(n_target, 1e-12) <= tol
                    println("FOUND MU (n1 already at target); mu: $mu1  n: $n1")
                    n1, E1, psi1, s1, H1 = solve_Ham(mu1, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
                    return mu1, n1, E1, psi1, s1, H1
                end
                
                # Check if mu interval is too small to refine further
                if abs(mu1 - mu0) < 1e-10
                    println("Mu interval too small ($(abs(mu1 - mu0))), returning best result...")
                    # Return the endpoint closer to target
                    if abs(n0 - n_target) < abs(n1 - n_target)
                        n0, E0, psi0, s0, H0 = solve_Ham(mu0, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
                        return mu0, n0, E0, psi0, s0, H0
                    else
                        n1, E1, psi1, s1, H1 = solve_Ham(mu1, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
                        return mu1, n1, E1, psi1, s1, H1
                    end
                end
                
                if abs(n1 - n0) > 1e-12
                    println("Trying secant step...")

                    # Update mu that is farther from target
                    if abs(n1 - n_target) > abs(n0 - n_target)
                        mu1 = mu1 - (n1 - n_target) * (mu1 - mu0) / (n1 - n0)
                        n1, E1, psi1, s1, H1 = solve_Ham(mu1, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
                        mu_new, n_new, E_new, psi_new, s_new, H_new = mu1, n1, E1, psi1, s1, H1
                    else
                        mu0 = mu0 + (n_target - n0) * (mu1 - mu0) / (n1 - n0)
                        n0, E0, psi0, s0, H0 = solve_Ham(mu0, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
                        mu_new, n_new, E_new, psi_new, s_new, H_new = mu0, n0, E0, psi0, s0, H0
                    end

                    # Ensure mu0 < mu1 for next iteration
                    if mu1 < mu0
                        mu0, mu1 = mu1, mu0
                        n0, n1 = n1, n0
                    end

                    if abs(n_new - n_target) / max(n_target, 1e-12) <= tol
                        println("FOUND MU (secant); mu: $mu_new  n: $n_new")
                        return mu_new, n_new, E_new, psi_new, s_new, H_new
                    end
                else
                    println("Secant unstable (n0 ≈ n1), falling back to bisection...")
                    
                    # If densities are identical and interval is tiny, we've converged numerically
                    if abs(mu1 - mu0) < 1e-8
                        println("Densities identical and mu interval small, returning closest to target...")
                        if abs(n0 - n_target) < abs(n1 - n_target)
                            return mu0, n0, E_new, psi_new, s_new, H_new
                        else
                            return mu1, n1, E_new, psi_new, s_new, H_new
                        end
                    end
                    
                    mu_new = 0.5 * (mu0 + mu1)
                    n_new, E_new, psi_new, s_new, H_new = solve_Ham(mu_new, model_params, alpha, beta, model_params[:density]; dmrg_kw...)
                    if abs(n_new - n_target) / max(n_target, 1e-12) <= tol
                        println("FOUND MU (bisection); mu: $mu_new  n: $n_new")
                        return mu_new, n_new, E_new, psi_new, s_new, H_new
                    end
                    if n_target > n_new
                        mu0, n0 = mu_new, n_new
                    else
                        mu1, n1 = mu_new, n_new
                    end
                end
            end
            error("Target density not achieved within maximum iterations in refinement loop")
        end
    end
    error("Target density not achieved within maximum iterations")
end

# Measure alpha, beta from correlators using ladder formulas (Appendix E)
function calculate_alpha_beta_measured(psi::MPS, s; L::Int, r_range::Int, z_c::Int, t_p::Float64, E_p::Float64, threshold::Float64=1e-5)
    # L is number of rungs
    # alpha[i, i', j, j'] for rungs i,i' and legs j,j'
    # beta[σ, i, i', j, j'] for spin σ, rungs i,i' and legs j,j'
    alpha_meas = zeros(Float64, L, L, 2, 2)
    beta_meas = zeros(Float64, 2, L, L, 2, 2)

    pref = 2 * t_p^2 / E_p

    # Compute all needed correlation matrices on MPS sites (2L total)
    # For pairing: <c_up c_dn>
    C_pair = correlation_matrix(psi, "Cup", "Cdn")
    # For exchange down: <c†_dn c_dn>
    C_exc_dn = correlation_matrix(psi, "Cdagdn", "Cdn")
    # For exchange up: <c†_up c_up>
    C_exc_up = correlation_matrix(psi, "Cdagup", "Cup")

    # Compute alpha and beta for each rung pair within r_range
    for i in 1:L, i_p in 1:L
        if abs(i - i_p) <= r_range
            # Map rung indices to MPS sites
            site_i0 = rung_leg_to_site(i, 0)
            site_i1 = rung_leg_to_site(i, 1)
            site_ip0 = rung_leg_to_site(i_p, 0)
            site_ip1 = rung_leg_to_site(i_p, 1)

            # Alpha pairing terms (Appendix E, Eqs. E10-E13)
            # Same-leg pairing (0,0): α_{i,i',0,0} = pref * (<c_{i',1,↑} c_{i,1,↓}> + 2<c_{i',0,↑} c_{i,0,↓}>)
            val = pref * (C_pair[site_ip1, site_i1] + 2 * C_pair[site_ip0, site_i0])
            alpha_meas[i, i_p, 1, 1] = (abs(val) > threshold) ? val : 0.0

            # Same-leg pairing (1,1): α_{i,i',1,1} = pref * (<c_{i',0,↑} c_{i,0,↓}> + 2<c_{i',1,↑} c_{i,1,↓}>)
            val = pref * (C_pair[site_ip0, site_i0] + 2 * C_pair[site_ip1, site_i1])
            alpha_meas[i, i_p, 2, 2] = (abs(val) > threshold) ? val : 0.0

            # Cross-leg pairing (1,0): α_{i,i',1,0} = 2*pref * <c_{i',0,↑} c_{i,1,↓}>
            val = 2 * pref * C_pair[site_ip0, site_i1]
            alpha_meas[i, i_p, 2, 1] = (abs(val) > threshold) ? val : 0.0

            # Cross-leg pairing (0,1): α_{i,i',0,1} = 2*pref * <c_{i',1,↑} c_{i,0,↓}>
            val = 2 * pref * C_pair[site_ip1, site_i0]
            alpha_meas[i, i_p, 1, 2] = (abs(val) > threshold) ? val : 0.0

            # Beta exchange terms (Appendix E, Eqs. E14-E17)
            # Down spin, same-leg (0,0): β_{i,i',0,0,↓} = pref * (<c†_{i,1,↓} c_{i',1,↓}> + 2<c†_{i,0,↓} c_{i',0,↓}>)
            val = pref * (C_exc_dn[site_i1, site_ip1] + 2 * C_exc_dn[site_i0, site_ip0])
            beta_meas[1, i, i_p, 1, 1] = (abs(val) > threshold) ? val : 0.0

            # Down spin, same-leg (1,1): β_{i,i',1,1,↓} = pref * (<c†_{i,0,↓} c_{i',0,↓}> + 2<c†_{i,1,↓} c_{i',1,↓}>)
            val = pref * (C_exc_dn[site_i0, site_ip0] + 2 * C_exc_dn[site_i1, site_ip1])
            beta_meas[1, i, i_p, 2, 2] = (abs(val) > threshold) ? val : 0.0

            # Down spin, cross-leg (1,0): β_{i,i',1,0,↓} = 2*pref * <c†_{i',0,↓} c_{i,1,↓}>
            val = 2 * pref * C_exc_dn[site_ip0, site_i1]
            beta_meas[1, i, i_p, 2, 1] = (abs(val) > threshold) ? val : 0.0

            # Down spin, cross-leg (0,1): β_{i,i',0,1,↓} = 2*pref * <c†_{i',1,↓} c_{i,0,↓}>
            val = 2 * pref * C_exc_dn[site_ip1, site_i0]
            beta_meas[1, i, i_p, 1, 2] = (abs(val) > threshold) ? val : 0.0

            # Up spin, same-leg (0,0): β_{i,i',0,0,↑} = pref * (<c†_{i,1,↑} c_{i',1,↑}> + 2<c†_{i,0,↑} c_{i',0,↑}>)
            val = pref * (C_exc_up[site_i1, site_ip1] + 2 * C_exc_up[site_i0, site_ip0])
            beta_meas[2, i, i_p, 1, 1] = (abs(val) > threshold) ? val : 0.0

            # Up spin, same-leg (1,1): β_{i,i',1,1,↑} = pref * (<c†_{i,0,↑} c_{i',0,↑}> + 2<c†_{i,1,↑} c_{i',1,↑}>)
            val = pref * (C_exc_up[site_i0, site_ip0] + 2 * C_exc_up[site_i1, site_ip1])
            beta_meas[2, i, i_p, 2, 2] = (abs(val) > threshold) ? val : 0.0

            # Up spin, cross-leg (1,0): β_{i,i',1,0,↑} = 2*pref * <c†_{i',0,↑} c_{i,1,↑}>
            val = 2 * pref * C_exc_up[site_ip0, site_i1]
            beta_meas[2, i, i_p, 2, 1] = (abs(val) > threshold) ? val : 0.0

            # Up spin, cross-leg (0,1): β_{i,i',0,1,↑} = 2*pref * <c†_{i',1,↑} c_{i,0,↑}>
            val = 2 * pref * C_exc_up[site_ip1, site_i0]
            beta_meas[2, i, i_p, 1, 2] = (abs(val) > threshold) ? val : 0.0
        end
    end

    return alpha_meas, beta_meas
end

# Order parameter: average of diagonal r=0 of ⟨ Cup_i Cdn_{i+r} ⟩, excluding 10-site edges
function order_parameter(psi::MPS, s)
    L = length(s)
    C = correlation_matrix(psi, "Cup", "Cdn")
    diag0 = [C[i, i] for i in 1:L]
    lo = max(1, 10)
    hi = max(lo, L - 10)
    return abs(sum(diag0[lo:hi]) / max(hi - lo + 1, 1))
end

# d-wave singlet order parameter for ladder
# Measures pairing with d-wave symmetry: positive along legs, negative across rungs
function dwave_order_parameter(psi::MPS, s; L_rungs::Int)
    C = correlation_matrix(psi, "Cup", "Cdn")
    
    # Leg-direction pairing (positive weight)
    leg_pair = 0.0
    for i_rung in 1:(L_rungs-1)
        for j_leg in 0:1
            site1 = rung_leg_to_site(i_rung, j_leg)
            site2 = rung_leg_to_site(i_rung + 1, j_leg)
            leg_pair += C[site1, site2]
        end
    end
    leg_pair /= (2 * (L_rungs - 1))
    
    # Rung-direction pairing (negative weight for d-wave)
    rung_pair = 0.0
    for i_rung in 1:L_rungs
        site0 = rung_leg_to_site(i_rung, 0)
        site1 = rung_leg_to_site(i_rung, 1)
        rung_pair += C[site0, site1]
    end
    rung_pair /= L_rungs
    
    # d-wave order parameter (without absolute value to preserve sign)
    return leg_pair - rung_pair
end

# Charge density wave order parameter
# Measures density modulation at wavevector q (default q=π for period-2 CDW)
function cdw_order_parameter(psi::MPS, s; L_rungs::Int, q::Float64=Float64(π))
    # Measure density on each site
    n = expect(psi, "Ntot")
    
    # Compute structure factor S(q) = (1/N) Σ_i n_i e^{iq·r_i}
    cdw_amp = 0.0 + 0.0im
    for i_rung in 1:L_rungs
        for j_leg in 0:1
            site = rung_leg_to_site(i_rung, j_leg)
            # Position along leg direction (rung index)
            r = i_rung - 1
            cdw_amp += n[site] * exp(im * q * r)
        end
    end
    cdw_amp /= (2 * L_rungs)
    
    return abs(cdw_amp)
end

# Main SCF loop: adjust mu for target density, then update alpha/beta until self-consistent
function main_loop(model_params; n_target::Float64, E_p::Float64, z_c::Int=4, alpha_list, beta_list, max_iter::Int=150, nsweeps=10, maxdim=200, cutoff=1e-10, damp=0.5)
    alpha = model_params[:alpha]
    beta = model_params[:beta]
    mu = model_params[:mu]
    U = model_params[:U]
    t_p = model_params[:t_p]
    r_range = model_params[:r_range]
    L = model_params[:L]
    outfile = model_params[:outfile]
    E = 0.0
    psi = nothing
    s = nothing

    for it in 1:max_iter
        mu, n_meas, E, psi, s, H = find_mu_for_target_density(model_params, alpha, beta, mu, n_target;
            dmrg_kw=(nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff))
        if TIME_LIMIT_EXCEEDED[] || peektimer() > TIME_LIMIT_SECONDS
            return alpha, beta, alpha_list, beta_list, mu, psi, E, s, H
        end
        println("Target density achieved with mu=$mu, n=$n_meas")

        alpha_meas, beta_meas = calculate_alpha_beta_measured(psi, s; L=L, r_range=r_range, z_c=z_c, t_p=t_p, E_p=E_p)
        alpha_list == [] ? alpha_list = alpha_meas : alpha_list = cat(alpha_list, alpha_meas, dims=length(size(alpha_meas))+1)
        beta_list == [] ? beta_list = beta_meas : beta_list = cat(beta_list, beta_meas, dims=length(size(beta_meas))+1)

        println("Checking {alpha,beta} vs measured")
        if close_ab(alpha, alpha_meas, beta, beta_meas, r_range; thresh=1e-3)
            println("Converged {alpha,beta}. mu=$mu, n=$n_meas\nExiting loop")
            return alpha, beta, alpha_list, beta_list, mu, psi, E, s, H
        end

        println("NOT CONVERGED; updating alpha, beta and continuing...")
        alpha = damp * alpha_meas + (1 - damp) * alpha
        beta  = damp * beta_meas  + (1 - damp) * beta
        model_params[:alpha] = alpha
        model_params[:beta] = beta
        model_params[:mu] = mu

        # Writing to data file
        F = h5open(outfile, "w")
        F["U"] = U
        F["t_p"] = t_p
        F["alpha"] = alpha
        F["beta"] = beta
        F["mu"] = mu
        F["E"] = E
        F["alpha_list"] = alpha_list
        F["beta_list"] = beta_list
        F["completed"] = false
        close(F)
    end

    @warn "Failed to converge alpha and beta within the maximum number of iterations"
    return alpha, beta, alpha_list, beta_list, mu, psi, E, s, nothing
end

# Convenience: run the full loop and then compute gap and order parameter
function run_loop(L::Int, t::Float64, U::Float64, t0::Float64, t_p::Float64, mu_init::Float64, n_target::Float64,
    r_range::Int, z_c::Int, E_p::Real, chi_max::Int=200; nsweeps=30, cutoff=1e-10)
    
    tick()

    outfile = "results_L_$(L)_U_$(U)_t0_$(t0)_t_p_$(t_p)_chi_$(chi_max)_gpu.h5"
    if (isfile(outfile))
        println("Resuming from checkpoint $outfile")
        F = h5open(outfile,"r")
        alpha = read(F, "alpha")
        beta = read(F, "beta")
        alpha_list = read(F, "alpha_list")
        beta_list = read(F, "beta_list")
        mu_init = read(F, "mu")
        close(F)
        println("Resuming with mu_init=$(mu_init)")
    else
        println("Starting fresh run")
        pref = 2 * t_p^2 / E_p  # Note: no z_c factor for ladders (see Appendix E)
        # Initialize alpha[L, L, 2, 2] with diagonal onsite pairing
        alpha = zeros(Float64, L, L, 2, 2)
        for i in 1:L
            alpha[i, i, 1, 1] = pref  # Onsite pairing leg 0
            alpha[i, i, 2, 2] = pref  # Onsite pairing leg 1
        end
        # Initialize beta[2, L, L, 2, 2] to zeros
        beta = zeros(Float64, 2, L, L, 2, 2)
        alpha_list = Vector{Any}()
        beta_list = Vector{Any}()
    end

    println("Running with t0=$(t0) t_p=$(t_p) U=$U L=$L chi_max=$(chi_max) E_p=$(E_p) mu_init=$(mu_init)")

    model_params = Dict{Symbol,Any}(
        :L => L,
        :t => t,
        :U => U,
        :V => 0.0,
        :t0 => t0,
        :t_p => t_p,
        :mu => mu_init,
        :r_range => r_range,
        :alpha => alpha,
        :beta => beta,
        :density => n_target,
        :outfile => outfile,
    )

    alpha, beta, alpha_list, beta_list, mu, psi, E, s, H = main_loop(model_params; n_target=n_target, E_p=E_p, z_c=z_c, alpha_list=alpha_list, beta_list=beta_list, nsweeps=nsweeps, maxdim=chi_max, cutoff=cutoff)

    if H === nothing
        println("Main loop returned nothing for H (convergence failure). Exiting.")
        return
    end

    if TIME_LIMIT_EXCEEDED[] || peektimer() > TIME_LIMIT_SECONDS
        return
    end


    E1, psi1 = run_dmrg_excited(s, H, psi, n_target; nsweeps=nsweeps, maxdim=chi_max, cutoff=cutoff)
    gap = E1 - E
    order_param = order_parameter(psi, s)
    dwave_op = dwave_order_parameter(psi, s; L_rungs=L)
    cdw_op = cdw_order_parameter(psi, s; L_rungs=L)

    println("Calculation finished. Saving results to $outfile")

    # Writing to data file
    F = h5open(outfile, "w")
    F["U"] = U
    F["t0"] = t0
    F["t_p"] = t_p
    F["alpha"] = alpha
    F["beta"] = beta
    F["mu"] = mu
    F["psi"] = ITensors.cpu(psi)
    F["order_param"] = order_param
    F["dwave_order_param"] = dwave_op
    F["cdw_order_param"] = cdw_op
    F["gap"] = gap
    F["E"] = E
    F["alpha_list"] = alpha_list
    F["beta_list"] = beta_list
    F["completed"] = true
    close(F)
    return
end

# -----------------------------
# CLI entry point
# -----------------------------

if length(ARGS) != 8
    println("Usage: julia main_loop_script_ladder_gpu.jl <L> <U> <t0> <t_p> <chi_max> <E_p> <mu_init> <density>")
    println("  L: number of rungs (total sites = 2L)")
    println("  U: onsite interaction (repulsive, U > 0 for ladders)")
    println("  t0: rung hopping strength")
    println("  t_p: inter-chain hopping")
    println("  chi_max: maximum bond dimension")
    println("  E_p: pair binding energy")
    println("  mu_init: initial chemical potential")
    println("  density: target particle density (e.g. 0.9375)")
    return
end

L = parse(Int, ARGS[1])
U = parse(Float64, ARGS[2])
t0 = parse(Float64, ARGS[3])
t_p = parse(Float64, ARGS[4])
chi_max = parse(Int, ARGS[5])
E_p = parse(Float64, ARGS[6])
mu_init = parse(Float64, ARGS[7])
density = parse(Float64, ARGS[8])

t = 1.0
n_target = density
r_range = 4
z_c = 4  # Still used in calculate_alpha_beta_measured

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(1)

result = run_loop(L, t, U, t0, t_p, mu_init, n_target, r_range, z_c, E_p, chi_max)
