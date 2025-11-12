using ITensors
using ITensorMPS
using HDF5
using LinearAlgebra
using Printf
using Serialization
using TickTock

# Build Electron site indices with Sz conservation only (no particle number conservation)
# Matches the TeNPy choice cons_Sz = "Sz", cons_N = None
function make_sites(L::Int)
    # conserve_sz=true, conserve_nf=false keeps Sz U(1), but not total N
    return siteinds("Electron", L; conserve_sz=true, conserve_nf=false)
end

# Optionally build Electron sites that DO conserve particle number (used for E_p calculation)
function make_sites_conserveN(L::Int)
    # Conserve both Nf and Sz
    return siteinds("Electron", L; conserve_sz=true, conserve_nf=true)
end

# Compute average density
function average_density(psi::MPS)
    return sum(expect(psi, "Ntot")) / length(psi)
end

# Build mean-field-augmented Hubbard MPO
# Parameters:
#  - L, t, U, mu, V (nn density-density), r_range, alpha[L,L], beta[2,L,L], optional phi_ext (Peierls flux per hop)
function build_MPO_MF(; L::Int, t::Float64, U::Float64, mu::Float64, V::Float64,
    r_range::Int, alpha::Array{Float64,2}, beta::Array{Float64,3},
    phi_ext::Union{Nothing,Float64}=nothing)
    s = make_sites(L)
    os = OpSum()

    # Onsite terms: -mu*Ntot + U*Nup*Ndn
    for i in 1:L
        add!(os, -mu, "Ntot", i)
        add!(os, U, "Nupdn", i)
    end

    # Nearest-neighbor hopping (open bc in supplied code)
    for i in 1:(L-1)
        hop = -t
        if phi_ext !== nothing
            # Simple Peierls phase e^{i*phi} for +x hops; use symmetric h.c.
            ϕ = 2π * phi_ext
            hop_fwd = hop * cos(ϕ)
            hop_im = hop * sin(ϕ)
            # Up spin
            add!(os, hop_fwd, "Cdagup", i, "Cup", i + 1)
            add!(os, hop_fwd, "Cdagup", i + 1, "Cup", i)
            add!(os, hop_im, "iCdagup", i, "Cup", i + 1)      # i * Cdagup * Cup
            add!(os, -hop_im, "iCdagup", i + 1, "Cup", i)      # -i * Cdagup * Cup (h.c.)
            # Down spin
            add!(os, hop_fwd, "Cdagdn", i, "Cdn", i + 1)
            add!(os, hop_fwd, "Cdagdn", i + 1, "Cdn", i)
            add!(os, hop_im, "iCdagdn", i, "Cdn", i + 1)
            add!(os, -hop_im, "iCdagdn", i + 1, "Cdn", i)
        else
            # Standard hermitian hopping
            add!(os, hop, "Cdagup", i, "Cup", i + 1)
            add!(os, hop, "Cdagup", i + 1, "Cup", i)
            add!(os, hop, "Cdagdn", i, "Cdn", i + 1)
            add!(os, hop, "Cdagdn", i + 1, "Cdn", i)
        end
    end

    # Optional nearest-neighbor density-density V N_i N_{i+1}
    if V != 0
        for i in 1:(L-1)
            # (Nup+Ndn)_i (Nup+Ndn)_{i+1} = expand
            add!(os, V, "Nup", i, "Nup", i + 1)
            add!(os, V, "Nup", i, "Ndn", i + 1)
            add!(os, V, "Ndn", i, "Nup", i + 1)
            add!(os, V, "Ndn", i, "Ndn", i + 1)
        end
    end

    # Pairing alpha terms
    # Includes onsite i==k as s-wave pairing source term
    for i in 1:L
        # onsite piece
        if alpha[i, i] != 0
            add!(os, -alpha[i, i], "Cup", i, "Cdn", i)
            add!(os, -alpha[i, i], "Cdagdn", i, "Cdagup", i) # h.c.
        end
        # finite range offsite
        for k in (i+1):min(i + r_range, L)
            a = alpha[i, k]
            if a != 0
                add!(os, -alpha[i, k], "Cup", i, "Cdn", k)
                add!(os, -alpha[i, k], "Cdagdn", k, "Cdagup", i) # h.c.
                add!(os, -alpha[k, i], "Cup", k, "Cdn", i)
                add!(os, -alpha[k, i], "Cdagdn", i, "Cdagup", k) # h.c.
            end
        end
    end

    # Normal beta terms: spin-resolved longer-range hoppings up to r_range
    Lbeta1, Lbeta2, Lbeta3 = size(beta)
    @assert Lbeta1 == 2 && Lbeta2 == L && Lbeta3 == L "beta must be of size (2,L,L)"
    for i in 1:L
        for r in 1:r_range
            j = i + r
            j > L && break
            bdn = beta[1, i, j]
            if bdn != 0
                add!(os, bdn, "Cdagdn", i, "Cdn", j)
                add!(os, bdn, "Cdagdn", j, "Cdn", i)
            end
            bup = beta[2, i, j]
            if bup != 0
                add!(os, bup, "Cdagup", i, "Cup", j)
                add!(os, bup, "Cdagup", j, "Cup", i)
            end
        end
    end

    H = MPO(os, s)
    return s, H
end

# DMRG ground state with product-state initialization
function run_dmrg_ground(s, H; nsweeps=10, maxdim=200, cutoff=1e-10)
    L = length(s)
    psi0 = productMPS(s, half_filling_product_state(L))

    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    return energy, psi
end

# First excited state energy via DMRG orthogonalization to ground state
function run_dmrg_excited(s, H; nsweeps=10, maxdim=200, cutoff=1e-10)
    E0, psi0 = run_dmrg_ground(s, H; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    L = length(s)
    psi1 = productMPS(s, half_filling_product_state(L))

    E1, psi1opt = dmrg(H, [psi0], psi1; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    return (E0, psi0), (E1, psi1opt), (E1 - E0)
end

# Solve one Hamiltonian instance and return (density, energy, psi)
function solve_Ham(mu, model_params, alpha, beta; nsweeps::Int=10, maxdim=200, cutoff=1e-10)
    L = model_params[:L]
    t = model_params[:t]
    U = model_params[:U]
    V = get(model_params, :V, 0.0)
    r_rg = model_params[:r_range]
    φ = get(model_params, :phi_ext, nothing)

    s, H = build_MPO_MF(L=L, t=t, U=U, mu=mu, V=V, r_range=r_rg, alpha=alpha, beta=beta, phi_ext=φ)
    E, psi = run_dmrg_ground(s, H; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    n̄ = average_density(psi)
    return n̄, E, psi, s, H
end

# Check convergence of alpha/beta by comparing diagonals up to r_range with relative tolerance
function close_ab(alpha, alpha_meas, beta, beta_meas, r_range; thresh=1e-4)
    eps = 1e-12
    for r in 0:r_range
        # alpha
        a = diag(alpha, r)
        am = diag(alpha_meas, r)
        if any(abs.(am .- a) ./ (abs.(a) .+ eps) .> thresh)
            return false
        end
        # beta down
        b = diag(view(beta, 1, :, :), r)
        bm = diag(view(beta_meas, 1, :, :), r)
        if any(abs.(bm .- b) ./ (abs.(b) .+ eps) .> thresh)
            return false
        end
        # beta up
        b = diag(view(beta, 2, :, :), r)
        bm = diag(view(beta_meas, 2, :, :), r)
        if any(abs.(bm .- b) ./ (abs.(b) .+ eps) .> thresh)
            return false
        end
    end
    return true
end

# Find mu to hit target density using secant with bisection fallback
function find_mu_for_target_density(model_params, alpha, beta, mu_init, n_target;
    tol=1e-5, delta_mu=0.01, max_iter=100, dmrg_kw)
    mu0 = mu_init
    n0, E0, psi0, s0, H0 = solve_Ham(mu0, model_params, alpha, beta; dmrg_kw...)
    
    if abs(n0 - n_target) / max(n_target, 1e-12) <= tol
        println("Same mu: $mu0 and continuing")
        return mu0, n0, E0, psi0, s0, H0
    end
    println("Not same mu, searching again")

    # 0 and 1 are fixed to be left and right. "new" keeps track of latest
    if n0 < n_target
        mu_new = mu0 + delta_mu
        n_new, E_new, psi_new, s_new, H_new = solve_Ham(mu_new, model_params, alpha, beta; dmrg_kw...)
        mu1, n1 = mu_new, n_new
    else
        mu1, n1 = mu0, n0
        mu_new = mu0 - delta_mu
        n_new, E_new, psi_new, s_new, H_new = solve_Ham(mu_new, model_params, alpha, beta; dmrg_kw...)
        mu0, n0 = mu_new, n_new
    end

    for it in 1:max_iter
        if peektimer() > (23.5 * 60 * 60)
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
            mu_new = mu0 + delta_mu
            n_new, E_new, psi_new, s_new, H_new = solve_Ham(mu_new, model_params, alpha, beta; dmrg_kw...)
            mu1, n1 = mu_new, n_new
        elseif (n_target <= n1 && n_target <= n0)
            println("n_target < n1 and n_target < n0 triggered")
            mu1, n1 = mu0, n0
            mu_new = mu0 - delta_mu
            n_new, E_new, psi_new, s_new, H_new = solve_Ham(mu_new, model_params, alpha, beta; dmrg_kw...)
            mu0, n0 = mu_new, n_new
        else
            # refinement loop
            println("\nRange found between mu0: $mu0 (n0: $n0) and mu1: $mu1 (n1: $n1); refining...")
            for j in 1:max_iter
                if peektimer() > (23.5 * 60 * 60)
                    println("Time limit exceeded, exiting loop")
                    return mu_new, n_new, E_new, psi_new, s_new, H_new
                end
                println("n0: $n0; n1: $n1; mu0: $mu0; mu1: $mu1")
                if abs(n1 - n0) > 1e-12
                    println("Trying secant step...")

                    # Update mu that is farther from target
                    if abs(n1 - n_target) > abs(n0 - n_target)
                        mu1 = mu1 - (n1 - n_target) * (mu1 - mu0) / (n1 - n0)
                        n1, E1, psi1, s1, H1 = solve_Ham(mu1, model_params, alpha, beta; dmrg_kw...)
                        mu_new, n_new, E_new, psi_new, s_new, H_new = mu1, n1, E1, psi1, s1, H1
                    else
                        mu0 = mu0 + (n_target - n0) * (mu1 - mu0) / (n1 - n0)
                        n0, E0, psi0, s0, H0 = solve_Ham(mu0, model_params, alpha, beta; dmrg_kw...)
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
                    println("Secant unstable, falling back to bisection...")
                    mu_new = 0.5 * (mu0 + mu1)
                    n_new, E_new, psi_new, s_new, H_new = solve_Ham(mu_new, model_params, alpha, beta; dmrg_kw...)
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

# Measure alpha,beta from correlators per the supplied definitions
function calculate_alpha_beta_measured(psi::MPS, s; L::Int, r_range::Int, z_c::Int, t_p::Float64, E_p::Float64)
    alpha_meas = zeros(Float64, L, L)
    beta_meas = zeros(Float64, 2, L, L)

    pref = 2 * z_c * t_p^2 / E_p

    # alpha from ⟨ Cup_i Cdn_j ⟩
    Calpha = pref .* correlation_matrix(psi, "Cup", "Cdn")
    # beta_down from ⟨ Cdagdn_i Cdn_j ⟩, beta_up from ⟨ Cdagup_i Cup_j ⟩
    Cbetad = pref .* correlation_matrix(psi, "Cdagdn", "Cdn")
    Cbetau = pref .* correlation_matrix(psi, "Cdagup", "Cup")

    for i in 1:L, j in 1:L
        if abs(i - j) <= r_range
            alpha_meas[i, j] = Calpha[i, j]
            beta_meas[1, i, j] = Cbetad[i, j]
            beta_meas[2, i, j] = Cbetau[i, j]
        end
    end

    return alpha_meas, beta_meas
end

# Order parameter: average of diagonal r=0 of ⟨ Cup_i Cdn_{i+r} ⟩, excluding 10-site edges
function order_parameter(psi::MPS, s)
    L = length(s)
    Corr = correlation_matrix(psi, "Cup", "Cdn")
    diag0 = [Corr[i, i] for i in 1:L]
    lo = max(1, 10)
    hi = max(lo, L - 10)
    return abs(sum(diag0[lo:hi]) / max(hi - lo + 1, 1))
end

# Main SCF loop: adjust mu for target density, then update alpha/beta until self-consistent
function main_loop(model_params; n_target::Float64, E_p::Float64, z_c::Int=4, alpha_list, beta_list, max_iter::Int=150, nsweeps=10, maxdim=200, cutoff=1e-10)
    alpha = model_params[:alpha]
    beta = model_params[:beta]
    mu = model_params[:mu]
    t_p = model_params[:t_p]
    r_range = model_params[:r_range]
    L = model_params[:L]
    outfile = model_params[:out_file]
    last_E = 0.0
    last_psi = nothing
    last_s = nothing

    for it in 1:max_iter
        mu, n_meas, E, psi, s, H = find_mu_for_target_density(model_params, alpha, beta, mu, n_target; tol=1e-5,
            dmrg_kw=(nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff))
        if peektimer() > (23.5 * 60 * 60)
            return alpha, beta, mu, n_meas, last_psi, last_E, last_s, H
        end
        println("Target density achieved with mu=$mu, n=$n_meas")

        alpha_meas, beta_meas = calculate_alpha_beta_measured(psi, s; L=L, r_range=r_range, z_c=z_c, t_p=t_p, E_p=E_p)
        alpha_list == [] ? alpha_list = alpha_meas : alpha_list = hcat(alpha_list, alpha_meas)
        beta_list == [] ? beta_list = beta_meas : beta_list = hcat(beta_list, beta_meas)

        println("Checking {alpha,beta} vs measured")
        if close_ab(alpha, alpha_meas, beta, beta_meas, r_range; thresh=1e-4)
            println("Converged {alpha,beta}. mu=$mu, n=$n_meas\nExiting loop")
            last_E, last_psi, last_s = E, psi, s
            model_params[:mu] = mu
            model_params[:alpha] = alpha
            model_params[:beta] = beta
            return alpha, beta, mu, n_meas, last_psi, last_E, last_s, H
        end

        println("NOT CONVERGED; updating alpha, beta and continuing...")
        alpha .= alpha_meas
        beta .= beta_meas
        model_params[:alpha] = alpha
        model_params[:beta] = beta
        model_params[:mu] = mu
        last_E, last_psi, last_s = E, psi, s

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
    return alpha, beta, mu, average_density(last_psi), last_psi, last_E, last_s, nothing
end

# Convenience: run the full loop and then compute gap and order parameter
function run_loop(L::Int, t::Float64, U::Float64, t_p::Float64, mu_init::Float64, n_target::Float64,
    r_range::Int, z_c::Int, E_p::Real, chi_max::Int=200; nsweeps=30, cutoff=1e-10)
    
    tick()

    outfile = "results_U_$(U)_t_p_$(t_p).h5"
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
        pref = 2 * z_c * t_p^2 / E_p
        alpha = pref .* Matrix{Float64}(I, L, L)
        beta = zeros(Float64, 2, L, L)
        alpha_list = Vector{Any}()
        beta_list = Vector{Any}()
    end

    println("Running with t_p=$(t_p) U=$U L=$L chi_max=$(chi_max) E_p=$(E_p) mu_init=$(mu_init)")

    model_params = Dict{Symbol,Any}(
        :L => L,
        :t => t,
        :U => U,
        :V => 0.0,
        :t_p => t_p,
        :mu => mu_init,
        :r_range => r_range,
        :alpha => alpha,
        :beta => beta,
        :out_file => outfile,
    )

    alpha, beta, mu, n̄, psi, E, s, H = main_loop(model_params; n_target=n_target, E_p=E_p, z_c=z_c, alpha_list=alpha_list, beta_list=beta_list, nsweeps=nsweeps, maxdim=chi_max, cutoff=cutoff)

    if peektimer() > (23.5 * 60 * 60)
        println("Time limit exceeded, exiting.")
        return
    end


    (E0, ψ0), (E1, ψ1), gap = run_dmrg_excited(s, H; nsweeps=nsweeps, maxdim=chi_max, cutoff=cutoff)
    order_param = order_parameter(psi, s)

    println("Calculation finished. Saving results to $outfile")

    # Writing to data file
    F = h5open(outfile, "w")
    F["U"] = U
    F["t_p"] = t_p
    F["alpha"] = alpha
    F["beta"] = beta
    F["mu"] = mu
    F["psi"] = psi
    F["order_param"] = order_param
    F["gap"] = gap
    F["E"] = E
    F["alpha_list"] = alpha_list
    F["beta_list"] = beta_list
    F["completed"] = true
    close(F)
    return
end

# -----------------------------
# Pair-binding energy E_p at half-filling n=0.5 (conserving N)
# -----------------------------

# Standard 1D Hubbard MPO (no MF terms), with chemical potential mu and t=1 by default
function build_MPO_Hubbard_standard(; L::Int, t::Float64=1.0, U::Float64=0.0, mu::Float64=0.0)
    s = make_sites_conserveN(L)
    os = OpSum()
    # Onsite
    for i in 1:L
        add!(os, U, "Nupdn", i)
        add!(os, -mu, "Ntot", i)
    end
    # NN hopping (open)
    for i in 1:(L-1)
        add!(os, -t, "Cdagup", i, "Cup", i + 1)
        add!(os, -t, "Cdagup", i + 1, "Cup", i)
        add!(os, -t, "Cdagdn", i, "Cdn", i + 1)
        add!(os, -t, "Cdagdn", i + 1, "Cdn", i)
    end
    H = MPO(os, s)
    return s, H
end

# DMRG with a specific product state (conserving N sector)
function run_dmrg_with_prodstate(s, H, prod_states::Vector{String}; nsweeps, maxdim, cutoff)
    psi0 = productMPS(s, prod_states)
    E, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    return E
end

# Construct a half-filling pattern
function half_filling_product_state(L::Int)
    states = fill("Emp", L)
    for i in 1:L
        if i % 4 == 1
            states[i] = "Up"
        elseif i % 2 == 1 && states[i] == "Emp"
            states[i] = "Dn"
        end
    end
    return states
end

# Pair binding energy: E_pair = 2 E(N+1,S=1/2) - E(N,S=0) - E(N+2,S=0)
function calculate_pair_binding_energy(L::Int, U::Float64; t::Float64=1.0)
    nsweeps = 20
    maxdim = 200
    cutoff = 1e-10

    s, H = build_MPO_Hubbard_standard(L=L, t=t, U=U, mu=0.0)

    # Base product state at nominal half-filling (approximate sector selection via initial state)
    base0 = half_filling_product_state(L)

    # E(N)
    EN = run_dmrg_with_prodstate(s, H, copy(base0); nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)

    # E(N+1): flip one Emp -> Up (if available)
    baseN1 = copy(base0)
    for i in 1:L
        if baseN1[i] == "Emp"
            baseN1[i] = "Up"
            break
        end
    end
    ENp1 = run_dmrg_with_prodstate(s, H, baseN1; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)

    # E(N+2): add one more Down similarly
    baseN2 = copy(baseN1)
    for i in 1:L
        if baseN2[i] == "Emp"
            baseN2[i] = "Dn"
            break
        end
    end
    ENp2 = run_dmrg_with_prodstate(s, H, baseN2; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)

    pairing_energy = 2 * ENp1 - EN - ENp2
    return pairing_energy
end

# -----------------------------
# CLI entry point
# -----------------------------

if length(ARGS) != 6
    println("Usage: julia main_loop_script.jl <L> <U> <t_p> <chi_max> <E_p> <mu_init>")
    return
end

L = parse(Int, ARGS[1])
U = parse(Float64, ARGS[2])
t_p = parse(Float64, ARGS[3])
chi_max = parse(Int, ARGS[4])
E_p = parse(Float64, ARGS[5])
mu_init = parse(Float64, ARGS[6])

t = 1.0
n_target = 0.5
r_range = 4
z_c = 4

ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(1)

result = run_loop(L, t, U, t_p, mu_init, n_target, r_range, z_c, E_p, chi_max)
