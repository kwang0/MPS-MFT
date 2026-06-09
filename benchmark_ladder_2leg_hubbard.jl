using ITensors
using ITensorMPS
using HDF5
using LinearAlgebra
using Printf
using Statistics

# Note: U(1) is conserved here
# to make a meaningful comparison between the 2-leg and MFT versions, we need to track pair-pair correlators in both
# Usage:  JULIA_NUM_THREADS=8 julia benchmark_ladder_2leg_hubbard.jl 16 8.0 0.0 1.2 0.9375 300 10 1e-7 smoke_test_L16.h5

function rung_leg_to_site(r::Int, l::Int)
    return 2 * (r - 1) + l + 1
end

site(r::Int, l::Int) = rung_leg_to_site(r, l)

function site_to_rung_leg(s::Int)
    r = div(s - 1, 2) + 1
    l = (s - 1) % 2
    return r, l
end

rungleg(i::Int) = site_to_rung_leg(i)

function make_sites_conserveN(L::Int)
    return siteinds("Electron", L; conserve_sz=true, conserve_nf=true)
end

# Construct a product state with exactly N, Nup, and Ndn particles
function product_state(L::Int, n::Float64)
    M = 2L
    N = round(Int, n * M)
    0 <= N <= 2M || error("Invalid filling: need 0 <= round(n*2L) <= 4L")
    Nup, Ndn = cld(N, 2), fld(N, 2)
    up_left, dn_left = Nup, Ndn
    states = fill("Emp", M)

    staggered_sites = [site(r, isodd(r) ? l : 1 - l) for r in 1:L for l in 0:1]
    extra = max(N - M, 0)
    for i in staggered_sites[1:extra]
        states[i] = "UpDn"
        up_left -= 1
        dn_left -= 1
    end

    singles = N - 2extra
    occ = staggered_sites[(extra + 1):(extra + singles)]
    for i in occ
        r, l = rungleg(i)
        use_up = (isodd(r + l) && up_left > 0) || dn_left == 0
        states[i] = use_up ? "Up" : "Dn"
        use_up ? (up_left -= 1) : (dn_left -= 1)
    end

    return states, N, Nup, Ndn
end

# Standard two-leg Hubbard ladder MPO
function build_MPO_Hubbard_ladder(s; L::Int, U::Float64, V::Float64, t0::Float64, t::Float64=1.0)
    os = OpSum()

    # Onsite Hubbard interaction
    for i in 1:2L
        add!(os, U, "Nupdn", i)
    end

    # Nearest-neighbor hopping and V along legs
    for r in 1:(L - 1)
        for l in 0:1
            site_i = site(r, l)
            site_ip = site(r + 1, l)
            add!(os, -t, "Cdagup", site_i, "Cup", site_ip)
            add!(os, -t, "Cdagup", site_ip, "Cup", site_i)
            add!(os, -t, "Cdagdn", site_i, "Cdn", site_ip)
            add!(os, -t, "Cdagdn", site_ip, "Cdn", site_i)

            if V != 0
                add!(os, V, "Nup", site_i, "Nup", site_ip)
                add!(os, V, "Nup", site_i, "Ndn", site_ip)
                add!(os, V, "Ndn", site_i, "Nup", site_ip)
                add!(os, V, "Ndn", site_i, "Ndn", site_ip)
            end
        end
    end

    # Rung hopping and rung V
    for r in 1:L
        site0 = site(r, 0)
        site1 = site(r, 1)
        add!(os, -t0, "Cdagup", site0, "Cup", site1)
        add!(os, -t0, "Cdagup", site1, "Cup", site0)
        add!(os, -t0, "Cdagdn", site0, "Cdn", site1)
        add!(os, -t0, "Cdagdn", site1, "Cdn", site0)

        if V != 0
            add!(os, V, "Nup", site0, "Nup", site1)
            add!(os, V, "Nup", site0, "Ndn", site1)
            add!(os, V, "Ndn", site0, "Nup", site1)
            add!(os, V, "Ndn", site0, "Ndn", site1)
        end
    end

    return MPO(os, s)
end

function make_sweeps(nsweeps::Int, chi::Int, cutoff::Float64)
    sw = Sweeps(nsweeps)
    schedule = vcat([20, 50, 100, 200], fill(chi, max(nsweeps - 4, 0)))[1:nsweeps]
    maxdim!(sw, [min(chi, d) for d in schedule]...)
    cutoff!(sw, cutoff)
    noise!(sw, [swp <= 4 ? 10.0^(-swp - 3) : 0.0 for swp in 1:nsweeps]...)
    return sw
end

function run_dmrg_ground(s, H, prod_states::Vector{String}; nsweeps, maxdim, cutoff)
    psi0 = productMPS(s, prod_states)
    E, psi = dmrg(H, psi0, make_sweeps(nsweeps, maxdim, cutoff); outputlevel=1)
    return E, psi
end

function cdw_pi(psi; L::Int)
    n = expect(psi, "Ntot")
    return abs(sum(n[site(r, l)] * (-1)^(r - 1) for r in 1:L, l in 0:1) / 2L)
end

function pair_terms(kind::Symbol, r::Int; dagger::Bool=false)
    terms = Tuple{Float64,Vector{Tuple{String,Int}}}[]
    if kind == :rung
        a, b = site(r, 0), site(r, 1)
        if dagger
            push!(terms, ( 1.0, [("Cdagup", a), ("Cdagdn", b)]))
            push!(terms, (-1.0, [("Cdagdn", a), ("Cdagup", b)]))
        else
            push!(terms, (-1.0, [("Cup", a), ("Cdn", b)]))
            push!(terms, ( 1.0, [("Cdn", a), ("Cup", b)]))
        end
    elseif kind == :leg0 || kind == :leg1
        l = kind == :leg0 ? 0 : 1
        a, b = site(r, l), site(r + 1, l)
        if dagger
            push!(terms, ( 0.5, [("Cdagup", a), ("Cdagdn", b)]))
            push!(terms, (-0.5, [("Cdagdn", a), ("Cdagup", b)]))
        else
            push!(terms, (-0.5, [("Cup", a), ("Cdn", b)]))
            push!(terms, ( 0.5, [("Cdn", a), ("Cup", b)]))
        end
    elseif kind == :dwave
        append!(terms, pair_terms(:leg0, r; dagger=dagger))
        append!(terms, pair_terms(:leg1, r; dagger=dagger))
        append!(terms, [(-c, ops) for (c, ops) in pair_terms(:rung, r; dagger=dagger)])
    else
        error("unknown pair field kind $kind")
    end
    return terms
end

function pair_mpo(s, i::Int, j::Int; kind::Symbol=:rung)
    os = OpSum()
    for (ca, opa) in pair_terms(kind, i; dagger=false), (cd, opd) in pair_terms(kind, j; dagger=true)
        opsites = Any[]
        for (op, site_index) in vcat(opa, opd)
            push!(opsites, op, site_index)
        end
        add!(os, ca * cd, opsites...)
    end
    return MPO(os, s)
end

pair_max_r(L::Int, kind::Symbol) = kind == :dwave ? L - 1 : L
all_starts(L::Int, ell::Int; kind::Symbol=:rung) = collect(1:(pair_max_r(L, kind) - ell))

function centered_starts(L::Int, ell::Int; kind::Symbol=:rung, npairs::Int=6)
    starts = all_starts(L, ell; kind=kind)
    c = (L + 1) / 2
    sort!(starts; by=i -> abs(i + ell / 2 - c))
    return sort(starts[1:min(npairs, length(starts))])
end

function D_of_ell(psi, s; L::Int, ell::Int, kind::Symbol=:rung, starts=all_starts(L, ell; kind=kind))
    vals = [real(inner(psi', pair_mpo(s, i, i + ell; kind=kind), psi)) for i in starts]
    return mean(vals), vals
end

function Dbar_rung(psi, s; L::Int)
    D1, D1_vals = D_of_ell(psi, s; L=L, ell=1, starts=centered_starts(L, 1))
    ells = collect(8:min(12, L - 1))
    if isempty(ells)
        return NaN, ells, Float64[], Float64[], D1, D1_vals
    end
    Dvals = Float64[first(D_of_ell(psi, s; L=L, ell=ell, starts=centered_starts(L, ell))) for ell in ells]
    if abs(D1) <= eps(Float64)
        ratios = fill(Float64(NaN), length(Dvals))
        return NaN, ells, Dvals, ratios, D1, D1_vals
    end
    ratios = Dvals ./ D1
    return mean(ratios), ells, Dvals, ratios, D1, D1_vals
end

function pair_expectation(psi, s, kind::Symbol, r::Int)
    os = OpSum()
    for (c, ops) in pair_terms(kind, r; dagger=false)
        opsites = Any[]
        for (op, site_index) in ops
            push!(opsites, op, site_index)
        end
        add!(os, c, opsites...)
    end
    return real(inner(psi', MPO(os, s), psi))
end

function fixedN_dwave_order_param(psi, s; L::Int)
    # Since U(1) conserved, this should theoretically be close to 0
    val = 0.0
    for r in 1:(L - 1)
        val += pair_expectation(psi, s, :dwave, r)
    end
    return val / (L - 1)
end

# Not relevant right now, but asked codex to generate these based on the MFT script; needs to be checked
function structure_factors(psi; L::Int)
    nup = expect(psi, "Nup")
    ndn = expect(psi, "Ndn")
    density = nup .+ ndn
    Cuu = correlation_matrix(psi, "Nup", "Nup")
    Cud = correlation_matrix(psi, "Nup", "Ndn")
    Cdu = correlation_matrix(psi, "Ndn", "Nup")
    Cdd = correlation_matrix(psi, "Ndn", "Ndn")
    Nr = [sum(density[site(r, l)] for l in 0:1) for r in 1:L]
    CNN = [sum(real(Cuu[site(r, l), site(rp, lp)] + Cud[site(r, l), site(rp, lp)] +
                    Cdu[site(r, l), site(rp, lp)] + Cdd[site(r, l), site(rp, lp)])
               for l in 0:1, lp in 0:1) for r in 1:L, rp in 1:L]
    CMM = [sum(real(Cuu[site(r, l), site(rp, lp)] - Cud[site(r, l), site(rp, lp)] -
                    Cdu[site(r, l), site(rp, lp)] + Cdd[site(r, l), site(rp, lp)])
               for l in 0:1, lp in 0:1) for r in 1:L, rp in 1:L]
    q = [2π * m / L for m in 0:(L - 1)]
    cdw = [real(sum(exp(im * qq * (r - rp)) * (CNN[r, rp] - Nr[r] * Nr[rp])
                    for r in 1:L, rp in 1:L) / L) for qq in q]
    sdw = [real(sum(exp(im * qq * (r - rp)) * CMM[r, rp]
                    for r in 1:L, rp in 1:L) / L) for qq in q]
    cdw_peak_i = length(cdw) > 1 ? argmax(cdw[2:end]) + 1 : 1
    sdw_peak_i = length(sdw) > 1 ? argmax(sdw[2:end]) + 1 : 1
    return q, cdw, sdw, cdw[cdw_peak_i], q[cdw_peak_i], sdw[sdw_peak_i], q[sdw_peak_i]
end

function parse_args(args)
    L       = length(args) >= 1 ? parse(Int, args[1]) : 64
    U       = length(args) >= 2 ? parse(Float64, args[2]) : 8.0
    V       = length(args) >= 3 ? parse(Float64, args[3]) : 0.0
    t0      = length(args) >= 4 ? parse(Float64, args[4]) : 1.4
    n       = length(args) >= 5 ? parse(Float64, args[5]) : 15 / 16
    chi     = length(args) >= 6 ? parse(Int, args[6]) : 1000
    nsweeps = length(args) >= 7 ? parse(Int, args[7]) : 80
    cutoff  = length(args) >= 8 ? parse(Float64, args[8]) : 1e-10
    outfile = length(args) >= 9 ? args[9] :
        joinpath("2leg_benchmark", "benchmark_ladder_L_$(L)_U_$(U)_V_$(V)_t0_$(t0)_chi_$(chi)_density_$(n).h5")
    length(args) <= 9 || error("Usage: julia benchmark_ladder_2leg_hubbard.jl [L U V t0 n chi nsweeps cutoff outfile]")
    L >= 2 && chi > 0 && nsweeps > 0 && cutoff > 0 || error("Need L>=2, chi>0, nsweeps>0, cutoff>0")
    return L, U, V, t0, n, chi, nsweeps, cutoff, outfile
end

function main(args=ARGS)
    L, U, V, t0, n, chi, nsweeps, cutoff, outfile = parse_args(args)
    states, N, Nup, Ndn = product_state(L, n)
    s = make_sites_conserveN(2L)
    H = build_MPO_Hubbard_ladder(s; L=L, U=U, V=V, t0=t0)

    BLAS.set_num_threads(1)
    ITensors.Strided.set_num_threads(1)
    Threads.nthreads() > 1 && ITensors.enable_threaded_blocksparse()

    println("Running with t0=$(t0) U=$U V=$V L=$L chi_max=$(chi) density=$(n)")
    println("N=$N Nup=$Nup Ndn=$Ndn nsweeps=$nsweeps cutoff=$cutoff")
    E, psi = run_dmrg_ground(s, H, states; nsweeps=nsweeps, maxdim=chi, cutoff=cutoff)

    Dbar_rung_value, _, _, _, D_rung_1, _ = Dbar_rung(psi, s; L=L)
    Delta_d_global_fixedN = fixedN_dwave_order_param(psi, s; L=L)
    density = expect(psi, "Ntot")
    n_meas = mean(density)
    cdw = cdw_pi(psi; L=L)
    _, _, _, cdw_peak, cdw_peak_q, sdw_peak, sdw_peak_q = structure_factors(psi; L=L)

    @printf("E = %.16g\n", E)
    @printf("n = %.16g\n", n_meas)
    @printf("Dbar_rung = %.16g\n", Dbar_rung_value)
    @printf("D_rung(1) = %.16g\n", D_rung_1)
    @printf("CDW(pi) = %.16g\n", cdw)
    @printf("CDW peak = %.16g at q = %.16g\n", cdw_peak, cdw_peak_q)
    @printf("SDW peak = %.16g at q = %.16g\n", sdw_peak, sdw_peak_q)

    outdir = dirname(outfile)
    if !isempty(outdir) && outdir != "."
        mkpath(outdir)
    end
    h5open(outfile, "w") do F
        F["L"] = L
        F["t"] = 1.0
        F["U"] = U
        F["V"] = V
        F["t0"] = t0
        F["density"] = n
        F["N_particles"] = N
        F["Nup"] = Nup
        F["Ndn"] = Ndn
        F["chi_max"] = chi
        F["nsweeps"] = nsweeps
        F["cutoff"] = cutoff
        F["E"] = E
        F["E_per_site"] = E / 2L
        F["num_sites"] = 2L
        F["measured_density"] = n_meas
        F["cdw_staggered_amplitude"] = cdw
        F["global_dwave_order_param_fixedN"] = Delta_d_global_fixedN
        F["Dbar_rung"] = Dbar_rung_value
        F["D_rung_1"] = D_rung_1
        F["cdw_structure_peak_value"] = cdw_peak
        F["cdw_structure_peak_q"] = cdw_peak_q
        F["sdw_structure_peak_value"] = sdw_peak
        F["sdw_structure_peak_q"] = sdw_peak_q
        F["completed"] = true
        write(F, "psi", psi)
    end
    println("saved $outfile")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
