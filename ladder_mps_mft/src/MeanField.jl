function make_sites(model::ModelSettings)
    return siteinds("Electron", 2 * model.L; conserve_sz=true, conserve_nfparity=true)
end

function initial_fields(model::ModelSettings; seed::Symbol=:pairing, amplitude::Real=1e-3, rng=MersenneTwister(1))
    alpha = zeros(Float64, model.L, model.L, 2, 2)
    beta = zeros(Float64, 2, model.L, model.L, 2, 2)
    mu_cdw = zeros(Float64, 2, 2 * model.L)
    if seed == :pairing
        for rung in 1:model.L, offset in 0:model.r_range
            target = rung + offset
            target <= model.L || continue
            for leg in 1:2, other_leg in 1:2
                offset == 0 && other_leg < leg && continue
                value = amplitude * (2rand(rng) - 1)
                alpha[rung, target, leg, other_leg] = value
                alpha[target, rung, other_leg, leg] = value
            end
        end
    elseif seed == :sdw
        for site in 1:(2 * model.L)
            phase = isodd(site) ? 1.0 : -1.0
            mu_cdw[1, site] = amplitude * phase
            mu_cdw[2, site] = -amplitude * phase
        end
    elseif seed == :cdw
        for site in 1:(2 * model.L)
            phase = isodd(div(site - 1, 2)) ? 1.0 : -1.0
            mu_cdw[:, site] .= amplitude * phase
        end
    elseif seed != :zero
        throw(ArgumentError("unknown seed '$seed'"))
    end
    return FieldState(alpha, beta, mu_cdw)
end

function configure_threading!(runtime::RuntimeSettings)
    BLAS.set_num_threads(runtime.blas_threads)
    try
        ITensors.NDTensors.Strided.set_num_threads(runtime.strided_threads)
    catch err
        @warn "could not set ITensor Strided threads" exception=(err, catch_backtrace())
    end
    blocksparse = try
        runtime.threaded_blocksparse ? ITensors.enable_threaded_blocksparse() : ITensors.disable_threaded_blocksparse()
        ITensors.using_threaded_blocksparse()
    catch err
        @warn "could not configure ITensor block-sparse threading" exception=(err, catch_backtrace())
        false
    end
    strided = try
        ITensors.NDTensors.Strided.get_num_threads()
    catch
        runtime.strided_threads
    end
    return (; julia=Threads.nthreads(), blas=BLAS.get_num_threads(), strided, blocksparse)
end

function build_mf_mpo(sites, model::ModelSettings, fields::FieldState, chemical_potential::Real)
    os = OpSum()
    for site in 1:(2 * model.L)
        add!(os, -chemical_potential, "Ntot", site)
        add!(os, fields.mu_cdw[1, site], "Ndn", site)
        add!(os, fields.mu_cdw[2, site], "Nup", site)
        add!(os, model.U, "Nupdn", site)
    end
    for rung in 1:(model.L - 1), leg in 0:1
        left = rung_leg_to_site(rung, leg)
        right = rung_leg_to_site(rung + 1, leg)
        for spin in ("up", "dn")
            add!(os, -model.t, "Cdag$spin", left, "C$spin", right)
            add!(os, -model.t, "Cdag$spin", right, "C$spin", left)
        end
    end
    for rung in 1:model.L
        left = rung_leg_to_site(rung, 0)
        right = rung_leg_to_site(rung, 1)
        for spin in ("up", "dn")
            add!(os, -model.t0, "Cdag$spin", left, "C$spin", right)
            add!(os, -model.t0, "Cdag$spin", right, "C$spin", left)
        end
    end
    if model.V != 0
        for rung in 1:(model.L - 1), leg in 0:1
            left = rung_leg_to_site(rung, leg)
            right = rung_leg_to_site(rung + 1, leg)
            add!(os, model.V, "Ntot", left, "Ntot", right)
        end
        for rung in 1:model.L
            add!(os, model.V, "Ntot", rung_leg_to_site(rung, 0), "Ntot", rung_leg_to_site(rung, 1))
        end
    end
    for i in 1:model.L, ip in 1:model.L
        abs(i - ip) <= model.r_range || continue
        for leg in 0:1, other_leg in 0:1
            site_i = rung_leg_to_site(i, leg)
            site_ip = rung_leg_to_site(ip, other_leg)
            alpha = fields.alpha[i, ip, leg + 1, other_leg + 1]
            if alpha != 0
                add!(os, -alpha, "Cup", site_ip, "Cdn", site_i)
                add!(os, -alpha, "Cdagdn", site_i, "Cdagup", site_ip)
            end
            site_i == site_ip && continue
            beta_down = fields.beta[1, i, ip, leg + 1, other_leg + 1]
            beta_up = fields.beta[2, i, ip, leg + 1, other_leg + 1]
            beta_down != 0 && add!(os, beta_down, "Cdagdn", site_i, "Cdn", site_ip)
            beta_up != 0 && add!(os, beta_up, "Cdagup", site_i, "Cup", site_ip)
        end
    end
    return MPO(os, sites)
end

function _threshold(value::Real, threshold::Real)
    return abs(value) > threshold ? Float64(value) : 0.0
end

function calculate_mean_fields(psi::MPS, model::ModelSettings; threshold::Real=0.0)
    pair = real.(correlation_matrix(psi, "Cup", "Cdn"))
    exchange_down = real.(correlation_matrix(psi, "Cdagdn", "Cdn"))
    exchange_up = real.(correlation_matrix(psi, "Cdagup", "Cup"))
    density_down = real.(expect(psi, "Ndn"))
    density_up = real.(expect(psi, "Nup"))
    correlations = CorrelationState(pair, exchange_down, exchange_up, density_down, density_up)
    prefactor = 2 * model.tp^2 / model.ep
    alpha = zeros(Float64, model.L, model.L, 2, 2)
    beta = zeros(Float64, 2, model.L, model.L, 2, 2)
    for i in 1:model.L, ip in 1:model.L
        abs(i - ip) <= model.r_range || continue
        i0, i1 = rung_leg_to_site(i, 0), rung_leg_to_site(i, 1)
        ip0, ip1 = rung_leg_to_site(ip, 0), rung_leg_to_site(ip, 1)
        if model.geometry == :cubic_frustrated
            avals = (
                prefactor * (pair[ip1, i1] + 2 * pair[ip0, i0]),
                prefactor * (pair[ip0, i0] + 2 * pair[ip1, i1]),
                2 * prefactor * pair[ip0, i1],
                2 * prefactor * pair[ip1, i0],
            )
            bvals_down = (
                prefactor * (exchange_down[i1, ip1] + 2 * exchange_down[i0, ip0]),
                prefactor * (exchange_down[i0, ip0] + 2 * exchange_down[i1, ip1]),
                2 * prefactor * exchange_down[ip0, i1],
                2 * prefactor * exchange_down[ip1, i0],
            )
            bvals_up = (
                prefactor * (exchange_up[i1, ip1] + 2 * exchange_up[i0, ip0]),
                prefactor * (exchange_up[i0, ip0] + 2 * exchange_up[i1, ip1]),
                2 * prefactor * exchange_up[ip0, i1],
                2 * prefactor * exchange_up[ip1, i0],
            )
        elseif model.geometry == :cubic_unfrustrated
            avals = (3 * prefactor * pair[ip1, i1], 3 * prefactor * pair[ip0, i0], 2 * prefactor * pair[ip1, i0], 2 * prefactor * pair[ip0, i1])
            bvals_down = (3 * prefactor * exchange_down[i1, ip1], 3 * prefactor * exchange_down[i0, ip0], 2 * prefactor * exchange_down[ip1, i0], 2 * prefactor * exchange_down[ip0, i1])
            bvals_up = (3 * prefactor * exchange_up[i1, ip1], 3 * prefactor * exchange_up[i0, ip0], 2 * prefactor * exchange_up[ip1, i0], 2 * prefactor * exchange_up[ip0, i1])
        else
            avals = (prefactor * pair[ip1, i1], prefactor * pair[ip0, i0], 0.0, 0.0)
            bvals_down = (prefactor * exchange_down[i1, ip1], prefactor * exchange_down[i0, ip0], 0.0, 0.0)
            bvals_up = (prefactor * exchange_up[i1, ip1], prefactor * exchange_up[i0, ip0], 0.0, 0.0)
        end
        for (index, (leg, other_leg)) in enumerate(((1, 1), (2, 2), (2, 1), (1, 2)))
            alpha[i, ip, leg, other_leg] = _threshold(avals[index], threshold)
            beta[1, i, ip, leg, other_leg] = _threshold(bvals_down[index], threshold)
            beta[2, i, ip, leg, other_leg] = _threshold(bvals_up[index], threshold)
            if rung_leg_to_site(i, leg - 1) == rung_leg_to_site(ip, other_leg - 1)
                beta[:, i, ip, leg, other_leg] .= 0.0
            end
        end
    end
    mu_cdw = zeros(Float64, 2, 2 * model.L)
    kernel = density_kernel(model.geometry, model.tp, model.ep)
    for rung in 1:model.L
        sites = [rung_leg_to_site(rung, 0), rung_leg_to_site(rung, 1)]
        mu_cdw[1, sites] .= kernel * (density_down[sites] .- 0.5)
        mu_cdw[2, sites] .= kernel * (density_up[sites] .- 0.5)
    end
    mu_cdw[abs.(mu_cdw) .<= threshold] .= 0.0
    return FieldState(alpha, beta, mu_cdw), correlations
end
