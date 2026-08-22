function connected_structure_factor(
    correlation::AbstractMatrix,
    expectation::AbstractVector,
    L::Integer;
    qx::Real,
    ky::Real,
)
    sites = 2 * Int(L)
    size(correlation) == (sites, sites) || throw(DimensionMismatch("correlation matrix must be 2L by 2L"))
    length(expectation) == sites || throw(DimensionMismatch("expectation vector must have length 2L"))
    total = 0.0 + 0.0im
    for left in 1:sites, right in 1:sites
        rung_left, leg_left = site_to_rung_leg(left)
        rung_right, leg_right = site_to_rung_leg(right)
        connected = correlation[left, right] - expectation[left] * expectation[right]
        phase = exp(im * (qx * (rung_left - rung_right) + ky * (leg_left - leg_right)))
        total += connected * phase
    end
    return real(total) / sites
end

function structure_factor_grid(correlation, expectation, L::Integer; ky_values=(0.0, Float64(pi)))
    qx_values = [2pi * m / L for m in 0:(L - 1)]
    values = [connected_structure_factor(correlation, expectation, L; qx, ky) for qx in qx_values, ky in ky_values]
    return (; qx=qx_values, ky=collect(ky_values), values)
end

function dominant_wavevector(grid)
    index = argmax(grid.values)
    qx_index, ky_index = Tuple(index)
    return (
        qx=grid.qx[qx_index],
        ky=grid.ky[ky_index],
        value=grid.values[index],
        qx_over_pi=grid.qx[qx_index] / pi,
        ky_over_pi=grid.ky[ky_index] / pi,
    )
end

_circular_momentum_distance(left::Real, right::Real) = abs(mod(left - right + pi, 2 * pi) - pi)

function estimate_k_rho(charge_grid; maximum_modes::Int=3)
    positive = [(charge_grid.qx[index], charge_grid.values[index, 1]) for index in 2:min(length(charge_grid.qx), maximum_modes + 1)]
    isempty(positive) && return (
        K_rho_site_normalized=NaN,
        K_rho_rung_normalized=NaN,
        slope=NaN,
        modes=0,
        convention="stored S uses 1/(2L); site K=pi*dS/dq, rung-total K=2pi*dS/dq",
    )
    q = first.(positive)
    s = last.(positive)
    slope = dot(q, s) / dot(q, q)
    return (
        K_rho_site_normalized=pi * slope,
        K_rho_rung_normalized=2 * pi * slope,
        slope,
        modes=length(q),
        convention="stored S uses 1/(2L); site K=pi*dS/dq, rung-total K=2pi*dS/dq at ky=0",
    )
end

function bond_entropy(psi::MPS, bond::Integer)
    1 <= bond < length(psi) || throw(ArgumentError("bond must lie between 1 and length(psi)-1"))
    state = copy(psi)
    orthogonalize!(state, bond)
    left = bond == 1 ? (siteind(state, bond),) : (linkind(state, bond - 1), siteind(state, bond))
    _, singular, _ = svd(state[bond], left)
    probabilities = Float64[]
    for index in 1:dim(singular, 1)
        value = singular[index, index]^2
        value > eps(Float64) && push!(probabilities, real(value))
    end
    normalization = sum(probabilities)
    normalization > 0 && (probabilities ./= normalization)
    entropy = -sum(p * log(p) for p in probabilities if p > 0)
    renyi2 = -log(sum(abs2, probabilities))
    return (; entropy, renyi2, probabilities)
end

function entanglement_profile(psi::MPS)
    values = [bond_entropy(psi, bond) for bond in 1:(length(psi) - 1)]
    return (
        bonds=collect(1:(length(psi) - 1)),
        entropy=[value.entropy for value in values],
        renyi2=[value.renyi2 for value in values],
    )
end

function estimate_central_charge(profile; edge_fraction::Real=0.2, unit_cell::Int=2)
    sites = length(profile.bonds) + 1
    sites % unit_cell == 0 || throw(ArgumentError("MPS length must be divisible by unit_cell"))
    length_units = div(sites, unit_cell)
    first_unit = max(1, ceil(Int, edge_fraction * length_units))
    last_unit = min(length_units - 1, floor(Int, (1 - edge_fraction) * length_units))
    selected = findall(profile.bonds) do bond
        bond % unit_cell == 0 && first_unit <= div(bond, unit_cell) <= last_unit
    end
    length(selected) >= 3 || return (
        central_charge=NaN,
        intercept=NaN,
        r2=NaN,
        points=length(selected),
        unit_cell,
        convention="OBC Calabrese-Cardy fit on inter-rung cuts; insufficient fit points",
    )
    x = [
        log((2 * length_units / pi) * sin(pi * div(profile.bonds[index], unit_cell) / length_units))
        for index in selected
    ]
    y = profile.entropy[selected]
    design = hcat(ones(length(x)), x)
    coefficients = design \ y
    prediction = design * coefficients
    ss_res = sum(abs2, y .- prediction)
    ss_tot = sum(abs2, y .- mean(y))
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : 1.0
    return (
        central_charge=6 * coefficients[2],
        intercept=coefficients[1],
        r2,
        points=length(selected),
        unit_cell,
        convention="OBC Calabrese-Cardy fit on inter-rung cuts: S(l)=(c/6)log[(2L/pi)sin(pi*l/L)]+s0",
    )
end

function singlet_pair_mpo(sites, annihilation_left::Int, annihilation_right::Int, creation_left::Int, creation_right::Int)
    os = OpSum()
    add!(os, -1.0, "Cup", annihilation_left, "Cdn", annihilation_right, "Cdagup", creation_left, "Cdagdn", creation_right)
    add!(os, 1.0, "Cup", annihilation_left, "Cdn", annihilation_right, "Cdagdn", creation_left, "Cdagup", creation_right)
    add!(os, 1.0, "Cdn", annihilation_left, "Cup", annihilation_right, "Cdagup", creation_left, "Cdagdn", creation_right)
    add!(os, -1.0, "Cdn", annihilation_left, "Cup", annihilation_right, "Cdagdn", creation_left, "Cdagup", creation_right)
    return MPO(os, sites)
end

function sign_resolved_pair_correlations(psi::MPS, model::ModelSettings)
    sites = siteinds(psi)
    rung_bonds = [(rung_leg_to_site(rung, 0), rung_leg_to_site(rung, 1)) for rung in 1:model.L]
    leg0_bonds = [(rung_leg_to_site(rung, 0), rung_leg_to_site(rung + 1, 0)) for rung in 1:(model.L - 1)]
    leg1_bonds = [(rung_leg_to_site(rung, 1), rung_leg_to_site(rung + 1, 1)) for rung in 1:(model.L - 1)]
    function matrix_for(bonds)
        return [
            real(inner(psi', singlet_pair_mpo(sites, a, b, c, d), psi))
            for (a, b) in bonds, (c, d) in bonds
        ]
    end
    return (
        rung=matrix_for(rung_bonds),
        leg0=matrix_for(leg0_bonds),
        leg1=matrix_for(leg1_bonds),
        convention="unnormalized singlet Delta_ab = c_up,a c_dn,b - c_dn,a c_up,b",
    )
end

function compute_ladder_diagnostics(psi::MPS, model::ModelSettings; full_pair_correlations::Bool=false)
    density = real.(expect(psi, "Ntot"))
    spin = real.(expect(psi, "Sz"))
    charge_correlation = real.(correlation_matrix(psi, "Ntot", "Ntot"))
    spin_correlation = real.(correlation_matrix(psi, "Sz", "Sz"))
    charge_grid = structure_factor_grid(charge_correlation, density, model.L)
    spin_grid = structure_factor_grid(spin_correlation, spin, model.L)
    entropy = entanglement_profile(psi)
    pair = full_pair_correlations ? sign_resolved_pair_correlations(psi, model) : nothing
    expected_spin_q = pi * model.density
    spin_peak = dominant_wavevector(spin_grid)
    return (
        density,
        spin,
        charge_structure=charge_grid,
        spin_structure=spin_grid,
        charge_peak=dominant_wavevector(charge_grid),
        spin_peak,
        expected_spin_q,
        spin_q_mismatch=_circular_momentum_distance(spin_peak.qx, expected_spin_q),
        K_rho=estimate_k_rho(charge_grid),
        entanglement=entropy,
        central_charge=estimate_central_charge(entropy; unit_cell=2),
        pair_correlations=pair,
    )
end

function fixed_sector_product_state(sites, particle_number::Integer, twice_sz::Integer; rng=MersenneTwister(1))
    iseven(particle_number + twice_sz) || throw(ArgumentError("N + 2Sz must be even"))
    n_up = div(particle_number + twice_sz, 2)
    n_down = particle_number - n_up
    0 <= n_up <= length(sites) || throw(ArgumentError("invalid up-spin count"))
    0 <= n_down <= length(sites) || throw(ArgumentError("invalid down-spin count"))
    n_up + n_down <= 2 * length(sites) || throw(ArgumentError("too many particles"))
    doublons = max(0, particle_number - length(sites))
    states = vcat(
        fill("UpDn", doublons),
        fill("Up", n_up - doublons),
        fill("Dn", n_down - doublons),
        fill("Emp", length(sites) - particle_number + doublons),
    )
    shuffle!(rng, states)
    return states
end

function write_diagnostics(
    path::AbstractString,
    diagnostics;
    state_sha256::AbstractString="",
    metadata=Dict{String,Any}(),
    immutable::Bool=false,
)
    immutable && ispath(path) && throw(ArgumentError("refusing to overwrite immutable diagnostics: $path"))
    mkpath(dirname(path))
    temporary = tempname(dirname(path))
    h5open(temporary, "w") do file
        file["schema_version"] = 1
        file["artifact_kind"] = "ladder_mps_mft_diagnostics"
        file["state_sha256"] = String(state_sha256)
        for (key, value) in metadata
            file[String(key)] = value isa Symbol ? String(value) : value
        end
        file["density"] = diagnostics.density
        file["spin"] = diagnostics.spin
        file["expected_spin_q"] = diagnostics.expected_spin_q
        file["spin_q_mismatch"] = diagnostics.spin_q_mismatch
        for (name, grid) in (("charge_structure", diagnostics.charge_structure), ("spin_structure", diagnostics.spin_structure))
            group = create_group(file, name)
            group["qx"] = grid.qx
            group["ky"] = grid.ky
            group["values"] = grid.values
        end
        for (name, peak) in (("charge_peak", diagnostics.charge_peak), ("spin_peak", diagnostics.spin_peak))
            group = create_group(file, name)
            for key in keys(peak)
                group[String(key)] = getproperty(peak, key)
            end
        end
        krho = create_group(file, "K_rho")
        for key in keys(diagnostics.K_rho)
            krho[String(key)] = getproperty(diagnostics.K_rho, key)
        end
        entropy = create_group(file, "entanglement")
        entropy["bonds"] = diagnostics.entanglement.bonds
        entropy["entropy"] = diagnostics.entanglement.entropy
        entropy["renyi2"] = diagnostics.entanglement.renyi2
        central = create_group(file, "central_charge")
        for key in keys(diagnostics.central_charge)
            central[String(key)] = getproperty(diagnostics.central_charge, key)
        end
        if diagnostics.pair_correlations !== nothing
            pair = create_group(file, "pair_correlations")
            pair["rung"] = diagnostics.pair_correlations.rung
            pair["leg0"] = diagnostics.pair_correlations.leg0
            pair["leg1"] = diagnostics.pair_correlations.leg1
            pair["convention"] = diagnostics.pair_correlations.convention
        end
    end
    mv(temporary, path; force=!immutable)
    return path
end

function write_sector_gaps(path::AbstractString, gaps; immutable::Bool=false)
    immutable && ispath(path) && throw(ArgumentError("refusing to overwrite immutable sector gaps: $path"))
    mkpath(dirname(path))
    temporary = tempname(dirname(path))
    h5open(temporary, "w") do file
        file["schema_version"] = 1
        file["artifact_kind"] = "ladder_sector_gaps"
        file["particle_number"] = gaps.particle_number
        file["spin_gap"] = gaps.spin_gap
        file["charge_gap"] = gaps.charge_gap
        file["hole_pair_binding"] = gaps.hole_pair_binding
        file["particle_pair_binding"] = gaps.particle_pair_binding
        group = create_group(file, "sector_energies")
        for ((particle_number, twice_sz), energy) in sort(collect(gaps.energies); by=first)
            group["N$(particle_number)_twoSz$(twice_sz)"] = energy
        end
    end
    mv(temporary, path; force=!immutable)
    return path
end

function build_bare_ladder_mpo(sites, model::ModelSettings)
    fields = FieldState(
        zeros(Float64, model.L, model.L, 2, 2),
        zeros(Float64, 2, model.L, model.L, 2, 2),
        zeros(Float64, 2, 2 * model.L),
    )
    return build_mf_mpo(sites, model, fields, 0.0)
end

function sector_energy(model::ModelSettings, dmrg_settings::DMRGSettings, particle_number::Int, twice_sz::Int)
    sites = siteinds("Electron", 2 * model.L; conserve_nf=true, conserve_sz=true)
    mpo = build_bare_ladder_mpo(sites, model)
    psi0 = productMPS(sites, fixed_sector_product_state(sites, particle_number, twice_sz))
    result = run_dmrg_ground(
        sites,
        mpo,
        particle_number / (2 * model.L),
        dmrg_settings;
        psi_init=psi0,
        deadline=time() + dmrg_settings.max_time_seconds,
    )
    result.timed_out && error("fixed-sector DMRG timed out for N=$particle_number, 2Sz=$twice_sz")
    return result.energy
end

function sector_resolved_gaps(model::ModelSettings, dmrg_settings::DMRGSettings)
    particle_number = round(Int, 2 * model.L * model.density)
    iseven(particle_number) || throw(ArgumentError("sector diagnostic currently requires even target N"))
    sectors = Dict{Tuple{Int,Int},Float64}()
    for key in ((particle_number, 0), (particle_number, 2), (particle_number - 2, 0),
                (particle_number - 1, 1), (particle_number + 1, 1), (particle_number + 2, 0))
        sectors[key] = sector_energy(model, dmrg_settings, key...)
    end
    e0 = sectors[(particle_number, 0)]
    return (
        particle_number,
        energies=sectors,
        spin_gap=sectors[(particle_number, 2)] - e0,
        charge_gap=0.5 * (sectors[(particle_number + 2, 0)] + sectors[(particle_number - 2, 0)] - 2 * e0),
        hole_pair_binding=sectors[(particle_number - 2, 0)] + e0 - 2 * sectors[(particle_number - 1, 1)],
        particle_pair_binding=sectors[(particle_number + 2, 0)] + e0 - 2 * sectors[(particle_number + 1, 1)],
    )
end
