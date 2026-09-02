function connected_covariance_matrix(
    correlation::AbstractMatrix,
    expectation::AbstractVector,
)
    size(correlation, 1) == size(correlation, 2) == length(expectation) ||
        throw(DimensionMismatch("correlation and expectation dimensions differ"))
    covariance = real.(correlation) .- real.(expectation) * transpose(real.(expectation))
    return Matrix(Symmetric((covariance + transpose(covariance)) / 2))
end

function leg_parity_covariance(covariance::AbstractMatrix, L::Integer)
    size(covariance) == (2L, 2L) || throw(DimensionMismatch(
        "leg-parity covariance must be a 2L by 2L matrix",
    ))
    even = zeros(Float64, L, L)
    odd = zeros(Float64, L, L)
    cross = zeros(Float64, L, L)
    for left in 1:L, right in 1:L
        l0, l1 = rung_leg_to_site(left, 0), rung_leg_to_site(left, 1)
        r0, r1 = rung_leg_to_site(right, 0), rung_leg_to_site(right, 1)
        even[left, right] = 0.5 * (
            covariance[l0, r0] + covariance[l0, r1] +
            covariance[l1, r0] + covariance[l1, r1]
        )
        odd[left, right] = 0.5 * (
            covariance[l0, r0] - covariance[l0, r1] -
            covariance[l1, r0] + covariance[l1, r1]
        )
        cross[left, right] = 0.5 * (
            covariance[l0, r0] - covariance[l0, r1] +
            covariance[l1, r0] - covariance[l1, r1]
        )
    end
    return (;
        even=Matrix(Symmetric((even + transpose(even)) / 2)),
        odd=Matrix(Symmetric((odd + transpose(odd)) / 2)),
        cross,
        cross_relative_norm=norm(cross) / max(norm(covariance), eps(Float64)),
    )
end

function _mode_fourier_summary(vector::AbstractVector, model::ModelSettings; bond_centered::Bool=false)
    positions = bond_centered ? collect(1:length(vector)) .+ 0.5 : collect(1:length(vector))
    mode_numbers = collect(0:div(model.L, 2))
    q_values = 2pi .* mode_numbers ./ model.L
    amplitudes = [
        abs(sum(vector .* exp.(-im .* q .* positions))) / sqrt(length(vector))
        for q in q_values
    ]
    index = argmax(amplitudes)
    edge_count = max(1, ceil(Int, 0.125 * length(vector)))
    edge_indices = union(1:edge_count, (length(vector) - edge_count + 1):length(vector))
    edge_weight = sum(abs2, vector[edge_indices])
    return (;
        mode_number=mode_numbers[index],
        q=q_values[index],
        q_over_pi=q_values[index] / pi,
        fourier_overlap=amplitudes[index],
        inverse_participation=sum(abs2.(vector) .^ 2),
        edge_weight,
    )
end

function covariance_eigensystem(
    covariance::AbstractMatrix,
    model::ModelSettings;
    top_modes::Integer,
    bond_centered::Bool=false,
)
    matrix = Matrix(Symmetric((real.(covariance) + transpose(real.(covariance))) / 2))
    decomposition = eigen(Symmetric(matrix))
    order = sortperm(decomposition.values; rev=true)
    count = min(Int(top_modes), length(order))
    selected = order[1:count]
    values = decomposition.values[selected]
    vectors = decomposition.vectors[:, selected]
    residuals = [
        norm(matrix * vectors[:, index] - values[index] * vectors[:, index])
        for index in 1:count
    ]
    summaries = [
        _mode_fourier_summary(vectors[:, index], model; bond_centered)
        for index in 1:count
    ]
    return (;
        covariance=matrix,
        eigenvalues=values,
        eigenvectors=vectors,
        residuals,
        mode_numbers=[summary.mode_number for summary in summaries],
        q=[summary.q for summary in summaries],
        q_over_pi=[summary.q_over_pi for summary in summaries],
        fourier_overlap=[summary.fourier_overlap for summary in summaries],
        inverse_participation=[summary.inverse_participation for summary in summaries],
        edge_weight=[summary.edge_weight for summary in summaries],
        trace=tr(matrix),
        minimum_eigenvalue=minimum(decomposition.values),
        negative_eigenvalue_weight=sum(abs, decomposition.values[decomposition.values .< 0]),
    )
end

function _power_law_fit(distance::AbstractVector, magnitude::AbstractVector)
    selected = findall(index -> distance[index] > 0 && magnitude[index] > 100eps(Float64), eachindex(distance))
    length(selected) >= 3 || return (exponent=NaN, intercept=NaN, r2=NaN, points=length(selected))
    x = log.(Float64.(distance[selected]))
    y = log.(Float64.(magnitude[selected]))
    design = hcat(ones(length(x)), x)
    coefficients = design \ y
    prediction = design * coefficients
    residual = sum(abs2, y .- prediction)
    total = sum(abs2, y .- mean(y))
    return (;
        exponent=-coefficients[2],
        intercept=coefficients[1],
        r2=total > 0 ? 1 - residual / total : 1.0,
        points=length(selected),
    )
end

function _decay_profile(matrix::AbstractMatrix, edge_fraction::Real)
    count = size(matrix, 1)
    size(matrix, 2) == count || throw(DimensionMismatch("decay matrix must be square"))
    edge = floor(Int, edge_fraction * count)
    first_site = 1 + edge
    last_site = count - edge
    maximum_distance = max(0, div(last_site - first_site, 2))
    distance = collect(1:maximum_distance)
    magnitude = [
        mean(abs(matrix[left, left + separation]) for left in first_site:(last_site - separation))
        for separation in distance
    ]
    return (; distance, magnitude)
end

function correlation_exponent_with_window_uncertainty(
    matrix::AbstractMatrix,
    edge_fractions::AbstractVector,
)
    fits = NamedTuple[]
    profiles = NamedTuple[]
    for fraction in edge_fractions
        profile = _decay_profile(matrix, fraction)
        fit = _power_law_fit(profile.distance, profile.magnitude)
        push!(profiles, (; edge_fraction=Float64(fraction), profile...))
        push!(fits, (; edge_fraction=Float64(fraction), fit...))
    end
    finite = [fit.exponent for fit in fits if isfinite(fit.exponent)]
    estimate = isempty(finite) ? NaN : median(finite)
    uncertainty = length(finite) < 2 ? NaN : (maximum(finite) - minimum(finite)) / 2
    return (; estimate, window_uncertainty=uncertainty, fits, profiles)
end

function _raw_map_norm(correlations::CorrelationState, model::ModelSettings)
    total = sum(abs2, correlations.density_down .- 0.5) +
        sum(abs2, correlations.density_up .- 0.5)
    for left in 1:(2 * model.L), right in 1:(2 * model.L)
        rung_left, _ = site_to_rung_leg(left)
        rung_right, _ = site_to_rung_leg(right)
        abs(rung_left - rung_right) <= model.r_range || continue
        total += abs2(correlations.pair[left, right])
        left == right && continue
        total += abs2(correlations.exchange_down[left, right])
        total += abs2(correlations.exchange_up[left, right])
    end
    return sqrt(total)
end

function _pair_class_matrix(pair, class::Symbol)
    field_class = Symbol(class, "_field")
    hasproperty(pair, field_class) || throw(ArgumentError(
        "pair correlations do not contain Hermitian-field class $field_class",
    ))
    matrix = getproperty(pair, field_class)
    # A unique non-onsite alpha coordinate is stored twice in the symmetric
    # field array. Dividing its operator by sqrt(2) makes this covariance use
    # the same Euclidean field metric as Stage 2.
    scale = class in (:rung, :leg0, :leg1) ? 0.5 : 1.0
    return scale .* Matrix(Symmetric((matrix + transpose(matrix)) / 2))
end

function compute_bare_stage1(
    psi::MPS,
    model::ModelSettings,
    settings::BareStage1Settings=BareStage1Settings(),
)
    diagnostics = compute_ladder_diagnostics(psi, model; full_pair_correlations=true)
    charge_covariance = connected_covariance_matrix(
        diagnostics.charge_correlation,
        diagnostics.density,
    )
    spin_covariance = connected_covariance_matrix(
        diagnostics.spin_correlation,
        diagnostics.spin,
    )
    charge_parity = leg_parity_covariance(charge_covariance, model.L)
    spin_parity = leg_parity_covariance(spin_covariance, model.L)
    charge = (;
        covariance=charge_covariance,
        cross_relative_norm=charge_parity.cross_relative_norm,
        even=covariance_eigensystem(
            charge_parity.even,
            model;
            top_modes=settings.top_modes,
        ),
        odd=covariance_eigensystem(
            charge_parity.odd,
            model;
            top_modes=settings.top_modes,
        ),
    )
    spin = (;
        covariance=spin_covariance,
        cross_relative_norm=spin_parity.cross_relative_norm,
        even=covariance_eigensystem(
            spin_parity.even,
            model;
            top_modes=settings.top_modes,
        ),
        odd=covariance_eigensystem(
            spin_parity.odd,
            model;
            top_modes=settings.top_modes,
        ),
    )
    pair = Dict{Symbol,NamedTuple}()
    for class in settings.pair_classes
        matrix = _pair_class_matrix(diagnostics.pair_correlations, class)
        pair[class] = covariance_eigensystem(
            matrix,
            model;
            top_modes=settings.top_modes,
            bond_centered=class in (:leg0, :leg1),
        )
    end
    _, raw_correlations = calculate_mean_fields(psi, model)
    rung_charge_covariance = 2 .* charge_parity.even
    charge_decay = correlation_exponent_with_window_uncertainty(
        rung_charge_covariance,
        settings.bulk_edge_fractions,
    )
    pair_decay = correlation_exponent_with_window_uncertainty(
        diagnostics.pair_correlations.rung_field,
        settings.bulk_edge_fractions,
    )
    minimum_covariance_eigenvalue = minimum(vcat(
        [charge.even.minimum_eigenvalue, charge.odd.minimum_eigenvalue,
         spin.even.minimum_eigenvalue, spin.odd.minimum_eigenvalue],
        [spectrum.minimum_eigenvalue for spectrum in values(pair)],
    ))
    return (;
        diagnostics,
        charge,
        spin,
        pair,
        raw_correlations,
        raw_map_norm=_raw_map_norm(raw_correlations, model),
        charge_decay,
        pair_decay,
        minimum_covariance_eigenvalue,
        covariance_psd_pass=minimum_covariance_eigenvalue >= -settings.covariance_psd_tol,
        covariance_psd_tol=settings.covariance_psd_tol,
    )
end

function _write_spectrum(group, spectrum)
    group["covariance"] = spectrum.covariance
    group["eigenvalues"] = spectrum.eigenvalues
    group["eigenvectors"] = spectrum.eigenvectors
    group["residual_norm"] = spectrum.residuals
    group["mode_number"] = spectrum.mode_numbers
    group["q"] = spectrum.q
    group["q_over_pi"] = spectrum.q_over_pi
    group["fourier_overlap"] = spectrum.fourier_overlap
    group["inverse_participation"] = spectrum.inverse_participation
    group["edge_weight"] = spectrum.edge_weight
    group["trace"] = spectrum.trace
    group["minimum_eigenvalue"] = spectrum.minimum_eigenvalue
    group["negative_eigenvalue_weight"] = spectrum.negative_eigenvalue_weight
    return group
end

function _write_decay(group, decay)
    group["estimate"] = decay.estimate
    group["window_uncertainty"] = decay.window_uncertainty
    for (index, fit) in enumerate(decay.fits)
        child = create_group(group, lpad(string(index), 3, '0'))
        child["edge_fraction"] = fit.edge_fraction
        child["exponent"] = fit.exponent
        child["intercept"] = fit.intercept
        child["r2"] = fit.r2
        child["points"] = fit.points
        child["distance"] = decay.profiles[index].distance
        child["magnitude"] = decay.profiles[index].magnitude
    end
    return group
end

function write_bare_stage1(
    path::AbstractString,
    result,
    model::ModelSettings;
    backbone_path::AbstractString="",
    config_path::AbstractString="",
    immutable::Bool=true,
)
    destination = abspath(path)
    immutable && ispath(destination) && throw(ArgumentError(
        "refusing to overwrite immutable Stage 1 artifact: $destination",
    ))
    mkpath(dirname(destination))
    temporary = tempname(dirname(destination))
    h5open(temporary, "w") do file
        file["schema_version"] = 2
        file["artifact_kind"] = "bare_ladder_stage1_covariance_screen"
        file["complete"] = true
        _write_backbone_model(create_group(file, "model"), model)
        file["raw_map_norm"] = result.raw_map_norm
        file["minimum_covariance_eigenvalue"] = result.minimum_covariance_eigenvalue
        file["covariance_psd_pass"] = result.covariance_psd_pass
        file["covariance_psd_tol"] = result.covariance_psd_tol

        normal = create_group(file, "normal")
        for (name, sector) in (("charge", result.charge), ("spin", result.spin))
            parent = create_group(normal, name)
            parent["site_covariance"] = sector.covariance
            parent["opposite_parity_relative_norm"] = sector.cross_relative_norm
            _write_spectrum(create_group(parent, "even"), sector.even)
            _write_spectrum(create_group(parent, "odd"), sector.odd)
        end
        pairing = create_group(file, "pairing")
        pairing["convention"] = result.diagnostics.pair_correlations.convention
        pairing["screened_covariance"] =
            "Re(<Delta_i Delta_j^dagger> + <Delta_i^dagger Delta_j>) for the Hermitian source Delta+Delta^dagger"
        pairing["field_metric"] = "unique symmetric alpha coordinate; non-onsite bond operators divided by sqrt(2)"
        pairing["candidate_construction"] =
            "complete real-space covariance within each onsite/rung/leg class; cross-class mixing is deferred to Stage 2 candidate orthonormalization"
        for class in sort(collect(keys(result.pair)); by=string)
            _write_spectrum(create_group(pairing, String(class)), result.pair[class])
        end
        decay = create_group(file, "decay_fits")
        _write_decay(create_group(decay, "charge_rung_total"), result.charge_decay)
        _write_decay(create_group(decay, "rung_pair"), result.pair_decay)

        raw = create_group(file, "zero_field_raw_map")
        raw["pair"] = result.raw_correlations.pair
        raw["exchange_down"] = result.raw_correlations.exchange_down
        raw["exchange_up"] = result.raw_correlations.exchange_up
        raw["density_down"] = result.raw_correlations.density_down
        raw["density_up"] = result.raw_correlations.density_up
        raw["density_reference"] = 0.5

        diagnostic = create_group(file, "diagnostics")
        diagnostic["density"] = result.diagnostics.density
        diagnostic["spin"] = result.diagnostics.spin
        diagnostic["charge_correlation"] = result.diagnostics.charge_correlation
        diagnostic["spin_correlation"] = result.diagnostics.spin_correlation
        for (name, grid) in (("charge_structure", result.diagnostics.charge_structure),
                             ("spin_structure", result.diagnostics.spin_structure))
            child = create_group(diagnostic, name)
            child["qx"] = grid.qx
            child["ky"] = grid.ky
            child["values"] = grid.values
        end
        diagnostic["entanglement_bonds"] = result.diagnostics.entanglement.bonds
        diagnostic["entanglement_entropy"] = result.diagnostics.entanglement.entropy
        diagnostic["entanglement_renyi2"] = result.diagnostics.entanglement.renyi2
        diagnostic["K_rho_site_normalized"] = result.diagnostics.K_rho.K_rho_site_normalized
        diagnostic["K_rho_rung_normalized"] = result.diagnostics.K_rho.K_rho_rung_normalized
        diagnostic["central_charge"] = result.diagnostics.central_charge.central_charge
        diagnostic["central_charge_r2"] = result.diagnostics.central_charge.r2

        provenance = create_group(file, "provenance")
        provenance["backbone_path"] = isempty(backbone_path) ? "" : abspath(backbone_path)
        provenance["backbone_sha256"] = isempty(backbone_path) ? "" : sha256_file(abspath(backbone_path))
        provenance["config_path"] = isempty(config_path) ? "" : abspath(config_path)
        provenance["config_sha256"] = isempty(config_path) ? "" : sha256_file(abspath(config_path))
        provenance["implementation_sha256"] = implementation_fingerprint()
        provenance["git_commit"] = _read_git("rev-parse", "HEAD")
        provenance["julia_threads"] = Threads.nthreads()
    end
    mv(temporary, destination; force=!immutable)
    return destination
end
