function field_energy_components(fields::FieldState, correlations::CorrelationState, model::ModelSettings)
    pair_energy = 0.0
    exchange_energy = 0.0
    for i in 1:model.L, ip in 1:model.L
        abs(i - ip) <= model.r_range || continue
        for leg in 0:1, other_leg in 0:1
            site_i = rung_leg_to_site(i, leg)
            site_ip = rung_leg_to_site(ip, other_leg)
            alpha = fields.alpha[i, ip, leg + 1, other_leg + 1]
            pair_energy -= 2 * alpha * correlations.pair[site_ip, site_i]
            site_i == site_ip && continue
            exchange_energy += fields.beta[1, i, ip, leg + 1, other_leg + 1] * correlations.exchange_down[site_i, site_ip]
            exchange_energy += fields.beta[2, i, ip, leg + 1, other_leg + 1] * correlations.exchange_up[site_i, site_ip]
        end
    end
    density_energy = dot(fields.mu_cdw[1, :], correlations.density_down) +
        dot(fields.mu_cdw[2, :], correlations.density_up)
    return (; pair=pair_energy, exchange=exchange_energy, density=density_energy)
end

function variational_energy(
    effective_eigenvalue::Real,
    chemical_potential::Real,
    fields::FieldState,
    correlations::CorrelationState,
    model::ModelSettings,
    ;
    interaction_fields::FieldState=fields,
    effective_expectation::Real=effective_eigenvalue,
    bare_ladder_energy::Real=NaN,
)
    linear = field_energy_components(fields, correlations, model)
    # `fields` are the fields applied by the partner transverse subsystem. At
    # a fixed point they equal the measured fields. On a physical p-cycle they
    # belong to the preceding orbit phase and must not be replaced by an
    # Anderson/linear average or by the current phase's outgoing fields.
    transverse_linear = field_energy_components(interaction_fields, correlations, model)
    density_delta_down = correlations.density_down .- 0.5
    density_delta_up = correlations.density_up .- 0.5
    pair_transverse = 0.5 * transverse_linear.pair
    exchange_transverse = 0.5 * transverse_linear.exchange
    density_transverse = 0.5 * (
        dot(interaction_fields.mu_cdw[1, :], density_delta_down) +
        dot(interaction_fields.mu_cdw[2, :], density_delta_up)
    )
    correction = pair_transverse + exchange_transverse + density_transverse -
        (linear.pair + linear.exchange + linear.density)
    particle_number = sum(correlations.density_down) + sum(correlations.density_up)
    mu_term = Float64(chemical_potential) * particle_number
    linear_total = linear.pair + linear.exchange + linear.density
    reconstructed_bare = Float64(effective_expectation) + mu_term - linear_total
    transverse_total = pair_transverse + exchange_transverse + density_transverse
    reconstructed_variational = reconstructed_bare + transverse_total
    direct_variational = isfinite(bare_ladder_energy) ? Float64(bare_ladder_energy) + transverse_total : reconstructed_variational
    identity_error = isfinite(bare_ladder_energy) ? reconstructed_bare - Float64(bare_ladder_energy) : NaN
    consistency_error = reconstructed_variational - direct_variational
    canonical = direct_variational
    grand = canonical - mu_term
    return EnergyBreakdown(;
        effective_eigenvalue=Float64(effective_eigenvalue),
        effective_expectation=Float64(effective_expectation),
        effective_eigenvalue_error=Float64(effective_eigenvalue) - Float64(effective_expectation),
        bare_ladder_energy=Float64(bare_ladder_energy),
        reconstructed_bare_ladder_energy=reconstructed_bare,
        hamiltonian_identity_error=identity_error,
        chemical_potential_term=mu_term,
        pair_field_energy=linear.pair,
        exchange_field_energy=linear.exchange,
        density_field_energy=linear.density,
        pair_transverse_energy=pair_transverse,
        exchange_transverse_energy=exchange_transverse,
        density_transverse_energy=density_transverse,
        double_counting_correction=correction,
        reconstructed_variational_energy=reconstructed_variational,
        direct_variational_energy=direct_variational,
        variational_consistency_error=consistency_error,
        canonical_variational_energy=canonical,
        grand_potential=grand,
    )
end
