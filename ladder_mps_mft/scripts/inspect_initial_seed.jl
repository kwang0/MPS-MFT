#!/usr/bin/env julia

using LadderMPSMFT
using Printf
using Random

length(ARGS) in (1, 2) || error(
    "usage: julia --project=. scripts/inspect_initial_seed.jl CONFIG.toml [PROFILE.tsv]",
)

config_path = abspath(ARGS[1])
settings = load_settings(config_path)
run = settings.run
run.initial_seed_protocol == :matched_mode || error(
    "inspect_initial_seed.jl is for explicit matched_mode configs; this config uses $(run.initial_seed_protocol)",
)

fields = initial_fields(
    settings.model;
    seed=run.initial_seed,
    amplitude=run.initial_amplitude,
    rng=MersenneTwister(run.random_seed),
    protocol=run.initial_seed_protocol,
    mode_number=run.initial_mode_number,
    mode_phase_pi=run.initial_mode_phase_pi,
    pairing_form_factor=run.initial_pairing_form_factor,
    leg_parity=run.initial_leg_parity,
    stripe_charge_to_spin_ratio=run.initial_stripe_charge_to_spin_ratio,
    stripe_pairing_to_spin_ratio=run.initial_stripe_pairing_to_spin_ratio,
    random_seed=run.random_seed,
)
metadata = initial_seed_metadata(settings.model, run)
L = settings.model.L

function site_source(rung, leg)
    site = 2 * (rung - 1) + leg
    down = fields.mu_cdw[1, site]
    up = fields.mu_cdw[2, site]
    return (charge=(down + up) / 2, spin=(down - up) / 2)
end

rows = NamedTuple[]
for rung in 1:L
    leg1 = site_source(rung, 1)
    leg2 = site_source(rung, 2)
    onsite = (fields.alpha[rung, rung, 1, 1] + fields.alpha[rung, rung, 2, 2]) / 2
    rung_pair = fields.alpha[rung, rung, 1, 2]
    leg_pair = rung < L ? (
        fields.alpha[rung, rung + 1, 1, 1] + fields.alpha[rung, rung + 1, 2, 2]
    ) / 2 : NaN
    push!(rows, (
        rung,
        charge_even=(leg1.charge + leg2.charge) / 2,
        charge_odd=(leg1.charge - leg2.charge) / 2,
        spin_even=(leg1.spin + leg2.spin) / 2,
        spin_odd=(leg1.spin - leg2.spin) / 2,
        pair_onsite_s=onsite,
        pair_rung_s=rung_pair,
        pair_leg_s=leg_pair,
        pair_extended_s=isfinite(leg_pair) ? leg_pair + rung_pair : NaN,
        pair_d_wave=isfinite(leg_pair) ? leg_pair - rung_pair : NaN,
    ))
end

@printf("config=%s\n", config_path)
@printf("channel=%s\n", run.initial_seed)
@printf("protocol=%s\n", metadata.protocol)
@printf("mode_number=%d\n", metadata.mode_number)
@printf("mode_wavevector_pi=%.12g\n", metadata.mode_wavevector_pi)
@printf("mode_phase_pi=%.12g\n", metadata.mode_phase_pi)
@printf("pairing_form_factor=%s\n", metadata.pairing_form_factor)
@printf("leg_parity=%s\n", metadata.resolved_leg_parity)
if run.initial_seed in (:stripe, :stripe_pairing)
    @printf("stripe_envelope_mode_number=%d\n", metadata.stripe_envelope_mode_number)
    @printf("stripe_spin_mode_number=%d\n", metadata.stripe_spin_mode_number)
    @printf("stripe_spin_wavevector_pi=%.12g\n", metadata.stripe_spin_wavevector_pi)
    @printf("stripe_charge_mode_number=%d\n", metadata.stripe_charge_mode_number)
    @printf("stripe_charge_wavevector_pi=%.12g\n", metadata.stripe_charge_wavevector_pi)
    @printf("stripe_charge_to_spin_ratio=%.12g\n", metadata.stripe_charge_to_spin_ratio)
    @printf("stripe_pairing_to_spin_ratio=%.12g\n", metadata.stripe_pairing_to_spin_ratio)
end
if run.initial_seed == :legacy_pairing
    @printf("legacy_pairing_random_seed=%d\n", metadata.legacy_pairing_random_seed)
    @printf("legacy_pairing_center_of_mass_structure=%s\n", metadata.legacy_pairing_center_of_mass_structure)
    @printf("legacy_pairing_beta_initialization=%s\n", metadata.legacy_pairing_beta_initialization)
    @printf("legacy_pairing_mu_cdw_initialization=%s\n", metadata.legacy_pairing_mu_cdw_initialization)
end
@printf("field_l2_per_physical_site=%.12g\n", field_l2_per_physical_site(fields, settings.model))
@printf("max_abs_alpha=%.12g\n", maximum(abs, fields.alpha))
@printf("max_abs_beta=%.12g\n", maximum(abs, fields.beta))
@printf("max_abs_mu_cdw=%.12g\n", maximum(abs, fields.mu_cdw))
@printf("initial_seed_fingerprint=%s\n", initial_seed_fingerprint(settings))

if length(ARGS) == 2
    output_path = abspath(ARGS[2])
    ispath(output_path) && error("refusing to overwrite seed profile: $output_path")
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        println(io, join(propertynames(first(rows)), '\t'))
        for row in rows
            println(io, join((getproperty(row, name) for name in propertynames(row)), '\t'))
        end
    end
    println("profile_path=$output_path")
end
