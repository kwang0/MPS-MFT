#!/usr/bin/env julia

using LadderMPSMFT
using TOML

length(ARGS) == 4 || error(
    "usage: julia --project=. scripts/prepare_phase1_square_seed_pilot.jl BASE_CONFIG.toml CONTROL_RUN FULL_RUN RUN_ID",
)

const COMMON_RANDOM_SEED = 1404
const INITIAL_AMPLITUDE = 1.0e-3
const STRIPE_CHARGE_TO_SPIN_RATIO = 0.2
const STRIPE_PAIRING_TO_SPIN_RATIO = 1.0
const POINT_CONTRACTS = (
    (
        point_id="square_t014_vm04",
        V=-0.4,
        t0=1.4,
        mu_initial=0.55,
        ep_signed=-0.24962435880865996,
    ),
    (
        point_id="square_t014_v000",
        V=0.0,
        t0=1.4,
        mu_initial=0.55,
        ep_signed=-0.14653773091916378,
    ),
)

base_path = abspath(ARGS[1])
control_run = abspath(ARGS[2])
full_run = abspath(ARGS[3])
run_id = ARGS[4]

isfile(base_path) || error("square seed-pilot base configuration not found: $base_path")
occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID: $run_id")

base_settings = load_settings(base_path)
base_settings.model.geometry == :square || error("square seed pilot must use square geometry")
base_settings.model.L == 64 || error("square seed pilot must use L=64")
base_settings.model.U == 8.0 || error("square seed pilot must use U=8")
point_matches = filter(
    contract -> base_settings.model.V == contract.V && base_settings.model.t0 == contract.t0,
    POINT_CONTRACTS,
)
length(point_matches) == 1 || error(
    "unsupported square seed-pilot point V=$(base_settings.model.V), t0=$(base_settings.model.t0)",
)
point_contract = only(point_matches)
base_settings.model.tp == 0.1 || error("square seed pilot must use t_perp=0.1")
base_settings.model.density == 0.9375 || error("square seed pilot must use density=0.9375")
base_settings.model.mu_initial == point_contract.mu_initial || error(
    "square seed pilot at $(point_contract.point_id) must begin from mu=$(point_contract.mu_initial)",
)
base_settings.model.ep_mode == :exact || error("square seed pilot must use an exact E_p registry row")
base_settings.model.ep_signed == point_contract.ep_signed || error(
    "square seed pilot at $(point_contract.point_id) resolved an unexpected E_p=$(base_settings.model.ep_signed)",
)
base_settings.runtime.backend == :gpu || error("square seed pilot must use the GPU backend")
base_settings.runtime.tensor_scalar_type == :float64 || error(
    "square seed pilot must request Float64 tensors",
)
base_settings.dmrg.nsweeps == 12 || error("square seed pilot must use 12 DMRG sweeps")
base_settings.dmrg.maxdim == 200 || error("square seed pilot must use chi=200")
base_settings.dmrg.cutoff == 1.0e-10 || error("square seed pilot must use cutoff=1e-10")
base_settings.dmrg.energy_tol == 1.0e-6 || error("square seed pilot must use DMRG energy_tol=1e-6")
base_settings.dmrg.mu_density_tol == 1.0e-3 || error(
    "square seed pilot must use mu_density_tol=1e-3",
)
base_settings.dmrg.mu_bracket_step == 0.01 || error(
    "square seed pilot must use initial mu bracket step=0.01",
)
base_settings.dmrg.mu_bracket_growth == 3.0 || error(
    "square seed pilot must use mu bracket growth=3",
)
base_settings.dmrg.mu_warm_start_noise == 1.0e-8 || error(
    "square seed pilot must use 1e-8 noise for warm-started mu re-solves",
)
base_settings.convergence.density_tol == 1.0e-3 || error(
    "square seed pilot must use outer density_tol=1e-3",
)
base_settings.convergence.variational_energy_tol == 1.0e-6 || error(
    "square seed pilot must use exploratory variational_energy_tol=1e-6",
)
base_settings.convergence.field_abs_tol == 1.0e-6 || error(
    "square seed pilot must use exploratory field_abs_tol=1e-6",
)
base_settings.convergence.field_rel_tol == 5.0e-3 || error(
    "square seed pilot must use exploratory field_rel_tol=5e-3",
)
base_settings.convergence.period2_oscillation_cosine_max == -0.5 || error(
    "square seed pilot must require a negative period-two step cosine",
)
base_settings.convergence.period2_two_step_ratio_max == 0.5 || error(
    "square seed pilot must require d2/d1 <= 0.5 for period two",
)
base_settings.convergence.slow_mode_cosine_min == 0.9 || error(
    "square seed pilot must apply slow-mode extrapolation above cosine 0.9",
)
base_settings.run.initial_seed_protocol == :matched_mode || error(
    "square seed-pilot base must opt in to matched_mode",
)
base_settings.run.initial_amplitude == INITIAL_AMPLITUDE || error(
    "square seed-pilot base must use initial_amplitude=1e-3",
)
base_settings.run.random_seed == COMMON_RANDOM_SEED || error(
    "square seed-pilot base must use the declared common random seed $COMMON_RANDOM_SEED",
)
base_settings.run.initial_stripe_charge_to_spin_ratio == STRIPE_CHARGE_TO_SPIN_RATIO || error(
    "square seed-pilot base must use stripe charge:spin ratio 0.2",
)
base_settings.run.initial_stripe_pairing_to_spin_ratio == STRIPE_PAIRING_TO_SPIN_RATIO || error(
    "square seed-pilot base must use stripe pairing:spin ratio 1",
)
base_settings.convergence.unmixed_cycle_probe || error(
    "square seed pilot must begin with the unmixed raw-map probe",
)
base_settings.convergence.probe_iterations == 20 || error(
    "square seed pilot must use exactly 20 raw-map probe updates",
)
base_settings.convergence.cycle_action == :continue || error(
    "square seed pilot must archive any unaccepted raw recurrence before optional acceleration",
)
base_settings.run.max_iterations == 80 || error("square seed pilot must allow 80 MF updates")

config_directory = joinpath(control_run, "configs")
ispath(config_directory) && !isempty(readdir(config_directory)) && error(
    "refusing to overwrite nonempty configuration directory: $config_directory",
)
mkpath(config_directory)
mkpath(joinpath(control_run, "results"))
mkpath(joinpath(full_run, "results"))

# The primary m=4 stripe envelope is read from the supplied converged square
# profile: its antiferromagnetic spin mode is L-1-m=59 and its charge second
# harmonic is 2m=8. The adjacent m=5 bank member (spin 58, charge 10) is
# predeclared before energy inspection. Pure pairing/stripe branches are
# symmetry-subspace controls; stripe_pairing starts allow both sectors to live.
# The legacy_pairing control follows the actual fresh-run legacy structure:
# alpha is random across relative bond/leg classes but constant along the
# center-of-mass direction, while beta and mu_cdw begin at exactly zero.
branches = (
    (
        label="square__pairing_dwave_m000_chi200_loose",
        branch="pairing_control",
        seed="pairing",
        mode=0,
        pairing_form_factor="d_wave",
        leg_parity="auto",
        charge_to_spin=STRIPE_CHARGE_TO_SPIN_RATIO,
        pairing_to_spin=0.0,
        analysis_role="pairing_symmetry_subspace_control",
    ),
    (
        label="square__legacy_pairing_mixed_chi200_loose",
        branch="legacy_pairing_control",
        seed="legacy_pairing",
        mode=0,
        pairing_form_factor="onsite_s",
        leg_parity="auto",
        charge_to_spin=STRIPE_CHARGE_TO_SPIN_RATIO,
        pairing_to_spin=0.0,
        analysis_role="legacy_translation_invariant_pairing_basin_control",
    ),
    (
        label="square__stripe_m004_chi200_loose",
        branch="stripe_control",
        seed="stripe",
        mode=4,
        pairing_form_factor="onsite_s",
        leg_parity="auto",
        charge_to_spin=STRIPE_CHARGE_TO_SPIN_RATIO,
        pairing_to_spin=0.0,
        analysis_role="normal_stripe_symmetry_subspace_control",
    ),
    (
        label="square__stripe_m005_chi200_loose",
        branch="stripe_control",
        seed="stripe",
        mode=5,
        pairing_form_factor="onsite_s",
        leg_parity="auto",
        charge_to_spin=STRIPE_CHARGE_TO_SPIN_RATIO,
        pairing_to_spin=0.0,
        analysis_role="normal_stripe_symmetry_subspace_control",
    ),
    (
        label="square__stripe_pairing_m004_chi200_loose",
        branch="coexistence",
        seed="stripe_pairing",
        mode=4,
        pairing_form_factor="d_wave",
        leg_parity="auto",
        charge_to_spin=STRIPE_CHARGE_TO_SPIN_RATIO,
        pairing_to_spin=STRIPE_PAIRING_TO_SPIN_RATIO,
        analysis_role="unrestricted_stripe_pairing_basin",
    ),
    (
        label="square__stripe_pairing_m005_chi200_loose",
        branch="coexistence",
        seed="stripe_pairing",
        mode=5,
        pairing_form_factor="d_wave",
        leg_parity="auto",
        charge_to_spin=STRIPE_CHARGE_TO_SPIN_RATIO,
        pairing_to_spin=STRIPE_PAIRING_TO_SPIN_RATIO,
        analysis_role="unrestricted_stripe_pairing_basin",
    ),
)

manifest_path = joinpath(control_run, "manifest.tsv")
ispath(manifest_path) && error("refusing to overwrite manifest: $manifest_path")
model_fingerprints = String[]
numerical_fingerprints = String[]
seed_fingerprints = String[]

open(manifest_path, "w") do io
    println(io, join((
        "label", "geometry", "point_id", "L", "U", "V", "t0", "tp", "density", "chi",
        "branch", "seed", "random_seed", "config", "config_sha256",
        "ep_mode", "ep_signed", "ep_abs", "ep_t0_lower", "ep_t0_upper",
        "ep_lower_signed", "ep_upper_signed", "ep_weight", "tp2_over_ep",
        "full_output_directory", "stateless_output_directory",
        "initial_seed_protocol", "initial_seed_fingerprint", "initial_amplitude",
        "initial_mode_number", "initial_mode_wavevector_pi", "initial_mode_phase_pi",
        "initial_pairing_form_factor", "initial_leg_parity_requested",
        "initial_leg_parity_resolved", "initial_seed_normalization",
        "stripe_envelope_mode_number", "stripe_spin_mode_number",
        "stripe_spin_wavevector_pi", "stripe_charge_mode_number",
        "stripe_charge_wavevector_pi", "stripe_charge_to_spin_ratio",
        "stripe_pairing_to_spin_ratio",
        "legacy_pairing_random_seed", "legacy_pairing_center_of_mass_structure",
        "legacy_pairing_beta_initialization", "legacy_pairing_mu_cdw_initialization",
        "analysis_role", "preliminary_energy_only", "dmrg_energy_tol", "dmrg_cutoff",
        "mu_density_tol", "outer_density_tol", "field_abs_tol", "field_rel_tol",
        "variational_energy_tol", "mu_bracket_step", "mu_bracket_growth",
        "mu_warm_start_noise", "period2_oscillation_cosine_max",
        "period2_two_step_ratio_max", "slow_mode_cosine_min",
    ), '\t'))

    for branch in branches
        raw = TOML.parsefile(base_path)
        run = raw["run"]
        full_output_directory = joinpath(full_run, "results", branch.label)
        stateless_output_directory = joinpath(control_run, "results", branch.label)
        run["output_directory"] = full_output_directory
        run["branch_label"] = branch.branch
        run["preparation"] = "square_targeted_mode_independent_seed_exploratory"
        run["direction"] = "none"
        run["seed_label"] = "targeted_$(branch.seed)_m$(lpad(branch.mode, 3, '0'))"
        run["random_seed"] = COMMON_RANDOM_SEED
        run["initial_seed"] = branch.seed
        run["initial_amplitude"] = INITIAL_AMPLITUDE
        run["initial_seed_protocol"] = "matched_mode"
        run["initial_mode_number"] = branch.mode
        run["initial_mode_phase_pi"] = 0.0
        run["initial_pairing_form_factor"] = branch.pairing_form_factor
        run["initial_leg_parity"] = branch.leg_parity
        run["initial_stripe_charge_to_spin_ratio"] = branch.charge_to_spin
        run["initial_stripe_pairing_to_spin_ratio"] = branch.pairing_to_spin
        for key in (
            "inherit_from", "inherit_sha256",
            "parent_checkpoint", "parent_sha256", "parent_orbit_phase",
            "resume_checkpoint", "resume_sha256",
        )
            pop!(run, key, nothing)
        end

        config_path = joinpath(config_directory, "$(branch.label).segment-001.toml")
        ispath(config_path) && error("refusing to overwrite square seed-pilot config: $config_path")
        open(config_path, "w") do config_io
            TOML.print(config_io, raw; sorted=true)
        end

        settings = load_settings(config_path)
        model = settings.model
        seed_metadata = initial_seed_metadata(model, settings.run)
        settings.run.random_seed == COMMON_RANDOM_SEED || error(
            "prepared branch lost the common product-state random seed: $(branch.label)",
        )
        settings.run.initial_seed == Symbol(branch.seed) || error(
            "prepared branch changed seed channel: $(branch.label)",
        )
        settings.run.initial_seed_protocol == :matched_mode || error(
            "prepared branch changed seed protocol: $(branch.label)",
        )
        settings.run.initial_mode_number == branch.mode || error(
            "prepared branch changed carrier mode: $(branch.label)",
        )
        settings.run.initial_mode_phase_pi == 0.0 || error(
            "prepared branch changed carrier phase: $(branch.label)",
        )
        settings.run.initial_pairing_form_factor == Symbol(branch.pairing_form_factor) || error(
            "prepared branch changed pairing form factor: $(branch.label)",
        )
        expected_parity = branch.seed in ("pairing", "legacy_pairing") ? :not_applicable :
            :mixed_even_charge_odd_spin
        seed_metadata.resolved_leg_parity == expected_parity || error(
            "prepared branch changed transverse parity: $(branch.label)",
        )
        if branch.seed in ("stripe", "stripe_pairing")
            seed_metadata.stripe_spin_mode_number == model.L - 1 - branch.mode || error(
                "prepared branch changed stripe spin harmonic: $(branch.label)",
            )
            seed_metadata.stripe_charge_mode_number == 2 * branch.mode || error(
                "prepared branch changed stripe charge harmonic: $(branch.label)",
            )
        end
        if branch.seed == "legacy_pairing"
            seed_metadata.legacy_pairing_random_seed == COMMON_RANDOM_SEED || error(
                "prepared legacy-like branch changed its field RNG seed: $(branch.label)",
            )
            seed_metadata.legacy_pairing_center_of_mass_structure ==
                "constant_by_relative_offset_and_leg_pair" || error(
                    "prepared legacy-like branch changed its spatial structure: $(branch.label)",
                )
            seed_metadata.legacy_pairing_beta_initialization == "zero" || error(
                "prepared legacy-like branch must initialize beta to zero: $(branch.label)",
            )
            seed_metadata.legacy_pairing_mu_cdw_initialization == "zero" || error(
                "prepared legacy-like branch must initialize mu_cdw to zero: $(branch.label)",
            )
        end

        push!(model_fingerprints, LadderMPSMFT.model_fingerprint(model))
        push!(numerical_fingerprints, LadderMPSMFT.numerical_fingerprint(settings))
        push!(seed_fingerprints, LadderMPSMFT.initial_seed_fingerprint(settings))
        println(io, join((
            branch.label,
            String(model.geometry),
            point_contract.point_id,
            string(model.L),
            string(model.U),
            string(model.V),
            string(model.t0),
            string(model.tp),
            string(model.density),
            string(settings.dmrg.maxdim),
            branch.branch,
            branch.seed,
            string(COMMON_RANDOM_SEED),
            config_path,
            LadderMPSMFT.sha256_file(config_path),
            String(model.ep_mode),
            string(model.ep_signed),
            string(model.ep),
            string(model.ep_t0_lower),
            string(model.ep_t0_upper),
            string(model.ep_lower_signed),
            string(model.ep_upper_signed),
            string(model.ep_interpolation_weight),
            string(model.tp^2 / model.ep),
            full_output_directory,
            stateless_output_directory,
            String(settings.run.initial_seed_protocol),
            LadderMPSMFT.initial_seed_fingerprint(settings),
            string(settings.run.initial_amplitude),
            string(seed_metadata.mode_number),
            string(seed_metadata.mode_wavevector_pi),
            string(seed_metadata.mode_phase_pi),
            String(seed_metadata.pairing_form_factor),
            String(seed_metadata.requested_leg_parity),
            String(seed_metadata.resolved_leg_parity),
            seed_metadata.normalization,
            string(seed_metadata.stripe_envelope_mode_number),
            string(seed_metadata.stripe_spin_mode_number),
            string(seed_metadata.stripe_spin_wavevector_pi),
            string(seed_metadata.stripe_charge_mode_number),
            string(seed_metadata.stripe_charge_wavevector_pi),
            string(seed_metadata.stripe_charge_to_spin_ratio),
            string(seed_metadata.stripe_pairing_to_spin_ratio),
            string(seed_metadata.legacy_pairing_random_seed),
            seed_metadata.legacy_pairing_center_of_mass_structure,
            seed_metadata.legacy_pairing_beta_initialization,
            seed_metadata.legacy_pairing_mu_cdw_initialization,
            branch.analysis_role,
            "true",
            string(settings.dmrg.energy_tol),
            string(settings.dmrg.cutoff),
            string(settings.dmrg.mu_density_tol),
            string(settings.convergence.density_tol),
            string(settings.convergence.field_abs_tol),
            string(settings.convergence.field_rel_tol),
            string(settings.convergence.variational_energy_tol),
            string(settings.dmrg.mu_bracket_step),
            string(settings.dmrg.mu_bracket_growth),
            string(settings.dmrg.mu_warm_start_noise),
            string(settings.convergence.period2_oscillation_cosine_max),
            string(settings.convergence.period2_two_step_ratio_max),
            string(settings.convergence.slow_mode_cosine_min),
        ), '\t'))
    end
end

length(unique(model_fingerprints)) == 1 || error(
    "square seed-pilot branches do not share one model fingerprint",
)
length(unique(numerical_fingerprints)) == 1 || error(
    "square seed-pilot branches do not share one numerical fingerprint",
)
length(unique(seed_fingerprints)) == length(branches) || error(
    "square seed-pilot branches do not have distinct seed fingerprints",
)

println("branch_count=$(length(branches))")
println("common_random_seed=$COMMON_RANDOM_SEED")
println("model_fingerprint=$(only(unique(model_fingerprints)))")
println("numerical_fingerprint=$(only(unique(numerical_fingerprints)))")
println("manifest_path=$manifest_path")
