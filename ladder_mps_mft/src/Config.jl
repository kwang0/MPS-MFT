const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

_table(raw, key) = haskey(raw, key) ? raw[key] : Dict{String,Any}()
_value(table, key, default) = haskey(table, key) ? table[key] : default

function _project_path(path::AbstractString)
    isempty(path) && return ""
    return isabspath(path) ? normpath(path) : normpath(joinpath(PROJECT_ROOT, path))
end

function load_settings(path::AbstractString)
    config_path = abspath(path)
    isfile(config_path) || throw(ArgumentError("configuration not found: $config_path"))
    raw = TOML.parsefile(config_path)
    model_raw = _table(raw, "model")
    ep_raw = _table(raw, "pair_binding")
    dmrg_raw = _table(raw, "dmrg")
    mixing_raw = _table(raw, "mixing")
    convergence_raw = _table(raw, "convergence")
    runtime_raw = _table(raw, "runtime")
    run_raw = _table(raw, "run")

    L = Int(_value(model_raw, "L", 64))
    U = Float64(_value(model_raw, "U", 8.0))
    V = Float64(_value(model_raw, "V", 0.0))
    t0 = Float64(_value(model_raw, "t0", 1.0))
    tp = Float64(_value(model_raw, "tp", 0.1))
    density = Float64(_value(model_raw, "density", 0.9375))
    registry_path = _project_path(String(_value(ep_raw, "registry", "data/E_p_values.csv")))
    allow_unbound = Bool(_value(run_raw, "allow_unbound_ep", false))
    selection = lookup_ep(
        registry_path;
        L,
        U,
        V,
        t0,
        density,
        tp,
        require_bound=!allow_unbound,
        allow_interpolation=Bool(_value(ep_raw, "allow_interpolation", false)),
    )

    model = ModelSettings(;
        L=L,
        t=Float64(_value(model_raw, "t", 1.0)),
        U=U,
        V=V,
        t0=t0,
        tp=tp,
        density=density,
        mu_initial=Float64(_value(model_raw, "mu_initial", 0.0)),
        r_range=Int(_value(model_raw, "r_range", 4)),
        geometry=normalize_geometry(_value(model_raw, "geometry", "cubic_frustrated")),
        ep=selection.denominator,
        ep_signed=selection.record.E_p,
        ep_source=selection.source_path,
        ep_mode=selection.mode,
        ep_t0_lower=selection.lower_record.t0,
        ep_t0_upper=selection.upper_record.t0,
        ep_lower_signed=selection.lower_record.E_p,
        ep_upper_signed=selection.upper_record.E_p,
        ep_interpolation_weight=selection.interpolation_weight,
        ep_lower_chi=selection.lower_record.chi,
        ep_upper_chi=selection.upper_record.chi,
    )
    dmrg = DMRGSettings(;
        nsweeps=Int(_value(dmrg_raw, "nsweeps", 12)),
        maxdim=Int(_value(dmrg_raw, "maxdim", 200)),
        cutoff=Float64(_value(dmrg_raw, "cutoff", 1e-10)),
        energy_tol=Float64(_value(dmrg_raw, "energy_tol", 1e-8)),
        eigsolve_krylovdim=Int(_value(dmrg_raw, "eigsolve_krylovdim", 8)),
        max_time_seconds=Float64(_value(dmrg_raw, "max_time_seconds", 23.5 * 3600)),
        output_level=Int(_value(dmrg_raw, "output_level", 1)),
        mu_density_tol=Float64(_value(dmrg_raw, "mu_density_tol", 2e-4)),
        mu_max_iterations=Int(_value(dmrg_raw, "mu_max_iterations", 16)),
        mu_bracket_step=Float64(_value(dmrg_raw, "mu_bracket_step", 0.05)),
        mu_bracket_growth=Float64(_value(dmrg_raw, "mu_bracket_growth", 2.0)),
        mu_interval_tol=Float64(_value(dmrg_raw, "mu_interval_tol", 1e-6)),
    )
    mixing = MixingSettings(;
        method=Symbol(lowercase(String(_value(mixing_raw, "method", "anderson")))),
        damping=Float64(_value(mixing_raw, "damping", 0.5)),
        minimum_damping=Float64(_value(mixing_raw, "minimum_damping", 0.05)),
        maximum_damping=Float64(_value(mixing_raw, "maximum_damping", 0.8)),
        memory=Int(_value(mixing_raw, "memory", 5)),
        regularization=Float64(_value(mixing_raw, "regularization", 1e-10)),
        adaptive=Bool(_value(mixing_raw, "adaptive", true)),
    )
    convergence = ConvergenceSettings(;
        field_abs_tol=Float64(_value(convergence_raw, "field_abs_tol", 1e-6)),
        field_rel_tol=Float64(_value(convergence_raw, "field_rel_tol", 5e-3)),
        density_tol=Float64(_value(convergence_raw, "density_tol", 1e-5)),
        variational_energy_tol=Float64(_value(convergence_raw, "variational_energy_tol", 1e-7)),
        hamiltonian_identity_tol=Float64(_value(convergence_raw, "hamiltonian_identity_tol", 1e-9)),
        effective_energy_consistency_tol=Float64(_value(convergence_raw, "effective_energy_consistency_tol", 1e-6)),
        stable_iterations=Int(_value(convergence_raw, "stable_iterations", 2)),
        max_period=Int(_value(convergence_raw, "max_period", 8)),
        period_repeats=Int(_value(convergence_raw, "period_repeats", 3)),
        period_abs_tol=Float64(_value(convergence_raw, "period_abs_tol", 2e-6)),
        period_rel_tol=Float64(_value(convergence_raw, "period_rel_tol", 1e-2)),
        unmixed_cycle_probe=Bool(_value(convergence_raw, "unmixed_cycle_probe", true)),
        probe_max_period=Int(_value(convergence_raw, "probe_max_period", 2)),
        probe_iterations=Int(_value(convergence_raw, "probe_iterations", 20)),
        accepted_periods=sort(unique(Int.(_value(convergence_raw, "accepted_periods", [1, 2])))),
        orbit_bulk_fraction=Float64(_value(convergence_raw, "orbit_bulk_fraction", 0.5)),
        cycle_action=Symbol(lowercase(String(_value(convergence_raw, "cycle_action", "stop")))),
        stagnation_window=Int(_value(convergence_raw, "stagnation_window", 10)),
        stagnation_min_relative_improvement=Float64(_value(convergence_raw, "stagnation_min_relative_improvement", 1e-2)),
        divergence_factor=Float64(_value(convergence_raw, "divergence_factor", 8.0)),
    )
    runtime = RuntimeSettings(;
        backend=Symbol(lowercase(String(_value(runtime_raw, "backend", "cpu")))),
        tensor_scalar_type=Symbol(lowercase(String(_value(runtime_raw, "tensor_scalar_type", "float64")))),
        blas_threads=Int(_value(runtime_raw, "blas_threads", 1)),
        strided_threads=Int(_value(runtime_raw, "strided_threads", 1)),
        threaded_blocksparse=Bool(_value(runtime_raw, "threaded_blocksparse", true)),
        conserve_sz=Bool(_value(runtime_raw, "conserve_sz", true)),
        conserve_nfparity=Bool(_value(runtime_raw, "conserve_nfparity", true)),
    )
    run = RunSettings(;
        output_directory=_project_path(String(_value(run_raw, "output_directory", "output"))),
        branch_label=String(_value(run_raw, "branch_label", "independent")),
        preparation=String(_value(run_raw, "preparation", "independent_seed")),
        direction=String(_value(run_raw, "direction", "none")),
        seed_label=String(_value(run_raw, "seed_label", "seed_1")),
        random_seed=Int(_value(run_raw, "random_seed", 1)),
        initial_seed=Symbol(lowercase(String(_value(run_raw, "initial_seed", "pairing")))),
        initial_amplitude=Float64(_value(run_raw, "initial_amplitude", 1e-3)),
        initial_seed_protocol=Symbol(lowercase(String(_value(run_raw, "initial_seed_protocol", "legacy")))),
        initial_mode_number=Int(_value(run_raw, "initial_mode_number", 0)),
        initial_mode_phase_pi=Float64(_value(run_raw, "initial_mode_phase_pi", 0.0)),
        initial_pairing_form_factor=Symbol(lowercase(String(_value(
            run_raw,
            "initial_pairing_form_factor",
            "onsite_s",
        )))),
        initial_leg_parity=Symbol(lowercase(String(_value(run_raw, "initial_leg_parity", "auto")))),
        initial_stripe_charge_to_spin_ratio=Float64(_value(
            run_raw,
            "initial_stripe_charge_to_spin_ratio",
            0.2,
        )),
        initial_stripe_pairing_to_spin_ratio=Float64(_value(
            run_raw,
            "initial_stripe_pairing_to_spin_ratio",
            1.0,
        )),
        inherit_from=haskey(run_raw, "inherit_from") ? _project_path(String(run_raw["inherit_from"])) : nothing,
        inherit_sha256=haskey(run_raw, "inherit_sha256") ? lowercase(String(run_raw["inherit_sha256"])) : nothing,
        parent_checkpoint=haskey(run_raw, "parent_checkpoint") ? _project_path(String(run_raw["parent_checkpoint"])) : nothing,
        parent_sha256=haskey(run_raw, "parent_sha256") ? lowercase(String(run_raw["parent_sha256"])) : nothing,
        parent_orbit_phase=haskey(run_raw, "parent_orbit_phase") ? Int(run_raw["parent_orbit_phase"]) : nothing,
        resume_checkpoint=haskey(run_raw, "resume_checkpoint") ? _project_path(String(run_raw["resume_checkpoint"])) : nothing,
        resume_sha256=haskey(run_raw, "resume_sha256") ? lowercase(String(run_raw["resume_sha256"])) : nothing,
        max_iterations=Int(_value(run_raw, "max_iterations", 80)),
        save_every=Int(_value(run_raw, "save_every", 1)),
        require_accepted_solution=Bool(_value(
            run_raw,
            "require_accepted_solution",
            _value(run_raw, "require_fixed_point", true),
        )),
        allow_unbound_ep=allow_unbound,
        quick_diagnostics=Bool(_value(run_raw, "quick_diagnostics", true)),
        full_pair_correlations=Bool(_value(run_raw, "full_pair_correlations", false)),
    )
    settings = ProjectSettings(; model, dmrg, mixing, convergence, runtime, run, config_path)
    validate_settings(settings)
    return settings
end

function validate_settings(settings::ProjectSettings)
    model = settings.model
    model.L >= 2 || throw(ArgumentError("model.L must be at least 2 rungs"))
    0 < model.density <= 2 || throw(ArgumentError("density must lie in (0,2] per site"))
    model.r_range >= 0 || throw(ArgumentError("r_range must be nonnegative"))
    model.tp >= 0 || throw(ArgumentError("tp must be nonnegative"))
    model.ep > 0 || throw(ArgumentError("the selected |E_p| must be positive"))
    settings.runtime.backend in (:cpu, :gpu) || throw(ArgumentError("runtime.backend must be cpu or gpu"))
    settings.runtime.tensor_scalar_type in (:float32, :float64) || throw(ArgumentError(
        "runtime.tensor_scalar_type must be float32 or float64",
    ))
    settings.runtime.blas_threads >= 1 || throw(ArgumentError("blas_threads must be positive"))
    settings.runtime.strided_threads >= 1 || throw(ArgumentError("strided_threads must be positive"))
    if settings.runtime.backend == :gpu
        !settings.runtime.conserve_sz && !settings.runtime.conserve_nfparity || throw(ArgumentError(
            "the validated GPU DMRG path requires conserve_sz=false and conserve_nfparity=false; " *
            "QN block-sparse CUDA is not the production backend",
        ))
        !settings.runtime.threaded_blocksparse || throw(ArgumentError(
            "threaded_blocksparse must be false for the dense GPU backend",
        ))
        settings.runtime.blas_threads == 1 && settings.runtime.strided_threads == 1 || throw(ArgumentError(
            "GPU runs require blas_threads=1 and strided_threads=1 to avoid CPU oversubscription",
        ))
    end
    settings.dmrg.nsweeps >= 1 || throw(ArgumentError("nsweeps must be positive"))
    settings.dmrg.maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    settings.dmrg.mu_density_tol > 0 || throw(ArgumentError("mu_density_tol must be positive"))
    settings.dmrg.mu_max_iterations >= 1 || throw(ArgumentError("mu_max_iterations must be positive"))
    settings.dmrg.mu_bracket_step > 0 || throw(ArgumentError("mu_bracket_step must be positive"))
    settings.dmrg.mu_bracket_growth > 1 || throw(ArgumentError("mu_bracket_growth must exceed 1"))
    settings.mixing.method in (:linear, :anderson) || throw(ArgumentError("mixing.method must be linear or anderson"))
    0 < settings.mixing.minimum_damping <= settings.mixing.damping <= settings.mixing.maximum_damping <= 1 ||
        throw(ArgumentError("mixing damping values must satisfy 0 < min <= damping <= max <= 1"))
    settings.mixing.memory >= 1 || throw(ArgumentError("Anderson memory must be positive"))
    settings.convergence.max_period >= 1 || throw(ArgumentError("max_period must be positive"))
    settings.convergence.hamiltonian_identity_tol > 0 || throw(ArgumentError("hamiltonian_identity_tol must be positive"))
    settings.convergence.effective_energy_consistency_tol > 0 || throw(ArgumentError("effective_energy_consistency_tol must be positive"))
    settings.convergence.period_repeats >= 2 || throw(ArgumentError("period_repeats must be at least 2"))
    settings.convergence.probe_max_period >= 2 || throw(ArgumentError("probe_max_period must be at least 2"))
    settings.convergence.probe_max_period <= settings.convergence.max_period || throw(ArgumentError(
        "probe_max_period cannot exceed max_period",
    ))
    !isempty(settings.convergence.accepted_periods) || throw(ArgumentError("accepted_periods cannot be empty"))
    1 in settings.convergence.accepted_periods || throw(ArgumentError("accepted_periods must include period 1"))
    all(period -> 1 <= period <= settings.convergence.probe_max_period, settings.convergence.accepted_periods) ||
        throw(ArgumentError("accepted_periods must lie between 1 and probe_max_period"))
    0 < settings.convergence.orbit_bulk_fraction <= 1 || throw(ArgumentError(
        "orbit_bulk_fraction must lie in (0,1]",
    ))
    minimum_probe_iterations = settings.convergence.probe_max_period *
        (settings.convergence.period_repeats + 1) + 1
    (!settings.convergence.unmixed_cycle_probe || settings.convergence.probe_iterations >= minimum_probe_iterations) ||
        throw(ArgumentError(
            "probe_iterations must be at least $minimum_probe_iterations for all-phase orbit validation",
        ))
    settings.convergence.cycle_action in (:stop, :continue) || throw(ArgumentError("cycle_action must be stop or continue"))
    settings.run.max_iterations >= 1 || throw(ArgumentError("run.max_iterations must be positive"))
    settings.run.save_every >= 1 || throw(ArgumentError("run.save_every must be positive"))
    settings.run.initial_seed in (
        :pairing,
        :legacy_pairing,
        :sdw,
        :cdw,
        :stripe,
        :stripe_pairing,
        :zero,
    ) ||
        throw(ArgumentError(
            "initial_seed must be pairing, legacy_pairing, sdw, cdw, stripe, " *
            "stripe_pairing, or zero",
        ))
    isfinite(settings.run.initial_amplitude) && settings.run.initial_amplitude >= 0 ||
        throw(ArgumentError("initial_amplitude must be finite and nonnegative"))
    settings.run.initial_seed_protocol in (:legacy, :matched_mode) || throw(ArgumentError(
        "initial_seed_protocol must be legacy or matched_mode",
    ))
    0 <= settings.run.initial_mode_number <= model.L - 1 || throw(ArgumentError(
        "initial_mode_number must lie between 0 and L-1",
    ))
    isfinite(settings.run.initial_mode_phase_pi) || throw(ArgumentError(
        "initial_mode_phase_pi must be finite",
    ))
    settings.run.initial_pairing_form_factor in (:onsite_s, :rung_s, :leg_s, :extended_s, :d_wave) ||
        throw(ArgumentError(
            "initial_pairing_form_factor must be onsite_s, rung_s, leg_s, extended_s, or d_wave",
        ))
    settings.run.initial_leg_parity in (:auto, :even, :odd) || throw(ArgumentError(
        "initial_leg_parity must be auto, even, or odd",
    ))
    isfinite(settings.run.initial_stripe_charge_to_spin_ratio) &&
        settings.run.initial_stripe_charge_to_spin_ratio > 0 || throw(ArgumentError(
            "initial_stripe_charge_to_spin_ratio must be finite and positive",
        ))
    isfinite(settings.run.initial_stripe_pairing_to_spin_ratio) &&
        settings.run.initial_stripe_pairing_to_spin_ratio >= 0 || throw(ArgumentError(
            "initial_stripe_pairing_to_spin_ratio must be finite and nonnegative",
        ))
    stripe_seed = settings.run.initial_seed in (:stripe, :stripe_pairing)
    legacy_pairing_seed = settings.run.initial_seed == :legacy_pairing
    legacy_pairing_seed && settings.run.initial_seed_protocol != :matched_mode &&
        throw(ArgumentError(
            "legacy_pairing requires initial_seed_protocol=matched_mode so its complete " *
            "field norm is matched to the other pilot seeds",
        ))
    if legacy_pairing_seed
        settings.run.initial_mode_number == 0 || throw(ArgumentError(
            "legacy_pairing is translation-invariant in the center-of-mass coordinate; " *
            "use initial_mode_number=0",
        ))
        settings.run.initial_leg_parity == :auto || throw(ArgumentError(
            "legacy_pairing draws all relative leg-pair classes; use initial_leg_parity=auto",
        ))
        settings.run.initial_pairing_form_factor == :onsite_s || throw(ArgumentError(
            "legacy_pairing draws a mixed relative-bond pairing form factor; leave " *
            "initial_pairing_form_factor=onsite_s as the unused sentinel",
        ))
    end
    stripe_seed && settings.run.initial_seed_protocol != :matched_mode && throw(ArgumentError(
        "stripe and stripe_pairing seeds require initial_seed_protocol=matched_mode",
    ))
    if stripe_seed
        settings.run.initial_leg_parity == :auto || throw(ArgumentError(
            "stripe seeds fix odd spin and even charge leg parity; use initial_leg_parity=auto",
        ))
        1 <= settings.run.initial_mode_number || throw(ArgumentError(
            "stripe envelope mode must be positive",
        ))
        2 * settings.run.initial_mode_number <= model.L - 1 || throw(ArgumentError(
            "stripe charge harmonic 2m must not exceed L-1",
        ))
        settings.run.initial_seed == :stripe_pairing &&
            settings.run.initial_stripe_pairing_to_spin_ratio <= 0 && throw(ArgumentError(
                "stripe_pairing requires a positive pairing-to-spin ratio",
            ))
    end
    if settings.run.initial_seed_protocol == :matched_mode
        resolved_parity = resolved_initial_leg_parity(
            settings.run.initial_seed,
            settings.run.initial_leg_parity,
        )
        settings.run.initial_seed == :cdw && resolved_parity == :even &&
            settings.run.initial_mode_number == 0 && settings.run.initial_amplitude > 0 &&
            throw(ArgumentError(
                "a matched_mode CDW seed with even leg parity requires a nonzero mode; " *
                "the uniform even charge source is redundant with chemical-potential targeting",
            ))
        settings.run.initial_seed in (:pairing, :stripe_pairing) &&
            settings.run.initial_pairing_form_factor in (:leg_s, :extended_s, :d_wave) &&
            model.r_range < 1 && throw(ArgumentError(
                "the selected matched pairing form factor requires model.r_range >= 1",
            ))
    end
    lineage_sources = count(source -> source !== nothing, (
        settings.run.inherit_from,
        settings.run.parent_checkpoint,
        settings.run.resume_checkpoint,
    ))
    lineage_sources <= 1 || throw(ArgumentError(
        "set at most one of inherit_from, parent_checkpoint, or resume_checkpoint",
    ))
    settings.run.inherit_from !== nothing && settings.run.inherit_sha256 === nothing &&
        throw(ArgumentError("inherit_sha256 is required with inherit_from"))
    settings.run.inherit_from === nothing && settings.run.inherit_sha256 !== nothing &&
        throw(ArgumentError("inherit_from is required with inherit_sha256"))
    settings.run.parent_checkpoint !== nothing && settings.run.parent_sha256 === nothing &&
        throw(ArgumentError("parent_sha256 is required with parent_checkpoint"))
    settings.run.parent_checkpoint === nothing && settings.run.parent_sha256 !== nothing &&
        throw(ArgumentError("parent_checkpoint is required with parent_sha256"))
    settings.run.parent_orbit_phase !== nothing && settings.run.parent_checkpoint === nothing &&
        throw(ArgumentError("parent_checkpoint is required with parent_orbit_phase"))
    settings.run.parent_orbit_phase !== nothing && settings.run.parent_orbit_phase < 1 &&
        throw(ArgumentError("parent_orbit_phase must be positive"))
    settings.run.resume_checkpoint !== nothing && settings.run.resume_sha256 === nothing &&
        throw(ArgumentError("resume_sha256 is required with resume_checkpoint"))
    settings.run.resume_checkpoint === nothing && settings.run.resume_sha256 !== nothing &&
        throw(ArgumentError("resume_checkpoint is required with resume_sha256"))
    return settings
end
