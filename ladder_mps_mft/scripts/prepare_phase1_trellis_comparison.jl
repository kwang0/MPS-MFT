#!/usr/bin/env julia

isdefined(@__MODULE__, :two_basin_references) || include("prepare_phase1_two_basin_grid.jl")

function validate_trellis_campaign(settings)
    m, d, c, r = settings.model, settings.dmrg, settings.convergence, settings.run
    m.geometry == :trellis && m.L == 64 && m.r_range == 4 || error("expected L64 range4 trellis")
    (m.t, m.U, m.t0, m.tau0, m.tau1, m.V, m.density) == (1.,8.,1.,.1,.1,0.,.9375) || error("trellis parameters changed")
    m.ep_mode == :exact && m.ep_signed < 0 || error("exact bound-pair registry row required")
    d.maxdim == 200 && d.energy_tol == c.dmrg_sweep_energy_tol == 1e-7 || error("chi/inner-DMRG contract changed")
    r.max_iterations == c.probe_iterations == 60 && c.minimum_iterations == 40 && c.stable_iterations == 10 || error("cell-sweep contract changed")
    c.channel_residuals && c.channel_noise_floor == 5e-7 && c.variational_energy_tol == 1e-7 || error("channel/energy gates changed")
    c.field_abs_tol == 1e-7 && c.field_rel_tol == 1e-4 && c.accepted_periods == [1] || error("stationary-field contract changed")
    r.parent_checkpoint === nothing && r.resume_checkpoint === nothing || error("fresh MPS required")
    return settings
end

function prepare_trellis_comparison(base_path, reference_path, control_run, full_run, run_id)
    occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID")
    base = validate_trellis_campaign(load_settings(base_path))
    raw_base = TOML.parsefile(base_path)
    stripe, pairing = two_basin_references(reference_path)
    templates = (stripe=blend_correlations(stripe, pairing), pairing=blend_correlations(pairing, stripe))
    control_run, full_run = abspath(control_run), abspath(full_run)
    for name in ("configs", "seeds")
        path = joinpath(control_run, name)
        isdir(path) && !isempty(readdir(path)) && error("refusing to overwrite $path")
    end
    ispath(joinpath(control_run, "manifest.tsv")) && error("manifest already exists")
    mkpath(joinpath(control_run, "configs")); mkpath(joinpath(control_run, "seeds"))
    mkpath(joinpath(full_run, "results"))
    rows = NamedTuple[]
    for cell in (:one_ladder, :two_ladder), family in (:stripe, :pairing)
        label = "trellis__$(cell)__$(family)_eps005_chi200_raw60"
        raw = deepcopy(raw_base)
        raw["model"]["trellis_cell"] = String(cell)
        raw["pair_binding"]["registry"] = joinpath(LadderMPSMFT.PROJECT_ROOT, "data", "E_p_values.csv")
        raw["run"]["branch_label"] = "$(cell)_$(family)_weak_other"
        raw["run"]["seed_label"] = "$(family)_eps005"
        raw["run"]["random_seed"] = 1404
        raw["run"]["output_directory"] = joinpath(full_run, "results", label)
        config_path = joinpath(control_run, "configs", label * ".segment-001.toml")
        seed_path = joinpath(control_run, "seeds", label * ".h5")
        open(config_path, "w") do io; TOML.print(io, raw); end
        model = load_settings(config_path).model
        count = LadderMPSMFT.trellis_ladders(model)
        correlations = getproperty(templates, family)
        fields = trellis_mean_fields(fill(correlations, count), model)
        h5open(seed_path, "w") do file
            file["artifact_kind"] = "trellis_correlation_seed"
            file["model_fingerprint"] = LadderMPSMFT.model_fingerprint(model)
            file["chemical_potential"] = model.mu_initial
            file["spatial_ladders"] = count
            LadderMPSMFT._write_correlations(create_group(file, "template_correlations"), correlations)
            for index in 1:count
                LadderMPSMFT._write_fields(create_group(file, "ladders/$(index == 1 ? "A" : "B")/fields"), fields[index])
                norm(fields[index].alpha) > 1e-8 || error("missing pairing perturbation")
                norm(fields[index].mu_cdw[2,:] - fields[index].mu_cdw[1,:]) > 1e-6 || error("missing spin perturbation")
            end
            LadderMPSMFT._write_dict(create_group(file, "seed_provenance"), Dict(
                "family" => String(family), "epsilon" => TWO_BASIN_EPSILON,
                "definition" => "0.95 primary + 0.05 competing measured correlations; identical template on each spatial ladder; target trellis kernel",
                "reference_bundle_sha256" => TWO_BASIN_REFERENCE_SHA,
                "stripe_source_sha256" => TWO_BASIN_SOURCE_HASHES.stripe,
                "pairing_source_sha256" => TWO_BASIN_SOURCE_HASHES.pairing,
                "fresh_mps" => true, "random_seed_each_ladder" => 1404,
                "is_scf_solution" => false,
            ))
        end
        raw["run"]["inherit_from"] = seed_path
        raw["run"]["inherit_sha256"] = LadderMPSMFT.sha256_file(seed_path)
        open(config_path, "w") do io; TOML.print(io, raw); end
        settings = validate_trellis_campaign(load_settings(config_path))
        push!(rows, (
            label, config=config_path, config_sha256=LadderMPSMFT.sha256_file(config_path),
            geometry="trellis", trellis_cell=String(cell), spatial_ladders=count, family=String(family),
            U=model.U, t0=model.t0, tau0=model.tau0, tau1=model.tau1, V=model.V,
            L=model.L, density=model.density, chi=settings.dmrg.maxdim,
            ep_signed=model.ep_signed, ep_denominator=model.ep, ep_mode=String(model.ep_mode),
            seed=seed_path, seed_sha256=settings.run.inherit_sha256, reference_sha256=TWO_BASIN_REFERENCE_SHA,
            model_fingerprint=LadderMPSMFT.model_fingerprint(model),
            numerical_fingerprint=LadderMPSMFT.numerical_fingerprint(settings),
            implementation_sha256=implementation_fingerprint(settings),
            ep_source_sha256=LadderMPSMFT.sha256_file(model.ep_source),
            full_output_directory=settings.run.output_directory,
            stateless_output_directory=joinpath(control_run, "results", label),
        ))
    end
    open(joinpath(control_run, "manifest.tsv"), "w") do io
        println(io, join(String.(keys(first(rows))), '\t'))
        for row in rows; println(io, join(values(row), '\t')); end
    end
    open(joinpath(control_run, "seed_contract.toml"), "w") do io
        TOML.print(io, Dict("run_id" => run_id, "branches" => 4, "epsilon" => TWO_BASIN_EPSILON,
            "reference_sha256" => TWO_BASIN_REFERENCE_SHA, "fresh_mps" => true,
            "maximum_cell_sweeps" => 60, "minimum_cell_sweeps" => 40, "stable_sweeps" => 10,
            "maximum_ladder_solves" => 360, "density_constraint" => "fixed n=0.9375 on each ladder",
            "update" => "simultaneous raw map in fixed coordinates; A/B are spatial states",
            "accepted_iteration_periods" => [1], "chi" => base.dmrg.maxdim))
    end
    println("Prepared four trellis branches, manifest=$(joinpath(control_run, "manifest.tsv"))")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 5 || error("usage: julia --project=. scripts/prepare_phase1_trellis_comparison.jl BASE REFERENCES CONTROL FULL RUN_ID")
    prepare_trellis_comparison(ARGS...)
end
