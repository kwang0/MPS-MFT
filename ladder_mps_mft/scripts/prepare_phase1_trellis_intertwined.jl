#!/usr/bin/env julia

# Reuse the frozen legacy-derived shape; do not change historical campaigns.
include("prepare_phase1_square_positive_v.jl")

const TRELLIS_INTERTWINED_POINTS = ((1.2,0.0), (1.2,0.2), (1.4,0.0), (1.4,0.2))
const TRELLIS_INTERTWINED_REGISTRY_SHA = "2209bd2ca3c1ad02c0e542d1a9d63ecf90fdfa49120ad9cc3af599a5b4bc1f0e"
const TRELLIS_INTERTWINED_FAMILY = "intertwined_lambda16"

function validate_trellis_intertwined(settings)
    m,d,c,r,x = settings.model,settings.dmrg,settings.convergence,settings.run,settings.mixing
    m.geometry == :trellis && m.trellis_cell == :two_ladder || error("two-ladder trellis required")
    (m.L,m.t,m.U,m.tp,m.tau0,m.tau1,m.density,m.r_range) ==
        (64,1.,8.,.1,.1,.1,.9375,4) || error("intertwined model contract changed")
    (m.t0,m.V) in TRELLIS_INTERTWINED_POINTS || error("unexpected intertwined coordinate")
    m.ep_mode == :exact && m.ep_signed < 0 || error("exact bound-pair registry row required")
    LadderMPSMFT.sha256_file(m.ep_source) == TRELLIS_INTERTWINED_REGISTRY_SHA || error("E_p registry changed")
    d.maxdim == 200 && d.energy_tol == c.dmrg_sweep_energy_tol == 1e-7 || error("chi/inner-DMRG contract changed")
    r.max_iterations == c.probe_iterations == 60 && c.minimum_iterations == 40 && c.stable_iterations == 10 || error("cell-sweep contract changed")
    c.unmixed_cycle_probe && x.method == :linear && !x.adaptive || error("raw updates required")
    x.minimum_damping == x.damping == x.maximum_damping == 1 || error("damping forbidden")
    c.channel_residuals && c.channel_noise_floor == 5e-7 && c.variational_energy_tol == 1e-7 || error("channel/energy gates changed")
    c.field_abs_tol == 1e-7 && c.field_rel_tol == 1e-4 && c.accepted_periods == [1] || error("stationary-field contract changed")
    r.parent_checkpoint === nothing && r.resume_checkpoint === nothing || error("fresh MPS required")
    r.random_seed == 1404 && r.quick_diagnostics && r.full_pair_correlations || error("initialization/measurement contract changed")
    return settings
end

function prepare_trellis_intertwined(base_path, recipe_path, control_run, full_run, run_id)
    occursin(r"^[A-Za-z0-9_.-]+$",run_id) || error("unsafe run ID")
    base = validate_trellis_intertwined(load_settings(base_path))
    base.run.inherit_from === nothing || error("base must not inherit another seed")
    raw_base = TOML.parsefile(base_path)
    recipe = positive_v_recipe(recipe_path)
    control_run,full_run = abspath(control_run),abspath(full_run)
    for name in ("configs","seeds")
        path = joinpath(control_run,name)
        isdir(path) && !isempty(readdir(path)) && error("refusing to overwrite $path")
    end
    ispath(joinpath(control_run,"manifest.tsv")) && error("manifest already exists")
    ispath(joinpath(full_run,"results")) && !isempty(readdir(joinpath(full_run,"results"))) && error("full results already exist")
    mkpath(joinpath(control_run,"configs")); mkpath(joinpath(control_run,"seeds"))
    mkpath(joinpath(full_run,"results"))
    rows = NamedTuple[]
    for (t0,V) in TRELLIS_INTERTWINED_POINTS
        point = "t0$(round(Int,10t0))_$(V == 0 ? "v000" : "vp02")"
        label = "trellis__two_ladder__$(TRELLIS_INTERTWINED_FAMILY)_$(point)_chi200_raw60"
        raw = deepcopy(raw_base)
        raw["model"]["t0"] = t0; raw["model"]["V"] = V
        raw["model"]["mu_initial"] = V == 0 ? 1.65 : 2.2
        raw["pair_binding"]["registry"] = base.model.ep_source
        raw["run"]["branch_label"] = TRELLIS_INTERTWINED_FAMILY
        raw["run"]["seed_label"] = TRELLIS_INTERTWINED_FAMILY
        raw["run"]["output_directory"] = joinpath(full_run,"results",label)
        config_path = joinpath(control_run,"configs",label*".segment-001.toml")
        seed_path = joinpath(control_run,"seeds",label*".h5")
        open(io -> TOML.print(io,raw),config_path,"w")
        model = validate_trellis_intertwined(load_settings(config_path)).model
        correlations = intertwined_correlations(recipe,model,16)
        # Same template in each ladder's local rung coordinates, as in the
        # existing trellis campaign. A/B fields use their distinct reciprocal maps.
        fields = trellis_mean_fields([correlations,correlations],model)
        h5open(seed_path,"w") do file
            file["artifact_kind"] = "trellis_correlation_seed"
            file["model_fingerprint"] = LadderMPSMFT.model_fingerprint(model)
            file["chemical_potential"] = model.mu_initial
            file["spatial_ladders"] = 2
            LadderMPSMFT._write_correlations(create_group(file,"template_correlations"),correlations)
            for (name,field) in zip(("A","B"),fields)
                LadderMPSMFT._write_fields(create_group(file,"ladders/$name/fields"),field)
                norm(field.alpha) > 1e-8 || error("missing pairing field")
                norm(field.mu_cdw[2,:]-field.mu_cdw[1,:]) > 1e-6 || error("missing spin field")
            end
            LadderMPSMFT._write_dict(create_group(file,"seed_provenance"),Dict(
                "family"=>TRELLIS_INTERTWINED_FAMILY,"fresh_mps"=>true,"is_scf_solution"=>false,
                "random_seed_each_ladder"=>1404,"source_kind"=>"incomplete_legacy_shape_recipe",
                "recipe_sha256"=>POSITIVE_V_RECIPE_SHA,"legacy_source_sha256"=>POSITIVE_V_LEGACY_SHA,
                "source_geometry"=>recipe["source_geometry"],"source_completed"=>false,
                "source_mean_density"=>recipe["source_mean_density"],
                "definition"=>"legacy-derived period-16 charge/pair envelope at spin walls; identical local-rung template on A/B; target rectangular trellis kernel",
                "pairing_phase"=>"same sign between charge peaks and same global phase on A/B",
                "charge_pairing_wavelength_rungs"=>16,"spin_envelope_wavelength_rungs"=>32,
                "local_rung_coordinate"=>"i-1/2","ladder_origins_rungs"=>[0.,-.5],
                "relative_shift_local_rungs"=>0.,"target_geometry"=>"trellis","trellis_cell"=>"two_ladder",
                "target_t0"=>t0,"target_V"=>V,"target_tau0"=>model.tau0,"target_tau1"=>model.tau1,
                "target_density_each_ladder"=>model.density,"target_ep"=>model.ep,"ep_mode"=>"exact",
                "charge_amplitude"=>recipe["charge_amplitude"],"spin_sz_amplitude"=>recipe["spin_sz_amplitude"],
                "pair_rung_mean"=>recipe["pair_rung_mean"],"pair_rung_modulation"=>recipe["pair_rung_modulation"]))
        end
        raw["run"]["inherit_from"] = seed_path
        raw["run"]["inherit_sha256"] = LadderMPSMFT.sha256_file(seed_path)
        open(io -> TOML.print(io,raw),config_path,"w")
        settings = validate_trellis_intertwined(load_settings(config_path))
        h5open(seed_path,"r") do file
            for (name,field) in zip(("A","B"),fields)
                saved = LadderMPSMFT._read_fields(file["ladders/$name/fields"])
                all(getfield(saved,k)==getfield(field,k) for k in fieldnames(FieldState)) || error("seed readback mismatch")
            end
        end
        push!(rows,(label,config=config_path,config_sha256=LadderMPSMFT.sha256_file(config_path),
            geometry="trellis",trellis_cell="two_ladder",spatial_ladders=2,family=TRELLIS_INTERTWINED_FAMILY,
            U=model.U,t0,V,tau0=model.tau0,tau1=model.tau1,L=model.L,density=model.density,chi=settings.dmrg.maxdim,
            charge_pairing_wavelength=16,spin_envelope_wavelength=32,
            ep_signed=model.ep_signed,ep_denominator=model.ep,ep_mode=String(model.ep_mode),
            seed=seed_path,seed_sha256=settings.run.inherit_sha256,recipe_sha256=POSITIVE_V_RECIPE_SHA,
            model_fingerprint=LadderMPSMFT.model_fingerprint(model),
            numerical_fingerprint=LadderMPSMFT.numerical_fingerprint(settings),
            implementation_sha256=implementation_fingerprint(settings),ep_source_sha256=TRELLIS_INTERTWINED_REGISTRY_SHA,
            full_output_directory=settings.run.output_directory,
            stateless_output_directory=joinpath(control_run,"results",label)))
    end
    length(rows)==4 && length(unique(r.model_fingerprint for r in rows))==4 || error("four distinct model points required")
    length(unique(r.numerical_fingerprint for r in rows))==1 || error("numerical controls differ")
    open(joinpath(control_run,"manifest.tsv"),"w") do io
        println(io,join(String.(keys(first(rows))),'\t'))
        for row in rows; println(io,join(values(row),'\t')); end
    end
    open(joinpath(control_run,"seed_contract.toml"),"w") do io
        TOML.print(io,Dict("run_id"=>run_id,"branches"=>4,"geometry"=>"trellis","trellis_cell"=>"two_ladder",
            "family"=>TRELLIS_INTERTWINED_FAMILY,"t0_values"=>[1.2,1.4],"V_values"=>[0.,.2],
            "charge_pairing_wavelength_rungs"=>16,"spin_envelope_wavelength_rungs"=>32,
            "recipe_sha256"=>POSITIVE_V_RECIPE_SHA,"registry_sha256"=>TRELLIS_INTERTWINED_REGISTRY_SHA,
            "fresh_mps"=>true,"interpolated_ep"=>false,"maximum_cell_sweeps"=>60,"minimum_cell_sweeps"=>40,
            "stable_sweeps"=>10,"maximum_ladder_solves"=>480,"chi"=>200,"accepted_iteration_periods"=>[1],
            "update"=>"simultaneous raw map; no Anderson, damping or external pinning",
            "density_constraint"=>"fixed n=0.9375 on each ladder"))
    end
    println("Prepared four two-ladder trellis intertwined starts: ",joinpath(control_run,"manifest.tsv"))
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==5 || error("usage: BASE RECIPE CONTROL_RUN FULL_RUN RUN_ID")
    prepare_trellis_intertwined(ARGS...)
end
