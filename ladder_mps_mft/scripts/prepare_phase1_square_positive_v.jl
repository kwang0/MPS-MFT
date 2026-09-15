#!/usr/bin/env julia
include("prepare_phase1_two_basin_grid.jl")

const POSITIVE_V_RECIPE_SHA = "786fa2e8846820f42aabbabb625c3a645558d25e5388afa4058abf63cf834d35"
const POSITIVE_V_LEGACY_SHA = "8a8f5b917d11259d34ce773cb2860fb20d809f5dccfd252342f69a4596b9fec6"

function positive_v_recipe(path)
    LadderMPSMFT.sha256_file(path) == POSITIVE_V_RECIPE_SHA || error("intertwined recipe hash mismatch")
    recipe = TOML.parsefile(path)
    recipe["source_sha256"] == POSITIVE_V_LEGACY_SHA || error("legacy source changed")
    !recipe["source_completed"] || error("recipe must retain the incomplete-source flag")
    return recipe
end

"""Synthetic correlations used only to construct initial fields, not an MPS."""
function intertwined_correlations(recipe, model, wavelength::Integer)
    wavelength in (8,16) && model.L == 64 || error("expected L=64, charge period 8 or 16")
    L, n = model.L, model.density
    A, S = recipe["charge_amplitude"], recipe["spin_sz_amplitude"]
    P0, P1 = recipe["pair_rung_mean"], recipe["pair_rung_modulation"]
    0 < P1 < P0 && 0 < A < 1-n || error("invalid intertwined amplitudes")
    # Sites lie at x=i-1/2. Pair/charge maxima share spin antiphase walls.
    # The modulation has one global pairing phase: this is not an imposed PDW.
    charge(x) = n + A*cospi(2x/wavelength)
    pairing(x) = P0 - P1*cospi(2x/wavelength)
    pair, dn, up = (zeros(2L,2L) for _ in 1:3)
    ndn, nup = zeros(2L), zeros(2L)
    for i in 1:L, l in 0:1
        site = LadderMPSMFT.rung_leg_to_site(i,l)
        sz = S*(-1)^(i-1+l)*cospi((i-.5)/wavelength)
        ndn[site], nup[site] = charge(i-.5)/2-sz, charge(i-.5)/2+sz
    end
    for i in 1:L, j in 1:L, l in 0:1, lp in 0:1
        d = abs(j-i)
        d <= model.r_range || continue
        a, b = LadderMPSMFT.rung_leg_to_site(i,l), LadderMPSMFT.rung_leg_to_site(j,lp)
        kind = l == lp ? "same_leg" : "cross_leg"
        pair[a,b] = recipe["pair_"*kind][d+1] * pairing((i+j-1)/2)
        dn[a,b] = up[a,b] = recipe["exchange_"*kind][d+1]
    end
    dn[diagind(dn)] .= ndn; up[diagind(up)] .= nup
    all(0 .<= ndn .<= 1) && all(0 .<= nup .<= 1) || error("invalid on-site occupations")
    abs(mean(ndn+nup)-n) < 1e-14 || error("synthetic seed density mismatch")
    return CorrelationState(pair,dn,up,ndn,nup)
end

function prepare_square_positive_v(base_path, reference_path, recipe_path, control_run, full_run, run_id)
    occursin(r"^[A-Za-z0-9_.-]+$",run_id) || error("unsafe run ID")
    raw_base = TOML.parsefile(base_path)
    base = validate_raw_basin_contract(load_settings(base_path))
    (base.model.t0,base.model.V,base.model.r_range) == (1.2,.2,4) || error("expected square (1.2,+0.2), r_range=4")
    (base.run.max_iterations,base.convergence.minimum_iterations,base.convergence.stable_iterations) == (60,40,10) || error("expected 60/40/10 contract")
    recipe = positive_v_recipe(recipe_path)
    stripe,pairing = two_basin_references(reference_path)
    specifications = [
        (family="stripe_weak_other",wavelength=0,correlations=blend_correlations(stripe,pairing)),
        (family="pairing_weak_other",wavelength=0,correlations=blend_correlations(pairing,stripe)),
        (family="intertwined_lambda08",wavelength=8,correlations=intertwined_correlations(recipe,base.model,8)),
        (family="intertwined_lambda16",wavelength=16,correlations=intertwined_correlations(recipe,base.model,16))]
    control_run,full_run = abspath(control_run),abspath(full_run)
    for name in ("configs","seeds")
        dir = joinpath(control_run,name)
        isdir(dir) && !isempty(readdir(dir)) && error("refusing to overwrite $dir")
    end
    isfile(joinpath(control_run,"manifest.tsv")) && error("manifest already exists")
    mkpath(joinpath(control_run,"configs")); mkpath(joinpath(control_run,"seeds")); mkpath(joinpath(full_run,"results"))
    rows = NamedTuple[]
    for spec in specifications
        label = "square__$(spec.family)_t012_vp02_chi200_raw"
        raw = deepcopy(raw_base)
        seed_path = joinpath(control_run,"seeds",label*".h5")
        config_path = joinpath(control_run,"configs",label*".segment-001.toml")
        output = joinpath(full_run,"results",label)
        raw["pair_binding"]["registry"] = base.model.ep_source
        raw["run"]["output_directory"] = output
        raw["run"]["branch_label"] = spec.family
        raw["run"]["seed_label"] = spec.family
        raw["run"]["random_seed"] = 1404
        fields = LadderMPSMFT.mean_fields_from_correlations(spec.correlations,base.model)
        provenance = Dict{String,Any}(
            "family"=>spec.family,"fresh_mps"=>true,"is_scf_solution"=>false,
            "definition"=>(spec.wavelength==0 ? "95% primary + 5% competing reference correlations" :
                "regular charge/pairing wave locked to antiphase spin nodes; legacy bulk amplitudes and relative bond form"),
            "target_geometry"=>"square","target_t0"=>1.2,"target_V"=>.2,"target_density"=>base.model.density,
            "ep_mode"=>"exact","target_ep"=>base.model.ep,
            "charge_pairing_wavelength_rungs"=>spec.wavelength,"spin_envelope_wavelength_rungs"=>2spec.wavelength,
            "source_kind"=>(spec.wavelength==0 ? "two_basin_references" : "incomplete_legacy_shape_recipe"))
        if spec.wavelength==0
            merge!(provenance,Dict("reference_bundle_sha256"=>TWO_BASIN_REFERENCE_SHA,"epsilon"=>.05,
                "stripe_source_sha256"=>TWO_BASIN_SOURCE_HASHES.stripe,"pairing_source_sha256"=>TWO_BASIN_SOURCE_HASHES.pairing))
        else
            merge!(provenance,Dict("recipe_sha256"=>POSITIVE_V_RECIPE_SHA,"legacy_source_sha256"=>POSITIVE_V_LEGACY_SHA,
                "source_geometry"=>recipe["source_geometry"],"source_completed"=>false,
                "source_mean_density"=>recipe["source_mean_density"],"pairing_phase"=>"same sign between charge peaks",
                "rung_coordinate"=>"x=i-1/2","spatial_phase_radians"=>0.,
                "charge_amplitude"=>recipe["charge_amplitude"],"spin_sz_amplitude"=>recipe["spin_sz_amplitude"],
                "pair_rung_mean"=>recipe["pair_rung_mean"],"pair_rung_modulation"=>recipe["pair_rung_modulation"]))
        end
        h5open(seed_path,"w") do f
            f["artifact_kind"] = "positive_v_derived_field_seed"
            f["chemical_potential"] = base.model.mu_initial
            f["model/transverse_geometry"] = "square"
            LadderMPSMFT._write_fields(create_group(f,"fields/restart"),fields)
            LadderMPSMFT._write_correlations(create_group(f,"template_correlations"),spec.correlations)
            LadderMPSMFT._write_dict(create_group(f,"seed_provenance"),provenance)
        end
        raw["run"]["inherit_from"] = seed_path
        raw["run"]["inherit_sha256"] = LadderMPSMFT.sha256_file(seed_path)
        open(config_path,"w") do io; TOML.print(io,raw); end
        settings = validate_raw_basin_contract(load_settings(config_path))
        readback = read_inherited_fields(seed_path)
        all(getfield(readback.fields,k)==getfield(fields,k) for k in fieldnames(FieldState)) || error("seed readback mismatch")
        push!(rows,(label=label,config=config_path,config_sha256=LadderMPSMFT.sha256_file(config_path),
            geometry="square",t0=1.2,V=.2,family=spec.family,charge_wavelength=spec.wavelength,
            spin_envelope_wavelength=2spec.wavelength,ep_mode="exact",ep_signed=settings.model.ep_signed,
            seed=seed_path,seed_sha256=raw["run"]["inherit_sha256"],
            model_fingerprint=LadderMPSMFT.model_fingerprint(settings.model),
            numerical_fingerprint=LadderMPSMFT.numerical_fingerprint(settings),
            implementation_sha256=implementation_fingerprint(settings),ep_source_sha256=LadderMPSMFT.sha256_file(settings.model.ep_source),
            full_output_directory=output,stateless_output_directory=joinpath(control_run,"results",label)))
    end
    for key in (:model_fingerprint,:numerical_fingerprint,:implementation_sha256,:ep_source_sha256)
        length(unique(getproperty(row,key) for row in rows))==1 || error("incompatible $key")
    end
    open(joinpath(control_run,"manifest.tsv"),"w") do io
        println(io,join(String.(keys(first(rows))),'\t'))
        for row in rows; println(io,join(values(row),'\t')); end
    end
    open(joinpath(control_run,"seed_contract.toml"),"w") do io
        TOML.print(io,Dict("run_id"=>run_id,"geometry"=>"square","t0"=>1.2,"V"=>.2,"branches"=>4,
            "maximum_iterations"=>60,"minimum_iterations"=>40,"stable_iterations"=>10,"chi"=>200,
            "fresh_mps"=>true,"recipe_sha256"=>POSITIVE_V_RECIPE_SHA,"reference_sha256"=>TWO_BASIN_REFERENCE_SHA,
            "intertwined_charge_wavelengths"=>[8,16],"update"=>"raw map; no Anderson; no external pinning fields"))
    end
    println("Prepared four square (1.2,+0.2) branches: ",joinpath(control_run,"manifest.tsv"))
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==6 || error("usage: BASE REFERENCES RECIPE CONTROL_RUN FULL_RUN RUN_ID")
    prepare_square_positive_v(ARGS...)
end
