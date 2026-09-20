#!/usr/bin/env julia
isdefined(@__MODULE__, :two_basin_references) || include("prepare_phase1_two_basin_grid.jl")

"""Reindex a seed only. This never changes the open-boundary Hamiltonian.
Permuting both matrix indices preserves symmetry and the density spectrum;
the finite-reference end structure is displaced too and is free to relax.
"""
function square_shift_seed(c::CorrelationState, shift::Integer)
    L = length(c.density_up) ÷ 2
    sites = [2(mod1(i-shift, L)-1)+leg for i in 1:L for leg in 1:2]
    return CorrelationState((ndims(getfield(c,k)) == 2 ? getfield(c,k)[sites,sites] :
        getfield(c,k)[sites] for k in fieldnames(CorrelationState))...)
end

function validate_square_two_ladder(settings)
    m, d, c, r = settings.model, settings.dmrg, settings.convergence, settings.run
    m.geometry == :square && m.spatial_cell == :two_ladder || error("expected square A/B cell")
    (m.L,m.t,m.U,m.t0,m.tp,m.density,m.r_range) == (64,1.,8.,1.4,.1,.9375,4) || error("square model contract changed")
    m.V in (-.4,-.2) && m.ep_mode == :exact && m.ep_signed < 0 || error("expected exact E_p at the paired square points")
    d.maxdim == 200 && d.energy_tol == c.dmrg_sweep_energy_tol == 1e-7 || error("chi/DMRG contract changed")
    r.max_iterations == c.probe_iterations == 60 && c.minimum_iterations == 40 && c.stable_iterations == 10 || error("60/40/10 iteration contract changed")
    c.channel_residuals && c.channel_noise_floor == 5e-7 && c.variational_energy_tol == 1e-7 || error("channel gates changed")
    c.field_abs_tol == 1e-7 && c.field_rel_tol == 1e-4 && c.accepted_periods == [1] || error("stationarity gates changed")
    r.quick_diagnostics && r.full_pair_correlations || error("full terminal measurements required")
    r.parent_checkpoint === nothing && r.resume_checkpoint === nothing || error("fresh MPS required")
    return settings
end

function prepare_square_two_ladder(base_path, reference_path, control_run, full_run, run_id)
    occursin(r"^[A-Za-z0-9_.-]+$",run_id) || error("unsafe run ID")
    validate_square_two_ladder(load_settings(base_path))
    raw_base = TOML.parsefile(base_path)
    stripe, pairing = two_basin_references(reference_path)
    shift = 8 # Half the nominal charge period; spin is shifted consistently.
    stripes = [stripe, square_shift_seed(stripe,shift)]
    control_run, full_run = abspath(control_run), abspath(full_run)
    for name in ("configs","seeds")
        path = joinpath(control_run,name)
        isdir(path) && !isempty(readdir(path)) && error("refusing to overwrite $path")
    end
    ispath(joinpath(control_run,"manifest.tsv")) && error("manifest already exists")
    mkpath(joinpath(control_run,"configs")); mkpath(joinpath(control_run,"seeds"))
    mkpath(joinpath(full_run,"results"))
    rows = NamedTuple[]
    for V in (-.4,-.2), family in (:stripe,:pairing)
        label = "square__two_ladder__$(family)_eps005_t014_vm0$(round(Int,-10V))_chi200_raw60"
        raw = deepcopy(raw_base)
        raw["model"]["V"] = V
        raw["model"]["mu_initial"] = V == -.4 ? .55 : 1.10
        raw["pair_binding"]["registry"] = joinpath(LadderMPSMFT.PROJECT_ROOT,"data","E_p_values.csv")
        raw["run"]["branch_label"] = "two_ladder_$(family)_weak_other"
        raw["run"]["seed_label"] = "$(family)_eps005_Bshift08"
        raw["run"]["output_directory"] = joinpath(full_run,"results",label)
        config = joinpath(control_run,"configs",label*".segment-001.toml")
        seed = joinpath(control_run,"seeds",label*".h5")
        open(config,"w") do io; TOML.print(io,raw); end
        model = load_settings(config).model
        correlations = [family == :stripe ? blend_correlations(s,pairing) :
            blend_correlations(pairing,s) for s in stripes]
        fields = spatial_cell_mean_fields(correlations,model)
        norm(fields[1].mu_cdw-fields[2].mu_cdw) > 1e-6 || error("A/B asymmetry missing")
        h5open(seed,"w") do file
            file["artifact_kind"] = "spatial_cell_correlation_seed"
            file["model_fingerprint"] = LadderMPSMFT.model_fingerprint(model)
            file["chemical_potential"] = model.mu_initial
            file["spatial_ladders"] = 2
            for (i,name) in enumerate(("A","B"))
                g = create_group(file,"ladders/$name")
                LadderMPSMFT._write_correlations(create_group(g,"template_correlations"),correlations[i])
                LadderMPSMFT._write_fields(create_group(g,"fields"),fields[i])
                norm(fields[i].alpha) > 1e-8 || error("missing pairing component")
                norm(fields[i].mu_cdw[2,:]-fields[i].mu_cdw[1,:]) > 1e-6 || error("missing spin component")
            end
            LadderMPSMFT._write_dict(create_group(file,"seed_provenance"),Dict(
                "family"=>String(family),"epsilon"=>TWO_BASIN_EPSILON,
                "definition"=>"95% primary + 5% competing reference; B stripe template displaced by 8 rungs; pairing template unshifted on A/B; target square kernel",
                "stripe_shift_rungs_A"=>0,"stripe_shift_rungs_B"=>shift,
                "shift_method"=>"cyclic site permutation of the seed correlations only; physical bonds remain open and unshifted",
                "reference_bundle_sha256"=>TWO_BASIN_REFERENCE_SHA,
                "stripe_source_sha256"=>TWO_BASIN_SOURCE_HASHES.stripe,
                "pairing_source_sha256"=>TWO_BASIN_SOURCE_HASHES.pairing,
                "fresh_mps"=>true,"random_seed_each_ladder"=>1404,"is_scf_solution"=>false))
        end
        raw["run"]["inherit_from"] = seed
        raw["run"]["inherit_sha256"] = LadderMPSMFT.sha256_file(seed)
        open(config,"w") do io; TOML.print(io,raw); end
        settings = validate_square_two_ladder(load_settings(config))
        push!(rows,(label,config,config_sha256=LadderMPSMFT.sha256_file(config),
            geometry="square",spatial_cell="two_ladder",spatial_ladders=2,family=String(family),
            t0=model.t0,V,L=model.L,density=model.density,chi=settings.dmrg.maxdim,
            ep_signed=model.ep_signed,ep_denominator=model.ep,ep_mode=String(model.ep_mode),
            seed,seed_sha256=settings.run.inherit_sha256,reference_sha256=TWO_BASIN_REFERENCE_SHA,
            stripe_shift_rungs_B=shift,model_fingerprint=LadderMPSMFT.model_fingerprint(model),
            numerical_fingerprint=LadderMPSMFT.numerical_fingerprint(settings),
            implementation_sha256=implementation_fingerprint(settings),
            ep_source_sha256=LadderMPSMFT.sha256_file(model.ep_source),
            full_output_directory=settings.run.output_directory,
            stateless_output_directory=joinpath(control_run,"results",label)))
    end
    for V in (-.4,-.2), key in (:model_fingerprint,:numerical_fingerprint,:implementation_sha256,:ep_source_sha256)
        length(unique(getproperty(r,key) for r in rows if r.V == V)) == 1 || error("unmatched $key")
    end
    open(joinpath(control_run,"manifest.tsv"),"w") do io
        println(io,join(String.(keys(first(rows))),'\t'))
        for row in rows; println(io,join(values(row),'\t')); end
    end
    open(joinpath(control_run,"seed_contract.toml"),"w") do io
        TOML.print(io,Dict("run_id"=>run_id,"geometry"=>"square","spatial_cell"=>"two_ladder",
            "branches"=>4,"t0"=>1.4,"V"=>[-.4,-.2],"epsilon"=>TWO_BASIN_EPSILON,
            "reference_sha256"=>TWO_BASIN_REFERENCE_SHA,"fresh_mps"=>true,
            "maximum_cell_sweeps"=>60,"minimum_cell_sweeps"=>40,"stable_sweeps"=>10,
            "maximum_ladder_solves"=>480,"chi"=>200,"stripe_shift_rungs_B"=>shift,
            "density_constraint"=>"fixed n=0.9375 on each ladder",
            "update"=>"simultaneous raw map; square bonds connect equal rungs; no shear or diagonal hopping",
            "accepted_iteration_periods"=>[1],"full_terminal_correlations"=>true))
    end
    println("Prepared four square A/B branches: ",joinpath(control_run,"manifest.tsv"))
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 5 || error("usage: BASE REFERENCES CONTROL FULL RUN_ID")
    prepare_square_two_ladder(ARGS...)
end
