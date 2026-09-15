#!/usr/bin/env julia
using HDF5, LadderMPSMFT, TOML

const SQUARE_FINISH_SOURCES = (
    (family="stripe", iterations=80, job="58093802",
     compact_sha="ba38078fd047653251f321d14995e89acc2c63d05cdba8e1e41fac6b2a3b5bb0",
     full_sha="9ad2d9ea1239727e577be2997a7a9f62e1e6aaeae0653bd8591448f55dc58ea5"),
    (family="pairing", iterations=62, job="58093803",
     compact_sha="8d18a149ef1b6707f2377b35f9b5ed9ada97f6a287d693b09ede6b0c7e0c5ab9",
     full_sha="d4b2ef33f8d969e23b519f08642f50eacfd158948c0b6710756613fd89344237"),
)

function square_finish_source(source_run, expected; compact_preview=false)
    label = "square__$(expected.family)_weak_other_t014_v000_chi200_raw"
    root = joinpath(source_run, "results", label)
    isdir(root) || error("missing original branch: $root")
    paths = [joinpath(dir,"state.h5") for (dir,_,names) in walkdir(root) if "state.h5" in names]
    length(paths) == 1 || error("expected exactly one original state for $label")
    compact_path = only(paths)
    LadderMPSMFT.sha256_file(compact_path) == expected.compact_sha || error("original compact hash mismatch: $label")
    metadata = h5open(compact_path,"r") do f
        Bool(read(f,"analysis_storage/is_stateless_copy")) || error("expected original compact state")
        String(read(f,"provenance/slurm_job_id")) == expected.job || error("original job mismatch")
        length(read(f,"history/iteration")) == expected.iterations || error("original history length mismatch")
        !Bool(read(f,"accepted")) || error("unexpected accepted source")
        String(read(f,"status")) in ("maximum_iterations","time_limit") || error("unsupported source status")
        Int(read(f,"fundamental_period")) == 0 || error("source is a periodic solution")
        String(read(f,"provenance/tensor_scalar_type")) == "float64" || error("expected Float64 source")
        String(read(f,"analysis_storage/full_artifact_sha256")) == expected.full_sha || error("full-source link hash mismatch")
        for component in ("alpha","beta","mu_cdw")
            read(f,"fields/restart/$component") == read(f,"fields/measured/$component") || error("restart is not the latest measured field")
        end
        (full_path=String(read(f,"analysis_storage/full_artifact_path")),
         model_fingerprint=String(read(f,"provenance/model_fingerprint")),
         chemical_potential=Float64(read(f,"chemical_potential")),
         config_sha=String(read(f,"provenance/config_sha256")),
         ep_sha=String(read(f,"provenance/ep_source_sha256")),
         status=String(read(f,"status")))
    end
    previous_config = joinpath(source_run,"configs",label*".segment-001.toml")
    LadderMPSMFT.sha256_file(previous_config) == metadata.config_sha || error("original config hash mismatch")
    if !compact_preview
        isfile(metadata.full_path) || error("full MPS is missing on scratch: $(metadata.full_path); compact states cannot resume")
        full_root = realpath(strip(read(joinpath(source_run,"full_storage_path.txt"),String)))
        relative = relpath(realpath(metadata.full_path),full_root)
        first(splitpath(relative)) == ".." && error("full source escapes original scratch run")
        LadderMPSMFT.sha256_file(metadata.full_path) == expected.full_sha || error("full source hash mismatch")
        h5open(metadata.full_path,"r") do full
            haskey(full,"psi") || error("full source has no MPS")
            String(read(full,"provenance/model_fingerprint")) == metadata.model_fingerprint || error("full/compact model mismatch")
            String(read(full,"provenance/slurm_job_id")) == expected.job || error("full source job mismatch")
            h5open(compact_path,"r") do compact
                for component in ("alpha","beta","mu_cdw")
                    read(full,"fields/restart/$component") == read(compact,"fields/measured/$component") || error("full/compact restart mismatch")
                end
            end
        end
    end
    return merge(expected,metadata,(label=label,compact_path=compact_path))
end

function prepare_square_two_basin_finish(base_path,source_run,control_run,full_run,run_id; compact_preview=false)
    occursin(r"^[A-Za-z0-9_.-]+$",run_id) || error("unsafe run ID")
    raw_base = TOML.parsefile(base_path)
    raw_base["pair_binding"]["registry"] = joinpath(LadderMPSMFT.PROJECT_ROOT,"data","E_p_values.csv")
    base = load_settings(base_path)
    base.model.geometry == :square && base.model.t0 == 1.4 && base.model.V == 0 || error("expected square (1.4,0)")
    base.run.max_iterations == 20 && base.convergence.minimum_iterations == 10 || error("expected 20-update / 10-minimum continuation")
    base.dmrg.maxdim == 200 && base.convergence.stable_iterations == 10 || error("expected chi=200 and ten fresh stable records")
    base.convergence.probe_iterations >= 20 && base.convergence.unmixed_cycle_probe || error("raw observation must cover continuation")
    base.mixing.method == :linear && !base.mixing.adaptive && base.mixing.damping == base.mixing.minimum_damping == base.mixing.maximum_damping == 1 || error("raw F(x) required")
    base.convergence.dmrg_sweep_energy_tol == base.dmrg.energy_tol || error("inner stopping/acceptance mismatch")
    base.convergence.channel_residuals && base.convergence.channel_noise_floor == 5e-7 || error("retain qualified square channel floor")
    isfile(joinpath(control_run,"manifest.tsv")) && error("refusing existing manifest")
    config_dir = joinpath(control_run,"configs")
    isdir(config_dir) && !isempty(readdir(config_dir)) && error("refusing existing configs")
    sources = [square_finish_source(source_run,s;compact_preview) for s in SQUARE_FINISH_SOURCES]
    all(s -> s.model_fingerprint == LadderMPSMFT.model_fingerprint(base.model),sources) || error("continuation changed the source Hamiltonian")
    all(s -> s.ep_sha == LadderMPSMFT.sha256_file(base.model.ep_source),sources) || error("E_p registry changed")
    mkpath(config_dir); mkpath(joinpath(full_run,"results"))
    rows = NamedTuple[]
    for source in sources
        label = source.label*"_finish20"
        raw = deepcopy(raw_base)
        # Keep the original model fingerprint, including mu_initial. The solver
        # takes its actual starting chemical potential from the parent checkpoint.
        run = raw["run"]
        for key in ("inherit_from","inherit_sha256","resume_checkpoint","resume_sha256","parent_checkpoint","parent_sha256","parent_orbit_phase")
            pop!(run,key,nothing)
        end
        run["output_directory"] = joinpath(abspath(full_run),"results",label)
        run["branch_label"] = source.family*"_weak_other_finish20"
        run["seed_label"] = source.family*"_eps005_continued"
        # Same model and full MPS; parent ancestry also enables the existing
        # plotting adapter to stitch the entire original measured history.
        run["parent_checkpoint"] = source.full_path
        run["parent_sha256"] = source.full_sha
        config_path = joinpath(abspath(config_dir),label*".segment-001.toml")
        open(config_path,"w") do io; TOML.print(io,raw;sorted=true); end
        settings = load_settings(config_path)
        LadderMPSMFT.model_fingerprint(settings.model) == source.model_fingerprint || error("prepared model mismatch")
        push!(rows,(label=label,config=config_path,config_sha256=LadderMPSMFT.sha256_file(config_path),
            geometry="square",t0=1.4,V=0.,family=source.family,
            parent_checkpoint=source.full_path,parent_sha256=source.full_sha,
            source_compact_state=source.compact_path,source_compact_sha256=source.compact_sha,
            parent_job_id=source.job,parent_iterations=source.iterations,parent_status=source.status,
            maximum_additional_iterations=20,full_source_verified=!compact_preview,
            model_fingerprint=LadderMPSMFT.model_fingerprint(settings.model),
            numerical_fingerprint=LadderMPSMFT.numerical_fingerprint(settings),
            implementation_sha256=implementation_fingerprint(settings),ep_source_sha256=source.ep_sha,
            full_output_directory=run["output_directory"],
            stateless_output_directory=joinpath(abspath(control_run),"results",label)))
    end
    for key in (:model_fingerprint,:numerical_fingerprint,:implementation_sha256,:ep_source_sha256)
        getproperty(rows[1],key) == getproperty(rows[2],key) || error("two continuation branches are not comparable")
    end
    open(joinpath(control_run,"manifest.tsv"),"w") do io
        println(io,join(String.(keys(first(rows))),'\t'))
        for row in rows; println(io,join(values(row),'\t')); end
    end
    open(joinpath(control_run,"continuation_contract.toml"),"w") do io
        TOML.print(io,Dict("run_id"=>run_id,"branches"=>2,"chi"=>200,
            "maximum_additional_iterations"=>20,"minimum_fresh_iterations"=>10,
            "full_sources_verified"=>!compact_preview,"preview_only"=>compact_preview,
            "boundary"=>"full MPS + latest measured fields; fresh convergence window; raw map; no automatic extensions"))
    end
    println("Prepared two square continuations; full_sources_verified=$(!compact_preview)")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (5,6) || error("usage: BASE SOURCE_RUN CONTROL_RUN FULL_RUN NEW_RUN [--compact-preview]")
    length(ARGS)==5 || ARGS[6]=="--compact-preview" || error("unknown option")
    prepare_square_two_basin_finish(ARGS[1:5]...;compact_preview=length(ARGS)==6)
end
