#!/usr/bin/env julia
# Read-only discovery, then immutable equal-time measurements. No DMRG/Slurm.
using LadderMPSMFT, HDF5, CSV, TOML
const LMF = LadderMPSMFT
const MEASUREMENT_CAMPAIGNS = [
    ("20260908_square_two_basin_95_5_80_anchors", 2),
    ("20260910_square_two_basin_95_5_40_remainder", 14),
    ("20260915_square_t014_v000_two_basin_finish20", 2),
    ("20260915_cubic_unfrustrated_two_basin_95_5_60", 18),
    ("20260915_square_two_basin_fine_cuts_95_5_60", 12),
    ("20260915_square_t012_vp02_four_seeds_60", 4),
    ("20260916_trellis_two_basin_comparison_60", 4),
]

function diagnostic_inventory(campaign_root; require_full=true)
    rows = NamedTuple[]
    problems = String[]
    for (campaign, expected) in MEASUREMENT_CAMPAIGNS
        directory = joinpath(campaign_root, campaign)
        manifest = joinpath(directory, "manifest.tsv")
        if !isfile(manifest)
            push!(problems, "$campaign: missing manifest"); continue
        end
        count = 0
        for branch in CSV.File(manifest; delim='\t', types=String)
            # The two V=0 anchor endpoints have explicit later continuations.
            campaign == first(MEASUREMENT_CAMPAIGNS)[1] && parse(Float64, branch.V) == 0 && continue
            count += 1
            label = String(branch.label)
            try
                compact_root = joinpath(directory, isdir(joinpath(directory,"stateless_results")) ? "stateless_results" : "results", label)
                candidates = String[]
                if isdir(compact_root)
                    for (root, _, files) in walkdir(compact_root)
                        "state.h5" in files && push!(candidates, joinpath(root,"state.h5"))
                    end
                end
                isempty(candidates) && error("no terminal state; wait for completion and refresh compact results")
                # Run directories begin with sortable UTC timestamps. A later
                # terminal attempt supersedes earlier attempts within a branch.
                sort!(candidates; by=p -> basename(dirname(p)))
                length(candidates) > 1 && basename(dirname(candidates[end])) == basename(dirname(candidates[end-1])) &&
                    error("ambiguous latest terminal states")
                compact = last(candidates)
                config = joinpath(directory,"configs",basename(String(branch.config)))
                isfile(config) || error("missing recorded configuration")
                config_hash = LMF.sha256_file(config)
                config_hash == branch.config_sha256 || error("configuration hash differs from manifest")
                item = h5open(compact,"r") do f
                    LMF._diagnostic_model(f)
                    read(f,"provenance/model_fingerprint") == branch.model_fingerprint || error("manifest/model mismatch")
                    read(f,"provenance/config_sha256") == config_hash || error("state/configuration mismatch")
                    status = String(read(f,"status"))
                    accepted = Bool(read(f,"accepted"))
                    accepted || status in ("maximum_iterations","time_limit","stagnated") || error("unsupported terminal status: $status")
                    stateless = haskey(f,"analysis_storage/is_stateless_copy") && Bool(read(f,"analysis_storage/is_stateless_copy"))
                    source = stateless ? String(read(f,"analysis_storage/full_artifact_path")) : compact
                    hash = stateless ? String(read(f,"analysis_storage/full_artifact_sha256")) : LMF.sha256_file(compact)
                    require_full && !isfile(source) && error("full MPS artifact unavailable: $source")
                    if require_full && stateless
                        filesize(source) == Int(read(f,"analysis_storage/full_artifact_size_bytes")) || error("full artifact size differs from mirror")
                    end
                    samples = LMF._diagnostic_samples(f)
                    (source_path=source, source_sha256=hash,
                     model_fingerprint=String(branch.model_fingerprint), status=status, accepted=accepted,
                     iteration=maximum(s.iteration for s in samples), samples=length(samples))
                end
                push!(rows, (; index=length(rows)+1, campaign, label, config_path=config, config_sha256=config_hash,
                             compact_path=compact, compact_sha256=LMF.sha256_file(compact), item...))
            catch error
                push!(problems, "$campaign/$label: $(sprint(showerror,error))")
            end
        end
        count == expected || push!(problems,"$campaign: $count branches, expected $expected")
    end
    return rows, problems
end

function prepare_measurements(campaign_root, destination)
    ispath(destination) && error("output exists; use its frozen manifest or choose a new run ID")
    rows, problems = diagnostic_inventory(campaign_root)
    isempty(problems) || error(join(problems,'\n'))
    length(rows) == 56 || error("expected 56 latest branch endpoints")
    mkpath(destination)
    CSV.write(joinpath(destination,"manifest.tsv"),rows;delim='\t')
    println("prepared_states=$(length(rows)) mps_measurements=$(sum(r.samples for r in rows))")
end

function run_measurement(manifest, index)
    rows = collect(CSV.File(manifest;delim='\t',types=String))
    1 <= index <= length(rows) || error("invalid manifest row")
    row = rows[index]
    LMF.sha256_file(row.config_path) == row.config_sha256 || error("configuration changed since preparation")
    configure_threading!(RuntimeSettings())
    destination = joinpath(dirname(abspath(manifest)),"results",row.campaign,row.label)
    paths = measure_state_diagnostics(row.source_path;output_directory=destination,
        full_pair_correlations=true,allow_unaccepted=true,reuse=true,
        expected_sha256=row.source_sha256,expected_model_fingerprint=row.model_fingerprint)
    receipt = joinpath(destination,"measurement_receipt.toml")
    if !isfile(receipt)
        temporary = tempname(destination)
        open(temporary,"w") do io
            TOML.print(io,Dict("source_path"=>String(row.source_path),"source_sha256"=>String(row.source_sha256),
                "files"=>basename.(paths),"sha256"=>LMF.sha256_file.(paths)))
        end
        mv(temporary,receipt)
    end
end

function measurement_status(directory)
    rows=CSV.File(joinpath(directory,"manifest.tsv");delim='\t',types=String)
    complete=0
    for row in rows
        receipt=joinpath(directory,"results",row.campaign,row.label,"measurement_receipt.toml")
        finished = if isfile(receipt)
            try
                saved=TOML.parsefile(receipt)
                saved["source_sha256"]==row.source_sha256 && length(saved["files"])==parse(Int,row.samples) &&
                    length(saved["sha256"])==length(saved["files"]) &&
                    all(LMF.sha256_file(joinpath(dirname(receipt),name))==hash for (name,hash) in zip(saved["files"],saved["sha256"]))
            catch
                false
            end
        else
            false
        end
        finished && (complete+=1)
        println("$(row.index)\t$(finished ? "MEASURED" : "MISSING")\t$(row.campaign)/$(row.label)")
    end
    println("complete=$complete/$(length(rows))")
end

function measurements_main(args)
    isempty(args) && error("usage: measure_latest_campaigns.jl plan ROOT [--local] | prepare ROOT OUT | run MANIFEST INDEX | status OUT")
    if args[1]=="plan"
        rows,problems=diagnostic_inventory(args[2];require_full=!("--local" in args))
        for row in rows
            println("$(row.index)\t$(row.campaign)\t$(row.label)\t$(row.status)\t$(row.samples)")
        end
        println("ready_states=$(length(rows))/56 mps_measurements=$(sum(r.samples for r in rows;init=0))")
        foreach(p -> println(stderr,p),problems)
        return isempty(problems)
    elseif args[1]=="prepare"
        prepare_measurements(abspath(args[2]),abspath(args[3]))
    elseif args[1]=="run"
        run_measurement(abspath(args[2]),parse(Int,args[3]))
    elseif args[1]=="status"
        measurement_status(abspath(args[2]))
    else
        error("unknown action")
    end
    return true
end

if abspath(PROGRAM_FILE)==@__FILE__
    measurements_main(ARGS) || exit(1)
end
