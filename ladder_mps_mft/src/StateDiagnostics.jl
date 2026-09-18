# Measurements are properties of a saved MPS, independently of SCF acceptance.
const STATE_DIAGNOSTICS_VERSION = "equal_time_v2"

terminal_diagnostics_enabled(diagnostic, run) =
    run.quick_diagnostics && (diagnostic.accepted || diagnostic.status == :maximum_iterations)

function _diagnostic_model(file)
    group = file["model"]
    values = Dict{Symbol,Any}()
    for key in fieldnames(ModelSettings)
        name = String(key)
        stored = key == :geometry ? "transverse_geometry" :
            startswith(name, "ep") ? replace(name, r"^ep" => "E_p") : name
        # Trellis stores the literal ModelSettings names; older ladders use E_p.
        haskey(group, name) && (stored = name)
        haskey(group, stored) || continue
        value = read(group, stored)
        values[key] = fieldtype(ModelSettings, key) == Symbol ? Symbol(value) : value
    end
    model = ModelSettings(; values...)
    model_fingerprint(model) == read(file, "provenance/model_fingerprint") ||
        throw(ArgumentError("stored model does not reproduce the source fingerprint"))
    return model
end

function _diagnostic_samples(file)
    accepted = Bool(read(file, "accepted"))
    period = Int(read(file, "fundamental_period"))
    trellis = String(read(file, "artifact_kind")) == "trellis_mps_mft_state"
    if trellis
        names = sort!(String.(collect(keys(file["ladders"]))))
        length(names) == Int(read(file, "spatial_ladders")) || error("incomplete spatial cell")
        names in (["A"], ["A", "B"]) || error("unknown spatial ladder names")
        return [(psi_path="ladders/$name/psi", suffix="_ladder_$name", ladder=name,
                 phase=0, iteration=Int(last(read(file, "ladders/$name/history/iteration")))) for name in names]
    elseif accepted && period > 1
        Bool(read(file, "orbit_validated")) || error("accepted orbit is not validated")
        names = sort!(String.(collect(keys(file["cycle_members"]))))
        length(names) == period || error("missing orbit phases")
        return [(psi_path="cycle_members/$name/psi", suffix="_phase_$name", ladder="",
                 phase=parse(Int, name), iteration=Int(read(file, "cycle_members/$name/iteration"))) for name in names]
    else
        accepted && period != 1 && error("accepted ladder has invalid period")
        return [(psi_path="psi", suffix="", ladder="", phase=accepted ? 1 : 0,
                 iteration=Int(last(read(file, "history/iteration"))))]
    end
end

"""
Measure a full saved state without modifying it. Unaccepted finite terminal
snapshots require explicit permission; they retain their status and period.
Validated temporal orbit phases and trellis spatial ladders are kept separate.
"""
function measure_state_diagnostics(state_path::AbstractString;
    output_directory::AbstractString=dirname(abspath(state_path)),
    full_pair_correlations::Bool=true, allow_unaccepted::Bool=false,
    expected_sha256::Union{Nothing,AbstractString}=nothing,
    expected_model_fingerprint::Union{Nothing,AbstractString}=nothing,
    reuse::Bool=false)
    source = abspath(state_path)
    source_hash = sha256_file(source)
    expected_sha256 === nothing || source_hash == expected_sha256 || error("source state SHA-256 mismatch")
    implementation = implementation_fingerprint()
    paths = String[]
    h5open(source, "r") do file
        haskey(file, "analysis_storage/is_stateless_copy") && error(
            "measurements require the full MPS artifact, not a stateless mirror")
        accepted = Bool(read(file, "accepted"))
        status = String(read(file, "status"))
        accepted || (allow_unaccepted && status in ("maximum_iterations", "time_limit", "stagnated")) ||
            error("state is not accepted or an explicitly allowed finite terminal snapshot: $status")
        model = _diagnostic_model(file)
        fingerprint = model_fingerprint(model)
        expected_model_fingerprint === nothing || fingerprint == expected_model_fingerprint ||
            error("state and requested model fingerprints differ")
        samples = _diagnostic_samples(file)
        all(s -> haskey(file, s.psi_path), samples) || error("state is missing an MPS")
        for sample in samples
            suffix = full_pair_correlations ? "" : "_basic"
            destination = joinpath(output_directory, "diagnostics$(sample.suffix)$suffix.h5")
            metadata = Dict{String,Any}(
                "measurement_version" => STATE_DIAGNOSTICS_VERSION,
                "measurement_implementation_sha256" => implementation,
                "state_path" => source, "model_fingerprint" => fingerprint,
                "accepted" => accepted, "status" => status,
                "solution_kind" => String(read(file, "solution_kind")),
                "period" => Int(read(file, "fundamental_period")),
                "phase" => sample.phase, "spatial_ladder" => sample.ladder,
                "spatial_ladders" => model.geometry == :trellis ? length(samples) : 1,
                "iteration" => sample.iteration, "source_mps_path" => sample.psi_path,
                "sample_kind" => !accepted ? "terminal_snapshot" :
                    isempty(sample.ladder) ? "accepted_solution_phase" : "accepted_spatial_ladder",
                "full_pair_correlations" => full_pair_correlations,
                "geometry" => String(model.geometry), "L" => model.L,
                "t0" => model.t0, "V" => model.V, "U" => model.U,
                "target_density" => model.density,
                "measurement_scope" => "equal-time intraladder; A/B are spatial; no new DMRG")
            if isfile(destination)
                reuse || error("refusing to overwrite immutable diagnostics: $destination")
                h5open(destination, "r") do existing
                    read(existing, "state_sha256") == source_hash || error("existing measurement has a different source")
                    for key in ("measurement_version", "measurement_implementation_sha256", "source_mps_path", "full_pair_correlations")
                        read(existing, key) == metadata[key] || error("existing measurement differs: $key")
                    end
                    Bool(read(existing, "measurement_complete")) || error("existing measurement is incomplete")
                end
                push!(paths, destination)
                println("diagnostics_reused=$destination")
                continue
            end
            println("measuring_state=$source mps=$(sample.psi_path) accepted=$accepted status=$status")
            flush(stdout)
            started = time()
            psi = read(file, sample.psi_path, MPS)
            diagnostics = compute_ladder_diagnostics(psi, model; full_pair_correlations)
            metadata["measurement_wall_seconds"] = time() - started
            metadata["measurement_complete"] = true
            metadata["measured_utc"] = string(now(UTC))
            write_diagnostics(destination, diagnostics; state_sha256=source_hash, metadata, immutable=true)
            push!(paths, destination)
            println("diagnostics_path=$destination seconds=$(metadata["measurement_wall_seconds"])")
            flush(stdout)
        end
    end
    return paths
end
