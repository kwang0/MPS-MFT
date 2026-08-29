#!/usr/bin/env julia

using HDF5
using LadderMPSMFT
using TOML

length(ARGS) == 5 || error(
    "usage: julia --project=. scripts/prepare_phase1_recurrence_competitors.jl BASE_CONFIG.toml RECURRENCE_CONTROL_RUN CONTROL_RUN FULL_RUN RUN_ID",
)

const PAIRING_SURVIVAL_FLOOR = 1.0e-4
const RECURRENCE_LABELS = (
    "unfrustrated__pairing_s1_phase001_chi400",
    "unfrustrated__pairing_s1_phase002_chi400",
    "unfrustrated__pairing_s2_chi400",
)

base_path = abspath(ARGS[1])
recurrence_run = abspath(ARGS[2])
control_run = abspath(ARGS[3])
full_run = abspath(ARGS[4])
run_id = ARGS[5]

isfile(base_path) || error("recurrence base configuration not found: $base_path")
isdir(recurrence_run) || error("recurrence control run not found: $recurrence_run")
occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID: $run_id")

campaign_kind_path = joinpath(recurrence_run, "campaign_kind.txt")
isfile(campaign_kind_path) || error(
    "conditional controls require a recurrence campaign prepared by the current launcher",
)
strip(read(campaign_kind_path, String)) == "recurrence" || error(
    "conditional-control source is not a recurrence campaign: $recurrence_run",
)

function latest_state(root::AbstractString)
    states = String[]
    isdir(root) || error("recurrence branch result directory not found: $root")
    for (directory, _, names) in walkdir(root)
        "state.h5" in names && push!(states, joinpath(directory, "state.h5"))
    end
    isempty(states) && error("no state.h5 found below $root")
    sort!(states; by=path -> (mtime(path), path))
    return last(states)
end

function gate_row(
    label::AbstractString,
    path::AbstractString,
    base_settings,
    full_root::AbstractString,
)
    expected_model = LadderMPSMFT.model_fingerprint(base_settings.model)
    expected_numerical = LadderMPSMFT.numerical_fingerprint(base_settings)
    expected_implementation = LadderMPSMFT.implementation_fingerprint()
    expected_ep = LadderMPSMFT.sha256_file(base_settings.model.ep_source)
    return h5open(path, "r") do file
        for required in (
            "status", "accepted", "fundamental_period", "unmixed_cycle_probe",
            "fields/measured/alpha", "model/transverse_geometry",
            "provenance/initial_seed", "provenance/model_fingerprint",
            "provenance/numerical_fingerprint", "provenance/implementation_sha256",
            "provenance/ep_source_sha256", "provenance/tensor_scalar_type",
            "analysis_storage/is_stateless_copy", "analysis_storage/full_artifact_path",
            "analysis_storage/full_artifact_sha256",
        )
            haskey(file, required) || error("gate state $path has no $required")
        end
        Bool(read(file, "analysis_storage/is_stateless_copy")) || error(
            "gate state is not a stateless analysis copy: $path",
        )
        recorded_full_path = String(read(file, "analysis_storage/full_artifact_path"))
        recorded_full_sha256 = lowercase(String(read(file, "analysis_storage/full_artifact_sha256")))
        isfile(recorded_full_path) || error(
            "gate state full artifact is unavailable; run this preparation on Perlmutter: $recorded_full_path",
        )
        full_path = realpath(recorded_full_path)
        relative = relpath(full_path, full_root)
        (relative == ".." || startswith(relative, "..$(Base.Filesystem.path_separator)")) && error(
            "gate state full artifact escapes the recurrence scratch run: $full_path",
        )
        LadderMPSMFT.sha256_file(full_path) == recorded_full_sha256 || error(
            "gate state full artifact SHA-256 differs from its stateless link: $label",
        )
        geometry = String(read(file, "model/transverse_geometry"))
        geometry == "cubic_unfrustrated" || error(
            "gate state $label has geometry $geometry rather than cubic_unfrustrated",
        )
        initial_seed = lowercase(String(read(file, "provenance/initial_seed")))
        initial_seed == "pairing" || error("gate state $label is not pairing-seeded")
        model_fingerprint = String(read(file, "provenance/model_fingerprint"))
        numerical_fingerprint = String(read(file, "provenance/numerical_fingerprint"))
        implementation_fingerprint = String(read(file, "provenance/implementation_sha256"))
        ep_source_sha256 = lowercase(String(read(file, "provenance/ep_source_sha256")))
        tensor_scalar_type = lowercase(String(read(file, "provenance/tensor_scalar_type")))
        model_fingerprint == expected_model || error(
            "gate state $label does not match the chi=400 control model fingerprint",
        )
        numerical_fingerprint == expected_numerical || error(
            "gate state $label does not match the chi=400 control numerical fingerprint",
        )
        implementation_fingerprint == expected_implementation || error(
            "gate state $label was produced by a different implementation; freeze one code tree for both stages",
        )
        ep_source_sha256 == expected_ep || error(
            "gate state $label does not match the current E_p registry hash",
        )
        tensor_scalar_type == "float64" || error("gate state $label is not Float64")

        status = Symbol(read(file, "status"))
        accepted = Bool(read(file, "accepted"))
        period = Int(read(file, "fundamental_period"))
        unmixed = Bool(read(file, "unmixed_cycle_probe"))
        alpha_max = maximum(abs, read(file, "fields/measured/alpha"))
        alpha_min_phase_max = alpha_max
        if status == :periodic_solution
            haskey(file, "cycle_members") || error(
                "accepted periodic gate state $label has no phase-resolved cycle members",
            )
            phase_names = sort!(String.(collect(keys(file["cycle_members"]))))
            length(phase_names) == period || error(
                "accepted periodic gate state $label has $(length(phase_names)) phases for period $period",
            )
            phase_alpha_maxima = Float64[]
            for phase_name in phase_names
                phase_path = "cycle_members/$phase_name/measured/alpha"
                haskey(file, phase_path) || error(
                    "accepted periodic gate state $label phase $phase_name has no measured pairing field",
                )
                push!(phase_alpha_maxima, maximum(abs, read(file, phase_path)))
            end
            alpha_max = maximum(phase_alpha_maxima)
            alpha_min_phase_max = minimum(phase_alpha_maxima)
        end
        accepted_solution = accepted && (
            status == :fixed_point ||
            (status == :periodic_solution && unmixed && period in base_settings.convergence.accepted_periods)
        )
        survives = accepted_solution && alpha_min_phase_max >= PAIRING_SURVIVAL_FLOOR
        return (
            label=String(label),
            path=String(path),
            sha256=LadderMPSMFT.sha256_file(path),
            full_path=String(full_path),
            full_sha256=recorded_full_sha256,
            status=String(status),
            accepted,
            period,
            unmixed,
            alpha_max,
            alpha_min_phase_max,
            survives,
            model_fingerprint,
            numerical_fingerprint,
            implementation_fingerprint,
            ep_source_sha256,
            tensor_scalar_type,
        )
    end
end

base_settings = load_settings(base_path)
base_settings.model.geometry == :cubic_unfrustrated || error(
    "conditional-control base must use cubic_unfrustrated geometry",
)
base_settings.runtime.backend == :gpu || error("conditional-control base must use the GPU backend")
base_settings.runtime.tensor_scalar_type == :float64 || error(
    "conditional-control base must request Float64 tensors",
)
base_settings.dmrg.maxdim == 400 || error("conditional-control base must use chi=400")
base_settings.convergence.cycle_action == :stop || error(
    "conditional controls must stop an unaccepted raw-map recurrence",
)
base_settings.convergence.probe_iterations == 20 || error(
    "conditional controls require the established 20-update raw-map probe",
)
base_settings.run.max_iterations == base_settings.convergence.probe_iterations + 1 || error(
    "conditional-control base must retain the raw-only Stage A execution boundary",
)
matched_mode = base_settings.run.initial_seed_protocol == :matched_mode
common_random_seed = base_settings.run.random_seed

manifest_path = joinpath(recurrence_run, "manifest.tsv")
isfile(manifest_path) || error("recurrence campaign has no manifest: $manifest_path")
manifest_labels = Tuple(split(line, '\t')[1] for line in readlines(manifest_path)[2:end])
(length(manifest_labels) == length(RECURRENCE_LABELS) &&
    Set(manifest_labels) == Set(RECURRENCE_LABELS)) || error(
    "conditional controls require exactly the three phase-resolved recurrence branches",
)

stateless_root = joinpath(recurrence_run, "stateless_results")
analysis_root = isdir(stateless_root) ? stateless_root : joinpath(recurrence_run, "results")
full_root_path = joinpath(recurrence_run, "full_storage_path.txt")
isfile(full_root_path) || error("recurrence campaign has no full_storage_path.txt")
full_root = realpath(strip(read(full_root_path, String)))
rows = [
    gate_row(label, latest_state(joinpath(analysis_root, label)), base_settings, full_root)
    for label in RECURRENCE_LABELS
]
phase_survivor = any(row -> row.label != RECURRENCE_LABELS[3] && row.survives, rows)
independent_survivor = only(row.survives for row in rows if row.label == RECURRENCE_LABELS[3])
phase_survivor || error(
    "conditional gate failed: neither phase-parent branch is an accepted pairing-bearing survivor",
)
independent_survivor || error(
    "conditional gate failed: independent pairing seed s2 is not an accepted pairing-bearing survivor",
)

config_directory = joinpath(control_run, "configs")
ispath(config_directory) && !isempty(readdir(config_directory)) && error(
    "refusing to overwrite nonempty configuration directory: $config_directory",
)
mkpath(config_directory)
mkpath(joinpath(control_run, "results"))
mkpath(joinpath(full_run, "results"))

gate_path = joinpath(control_run, "conditional_gate.tsv")
ispath(gate_path) && error("refusing to overwrite conditional gate record: $gate_path")
open(gate_path, "w") do io
    println(io, join((
        "label", "state", "state_sha256", "full_state", "full_state_sha256",
        "status", "accepted", "period", "unmixed", "alpha_max", "alpha_min_phase_max",
        "pairing_survival_floor", "survives", "model_fingerprint",
        "numerical_fingerprint", "implementation_fingerprint", "ep_source_sha256",
        "tensor_scalar_type",
    ), '\t'))
    for row in rows
        println(io, join((
            row.label, row.path, row.sha256, row.full_path, row.full_sha256,
            row.status, row.accepted, row.period, row.unmixed, row.alpha_max,
            row.alpha_min_phase_max, PAIRING_SURVIVAL_FLOOR, row.survives, row.model_fingerprint,
            row.numerical_fingerprint, row.implementation_fingerprint, row.ep_source_sha256,
            row.tensor_scalar_type,
        ), '\t'))
    end
end

branches = (
    (
        label="unfrustrated__sdw_s2_chi400",
        branch="sdw",
        seed="sdw",
        seed_label="sdw_s2",
        random_seed=1203,
    ),
    (
        label="unfrustrated__cdw_s2_chi400",
        branch="cdw",
        seed="cdw",
        seed_label="cdw_s2",
        random_seed=1304,
    ),
)

output_manifest = joinpath(control_run, "manifest.tsv")
ispath(output_manifest) && error("refusing to overwrite manifest: $output_manifest")
numerical_fingerprints = String[]
open(output_manifest, "w") do io
    println(io, join((
        "label", "geometry", "branch", "seed", "random_seed", "config", "config_sha256",
        "ep_mode", "ep_signed", "ep_abs", "ep_t0_lower", "ep_t0_upper",
        "ep_lower_signed", "ep_upper_signed", "ep_weight", "tp2_over_ep",
        "parent_checkpoint", "parent_sha256", "parent_orbit_phase", "parent_status",
        "parent_numerical_fingerprint", "parent_tensor_scalar_type",
        "source_phase_iteration", "source_phase_update_mode",
        "full_output_directory", "stateless_output_directory",
        "initial_seed_protocol", "initial_seed_fingerprint",
    ), '\t'))
    for branch in branches
        raw = TOML.parsefile(base_path)
        run = raw["run"]
        full_output_directory = joinpath(full_run, "results", branch.label)
        stateless_output_directory = joinpath(control_run, "results", branch.label)
        run["output_directory"] = full_output_directory
        run["branch_label"] = branch.branch
        run["preparation"] = "conditional_independent_seed"
        run["direction"] = "none"
        run["seed_label"] = matched_mode ? "matched_$(branch.seed_label)" : branch.seed_label
        run["random_seed"] = matched_mode ? common_random_seed : branch.random_seed
        run["initial_seed"] = branch.seed
        run["max_iterations"] = 80
        for key in (
            "inherit_from", "inherit_sha256",
            "parent_checkpoint", "parent_sha256", "parent_orbit_phase",
            "resume_checkpoint", "resume_sha256",
        )
            pop!(run, key, nothing)
        end

        config_path = joinpath(config_directory, "$(branch.label).segment-001.toml")
        ispath(config_path) && error("refusing to overwrite conditional-control config: $config_path")
        open(config_path, "w") do config_io
            TOML.print(config_io, raw; sorted=true)
        end
        settings = load_settings(config_path)
        fingerprint = LadderMPSMFT.numerical_fingerprint(settings)
        push!(numerical_fingerprints, fingerprint)
        fingerprint == LadderMPSMFT.numerical_fingerprint(base_settings) || error(
            "prepared conditional control changed the recurrence numerical fingerprint",
        )
        settings.model.geometry == :cubic_unfrustrated || error(
            "prepared conditional control changed geometry: $(branch.label)",
        )
        settings.run.max_iterations == 80 || error(
            "prepared conditional control lost its fixed-point-acceleration budget",
        )
        model = settings.model
        println(io, join((
            branch.label,
            String(model.geometry),
            branch.branch,
            branch.seed,
            string(run["random_seed"]),
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
            "", "", "0", "", "", "", "0", "",
            full_output_directory,
            stateless_output_directory,
            String(settings.run.initial_seed_protocol),
            initial_seed_fingerprint(settings),
        ), '\t'))
    end
end

length(unique(numerical_fingerprints)) == 1 || error(
    "conditional controls do not share one numerical fingerprint",
)
println("gate_run=$recurrence_run")
println("pairing_survival_floor=$PAIRING_SURVIVAL_FLOOR")
println("phase_parent_survivor=true")
println("independent_seed_survivor=true")
println("branch_count=$(length(branches))")
println("numerical_fingerprint=$(only(unique(numerical_fingerprints)))")
println("gate_path=$gate_path")
println("manifest_path=$output_manifest")
