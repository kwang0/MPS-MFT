#!/usr/bin/env julia

using HDF5
using LadderMPSMFT
using LinearAlgebra
using Statistics
using TOML

const TWO_BASIN_REFERENCE_SHA = "e01a1ea7d6be813110870d26377db0df529d1584816c946af78584e1e1fbddc1"
const TWO_BASIN_EPSILON = 0.05
const TWO_BASIN_SOURCE_HASHES = (
    stripe="ae6a3bfe76ca8f06f2396fd731b18bca8539e0b7ee68df016cc9156fdceeb074",
    pairing="8a1cf2d64d2fbe0eb59521192b829cab43e19a4d7ac026519ea847f6ac0778b8",
)

function two_basin_references(path)
    LadderMPSMFT.sha256_file(path) == TWO_BASIN_REFERENCE_SHA || error("reference bundle hash mismatch")
    h5open(path, "r") do file
        String(read(file, "artifact_kind")) == "two_basin_correlation_templates" || error("wrong reference kind")
        read(file, "L") == 64 && read(file, "chi") == 200 || error("reference size/chi mismatch")
        map((:stripe, :pairing)) do family
            group = file[String(family)]
            String(read(group, "source_sha256")) == getproperty(TWO_BASIN_SOURCE_HASHES, family) || error("source hash mismatch")
            values = [Float64.(read(group, String(key))) for key in fieldnames(CorrelationState)]
            all(value -> all(isfinite, value), values) || error("nonfinite template")
            CorrelationState(values...)
        end
    end
end

function blend_correlations(primary::CorrelationState, perturbation::CorrelationState; epsilon=TWO_BASIN_EPSILON)
    0 < epsilon < 0.5 || error("perturbation fraction must lie in (0,0.5)")
    CorrelationState(((1 - epsilon) .* getfield(primary, key) .+ epsilon .* getfield(perturbation, key)
                      for key in fieldnames(CorrelationState))...)
end

function validate_raw_basin_contract(settings)
    settings.model.geometry == :square && settings.model.L == 64 || error("expected square L=64")
    settings.model.U == 8 && settings.model.tp == 0.1 && settings.model.density == 0.9375 || error("model contract changed")
    settings.dmrg.maxdim == 200 || error("expected chi=200")
    settings.mixing.method == :linear && !settings.mixing.adaptive || error("Anderson/adaptive mixing is forbidden")
    settings.mixing.minimum_damping == settings.mixing.damping == settings.mixing.maximum_damping == 1 || error("fallback must be raw F(x)")
    settings.convergence.unmixed_cycle_probe || error("raw-map observation is required")
    settings.convergence.probe_iterations >= settings.run.max_iterations || error("probe must cover the entire run")
    settings.convergence.minimum_iterations >= 50 || error("at least 50 evaluations are required")
    settings.convergence.channel_residuals || error("channel gates are required")
    settings.convergence.dmrg_sweep_energy_tol == settings.dmrg.energy_tol || error("inner DMRG acceptance must match its stopping tolerance")
    settings.convergence.stable_iterations >= 5 || error("at least five stable records are required")
    settings.model.ep_mode == :exact || error("exact E_p is required")
    settings.run.parent_checkpoint === nothing && settings.run.resume_checkpoint === nothing || error("fresh MPS required")
    return settings
end

function prepare_two_basin_grid(base_path, reference_path, control_run, full_run, run_id; stage="anchors")
    stage in ("anchors", "remainder", "grid") || error("stage must be anchors, remainder, or grid")
    occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID")
    control_run, full_run = abspath(control_run), abspath(full_run)
    raw_base = TOML.parsefile(base_path)
    validate_raw_basin_contract(load_settings(base_path))
    stripe, pairing = two_basin_references(reference_path)
    seeds = (stripe=blend_correlations(stripe, pairing), pairing=blend_correlations(pairing, stripe))
    for name in ("configs", "seeds")
        directory = joinpath(control_run, name)
        isdir(directory) && !isempty(readdir(directory)) && error("refusing to overwrite $directory")
    end
    isfile(joinpath(control_run, "manifest.tsv")) && error("manifest already exists")
    mkpath(joinpath(control_run, "configs")); mkpath(joinpath(control_run, "seeds"))
    mkpath(joinpath(full_run, "results"))
    rows = NamedTuple[]
    for t0 in (1.0, 1.2, 1.4), V in (-0.4, -0.2, 0.0)
        anchor = t0 == 1.4 && V in (-0.4, 0.0)
        stage == "anchors" && !anchor && continue
        stage == "remainder" && anchor && continue
        point = "t0$(round(Int, t0 * 10))_" * (V == 0 ? "v000" : "vm0$(round(Int, -10V))")
        point_fingerprints = NamedTuple[]
        for family in (:stripe, :pairing)
            label = "square__$(family)_weak_other_$(point)_chi200_raw"
            raw = deepcopy(raw_base)
            raw["model"]["t0"] = t0; raw["model"]["V"] = V
            raw["model"]["mu_initial"] = V == 0 ? 1.65 : V == -0.2 ? 1.10 : 0.55
            raw["pair_binding"]["registry"] = joinpath(LadderMPSMFT.PROJECT_ROOT, "data", "E_p_values.csv")
            output = joinpath(full_run, "results", label)
            seed_path = joinpath(control_run, "seeds", label * ".h5")
            config_path = joinpath(control_run, "configs", label * ".segment-001.toml")
            raw["run"]["output_directory"] = output
            raw["run"]["branch_label"] = String(family) * "_weak_other"
            raw["run"]["seed_label"] = String(family) * "_eps005"
            raw["run"]["random_seed"] = 1404
            # Resolve the target Hamiltonian first; all fields use its kernel.
            open(config_path, "w") do io; TOML.print(io, raw); end
            model = load_settings(config_path).model
            correlations = getproperty(seeds, family)
            fields = LadderMPSMFT.mean_fields_from_correlations(correlations, model)
            h5open(seed_path, "w") do file
                file["artifact_kind"] = "two_basin_derived_field_seed"
                file["chemical_potential"] = model.mu_initial
                file["model/transverse_geometry"] = "square"
                LadderMPSMFT._write_fields(create_group(file, "fields/restart"), fields)
                LadderMPSMFT._write_correlations(create_group(file, "template_correlations"), correlations)
                provenance = create_group(file, "seed_provenance")
                LadderMPSMFT._write_dict(provenance, Dict(
                    "family" => String(family), "epsilon" => TWO_BASIN_EPSILON,
                    "definition" => "0.95 primary correlations + 0.05 competing correlations; target mean-field kernel",
                    "stripe_source_sha256" => TWO_BASIN_SOURCE_HASHES.stripe,
                    "pairing_source_sha256" => TWO_BASIN_SOURCE_HASHES.pairing,
                    "reference_bundle_sha256" => TWO_BASIN_REFERENCE_SHA,
                    "target_t0" => t0, "target_V" => V, "target_ep" => model.ep,
                    "source_L" => 64, "target_L" => 64, "fresh_mps" => true,
                    "is_scf_solution" => false,
                ))
            end
            seed_sha = LadderMPSMFT.sha256_file(seed_path)
            raw["run"]["inherit_from"] = seed_path
            raw["run"]["inherit_sha256"] = seed_sha
            open(config_path, "w") do io; TOML.print(io, raw); end
            settings = validate_raw_basin_contract(load_settings(config_path))
            readback = read_inherited_fields(seed_path)
            for component in (:alpha, :beta, :mu_cdw)
                getfield(readback.fields, component) == getfield(fields, component) || error("seed readback mismatch")
            end
            norm(fields.alpha) > 1e-8 || error("missing pairing perturbation")
            norm(fields.mu_cdw[2, :] .- fields.mu_cdw[1, :]) > 1e-6 || error("missing stripe perturbation")
            fp = (model_fingerprint=LadderMPSMFT.model_fingerprint(model),
                  numerical_fingerprint=LadderMPSMFT.numerical_fingerprint(settings),
                  implementation_sha256=implementation_fingerprint(settings),
                  ep_source_sha256=LadderMPSMFT.sha256_file(model.ep_source))
            push!(point_fingerprints, fp)
            push!(rows, merge((label=label, config=config_path, config_sha256=LadderMPSMFT.sha256_file(config_path),
                geometry="square", t0=t0, V=V, family=String(family), epsilon=TWO_BASIN_EPSILON,
                seed=seed_path, seed_sha256=seed_sha, reference_sha256=TWO_BASIN_REFERENCE_SHA,
                full_output_directory=output, stateless_output_directory=joinpath(control_run, "results", label)), fp))
        end
        first(point_fingerprints) == last(point_fingerprints) || error("paired branches have incompatible fingerprints")
    end
    manifest_path = joinpath(control_run, "manifest.tsv")
    open(manifest_path, "w") do io
        println(io, join(String.(keys(first(rows))), '\t'))
        for row in rows; println(io, join(values(row), '\t')); end
    end
    open(joinpath(control_run, "seed_contract.toml"), "w") do io
        TOML.print(io, Dict("run_id" => run_id, "stage" => stage, "epsilon" => TWO_BASIN_EPSILON,
            "reference_sha256" => TWO_BASIN_REFERENCE_SHA, "branches" => length(rows),
            "maximum_iterations" => 80, "minimum_iterations" => 50, "chi" => 200,
            "update" => "unmixed raw map throughout; no Anderson", "fresh_mps" => true))
    end
    println("Prepared $(length(rows)) branches ($stage), manifest=$manifest_path")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 6 || error("usage: julia --project=. scripts/prepare_phase1_two_basin_grid.jl BASE.toml REFERENCES.h5 CONTROL_RUN FULL_RUN RUN_ID anchors|remainder|grid")
    prepare_two_basin_grid(ARGS[1:5]...; stage=ARGS[6])
end
