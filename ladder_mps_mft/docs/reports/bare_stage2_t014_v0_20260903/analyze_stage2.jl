using LadderMPSMFT
using HDF5
using LinearAlgebra
using Printf
using Statistics

const REPORT_DIR = @__DIR__
const PROJECT_DIR = normpath(joinpath(REPORT_DIR, "..", "..", ".."))
const STAGE2_DIR = joinpath(
    PROJECT_DIR,
    "output",
    "bare_stage2",
    "20260902_bare_t014_v0_stage2",
)
const STAGE2_RESULTS = joinpath(STAGE2_DIR, "stateless_results")
const SCF_DIR = joinpath(
    PROJECT_DIR,
    "output",
    "phase1_gpu",
    "20260902_phase1_square_t014_v000_seed_chi200_loose_cuda130",
)

function write_tsv(path::AbstractString, rows, columns)
    open(path, "w") do io
        println(io, join(string.(columns), '\t'))
        for row in rows
            values = map(columns) do column
                value = getproperty(row, column)
                if value isa AbstractFloat
                    isfinite(value) ? @sprintf("%.16g", value) : string(value)
                else
                    replace(string(value), '\t' => ' ', '\n' => ' ')
                end
            end
            println(io, join(values, '\t'))
        end
    end
    return path
end

component_norm(component, model) = sqrt(sum(abs2, component) / (2 * model.L))

function component_norms(fields::FieldState, model)
    return (
        alpha=component_norm(fields.alpha, model),
        beta=component_norm(fields.beta, model),
        mu_cdw=component_norm(fields.mu_cdw, model),
        total=field_metric_norm(fields, model),
    )
end

function add_fields(left::FieldState, right::FieldState)
    return FieldState(
        left.alpha .+ right.alpha,
        left.beta .+ right.beta,
        left.mu_cdw .+ right.mu_cdw,
    )
end

function beta_class_background(fields::FieldState, model)
    beta = zeros(Float64, size(fields.beta))
    for spin in 1:2, offset in -model.r_range:model.r_range, leg in 1:2, other_leg in 1:2
        positions = Tuple{Int,Int}[]
        for rung in 1:model.L
            target = rung + offset
            1 <= target <= model.L || continue
            push!(positions, (rung, target))
        end
        isempty(positions) && continue
        average = mean(fields.beta[spin, rung, target, leg, other_leg] for (rung, target) in positions)
        for (rung, target) in positions
            beta[spin, rung, target, leg, other_leg] = average
        end
    end
    return beta
end

function split_background(fields::FieldState, model)
    beta_background = beta_class_background(fields, model)
    mu_average = mean(fields.mu_cdw)
    mu_background = fill(mu_average, size(fields.mu_cdw))
    background = FieldState(
        zeros(Float64, size(fields.alpha)),
        beta_background,
        mu_background,
    )
    residual = subtract_fields(fields, background)
    return (; background, residual, mu_average)
end

function mu_profiles(mu_cdw, L)
    charge_even = zeros(Float64, L)
    charge_odd = zeros(Float64, L)
    spin_even = zeros(Float64, L)
    spin_odd = zeros(Float64, L)
    for rung in 1:L
        site1 = rung_leg_to_site(rung, 0)
        site2 = rung_leg_to_site(rung, 1)
        charge1 = (mu_cdw[1, site1] + mu_cdw[2, site1]) / 2
        charge2 = (mu_cdw[1, site2] + mu_cdw[2, site2]) / 2
        spin1 = (mu_cdw[2, site1] - mu_cdw[1, site1]) / 2
        spin2 = (mu_cdw[2, site2] - mu_cdw[1, site2]) / 2
        charge_even[rung] = (charge1 + charge2) / 2
        charge_odd[rung] = (charge1 - charge2) / 2
        spin_even[rung] = (spin1 + spin2) / 2
        spin_odd[rung] = (spin1 - spin2) / 2
    end
    return (; charge_even, charge_odd, spin_even, spin_odd)
end

function nearest_neighbor_beta_profile(fields::FieldState, model)
    result = zeros(Float64, model.L - 1)
    for rung in 1:(model.L - 1)
        result[rung] = mean(
            fields.beta[spin, rung, rung + 1, leg, leg]
            for spin in 1:2, leg in 1:2
        )
    end
    return result
end

function centered_cosine(left, right)
    l = left .- mean(left)
    r = right .- mean(right)
    denominator = norm(l) * norm(r)
    return denominator > eps(Float64) ? dot(l, r) / denominator : NaN
end

function field_cosine(left::FieldState, right::FieldState, model)
    denominator = field_metric_norm(left, model) * field_metric_norm(right, model)
    return denominator > eps(Float64) ? field_metric_dot(left, right, model) / denominator : NaN
end

function find_single_state(branch_directory::AbstractString)
    paths = String[]
    for (root, _, files) in walkdir(branch_directory)
        "state.h5" in files && push!(paths, joinpath(root, "state.h5"))
    end
    length(paths) == 1 || error("expected one state.h5 under $branch_directory, found $(length(paths))")
    return only(paths)
end

function read_fields(group)
    return FieldState(
        Float64.(read(group, "alpha")),
        Float64.(read(group, "beta")),
        Float64.(read(group, "mu_cdw")),
    )
end

function first_history_fields(file, source::AbstractString)
    group = file["history/fields/$source"]
    function first_slice(component)
        values = Float64.(read(group, component))
        return copy(selectdim(values, ndims(values), 1))
    end
    return FieldState(first_slice("alpha"), first_slice("beta"), first_slice("mu_cdw"))
end

function short_seed_label(label::AbstractString)
    occursin("stripe_pairing_m004", label) && return "stripe + d-wave, m=4"
    occursin("stripe_pairing_m005", label) && return "stripe + d-wave, m=5"
    occursin("stripe_m004", label) && return "stripe, m=4"
    occursin("stripe_m005", label) && return "stripe, m=5"
    occursin("pairing_dwave", label) && return "d-wave, q=0"
    occursin("legacy_pairing", label) && return "legacy pairing"
    return String(label)
end

function duration_seconds(value::AbstractString)
    pieces = split(value, ':')
    if length(pieces) == 2
        return 60 * parse(Int, pieces[1]) + parse(Float64, pieces[2])
    end
    length(pieces) == 3 || error("unsupported duration $value")
    first_piece = pieces[1]
    days = 0
    hours = 0
    if occursin('-', first_piece)
        day_piece, hour_piece = split(first_piece, '-'; limit=2)
        days = parse(Int, day_piece)
        hours = parse(Int, hour_piece)
    else
        hours = parse(Int, first_piece)
    end
    return 86400 * days + 3600 * hours + 60 * parse(Int, pieces[2]) + parse(Float64, pieces[3])
end

function memory_gib(value::AbstractString)
    isempty(value) && return NaN
    endswith(value, "K") && return parse(Float64, chop(value; tail=1)) / 1024^2
    endswith(value, "M") && return parse(Float64, chop(value; tail=1)) / 1024
    endswith(value, "G") && return parse(Float64, chop(value; tail=1))
    error("unsupported memory value $value")
end

config_path = joinpath(STAGE2_DIR, "config.toml")
bank_path = joinpath(STAGE2_RESULTS, "candidate_bank.h5")
normal_reference_path = joinpath(STAGE2_RESULTS, "normal_reference.h5")
discovery_path = joinpath(STAGE2_RESULTS, "stage2_discovery.h5")

project = load_settings(config_path)
bank = read_stage2_candidate_bank(bank_path)
reference = h5open(normal_reference_path, "r") do file
    String(read(file, "artifact_kind")) == "bare_ladder_stage2_normal_reference" ||
        error("unexpected Stage 2 normal-reference artifact")
    Bool(read(file, "complete")) || error("incomplete Stage 2 normal-reference artifact")
    Bool(read(file, "scientifically_accepted")) ||
        error("Stage 2 normal reference did not pass its scientific gates")
    group = file["zero_field_correlations"]
    correlations = CorrelationState(
        Float64.(read(group, "pair")),
        Float64.(read(group, "exchange_down")),
        Float64.(read(group, "exchange_up")),
        Float64.(read(group, "density_down")),
        Float64.(read(group, "density_up")),
    )
    (;
        correlations,
        density=Float64(read(file, "density")),
        energy=Float64(read(file, "dmrg/energy")),
        recorded_full_bank_sha256=String(read(file, "sources/candidate_bank_sha256")),
    )
end
geometries = (:cubic_frustrated, :cubic_unfrustrated, :square)

bare_fields = Dict{Symbol,FieldState}()
bare_splits = Dict{Symbol,NamedTuple}()
field_rows = NamedTuple[]
profile_rows = NamedTuple[]
beta_class_rows = NamedTuple[]

for geometry in geometries
    geometry_model = LadderMPSMFT._model_with_geometry(
        project.model,
        geometry;
        ep_signed=bank.kernel_pair_binding,
    )
    fields = mean_fields_from_correlations(reference.correlations, geometry_model; threshold=0.0)
    split = split_background(fields, geometry_model)
    bare_fields[geometry] = fields
    bare_splits[geometry] = split
    norms = component_norms(fields, geometry_model)
    background_norms = component_norms(split.background, geometry_model)
    residual_norms = component_norms(split.residual, geometry_model)
    profiles = mu_profiles(fields.mu_cdw, geometry_model.L)
    beta_nn = nearest_neighbor_beta_profile(fields, geometry_model)
    push!(field_rows, (;
        geometry=String(geometry),
        alpha_norm=norms.alpha,
        beta_norm=norms.beta,
        mu_cdw_norm=norms.mu_cdw,
        total_norm=norms.total,
        background_norm=background_norms.total,
        modulation_norm=residual_norms.total,
        modulation_fraction=residual_norms.total / norms.total,
        beta_background_norm=background_norms.beta,
        beta_modulation_norm=residual_norms.beta,
        beta_modulation_fraction=residual_norms.beta / norms.beta,
        mu_uniform_value=split.mu_average,
        mu_uniform_norm=background_norms.mu_cdw,
        mu_modulation_norm=residual_norms.mu_cdw,
        charge_even_modulation_rms=std(profiles.charge_even; corrected=false),
        charge_even_modulation_max=maximum(abs, profiles.charge_even .- mean(profiles.charge_even)),
        charge_odd_rms=sqrt(mean(abs2, profiles.charge_odd)),
        spin_even_rms=sqrt(mean(abs2, profiles.spin_even)),
        spin_odd_rms=sqrt(mean(abs2, profiles.spin_odd)),
        beta_nn_mean=mean(beta_nn),
        beta_nn_modulation_rms=std(beta_nn; corrected=false),
        alpha_max=maximum(abs, fields.alpha),
        beta_max=maximum(abs, fields.beta),
        mu_cdw_max=maximum(abs, fields.mu_cdw),
    ))
    for rung in 1:geometry_model.L
        push!(profile_rows, (;
            geometry=String(geometry),
            rung,
            charge_even=profiles.charge_even[rung],
            charge_even_centered=profiles.charge_even[rung] - mean(profiles.charge_even),
            charge_odd=profiles.charge_odd[rung],
            spin_even=profiles.spin_even[rung],
            spin_odd=profiles.spin_odd[rung],
            beta_nn=rung <= length(beta_nn) ? beta_nn[rung] : NaN,
            beta_nn_centered=rung <= length(beta_nn) ? beta_nn[rung] - mean(beta_nn) : NaN,
        ))
    end
    for offset in 0:geometry_model.r_range, bond_class in (:same_leg, :cross_leg)
        values = Float64[]
        for spin in 1:2, leg in 1:2, other_leg in 1:2
            (bond_class == :same_leg) == (leg == other_leg) || continue
            for rung in 1:(geometry_model.L - offset)
                push!(values, fields.beta[spin, rung, rung + offset, leg, other_leg])
            end
        end
        push!(beta_class_rows, (;
            geometry=String(geometry),
            offset,
            bond_class=String(bond_class),
            count=length(values),
            mean=mean(values),
            rms=sqrt(mean(abs2, values)),
            spatial_std=std(values; corrected=false),
            minimum=minimum(values),
            maximum=maximum(values),
        ))
    end
end

write_tsv(
    joinpath(REPORT_DIR, "bare_field_summary.tsv"),
    field_rows,
    propertynames(first(field_rows)),
)
write_tsv(
    joinpath(REPORT_DIR, "bare_field_profiles.tsv"),
    profile_rows,
    propertynames(first(profile_rows)),
)
write_tsv(
    joinpath(REPORT_DIR, "beta_bond_class_summary.tsv"),
    beta_class_rows,
    propertynames(first(beta_class_rows)),
)

spectrum_rows = NamedTuple[]
for geometry in geometries
    profiles = mu_profiles(bare_fields[geometry].mu_cdw, project.model.L)
    signal = profiles.charge_even .- mean(profiles.charge_even)
    for mode_number in 1:(project.model.L - 1)
        template = matched_mode_profile(project.model; mode_number, phase_pi=0.0)
        coefficient = dot(signal, template) / dot(template, template)
        push!(spectrum_rows, (;
            geometry=String(geometry),
            mode_number,
            q_over_pi=mode_number / (project.model.L - 1),
            coefficient,
            absolute_profile_cosine=abs(centered_cosine(signal, template)),
        ))
    end
end
write_tsv(
    joinpath(REPORT_DIR, "bare_charge_mode_overlap.tsv"),
    spectrum_rows,
    propertynames(first(spectrum_rows)),
)

square_model = LadderMPSMFT._model_with_geometry(
    project.model,
    :square;
    ep_signed=bank.kernel_pair_binding,
)
square_bare = bare_fields[:square]
square_modulation = bare_splits[:square].residual

seed_rows = NamedTuple[]
seed_profile_rows = NamedTuple[]
first_step_rows = NamedTuple[]
scf_history_rows = NamedTuple[]
seed_fields = Dict{String,FieldState}()
endpoint_rows = NamedTuple[]
endpoint_fields = Dict{String,FieldState}()

results_directory = joinpath(SCF_DIR, "results")
for branch in sort(readdir(results_directory; join=true))
    isdir(branch) || continue
    branch_label = basename(branch)
    state_path = find_single_state(branch)
    h5open(state_path, "r") do file
        initial = read_fields(file["fields/initial"])
        first_measured = first_history_fields(file, "measured")
        first_delta = subtract_fields(first_measured, square_bare)
        endpoint = read_fields(file["fields/restart"])
        endpoint_delta = subtract_fields(endpoint, square_bare)
        endpoint_normal = FieldState(
            zeros(Float64, size(endpoint.alpha)),
            endpoint.beta,
            endpoint.mu_cdw,
        )
        endpoint_normal_delta = subtract_fields(endpoint_normal, square_bare)
        seed_fields[branch_label] = initial
        endpoint_fields[branch_label] = endpoint
        initial_norms = component_norms(initial, square_model)
        initial_profiles = mu_profiles(initial.mu_cdw, square_model.L)
        bare_profiles = mu_profiles(square_bare.mu_cdw, square_model.L)
        full_projection = field_metric_dot(square_bare, initial, square_model) /
            max(field_metric_dot(initial, initial, square_model), eps(Float64))
        modulation_projection = field_metric_dot(square_modulation, initial, square_model) /
            max(field_metric_dot(initial, initial, square_model), eps(Float64))
        push!(seed_rows, (;
            branch=branch_label,
            seed=short_seed_label(branch_label),
            total_norm=initial_norms.total,
            alpha_norm=initial_norms.alpha,
            beta_norm=initial_norms.beta,
            mu_cdw_norm=initial_norms.mu_cdw,
            bare_full_cosine=field_cosine(square_bare, initial, square_model),
            bare_modulation_cosine=field_cosine(square_modulation, initial, square_model),
            bare_full_projection_in_seed_units=full_projection,
            bare_modulation_projection_in_seed_units=modulation_projection,
            charge_even_profile_cosine=centered_cosine(
                bare_profiles.charge_even,
                initial_profiles.charge_even,
            ),
        ))
        for rung in 1:square_model.L
            push!(seed_profile_rows, (;
                branch=branch_label,
                seed=short_seed_label(branch_label),
                rung,
                charge_even=initial_profiles.charge_even[rung],
                charge_odd=initial_profiles.charge_odd[rung],
                spin_even=initial_profiles.spin_even[rung],
                spin_odd=initial_profiles.spin_odd[rung],
            ))
        end
        delta_norms = component_norms(first_delta, square_model)
        measured_norms = component_norms(first_measured, square_model)
        gain = field_metric_dot(first_delta, initial, square_model) /
            max(field_metric_dot(initial, initial, square_model), eps(Float64))
        history_mu = Float64.(read(file, "history/chemical_potential"))
        history_density = Float64.(read(file, "history/density"))
        history_mu_evaluations = Int.(read(file, "history/mu_evaluations"))
        history_wall = Float64.(read(file, "history/wall_seconds"))
        history_residual = Float64.(read(file, "history/field_rel_residual"))
        for iteration in eachindex(history_mu)
            push!(scf_history_rows, (;
                branch=branch_label,
                seed=short_seed_label(branch_label),
                iteration,
                chemical_potential=history_mu[iteration],
                density=history_density[iteration],
                mu_evaluations=history_mu_evaluations[iteration],
                wall_hours=history_wall[iteration] / 3600,
                field_rel_residual=history_residual[iteration],
            ))
        end
        push!(first_step_rows, (;
            branch=branch_label,
            seed=short_seed_label(branch_label),
            first_measured_norm=measured_norms.total,
            distance_from_stage2_bare=delta_norms.total,
            relative_distance_from_stage2_bare=delta_norms.total / field_metric_norm(square_bare, square_model),
            delta_alpha_norm=delta_norms.alpha,
            delta_beta_norm=delta_norms.beta,
            delta_mu_cdw_norm=delta_norms.mu_cdw,
            response_gain_along_seed=gain,
            first_chemical_potential=history_mu[1],
            first_density=history_density[1],
            first_mu_evaluations=history_mu_evaluations[1],
            first_iteration_wall_hours=history_wall[1] / 3600,
            final_chemical_potential=Float64(read(file, "chemical_potential")),
            iteration_count=length(history_mu),
        ))
        endpoint_norms = component_norms(endpoint, square_model)
        endpoint_delta_norms = component_norms(endpoint_delta, square_model)
        endpoint_normal_delta_norms = component_norms(endpoint_normal_delta, square_model)
        endpoint_profiles = mu_profiles(endpoint.mu_cdw, square_model.L)
        endpoint_split = split_background(endpoint, square_model)
        push!(endpoint_rows, (;
            branch=branch_label,
            seed=short_seed_label(branch_label),
            accepted=Bool(read(file, "accepted")),
            status=String(read(file, "status")),
            iterations=length(history_mu),
            total_norm=endpoint_norms.total,
            alpha_norm=endpoint_norms.alpha,
            beta_norm=endpoint_norms.beta,
            mu_cdw_norm=endpoint_norms.mu_cdw,
            background_norm=field_metric_norm(endpoint_split.background, square_model),
            nonuniform_norm=field_metric_norm(endpoint_split.residual, square_model),
            distance_from_bare=endpoint_delta_norms.total,
            normal_distance_from_bare=endpoint_normal_delta_norms.total,
            cosine_with_bare=field_cosine(endpoint, square_bare, square_model),
            charge_even_modulation_rms=std(endpoint_profiles.charge_even; corrected=false),
            spin_odd_rms=sqrt(mean(abs2, endpoint_profiles.spin_odd)),
            alpha_max=maximum(abs, endpoint.alpha),
            beta_max=maximum(abs, endpoint.beta),
            mu_cdw_max=maximum(abs, endpoint.mu_cdw),
            chemical_potential=Float64(read(file, "chemical_potential")),
            corrected_energy=Float64(read(file, "solution_target_density_corrected_variational_energy")),
            fixed_point_rel_residual=Float64(read(file, "fixed_point_rel_residual")),
        ))
    end
end

endpoint_pair_rows = NamedTuple[]
endpoint_labels = sort(collect(keys(endpoint_fields)))
for left_index in eachindex(endpoint_labels)
    for right_index in (left_index + 1):length(endpoint_labels)
        left_label = endpoint_labels[left_index]
        right_label = endpoint_labels[right_index]
        left = endpoint_fields[left_label]
        right = endpoint_fields[right_label]
        delta = subtract_fields(left, right)
        left_normal = FieldState(zeros(Float64, size(left.alpha)), left.beta, left.mu_cdw)
        right_normal = FieldState(zeros(Float64, size(right.alpha)), right.beta, right.mu_cdw)
        normal_delta = subtract_fields(left_normal, right_normal)
        push!(endpoint_pair_rows, (;
            left_seed=short_seed_label(left_label),
            right_seed=short_seed_label(right_label),
            full_distance=field_metric_norm(delta, square_model),
            normal_distance=field_metric_norm(normal_delta, square_model),
            alpha_distance=component_norm(delta.alpha, square_model),
            field_cosine=field_cosine(left, right, square_model),
            corrected_energy_difference=abs(
                only(row.corrected_energy for row in endpoint_rows if row.branch == left_label) -
                only(row.corrected_energy for row in endpoint_rows if row.branch == right_label),
            ),
        ))
    end
end

write_tsv(joinpath(REPORT_DIR, "current_seed_summary.tsv"), seed_rows, propertynames(first(seed_rows)))
write_tsv(joinpath(REPORT_DIR, "current_seed_profiles.tsv"), seed_profile_rows, propertynames(first(seed_profile_rows)))
write_tsv(joinpath(REPORT_DIR, "first_step_comparison.tsv"), first_step_rows, propertynames(first(first_step_rows)))
write_tsv(joinpath(REPORT_DIR, "scf_iteration_history.tsv"), scf_history_rows, propertynames(first(scf_history_rows)))
write_tsv(joinpath(REPORT_DIR, "scf_endpoint_summary.tsv"), endpoint_rows, propertynames(first(endpoint_rows)))
write_tsv(joinpath(REPORT_DIR, "scf_endpoint_pairwise.tsv"), endpoint_pair_rows, propertynames(first(endpoint_pair_rows)))

basis_labels = Dict{Int,String}()
for item in bank.basis
    basis_labels[item.basis_index] = item.label
end

eigenvalue_rows = NamedTuple[]
composition_rows = NamedTuple[]
mode_seed_overlap_rows = NamedTuple[]
leakage_rows = NamedTuple[]

h5open(discovery_path, "r") do file
    for geometry in geometries
        geometry_group = file["geometry_maps/$(String(geometry))"]
        leakage = Float64.(read(geometry_group, "column_leakage_relative"))
        beta_fraction = Float64.(read(geometry_group, "column_beta_fraction"))
        for basis_index in eachindex(leakage)
            push!(leakage_rows, (;
                geometry=String(geometry),
                basis_index,
                basis_label=basis_labels[basis_index],
                leakage_relative=leakage[basis_index],
                beta_fraction=beta_fraction[basis_index],
            ))
        end
        for block in (:normal, :pair)
            indices = block == :normal ? bank.normal_indices : bank.pair_indices
            group = geometry_group[String(block)]
            values = complex.(
                Float64.(read(group, "eigenvalue_real")),
                Float64.(read(group, "eigenvalue_imag")),
            )
            vectors = complex.(
                Float64.(read(group, "eigenvector_real")),
                Float64.(read(group, "eigenvector_imag")),
            )
            residuals = Float64.(read(group, "residual_norm"))
            critical_tp = Float64.(read(group, "critical_tp"))
            for rank in eachindex(values)
                vector = copy(vectors[:, rank])
                pivot = argmax(abs.(vector))
                abs(vector[pivot]) > 0 && (vector .*= conj(vector[pivot]) / abs(vector[pivot]))
                push!(eigenvalue_rows, (;
                    geometry=String(geometry),
                    block=String(block),
                    rank,
                    eigenvalue_real=real(values[rank]),
                    eigenvalue_imag=imag(values[rank]),
                    abs_eigenvalue=abs(values[rank]),
                    critical_tp=critical_tp[rank],
                    residual_norm=residuals[rank],
                    unstable=abs(values[rank]) > 1,
                    recurrence_character=real(values[rank]) < -1 ? "oscillatory" :
                        real(values[rank]) > 1 ? "monotone" : "subcritical",
                ))
                if rank <= 3
                    order = sortperm(abs.(vector); rev=true)
                    cumulative = 0.0
                    for local_index in order
                        coefficient = vector[local_index]
                        weight = abs2(coefficient)
                        cumulative += weight
                        push!(composition_rows, (;
                            geometry=String(geometry),
                            block=String(block),
                            rank,
                            eigenvalue_real=real(values[rank]),
                            basis_index=indices[local_index],
                            basis_label=basis_labels[indices[local_index]],
                            coefficient_real=real(coefficient),
                            coefficient_imag=imag(coefficient),
                            weight,
                            cumulative_weight=cumulative,
                        ))
                    end
                end
                if rank == 1
                    mode_fields = zero_field_state(square_model)
                    for local_index in eachindex(indices)
                        mode_fields = add_fields(
                            mode_fields,
                            scale_fields(bank.basis[indices[local_index]].fields, real(vector[local_index])),
                        )
                    end
                    for (branch_label, seed) in seed_fields
                        push!(mode_seed_overlap_rows, (;
                            geometry=String(geometry),
                            block=String(block),
                            rank,
                            eigenvalue_real=real(values[rank]),
                            branch=branch_label,
                            seed=short_seed_label(branch_label),
                            absolute_cosine=abs(field_cosine(mode_fields, seed, square_model)),
                            signed_cosine=field_cosine(mode_fields, seed, square_model),
                        ))
                    end
                end
            end
        end
    end
end

write_tsv(joinpath(REPORT_DIR, "eigenvalue_summary.tsv"), eigenvalue_rows, propertynames(first(eigenvalue_rows)))
write_tsv(joinpath(REPORT_DIR, "eigenmode_composition.tsv"), composition_rows, propertynames(first(composition_rows)))
write_tsv(joinpath(REPORT_DIR, "dominant_mode_seed_overlap.tsv"), mode_seed_overlap_rows, propertynames(first(mode_seed_overlap_rows)))
write_tsv(joinpath(REPORT_DIR, "projected_leakage.tsv"), leakage_rows, propertynames(first(leakage_rows)))

allocation_rows = Dict{String,NamedTuple}()
for line in readlines(joinpath(STAGE2_DIR, "logs", "sacct-allocations.txt"))
    fields = split(line, '|'; keepempty=true)
    length(fields) >= 7 || continue
    elapsed_seconds = parse(Float64, fields[5])
    allocated_logical_cpus = parse(Int, fields[6])
    requested_memory_gib = memory_gib(fields[7])
    allocation_rows[fields[1]] = (;
        job_id=fields[1],
        job_name=fields[2],
        elapsed_seconds,
        allocated_logical_cpus,
        requested_memory_gib,
    )
end

step_rows = Dict{String,NamedTuple}()
for line in readlines(joinpath(STAGE2_DIR, "logs", "sacct-steps.txt"))
    fields = split(line, '|'; keepempty=true)
    length(fields) >= 10 || continue
    endswith(fields[1], ".0") || continue
    base_job = first(split(fields[1], '.'; limit=2))
    isempty(fields[8]) && continue
    step_rows[base_job] = (;
        total_cpu_seconds=duration_seconds(fields[8]),
        max_rss_gib=memory_gib(fields[10]),
        step_allocated_logical_cpus=parse(Int, fields[6]),
    )
end

resource_rows = NamedTuple[]
for job_id in sort(collect(keys(allocation_rows)))
    allocation = allocation_rows[job_id]
    startswith(allocation.job_name, "lmf-s2-normal") ||
        startswith(allocation.job_name, "lmf-s2-zero") ||
        startswith(allocation.job_name, "lmf-s2-pair") || continue
    haskey(step_rows, job_id) || continue
    step = step_rows[job_id]
    average_busy_cores = step.total_cpu_seconds / allocation.elapsed_seconds
    category = occursin("pair", allocation.job_name) ? "pair" : "normal"
    push!(resource_rows, (;
        job_id,
        job_name=allocation.job_name,
        category,
        elapsed_hours=allocation.elapsed_seconds / 3600,
        allocated_logical_cpus=allocation.allocated_logical_cpus,
        billed_node_fraction=allocation.allocated_logical_cpus / 256,
        node_hours=allocation.elapsed_seconds / 3600 * allocation.allocated_logical_cpus / 256,
        step_allocated_logical_cpus=step.step_allocated_logical_cpus,
        total_cpu_hours=step.total_cpu_seconds / 3600,
        average_busy_cores,
        step_cpu_efficiency=average_busy_cores / step.step_allocated_logical_cpus,
        billed_cpu_efficiency=average_busy_cores / allocation.allocated_logical_cpus,
        requested_memory_gib=allocation.requested_memory_gib,
        max_rss_gib=step.max_rss_gib,
        memory_request_utilization=step.max_rss_gib / allocation.requested_memory_gib,
    ))
end
write_tsv(joinpath(REPORT_DIR, "resource_efficiency.tsv"), resource_rows, propertynames(first(resource_rows)))

total_node_hours = sum(row.node_hours for row in resource_rows)
overhead_node_hours = sum(
    allocation.elapsed_seconds / 3600 * allocation.allocated_logical_cpus / 256
    for allocation in values(allocation_rows)
    if !haskey(step_rows, allocation.job_id) ||
        !(startswith(allocation.job_name, "lmf-s2-normal") ||
          startswith(allocation.job_name, "lmf-s2-zero") ||
          startswith(allocation.job_name, "lmf-s2-pair"))
)
total_node_hours += overhead_node_hours
normal_node_hours = sum(row.node_hours for row in resource_rows if row.category == "normal")
pair_node_hours = sum(row.node_hours for row in resource_rows if row.category == "pair")

open(joinpath(REPORT_DIR, "analysis_summary.txt"), "w") do io
    println(io, "stage2_reference_energy=$(reference.energy)")
    println(io, "stage2_reference_density=$(reference.density)")
    println(io, "stage2_reference_chemical_potential=$(bank.chemical_potential)")
    println(io, "kernel_pair_binding=$(bank.kernel_pair_binding)")
    println(io, "kernel_prefactor=$(bank.kernel_prefactor)")
    println(io, "square_bare_field_norm=$(field_metric_norm(square_bare, square_model))")
    println(io, "square_bare_modulation_norm=$(field_metric_norm(square_modulation, square_model))")
    println(io, "total_measured_node_hours=$total_node_hours")
    println(io, "normal_measured_node_hours=$normal_node_hours")
    println(io, "pair_measured_node_hours=$pair_node_hours")
    println(io, "overhead_measured_node_hours=$overhead_node_hours")
end

println("analysis outputs written to $REPORT_DIR")
