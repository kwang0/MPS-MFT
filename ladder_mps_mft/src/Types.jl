const SUPPORTED_GEOMETRIES = (:cubic_frustrated, :cubic_unfrustrated, :square)

Base.@kwdef struct ModelSettings
    L::Int = 64
    t::Float64 = 1.0
    U::Float64 = 8.0
    V::Float64 = 0.0
    t0::Float64 = 1.0
    tp::Float64 = 0.1
    density::Float64 = 0.9375
    mu_initial::Float64 = 0.0
    r_range::Int = 4
    geometry::Symbol = :cubic_frustrated
    ep::Float64 = NaN
    ep_signed::Float64 = NaN
    ep_source::String = ""
    ep_mode::Symbol = :exact
    ep_t0_lower::Float64 = NaN
    ep_t0_upper::Float64 = NaN
    ep_lower_signed::Float64 = NaN
    ep_upper_signed::Float64 = NaN
    ep_interpolation_weight::Float64 = 0.0
    ep_lower_chi::Int = 0
    ep_upper_chi::Int = 0
end

Base.@kwdef struct DMRGSettings
    nsweeps::Int = 12
    maxdim::Int = 200
    cutoff::Float64 = 1e-10
    energy_tol::Float64 = 1e-8
    eigsolve_krylovdim::Int = 8
    max_time_seconds::Float64 = 23.5 * 3600
    output_level::Int = 1
    mu_density_tol::Float64 = 2e-4
    mu_max_iterations::Int = 16
    mu_bracket_step::Float64 = 0.05
    mu_bracket_growth::Float64 = 2.0
    mu_interval_tol::Float64 = 1e-6
end

Base.@kwdef struct MixingSettings
    method::Symbol = :anderson
    damping::Float64 = 0.5
    minimum_damping::Float64 = 0.05
    maximum_damping::Float64 = 0.8
    memory::Int = 5
    regularization::Float64 = 1e-10
    adaptive::Bool = true
end

Base.@kwdef struct ConvergenceSettings
    field_abs_tol::Float64 = 1e-6
    field_rel_tol::Float64 = 5e-3
    density_tol::Float64 = 1e-5
    variational_energy_tol::Float64 = 1e-7
    hamiltonian_identity_tol::Float64 = 1e-9
    effective_energy_consistency_tol::Float64 = 1e-6
    stable_iterations::Int = 2
    max_period::Int = 8
    period_repeats::Int = 3
    period_abs_tol::Float64 = 2e-6
    period_rel_tol::Float64 = 1e-2
    unmixed_cycle_probe::Bool = true
    probe_max_period::Int = 2
    probe_iterations::Int = 20
    accepted_periods::Vector{Int} = [1, 2]
    orbit_bulk_fraction::Float64 = 0.5
    cycle_action::Symbol = :stop
    stagnation_window::Int = 10
    stagnation_min_relative_improvement::Float64 = 1e-2
    divergence_factor::Float64 = 8.0
end

Base.@kwdef struct RuntimeSettings
    backend::Symbol = :cpu
    tensor_scalar_type::Symbol = :float64
    blas_threads::Int = 1
    strided_threads::Int = 1
    threaded_blocksparse::Bool = true
    conserve_sz::Bool = true
    conserve_nfparity::Bool = true
end

Base.@kwdef struct RunSettings
    output_directory::String = "output"
    branch_label::String = "independent"
    preparation::String = "independent_seed"
    direction::String = "none"
    seed_label::String = "seed_1"
    random_seed::Int = 1
    initial_seed::Symbol = :pairing
    initial_amplitude::Float64 = 1e-3
    inherit_from::Union{Nothing,String} = nothing
    inherit_sha256::Union{Nothing,String} = nothing
    parent_checkpoint::Union{Nothing,String} = nothing
    parent_sha256::Union{Nothing,String} = nothing
    parent_orbit_phase::Union{Nothing,Int} = nothing
    resume_checkpoint::Union{Nothing,String} = nothing
    resume_sha256::Union{Nothing,String} = nothing
    max_iterations::Int = 80
    save_every::Int = 1
    require_accepted_solution::Bool = true
    allow_unbound_ep::Bool = false
    quick_diagnostics::Bool = true
    full_pair_correlations::Bool = false
end

Base.@kwdef struct ProjectSettings
    model::ModelSettings
    dmrg::DMRGSettings = DMRGSettings()
    mixing::MixingSettings = MixingSettings()
    convergence::ConvergenceSettings = ConvergenceSettings()
    runtime::RuntimeSettings = RuntimeSettings()
    run::RunSettings = RunSettings()
    config_path::String = ""
end

struct FieldState
    alpha::Array{Float64,4}
    beta::Array{Float64,5}
    mu_cdw::Array{Float64,2}
end

Base.copy(fields::FieldState) = FieldState(copy(fields.alpha), copy(fields.beta), copy(fields.mu_cdw))

struct CorrelationState
    pair::Matrix{Float64}
    exchange_down::Matrix{Float64}
    exchange_up::Matrix{Float64}
    density_down::Vector{Float64}
    density_up::Vector{Float64}
end

Base.@kwdef struct EnergyBreakdown
    effective_eigenvalue::Float64
    effective_expectation::Float64
    effective_eigenvalue_error::Float64
    bare_ladder_energy::Float64
    reconstructed_bare_ladder_energy::Float64
    hamiltonian_identity_error::Float64
    chemical_potential_term::Float64
    pair_field_energy::Float64
    exchange_field_energy::Float64
    density_field_energy::Float64
    pair_transverse_energy::Float64
    exchange_transverse_energy::Float64
    density_transverse_energy::Float64
    double_counting_correction::Float64
    reconstructed_variational_energy::Float64
    direct_variational_energy::Float64
    variational_consistency_error::Float64
    canonical_variational_energy::Float64
    grand_potential::Float64
end

Base.@kwdef struct ConvergenceDiagnostic
    status::Symbol = :insufficient_history
    accepted::Bool = false
    reason::String = "insufficient history"
    solution_kind::Symbol = :none
    fundamental_period::Int = 0
    orbit_validated::Bool = false
    unmixed_probe::Bool = false
    solution_canonical_variational_energy::Float64 = NaN
    orbit_energy_spread::Float64 = NaN
    orbit_density_contrast::Float64 = NaN
    fixed_point_abs_residual::Float64 = Inf
    fixed_point_rel_residual::Float64 = Inf
    cycle_abs_residual::Float64 = Inf
    cycle_rel_residual::Float64 = Inf
    density_error::Float64 = Inf
    variational_energy_change::Float64 = Inf
    hamiltonian_identity_error_per_site::Float64 = Inf
    effective_eigenvalue_error_per_site::Float64 = Inf
    best_iteration::Int = 0
end

Base.@kwdef struct IterationRecord
    iteration::Int
    update_mode::Symbol = :unknown
    applied::FieldState
    measured::FieldState
    correlations::CorrelationState
    density::Float64
    chemical_potential::Float64
    mu_search_status::Symbol
    mu_evaluations::Int
    mu_density_converged::Bool
    effective_energy::Float64
    variational::EnergyBreakdown
    field_abs_residual::Float64
    field_rel_residual::Float64
    wall_seconds::Float64
end

struct EpRecord
    L::Int
    U::Float64
    V::Float64
    t0::Float64
    density::Float64
    chi::Int
    E_N::Float64
    E_p::Float64
    rel_diff::Float64
end

Base.@kwdef struct EpSelection
    record::EpRecord
    denominator::Float64
    source_path::String
    bound_pair::Bool
    tp_below_pair_binding::Bool
    mode::Symbol = :exact
    lower_record::EpRecord = record
    upper_record::EpRecord = record
    interpolation_weight::Float64 = 0.0
end
