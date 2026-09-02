module LadderMPSMFT

using CSV
using Dates
using HDF5
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf
using Random
using SHA
using Statistics
using TOML

include("Types.jl")
include("Geometry.jl")
include("EpRegistry.jl")
include("Config.jl")
include("Device.jl")
include("Mixing.jl")
include("Convergence.jl")
include("MeanField.jl")
include("Variational.jl")
include("Diagnostics.jl")
include("Provenance.jl")
include("Storage.jl")
include("Solver.jl")
include("Backbone.jl")
include("BareStage1.jl")
include("BareStage2.jl")
include("Selection.jl")

export SUPPORTED_GEOMETRIES,
       ModelSettings,
       DMRGSettings,
       MixingSettings,
       ConvergenceSettings,
       RuntimeSettings,
       RunSettings,
       ProjectSettings,
       LadderBackboneSettings,
       BareStage1Settings,
       BareStage2Settings,
       FieldState,
       CorrelationState,
       EnergyBreakdown,
       ConvergenceDiagnostic,
       IterationRecord,
       EpRecord,
       EpSelection,
       normalize_geometry,
       rung_leg_to_site,
       site_to_rung_leg,
       density_kernel,
       load_ep_registry,
       lookup_ep,
       validate_weak_coupling,
       load_settings,
       validate_settings,
       ensure_backend!,
       gpu_linalg_preflight!,
       move_to_backend,
       move_to_cpu,
       backend_metadata,
       density_product_state,
       initial_fields,
       matched_mode_profile,
       initial_mode_wavevector_pi,
       resolved_initial_leg_parity,
       field_l2_per_physical_site,
       initial_seed_metadata,
       calculate_mean_fields,
       mean_fields_from_correlations,
       configure_threading!,
       build_mf_mpo,
       field_energy_components,
       variational_energy,
       detect_period,
       assess_convergence,
       mix_fields!,
       compute_ladder_diagnostics,
       sector_resolved_gaps,
       load_ladder_backbone_settings,
       load_bare_stage1_settings,
       load_bare_stage2_settings,
       backbone_siteinds,
       backbone_sector_keys,
       backbone_sector_label,
       spread_fixed_sector_product_state,
       last_five_sweep_change,
       run_backbone_sector,
       write_backbone_stage_checkpoint,
       read_backbone_stage_checkpoint,
       write_backbone_sector,
       read_backbone_sector,
       backbone_energy_summary,
       backbone_validity,
       backbone_chi_dependence,
       write_backbone_artifact,
       read_backbone_ground_state,
       connected_covariance_matrix,
       leg_parity_covariance,
       covariance_eigensystem,
       correlation_exponent_with_window_uncertainty,
       compute_bare_stage1,
       write_bare_stage1,
       zero_field_state,
       field_metric_dot,
       field_metric_norm,
       scale_fields,
       subtract_fields,
       normalize_fields,
       build_stage2_candidates,
       orthonormalize_stage2_candidates,
       write_stage2_candidate_bank,
       read_stage2_candidate_bank,
       run_stage2_normal_reference,
       read_stage2_normal_reference,
       run_stage2_pair_reference,
       read_stage2_pair_reference,
       field_conjugate_expectation,
       run_stage2_discovery_probe,
       assemble_stage2_discovery,
       read_stage2_validation_direction,
       run_stage2_validation_probe,
       assemble_stage2_validation,
       write_diagnostics,
       write_sector_gaps,
       write_checkpoint,
       write_stateless_copy,
       mirror_stateless_tree,
       read_checkpoint,
       read_field_history,
       read_inherited_fields,
       read_orbit_phase_states,
       collect_provenance,
       implementation_fingerprint,
       tree_fingerprint,
       initial_seed_fingerprint,
       find_mu_for_density,
       run_dmrg_ground,
       run_scf,
       select_completed_runs,
       compare_variational_branches

end
