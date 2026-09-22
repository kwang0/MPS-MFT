#!/usr/bin/env julia
include("prepare_phase1_two_basin_grid.jl")

# Four independently seeded coordinates, not parameter continuations.
const SQUARE_TP_SCAN = [
    (t0=1.4, V=V, tp=tp, cut=V == 0 ? "fixed_t014_v000" : "fixed_t014_vm02",
     point="tp$(lpad(round(Int,1000tp),3,'0'))_t014_$(V == 0 ? "v000" : "vm02")")
    for (V,tp) in ((0.0,0.06), (0.0,0.08), (-0.2,0.12), (-0.2,0.14))
]

function prepare_square_tp_scan(base, reference, control, full, run_id)
    settings = load_settings(base)
    (settings.model.t, settings.model.t0, settings.model.V, settings.model.r_range) ==
        (1.0,1.4,0.0,4) || error("expected square t0=1.4 reference base")
    settings.model.spatial_cell == :one_ladder || error("expected one-ladder square scan")
    (settings.run.max_iterations, settings.convergence.minimum_iterations,
     settings.convergence.stable_iterations) == (60,40,10) || error("expected 60/40/10 scan contract")
    LadderMPSMFT.sha256_file(settings.model.ep_source) ==
        "2209bd2ca3c1ad02c0e542d1a9d63ecf90fdfa49120ad9cc3af599a5b4bc1f0e" || error("E_p registry changed")
    return prepare_two_basin_grid(base, reference, control, full, run_id;
        stage="tp_scan", geometry=:square, coordinates=SQUARE_TP_SCAN)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 5 || error("usage: BASE REFERENCES CONTROL_RUN FULL_RUN RUN_ID")
    prepare_square_tp_scan(ARGS...)
end
