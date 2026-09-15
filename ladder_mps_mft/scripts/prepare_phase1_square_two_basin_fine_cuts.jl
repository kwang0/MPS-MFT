#!/usr/bin/env julia
include("prepare_phase1_two_basin_grid.jl")

const TWO_BASIN_FINE_CUTS = vcat(
    [(t0=1.4,V=V,cut="fixed_t014",axis=:V,lower=-.2,upper=0.,
      point="t01400_vm$(lpad(round(Int,-1000V),4,'0'))",mu_initial=1.65+2.75V)
     for V in (-.05,-.1,-.15)],
    [(t0=t0,V=-.4,cut="fixed_vm04",axis=:t0,lower=1.2,upper=1.4,
      point="t0$(round(Int,1000t0))_vm0400",mu_initial=.55)
     for t0 in (1.25,1.3,1.35)])

function prepare_square_two_basin_fine_cuts(base,reference,control,full,run_id)
    settings = load_settings(base)
    (settings.run.max_iterations,settings.convergence.minimum_iterations,
     settings.convergence.stable_iterations) == (60,40,10) || error("expected 60/40/10 fine-cut contract")
    # Pin the coarse input table so a future registry update cannot silently
    # replace the user's requested endpoint interpolation with another model.
    LadderMPSMFT.sha256_file(settings.model.ep_source) ==
        "2209bd2ca3c1ad02c0e542d1a9d63ecf90fdfa49120ad9cc3af599a5b4bc1f0e" || error("coarse E_p registry changed")
    return prepare_two_basin_grid(base,reference,control,full,run_id;
        stage="cuts",geometry=:square,coordinates=TWO_BASIN_FINE_CUTS)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS)==5 || error("usage: BASE REFERENCES CONTROL_RUN FULL_RUN RUN_ID")
    prepare_square_two_basin_fine_cuts(ARGS...)
end
