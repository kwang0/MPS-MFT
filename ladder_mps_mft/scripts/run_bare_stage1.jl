#!/usr/bin/env julia

using LadderMPSMFT
using Printf

length(ARGS) == 3 || error(
    "usage: julia --project=. scripts/run_bare_stage1.jl " *
    "CONFIG.toml BACKBONE.h5 OUTPUT.h5",
)
config_path, backbone_path, output_path = abspath.(ARGS)
ispath(output_path) && error("refusing to overwrite immutable Stage 1 artifact: $output_path")
settings = load_settings(config_path)
settings.runtime.backend == :cpu || error("Stage 1 covariance screening requires runtime.backend=cpu")
stage1 = load_bare_stage1_settings(config_path)
threading = configure_threading!(settings.runtime)
ground = read_backbone_ground_state(backbone_path)
length(ground.psi) == 2 * settings.model.L || error("backbone MPS length differs from the configuration")
LadderMPSMFT.sha256_file(config_path) == ground.config_sha256 || error(
    "Stage 1 requires the exact configuration used to build the backbone",
)
LadderMPSMFT.implementation_fingerprint() == ground.implementation_sha256 || error(
    "Stage 1 implementation differs from the code that built the backbone",
)

println("stage1_backbone=$backbone_path")
println("stage1_backbone_sha256=$(ground.backbone_sha256)")
println("stage1_ground_energy=$(ground.energy)")
println("stage1_chemical_potential=$(ground.chemical_potential)")
println("julia_threads=$(threading.julia)")
println("threaded_blocksparse=$(threading.blocksparse)")
println("stage1_step=compute_charge_spin_pair_covariances")
result = compute_bare_stage1(ground.psi, settings.model, stage1)
write_bare_stage1(
    output_path,
    result,
    settings.model;
    backbone_path,
    config_path,
    immutable=true,
)

summary_path = splitext(output_path)[1] * "_summary.tsv"
open(summary_path, "w") do io
    println(io, "sector\tparity_or_class\trank\teigenvalue\tmode_number\tq_over_pi\tfourier_overlap\tresidual_norm\tedge_weight")
    for (sector_name, sector) in (("charge", result.charge), ("spin", result.spin))
        for parity in (:even, :odd)
            spectrum = getproperty(sector, parity)
            for rank in eachindex(spectrum.eigenvalues)
                @printf(
                    io,
                    "%s\t%s\t%d\t%.16g\t%d\t%.16g\t%.16g\t%.16g\t%.16g\n",
                    sector_name,
                    String(parity),
                    rank,
                    spectrum.eigenvalues[rank],
                    spectrum.mode_numbers[rank],
                    spectrum.q_over_pi[rank],
                    spectrum.fourier_overlap[rank],
                    spectrum.residuals[rank],
                    spectrum.edge_weight[rank],
                )
            end
        end
    end
    for class in sort(collect(keys(result.pair)); by=string)
        spectrum = result.pair[class]
        for rank in eachindex(spectrum.eigenvalues)
            @printf(
                io,
                "pair\t%s\t%d\t%.16g\t%d\t%.16g\t%.16g\t%.16g\t%.16g\n",
                String(class),
                rank,
                spectrum.eigenvalues[rank],
                spectrum.mode_numbers[rank],
                spectrum.q_over_pi[rank],
                spectrum.fourier_overlap[rank],
                spectrum.residuals[rank],
                spectrum.edge_weight[rank],
            )
        end
    end
end

println("stage1_path=$output_path")
println("stage1_sha256=$(LadderMPSMFT.sha256_file(output_path))")
println("stage1_summary=$summary_path")
println("raw_map_norm=$(result.raw_map_norm)")
println("minimum_covariance_eigenvalue=$(result.minimum_covariance_eigenvalue)")
println("covariance_psd_pass=$(result.covariance_psd_pass)")
println("charge_decay_exponent=$(result.charge_decay.estimate)")
println("charge_decay_window_uncertainty=$(result.charge_decay.window_uncertainty)")
println("pair_decay_exponent=$(result.pair_decay.estimate)")
println("pair_decay_window_uncertainty=$(result.pair_decay.window_uncertainty)")
