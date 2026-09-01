#!/usr/bin/env julia

using LadderMPSMFT
using Printf

length(ARGS) >= 2 || error("usage: julia --project=. scripts/compare_branches.jl STATE1.h5 STATE2.h5 [STATE3.h5 ...]")
rows = compare_variational_branches(ARGS)
reference = first(rows).energy
println("rank,branch,geometry,solution_kind,period,target_density_corrected_variational_energy,delta_energy,path")
for (rank, row) in enumerate(rows)
    @printf("%d,%s,%s,%s,%d,%.16g,%.16g,%s\n", rank, row.branch, row.geometry,
        row.solution_kind, row.period, row.energy, row.energy - reference, row.path)
end
