#!/usr/bin/env julia

using HDF5

for raw_path in ARGS
    path = abspath(raw_path)
    h5open(path, "r") do file
        status = String(read(file, "status"))
        accepted = Bool(read(file, "accepted"))
        period = Int(read(file, "fundamental_period"))
        energy = Float64(read(file, "solution_canonical_variational_energy"))
        println(join((path, status, string(accepted), string(period), string(energy)), '\t'))
    end
end
