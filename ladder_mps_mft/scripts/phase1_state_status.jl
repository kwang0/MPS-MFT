#!/usr/bin/env julia

using HDF5

for raw_path in ARGS
    path = abspath(raw_path)
    h5open(path, "r") do file
        status = String(read(file, "status"))
        accepted = Bool(read(file, "accepted"))
        period = Int(read(file, "fundamental_period"))
        energy = Float64(read(file, "solution_canonical_variational_energy"))
        tensor_path = "psi/MPS[1]/storage/data"
        tensor_scalar_type = haskey(file, tensor_path) ? string(eltype(read(file, tensor_path))) : "unknown"
        println(join((path, status, string(accepted), string(period), string(energy), tensor_scalar_type), '\t'))
    end
end
