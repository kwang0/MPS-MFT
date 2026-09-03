#!/usr/bin/env julia

using HDF5

for raw_path in ARGS
    path = abspath(raw_path)
    h5open(path, "r") do file
        status = String(read(file, "status"))
        accepted = Bool(read(file, "accepted"))
        period = Int(read(file, "fundamental_period"))
        canonical_energy = Float64(read(file, "solution_canonical_variational_energy"))
        target_corrected_energy = haskey(
            file,
            "solution_target_density_corrected_variational_energy",
        ) ? Float64(read(file, "solution_target_density_corrected_variational_energy")) : NaN
        tensor_path = "psi/MPS[1]/storage/data"
        tensor_scalar_type = if haskey(file, tensor_path)
            string(eltype(read(file, tensor_path)))
        elseif haskey(file, "model/tensor_scalar_type")
            String(read(file, "model/tensor_scalar_type"))
        elseif haskey(file, "provenance/tensor_scalar_type")
            String(read(file, "provenance/tensor_scalar_type"))
        else
            "unknown"
        end
        println(join((
            path,
            status,
            string(accepted),
            string(period),
            string(canonical_energy),
            string(target_corrected_energy),
            tensor_scalar_type,
        ), '\t'))
    end
end
