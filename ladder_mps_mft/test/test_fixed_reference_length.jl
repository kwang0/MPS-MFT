# Focused config tests; may run directly or be included by runtests.jl.
using Test
using LadderMPSMFT
using TOML

@testset "explicit fixed reference-length pair binding" begin
    project = normpath(joinpath(@__DIR__, ".."))
    base = joinpath(project, "configs", "phase1_gpu_square_v0_chi400_tight_compare.toml")
    original = load_settings(base)
    mktempdir() do temporary
        path = joinpath(temporary, "reference.toml")
        function load_raw(raw)
            open(path, "w") do io
                TOML.print(io, raw)
            end
            return load_settings(path)
        end
        raw = TOML.parsefile(base)
        raw["pair_binding"]["reference_L"] = 64
        same = load_raw(raw)
        @test same.model.ep_mode == :exact
        @test LadderMPSMFT.model_fingerprint(same.model) == LadderMPSMFT.model_fingerprint(original.model)
        fingerprints = String[]
        for L in (96, 128)
            raw["model"]["L"] = L
            result = load_raw(raw)
            @test result.model.L == L
            @test result.model.ep_reference_L == 64
            @test result.model.ep_mode == :fixed_reference_length
            @test result.model.ep_signed == original.model.ep_signed
            @test result.model.tp^2 / result.model.ep == original.model.tp^2 / original.model.ep
            push!(fingerprints, LadderMPSMFT.model_fingerprint(result.model))
        end
        @test length(unique(fingerprints)) == 2
        delete!(raw["pair_binding"], "reference_L")
        @test_throws ArgumentError load_raw(raw)  # No implicit L=64 fallback.
        raw["pair_binding"]["reference_L"] = 0
        @test_throws ArgumentError load_raw(raw)
        raw["pair_binding"]["reference_L"] = 64
        raw["pair_binding"]["allow_interpolation"] = true
        @test_throws ArgumentError load_raw(raw)
    end
end
