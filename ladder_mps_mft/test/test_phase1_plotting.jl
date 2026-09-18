# Standalone plotting regression checks; use an environment with HDF5/PyPlot.
# No DMRG or solver imports. Optional ARGS[1]: a real one-ladder trellis state.
ENV["MPLBACKEND"] = "Agg"
using Test
include(joinpath(@__DIR__, "..", "plot_phase1_mf_observables.jl"))

function plotting_fixture(path; members=nothing, values=[1.0], parent="")
    h5open(path, "w") do f
        f["model/L"] = 8
        f["model/r_range"] = 2
        f["model/transverse_geometry"] = members === nothing ? "square" : "trellis"
        f["model/trellis_cell"] = members === nothing ? "" : length(members) == 1 ? "one_ladder" : "two_ladder"
        f["provenance/parent_checkpoint"] = parent
        f["status"] = "maximum_iterations"
        f["accepted"] = false
        # Deliberately omit initial_state_source, as the trellis driver does.
        roots = members === nothing ? [f] : [create_group(f, "ladders/$m") for m in members]
        for (root, value) in zip(roots, values)
            root["history/iteration"] = collect(1:3)
            root["history/fields/seed_iteration"] = 0
            for (key, shape) in (("alpha", (8,8,2,2)), ("beta", (2,8,8,2,2)), ("mu_cdw", (2,16)))
                root["fields/initial/$key"] = fill(value, shape)
                root["history/fields/seed/$key"] = fill(value, shape)
                root["fields/applied/$key"] = fill(value+2, shape)
                root["fields/measured/$key"] = fill(value+3, shape)
                root["fields/restart/$key"] = fill(value+3, shape)
                for (source, offset) in (("applied",0), ("measured",1))
                    root["history/fields/$source/$key"] = cat(
                        (fill(value+i+offset, shape) for i in 0:2)...; dims=length(shape)+1)
                end
            end
        end
    end
    return path
end

@testset "Phase 1 spatial-ladder plotting" begin
    mktempdir() do dir
        square = plotting_fixture(joinpath(dir,"square.h5"); values=[2.0])
        one = plotting_fixture(joinpath(dir,"one.h5"); members=["A"])
        two = plotting_fixture(joinpath(dir,"two.h5"); members=["A","B"], values=[1.0,7.0])
        for (path, ladder, value) in ((square,nothing,2.0), (one,nothing,1.0),
                                     (two,:A,1.0), (two,:B,7.0), (two,"b",7.0))
            @test all(phase1_seed_fields(path; ladder).alpha .== value)
            @test all(_p1_snapshot_fields(path,:restart; ladder).alpha .== value+3)
            history = _p1_complete_history(path,:measured; ladder)
            @test history.iterations == [0,1,2,3]
            @test vec(history.alpha[1,1,1,1,:]) == value .+ (0:3)
            @test _p1_complete_history(path,:measured; ladder, include_seed=false).iterations == [1,2,3]
            @test vec(_p1_complete_history(path,:applied; ladder, include_seed=false).alpha[1,1,1,1,:]) == value .+ (0:2)
            snapshots = phase1_saved_mf_snapshots(path; ladder)
            @test getfield.(snapshots,:iteration) == [0,3]
            @test all(last(snapshots).fields.alpha .== value+3)
            fig = plot_phase1_mf_profiles_and_middle_histories(path; ladder)
            @test collect(fig.axes[2].lines[1].get_xdata()) == [1,2,3,4]
            @test collect(fig.axes[2].lines[1].get_ydata()) == 2 .* (value .+ (0:3))
            slider = _FIGURE_CALLBACK_REFS[fig].slider
            @test slider.valmax == 4
            slider.set_val(1)
            @test all(collect(fig.axes[1].lines[1].get_ydata()) .== 2value)
            slider.set_val(4)
            @test all(collect(fig.axes[1].lines[1].get_ydata()) .== 2(value+3))
            if path != square
                @test occursin("ladder=$(uppercase(String(something(ladder,:A))))", String(fig._suptitle.get_text()))
            end
            PyPlot.close(fig)
            seedfig = plot_phase1_seed_profiles(path; ladder)
            @test all(collect(seedfig.axes[1].lines[1].get_ydata()) .== 2value)
            PyPlot.close(seedfig)
        end
        @test_throws ArgumentError phase1_seed_fields(one; ladder=:B)
        @test_throws ArgumentError phase1_seed_fields(square; ladder=:B)
        @test_throws ArgumentError _p1_complete_history(two,:measured; ladder=:C)
        child = plotting_fixture(joinpath(dir,"child.h5"); members=["A","B"], values=[4.0,10.0], parent=two)
        for (ladder,value) in ((:A,1.0),(:B,7.0))
            h = _p1_complete_history(child,:measured; ladder)
            stitched = _p1_stitch_parent_measured_history(child,h; include_seed=true,ladder)
            @test stitched.parent_updates == 3
            @test stitched.continuation_updates == 3
            @test vec(stitched.history.alpha[1,1,1,1,:]) == value .+ (0:6)
        end
    end
end

if !isempty(ARGS)
    @testset "Synced trellis full-history render" begin
        state = abspath(ARGS[1])
        history = _p1_complete_history(state,:measured)
        seed = phase1_seed_fields(state)
        h5open(state,"r") do f
            @test seed.alpha == read(f,"ladders/A/fields/initial/alpha")
            @test history.alpha[:,:,:,:,2:end] == read(f,"ladders/A/history/fields/measured/alpha")
        end
        out = joinpath(@__DIR__,"..","output","plot_validation","trellis_mf_and_middle_histories.png")
        fig = plot_phase1_mf_profiles_and_middle_histories(state; savepath=out,dpi=90)
        @test length(fig.axes[2].lines[1].get_xdata()) == length(history.iterations)
        @test _FIGURE_CALLBACK_REFS[fig].slider.valmax == length(history.iterations)
        @test isfile(out)
        PyPlot.close(fig)
        println("rendered=$out")
    end
end
