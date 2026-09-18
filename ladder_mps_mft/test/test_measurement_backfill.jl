using Test, HDF5, ITensors, ITensorMPS, LadderMPSMFT, CSV, TOML
include("../scripts/measure_latest_campaigns.jl")

@testset "retroactive manifest worker and receipts" begin
    mktempdir() do directory
        config=joinpath(directory,"config.toml"); write(config,"# immutable fixture\n")
        state=joinpath(directory,"state.h5")
        model=ModelSettings(L=2,r_range=1,ep=.1,ep_signed=-.1)
        sites=siteinds("Electron",4;conserve_qns=false)
        psi=MPS(sites,["Up","Dn","Up","Dn"])
        h5open(state,"w") do f
            f["artifact_kind"]="ladder_mps_mft_state"
            f["accepted"]=false; f["status"]="maximum_iterations"
            f["solution_kind"]="none"; f["fundamental_period"]=0
            f["psi"]=psi; f["history/iteration"]=[60]
            LMF._write_dict(create_group(f,"model"),Dict(String(k)=>getfield(model,k) for k in fieldnames(ModelSettings)))
            f["provenance/model_fingerprint"]=LMF.model_fingerprint(model)
        end
        row=(index=1,campaign="synthetic",label="stripe",config_path=config,config_sha256=LMF.sha256_file(config),
             source_path=state,source_sha256=LMF.sha256_file(state),model_fingerprint=LMF.model_fingerprint(model),samples=1)
        manifest=joinpath(directory,"manifest.tsv")
        CSV.write(manifest,[row];delim='\t')
        run_measurement(manifest,1)
        receipt=joinpath(directory,"results","synthetic","stripe","measurement_receipt.toml")
        @test isfile(receipt)
        saved=TOML.parsefile(receipt)
        @test saved["files"]==["diagnostics.h5"]
        output=joinpath(dirname(receipt),only(saved["files"]))
        @test LMF.sha256_file(output)==only(saved["sha256"])
        @test h5read(output,"measurement_implementation_sha256")==implementation_fingerprint()
        @test !h5read(output,"accepted")
        previous=LMF.sha256_file(output)
        run_measurement(manifest,1)
        @test LMF.sha256_file(output)==previous
        @test LMF.sha256_file(state)==row.source_sha256
        @test_throws ErrorException run_measurement(manifest,2)
        write(config,"# changed\n")
        @test_throws ErrorException run_measurement(manifest,1)
        measurement_status(directory)
    end
end
