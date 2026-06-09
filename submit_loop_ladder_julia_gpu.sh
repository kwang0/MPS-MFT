#!/bin/bash
#SBATCH -A m4863_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o ./logs_slurm/slurm-%j.out

# Arguments
# $1 = L
# $2 = U
# $3 = V
# $4 = t0
# $5 = t_p
# $6 = chi_max
# $7 = E_p
# $8 = mu_init
# $9 = density

export SLURM_CPU_BIND="cores"

module load julia
module load cudatoolkit
module load cray-mpich
module load libfabric
module load python
conda activate tenpy-env

outfile_name="results_L_${1}_U_${2}_V_${3}_t0_${4}_t_p_${5}_chi_${6}_density_${9}_gpu_nodamping.h5"

echo "outfile_name: $outfile_name"

mkdir -p logs_julia

srun -u julia main_loop_script_ladder_gpu.jl "$@" \
  | tee -a logs_julia/dmrg_ladder_L_${1}_U_${2}_V_${3}_t0_${4}_t_p_${5}_chi_${6}_density_${9}_gpu.log

python <<END
import h5py, sys
try:
    if (h5py.File("$outfile_name", 'r')['completed'][...]):
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    print("Check failed:", e)
    sys.exit(1)
END

if [ $? -ne 0 ]; then
    echo "Not completed, resubmitting..."
    sbatch --dependency=afterany:$SLURM_JOB_ID submit_loop_ladder_julia_gpu.sh "$@"
else
    echo "Run completed successfully!"
fi
    
#Monitor job with: tail -f logs_julia/dmrg_ladder_L_${1}_U_${2}_V_${3}_t0_${4}_t_p_${5}_chi_${6}_gpu.log
