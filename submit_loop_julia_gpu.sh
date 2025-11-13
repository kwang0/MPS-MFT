#!/bin/bash
#SBATCH -A m4863_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o ./logs_slurm/slurm-%j.out

# Arguments
# $1 = L
# $2 = U
# $3 = t_p
# $4 = chi_max
# $5 = E_p
# $6 = mu_init

export SLURM_CPU_BIND="cores"

module load python
conda activate tenpy-env

outfile_name="results_U_${2}_t_p_${3}_gpu.h5"

echo "outfile_name: $outfile_name"

mkdir -p logs_julia

srun -u julia main_loop_script_gpu.jl $1 $2 $3 $4 $5 $6 \
  | tee -a logs_julia/dmrg_L_${1}_U_${2}_t_p_${3}_chi_${4}_gpu.log

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
    sbatch --dependency=afterany:$SLURM_JOB_ID submit_loop_julia_gpu.sh "$@"
else
    echo "Run completed successfully!"
fi
    
#Monitor job with: tail -f logs_julia/dmrg_L_*_${1}_U_${2}_t_p_${3}_chi_${4}_gpu.log
