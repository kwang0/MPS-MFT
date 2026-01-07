#!/bin/bash
#SBATCH -A m4863
#SBATCH -C cpu
#SBATCH -c 256
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -o ./logs_slurm/slurm-%j.out

# Arguments
# $1 = L
# $2 = U
# $3 = t0
# $4 = t_p
# $5 = chi_max
# $6 = E_p
# $7 = mu_init
# $8 = density

export OMP_NUM_THREADS=256
export MKL_NUM_THREADS=256

module load python
conda activate tenpy-env

outfile_name="results_L_${1}_U_${2}_t0_${3}_t_p_${4}_chi_${5}.h5"

echo "outfile_name: $outfile_name"

mkdir -p logs_julia

srun -u julia main_loop_script_ladder.jl $1 $2 $3 $4 $5 $6 $7 $8 \
  | tee -a logs_julia/dmrg_ladder_L_${1}_U_${2}_t0_${3}_t_p_${4}_chi_${5}.log

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
    sbatch --dependency=afterany:$SLURM_JOB_ID submit_loop_ladder_julia.sh "$@"
else
    echo "Run completed successfully!"
fi
    
#Monitor job with: tail -f logs_julia/dmrg_ladder_L_${1}_U_${2}_t0_${3}_t_p_${4}_chi_${5}.log
