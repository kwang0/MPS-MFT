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
# $3 = t_p
# $4 = chi_max
# $5 = E_p
# $6 = mu_init

module load python
conda activate tenpy-env
export PYTHONUNBUFFERED=1

outfile_name="results_U_${2}_t_p${3}.pkl"

srun -u python main_loop_script.py $1 $2 $3 $4 $5 $6 \
  | tee -a logs/dmrg_L_${1}_U_${2}_t_p_${3}_chi_${4}.log

completed=$(python - <<END
import pickle, sys
try:
    with open("$outfile_name", "rb") as f:
        data = pickle.load(f)
    sys.exit(0 if data.get("completed", False) else 1)
except:
    sys.exit(1)
END
)

if [ $? -ne 0 ]; then
    echo "Not completed, resubmitting..."
    sbatch --dependency=afterany:$SLURM_JOB_ID submit_loop.sh "$@"
else
    echo "Run completed successfully!"
fi

    
#Monitor job with: tail -f logs/dmrg_L_*_${1}_U_${2}_t_p_${3}_chi_${4}.log
