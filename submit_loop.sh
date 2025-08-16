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

python main_loop_script.py $1 $2 $3 $4 $5 $6 > logs/dmrg_L_${1}_U_${2}_t_p_${3}_chi_${4}.log

exit 0