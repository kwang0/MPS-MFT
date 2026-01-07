#!/bin/bash
#SBATCH -A m4863
#SBATCH -C cpu
#SBATCH -c 256
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -o ./logs_slurm/slurm-%j.out

# Arguments
# $1 = L (number of rungs)
# $2 = U (onsite interaction)
# $3 = t0 (rung hopping)
# $4 = t_p (inter-chain hopping)
# $5 = chi_max (bond dimension)
# $6 = E_p (pair binding energy)
# $7 = mu_init (initial chemical potential)
# $8 = density (target density)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=256

module load julia

mkdir -p logs_slurm

srun -u julia main_loop_script_ladder.jl $1 $2 $3 $4 $5 $6 $7 $8
