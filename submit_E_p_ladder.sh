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
# $4 = density (target particle density)

export OMP_NUM_THREADS=256
export MKL_NUM_THREADS=256

module load julia
module load python
conda activate tenpy-env

mkdir -p logs_julia

# Note: calculate_E_p_ladder.jl doesn't create an h5 file, just computes E_p
srun -u julia -t 32 calculate_E_p_ladder.jl $1 $2 $3 $4 $5 \
  | tee -a logs_julia/E_p_ladder_L_${1}_U_${2}_t0_${3}_density_${4}_offset_${5}.log

# No completion check needed - script prints result and exits
echo "E_p calculation completed"
    
#Monitor job with: tail -f logs_julia/E_p_ladder_L_${1}_U_${2}_t0_${3}_density_${4}.log


#sbatch submit_E_p_ladder.sh 16 8.0 1.0 0.9375