#!/bin/bash
#SBATCH -A m4863
#SBATCH -C cpu
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -q shared
#SBATCH -t 48:00:00
#SBATCH -o ./logs_slurm/slurm-%j.out

# Shared QOS: charged only for the node fraction used (-c 64 = 1/4 node).
# Memory defaults to the same fraction (~128 GB); raise with --mem if DMRG
# needs more, noting the charge scales with max(cpu, memory) fraction.
# Shared allows at most half a node (-c 128); a full node needs -q regular.

# Arguments
# $1 = L (number of rungs)
# $2 = U (onsite interaction)
# $3 = V (nearest-neighbor interaction)
# $4 = t0 (rung hopping)
# $5 = density (target particle density)
# Optional additional arguments are passed through to calculate_E_p_ladder.jl,
# for example: --inherit-from previous.h5 --outfile current.h5 --force

# Perlmutter Slurm counts hardware threads as CPUs. Requesting -c 64 gives
# a 32-thread Julia process one physical core per Julia thread.
export TASK_CPUS="${SLURM_CPUS_PER_TASK:-64}"
export JULIA_THREADS="${JULIA_THREADS:-$((TASK_CPUS / 2))}"

if [[ "$((2 * JULIA_THREADS))" -gt "${TASK_CPUS}" ]]; then
  echo "JULIA_THREADS=${JULIA_THREADS} needs about $((2 * JULIA_THREADS)) Slurm CPUs on Perlmutter for one physical core per Julia thread."
  echo "Current SLURM_CPUS_PER_TASK=${TASK_CPUS}; reduce JULIA_THREADS or submit with a larger --cpus-per-task."
fi

# ITensor block-sparse contractions use Julia threads. Keep BLAS/OpenMP libraries
# single-threaded to avoid oversubscribing the Julia worker threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load julia
module load python
conda activate tenpy-env

mkdir -p logs_julia

echo "Running on ${SLURM_JOB_NODELIST:-unknown-node} with SLURM_NTASKS=${SLURM_NTASKS:-unset}, SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}, TASK_CPUS=${TASK_CPUS}, JULIA_THREADS=${JULIA_THREADS}"

# calculate_E_p_ladder.jl writes an HDF5 checkpoint with E_N_* and psi_N_* entries.
srun -u -n 1 --ntasks-per-node=1 --cpu-bind=cores -c "${TASK_CPUS}" julia -t "${JULIA_THREADS}" calculate_E_p_ladder.jl "$1" "$2" "$3" "$4" "$5" "${@:6}" \
  | tee -a logs_julia/E_p_ladder_L_${1}_U_${2}_V_${3}_t0_${4}_density_${5}_chi_1000.log

# No completion check needed here - script prints result and writes completed=true.
echo "E_p calculation completed"
    
#Monitor job with: tail -f logs_julia/E_p_ladder_L_${1}_U_${2}_V_${3}_t0_${4}_density_${5}_chi_1000.log


#sbatch submit_E_p_ladder.sh 16 8.0 0.0 1.0 0.9375
# For a full CPU node test (exceeds shared QOS half-node limit, so use regular):
# sbatch -q regular --cpus-per-task=256 --export=ALL,JULIA_THREADS=128 submit_E_p_ladder.sh 16 8.0 0.0 1.0 0.9375
