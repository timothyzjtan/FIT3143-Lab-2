#!/bin/bash
#SBATCH --job-name=lab2-mpi
#SBATCH --partition=defq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:28:00
#SBATCH --array=1-3
#SBATCH --output=bench/slurm/logs/%x-%A_%a.out
#
# Task 1 across four nodes (64 ranks). srun fills nodes in block order, so
# P <= 16 stays on one node (comparable to the single-host runs) and P > 16 is
# genuinely distributed-memory. Array split as in baselines.sbatch.
set -euo pipefail
source bench/slurm/env.sh

export LAUNCH="$LAUNCH_MPI"
export FIXED_PROCS=16
export PROC_LIST="1 2 4 8 12 16 24 32 48 64"
case "$SLURM_ARRAY_TASK_ID" in
    1) export SWEEPS=A LADDER_SLICE=1:15 ;;
    2) export SWEEPS=A LADDER_SLICE=16:30 ;;
    3) export SWEEPS="B C" ;;
esac
bench/sweep-mpi.sh
echo "finished $(date)"
