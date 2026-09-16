#!/bin/bash
#SBATCH --job-name=lab2-8node
#SBATCH --partition=defq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:20:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
#
# Optional: 128-rank Sweep B points only. Submitted last; dropped if it does
# not schedule in time.
set -euo pipefail
source bench/slurm/env.sh
export SWEEPS=B
LAUNCH="$LAUNCH_MPI" PROC_LIST="64 96 128" bench/sweep-mpi.sh
LAUNCH="$LAUNCH_HYBRID" GRID="8:16 16:8 32:4 128:1" bench/sweep-hybrid.sh
echo "finished $(date)"
