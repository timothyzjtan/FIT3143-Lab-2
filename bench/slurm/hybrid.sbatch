#!/bin/bash
#SBATCH --job-name=lab2-hybrid
#SBATCH --partition=defq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:28:00
#SBATCH --array=1-3
#SBATCH --output=bench/slurm/logs/%x-%A_%a.out
#
# Task 2 across four nodes (64 cores). Sweep B holds the worker count at
# 16 / 32 / 64 and varies the rank:thread split, plus threads-per-rank at a
# fixed P=4 for figure 8. 64x1, 32x1 and 16x1 are the flat-MPI reference
# points. Array split as in baselines.sbatch.
set -euo pipefail
source bench/slurm/env.sh

export LAUNCH="$LAUNCH_HYBRID"
export FIXED_PROCS=4
export FIXED_THREADS=16
export GRID="1:16 2:8 4:4 8:2 16:1 2:16 4:8 8:4 16:2 32:1 4:16 8:8 16:4 32:2 64:1 4:1 4:2 4:8"
case "$SLURM_ARRAY_TASK_ID" in
    1) export SWEEPS=A LADDER_SLICE=1:15 ;;
    2) export SWEEPS=A LADDER_SLICE=16:30 ;;
    3) export SWEEPS="B C" ;;
esac
bench/sweep-hybrid.sh
echo "finished $(date)"
