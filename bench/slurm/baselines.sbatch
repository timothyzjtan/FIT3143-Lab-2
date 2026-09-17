#!/bin/bash
#SBATCH --job-name=lab2-baselines
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --array=1-5
#SBATCH --output=bench/slurm/logs/%x-%A_%a.out
#
# Week 4 baselines (serial, pthreads, OpenMP) on one 16-core node. Supplies the
# serial denominator analyse.py needs, measured on the same hardware as the MPI
# and hybrid sweeps.
#
# Split into five array tasks because defq caps a job at 20 minutes (MaxTime
# was 30 min until ~2026-09-17; jobs asking for more sit at PartitionTimeLimit
# forever rather than failing at submit, so check `scontrol show partition defq`
# before trusting any of these estimates). The A
# ladder costs ~38 min in total on CAAS and its per-n cost grows as ~n^1.5, so
# an even 15/15 split is not an even split of TIME: measured on job 133801,
# task 1 (points 1-15) finished in 9 min while task 2 (points 16-30) hit the
# 28-minute wall and lost n=1e8. The slices below are balanced by measured cost
# instead of by count -- 1:18, 19:25, 26:30 are ~12.5, ~12.1 and ~13.7 min --
# and LADDER_SLICE still keeps the exact 30-point values. Task 4 does B and C.
set -euo pipefail
source bench/slurm/env.sh

export N_FIXED=100000000
export FIXED_THREADS=16
export THREAD_LIST="1 2 4 6 8 10 12 14 16"
case "$SLURM_ARRAY_TASK_ID" in
    1) export SWEEPS=A LADDER_SLICE=1:18 ;;
    2) export SWEEPS=A LADDER_SLICE=19:25 ;;
    3) export SWEEPS=A LADDER_SLICE=26:30 ;;
    4) export SWEEPS=B   THREAD_LIST="1 2 4" ;;
    5) export SWEEPS="B C" THREAD_LIST="6 8 10 12 14 16" ;;
esac
bench/sweep.sh
echo "finished $(date)"
