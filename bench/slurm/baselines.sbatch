#!/bin/bash
#SBATCH --job-name=lab2-baselines
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:28:00
#SBATCH --array=1-3
#SBATCH --output=bench/slurm/logs/%x-%A_%a.out
#
# Week 4 baselines (serial, pthreads, OpenMP) on one 16-core node. Supplies the
# serial denominator analyse.py needs, measured on the same hardware as the MPI
# and hybrid sweeps.
#
# Split into three array tasks because CAAS caps a job at 30 minutes and the
# serial runs alone are ~45 min: tasks 1 and 2 take half the n ladder each
# (LADDER_SLICE keeps the exact 30-point values), task 3 does sweeps B and C.
set -euo pipefail
source bench/slurm/env.sh

export FIXED_THREADS=16
export THREAD_LIST="1 2 4 6 8 10 12 14 16"
case "$SLURM_ARRAY_TASK_ID" in
    1) export SWEEPS=A LADDER_SLICE=1:15 ;;
    2) export SWEEPS=A LADDER_SLICE=16:30 ;;
    3) export SWEEPS="B C" ;;
esac
bench/sweep.sh
echo "finished $(date)"
