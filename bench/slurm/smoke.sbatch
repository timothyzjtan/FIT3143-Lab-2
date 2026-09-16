#!/bin/bash
#SBATCH --job-name=lab2-smoke
#SBATCH --partition=defq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
#
# Pipeline check before spending queue time on the full sweeps: two nodes,
# every Task 1 scheme and the hybrid, each compared byte-for-byte against the
# serial reference.
set -uo pipefail
source bench/slurm/env.sh

N=10000000
S="$(mktemp -d)"; trap 'rm -rf "$S"' EXIT
fail=0

echo "--- nodes in use"
srun -n "$SLURM_NTASKS" hostname | sort | uniq -c

echo "--- serial reference (n=$N)"
week-4-lab-1/task1 "$N" --csv --out "$S/ref.txt"

check() { if cmp -s "$S/ref.txt" "$S/out.txt"; then echo "  ok   $*"; else echo "  MISMATCH $*"; fail=1; fi; }

echo "--- task1, $SLURM_NTASKS ranks across $SLURM_JOB_NUM_NODES nodes"
for scheme in block cyclic blockcyclic dynamic; do
    $LAUNCH_MPI "$SLURM_NTASKS" lab-2/task1 "$N" --scheme "$scheme" --csv --out "$S/out.txt" || fail=1
    check "task1 $scheme"
done

echo "--- task2 hybrid, 2 ranks x 4 threads"
for scheme in block blockcyclic dynamic; do
    $LAUNCH_HYBRID 2 $LAUNCH_THREADS_FLAG 4 lab-2/task2 "$N" --threads 4 --scheme "$scheme" --csv --out "$S/out.txt" || fail=1
    check "task2 $scheme"
done

echo "finished $(date)"
[ "$fail" -eq 0 ] && echo "SMOKE PASS" || { echo "SMOKE FAIL"; exit 1; }
