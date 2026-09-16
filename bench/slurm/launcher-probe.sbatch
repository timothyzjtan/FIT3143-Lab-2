#!/bin/bash
#SBATCH --job-name=lab2-probe
#SBATCH --partition=defq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:08:00
#SBATCH --output=bench/slurm/logs/%x-%j.out
# Which srun MPI plugin actually launches this Open MPI build across nodes?
set -uo pipefail
source bench/slurm/env.sh
S="$(mktemp -d)"; trap 'rm -rf "$S"' EXIT
week-4-lab-1/task1 1000000 --out "$S/ref.txt" >/dev/null
for mpi in pmi2 pmix_v3 pmix; do
    echo "--- srun --mpi=$mpi"
    t0=$(date +%s)
    if timeout 75 srun --mpi=$mpi -n 4 lab-2/task1 1000000 --scheme block --csv --out "$S/out.txt"; then
        cmp -s "$S/ref.txt" "$S/out.txt" && echo "  $mpi: OK ($(( $(date +%s)-t0 ))s)" || echo "  $mpi: ran but WRONG OUTPUT"
    else
        echo "  $mpi: FAILED/HUNG ($(( $(date +%s)-t0 ))s)"
    fi
done
echo "--- mpirun"
timeout 60 mpirun -np 4 lab-2/task1 1000000 --scheme block --csv --out "$S/out.txt" && echo "  mpirun: OK" || echo "  mpirun: FAILED"
echo "finished $(date)"
