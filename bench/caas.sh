#!/usr/bin/env bash
# Move the repo to/from Monash CAAS (SLURM cluster) and watch the queue.
#
#   bench/caas.sh push      # rsync sources up (never binaries or results)
#   bench/caas.sh build     # module load + make on the headnode
#   bench/caas.sh submit X  # sbatch bench/slurm/X.sbatch
#   bench/caas.sh status    # my jobs in the queue
#   bench/caas.sh pull      # fetch bench/results-caas/ and job logs
#   bench/caas.sh analyse   # analyse the cluster results into bench/analysis-caas/
#
# Needs a "caas" entry in ~/.ssh/config with key auth.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST=caas
REMOTE='~/FIT3143-Lab-2'

case "${1:-}" in
push)
    rsync -az --delete \
        --exclude .git --exclude .DS_Store --exclude '*.pdf' \
        --exclude 'bench/results*' --exclude 'bench/analysis*' --exclude 'bench/slurm/logs' \
        --exclude 'lab-2/task1' --exclude 'lab-2/task2' \
        --exclude 'week-4-lab-1/task1' --exclude 'week-4-lab-1/task2' --exclude 'week-4-lab-1/task3' \
        --exclude '*_output.txt' --exclude '*.txt' \
        "$REPO_ROOT/" "$HOST:$REMOTE/"
    ;;
build)
    ssh "$HOST" "cd $REMOTE && module load openmpi && make -C week-4-lab-1 && make -C lab-2 && mpicc --version | head -1"
    ;;
submit)
    ssh "$HOST" "cd $REMOTE && mkdir -p bench/slurm/logs && sbatch bench/slurm/${2:?job name}.sbatch"
    ;;
status)
    ssh "$HOST" 'squeue -u $USER -o "%.8i %.14j %.2t %.10M %.4D %R"'
    ;;
pull)
    rsync -az "$HOST:$REMOTE/bench/results-caas/" "$REPO_ROOT/bench/results-caas/" 2>/dev/null || true
    rsync -az "$HOST:$REMOTE/bench/slurm/logs/" "$REPO_ROOT/bench/slurm/logs/"
    ;;
analyse)
    # Explicit paths, not the default glob: analyse.py would otherwise
    # concatenate bench/results/ (single-host Mac data) with the cluster runs
    # and silently average two different machines into one median.
    cd "$REPO_ROOT"
    python3 bench/analyse.py \
        --baselines 'bench/results-caas/baselines-*.csv' \
        --mpi       'bench/results-caas/mpi-*.csv' \
        --hybrid    'bench/results-caas/hybrid-*.csv' \
        --out bench/analysis-caas
    python3 bench/report.py --out bench/analysis-caas
    echo "-> bench/analysis-caas/report.html"
    ;;
*)
    sed -n '2,11p' "$0"; exit 1 ;;
esac
