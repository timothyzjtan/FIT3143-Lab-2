#!/usr/bin/env bash
#
# FIT3143 Lab #2 - benchmark sweep for the Open MPI implementation (Task 1).
#
#     docker/run.sh bench/sweep-mpi.sh
#     docker/run.sh bench/sweep-mpi.sh --quick
#
# Kept separate from bench/sweep.sh because the MPI programme reports extra
# phases (broadcast, gather, sort) that the Week 4 baselines have no analogue
# for. Mixing two column layouts in one CSV would make the analysis fragile;
# the two files are joined on (n, threads/procs) at analysis time instead.
#
# Sweeps mirror the baseline script so the graphs line up:
#   Sweep A  vary n, P fixed        -> runtime/speedup vs problem size
#   Sweep B  vary P, n fixed        -> speedup vs process count
#   Sweep C  vary scheme, n+P fixed -> partitioning comparison
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO_ROOT/lab-2/task1"
OUT_DIR="$REPO_ROOT/bench/results"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

REPS=3
FIXED_PROCS=8
CHUNK=4096
# dynamic is the default scheme for the n- and P-sweeps: it measured best at
# n=1e7 and is the only scheme that adapts to cores of differing speed, which
# this host has (4 performance + 6 efficiency cores).
MAIN_SCHEME=dynamic
SCHEMES="block cyclic blockcyclic dynamic"

# Same ladder as the baseline sweep so the two are directly comparable.
N_MIN=20000000
N_MAX=100000000
N_STEPS=30
N_FIXED=30000000

if [ "$QUICK" -eq 1 ]; then
    REPS=1; N_MIN=1000000; N_MAX=4000000; N_STEPS=3
    N_FIXED=2000000; PROC_LIST="1 2 4"
else
    PROC_LIST="1 2 3 4 5 6 7 8 9 10 12 14 16"
fi

[ -x "$BIN" ] || { echo "error: $BIN not built. Run: make -C lab-2" >&2; exit 1; }

mkdir -p "$OUT_DIR"
RESULTS="$OUT_DIR/mpi-$(date +%Y%m%d-%H%M%S).csv"
echo "impl,n,procs,threads,prime_count,t_alloc,t_bcast,t_compute,t_comm,t_sort,t_io,t_total,imbalance,rep,sweep" > "$RESULTS"

# run <sweep> <rep> <procs> <n> <scheme> [extra args...]
run() {
    local sweep="$1" rep="$2" procs="$3" n="$4" scheme="$5"; shift 5
    local row
    row="$(mpirun -np "$procs" "$BIN" "$n" --scheme "$scheme" --chunk "$CHUNK" \
           --csv --out "$SCRATCH/out.txt" "$@")"
    echo "${row},${rep},${sweep}" >> "$RESULTS"
}

LADDER="$(awk -v lo="$N_MIN" -v hi="$N_MAX" -v steps="$N_STEPS" 'BEGIN {
    for (i = 0; i < steps; i++)
        printf "%d\n", int(lo * exp((log(hi)-log(lo)) * i / (steps-1)) / 1000) * 1000
}' | sort -n -u)"
LADDER_COUNT="$(echo "$LADDER" | wc -l | tr -d ' ')"

echo "=============================================="
echo " FIT3143 Lab 2 - Open MPI sweep"
echo " cores    : $(nproc)   reps: $REPS   scheme: $MAIN_SCHEME"
echo " n values : $LADDER_COUNT ($N_MIN .. $N_MAX)"
echo " results  : $RESULTS"
echo "=============================================="

# A serial reference is produced once per n so correctness can be checked
# against it - a partitioning bug that only appears at some size would
# otherwise be invisible in timing data.
SERIAL="$REPO_ROOT/week-4-lab-1/task1"

echo
echo "--- Sweep A: increasing n (P=$FIXED_PROCS, scheme=$MAIN_SCHEME) ---"
for n in $LADDER; do
    echo "  n = $n"
    for rep in $(seq 1 "$REPS"); do
        run A "$rep" "$FIXED_PROCS" "$n" "$MAIN_SCHEME"
    done
    "$SERIAL" "$n" --out "$SCRATCH/ref.txt" >/dev/null
    cmp -s "$SCRATCH/ref.txt" "$SCRATCH/out.txt" || {
        echo "ERROR: MPI output disagrees with serial at n=$n" >&2; exit 1; }
done

echo
echo "--- Sweep B: increasing processes (n=$N_FIXED, scheme=$MAIN_SCHEME) ---"
"$SERIAL" "$N_FIXED" --out "$SCRATCH/ref.txt" >/dev/null
for p in $PROC_LIST; do
    echo "  P = $p"
    for rep in $(seq 1 "$REPS"); do
        run B "$rep" "$p" "$N_FIXED" "$MAIN_SCHEME"
    done
    cmp -s "$SCRATCH/ref.txt" "$SCRATCH/out.txt" || {
        echo "ERROR: MPI output disagrees with serial at P=$p" >&2; exit 1; }
done

echo
echo "--- Sweep C: partitioning scheme comparison (n=$N_FIXED, P=$FIXED_PROCS) ---"
for s in $SCHEMES; do
    echo "  scheme = $s"
    for rep in $(seq 1 "$REPS"); do
        run C "$rep" "$FIXED_PROCS" "$N_FIXED" "$s"
    done
    cmp -s "$SCRATCH/ref.txt" "$SCRATCH/out.txt" || {
        echo "ERROR: MPI output disagrees with serial for scheme=$s" >&2; exit 1; }
done

echo
echo "Done. $(( $(wc -l < "$RESULTS") - 1 )) rows -> $RESULTS"
