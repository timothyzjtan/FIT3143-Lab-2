#!/usr/bin/env bash
#
# FIT3143 Lab #2 - benchmark sweep for the hybrid MPI+OpenMP implementation
# (Task 2).
#
#     docker/run.sh bench/sweep-hybrid.sh
#     docker/run.sh bench/sweep-hybrid.sh --quick
#
# Separate CSV from the Task 1 sweep because the hybrid reports an extra phase
# (t_lsort, the concurrent per-rank sort) and a second parallelism dimension
# (threads per rank). The files are joined on n and on total_workers = P x T at
# analysis time.
#
# Sweeps:
#   A  vary n, P and T fixed          -> runtime/speedup vs problem size
#   B  vary total workers along the   -> speedup vs degree of parallelism,
#      P x T grid at fixed n             and where the P/T split matters
#   C  vary scheme, n/P/T fixed       -> partitioning comparison
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO_ROOT/lab-2/task2"
SERIAL="$REPO_ROOT/week-4-lab-1/task1"
OUT_DIR="$REPO_ROOT/bench/results"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

REPS=3
CHUNK=4096
MAIN_SCHEME=${MAIN_SCHEME:-dynamic}
# SWEEPS selects which of A/B/C to run, so one sweep can be re-measured
# without spending the time to redo the other two.
SWEEPS="${SWEEPS:-A B C}"

# MANDATORY for the hybrid, and the single most important flag in this file.
# Open MPI binds a rank to one core (np<=2) or one socket (np>2) by default.
# That is right for Task 1, where a rank is one thread, but for the hybrid it
# pins every OpenMP thread of a rank onto that rank's core: measured at
# n=2e7, P=1, T=8 the compute phase was 1.147 s bound vs 0.258 s unbound, i.e.
# the threads contributed NOTHING and the sweep silently measured MPI-only
# scaling. --map-by slot:PE=T would be the tighter choice but cannot express
# the oversubscribed points of the P x T grid (P*T up to 16 on 10 cores).
BIND="--bind-to none"
SCHEMES="block blockcyclic dynamic"

# Task 1's sweeps fix P=8; the hybrid fixes P x T = 8 as well so the two are
# compared at the same degree of parallelism rather than the same rank count.
FIXED_PROCS=4
FIXED_THREADS=2

N_MIN=20000000
N_MAX=100000000
N_STEPS=30
N_FIXED=30000000

# (procs, threads) pairs for Sweep B. Covers the same 1..16 worker ladder as
# the Task 1 P-sweep, and at several worker counts includes more than one
# split so the cost of an MPI rank versus an OpenMP thread is measurable.
GRID="1:1 2:1 1:2 3:1 4:1 2:2 1:4 5:1 6:1 3:2 2:3 8:1 4:2 2:4 1:8 10:1 5:2 12:1 6:2 4:3 3:4 16:1 8:2 4:4"

if [ "$QUICK" -eq 1 ]; then
    REPS=1; N_MIN=1000000; N_MAX=4000000; N_STEPS=3
    N_FIXED=2000000; GRID="1:1 2:2 4:2"
fi

[ -x "$BIN" ]    || { echo "error: $BIN not built. Run: make -C lab-2" >&2; exit 1; }
[ -x "$SERIAL" ] || { echo "error: $SERIAL not built. Run: make -C week-4-lab-1" >&2; exit 1; }

mkdir -p "$OUT_DIR"
RESULTS="$OUT_DIR/hybrid-$(date +%Y%m%d-%H%M%S).csv"
echo "impl,n,procs,threads,prime_count,t_alloc,t_bcast,t_compute,t_lsort,t_comm,t_sort,t_io,t_total,imbalance,workers,rep,sweep" > "$RESULTS"

# run <sweep> <rep> <procs> <threads> <n> <scheme>
run() {
    local sweep="$1" rep="$2" procs="$3" threads="$4" n="$5" scheme="$6"
    local row
    row="$(mpirun --allow-run-as-root --oversubscribe $BIND -np "$procs" "$BIN" "$n" \
           --threads "$threads" --scheme "$scheme" --chunk "$CHUNK" \
           --csv --out "$SCRATCH/out.txt")"
    echo "${row},${rep},${sweep}" >> "$RESULTS"
}

# Every configuration is checked against the serial reference: a decomposition
# bug that only shows at some size or some P/T split would otherwise hide in
# the timing data.
check() {
    cmp -s "$SCRATCH/ref.txt" "$SCRATCH/out.txt" || {
        echo "ERROR: hybrid output disagrees with serial ($*)" >&2; exit 1; }
}

LADDER="$(awk -v lo="$N_MIN" -v hi="$N_MAX" -v steps="$N_STEPS" 'BEGIN {
    for (i = 0; i < steps; i++)
        printf "%d\n", int(lo * exp((log(hi)-log(lo)) * i / (steps-1)) / 1000) * 1000
}' | sort -n -u)"
LADDER_COUNT="$(echo "$LADDER" | wc -l | tr -d ' ')"

echo "=============================================="
echo " FIT3143 Lab 2 - hybrid MPI+OpenMP sweep"
echo " cores    : $(nproc)   reps: $REPS   scheme: $MAIN_SCHEME"
echo " fixed    : P=$FIXED_PROCS T=$FIXED_THREADS ($((FIXED_PROCS*FIXED_THREADS)) workers)"
echo " n values : $LADDER_COUNT ($N_MIN .. $N_MAX)"
echo " results  : $RESULTS"
echo "=============================================="

echo
case " $SWEEPS " in *" A "*)
echo "--- Sweep A: increasing n (P=$FIXED_PROCS T=$FIXED_THREADS, scheme=$MAIN_SCHEME) ---"
for n in $LADDER; do
    echo "  n = $n"
    for rep in $(seq 1 "$REPS"); do
        run A "$rep" "$FIXED_PROCS" "$FIXED_THREADS" "$n" "$MAIN_SCHEME"
    done
    "$SERIAL" "$n" --out "$SCRATCH/ref.txt" >/dev/null
    check "n=$n"
done

echo
;; esac

case " $SWEEPS " in *" B "*)
echo "--- Sweep B: P x T grid (n=$N_FIXED, scheme=$MAIN_SCHEME) ---"
"$SERIAL" "$N_FIXED" --out "$SCRATCH/ref.txt" >/dev/null
for pt in $GRID; do
    p="${pt%%:*}"; t="${pt##*:}"
    echo "  P = $p  T = $t  (workers = $((p*t)))"
    for rep in $(seq 1 "$REPS"); do
        run B "$rep" "$p" "$t" "$N_FIXED" "$MAIN_SCHEME"
    done
    check "P=$p T=$t"
done

echo
;; esac

case " $SWEEPS " in *" C "*)
echo "--- Sweep C: scheme comparison (n=$N_FIXED, P=$FIXED_PROCS T=$FIXED_THREADS) ---"
for s in $SCHEMES; do
    echo "  scheme = $s"
    for rep in $(seq 1 "$REPS"); do
        run C "$rep" "$FIXED_PROCS" "$FIXED_THREADS" "$N_FIXED" "$s"
    done
    check "scheme=$s"
done

echo
;; esac

echo "Done. $(( $(wc -l < "$RESULTS") - 1 )) rows -> $RESULTS"
