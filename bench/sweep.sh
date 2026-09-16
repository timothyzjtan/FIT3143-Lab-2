#!/usr/bin/env bash
#
# FIT3143 Lab #2 - benchmark sweep for the Week 4 baselines.
#
# Run inside the container so every number comes from one toolchain:
#     docker/run.sh bench/sweep.sh
#     docker/run.sh bench/sweep.sh --quick     # smoke test, ~1 min
#
# Design - three separate sweeps rather than one full cross-product. The full
# grid (30 values of n x 16 thread counts x 3 reps x 3 implementations) is
# ~4300 runs and would take hours, while the specification only ever asks for
# one variable to move at a time:
#
#   Sweep A  vary n, threads fixed          -> "increasing problem size n" graphs
#   Sweep B  vary thread count, n fixed     -> "increasing number of threads" graphs
#   Sweep C  vary OpenMP schedule, n+T fixed -> partitioning-strategy comparison
#
# Raw per-repetition rows are written, not medians. Medians are computed at
# analysis time so the underlying spread stays inspectable - if a configuration
# is noisy we want to see that, not have it averaged away.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$REPO_ROOT/week-4-lab-1"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/bench/results}"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

REPS=3
FIXED_THREADS="${FIXED_THREADS:-8}"
# SWEEPS selects which of A/B/C to run, mirroring sweep-mpi.sh/sweep-hybrid.sh,
# so one sweep can be re-measured (or split across jobs) without redoing all.
SWEEPS="${SWEEPS:-A B C}"
# (schedule list for Sweep C is defined inline below, as kind+chunk pairs)

# The specification requires at least 30 distinct values of n, and warns
# against runtimes under 1 second (measurement noise dominates).
#
# Calibrated, not guessed: serial t_compute was measured at 2.02 s for
# n = 3e7 on this machine. Trial division costs ~n^1.5/ln n overall, so the
# 1-second crossing is near n = 1.9e7; the ladder starts just above it at 2e7.
# Re-measure and raise N_MIN on faster hardware.
N_MIN=20000000
N_MAX=100000000
N_STEPS=30

# Fixed n for the thread and schedule sweeps: large enough to be well clear of
# the noise floor, small enough that 16 thread counts x 3 reps stays quick.
N_FIXED=30000000

if [ "$QUICK" -eq 1 ]; then
    REPS=1
    N_MIN=1000000
    N_MAX=4000000
    N_STEPS=3
    N_FIXED=2000000
    THREAD_LIST="1 2 4"
else
    THREAD_LIST="${THREAD_LIST:-1 2 3 4 5 6 7 8 9 10 12 14 16}"
fi

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS="$OUT_DIR/baselines-$STAMP${RUN_TAG:+-$RUN_TAG}.csv"

for b in task1 task2 task3; do
    if [ ! -x "$BIN_DIR/$b" ]; then
        echo "error: $BIN_DIR/$b not built. Run: make -C week-4-lab-1" >&2
        exit 1
    fi
done

echo "impl,n,procs,threads,prime_count,t_alloc,t_compute,t_merge,t_io,t_total,rep,sweep" > "$RESULTS"

# run <sweep-label> <rep> <command...>
run() {
    local sweep="$1"; shift
    local rep="$1"; shift
    local row
    row="$("$@")"
    echo "${row},${rep},${sweep}" >> "$RESULTS"
}

# Log-spaced ladder of N_STEPS values from N_MIN to N_MAX, de-duplicated.
build_ladder() {
    awk -v lo="$N_MIN" -v hi="$N_MAX" -v steps="$N_STEPS" 'BEGIN {
        for (i = 0; i < steps; i++) {
            v = lo * exp((log(hi) - log(lo)) * i / (steps - 1))
            printf "%d\n", int(v / 1000) * 1000   # round to a tidy value
        }
    }' | sort -n -u
}

LADDER="$(build_ladder)"
# LADDER_SLICE="from:to" keeps only that 1-indexed range of the ladder, so one
# long sweep can be split across several jobs without changing any n value
# (CAAS enforces a 30-minute wall limit). Empty = the whole ladder.
if [ -n "${LADDER_SLICE:-}" ]; then
    LADDER="$(echo "$LADDER" | sed -n "${LADDER_SLICE%%:*},${LADDER_SLICE##*:}p")"
fi
LADDER_COUNT="$(echo "$LADDER" | wc -l | tr -d ' ')"

echo "=============================================="
echo " FIT3143 Lab 2 - baseline sweep"
echo " host cores : $(nproc)"
echo " reps       : $REPS"
echo " n values   : $LADDER_COUNT (min $N_MIN, max $N_MAX)"
echo " results    : $RESULTS"
echo "=============================================="

if [ "$LADDER_COUNT" -lt 30 ] && [ "$QUICK" -eq 0 ]; then
    echo "WARNING: only $LADDER_COUNT distinct values of n - the specification"
    echo "         requires at least 30. Increase N_STEPS or widen the range." >&2
fi

# ---------------------------------------------------------------- Sweep A ---
echo
case " $SWEEPS " in *" A "*)
echo "--- Sweep A: increasing n (threads fixed at $FIXED_THREADS) ---"
for n in $LADDER; do
    echo "  n = $n"
    for rep in $(seq 1 "$REPS"); do
        run A "$rep" "$BIN_DIR/task1" "$n" --csv --out "$SCRATCH/a1.txt"
        run A "$rep" "$BIN_DIR/task2" "$n" "$FIXED_THREADS" --csv --out "$SCRATCH/a2.txt"
        run A "$rep" "$BIN_DIR/task3" "$n" "$FIXED_THREADS" --csv --out "$SCRATCH/a3.txt"
    done
    # Correctness is re-checked at every n: a partitioning bug that only shows
    # up at a particular size would otherwise be invisible in timing data.
    if ! cmp -s "$SCRATCH/a1.txt" "$SCRATCH/a2.txt" || \
       ! cmp -s "$SCRATCH/a1.txt" "$SCRATCH/a3.txt"; then
        echo "ERROR: outputs disagree at n=$n - results are not trustworthy" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------- Sweep B ---
;; esac

case " $SWEEPS " in *" B "*)
echo
echo "--- Sweep B: increasing threads (n fixed at $N_FIXED) ---"
for rep in $(seq 1 "$REPS"); do
    run B "$rep" "$BIN_DIR/task1" "$N_FIXED" --csv --out "$SCRATCH/b1.txt"
done
for t in $THREAD_LIST; do
    echo "  threads = $t"
    for rep in $(seq 1 "$REPS"); do
        run B "$rep" "$BIN_DIR/task2" "$N_FIXED" "$t" --csv --out "$SCRATCH/b2.txt"
        run B "$rep" "$BIN_DIR/task3" "$N_FIXED" "$t" --csv --out "$SCRATCH/b3.txt"
    done
    if ! cmp -s "$SCRATCH/b1.txt" "$SCRATCH/b2.txt" || \
       ! cmp -s "$SCRATCH/b1.txt" "$SCRATCH/b3.txt"; then
        echo "ERROR: outputs disagree at threads=$t" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------- Sweep C ---
;; esac

case " $SWEEPS " in *" C "*)
echo
echo "--- Sweep C: OpenMP schedule comparison (n=$N_FIXED, T=$FIXED_THREADS) ---"
# chunk 0 = the implementation default. For `static` that means ONE contiguous
# block per thread - the load-imbalanced scheme, and the same partitioning the
# pthread version uses. Any explicit chunk deals chunks round-robin instead,
# which already balances most of the sqrt(k) cost gradient. Both are measured
# so the comparison shows what the schedule *kind* is actually worth.
for cfg in "static 0" "static 4096" "dynamic 4096" "guided 0"; do
    set -- $cfg
    echo "  schedule = $1, chunk = $2"
    for rep in $(seq 1 "$REPS"); do
        run C "$rep" "$BIN_DIR/task3" "$N_FIXED" "$FIXED_THREADS" \
            --schedule "$1" --chunk "$2" --csv --out "$SCRATCH/c3.txt"
    done
done

;; esac

echo
echo "Done. $(( $(wc -l < "$RESULTS") - 1 )) rows -> $RESULTS"
echo "Median t_total per configuration:"
# Portable median: group by impl|n|threads, sort each group's times with sort(1)
# rather than awk's asort(), which is a gawk extension and absent from the
# mawk shipped in the container (and not guaranteed on CAAS either).
awk -F, 'NR>1 {printf "%s|%s|%s %s\n", $1, $2, $4, $10}' "$RESULTS" \
  | sort -k1,1 -k2,2g \
  | awk '{k=$1; v[k]=v[k]" "$2; c[k]++}
         END {for (k in c) {
                n = split(v[k], a, " ")
                m = (n % 2) ? a[(n+1)/2] : (a[n/2] + a[n/2+1]) / 2
                printf "  %-34s n=%-3d median %.4f s\n", k, n, m
              }}' \
  | sort
