# Archive: CAAS run of 2026-09-17, `dynamic` scheme (superseded)

Jobs 133801 (baselines), 133803 (mpi), 133804 (hybrid), each a 1-3 array.

**Why this data is not usable for the report.** Every parallel row was measured
with `--scheme dynamic`, under which rank 0 is a pure dispatcher: it computes no
primes in Task 1 and creates no OpenMP threads in Task 2. That violates the
spec's explicit "including the root process" requirement for both tasks, so the
sweeps were re-run with `blockcyclic` as the default.

Row counts: 120 `mpi-dynamic`, 144 `hybrid-dynamic`, vs 3 each for the
compliant schemes (sweep C only).

**Known defect.** Array task `133801_2` hit TIMEOUT at 00:28:03 and lost the
n=1e8 point: the A ladder was split 15/15 by count, but per-n cost grows as
~n^1.5, so points 16-30 needed ~29 min against a 28-min wall.

Kept for the measurements that remain valid regardless of scheme -- CAAS serial
timings (n=1e8 -> 53.7 s, ~4.7x the Mac), which is what the re-run's array
slicing and N_FIXED=1e8 were sized from.
