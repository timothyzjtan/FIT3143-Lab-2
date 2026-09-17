# FIT3143 Lab 2 — prime search with MPI and hybrid MPI+OpenMP

Task 1 is a distributed prime search over MPI ranks; Task 2 adds OpenMP threads
inside each rank. Both are benchmarked against the Week 4 serial / pthreads /
OpenMP baselines, on a Mac via Docker and on the Monash CAAS SLURM cluster.

## Governing documents — read before changing behaviour

All work here is graded against two files in the repo root. **They are the
authority; this file is only a summary.** Re-read them before any change that
touches what is measured, plotted, or output:

- `Lab #2 - Assessment Specification.md` — the requirements (Tasks 0-4,
  required graphs, submission checklist).
- `Lab #2 Rubric.md` — the HD/D/C/P/N bands the above is marked against.

Binding constraints they impose:

- **Output**: a *sorted* list of primes strictly less than `n`, written to a text
  file by the root process. `n` arrives as a command-line argument to root and is
  disseminated to all ranks (and, in Task 2, reachable by every thread).
- **Root must do real work** — see Non-obvious rules below. Spec wording is
  "Each process (including the root process)" for Task 1 and "Each MPI process
  (including the root process) will create a set of threads" for Task 2.
- **Speedup is `t_total`**, explicitly "including communication, computation,
  sorting and file writing", against the Week 4 **serial** (`week-4-lab-1/task1`)
  at the same `n`. Must be > 1; HD wants close to linear in core count.
- **≥ 30 distinct values of n** (hence the 30-point ladder), and avoid reporting
  configurations under ~1 s.
- **Sweep processes and threads from 1 up to at least the core count**, and past
  it to show oversubscription behaviour.
- **Cluster analysis (CAAS or local) is required for both HD and D** on Task 1
  *and* Task 2 — not optional.
- **Seven required graphs**: 3 for Task 1 (runtime vs n; speedup vs n; speedup vs
  MPI process count), 2 for Task 2 (vs Task 1 with increasing threads; vs
  pthreads/OpenMP at *matched total worker count*), 2 for Task 3 (empirical vs
  theoretical, for Task 1 and for the hybrid). `report.py` emits 8 figures
  covering these.
- **Amdahl and Gustafson take different serial fractions** — Amdahl's from the
  serial run at the same `n`, Gustafson's from the parallel run being scored.
  `analyse.py` already separates them; do not feed both the same number.
- Submission: `task1.c`, `task2.c`, slides/documentation, optional Task 3 notes,
  and a separate AI-declaration PDF if not declared inline.

## Directories

| path | contents |
|---|---|
| `lab-2/` | `task1.c` (MPI), `task2.c` (hybrid), `Makefile`, and the `task3-*` / `task4-*` report documents |
| `week-4-lab-1/` | `task1.c` serial, `task2.c` pthreads, `task3.c` OpenMP. These are the **measurement instrument**, not old work — the serial binary is the speedup denominator and the correctness oracle |
| `bench/` | `sweep.sh` (baselines), `sweep-mpi.sh` (Task 1), `sweep-hybrid.sh` (Task 2), `analyse.py` → `summary.csv`, `report.py` → 8 SVGs + `report.html`, `caas.sh` cluster driver |
| `bench/slurm/` | one `.sbatch` per job (`smoke`, `baselines`, `mpi`, `hybrid`, `mpi-8node`, `launcher-probe`) plus `env.sh`, sourced by all of them |
| `bench/results/` | Mac/Docker CSVs (gitignored) |
| `bench/results-caas/` | cluster CSVs (gitignored). **Never pool with `results/`** — different hardware, pooling silently averages two machines into one median |
| `bench/analysis/`, `bench/analysis-caas/` | generated figures and `summary.csv` |
| `bench/archive-caas-dynamic-20260917/` | superseded cluster run (non-compliant `dynamic` scheme); kept for its still-valid CAAS serial timings |
| `docker/` | `Dockerfile` and `run.sh` (bind-mounts the repo at `/work`, caps `--cpus`) |

Root also holds `HANDOVER.md` (deliverable status — that churns, so it is
deliberately not duplicated here) alongside the spec and rubric named above.

## Build and run

    docker/run.sh make -C lab-2          # and make -C week-4-lab-1
    bench/sweep.sh --quick               # fast smoke sweep, tiny n

    bench/caas.sh push | build | submit <job> | status | pull | analyse

`analyse.py` must run before `report.py`; `caas.sh analyse` does both with
explicit `results-caas` globs.

## Non-obvious rules

- **Root must do real work.** The spec requires every process *including the
  root* to compute primes (Task 1) and to create threads (Task 2).
  `--scheme dynamic` makes rank 0 a pure dispatcher and violates both. Defaults
  are `blockcyclic` everywhere; `dynamic` survives only as a sweep-C comparison
  point. Do not restore it as a default.
- **CAAS launcher is plain `srun` with no `--mpi` flag.** `--mpi=pmix` and
  `--mpi=pmi2` both hang before the step launches; `mpirun` cannot spawn `orted`
  on compute nodes from inside an allocation.
- **`defq` `MaxTime` is 00:20:00** (it was 30 min until ~2026-09-17). A job
  asking for more sits at `PartitionTimeLimit` *forever* rather than failing at
  submit time. Run `scontrol show partition defq` before trusting any runtime
  estimate in these scripts.
- **`OMPI_MCA_orte_tmpdir_base=/tmp`** must stay set. Sweep scratch lives on NFS
  so all ranks share it, but Open MPI follows `TMPDIR` too — its shared-memory
  backing file on NFS pushes intra-node traffic over the network, a silent
  slowdown that lands in `t_compute` and corrupts every speedup number.
- **Sweep A's cost is dominated by the serial correctness run at each n**, not
  by the parallel run. Per-n cost grows ~n^1.5, so split ladders by *measured
  cost*, never by point count — an even 15/15 split is a 9-minute task and a
  29-minute one.
- CAAS single-core is ≈4.7× slower than the Mac (n=1e8 → 53.7 s serial).
- `t_compute` is reduced with `MPI_MAX` (the slowest rank gates the phase);
  `MAX/MIN` is reported as the imbalance figure.

## Sweep environment knobs

All three sweep scripts honour these (`${VAR:-default}`), which is how the
sbatch files parameterise them:

    REPS  N_MIN  N_MAX  N_STEPS  N_FIXED  LADDER_SLICE  SWEEPS
    PROC_LIST  THREAD_LIST  GRID  MAIN_SCHEME  SCHEMES  CHUNK
    LAUNCH  LAUNCH_THREADS_FLAG  FIXED_PROCS  FIXED_THREADS
    OUT_DIR  RUN_TAG

Plus a `--quick` flag. `SWEEPS` selects among A (vary n), B (vary workers),
C (vary partitioning scheme). `LADDER_SLICE="from:to"` takes a 1-indexed range
of the 30-point ladder without changing any n value, so one sweep can span
several jobs. `RUN_TAG` keeps concurrent array tasks from colliding on CSV names.

## CSV schemas

The three families differ, and `analyse.py` depends on the difference:

    baselines  impl,n,procs,threads,prime_count,t_alloc,t_compute,t_merge,t_io,t_total,rep,sweep
    mpi        impl,n,procs,threads,prime_count,t_alloc,t_bcast,t_compute,t_comm,t_sort,t_io,t_total,imbalance,rep,sweep
    hybrid     impl,n,procs,threads,prime_count,t_alloc,t_bcast,t_compute,t_lsort,t_comm,t_sort,t_io,t_total,imbalance,workers,rep,sweep

Speedup denominators come from `impl == "serial"` rows matched on `n`. Every
parallel row needs a serial row at the **same n** or it silently drops out of
the analysis — so `N_FIXED` must agree across the baselines, mpi and hybrid
jobs.
