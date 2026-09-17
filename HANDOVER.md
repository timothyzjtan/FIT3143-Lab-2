# Handover — Lab 2 working-tree changes

**Status:** 4 files modified, **nothing committed**. Nothing compile-checked
(the Windows box has no Docker daemon running) — run `make -C lab-2` first.

---

## 1. What changed

| File | Change |
|---|---|
| `lab-2/task2.c` | default `--scheme` is now `blockcyclic` (was `dynamic`); 3 lines incl. the `usage()` text |
| `bench/sweep-mpi.sh` | `MAIN_SCHEME` default `dynamic` → `blockcyclic`, plus comment rewrite |
| `bench/sweep-hybrid.sh` | same |
| `bench/sweep.sh`, `sweep-mpi.sh`, `sweep-hybrid.sh` | `N_MIN` `N_MAX` `N_STEPS` `N_FIXED` `REPS` changed from plain assignment to `${VAR:-default}` |

`lab-2/task1.c` untouched. No docs, figures or data touched.

---

## 2. Why the scheme change (the important one)

The spec states, twice and explicitly:

> "Each process **(including the root process)** will be tasked with a share of
> the workload to compute the prime number." — Task 1
>
> "Each MPI process **(including the root process)** will create a set of
> threads to compute the prime numbers in parallel." — Task 2

Under `--scheme dynamic`, rank 0 is a pure dispatcher. In `task1.c` the
`rank == 0` branch contains no `scan_range` call at all; in `task2.c` it
contains no `omp parallel` region, so the root creates zero threads. **That
violates an explicit coding requirement**, independent of performance.

`task1.c` already defaulted to `blockcyclic`. `task2.c` did not — and the run
command in its own file header carries no `--scheme` flag:

    mpirun --bind-to none -np 4 ./task2 10000000 --threads 2

So the invocation the file documents was landing on the non-compliant path. A
marker who compiles it and runs the documented command hits the violation
immediately. Now fixed.

Performance agrees with the compliance argument. From sweep A, n=1e8, P=8:

| scheme | speedup |
|---|---|
| `mpi-blockcyclic` | **3.72x** |
| `openmp` | 3.73x |
| `mpi-dynamic` | 2.47x |

`dynamic` gives up 1/P of the machine to the master before it schedules
anything, then serialises ~7,325 chunk requests through that one rank. On the
cluster at 64-128 ranks this gets worse, not better.

**`dynamic` was NOT deleted.** It is still in both source files and still in
`SCHEMES`, so sweep C and fig5 keep the four-way comparison. We keep the
evidence that we explored the options — the spec asks for exactly that — we
just stop using the losing scheme for the headline numbers.

---

## 3. Why `N_FIXED` became overridable — action needed from you

Sweep B is pinned at n = 30,000,000 while `mpi.sbatch` runs
`PROC_LIST="1 2 4 8 12 16 24 32 48 64"` and `mpi-8node.sbatch` goes to 128:

| ranks | projected sweep-B runtime at n=30M |
|---|---|
| 16 | ~0.18 s |
| 32 | ~0.09 s |
| 64 | ~0.04 s |

Spec: *"avoid showcasing runtime results that are too small (e.g., < 1 seconds)
that can be easily be affected by measurement noises."* At 32-64 ranks we would
be 10-25x under that floor, and the noise would read as an efficiency collapse
that isn't real.

`N_FIXED` and friends were plain assignments, so `export N_FIXED=...` from an
sbatch script was silently clobbered. They now honour the environment.

**To do:** add the same line to **all three** of `baselines.sbatch`,
`mpi.sbatch` and `hybrid.sbatch`:

    export N_FIXED=200000000

All three, because `analyse.py` pairs each parallel row with the serial row **at
the same n**. If baselines stays at 30M, every sweep-B row comes back with no
speedup to divide by.

200M puts the 64-rank point near 1 s and the 1-rank point near 50 s. Watch the
28-minute cap — `baselines.sbatch` already notes the serial runs are ~45 min at
the current ladder. If array task 1 times out, drop to 150M.

Sweep A's ladder (20M-100M) is fine as-is and needs no change.

---

## 4. Consequence: the committed figures are now stale

`bench/analysis/` was generated before this change:

- figs **1, 2** came from `blockcyclic` (`analyse.py`'s `pick_A` selects the
  fastest variant per family in sweep A)
- figs **3, 4, 6, 7, 8** came from `dynamic` (sweep B only ever ran that scheme)

Both are labelled "MPI (Task 1)". So the text quoting 3.72x and the graph
showing 2.44x refer to different programs, with nothing explaining the gap.
Re-running sweep B fixes the inconsistency and raises the headline number.

---

## 5. Findings with no code change — worth knowing

1. `task4-documentation.md` §2.1 says "Open MPI 5.x". It is **4.1.6** in Docker
   (Ubuntu 24.04 ships 4.1.6-7ubuntu2) and **4.1.5** on CAAS per `env.sh`. That
   also means the v5.0.x man pages the spec links are the wrong version for us —
   v4.1 docs live at `open-mpi.org/doc/v4.1/`, and use "socket" where v5 says
   "package".
2. §2.1 says "30 distinct values of *n*" — `summary.csv` has **31**.
3. The Discussion blames "two processes are slower than one" on communication
   overhead. It was the idle master: at P=2, `dynamic` has exactly one worker
   (`t_compute` 2.144 s vs serial 2.049 s). Goes away with the scheme change.
4. §3.2's scheme table and every speedup quoted in prose need refreshing after
   the re-run.
5. `week-4-lab-1/task3.c` (OpenMP) writes a flag per candidate into an n-byte
   array then does an **O(n) serial compaction**, while serial, pthreads and both
   MPI versions append primes directly. So OpenMP's measured serial fraction
   includes work the MPI versions never do — its non-compute time is visibly
   higher and the gap scales with n (0.025 s at 30M, 0.090 s at 100M). Worth one
   sentence in §5 rather than a code change; altering it would invalidate the
   existing data.
6. `task1.c:376` guards `local.count > INT_MAX` before `MPI_Gatherv`, but four
   lines later `displs[i] = (int) acc` truncates a *cumulative* long. Unreachable
   at our n (needs ~5e10) — the error message just overclaims.
7. `task2.c:69`'s `--bind-to` comment misses the third default (oversubscribed →
   bind to none) and doesn't mention that on CAAS the launcher is
   `srun --cpu-bind=none`, not `mpirun --bind-to` — which `env.sh` already gets
   right.
8. `MPI_Get_processor_name` is not used anywhere. The SLURM logs record the node
   list so we aren't evidence-less for the multi-node requirement, but per-rank
   node IDs in the CSV would be stronger for the HD/D band.

---

## 6. Still outstanding

**Blocking:**

- **No cluster run has executed yet.** `bench/slurm/logs/`, `bench/results-caas/`
  and `bench/analysis-caas/` are all absent. The rubric requires cluster analysis
  for both the HD *and* D bands on Task 1 and Task 2 — 30% of the mark is capped
  at C until this runs.
- `⟨Name 1⟩` / `⟨Student ID⟩` / `⟨email⟩` placeholders — 4 across both task4 docs.
- AI declaration still `⟨Required…⟩` in both. The checklist wants a separate PDF.
- No PDF/pptx export. 1,214 lines of markdown across three documents against a
  7-minute slot.

**Hygiene:**

- `README.md` currently reads `test again and again`.
- Tracked junk: `lab-2/task1`, `lab-2/task2` (ARM64 ELF binaries), `.DS_Store`,
  `bench/__pycache__/*.pyc`. `.gitignore` covers none of them.

---

## 7. Suggested order

    make -C lab-2                    # picks up the task2.c default
    # add `export N_FIXED=...` to the three sbatch files
    bench/caas.sh push
    bench/caas.sh build
    bench/caas.sh submit smoke       # wait for SMOKE PASS before anything else
    bench/caas.sh submit baselines
    bench/caas.sh submit mpi
    bench/caas.sh submit hybrid
    bench/caas.sh submit mpi-8node   # optional, last
    bench/caas.sh pull
    bench/caas.sh analyse            # -> bench/analysis-caas/report.html

Then update the doc numbers from `bench/analysis-caas/` and clear the submission
items in §6.
