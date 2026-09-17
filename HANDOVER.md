# Handover — FIT3143 Lab 2

**Status as of 2026-09-17.** Code complete and verified on the cluster. The
cluster sweeps have run and been analysed. Task 3's write-up is rebuilt against
the new data. **The two Task 4 documents still carry pre-cluster numbers and are
the main remaining work.**

Repo-level orientation (directories, sweep knobs, CSV schemas, operational
traps) lives in `CLAUDE.md`. This file tracks only *deliverable status*.

---

## 1. Where things stand

| Item | State |
|---|---|
| `lab-2/task1.c` (Task 1, MPI) | Done. Never needed changing — already defaulted to `blockcyclic` |
| `lab-2/task2.c` (Task 2, hybrid) | Done. Default scheme changed `dynamic` → `blockcyclic` |
| Correctness | Verified on 2 nodes: all 7 scheme configurations matched serial byte-for-byte (664,579 primes at n=1e7) |
| Cluster sweeps | Done. 11/11 array tasks COMPLETED, no errors, 30 distinct *n* |
| `bench/analysis-caas/` | Regenerated: `report.html` + 8 SVGs |
| `lab-2/task3-performance-evaluation.md` | **Rewritten** against cluster data |
| `lab-2/task4-documentation.md` | **Stale** — pre-cluster numbers, 3 placeholders |
| `lab-2/task4-presentation.md` | **Stale** — same |

### Headline results (CAAS, 4 nodes × 16 cores, n = 1e8)

Serial reference 53.72 s. Amdahl's s = 0.0090, ceiling 110.9×.

All from **sweep B**, so they agree with the tables in
`task3-performance-evaluation.md` (sweep A medians differ by ~1%):

| implementation | config | speedup |
|---|---|---|
| Hybrid (Task 2) | 8×8 | **30.37×** |
| Hybrid (Task 2) | 4×16 | 30.01× |
| MPI (Task 1) | 64×1 | 23.87× |
| OpenMP | 1×16 | 13.39× |
| MPI (Task 1) | 16×1 | 12.26× |
| Pthreads | 1×16 | 10.21× |

The scheme fix moved the Task 1 headline from 2.47× to 23.87×. `blockcyclic`
is both the specification-compliant choice and the fastest one, in both tasks —
there is no trade-off to defend.

---

## 2. Next steps, in order

**1. Rewrite `lab-2/task4-documentation.md` (604 lines).** Largest remaining
job. Every quoted speedup, the §3.2 scheme table and the Discussion all predate
the cluster. Three specific corrections beyond the numbers:

- §2.1 says "Open MPI 5.x". It is **4.1.6** in Docker and **4.1.5** on CAAS.
  The v5.0.x man pages the spec links are the wrong version for us — v4.1 docs
  are at `open-mpi.org/doc/v4.1/`, and say "socket" where v5 says "package".
- §2.1's "30 distinct values of *n*" is now exactly right for the CAAS data
  (it was 31 on the old Mac run). Leave as-is, but re-point it at the cluster.
- The Discussion blames "two processes are slower than one" on communication
  overhead. That result **no longer exists** — it was the idle master under
  `dynamic`. P=2 now gives 1.96×. Delete the passage rather than rewording it.

Use `lab-2/task3-performance-evaluation.md` as the model; its numbers are
already verified against `bench/analysis-caas/summary.csv`.

**2. Rewrite `lab-2/task4-presentation.md` (370 lines).** Same numbers, cut to
a 6-7 minute delivery. The rubric's HD band requires 6-7 min and explicitly
rewards parallel-computing terminology over general computing terms.

**3. Decide how the figures ship.** `bench/analysis-caas/` is **gitignored**
(`.gitignore:10`), but the Task 3 document now references
`bench/analysis-caas/figures/fig6.svg` and `fig7.svg`. As things stand those
figures are not in the repo and would not reach a marker. Either un-ignore the
directory, or copy the 8 SVGs to a tracked location and re-point the docs.

**4. Fill the placeholders.** 3 in `task4-documentation.md` (lines 5, 6, 560),
3 in `task4-presentation.md`: names, student IDs, `@student.monash.edu`
addresses, and the AI declaration. The submission checklist wants the AI
declaration as a **separate PDF** if it is not inline.

**5. Export to PDF/pptx.** Nothing has been exported. The checklist asks for
slides or documentation in a presentable format.

**6. Hygiene** (small, do before submitting):

- `README.md` reads `test again and again`.
- Still tracked and shouldn't be: `lab-2/task1`, `lab-2/task2` (aarch64 ELF
  binaries), `.DS_Store`, `bench/__pycache__/analyse.cpython-314.pyc`.
  `.gitignore` covers the `week-4-lab-1/` binaries but not the `lab-2/` ones.
- `bench/slurm/env.sh:4` still says the partition has a "30-minute hard limit".
  It is **20 minutes** now — see §3.

---

## 3. Open findings — still true, no action taken

1. **`defq`'s wall limit dropped to 00:20:00** (it was ≥28 min on 2026-09-17
   morning). A job requesting more sits at `PartitionTimeLimit` *forever*
   rather than failing at submit. Always check `scontrol show partition defq`
   before trusting a runtime estimate. The `.sbatch` files are sized for 20 min;
   `env.sh`'s comment is stale.
2. **`week-4-lab-1/task3.c` (OpenMP) does an O(n) serial compaction** — it
   writes a flag per candidate into an n-byte array (`task3.c:180`) then scans
   it (`:193`), while serial, pthreads and both MPI versions append primes
   directly. So OpenMP's measured serial fraction includes work the MPI versions
   never do. Worth one sentence in the write-up; changing the code would
   invalidate the existing data.
3. **`lab-2/task1.c:402`** — `displs[i] = (int) acc` truncates a *cumulative*
   long, four lines after a guard that only checks the per-rank count against
   `INT_MAX`. Unreachable at our n (would need ~5e10); the error message just
   overclaims.
4. **`lab-2/task2.c:69`'s `--bind-to` comment** misses the oversubscribed
   default and doesn't mention that on CAAS the launcher is
   `srun --cpu-bind=none`, not `mpirun --bind-to`. `env.sh` already gets this
   right.
5. **`MPI_Get_processor_name` is unused.** SLURM logs record the node list, so
   the multi-node requirement is evidenced, but per-rank node IDs in the CSV
   would be stronger for the HD/D band.
6. **`dynamic` was deliberately kept**, not deleted. It remains selectable in
   both programs and in `SCHEMES`, so sweep C keeps the four-way comparison and
   we retain evidence that alternatives were explored — which the specification
   explicitly asks for. It is simply not the default, and not used for any
   headline number.

---

## 4. Re-running the cluster sweeps

```sh
bench/caas.sh push
bench/caas.sh build
bench/caas.sh submit smoke        # wait for SMOKE PASS before anything else
bench/caas.sh submit baselines    # 4-task array
bench/caas.sh submit mpi          # 3-task array
bench/caas.sh submit hybrid       # 3-task array
bench/caas.sh submit mpi-8node    # optional, last
bench/caas.sh pull
bench/caas.sh analyse             # -> bench/analysis-caas/report.html
```

**Before re-analysing, clear stale CSVs out of `bench/results-caas/`.**
`analyse.py` globs the whole directory and will silently pool runs from
different `N_FIXED` values into one median. The superseded 2026-09-17 `dynamic`
run is preserved in `bench/archive-caas-dynamic-20260917/` (local) and
`bench/results-caas-retired/` (cluster).
