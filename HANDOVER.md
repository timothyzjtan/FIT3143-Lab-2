# Handover — FIT3143 Lab 2

**Status as of 2026-09-17 (later).** Code complete and verified on the cluster.
The cluster sweeps have run and been analysed. Task 3's write-up is now
**complete against all six axes the specification lists** (empirical and
theoretical, each over n / processes / threads-per-process). **The two Task 4
documents still carry pre-cluster numbers and are the only substantial work
left.**

> **Uncommitted right now:** `bench/slurm/env.sh` (+13/−2) and
> `lab-2/task3-performance-evaluation.md` (+237/−25). Commit or review before
> pulling.

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
| `lab-2/task3-performance-evaluation.md` | **Complete** — 531 lines, all six spec axes, verified arithmetic *(uncommitted)* |
| `bench/slurm/env.sh` | OpenMPI module pinned to the documented name *(uncommitted)* |
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

**3. ~~Decide how the figures ship.~~ RESOLVED** by commit `3525258` —
`bench/analysis-caas/` is now tracked (10 files, including all 8 SVGs), so the
Task 3 document's references resolve for a marker.

**4. Fill the placeholders.** 3 in `task4-documentation.md` (lines 5, 6, 560),
3 in `task4-presentation.md`: names, student IDs, `@student.monash.edu`
addresses, and the AI declaration. The submission checklist wants the AI
declaration as a **separate PDF** if it is not inline.

**5. Export to PDF/pptx.** Nothing has been exported. The checklist asks for
slides or documentation in a presentable format.

**6. Hygiene** (small, do before submitting):

- `README.md` still reads `test again and again`.
- Still tracked and shouldn't be: `.DS_Store` and
  `bench/__pycache__/analyse.cpython-314.pyc`. (The `lab-2/` binaries were
  untracked in `cf41c51` — these two remain.)
- ~~`env.sh:4`'s "30-minute hard limit"~~ — **fixed**; it now states 20 minutes
  and says to check `scontrol show partition defq` before trusting a `--time`.

---

## 3. Open findings and measurement caveats

1. **`defq`'s wall limit dropped to 00:20:00** (it was ≥28 min on 2026-09-17
   morning). A job requesting more sits at `PartitionTimeLimit` *forever*
   rather than failing at submit. Always check `scontrol show partition defq`
   before trusting a runtime estimate. The `.sbatch` files are sized for 20 min,
   and `env.sh`'s comment now says so.
2. **`week-4-lab-1/task3.c` (OpenMP) does an O(n) serial compaction** — it
   writes a flag per candidate into an n-byte array (`task3.c:180`) then scans
   it (`:193`), while serial, pthreads and both MPI versions append primes
   directly. So OpenMP's measured serial fraction includes work the MPI versions
   never do. **Now stated in the Task 3 write-up** (the "Comparison against the
   Week 4 baselines" section), framed as making OpenMP's low Karp–Flatt figure
   conservative rather than flattering. Changing the code would invalidate the
   existing data, so it stays as a caveat.
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

7. **`analyse.py` drops the `imbalance` column.** Both programs reduce
   `t_compute` with `MPI_MAX` *and* `MPI_MIN` and emit the ratio, but it only
   survives in `bench/results-caas/mpi-*.csv`, not in `summary.csv`. The
   measured figures are worth having: at n=100M, 16 ranks —
   `block` **4.54×**, `blockcyclic` 1.04×, `cyclic` 1.01×, `dynamic` 1.00×.
   Consider propagating the column next time `analyse.py` is touched.
8. **The imbalance measurement confirms a prior prediction.** Modelling chunk
   cost as ∝ √k and summing per rank predicted 3.4–4.1× for `block` and
   1.0012× for `blockcyclic`; measured 4.54× and 1.04×. Same order, and
   `blockcyclic`'s near-perfect balance predicted almost exactly. This is in the
   write-up as a model-then-measurement result.
9. **`dynamic` achieves *perfect* balance (1.00×) and still loses.** That kills
   the "it was just badly tuned" counter-argument: balance was never the binding
   constraint — there was only 4% left to recover, and it spends a whole rank
   plus per-chunk request traffic to get it. Strong Q&A material.
10. **43 of 207 measured configurations run under 1 second**, the fastest being
    the hybrid at 4×16 finishing n=20M in **0.33 s** — against the
    specification's explicit warning about sub-second results. All are
    high-worker-count runs at n ≤ 54M. The effect is visible: hybrid speedup
    scatters 14.4% between adjacent low-*n* points. The write-up now states this
    openly and pins every headline figure to n=100M (fastest configuration
    1.76 s). **Do not quote a low-*n* hybrid point as a precise result.**
11. **No discontinuity at the node boundary.** W=16 fits one node, W=24 spans
    two, yet efficiency falls smoothly (76.6% → 70.1%) and *e* dips slightly
    (0.0203 → 0.0185). The interconnect shows up as a trend past W≥48, not as a
    step — which is what the fixed-gather-cost model predicts.
12. **No oversubscription data on CAAS.** Worker counts stop at exactly 64 on a
    64-core allocation, so the specification's "what if you exceed the core
    count" question is answered from the old laptop runs. Legitimate and
    labelled as such, but `mpi-8node` at 96/128 ranks would put it on the
    cluster if it ever schedules.

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
