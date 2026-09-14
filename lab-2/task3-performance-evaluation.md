# Task 3 — Performance Evaluation with Amdahl's and Gustafson's Laws

FIT3143 Lab #2. Supporting notes for the calculations and experimental design
behind slide section (c). All numbers below come from
`bench/analysis/summary.csv`, produced by `bench/analyse.py` from the raw
per-repetition CSVs in `bench/results/`.

**Machine.** 10 physical cores. Worker counts past 10 are measured deliberately,
to show what oversubscription costs. Every configuration is repeated 3 times and
reduced by **median**, not mean: one scheduling hiccup moves a mean of three but
leaves a median untouched, and these runs share the host with the OS.

**Denominator.** Every speedup in this document is against the **Week 4 Task 1
serial program at the same n**, as the specification mandates — never against
the parallel program's own one-worker run.

---

## 1. What has to be measured, and why the two laws need different experiments

The specification's warning is the crux of this task: *"these two laws have
different measurement assumptions, and you will need to design measurement
experiments to compute the serial/parallel fractions correctly."*

They cannot be fed the same measured fraction.

| | Amdahl | Gustafson |
|---|---|---|
| Held fixed | problem size *n* | execution time |
| Question | how much faster is one fixed workload? | how much more work fits in the same time? |
| Formula | S(W) = 1 / (s + p/W) | S(W) = s + p·W |
| *s* is the serial share **of what** | the **one-worker** execution | the **parallel** execution as actually run |
| So *s* is measured on | the serial program at that *n* | the P×T run being scored |

Feeding Amdahl a fraction taken from a P-way run is circular, and it is the
most common way this task goes wrong. As workers are added the parallel part
shrinks while the serial part does not, so the *measured* serial share of a
parallel run rises with W — in our data from 0.028 at W=1 to 0.152 at W=8.
Substituting that into Amdahl makes the "ceiling" sag as W grows, and a ceiling
that moves with the thing it is supposed to bound is not a bound at all.

So two separate experiments are run.

### Experiment A — Amdahl's fraction, from the serial program

The serial program (`week-4-lab-1/task1.c`) is instrumented with four
phase timers that partition its whole runtime:

```
t_alloc     allocate the result buffer            serial
t_compute   the trial-division search             PARALLEL (this is p)
t_merge     assemble the ordered result           serial
t_io        write the output file                 serial
t_total     whole run
```

```
s = median(t_alloc + t_merge + t_io) / (that + median(t_compute))
p = 1 - s
```

This depends only on *n*. It is the fixed-workload experiment Amdahl assumes,
and it is why the Amdahl curve in figures 6 and 7 is identical for the MPI and
hybrid implementations — correctly so, since both parallelise the same phase of
the same algorithm.

Measured on the serial run:

| n | s | p | ceiling S(∞) = 1/s |
|---|---|---|---|
| 20,000,000 | 0.0288 | 0.9712 | 34.8× |
| 43,497,000 | 0.0207 | 0.9793 | 48.4× |
| 100,000,000 | 0.0140 | 0.9860 | 71.7× |

**The serial fraction falls as n grows.** Trial division costs roughly
O(n^1.5/log n) while allocation, ordering and file I/O scale with π(n) ≈
n/log n. The parallel part outgrows the serial part, so the Amdahl ceiling
*rises* with n — 34.8× to 71.7× across our range. This is the answer to the
"increasing problem size" axis of the theoretical analysis: bigger n does not
make the program faster, it raises the limit on how much faster parallelism
could ever make it.

### Experiment B — Gustafson's fraction, from each parallel run

Gustafson's *s* is by construction the serial share of the parallel execution,
so it is measured on the run being scored. Both parallel programs are
instrumented with the same partition of their runtime:

```
t_alloc     buffer allocation                              serial
t_bcast     MPI_Bcast of n to every rank                   serial
t_compute   the distributed search   (MPI_MAX over ranks)  PARALLEL
t_lsort     each rank ordering its own results             PARALLEL  [Task 2]
t_comm      MPI_Gather + MPI_Gatherv of results            serial
t_sort      root ordering the gathered result              serial
t_io        root writing the output file                   serial
t_total     MPI_Init to just before MPI_Finalize
```

Two measurement decisions matter here:

- **`t_compute` is reduced with `MPI_MAX`, not averaged.** The wall-clock cost
  of a phase is when its *slowest* rank finishes; an average would hide load
  imbalance, which is exactly what the partitioning experiment is looking for.
  `MPI_MIN` is reduced alongside it and the ratio reported as `imbalance`.
- **`t_lsort` counts as parallel work.** Every rank runs it simultaneously on
  disjoint data. Charging it as serial would inflate *s* for the hybrid alone
  and make the two implementations incomparable.

A barrier precedes the compute timer so it measures the phase rather than
accumulated rank skew.

### The cross-check — Karp–Flatt

Neither law charges anything for communication, and both therefore over-predict.
To find out *which* of the two possible causes explains the shortfall, Amdahl is
inverted on the **measured** speedup to recover the experimentally determined
serial fraction:

```
e = (1/S - 1/W) / (1 - 1/W)          (undefined at W = 1)
```

If *e* is flat in W, the gap really is irreducible serial work. If *e* climbs
with W, the loss is **parallel overhead** that grows with the worker count —
broadcast, gather, memory bandwidth, oversubscription — which neither law
models. This distinction is what makes the empirical-vs-theoretical comparison
diagnostic rather than merely descriptive.

---

## 2. Results (n = 30M, sweep B)

Amdahl's s = 0.0236 at this n, giving a ceiling of 42.4×.

| W | OpenMP measured | MPI (Task 1) measured | Hybrid (Task 2) measured | Amdahl | Gustafson (MPI) |
|---|---|---|---|---|---|
| 1 | 1.00× | 1.00× | 0.99× | 1.00× | 1.00× |
| 2 | 1.90× | **0.94×** | 1.81× | 1.95× | 1.96× |
| 4 | 3.41× | 1.99× | 2.85× | 3.74× | 3.67× |
| 8 | 3.67× | 2.44× | 2.70× | 6.87× | 6.94× |
| 10 | 3.81× | 2.15× | 2.28× | 8.25× | 8.57× |
| 16 | 3.72× | 2.03× | 2.04× | 11.82× | 14.20× |

Karp–Flatt on the same runs:

| W | OpenMP e | MPI e | Hybrid e |
|---|---|---|---|
| 2 | 0.053 | **1.127** | 0.102 |
| 4 | 0.057 | 0.338 | 0.134 |
| 8 | 0.168 | 0.325 | 0.281 |
| 10 | 0.180 | 0.407 | 0.376 |
| 16 | 0.220 | 0.456 | 0.456 |

Read against the true serial fraction of 0.024, every one of these is far too
large, and all three climb with W. **The shortfall is overhead, not serial
work.** OpenMP's e stays under 0.22 — threads share an address space, so there
is nothing to gather. MPI's sits near 0.33–0.46, an order of magnitude above the
program's actual serial fraction, because every rank's result vector has to
cross a process boundary through `MPI_Gatherv` and the root has to merge them.

The MPI value at W = 2 (e = 1.13, speedup 0.94×) is the clearest single result
in the set: **two ranks are slower than one.** At that point the gather of half
the primes costs more than halving the search saved. This is the honest answer
to "will more processes always increase the speedup" — here, going from one
process to two makes it strictly worse.

---

## 3. Answers to the questions the specification poses

**How does the actual speedup compare against the theoretical speedup?**
It is far below both, and the gap widens with W: at 8 workers, MPI achieves
2.44× against an Amdahl prediction of 6.87×, and at 16 workers 2.03× against
11.82×. The two laws bracket nothing useful here because both model only the
division of work and neither models its *re-collection*. Karp–Flatt localises
the discrepancy: e rises monotonically with W in all three implementations, so
the missing speedup is communication and memory-bandwidth overhead, not serial
code. Gustafson runs *above* Amdahl past W = 8 (14.20× vs 11.82× at W = 16),
which is expected — it assumes the workload grows with the machine, and ours
does not, so for this fixed-n experiment Amdahl is the more honest model and
Gustafson is reported to show what the same code would promise under a
weak-scaling regime.

**Will more MPI processes always increase the speedup?**
No, in either sense. Theoretically Amdahl saturates: at n = 30M the ceiling is
42.4× no matter how many processes are added, and the returns die long before
that — W = 40 buys 20.8×, less than half the ceiling, and reaching 90% of it
would take roughly 380 workers. Empirically it is worse than saturation — the
measured MPI curve *peaks at W = 9 (2.51×) and then declines*, to 2.15× at 10
and 2.03× at 16. Past 10 workers there are no more physical cores, so extra
ranks add gather traffic and context switching without adding compute. And at
the small end, W = 2 is slower than W = 1.

**How does the workload distribution affect the speedup?**
Substantially, and it is the one variable that costs nothing to change. At
n = 30M with 8 workers (sweep C), holding everything else fixed:

| scheme | MPI speedup |
|---|---|
| blockcyclic | **2.82×** |
| cyclic | 2.67× |
| dynamic | 2.38× |
| block | 2.09× |

`block` is worst because trial-division cost grows with the candidate's
magnitude — the rank holding the top block does far more work than the rank
holding the bottom one, and the phase ends when the slowest rank ends.
`blockcyclic` interleaves small strided chunks so every rank draws a mix of
cheap and expensive candidates, recovering 35% over `block` on identical
hardware and identical arithmetic. `dynamic` balances well in principle but
pays master–worker request traffic that, at this granularity, exceeds the
imbalance it removes.

**Will the speedup results be the same across different machines?**
No. The Amdahl ceiling is a property of the *program and n* and will transfer
(s = 0.024 at 30M is a property of the algorithm's phase mix). Everything
empirical is a property of the *host*: core count sets where the curve turns
over, memory bandwidth per core sets how early efficiency decays, and
interconnect sets the gather cost that dominates our Karp–Flatt numbers. A
machine with more cores would move the peak right; one with faster
core-to-core bandwidth would lower e and close the gap to theory without any
code change.

---

## 4. Reproducing

```sh
bench/sweep.sh          # serial, pthreads, OpenMP baselines
bench/sweep-mpi.sh      # Task 1
bench/sweep-hybrid.sh   # Task 2
python3 bench/analyse.py --out bench/analysis    # fractions, both laws, Karp-Flatt
python3 bench/report.py  --out bench/analysis    # figures 1-7 + report.html
```

Figures 6 and 7 (`bench/analysis/figures/fig6.svg`, `fig7.svg`) are the two
graphs section (c) requires: measured against Amdahl against Gustafson, for
Task 1 over increasing MPI processes and for Task 2 over increasing ranks ×
threads.
