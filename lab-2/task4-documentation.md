# Prime Search with the Message Passing Interface

**FIT3143 Parallel Computing — Lab #2 Report (Task 4)**

⟨Name 1⟩, ⟨Student ID⟩, ⟨email⟩@student.monash.edu
⟨Name 2⟩, ⟨Student ID⟩, ⟨email⟩@student.monash.edu

---

## Contents

1. [Introduction](#1-introduction)
2. [Experimental setup](#2-experimental-setup)
3. [Task 1 — Prime search with Open MPI](#3-task-1--prime-search-with-open-mpi)
4. [Task 2 — Hybrid Open MPI + OpenMP](#4-task-2--hybrid-open-mpi--openmp)
5. [Task 3 — Amdahl's and Gustafson's Laws](#5-task-3--amdahls-and-gustafsons-laws)
6. [Discussion](#6-discussion)
7. [Conclusions](#7-conclusions)
8. [Appendices](#8-appendices)

---

## 1. Introduction

This report extends the Week 4 prime search in two directions: a
message-passing version using Open MPI (Task 1) and a hybrid version that adds
OpenMP threads inside each MPI process (Task 2). Both are then evaluated
against Amdahl's and Gustafson's Laws (Task 3).

The problem is to find every prime strictly less than *n*, supplied on the
command line, and write the sorted result to a text file. The root process
broadcasts *n*, every process — the root included — searches a share of the
range, and the root gathers, sorts and writes the result.

**The trial-division kernel is byte-identical across all five
implementations** — the Week 4 serial, pthreads and OpenMP programs, and the
two documented here. Speedup is T_serial / T_parallel, so holding the
per-candidate work constant is what makes the comparison measure parallelism
rather than a change of algorithm. No sieve is used anywhere.

All results were measured on the Monash CAAS cluster. An earlier single-laptop
run is mentioned twice, labelled as such, for what it shows qualitatively.

---

## 2. Experimental setup

### 2.1 Platform

| Property | Value |
|---|---|
| Cluster | Monash CAAS, partition `defq` |
| Nodes | 4 × AMD EPYC 7763, 16 cores each — **64 cores total** |
| Interconnect | Gigabit ethernet |
| Toolchain | Open MPI 4.1.5, GCC 11.2, `-O2 -fopenmp`, launched with `srun` |
| Problem sizes | 30 distinct *n*, 20,000,000 → 100,000,000, geometric spacing |
| Repetitions | 3 per configuration, reduced by **median** |
| Baseline | Week 4 serial program, re-measured on the same nodes, at the same *n* |

The serial reference runs 5.49 s at *n* = 20M and 53.72 s at *n* = 100M.
Up to 16 workers fit on one node; every configuration above 16 spans several
nodes and pays for the interconnect. Nothing is oversubscribed — worker counts
stop at the 64-core allocation.

**Scheme.** Every headline figure uses `blockcyclic` with 4096-candidate
chunks, the default in both programs. The master–worker `dynamic` scheme
appears only in the scheme comparison (§3.3): it reserves rank 0 as a pure
dispatcher, which violates the specification's requirement that every process,
including the root, share the work.

**Problem size.** The specification warns against sub-second runtimes. 43 of
the 207 configurations fall below 1 s — all high-worker-count runs at
*n* ≤ 54M — and the hybrid's speedup visibly scatters (14% between adjacent
points) at the low end. **Every headline figure is therefore quoted at
*n* = 100M**, where the fastest configuration still takes 1.76 s. The low-*n*
points are kept for the trend.

### 2.2 Measurement design

Both parallel programs are timed in phases that partition the whole run, so the
serial and parallel fractions in §5 are *measured*, not assumed:

| Phase | Meaning | Class |
|---|---|---|
| `t_alloc` | buffer allocation | serial |
| `t_bcast` | `MPI_Bcast` of *n* | serial |
| `t_compute` | the distributed search | **parallel** |
| `t_lsort` | each rank sorting its own results (Task 2) | **parallel** |
| `t_comm` | `MPI_Gather` + `MPI_Gatherv` | serial |
| `t_sort` | root sorting the gathered result | serial |
| `t_io` | root writing the file | serial |
| `t_total` | `MPI_Init` to just before `MPI_Finalize` | — |

Three decisions shape every number that follows:

- **Median, not mean.** CAAS nodes may be shared with other workloads; one
  perturbed repetition moves a mean of three but not the median.
- **`t_compute` is reduced with `MPI_MAX`.** A phase ends when its *slowest*
  rank ends. `MPI_MIN` is reduced alongside it and the ratio reported as
  `imbalance` — the quantity §3.3 exists to measure.
- **`t_lsort` counts as parallel.** Every rank runs it simultaneously on
  disjoint data; charging it as serial would penalise the hybrid alone.

The serial baseline is timed with the matching subset (`t_alloc`, `t_compute`,
`t_merge`, `t_io`).

---

## 3. Task 1 — Prime search with Open MPI

### 3.1 Implementation and optimisations

Rank 0 parses *n* and broadcasts it with `MPI_Bcast`; every rank, including
rank 0, searches its share; results return in one `MPI_Gather` of counts
followed by one `MPI_Gatherv` of values; rank 0 sorts if needed and writes the
file. The optimisations that matter, in order of effect:

- **Kernel.** `is_prime` rejects even candidates in one modulo and tests only
  odd divisors up to ⌊√k⌋ — half the divisions of a naive loop. It is frozen,
  byte-for-byte, across all five programs.
- **No wasted candidates.** `cyclic` walks odd candidates only, starting at
  3 + 2r with stride 2P, so no rank is ever dealt a run of even numbers that
  the kernel would reject one by one.
- **Memory sized to the answer, not the input.** Each rank's buffer is seeded
  from the Rosser–Schoenfeld bound π(x) < 1.26 x / ln x divided by P, rather
  than `malloc(n)` — ~3.4 MB per rank at *n* = 100M and 16 ranks instead of
  800 MB — and grows only if the bound is exceeded (it never is).
- **One collective, not P messages.** Counts are gathered first so the root
  can build exact displacements; the values then move in a single `Gatherv`.
- **No sort where none is needed.** `block` ranks own ascending ranges, so
  the gathered vector is already ordered and the root skips `qsort`.
- **Honest timing.** A barrier precedes the compute timer, `t_compute` is
  reduced with `MPI_MAX` and `MPI_MIN`, and `t_total` spans `MPI_Init` to
  `MPI_Finalize`, so the speedup reported includes broadcast, gather, sort
  and file writing, as the specification requires.

### 3.2 Partitioning schemes

Trial division tests divisors up to √k, so **the cost of a candidate grows with
its value**. Dividing the *range* evenly does not divide the *cost* evenly. Four
schemes were implemented, all selectable with `--scheme`:

| Scheme | Allocation | Balance | Sort at root |
|---|---|---|---|
| `block` | rank *r* takes one contiguous range | poor — high ranks do more | not needed |
| `cyclic` | rank *r* takes k = 2+r, 2+r+P, … | very good | required |
| `blockcyclic` | 4096-candidate chunks dealt round-robin | very good | required |
| `dynamic` | rank 0 hands out chunks on request | best in principle | required |

`block` is the only scheme whose gathered results are already in order; the
others interleave and the root must sort. Sorting is timed separately and
charged only to the schemes that perform it.

### 3.3 Which scheme performs best

*n* = 100M, 16 ranks (one node), so the scheme is the only variable.
Imbalance is the measured `MPI_MAX / MPI_MIN` ratio of `t_compute`.

![Partitioning scheme comparison](../bench/analysis-caas/figures/fig5.svg)

| Scheme | Speedup | Imbalance | Karp–Flatt *e* |
|---|---|---|---|
| **`blockcyclic`** | **12.35×** | 1.04× | 0.020 |
| `cyclic` | 12.34× | 1.01× | 0.020 |
| `dynamic` | 11.54× | **1.00×** | 0.026 |
| `block` | 10.04× | **4.54×** | 0.040 |

`blockcyclic` and `cyclic` are indistinguishable and beat `block` by **23%** on
identical hardware and arithmetic — the cheapest win in the study. `block`'s
imbalance was **predicted before it was measured**: summing a √k cost per chunk
gives 3.4–4.1× for `block` and 1.0012× for `blockcyclic`, against 4.54× and
1.04× measured. `blockcyclic` is preferred over `cyclic` because a rank walks
4096 contiguous candidates at a time, keeping the cache locality a stride-P
interleave loses.

`dynamic` is the most instructive row. Its balance is **perfect** — exactly as a
master–worker scheduler promises — and it still loses 6.6%, because balance was
never the binding constraint. There was only 4% of imbalance left to recover,
and it spends a whole rank (15 of 16 compute, predicting a 6.3% loss) plus
request traffic to recover it.

### 3.4 Runtime against problem size *(required graph a-1)*

![Runtime against n](../bench/analysis-caas/figures/fig1.svg)

Log–log axes, 30 values of *n*. Task 1 at 16 ranks, the shared-memory
baselines at 16 threads, the hybrid at 4 ranks × 16 threads. Straight lines
mean runtime is a power law in *n*; the parallel lines sit below serial by a
near-constant offset, which is what constant speedup looks like in log space.

At *n* = 100M:

| Implementation | Configuration | Runtime |
|---|---|---|
| Serial (Week 4) | 1 core | 53.64 s |
| **Hybrid (Task 2)** | 4×16, four nodes | **1.756 s** |
| OpenMP | 1×16 | 3.977 s |
| **MPI (Task 1)** | 16×1 | 4.366 s |
| pthreads | 1×16 | 5.192 s |

### 3.5 Speedup against problem size *(required graph a-2)*

![Speedup against n](../bench/analysis-caas/figures/fig2.svg)

| *n* | serial | **MPI 16×1** | of Amdahl | Hybrid 4×16 | of Amdahl | OpenMP 1×16 | pthreads 1×16 |
|---|---|---|---|---|---|---|---|
| 20,000,000 | 5.49 s | 9.40× | 77% | 16.85× | 62% | 11.52× | 9.34× |
| 29,494,000 | 9.52 s | 10.33× | 84% | 18.79× | 66% | 11.46× | 9.06× |
| 43,497,000 | 16.48 s | 11.04× | 84% | 21.04× | 63% | 12.39× | 9.71× |
| 67,808,000 | 30.93 s | 12.16× | 89% | 24.41× | 65% | 12.45× | 9.93× |
| 100,000,000 | 53.72 s | **12.30×** | **87%** | **30.59×** | **75%** | 13.51× | 10.35× |

**Speedup improves with *n* for every implementation, and the MPI-based ones
gain most** — Task 1 +31%, Task 2 +82%, against OpenMP +17% and pthreads +11%.
Two effects compound: the serial fraction falls with *n* (§5.2), raising the
Amdahl ceiling for everyone; and the gather is a fixed cost set by π(n) while
compute grows as n^1.5 / log n, so communication shrinks as a *share* of the
run. The "of Amdahl" columns show the second effect — only the MPI-based
implementations *close* on their bound. Message passing is penalised at small
*n* and rewarded at large *n*.

### 3.6 Speedup against process count *(required graph a-3)*

*n* = 100M, one rank per core, 1 to 64 ranks. Ranks 1–16 sit on one node.

![Speedup against worker count](../bench/analysis-caas/figures/fig3.svg)

| Ranks | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Speedup** | 1.00× | 1.96× | 3.76× | 7.00× | 9.89× | 12.26× | 16.82× | 20.88× | 23.34× | **23.87×** |
| Efficiency | 100% | 98% | 94% | 87% | 82% | 77% | 70% | 65% | 49% | 37% |

Within one node the speedup is close to linear — 98% of ideal at 2 ranks, 94%
at 4, 87% at 8, 77% at 16 — and there is
**no step at the node boundary** — efficiency falls smoothly through 16 → 24
ranks. The curve **saturates rather than collapses**: 32 → 48 ranks buys 12%
more speedup, 48 → 64 buys 2% — the last 16 ranks return 2% for 33% more
hardware. §5.4 explains why.

On the earlier laptop run (10 cores, superseded scheme, numbers not quoted)
the curve *peaked and declined* past the core count. Both are real; they are
different regimes. Ranks beyond the cores add context switching to fixed
compute; ranks on new nodes add compute, just at a falling return.

---

## 4. Task 2 — Hybrid Open MPI + OpenMP

### 4.1 Two-level decomposition

**A chunk is the unit of work at both levels.** The MPI scheme decides which
chunks a rank owns; `#pragma omp for schedule(dynamic)` distributes that rank's
chunks across its threads. Every process, the root included, creates its own
OpenMP team; *n* reaches every thread through the rank's address space.

Inside a rank, `schedule(dynamic, 1)` hands chunks to whichever thread is
free, so the √k cost gradient balances across threads as it does across ranks.
Each thread appends into its **own private buffer** — there is no critical
section or atomic in the hot loop — and the rank concatenates and sorts them
once the team finishes. Buffers are sized from the same π(n) bound as Task 1,
divided by ranks × threads.

Two consequences follow. Threads pull chunks dynamically, so each rank sorts
its own results (`t_lsort`, parallel) before the gather; the root then sorts
globally only when the MPI scheme interleaves. And `cyclic` is absent from the
hybrid — it strides per candidate, so there is no chunk for the OpenMP level to
split. Under `dynamic`, rank 0's whole team idles with it: at 4×16 that is 16
of 64 workers, and the measured loss (30.33× → 21.78×, −28%) is close to the
25% the idle workers alone predict.

**One artifact worth recording.** Early hybrid results were by far the slowest
in the study, because the launcher's default binding pins each rank to one core
and leaves its threads nowhere to run. Every hybrid run therefore disables
binding (`srun --cpu-bind=none` on the cluster, `mpirun --bind-to none` in
Docker); everything measured before this was found was discarded.

### 4.2 Threads per rank at a fixed process count *(required graph b-1)*

Ranks fixed at 4, threads varied inside them, *n* = 100M. Task 1 has no
threads, so it is a flat line at its 4-rank speedup of 3.76×.

![Threads per rank at fixed process count](../bench/analysis-caas/figures/fig8.svg)

| Threads per rank | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Total workers | 4 | 8 | 16 | 32 | 64 |
| **Hybrid** | 3.79× | 7.06× | 12.64× | 20.98× | **30.01×** |
| Task 1 at 4×1 | 3.76× | 3.76× | 3.76× | 3.76× | 3.76× |
| Efficiency | 95% | 88% | 79% | 66% | 47% |
| Karp–Flatt *e* | 0.0186 | 0.0190 | 0.0177 | 0.0169 | 0.0180 |

**Karp–Flatt is flat across the entire thread axis.** On the rank axis (§3.6)
*e* held near 0.020 to 32 ranks and then climbed to 0.027 at 64. Adding ranks
accumulates overhead; adding threads does not — a thread shares its rank's
address space and contributes nothing to the gather volume or the message
count at the root. The same holds at P = 8 (12.78× → 30.37×, *e* 0.017–0.019)
and P = 16 (12.66× → 29.70×, *e* 0.018), so P = 4 is not a lucky choice.

### 4.3 Hybrid against shared-memory at matched worker counts *(required graph b-2)*

P ranks × T threads compared against OpenMP and pthreads at P × T threads. The
shared-memory programs cannot leave a node, so that comparison ends at 16.

![Speedup against worker count](../bench/analysis-caas/figures/fig3.svg)

| Total workers | 4 | 8 | 16 |
|---|---|---|---|
| OpenMP | 3.82× | 7.31× | 13.39× |
| pthreads | 2.88× | 5.45× | 10.21× |
| **Hybrid** (best split) | 3.79× (4×1) | 7.06× (4×2) | 12.78× (8×2) |
| MPI (Task 1) | 3.76× | 7.00× | 12.26× |

Within one node the hybrid is within 5% of OpenMP and ahead of pthreads, whose
static block partitioning suffers the same imbalance as `block` in §3.3.

Beyond 16 workers only the MPI-based programs can go, so the comparison
becomes hybrid against the *same program* at one thread per rank — the control
that isolates the threading level:

| Total workers | Best split | Speedup | Same program at P×1 | Advantage |
|---|---|---|---|---|
| 16 | 8×2 | 12.78× | 12.66× | +1% |
| 32 | 2×16 | 21.11× | 18.65× | +13% |
| 64 | 8×8 | **30.37×** | 22.52× | **+35%** |

At 64 workers the order is 8×8 (30.37×), 4×16 (30.01×), 16×4 (29.70×), 32×2
(24.78×), 64×1 (22.52×): **every split with ≥ 8 threads per rank beats every
split with ≤ 2**, and the three low-rank routes land within 2% of each other.
Once ranks are not the constraint, how the rest are split barely matters — only
that they are not spent on ranks.

The 64×1 hybrid (22.52×) trails Task 1 at 64 ranks (23.87×) by ~6%: the
standing cost of team setup and `t_lsort`, paid even with one thread. It is
what Task 1 saves by having one level, and what the hybrid recovers many times
over by using two.

---

## 5. Task 3 — Amdahl's and Gustafson's Laws

### 5.1 Two laws, two experiments

The specification warns that the two laws "have different measurement
assumptions". They cannot be fed the same fraction:

| | **Amdahl** | **Gustafson** |
|---|---|---|
| Held fixed | problem size *n* | execution time |
| Formula | S(W) = 1 / (s + p/W) | S(W) = s + p·W |
| *s* is the serial share of | the **one-worker** run | the **parallel** run as executed |
| So *s* is measured on | the serial program at that *n* | each P×T run being scored |

Feeding Amdahl a parallel run's fraction is circular: the measured serial share
of a parallel run *rises* with W (here 0.015 at W = 1 to 0.42 at W = 64,
because the parallel part shrinks and the serial part does not), so the
"ceiling" would sag with the very thing it is meant to bound. Gustafson's
fraction is *supposed* to behave that way — it describes how much of a larger
machine's time goes to coordination — so it is taken from each parallel run
using the phase table in §2.2.

### 5.2 Amdahl's fraction, from the serial program

```
s = median(t_alloc + t_merge + t_io) / [ that + median(t_compute) ]
```

| *n* | *s* | Ceiling 1/s |
|---|---|---|
| 20,000,000 | 0.0212 | 47.1× |
| 45,979,000 | 0.0149 | 67.0× |
| 100,000,000 | 0.0090 | **110.9×** |

**The serial fraction falls as *n* grows.** Trial division costs
~O(n^1.5 / log n) while allocation and I/O scale with π(n) ≈ n / log n. A
larger *n* does not make the program faster — it raises the limit on how much
faster parallelism could make it. The fraction depends only on *n*, so the
Amdahl curve is the same for both tasks, correctly: they parallelise the same
phase of the same algorithm.

### 5.3 Karp–Flatt: separating serial work from overhead

Neither law charges for communication, so both over-predict. Inverting Amdahl
on the *measured* speedup recovers the experimentally determined serial
fraction, `e = (1/S − 1/W) / (1 − 1/W)`, and its *trend* says why the
prediction missed: flat *e* is genuine serial work, rising *e* is overhead that
grows with W, falling *e* is load imbalance at small W.

At *n* = 100M (true serial fraction 0.0090):

| Workers | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| **MPI** | 0.0229 | 0.0211 | 0.0205 | 0.0203 | 0.0172 | 0.0225 | **0.0267** |
| Hybrid (best split) | — | 0.0186 | 0.0190 | 0.0168 | 0.0167 | — | 0.0176 |
| OpenMP | 0.018 | 0.016 | 0.014 | 0.013 | — | — | — |
| pthreads | **0.275** | 0.130 | 0.067 | 0.038 | — | — | — |

- **MPI is flat at ≈ 0.020 to 32 ranks, then rises.** The program scales as
  well as its serial side allows until the interconnect starts to cost more
  than it saves at 48+. Even the flat value is 2.2× the true serial fraction:
  the residual is the gather and the root's sort, which the serial program
  never does and which the model can only absorb into the one term it has.
- **The hybrid is flat everywhere** — its best splits reach 64 workers with at
  most 16 ranks, so the gather never grows.
- **OpenMP is the lowest in the study** (0.013): nothing gathered, and
  `schedule(dynamic)` balances the √k gradient. This is despite the Week 4
  OpenMP program compacting an *n*-byte flag array with an O(n) serial sweep
  the other programs do not have, so the comparison is conservative.
- **pthreads *falls* with W** — load imbalance from static block partitioning,
  worst with two blocks. The shared-memory counterpart of `block`.

### 5.4 Measured against theoretical *(required graphs c-1 and c-2)*

**Task 1, *n* = 100M, s = 0.0090:**

![Task 1 measured vs theoretical](../bench/analysis-caas/figures/fig6.svg)

| Ranks | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| **Measured** | 1.00× | 1.96× | 3.76× | 7.00× | 12.26× | 20.88× | 23.34× | 23.87× |
| Amdahl | 1.00× | 1.98× | 3.89× | 7.53× | 14.09× | 25.01× | 33.72× | 40.82× |
| Gustafson | 1.00× | 1.97× | 3.81× | 7.21× | 12.54× | 21.33× | 27.99× | 37.49× |

**Task 2, ranks fixed at 4, threads increasing:**

![Task 2 measured vs theoretical](../bench/analysis-caas/figures/fig7.svg)

| Workers (4 × T) | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| **Measured** | 3.79× | 7.06× | 12.64× | 20.98× | 30.01× |
| Amdahl | 3.89× | 7.53× | 14.09× | 25.01× | 40.82× |
| Gustafson | 3.84× | 7.29× | 13.47× | 22.67× | 36.75× |

**Measurement tracks Amdahl to about 32 workers and diverges after** — Task 1
reaches 93% of its prediction at 8 ranks, 83% at 32, 58% at 64. The hybrid
gets closer at every W (74% at 64) because it reaches the same worker count
with a quarter of the ranks.

**Gustafson sits below Amdahl throughout** (37.49× vs 40.82× at 64). Its *s*
comes from the parallel run and grows with W, pulling s + p·W down as the
machine grows; Amdahl's stays fixed. For this fixed-*n* experiment Amdahl is
the right model; Gustafson shows what the same code would promise under weak
scaling, where the comparison would invert.

**The 64-rank shortfall is 0.93 s** (Amdahl's 40.82× is 1.32 s against the
53.72 s reference; measured is 2.25 s), and it is made of three things Amdahl's
*s* cannot see because the serial program has none of them:

1. **A fixed-size gather.** π(10⁸) = 5,761,455 primes × 8 B = 46.1 MB, a
   0.37 s floor at 1 Gb/s — 40% of the shortfall by itself. Compute per rank
   falls as 1/W; this does not fall at all.
2. **A serial sort at the root.** Interleaved chunks arrive unordered, so the
   root `qsort`s 5.76 M values. The serial program appends in order and its
   `t_merge` is exactly zero.
3. **Per-message latency × 64.** `MPI_Gatherv` at 64 ranks is 64 transfers to
   one receiver, however little each contributes.

Both laws model how work is *divided*; neither models how it is
*re-collected*. That is why the curves diverge, and why Karp–Flatt reads 2.2×
more "serial work" than the algorithm has. For the hybrid the laws see only
P × T and predict one value per worker count; the measurements do not (30.37×
at 8×8 against 22.52× at 64×1), because threads skip the gather and ranks pay
for it — the gap figure 7 exposes.

---

## 6. Discussion

**How does the actual speedup compare against the theoretical speedup?**
It tracks Amdahl to ~32 workers (93% at 8, 83% at 32) and falls to 58% at 64.
Karp–Flatt localises the divergence: *e* is flat while the curve tracks theory
and turns up exactly where it stops, so the loss is serial-side work plus a
fixed gather and sort up to 32 ranks, and growing communication beyond (§5.4).
Gustafson sits below Amdahl because its fraction comes from the parallel run;
Amdahl is the right model for this fixed-*n* experiment.

**Will more MPI processes always increase the speedup?**
No. Theoretically Amdahl caps *n* = 100M at 110.9× regardless of process
count. Empirically the cluster saturates rather than collapses — monotonic to
64 ranks, but 48 → 64 buys 2% for 33% more hardware, and efficiency falls
87% → 77% → 65% → 37% at 8/16/32/64. Extrapolating the Karp–Flatt trend, ranks
past ~80 would be negative. On a single host with more ranks than cores the
curve declines outright (§3.6).

**How does the workload distribution affect the speedup?**
By 23% at zero cost: `blockcyclic` 12.35× against `block` 10.04× at 16 ranks,
because trial-division cost grows with candidate magnitude and the rank holding
the top block gates the phase (imbalance 4.54×, predicted from a √k model). A
`dynamic` scheduler balances perfectly and still loses, because balance was not
the constraint and it spends a rank to fix it (§3.3).

**Will the speed-up results be the same across different machines?**
No. The Amdahl ceiling transfers — it is a property of the algorithm's phase
mix and *n* — but nothing empirical does. A CAAS core is ~4.7× slower than the
laptop the code was developed on (53.7 s vs 11.4 s serial at 100M), yet the
cluster reaches 23.9× where the laptop reached under 4×: absolute speed and
scalability are independent. Core count sets where the curve turns, memory
bandwidth sets how early efficiency decays, and the interconnect sets the
0.37 s gather floor; InfiniBand would narrow the gap to theory with no code
change.

**Would you recommend Open MPI for prime searching, against POSIX threads or
OpenMP?**
Yes, in hybrid form, for the case that needs it. Inside one node OpenMP is the
better tool (13.39× vs MPI's 12.26× at 16, and nothing to gather). But OpenMP
cannot leave the node, and the study's fastest result — 30.37× at 8×8 — is more
than twice the best any shared-memory program reaches here. Within MPI the rule
is to spend workers on threads, not ranks: every 64-worker split with ≥ 8
threads per rank beats every split with ≤ 2, by up to 35% (§4.3). The caveat is
that prime search is unusually hostile to message passing — 46 MB of result for
~2 s of parallel compute, all of which must cross a process boundary. A
workload returning a scalar would track Amdahl far more closely.

---

## 7. Conclusions

1. **Partitioning is the cheapest win, and the compliant choice is the fastest
   one.** `blockcyclic` beats `block` by 23%; its 1.04× imbalance against
   `block`'s 4.54× was predicted before it was measured.
2. **On a cluster, more ranks saturate rather than collapse.** Task 1 is
   monotonic to 64 ranks, but efficiency falls from 87% at 8 to 37% at 64.
3. **The shortfall against theory is the gather and the sort, not serial
   code.** Karp–Flatt *e* ≈ 0.020 is flat to 32 ranks against a true serial
   fraction of 0.0090; the residual is work the serial program never does.
4. **The hybrid is the fastest implementation, and wins by adding workers as
   threads.** 30.37× at 8×8, +35% over flat MPI at 64 workers, with *e* flat
   across the whole thread axis.
5. **Neither law is useful here without an overhead term.** Both model how
   work is divided; neither models how it is re-collected, which is where the
   lost speedup goes.

### 7.1 Limitations

- **Sub-second runs at low *n*.** 43 of 207 configurations finish under 1 s;
  headline figures are pinned to *n* = 100M for that reason, and individual
  low-*n* points should be read as trend, not as precise values.
- **No oversubscription on the cluster.** Worker counts stop at the 64-core
  allocation, so the "more workers than cores" regime is evidenced only by the
  earlier laptop run, which used the superseded `dynamic` scheme.
- **Gigabit interconnect.** The 0.37 s gather floor is a property of this
  cluster; the shortfall analysis in §5.4 would look different on InfiniBand.
- **Speedup is far from linear at 64 ranks** (37% efficiency). This is the
  gather and sort, not the partitioning, and the hybrid recovers much of it —
  but flat MPI on this problem does not scale linearly past one node.
- **The OpenMP baseline is not algorithmically identical.** It compacts an
  *n*-byte flag array with an O(n) serial sweep the other programs lack, so
  its serial fraction is overstated relative to the MPI programs.
- **The measured `imbalance` column** survives only in the raw CSVs; the
  analysis script does not propagate it to `summary.csv`.

### 7.2 Future work

- **Overlap the gather with the search** — `MPI_Igatherv`, or have each rank
  stream results as it produces them — to take the fixed 0.37 s off the
  critical path, the single largest term in the 64-rank shortfall.
- **Replace the root's `qsort` with a k-way merge** of the already-sorted
  per-rank runs (each rank's `t_lsort` output is ordered), turning an
  O(N log N) serial phase into O(N log P).
- **Per-rank node identifiers in the CSV** via `MPI_Get_processor_name`, so
  that the node-boundary result in §3.6 can be shown per rank rather than
  inferred from the SLURM allocation.
- **Larger *n* and more nodes**, to test whether the Karp–Flatt upturn at
  48+ ranks moves right as the compute phase grows, as the fixed-cost model
  predicts.
- A sieve would make the program faster but the experiment worse; it is
  deliberately out of scope for a speedup study.

---

## 8. Appendices

### Appendix A — AI declaration

⟨Required. Declare Generative-AI use during the preparation period and attach
the prompt records as PDF, or state that none was used. AI tools were not used
during the presentation, oral or coding interview sessions.⟩

### Appendix B — Reproducing the results

On the CAAS cluster (every number in this report):

```sh
bench/caas.sh push && bench/caas.sh build
bench/caas.sh submit smoke        # wait for SMOKE PASS
bench/caas.sh submit baselines    # serial, pthreads, OpenMP
bench/caas.sh submit mpi          # Task 1
bench/caas.sh submit hybrid       # Task 2
bench/caas.sh pull && bench/caas.sh analyse   # -> bench/analysis-caas/
```

Clear stale CSVs out of `bench/results-caas/` before re-analysing — the
analysis globs the directory and would pool runs at different problem sizes.
`defq`'s wall limit is 20 minutes; a job asking for more waits forever rather
than failing. On a single host via Docker, `bench/sweep.sh`, `sweep-mpi.sh` and
`sweep-hybrid.sh` followed by `bench/analyse.py` and `bench/report.py` do the
same. `bench/analysis-caas/summary.csv` holds every reduced quantity used here;
the `imbalance` column is in the raw `mpi-*.csv` files.

### Appendix C — Figure index

| Figure | Shows | Requirement |
|---|---|---|
| `fig1.svg` | runtime against *n* | (a) 1 |
| `fig2.svg` | speedup against *n* | (a) 2 |
| `fig3.svg` | speedup against worker count | (a) 3, (b) 2 |
| `fig4.svg` | parallel efficiency | supporting |
| `fig5.svg` | partitioning scheme comparison | supporting |
| `fig6.svg` | Task 1 measured vs theoretical | (c) 1 |
| `fig7.svg` | Task 2 measured vs theoretical | (c) 2 |
| `fig8.svg` | threads per rank at fixed process count | (b) 1 |

All in `bench/analysis-caas/figures/`.

### Appendix D — Source files

| File | Contents |
|---|---|
| `lab-2/task1.c` | Task 1 — Open MPI, four partitioning schemes |
| `lab-2/task2.c` | Task 2 — hybrid MPI + OpenMP |
| `lab-2/task3-performance-evaluation.md` | Task 3 — full derivations and experimental design |
| `week-4-lab-1/task1.c` | serial baseline (speedup denominator) |
| `week-4-lab-1/task2.c` | POSIX threads baseline |
| `week-4-lab-1/task3.c` | OpenMP baseline |
