# Prime Search with the Message Passing Interface

**FIT3143 Parallel Computing — Lab #2 Report (Task 4)**

⟨Name 1⟩, ⟨Student ID⟩, ⟨email⟩@student.monash.edu
⟨Name 2⟩, ⟨Student ID⟩, ⟨email⟩@student.monash.edu

---

## Contents

1. [Introduction](#1-introduction)
2. [Experimental setup and measurement design](#2-experimental-setup-and-measurement-design)
3. [Task 1 — Prime search with Open MPI](#3-task-1--prime-search-with-open-mpi)
4. [Task 2 — Hybrid Open MPI + OpenMP](#4-task-2--hybrid-open-mpi--openmp)
5. [Task 3 — Performance evaluation with Amdahl's and Gustafson's Laws](#5-task-3--performance-evaluation-with-amdahls-and-gustafsons-laws)
6. [Discussion](#6-discussion)
7. [Conclusions](#7-conclusions)
8. [Appendices](#8-appendices)

---

## 1. Introduction

This report documents a distributed implementation of the prime search problem
introduced in Week 4 (Lab #1) and extends it in two directions: a pure
message-passing version using Open MPI (Task 1), and a hybrid version combining
message passing across processes with shared-memory threading inside each
process (Task 2). It then evaluates both empirically and against the
theoretical bounds given by Amdahl's and Gustafson's Laws (Task 3).

The problem is to find every prime strictly less than an integer *n* supplied
on the command line, and to write the complete result to a text file in
ascending order. The root process reads *n*, disseminates it to every other
process with `MPI_Bcast`, each process — the root included — computes a share
of the candidate range, and the root gathers, orders and writes the result.

**The single most important property of this study is that the trial-division
kernel is byte-identical across all five implementations** — the Week 4 serial,
POSIX threads and OpenMP versions, and the two implementations documented here.
Speedup is defined as T_serial / T_parallel, so holding the per-candidate work
constant is what makes the comparison measure parallelism rather than a change
of algorithm. No sieve is used anywhere; a faster algorithm would produce better
runtimes and a *worse* experiment.

All results in this report were measured on the Monash CAAS cluster across four
compute nodes. An earlier single-host run on a laptop is referred to in two
places, each time labelled as such, and only for what it shows qualitatively.

---

## 2. Experimental setup and measurement design

### 2.1 Platform

| Property | Value |
|---|---|
| Cluster | Monash CAAS, partition `defq` |
| Nodes | 4 × AMD EPYC 7763, 16 cores per node — **64 cores total** |
| Interconnect | Gigabit ethernet (not InfiniBand) |
| Toolchain | Open MPI 4.1.5, GCC 11.2, `-O2 -fopenmp` |
| Launcher | `srun` under SLURM; hybrid runs use `srun --cpu-bind=none` |
| Problem sizes | 30 distinct values of *n*, from 20,000,000 to 100,000,000, spaced geometrically |
| Repetitions | 3 per configuration |
| Reduction | **Median** across repetitions |
| Baseline | Week 4 Task 1 serial program, re-measured on the same nodes, at the same *n* |

The serial reference runs 5.49 s at *n* = 20M and 53.72 s at *n* = 100M. A
single CAAS core is roughly 4.7× slower than the laptop core the code was
developed on; that ratio is returned to in §6.

Worker counts up to 16 fit on one node; every configuration above 16 workers is
genuinely distributed across separate physical machines and pays for the
interconnect. Worker counts stop at exactly 64, the size of the allocation, so
nothing on the cluster is oversubscribed.

**Partitioning scheme.** Every headline figure uses `blockcyclic` with a chunk
of 4096 candidates, the default in both programs. The alternative schemes are
measured in §3.2. The master–worker `dynamic` scheme appears there only: it
reserves rank 0 as a pure dispatcher, which both costs a worker and violates the
specification's requirement that every process, *including the root*, take a
share of the work.

**Sub-second runs.** The specification warns against reporting runtimes under a
second. Of 207 measured configurations, 43 fall below it — all high-worker-count
runs at *n* ≤ 54M, the quickest being the hybrid at 4×16 finishing *n* = 20M in
0.33 s. The effect is visible: across the lowest eight values of *n* the
hybrid's speedup scatters by 14% between adjacent points. **Every headline
figure in this report is therefore quoted at *n* = 100M**, where the fastest
configuration still takes 1.76 s and the ranking between implementations is
stable. The low-*n* points are retained because the specification asks for 30
values and the *trend* across them is sound.

### 2.2 Why the median, and why `MPI_MAX`

Two measurement decisions shape every number in this report.

**Repetitions are reduced by median, not mean.** The CAAS handout notes that its
nodes may be shared with research workloads. A single perturbed repetition
displaces the mean of three samples substantially while leaving the median
untouched.

**`t_compute` is reduced across ranks with `MPI_MAX`, not averaged.** The
wall-clock cost of a parallel phase is determined by when its *slowest* rank
finishes, not by the average rank. Averaging would conceal load imbalance —
which is precisely the quantity the partitioning experiments in §3.2 exist to
measure. `MPI_MIN` is reduced alongside it and the ratio between them recorded
as an explicit `imbalance` column.

### 2.3 Phase-resolved timing

Both parallel programs are instrumented so that the phases partition the whole
run. This is what makes the theoretical analysis in §5 possible: the serial and
parallel fractions are *measured*, not assumed.

| Phase | Meaning | Classification |
|---|---|---|
| `t_alloc` | buffer allocation | serial |
| `t_bcast` | `MPI_Bcast` of *n* to every rank | serial |
| `t_compute` | the distributed prime search | **parallel** |
| `t_lsort` | each rank ordering its own results (Task 2) | **parallel** |
| `t_comm` | `MPI_Gather` + `MPI_Gatherv` of results | serial |
| `t_sort` | root ordering the gathered result | serial |
| `t_io` | root writing the output file | serial |
| `t_total` | `MPI_Init` to just before `MPI_Finalize` | — |

Two classifications deserve justification. `t_lsort` is counted as **parallel**
work because every rank executes it simultaneously on disjoint data; charging it
as serial would inflate the serial fraction for the hybrid alone and make the
two implementations incomparable. A barrier precedes the compute timer so that
`t_compute` measures the phase itself rather than accumulated rank skew.

The serial baseline is instrumented with the corresponding smaller set —
`t_alloc`, `t_compute`, `t_merge`, `t_io` — since it has nothing to broadcast
and nothing to gather. Section 5.2 depends on this.

---

## 3. Task 1 — Prime search with Open MPI

### 3.1 Partitioning schemes

Trial division tests divisors up to √k, so **the cost of a candidate grows with
its value.** A scheme that hands each rank an equal-sized contiguous range
therefore hands the high-numbered ranks more work than the low-numbered ones.
Load balance is consequently not a matter of dividing the range evenly — it is a
matter of dividing the *cost* evenly, and those are different problems.

Four schemes were implemented so that the trade-off could be measured rather
than asserted. All are selectable at runtime via `--scheme`.

| Scheme | Allocation | Balance | Sort at root |
|---|---|---|---|
| `block` | rank *r* takes one contiguous range | poor — high ranks do more | **not needed** |
| `cyclic` | rank *r* takes k = 2+r, 2+r+P, 2+r+2P, … | very good | required |
| `blockcyclic` | chunks of `--chunk` candidates dealt round-robin | very good | required |
| `dynamic` | master–worker; rank 0 issues chunks on request | best in principle | required |

`block` is the only scheme requiring no global sort: each rank owns one
contiguous ascending range, so the gathered results concatenate in order.
Interleaving schemes scatter each rank's primes across [2, *n*), so the root must
sort after gathering.

**Sorting is timed separately and charged only to the schemes that actually
perform it.** Charging `block` for a sort it never runs would misrepresent its
true total cost and make the comparison meaningless.

`dynamic` differs from the other three in kind, not degree: rank 0 dispatches
chunks and computes nothing. That is a design the specification rules out —
"each process (including the root process)" must search — so it is retained
here as a comparison point and as evidence that the alternative was explored,
not as a candidate default.

### 3.2 Which scheme performs best

Measured at *n* = 100M with 16 ranks (one full node), so that the partitioning
scheme is the only variable. The imbalance column is the measured
`MPI_MAX / MPI_MIN` ratio of `t_compute` across ranks — 4.54× means the slowest
rank spent 4.54 times as long computing as the fastest.

![Partitioning scheme comparison](../bench/analysis-caas/figures/fig5.svg)

| Scheme | Speedup vs serial | Measured imbalance | Karp–Flatt *e* |
|---|---|---|---|
| **`blockcyclic`** | **12.35×** | 1.04× | 0.020 |
| `cyclic` | 12.34× | 1.01× | 0.020 |
| `dynamic` | 11.54× | **1.00×** | 0.026 |
| `block` | 10.04× | **4.54×** | 0.040 |

`blockcyclic` and `cyclic` are statistically indistinguishable, and both beat
`block` by 23% on identical hardware running identical arithmetic. This is the
cheapest performance available anywhere in the study — it costs no hardware and
no algorithmic change, only a different indexing rule.

**The `block` result was predicted before it was measured.** Modelling each
chunk's cost as proportional to √k and summing per rank gives an expected
imbalance of 3.4× (cost ∝ √k⁄ln k) to 4.1× (cost ∝ √k) for `block`, against
1.0012× for `blockcyclic`. Measured: 4.54× and 1.04×. The model slightly
understates `block` — it ignores cache behaviour and the fact that the highest
range also stores the most results — but it is the right order, and it predicts
`blockcyclic`'s near-perfect balance almost exactly. Interleaving balances any
smooth monotonic cost function, and √k is smooth and monotonic. The imbalance
maps directly onto Karp–Flatt: `block`'s *e* is double `blockcyclic`'s.

`blockcyclic` is preferred over `cyclic` on grounds the measurement cannot
resolve at this scale: a rank walks 4096 contiguous candidates before moving to
its next chunk, preserving the cache locality that a stride-P interleave loses.

`dynamic` is the most informative row of the four. Its imbalance is **1.00×** —
perfect, better than every static scheme, exactly as a master–worker scheduler
promises. And it is still 6.6% slower than `blockcyclic`. That rules out tuning
as the explanation: the scheme does precisely what it is designed to do and
loses anyway, because balance was never the binding constraint. There was only
4% of imbalance left to recover, and it spends a whole rank (15 of 16 compute,
predicting a 1/16 ≈ 6.3% loss — almost exactly the measured gap) plus per-chunk
request traffic to recover it.

### 3.3 Runtime against problem size *(required graph a-1)*

![Runtime against n](../bench/analysis-caas/figures/fig1.svg)

Log–log axes, 30 values of *n* from 20M to 100M. Task 1 runs at 16 ranks (one
node), the shared-memory baselines at 16 threads, and the hybrid at 4 ranks × 16
threads (four nodes). Straight lines on log–log axes indicate that runtime
follows a clean power law in *n*, as expected for trial division. The parallel
lines sit below serial by a near-constant vertical offset, which is what a
roughly constant speedup looks like in log space.

At *n* = 100,000,000:

| Implementation | Configuration | Runtime | |
|---|---|---|---|
| Serial (Week 4 Task 1) | 1 core | 53.64 s | baseline |
| **Hybrid (Task 2)** | 4×16, four nodes | **1.756 s** | fastest |
| OpenMP | 1×16 | 3.977 s | fastest single-node |
| **MPI (Task 1)** | 16×1 | 4.366 s | |
| POSIX threads | 1×16 | 5.192 s | |

### 3.4 Empirical speedup against problem size *(required graph a-2)*

![Speedup against n](../bench/analysis-caas/figures/fig2.svg)

| *n* | serial | **MPI 16×1** | vs its Amdahl | Hybrid 4×16 | vs its Amdahl | OpenMP 1×16 | Pthreads 1×16 |
|---|---|---|---|---|---|---|---|
| 20,000,000 | 5.49 s | 9.40× | 77% | 16.85× | 62% | 11.52× | 9.34× |
| 29,494,000 | 9.52 s | 10.33× | 84% | 18.79× | 66% | 11.46× | 9.06× |
| 43,497,000 | 16.48 s | 11.04× | 84% | 21.04× | 63% | 12.39× | 9.71× |
| 67,808,000 | 30.93 s | 12.16× | 89% | 24.41× | 65% | 12.45× | 9.93× |
| 100,000,000 | 53.72 s | **12.30×** | **87%** | **30.59×** | **75%** | 13.51× | 10.35× |

**Speedup improves as *n* grows for every implementation, and the two
MPI-based ones gain most**: Task 1 rises 9.40× → 12.30× (+31%) and Task 2
16.85× → 30.59× (+82%), against OpenMP's +17% and pthreads' +11%.

Two mechanisms compound. The first is the falling serial fraction demonstrated
in §5.2: the Amdahl ceiling itself rises with *n*. The second is specific to
message passing — the gather is a fixed cost set by π(n), while the parallel
part grows as n^1.5 / log n, so communication shrinks as a *proportion* of the
run. That is what the "vs its Amdahl" columns show: Task 1 closes from 77% of
its bound to 87%, and Task 2 from 62% to 75%. The shared-memory baselines have
no such term and do not close. Message passing is penalised at small *n* and
rewarded at large *n*.

### 3.5 Empirical speedup against process count *(required graph a-3)*

Measured at *n* = 100M, one rank per core, from 1 rank to the full 64-core
allocation. Ranks 1–16 sit on one node; 24 upward span two or more.

![Speedup against worker count](../bench/analysis-caas/figures/fig3.svg)

| Ranks | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Speedup** | 1.00× | 1.96× | 3.76× | 7.00× | 9.89× | 12.26× | 16.82× | 20.88× | 23.34× | **23.87×** |
| Efficiency | 100% | 98% | 94% | 87% | 82% | 77% | 70% | 65% | 49% | 37% |

Three features stand out.

**Scaling is near-linear on one node.** 98% efficiency at 2 ranks, 87% at 8,
77% at 16. At the small end there is no anomaly: two ranks give 1.96×, because
both of them compute.

**There is no discontinuity at the node boundary.** W = 16 fits one node and
W = 24 spans two, yet efficiency falls smoothly through the crossing (77% → 70%).
The interconnect does not announce itself with a step when the first message
leaves the node; it shows up as a trend once the compute phase has shrunk
enough for a fixed-cost gather to matter — which §5.5 quantifies.

**The curve saturates rather than collapses.** 32 → 48 ranks buys 12% more
speedup; 48 → 64 buys 2%. The marginal rank is still positive at 64 but nearly
worthless: the last 16 ranks return 2% more speedup for 33% more hardware.

For comparison, an earlier run of the same code on a single 10-core laptop
(using the superseded `dynamic` scheme, so its absolute numbers are not quoted
here) showed the MPI curve *peaking and then declining* once the rank count
exceeded the physical cores. Both effects are real; they are different regimes.
With more ranks than cores, each extra rank adds context switching to a fixed
amount of compute and actively hurts. With ranks spread across genuinely
separate nodes, they keep helping, just by less and less.

---

## 4. Task 2 — Hybrid Open MPI + OpenMP

### 4.1 Two-level work decomposition

The hybrid exposes two levels of parallelism: distributed memory across MPI
ranks, which do not share an address space and must exchange results through
messages, and shared memory within each rank, where OpenMP threads cooperate
over that rank's candidates without any copying.

The design decision that keeps the two levels from interfering is that **a chunk
is the unit of work at both levels**:

```
MPI level      decides which chunks a rank owns    (--scheme: block | blockcyclic | dynamic)
OpenMP level   distributes that rank's chunks       #pragma omp for schedule(dynamic)
               across its threads
```

Every MPI process, the root included, creates its OpenMP team and searches its
own chunks; *n* is broadcast to every rank and is visible to every thread
through the rank's address space.

The pure `cyclic` scheme from Task 1 is deliberately absent from the hybrid: it
strides per candidate rather than per chunk, so it offers no natural chunk
decomposition for the OpenMP level to distribute. Under `dynamic`, rank 0 hands
out *batches* of chunks — one batch being `threads × chunk` candidates — so
that each worker rank always receives enough work to occupy all of its threads.
The cost is that rank 0's entire OpenMP team idles with it: at 4 ranks × 16
threads that is 16 of 64 workers, a quarter of the machine, and the measured
loss (30.33× for `blockcyclic` against 21.78× for `dynamic`, −28%) is close to
the 25% the lost workers alone predict.

**Result ordering.** Threads within a rank pull chunks dynamically, so their
private buffers hold scattered values and each rank must sort its own results
before gathering. That local sort runs concurrently on every rank and is
therefore classified as parallel work (§2.3). After the gather, a global sort at
the root is required only when the MPI-level scheme interleaves chunks between
ranks; under `block` the ranks' already-sorted runs concatenate in ascending
order and the root sorts nothing.

**A measurement artifact worth documenting.** Initial hybrid results showed it
as by far the slowest implementation. The cause was the launcher's default core
binding, which pins each rank to a single core — leaving the OpenMP threads
inside that rank with nowhere to run. Every hybrid measurement therefore
disables binding: `mpirun --bind-to none` in the Docker development environment
and `srun --cpu-bind=none` on the cluster. All results collected before this
was identified were discarded and the sweep re-run.

### 4.2 Threads per rank at a fixed process count *(required graph b-1)*

With MPI ranks fixed at 4 and OpenMP threads varied inside them, *n* = 100M.
Task 1 has no threads, so it appears as a flat reference at its 4-process
speedup of 3.76×. Every additional worker here arrives as a *thread*, never as a
process.

![Threads per rank at fixed process count](../bench/analysis-caas/figures/fig8.svg)

| Threads per rank | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Total workers | 4 | 8 | 16 | 32 | 64 |
| **Hybrid speedup** | 3.79× | 7.06× | 12.64× | 20.98× | **30.01×** |
| Task 1 at 4×1 | 3.76× | 3.76× | 3.76× | 3.76× | 3.76× |
| Efficiency | 95% | 88% | 79% | 66% | 47% |
| Karp–Flatt *e* | 0.0186 | 0.0190 | 0.0177 | 0.0169 | 0.0180 |

**Adding threads inside four ranks scales almost as well as adding ranks did,
and with no accumulating overhead.** Karp–Flatt is flat across the entire
thread axis — 0.0169 to 0.0190, with no trend. Set that against the rank axis
in §3.5, where *e* held near 0.020 to 32 ranks and then climbed to 0.0225 at 48
and 0.0267 at 64. Adding threads does not accumulate overhead; adding ranks
does.

The mechanism is direct. A thread shares its rank's address space, so a worker
added as a thread contributes *nothing* to the gather volume or to the message
count at the root. A worker added as a rank contributes to both. Scaling by
threads is therefore free of the term that bends the MPI curve — which is the
whole case for the hybrid.

The same pattern holds at other rank counts, which rules out P = 4 being a
lucky choice:

| Fixed ranks | Threads | Speedup range | *e* range |
|---|---|---|---|
| 4 | 1 → 16 | 3.79× → 30.01× | 0.0169–0.0190 |
| 8 | 2 → 8 | 12.78× → 30.37× | 0.0168–0.0189 |
| 16 | 1 → 4 | 12.66× → 29.70× | 0.0176–0.0183 |

### 4.3 Hybrid against shared-memory at matched worker counts *(required graph b-2)*

Comparing the hybrid against the shared-memory implementations with the total
worker count matched: a hybrid configuration of P ranks × T threads is compared
against OpenMP and pthreads running P × T threads. The shared-memory programs
cannot leave a node, so the comparison against them ends at 16 workers.

![Speedup against worker count](../bench/analysis-caas/figures/fig3.svg)

| Total workers | 4 | 8 | 16 |
|---|---|---|---|
| OpenMP | 3.82× | 7.31× | 13.39× |
| pthreads | 2.88× | 5.45× | 10.21× |
| **Hybrid** (best split) | 3.79× (4×1) | 7.06× (4×2) | 12.78× (8×2) |
| MPI (Task 1) | 3.76× | 7.00× | 12.26× |

**Within one node the hybrid sits within 5% of OpenMP** — 12.78× against 13.39×
at 16 workers — and beats pthreads throughout. OpenMP's `schedule(dynamic)`
balances the √k cost gradient with no gather at all; pthreads' static block
partitioning suffers the same imbalance §3.2 measured for `block`.

Beyond 16 workers only the two MPI-based implementations can continue, so the
matched comparison becomes hybrid against flat MPI. The honest control is the
hybrid program itself run with one thread per rank, which isolates the
threading level by holding the program constant:

| Total workers | Best hybrid split | Speedup | Same program at P×1 | Advantage |
|---|---|---|---|---|
| 16 | 8×2 | 12.78× | 12.66× | +1% |
| 32 | 2×16 | 21.11× | 18.65× | +13% |
| 64 | 8×8 | **30.37×** | 22.52× | **+35%** |

At 64 workers the full ordering is 8×8 (30.37×), 4×16 (30.01×), 16×4 (29.70×),
32×2 (24.78×), 64×1 (22.52×). **Every configuration with 8 or more threads per
rank beats every configuration with 2 or fewer.** At 16 workers there is little
gather to avoid and the splits are indistinguishable; at 64 avoiding it is worth
a third of the runtime. The three routes to 64 workers that keep the rank count
low — 4×16, 8×8, 16×4 — land within 2% of each other: once ranks are not the
constraint, *how* the remaining workers are split barely matters, only that
they are not spent on ranks.

One number needs reconciling. The hybrid at 64×1 reads 22.52× while Task 1 at
64 ranks reads 23.87×. The ~6% gap is the standing cost of the hybrid structure
— OpenMP team setup and the per-rank `t_lsort` phase — paid even when the team
has a single thread. It is the figure Task 1 buys by not having a second level,
and the figure the hybrid recovers many times over once the second level is
used.

The conclusion is that **the hybrid is the fastest implementation in the study,
and it wins by spending workers on threads rather than ranks.** On a single node
it matches OpenMP; across nodes it is the only implementation that can go, and
the split that minimises rank count is the one to choose.

---

## 5. Task 3 — Performance evaluation with Amdahl's and Gustafson's Laws

### 5.1 The two laws require two different experiments

The specification warns that the two laws "have different measurement
assumptions, and you will need to design measurement experiments to compute the
serial/parallel fractions correctly." This is the crux of the task, and the two
laws cannot be fed the same measured fraction.

| | **Amdahl** | **Gustafson** |
|---|---|---|
| Held fixed | problem size *n* | execution time |
| Question | how much faster is one fixed workload? | how much more work fits in the same time? |
| Formula | S(W) = 1 / (s + p/W) | S(W) = s + p·W |
| *s* is the serial share **of what** | the **one-worker** execution | the **parallel** execution as run |
| Therefore measured on | serial `task1` at that *n* | the P×T run being scored |

**Why supplying Amdahl with a parallel run's fraction is wrong.** As workers are
added, the parallel part shrinks while the serial part does not, so the measured
serial share of a parallel run *rises* with W — in our data from 0.015 at one
worker to 0.42 at sixty-four. Substituting that into Amdahl's formula makes the
predicted ceiling sag as W grows, and a ceiling that moves with the quantity it
is supposed to bound is not a bound at all. This is a circular measurement, and
avoiding it requires running two separate experiments.

### 5.2 Experiment A — Amdahl's fraction, from the serial program

Amdahl's *s* is a property of the workload, so it is measured on the serial
program at the same *n* — the one-worker execution whose time the law is
dividing:

```
s = median(t_alloc + t_merge + t_io) / [ that + median(t_compute) ]
p = 1 − s
```

| *n* | *s* | *p* | Ceiling S(∞) = 1/s |
|---|---|---|---|
| 20,000,000 | 0.0212 | 0.9788 | 47.1× |
| 45,979,000 | 0.0149 | 0.9851 | 67.0× |
| 100,000,000 | 0.0090 | 0.9910 | **110.9×** |

**The serial fraction falls as *n* grows, so the Amdahl ceiling rises.** Trial
division costs approximately O(n^1.5 / log n) while allocation, ordering and
file I/O scale with π(n) ≈ n / log n. The parallel part outgrows the serial part.

This is the answer to the "increasing problem size" axis of the theoretical
analysis, and it is worth stating precisely: **a larger *n* does not make the
program faster — it raises the limit on how much faster parallelism could ever
make it.** Because this fraction depends only on *n*, the Amdahl curve is
identical for the MPI and hybrid implementations, correctly so, since both
parallelise the same phase of the same algorithm.

### 5.3 Experiment B — Gustafson's fraction, from each parallel run

Gustafson's *s* is by construction the serial share of the parallel execution —
the scaled workload as actually executed — so it is measured on the run being
scored, using the phase decomposition of §2.3, at each (*n*, P, T).

Unlike Amdahl's, this fraction rises with worker count (0.015 → 0.42 from 1 to
64 workers), which under Gustafson's assumptions is expected rather than
pathological: the law asks how much more work a larger machine completes in the
same time, so the growing coordination share is part of what is being described.

### 5.4 The Karp–Flatt cross-check

Neither law charges anything for communication, so both over-predict. To
identify *which* of the two possible causes explains the shortfall, Amdahl's
formula is inverted on the **measured** speedup to recover the experimentally
determined serial fraction:

```
e = (1/S − 1/W) / (1 − 1/W)          undefined at W = 1
```

If *e* is flat in W, the gap is genuine irreducible serial work. If *e* climbs
with W, the loss is **parallel overhead** — broadcast, gather, memory bandwidth
— which neither law models. If *e* *falls* with W, the loss at small W was load
imbalance.

At *n* = 100M, where the true serial fraction is 0.0090:

| Workers | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| **MPI *e*** | 0.0229 | 0.0211 | 0.0205 | 0.0203 | 0.0172 | 0.0225 | **0.0267** |
| Hybrid *e* (best split) | — | 0.0186 | 0.0190 | 0.0168 | 0.0167 | — | 0.0176 |
| OpenMP *e* | 0.018 | 0.016 | 0.014 | 0.013 | — | — | — |
| pthreads *e* | **0.275** | 0.130 | 0.067 | 0.038 | — | — | — |

Each row has a different signature.

**MPI is flat at *e* ≈ 0.018–0.023 from 2 through 32 ranks, then rises at 48
and 64.** This is the central diagnostic result. A flat *e* means the shortfall
against Amdahl is genuine serial-side work, not overhead that grows with the
machine — the program scales as well as its serial fraction allows. The rise at
W ≥ 48 is where communication finally begins to cost more than it saves. Note
that even the flat value sits about 2.2× above the true serial fraction: that
residual is the gather and the root's sort, work the serial program never does,
which the model can only absorb into the one term it owns.

**The hybrid is flat everywhere**, at or slightly below MPI's plateau, and does
not rise at 64 — because its best splits reach 64 workers with only 4–16
ranks, so the message count and gather at the root never grow past what
16 ranks would cost.

**OpenMP's *e* is the lowest in the study** (0.013 at 16 threads): nothing is
gathered, and `schedule(dynamic)` balances the √k cost gradient. One caveat in
its disfavour: the Week 4 OpenMP program compacts an *n*-byte flag array with an
O(n) serial sweep, where every other program appends primes directly, so it
carries serial work the MPI versions never do — and still records the lowest
*e*. The comparison is conservative rather than flattering.

**pthreads' *e* starts at 0.275 and *falls* with W** — the signature of load
imbalance rather than overhead. Its static block partitioning gives the thread
holding the high-numbered candidates far more work, and that penalty is
proportionally worst when there are only two blocks. It is the shared-memory
counterpart of §3.2's `block` result.

### 5.5 Measured against theoretical speedup *(required graphs c-1 and c-2)*

**Task 1 (Open MPI), *n* = 100M, Amdahl s = 0.0090, ceiling 110.9×:**

![Task 1 measured vs theoretical](../bench/analysis-caas/figures/fig6.svg)

| Ranks | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| **Measured** | 1.00× | 1.96× | 3.76× | 7.00× | 12.26× | 20.88× | 23.34× | 23.87× |
| Amdahl | 1.00× | 1.98× | 3.89× | 7.53× | 14.09× | 25.01× | 33.72× | 40.82× |
| Gustafson | 1.00× | 1.97× | 3.81× | 7.21× | 12.54× | 21.33× | 27.99× | 37.49× |

**Task 2 (hybrid MPI + OpenMP), same *n*, ranks fixed at 4, threads increasing:**

![Task 2 measured vs theoretical](../bench/analysis-caas/figures/fig7.svg)

| Workers (4 × T) | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| **Measured** | 3.79× | 7.06× | 12.64× | 20.98× | 30.01× |
| Amdahl | 3.89× | 7.53× | 14.09× | 25.01× | 40.82× |
| Gustafson | 3.84× | 7.29× | 13.47× | 22.67× | 36.75× |

Three observations follow.

**Measurement tracks Amdahl closely to about 32 workers and diverges after.**
Task 1 achieves 93% of its Amdahl prediction at 8 ranks, 83% at 32 and 58% at
64. The Amdahl column is identical for both tasks at equal worker count,
correctly — it depends only on *n* — and the hybrid comes closer to it at every
W (74% at 64 against Task 1's 58%) because it reaches the same worker count
with a quarter of the ranks.

**Gustafson sits below Amdahl at every worker count** — 37.49× against 40.82×
at 64 — with the gap widening to 48 and narrowing thereafter. This is not a
contradiction but a direct consequence of the two experiments. Gustafson's *s*
is taken from the parallel run, where it grows with W (0.015 → 0.42), so
s + p·W is pulled down exactly as the machine grows; Amdahl's *s* stays fixed
at 0.0090. For this fixed-*n* experiment **Amdahl is the appropriate model** —
the workload does not grow with the machine — and Gustafson is reported to show
what the same code would promise under a weak-scaling regime, where the
comparison would invert.

**The shortfall at 64 ranks has three identifiable parts, none of which exist
in the program Amdahl's *s* was measured on.** Against the 53.72 s serial
reference, Amdahl's 40.82× is 1.32 s and the measured 23.87× is 2.25 s — a
shortfall of 0.93 s.

1. *The gather is a fixed cost that does not shrink with W.* Every prime must
   reach rank 0 exactly once, so the volume is set by π(n): 5,761,455 primes ×
   8 bytes = 46.1 MB, which at 1 Gb/s is a 0.37 s floor before latency — about
   40% of the shortfall on its own. Compute per rank falls as 1/W; this does
   not fall at all, so its *share* of the run grows with every rank added.
2. *The root's global sort is a serial phase the baseline does not have.*
   `blockcyclic` interleaves chunks, so the root must `qsort` 5.76 million
   longs. The serial program appends in ascending order by construction — its
   `t_merge` is measured as exactly zero — so this phase is structurally
   invisible to Amdahl's *s*.
3. *Message latency scales with rank count, not volume.* `MPI_Gatherv` at 64
   ranks is 64 transfers to one receiver, and the per-message cost is paid 64
   times regardless of how little each rank contributes.

This is why neither law brackets the measurement: both model how work is
*divided*, neither models how it is *re-collected*. It is also exactly what
Karp–Flatt detects — *e* ≈ 0.020 against a true serial fraction of 0.0090 is
the model reporting 2.2× more "serial work" than the algorithm has, because it
is absorbing the gather and the sort into the only term it owns.

For the hybrid, both laws see only the *product* P × T and therefore predict a
single value for every split with the same worker count. The measurements do
not: at 64 workers, 8×8 gives 30.37× while 64×1 gives 22.52×, because threads
inside a rank skip the gather entirely while ranks pay for it. That modelling
gap is precisely what figure 7 exposes.

---

## 6. Discussion

**How does the actual speedup compare against the theoretical speedup?**
It tracks Amdahl closely to about 32 workers and diverges after: at 8 ranks MPI
achieves 7.00× against a prediction of 7.53× (93%), at 32 ranks 20.88× against
25.01× (83%), and at 64 ranks 23.87× against 40.82× (58%). Karp–Flatt localises
the divergence — *e* is flat at ≈0.020 while the curve tracks theory and turns
upward exactly where it stops. So the loss is irreducible serial work plus a
fixed gather and sort up to 32 ranks, and growing communication overhead
beyond. Gustafson sits below Amdahl throughout because its fraction is taken
from the parallel run, where the serial share grows with the machine; for this
fixed-*n* experiment Amdahl is the appropriate model.

**Will more MPI processes always increase the speedup?**
No, though on this hardware the failure is saturation rather than collapse.
Theoretically Amdahl saturates: at *n* = 100M the ceiling is 110.9× no matter
how many processes are added. Empirically the curve is monotonic to 64 ranks
but flattens hard — 32 → 48 ranks buys 12% more speedup, 48 → 64 buys 2%.
Efficiency tells the story more plainly: 87% at 8 ranks, 77% at 16, 65% at 32,
37% at 64. The marginal rank is still positive but nearly worthless, and
extrapolating the Karp–Flatt trend, ranks past roughly 80 would make it
negative. On the earlier single-host laptop run the curve *did* decline once
ranks exceeded physical cores — a different regime, where extra ranks add
context switching to a fixed amount of compute rather than recruiting new
hardware.

**How does the workload distribution affect the speedup?**
Substantially, and it is the one variable that costs nothing to change. At
*n* = 100M with 16 ranks, `blockcyclic` (12.35×, imbalance 1.04×) beats `block`
(10.04×, imbalance 4.54×) by 23% on identical hardware running identical
arithmetic. `block` loses because trial-division cost grows with candidate
magnitude, so the rank holding the highest range gates the whole phase — an
effect predicted at 3.4–4.1× from a √k cost model before it was measured at
4.54×. `blockcyclic` interleaves chunks so that every rank draws a mixture of
cheap and expensive candidates, while chunking preserves the cache locality
that pure `cyclic` sacrifices. And `dynamic`, which balances *perfectly* at
1.00×, is still slower than `blockcyclic`: balance was never the binding
constraint, and the scheme spends a whole rank to recover the 4% that was left.

**Will the speed-up results be the same across different machines?**
No, and there is direct evidence. The Amdahl ceiling is a property of the
program and *n* and transfers — s = 0.0090 at 100M reflects the algorithm's
phase mix, not the hardware. Everything empirical is a property of the host. A
single CAAS core is roughly 4.7× slower than the laptop core the code was
developed on (53.7 s against 11.4 s for the serial run at *n* = 100M), yet the
cluster reaches 23.9× on Task 1 where the laptop reached under 4× — absolute
speed and scalability are independent. Core count sets where the curve turns
over, memory bandwidth per core sets how early efficiency decays, and the
interconnect sets the gather cost that dominates the Karp–Flatt figures past 32
ranks. A machine with InfiniBand rather than Gigabit ethernet would lower the
0.37 s gather floor and narrow the gap to theory with no code change at all.

**Would you recommend Open MPI for prime searching, against POSIX threads or
OpenMP?**
Yes — for the case where it is needed, and in hybrid form. Within one node,
OpenMP is the better tool: 13.39× at 16 threads against MPI's 12.26×, simpler,
and with nothing to gather. But OpenMP cannot leave the node, and that is where
the study's fastest result comes from: the hybrid at 8 ranks × 8 threads
reaches 30.37×, more than twice the best any shared-memory implementation can
achieve on this hardware. The recommendation within the MPI family is equally
clear — spend workers on threads, not ranks. Every 64-worker configuration with
8 or more threads per rank beats every configuration with 2 or fewer, by up to
35%, because a thread contributes nothing to the gather and a rank contributes
to everything.

The caveat is specific to this problem. Prime search has an unusually high
result-volume to compute-volume ratio — 46 MB of output for a search whose
parallel part takes about two seconds at this scale — and every one of those
bytes must cross a process boundary to be gathered and sorted at the root. A
workload that returned a single scalar would pay none of that and would track
Amdahl far more closely on the same hardware. Message passing is the right tool
for this problem across nodes; it is a tool that this problem is unusually
good at punishing.

---

## 7. Conclusions

1. **Partitioning is the cheapest available win, and the compliant choice is
   also the fastest.** `blockcyclic` over `block` is +23% for no hardware change
   and no algorithmic change, with a measured imbalance of 1.04× against 4.54×
   — predicted from a √k cost model before it was measured.
2. **More processes saturate rather than collapse on a cluster.** Task 1 is
   monotonic to 64 ranks but the last 16 return 2% more speedup for 33% more
   hardware; efficiency falls from 87% at 8 ranks to 37% at 64.
3. **The shortfall against theory is the gather and the sort, not serial
   code.** Karp–Flatt *e* is flat at ≈0.020 to 32 ranks against a true serial
   fraction of 0.0090; the 2.2× residual is work the serial program never does,
   and *e* only turns upward once the interconnect dominates at 48+ ranks.
4. **The hybrid is the fastest implementation in the study, and it wins by
   adding workers as threads.** 30.37× at 8×8, +35% over flat MPI at the same
   64 workers, with Karp–Flatt flat across the entire thread axis.
5. **Neither law is useful here without an overhead term.** Amdahl and
   Gustafson both model how work is divided; neither models how it is
   re-collected, which is where this program actually spends its lost speedup.

---

## 8. Appendices

### Appendix A — AI declaration

⟨Required. Declare Generative-AI use during the preparation period and attach
the prompt records as PDF, or state that none was used. AI tools were not used
during the presentation, oral or coding interview sessions.⟩

### Appendix B — Reproducing the results

On the CAAS cluster (every number in this report):

```sh
bench/caas.sh push
bench/caas.sh build
bench/caas.sh submit smoke        # wait for SMOKE PASS before anything else
bench/caas.sh submit baselines    # serial, pthreads, OpenMP
bench/caas.sh submit mpi          # Task 1
bench/caas.sh submit hybrid       # Task 2
bench/caas.sh pull
bench/caas.sh analyse             # -> bench/analysis-caas/report.html
```

Two traps. **Clear stale CSVs out of `bench/results-caas/` before
re-analysing** — `analyse.py` globs the whole directory and will silently pool
runs taken at different problem sizes into one median. And **check
`scontrol show partition defq` before trusting any `--time` in the `.sbatch`
files**: `defq`'s wall limit is 00:20:00, and a job requesting more sits at
`PartitionTimeLimit` indefinitely rather than being rejected at submit time.

On a single host via Docker (development and the single-host comparison):

```sh
bench/sweep.sh          # serial, pthreads, OpenMP baselines
bench/sweep-mpi.sh      # Task 1
bench/sweep-hybrid.sh   # Task 2

python3 bench/analyse.py --out bench/analysis   # both laws + Karp-Flatt
python3 bench/report.py  --out bench/analysis   # figures 1-8 + report.html
```

Individual sweeps can be re-measured in isolation via the `SWEEPS` and
`MAIN_SCHEME` environment variables. Raw per-repetition CSVs are retained in
`bench/results-caas/`; `bench/analysis-caas/summary.csv` holds the reduced
medians and every derived quantity used in this report. The measured
`imbalance` column is carried only in the raw `mpi-*.csv` files.

### Appendix C — Figure index

| Figure | Shows | Specification requirement |
|---|---|---|
| `fig1.svg` | runtime against *n* | (a) 1 |
| `fig2.svg` | empirical speedup against *n* | (a) 2 |
| `fig3.svg` | speedup against worker count | (a) 3, (b) 2 |
| `fig4.svg` | parallel efficiency | supporting |
| `fig5.svg` | partitioning scheme comparison | supporting |
| `fig6.svg` | Task 1 measured vs theoretical | (c) 1 |
| `fig7.svg` | Task 2 measured vs theoretical | (c) 2 |
| `fig8.svg` | threads per rank at fixed process count | (b) 1 |

All figures are in `bench/analysis-caas/figures/` as standalone SVG.

### Appendix D — Source files

| File | Contents |
|---|---|
| `lab-2/task1.c` | Task 1 — Open MPI, four partitioning schemes |
| `lab-2/task2.c` | Task 2 — hybrid MPI + OpenMP, two-level decomposition |
| `lab-2/task3-performance-evaluation.md` | Task 3 — full derivations and experimental design |
| `week-4-lab-1/task1.c` | serial baseline (speedup denominator) |
| `week-4-lab-1/task2.c` | POSIX threads baseline |
| `week-4-lab-1/task3.c` | OpenMP baseline |
