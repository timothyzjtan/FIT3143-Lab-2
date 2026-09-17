# Task 3 — Performance Evaluation with Amdahl's and Gustafson's Laws

FIT3143 Lab #2. Supporting notes for the calculations and experimental design
behind slide section (c). All numbers below come from
`bench/analysis-caas/summary.csv`, produced by `bench/analyse.py` from the raw
per-repetition CSVs in `bench/results-caas/`.

**Machine.** Monash CAAS, partition `defq`: AMD EPYC 7763 nodes, 16 cores each,
connected by **Gigabit ethernet** (not InfiniBand — the unit's CAAS handout is
explicit on this, and the shortfall analysis below shows it is the binding
constraint at high rank counts). Every figure here is measured across **4 nodes / 64 cores**, so worker
counts above 16 are genuinely distributed across separate physical machines
rather than oversubscribed onto one. Every configuration is repeated 3 times and
reduced by **median**, not mean: the handout notes that CAAS nodes "may be shared
with research clusters", so a neighbouring job can perturb any single
repetition — a median of three absorbs that where a mean would not.

**Problem sizes, and an honest limit on them.** The *n*-sweep covers **30
distinct values** from 20,000,000 to 100,000,000, spaced geometrically. The
serial reference runs 5.49 s at n = 20M and 53.72 s at n = 100M.

The specification warns against reporting runtimes under a second, and at the
*low* end of the ladder the fastest configurations cross that line: 43 of the 207
measured configurations fall below 1 s, the quickest being the hybrid at 4×16
finishing n = 20M in **0.33 s**. All of them are high-worker-count runs at
n ≤ 54M; every configuration at n ≥ 57M is above the threshold.

The effect is visible rather than hypothetical. Across the lowest eight values of
*n*, the hybrid's speedup scatters by 14.4% between adjacent points — neighbouring
problem sizes differing by 6% produce speedups differing by far more than that,
which is measurement noise, not a property of the algorithm. **Every headline
figure in this document is therefore quoted at n = 100M**, where the fastest
configuration still takes 1.76 s and the ranking between implementations is
stable. The low-*n* points are retained because the specification asks for 30
values and because the *trend* across them is sound; individual points there
should not be read as precise.

**Denominator.** Every speedup in this document is against the **Week 4 Task 1
serial program at the same n**, as the specification mandates — never against
the parallel program's own one-worker run. The serial reference is re-measured
on CAAS, so both numerator and denominator come from the same hardware.

**Partitioning scheme.** All headline figures use `blockcyclic` with a chunk of
**4096 candidates** (the default in both programs and in the sweep scripts).
Chunk size is what separates `blockcyclic` from `cyclic`: both interleave, but
`blockcyclic` walks 4096 contiguous candidates before handing the next chunk to
the next rank, preserving cache locality that a stride-1 interleave loses. The
master–worker `dynamic` scheme is reported only in the scheme comparison
(§3, last question): it reserves rank 0 as a pure dispatcher, which both costs
a worker and violates the specification's requirement that *every* process,
including the root, take a share of the work.

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
parallel run rises with W — in our data from **0.015 at W=1 to 0.42 at W=64**.
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
| 20,000,000 | 0.0212 | 0.9788 | 47.1× |
| 45,979,000 | 0.0149 | 0.9851 | 67.0× |
| 100,000,000 | 0.0090 | 0.9910 | 110.9× |

**The serial fraction falls as n grows.** Trial division costs roughly
O(n^1.5/log n) while allocation, ordering and file I/O scale with π(n) ≈
n/log n. The parallel part outgrows the serial part, so the Amdahl ceiling
*rises* with n — 47.1× to 110.9× across our range. This is the answer to the
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

## 2. Results (4 nodes × 16 cores)

### Empirical speedup against problem size

The first axis the specification asks for. Sweep A holds the configuration fixed
and walks *n* across all 30 values; Task 1 runs at 16×1 and Task 2 at 4×16, so
the two are each measured against the serial program at the same *n* rather than
against each other.

| n | serial | MPI 16×1 | vs its Amdahl | hybrid 4×16 | vs its Amdahl | OpenMP 1×16 | Pthreads 1×16 |
|---|---|---|---|---|---|---|---|
| 20,000,000 | 5.49 s | 9.40× | 77% | 16.85× | 62% | 11.52× | 9.34× |
| 29,494,000 | 9.52 s | 10.33× | 84% | 18.79× | 66% | 11.46× | 9.06× |
| 43,497,000 | 16.48 s | 11.04× | 84% | 21.04× | 63% | 12.39× | 9.71× |
| 67,808,000 | 30.93 s | 12.16× | 89% | 24.41× | 65% | 12.45× | 9.93× |
| 100,000,000 | 53.72 s | **12.30×** | **87%** | **30.59×** | **75%** | 13.51× | 10.35× |

**Every implementation gets faster relative to serial as *n* grows**, and the two
MPI-based ones gain most: Task 1 rises 9.40× → 12.30× (+31%) and Task 2
16.85× → 30.59× (+82%), against OpenMP's +17% and Pthreads' +11%.

Two mechanisms compound here. The first is the one §1 predicts: the serial
fraction falls from 0.0212 to 0.0090 across this range, so the Amdahl ceiling
itself rises. The second is specific to the distributed implementations — the
gather is a fixed cost set by π(n), while the parallel part grows as
n^1.5/log n, so communication shrinks as a *proportion* of the run. That is
visible in the "vs its Amdahl" columns: Task 1 closes from 77% of its bound to
87%, and Task 2 from 62% to 75%. **The shared-memory baselines have no such
term and do not close.**

This is the direct answer to the specification's "increasing problem size"
axis for the empirical evaluation, and it is the strongest argument in the study
for using MPI on this problem: message passing is penalised at small *n* and
rewarded at large *n*.

### Task 1 — Open MPI, increasing rank count

Fixing *n* = 100M, where Amdahl's s = 0.0090 gives a ceiling of 110.9×.

| W (ranks) | measured | Amdahl | Gustafson | Karp–Flatt e | efficiency |
|---|---|---|---|---|---|
| 1 | 1.00× | 1.00× | 1.00× | — | 100% |
| 2 | 1.96× | 1.98× | 1.97× | 0.0229 | 98% |
| 4 | 3.76× | 3.89× | 3.81× | 0.0211 | 94% |
| 8 | 7.00× | 7.53× | 7.21× | 0.0205 | 87% |
| 12 | 9.89× | 10.92× | 10.20× | 0.0194 | 82% |
| 16 | 12.26× | 14.09× | 12.54× | 0.0203 | 77% |
| 24 | 16.82× | 19.88× | 17.34× | 0.0185 | 70% |
| 32 | 20.88× | 25.01× | 21.33× | 0.0172 | 65% |
| 48 | 23.34× | 33.72× | 27.99× | 0.0225 | 49% |
| 64 | **23.87×** | 40.82× | 37.49× | 0.0267 | 37% |

**Karp–Flatt is essentially flat at e ≈ 0.018–0.023 from W=2 through W=32, then
rises at 48 and 64.** This is the central diagnostic result. A flat *e* means the
shortfall against Amdahl is genuine serial work, not overhead that grows with
the machine — the program scales as well as its serial fraction allows. The
rise at W≥48 (0.0172 → 0.0225 → 0.0267) is where communication finally begins
to cost more than it saves: 48 ranks buy 23.34× and 64 buy only 23.87×, so the
last 16 ranks return 2% more speedup for 33% more hardware.

Note that measured *e* (≈0.020) still sits about 2.2× above the true serial
fraction (0.0090). That residual gap is the gather: every rank's result vector
must cross a process boundary through `MPI_Gatherv` and be merged at the root,
work the serial program never does.

### What the shortfall is actually made of

Amdahl predicts 40.82× at W=64 and we measure 23.87×. In wall-clock terms
against the 53.72 s serial reference, that is 1.32 s predicted against 2.25 s
measured — a shortfall of **0.93 s** to account for. Three costs explain it, and
the important point is that **none of them exist in the program Amdahl's *s* was
measured on**:

**1. The gather is a fixed cost that does not shrink with W.** Every prime found
must reach rank 0 exactly once, so the volume is set by π(n), not by the number
of ranks:

```
π(1e8) = 5,761,455 primes x 8 bytes = 46.1 MB
46.1 MB / 125 MB/s (1 Gb/s)        = 0.369 s floor, before latency
```

Compute time per rank falls as 1/W; this does not fall at all. So the *fraction*
of runtime spent gathering grows with every rank added, which is precisely the
shape the measured curve has — and 0.369 s is ~40% of the 0.93 s shortfall.

**2. The root's global sort is a serial phase the baseline does not have.**
`blockcyclic` interleaves chunks between ranks, so the gathered result arrives
unordered and the root must `qsort` 5,761,455 longs. The serial program never
sorts anything — it appends in ascending order by construction, which is why its
`t_merge` is measured as exactly zero. Amdahl's *s* was taken from that program,
so this phase is structurally invisible to the model. Its measured cost is the
`t_sort` column of `bench/results-caas/mpi-*.csv`.

**3. Message latency scales with rank count, not volume.** `MPI_Gatherv` at 64
ranks is 64 transfers to one receiver, and the per-message cost is paid 64 times
regardless of how little each rank contributes.

One negative result worth recording: there is **no discontinuity at the node
boundary**. Ranks 1–16 fit on a single node and 24 upward span two or more, yet
efficiency falls smoothly through that crossing (76.6% at W=16, 70.1% at W=24)
and *e* actually dips slightly (0.0203 → 0.0185). The interconnect does not
announce itself with a step when the first message leaves the node; it shows up
as a trend once the compute phase has shrunk enough for a fixed ~0.37 s gather to
matter — which is at W≥48, exactly where *e* turns upward. That is the behaviour
the fixed-cost model above predicts, and it is evidence for it.

This is the honest answer to why neither law brackets the measurement: both model
how work is *divided*, neither models how it is *re-collected*. It is also
exactly what Karp–Flatt detects — *e* ≈ 0.020 against a true serial fraction of
0.0090 is the model reporting 2.2× more "serial work" than the algorithm has,
because it is absorbing the gather and the sort into the only term it owns.

A corollary worth stating: this problem is unusually hostile to message passing.
Prime search has a very high **result-volume to compute-volume ratio** — 46 MB of
output for a search whose parallel part takes ~2 s at this scale. A workload that
returned a single scalar (a sum, a maximum, a count) would pay none of cost 1 or
cost 2, and would track Amdahl far more closely on the same hardware.

### Task 2 — hybrid, and the rank×thread trade-off at fixed worker count

The hybrid's value is not visible at small W, and becomes decisive at large W.
Holding the total worker count fixed and varying the P:T split:

| total W | best split | speedup | same program at P×1 | hybrid advantage |
|---|---|---|---|---|
| 16 | 8×2 | 12.78× | 12.66× | +1% |
| 32 | 2×16 | 21.11× | 18.65× | +13% |
| 64 | 8×8 | **30.37×** | 22.52× | **+35%** |

The fourth column is `task2.c` run with **one thread per rank**, not `task1.c`.
That is the honest control for this comparison — it isolates the threading level
by holding the program constant. It also explains why 64×1 here reads 22.52×
while the Task 1 table above reads 23.87× at the same 64 ranks: the hybrid still
pays `omp` team setup and its per-rank `t_lsort` phase even when the team has a
single thread. The ~6% gap between the two is the standing cost of the hybrid
structure, and is the figure Task 1 buys by not having it.

At W=64 the full ordering is 8×8 (30.37×), 4×16 (30.01×), 16×4 (29.70×),
32×2 (24.78×), 64×1 (22.52×). **Every configuration with 8 or more threads per
rank beats every configuration with 2 or fewer.** The mechanism is direct: a
thread shares its rank's address space, so work split across threads costs no
gather traffic, while work split across ranks does. At W=16 there is little
gather to avoid and the splits are indistinguishable; at W=64 avoiding it is
worth a third of the runtime.

This is the answer to the specification's Task 2 graph requirement comparing
hybrid against flat MPI *at matched total worker count* — and the comparison
only favours the hybrid once the rank count is high enough for communication
to matter.

### Threads per MPI process, at a fixed rank count

The specification's third axis, and the one that separates Task 2 from Task 1
most cleanly. Here the rank count is held at **P = 4** and only the OpenMP team
size varies, so every additional worker arrives as a *thread* rather than a
process — n = 100M throughout:

| T (threads/rank) | total W | measured | Amdahl | Gustafson | Karp–Flatt e | efficiency |
|---|---|---|---|---|---|---|
| 1 | 4 | 3.79× | 3.89× | 3.84× | 0.0186 | 95% |
| 2 | 8 | 7.06× | 7.53× | 7.29× | 0.0190 | 88% |
| 4 | 16 | 12.64× | 14.09× | 13.47× | 0.0177 | 79% |
| 8 | 32 | 20.98× | 25.01× | 22.67× | 0.0169 | 66% |
| 16 | 64 | **30.01×** | 40.82× | 36.75× | 0.0180 | 47% |

**Karp–Flatt is flat across the entire thread axis — 0.0169 to 0.0190, with no
trend.** Set that against the rank axis in the Task 1 table, where *e* held near
0.020 to W=32 and then climbed to 0.0225 at 48 and 0.0267 at 64. Adding threads
does not accumulate overhead; adding ranks does.

The mechanism is the one the shortfall analysis above identifies. A thread shares
its rank's address space, so a worker added as a thread contributes **nothing** to
the 46 MB gather or to the message count. A worker added as a rank contributes to
both. Scaling by threads is therefore free of the term that bends the MPI curve,
which is the whole case for the hybrid.

The same pattern holds at other rank counts, which rules out P = 4 being a lucky
choice:

| fixed P | T range | speedup range | e range |
|---|---|---|---|
| 4 | 1 → 16 | 3.79× → 30.01× | 0.0169–0.0190 |
| 8 | 2 → 8 | 12.78× → 30.37× | 0.0168–0.0189 |
| 16 | 1 → 4 | 12.66× → 29.70× | 0.0176–0.0183 |

Note also that the three rows reaching W = 64 by different routes — 4×16, 8×8 and
16×4 — land within 2% of each other (30.01×, 30.37×, 29.70×). Once the rank count
is low enough that the gather is not the constraint, *how* the remaining workers
are split barely matters; what matters is not spending them on ranks.

Both theoretical columns behave as §1 predicts. Amdahl depends only on W, so it
is identical to the Task 1 row at the same worker count — correctly, since both
implementations parallelise the same phase of the same algorithm. Gustafson sits
between measurement and Amdahl throughout, because its *s* is taken from the
parallel run and rises with W.

### Comparison against the Week 4 baselines

Karp–Flatt also separates the two shared-memory baselines cleanly at n=100M:

| W | OpenMP | e | Pthreads | e |
|---|---|---|---|---|
| 2 | 1.96× | 0.018 | 1.57× | **0.275** |
| 4 | 3.82× | 0.016 | 2.88× | 0.130 |
| 8 | 7.31× | 0.014 | 5.45× | 0.067 |
| 16 | 13.39× | 0.013 | 10.21× | 0.038 |

OpenMP's *e* is the lowest in the entire study (0.013 at W=16) — nothing is
gathered, and `schedule(dynamic)` balances the sqrt(k) cost gradient.

One caveat on that comparison, in OpenMP's disfavour. `week-4-lab-1/task3.c`
does not collect primes the way the other four programs do: it writes a flag per
candidate into an *n*-byte array and then compacts it with an **O(n) serial
sweep**, where the serial, pthreads and both MPI versions append primes directly
(≈π(n) writes). So OpenMP carries serial work the MPI implementations never do,
and still records the lowest *e* in the study. The comparison is therefore
conservative rather than flattering — but it is not algorithmically
like-for-like, and the two should not be read as identical programs differing
only in threading model. Pthreads'
*e* starts at 0.275 and *falls* with W, the signature of **load imbalance rather
than overhead**: its static block partitioning gives the thread holding the
high-numbered candidates far more work, and that penalty is proportionally
worst when there are only two blocks.

---

## 3. Answers to the questions the specification poses

**How does the actual speedup compare against the theoretical speedup?**
It tracks Amdahl closely to about 32 workers and diverges after. At W=8 MPI
achieves 7.00× against a prediction of 7.53× (93% of theory); at W=32, 20.88×
against 25.01× (83%); at W=64, 23.87× against 40.82× (58%). Karp–Flatt
localises the divergence: *e* is flat at ≈0.020 while the curve tracks theory,
and turns upward exactly where it stops. So the loss is irreducible serial work
plus a fixed gather cost up to 32 ranks, and communication overhead beyond.
Gustafson sits **below** Amdahl at every worker count here (37.49× vs 40.82× at
W=64), and the gap widens to W=48 before narrowing. That ordering is a direct
consequence of the two experiments: Gustafson's *s* is taken from the parallel
run, where it grows with W (0.015 → 0.42), so `s + p·W` is pulled down exactly
as the machine grows. Amdahl's *s* stays fixed at 0.0090. For this fixed-n
experiment Amdahl is the appropriate model — the workload does not grow with the
machine — and Gustafson is reported to show what the same code would promise
under a weak-scaling regime, where the comparison would invert.

**Will more MPI processes always increase the speedup?**
No, though on this hardware the failure is saturation rather than collapse.
Theoretically Amdahl saturates: at n=100M the ceiling is 110.9× no matter how
many processes are added. Empirically the curve is monotonic to 64 ranks but
flattens hard — 32→48 ranks buys 12% more speedup, 48→64 buys 2%. Efficiency
tells the story more plainly: 87% at 8 ranks, 77% at 16, 65% at 32, 37% at 64.
The marginal rank is still positive but nearly worthless, and extrapolating the
Karp–Flatt trend, additional ranks past ~80 would make it negative.

This is worth contrasting with our earlier measurements on a single 10-core
laptop, where the MPI curve *peaked and declined* past the physical core count.
Both effects are real; they are different regimes. With more ranks than cores,
extra ranks add context-switching to a fixed amount of compute and actively
hurt. With ranks spread across genuinely separate nodes, they keep helping, just
by less and less.

**How does the workload distribution affect the speedup?**
Substantially, and it is the one variable that costs nothing to change. At
n=100M with 16 ranks (sweep C), holding everything else fixed:

| scheme | MPI speedup | measured imbalance |
|---|---|---|
| blockcyclic | **12.35×** | 1.04× |
| cyclic | 12.34× | 1.01× |
| dynamic | 11.54× | 1.00× |
| block | 10.04× | **4.54×** |

The imbalance column is measured directly, not inferred: each program reduces
`t_compute` with both `MPI_MAX` and `MPI_MIN` and reports their ratio, so 4.54×
means the slowest rank spent 4.54 times as long computing as the fastest. (The
figure is carried in the `imbalance` column of `bench/results-caas/mpi-*.csv`;
`analyse.py` does not propagate it to `summary.csv`.)

`block` is worst because trial-division cost grows with the candidate's
magnitude — the rank holding the top block does far more work than the rank
holding the bottom one, and the phase ends when the slowest rank ends. Its
Karp–Flatt *e* of 0.040 is double `blockcyclic`'s 0.020, and the 4.54× imbalance
is the direct cause.

**This was predicted before it was measured.** Modelling each chunk's cost as
proportional to √k and summing per rank gives an expected imbalance of 3.4×
(using cost ∝ √k⁄ln k) to 4.1× (using cost ∝ √k) for `block`, against **1.0012×**
for `blockcyclic`. Measured: 4.54× and 1.04×. The model slightly understates
`block` — it ignores cache behaviour and the fact that the highest range also
stores the most results — but it is the right order, and it predicts
`blockcyclic`'s near-perfect balance almost exactly. Interleaving balances any
smooth monotonic cost function, and √k is smooth and monotonic.

`blockcyclic` and `cyclic` are statistically indistinguishable here and recover
23% over `block` on identical hardware and identical arithmetic.

The `dynamic` row is the most informative of the four. Its imbalance is
**1.00×** — perfect, better than every static scheme, exactly as a master–worker
scheduler promises. And it is still slower than `blockcyclic`. That rules out
tuning as the explanation: the scheme does precisely what it is designed to do
and loses anyway, because balance was never the binding constraint. There was
only 4% of imbalance left for it to recover, and it spends a whole rank plus
per-chunk request traffic to recover it.

`dynamic` is third, and its deficit has a structural cause rather than a tuning
one: rank 0 acts purely as a dispatcher, so only 15 of 16 ranks compute. That
alone predicts a 1/16 loss, and the measured gap (12.35× → 11.54×, 6.6%) is
close to it. The same effect is larger in the hybrid, because there a
reserved rank takes its whole OpenMP team out with it: at the sweep-C
configuration of 4 ranks × 16 threads, rank 0 idles **16 of the 64 workers**, a
quarter of the machine. Measured, 30.33× for `blockcyclic` against 21.78× for
`dynamic` — a 28% loss, against the 25% the lost workers alone predict. This is why `blockcyclic` is the default in
both programs — it is both the specification-compliant choice and the faster one.

**Will the speedup results be the same across different machines?**
No, and we have direct evidence. The Amdahl ceiling is a property of the
*program and n* and transfers (s = 0.0090 at n=100M is a property of the
algorithm's phase mix). Everything empirical is a property of the host. A single
CAAS core is roughly 4.7× slower than the laptop core we first measured on
(53.7 s versus 11.4 s for the serial run at n=100M), yet CAAS reaches 23.9×
where the laptop reached under 4× — absolute speed and scalability are
independent. Core count sets where the curve turns over, memory bandwidth per
core sets how early efficiency decays, and the interconnect sets the gather cost
that dominates our Karp–Flatt numbers past 32 ranks.

---

## 4. Reproducing

On the cluster (the numbers in this document):

```sh
bench/caas.sh push
bench/caas.sh build
bench/caas.sh submit smoke        # wait for SMOKE PASS
bench/caas.sh submit baselines    # serial, pthreads, OpenMP
bench/caas.sh submit mpi          # Task 1
bench/caas.sh submit hybrid       # Task 2
bench/caas.sh pull
bench/caas.sh analyse             # -> bench/analysis-caas/report.html
```

Two traps, both learned the hard way. **Clear stale CSVs out of
`bench/results-caas/` before re-analysing** — `analyse.py` globs the whole
directory and will silently pool runs taken at different `N_FIXED` into one
median. And **check `scontrol show partition defq` before trusting any `--time`
in the `.sbatch` files**: `defq`'s wall limit is 00:20:00, and a job requesting
more sits at `PartitionTimeLimit` indefinitely rather than being rejected at
submit time.

On a single host via Docker (the oversubscription comparison in §3):

```sh
bench/sweep.sh          # serial, pthreads, OpenMP baselines
bench/sweep-mpi.sh      # Task 1
bench/sweep-hybrid.sh   # Task 2
python3 bench/analyse.py --out bench/analysis
python3 bench/report.py  --out bench/analysis
```

Figures 6 and 7 (`bench/analysis-caas/figures/fig6.svg`, `fig7.svg`) are the two
graphs section (c) requires: measured against Amdahl against Gustafson, for
Task 1 over increasing MPI processes and for Task 2 over increasing ranks ×
threads.
