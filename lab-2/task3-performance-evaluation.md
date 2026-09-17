# Task 3 — Performance Evaluation with Amdahl's and Gustafson's Laws

FIT3143 Lab #2. Supporting notes for the calculations and experimental design
behind slide section (c). All numbers below come from
`bench/analysis-caas/summary.csv`, produced by `bench/analyse.py` from the raw
per-repetition CSVs in `bench/results-caas/`.

**Machine.** Monash CAAS, partition `defq`: AMD EPYC 7763 nodes, 16 cores each.
Every figure here is measured across **4 nodes / 64 cores**, so worker counts
above 16 are genuinely distributed across separate physical machines rather than
oversubscribed onto one. Every configuration is repeated 3 times and reduced by
**median**, not mean: one scheduling hiccup moves a mean of three but leaves a
median untouched.

**Denominator.** Every speedup in this document is against the **Week 4 Task 1
serial program at the same n**, as the specification mandates — never against
the parallel program's own one-worker run. The serial reference is re-measured
on CAAS, so both numerator and denominator come from the same hardware.

**Partitioning scheme.** All headline figures use `blockcyclic`. The
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

## 2. Results (n = 100M, sweep B, 4 nodes × 16 cores)

Amdahl's s = 0.0090 at this n, giving a ceiling of 110.9×.

### Task 1 — Open MPI, increasing rank count

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

### Task 2 — hybrid, and the rank×thread trade-off at fixed worker count

The hybrid's value is not visible at small W, and becomes decisive at large W.
Holding the total worker count fixed and varying the P:T split:

| total W | best split | speedup | flat MPI (P×1) | hybrid advantage |
|---|---|---|---|---|
| 16 | 8×2 | 12.78× | 12.66× | +1% |
| 32 | 2×16 | 21.11× | 18.65× | +13% |
| 64 | 8×8 | **30.37×** | 22.52× | **+35%** |

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

### Comparison against the Week 4 baselines

Karp–Flatt also separates the two shared-memory baselines cleanly at n=100M:

| W | OpenMP | e | Pthreads | e |
|---|---|---|---|---|
| 2 | 1.96× | 0.018 | 1.57× | **0.275** |
| 4 | 3.82× | 0.016 | 2.88× | 0.130 |
| 8 | 7.31× | 0.014 | 5.45× | 0.067 |
| 16 | 13.39× | 0.013 | 10.21× | 0.038 |

OpenMP's *e* is the lowest in the entire study (0.013 at W=16) — nothing is
gathered, and `schedule(dynamic)` balances the sqrt(k) cost gradient. Pthreads'
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

| scheme | MPI speedup |
|---|---|
| blockcyclic | **12.35×** |
| cyclic | 12.34× |
| dynamic | 11.54× |
| block | 10.04× |

`block` is worst because trial-division cost grows with the candidate's
magnitude — the rank holding the top block does far more work than the rank
holding the bottom one, and the phase ends when the slowest rank ends. Its
Karp–Flatt *e* is 0.040, double `blockcyclic`'s 0.020, confirming imbalance
rather than communication as the cause. `blockcyclic` and `cyclic` are
statistically indistinguishable here and recover 23% over `block` on identical
hardware and identical arithmetic.

`dynamic` is third, and its deficit has a structural cause rather than a tuning
one: rank 0 acts purely as a dispatcher, so only 15 of 16 ranks compute. That
alone predicts a 1/16 loss, and the measured gap (12.35× → 11.54×, 6.6%) is
close to it. The same effect is larger in the hybrid, where the idle rank owns
16 threads: 30.33× for `blockcyclic` against 21.78× for `dynamic`, a 28% loss
from reserving one rank in four. This is why `blockcyclic` is the default in
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
