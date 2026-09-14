# FIT3143 Lab #2 — Prime Search with Open MPI
### Presentation slides (Task 4)

> **Fill in before submitting:** student names, IDs, Monash email addresses on
> the title slide, and the AI declaration in Appendix A. The specification
> requires both.
>
> **Figures** referenced as `figN.svg` live in `bench/analysis/figures/`.
> Drop them straight into the slide — standalone SVGs, sized 720×400.
>
> **Budget:** (a) 3 min · (b) 2 min · (c) 2 min · Q&A 2 min. Slide count is
> deliberately low; the marks are for explanation, not for text on screen.

---

## Slide 1 — Title

# Prime Search with Open MPI
## FIT3143 Lab #2 — Parallel Computing

**Team:** ⟨Name 1⟩ (⟨ID⟩, ⟨email⟩@student.monash.edu)
⟨Name 2⟩ (⟨ID⟩, ⟨email⟩@student.monash.edu)

**Problem.** Find every prime strictly less than *n*, output sorted.
**Constant across all five implementations:** the trial-division kernel.
So every difference measured is *parallelism*, not arithmetic.

> **Notes (15s):** Open with the one sentence that makes the whole comparison
> valid — the kernel never changes, so speedup measures parallelism alone.

---

## Slide 2 — Experimental setup

| | |
|---|---|
| Host | 10 cores — 4 performance + 6 efficiency (Apple silicon), Docker |
| Problem sizes | **30 values of n**, 20,000,000 → 100,000,000 |
| Repetitions | 3 per configuration, reduced by **median** |
| Baseline | Week 4 Task 1 **serial** — the denominator for every speedup |
| Timing | `MPI_Wtime`, phase-resolved; `t_compute` reduced with `MPI_MAX` |

Serial reference at n = 100M: **11.417 s**

> **Notes (20s):** Two things to say out loud: median not mean, because these
> runs share a laptop with the OS; and `MPI_MAX` not average for compute,
> because a phase ends when its *slowest* rank ends — averaging would hide the
> load imbalance we are trying to measure.

---

# (a) Task 1 — Open MPI · 3 minutes

---

## Slide 3 — Partitioning: four schemes, measured not asserted

Trial division costs ≈ √k, so **cost grows with the candidate's value.**
Equal-sized contiguous ranges therefore hand the high ranks more work.

| Scheme | Idea | Sort at root? |
|---|---|---|
| `block` | one contiguous range per rank | **no** — arrives ordered |
| `cyclic` | rank *r* takes k = 2+r, 2+r+P, … | yes |
| `blockcyclic` | chunks dealt round-robin | yes |
| `dynamic` | master–worker, chunk on request | yes |

Sorting is timed separately and charged **only** to schemes that need it —
charging `block` for a sort it never runs would misrepresent its real cost.

> **Notes (45s):** Lead with *why* balance is non-trivial here: the workload has
> a cost gradient, so the naive split is the imbalanced one. Mention that
> `dynamic` is the only scheme that adapts to cores of *different speeds*,
> which matters on this host — 4 performance + 6 efficiency cores means equal
> work on unequal cores is imbalanced by construction.

---

## Slide 4 — Which scheme wins → `fig5.svg`

**n = 30M, 8 workers — the only variable is how the range is divided.**

| Scheme | Speedup |
|---|---|
| **blockcyclic** | **2.82×** |
| cyclic | 2.67× |
| dynamic | 2.38× |
| block | 2.09× |

**`blockcyclic` beats `block` by 35% on identical hardware and identical arithmetic.**

> **Notes (30s):** The headline: partitioning alone is worth 35%, for free.
> `block` loses to the cost gradient. `dynamic` balances best in principle but
> pays master–worker request traffic that, at this granularity, costs more than
> the imbalance it removes — a good example of overhead beating theory.

---

## Slide 5 — Runtime against n → `fig1.svg`  *(required graph a-1)*

Log–log, 30 values of n. All parallel implementations run 8 workers.

At **n = 100M**: serial 11.42 s → OpenMP 3.06 s · **MPI 3.07 s** · pthreads 3.18 s

> **Notes (25s):** Straight lines on log–log mean runtime is a clean power law
> in n; the parallel lines sit *below* serial by a near-constant offset, which
> is what a constant speedup looks like in log space. MPI is level with OpenMP
> at the top end.

---

## Slide 6 — Speedup against n → `fig2.svg`  *(required graph a-2)*

**Speedup at n = 100M (vs serial, 8 workers):**

| OpenMP | **MPI (Task 1)** | pthreads | Hybrid |
|---|---|---|---|
| 3.73× | **3.72×** | 3.59× | 2.97× |

Speedup **improves with n** — the fixed costs (broadcast, gather, file I/O)
are amortised over more compute.

> **Notes (25s):** MPI catches OpenMP once n is large enough to pay for the
> gather. That is the core trade: MPI's cost is roughly fixed per run, so
> bigger problems hide it.

---

## Slide 7 — Speedup against process count → `fig3.svg`  *(required graph a-3)*

**n = 30M. MPI processes vs equal thread counts for OpenMP/pthreads.**

- MPI **peaks at 9 processes (2.51×)**, then *declines* — 2.15× at 10, 2.03× at 16
- **MPI at 2 processes = 0.94× — slower than serial**
- OpenMP plateaus around 3.8× and stays there

> **Notes (35s):** Spend your time on the two-process anomaly — it is the most
> revealing number in Task 1. With two ranks, gathering half the primes across
> a process boundary costs more than halving the search saved. Then the decline
> past 10: no cores left, so extra ranks add gather traffic and context
> switching without adding compute.

---

# (b) Task 2 — Hybrid MPI + OpenMP · 2 minutes

---

## Slide 8 — Two-level decomposition

A **chunk** is the unit of work at *both* levels, so the levels don't fight:

```
MPI level      which chunks a rank owns          (block | blockcyclic | dynamic)
OpenMP level   that rank's chunks across threads (#pragma omp for schedule(dynamic))
```

- Threads inside a rank **share the address space** — no copying, no gather
- Each rank sorts its own results **concurrently** → counted as parallel work
- Global sort at root **only** when the scheme interleaves chunks
- `--bind-to none` is mandatory: MPI's default binding pins each rank to one
  core and strangles its threads

> **Notes (35s):** The binding point is worth 10 seconds — we measured the
> hybrid as slowest until we found MPI was pinning every rank to a single core,
> so the OpenMP threads inside had nowhere to run. Results before that fix were
> discarded and the sweep re-run.

---

## Slide 9 — Threads per rank at fixed processes → `fig8.svg`  *(required graph b-1)*

**MPI ranks fixed at 4. Task 1 has no threads → flat reference at 1.99×.**

| Threads per rank | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Hybrid speedup | 2.10× | 2.07× | 2.18× | **1.78×** |

**Adding threads inside 4 ranks buys essentially nothing, then hurts.**
At 4×4 = 16 workers we are oversubscribing a 10-core host.

Best hybrid result is at the *other* extreme — **1 rank × 4 threads = 2.85×**,
i.e. the configuration that is barely distributed at all.

> **Notes (35s):** Honest finding: for this problem the second level of
> parallelism does not pay. Once the ranks already saturate the cores, threads
> inside them just subdivide the same silicon. The hybrid only wins when you
> *cut* the rank count — which is really an argument for OpenMP.

---

## Slide 10 — Hybrid vs shared-memory at matched worker counts → `fig3.svg`  *(required graph b-2)*

Total workers = ranks × threads, matched against OpenMP/pthreads thread counts.

| Workers | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| OpenMP | 1.90× | 3.41× | 3.67× | 3.72× |
| **Hybrid** | 1.81× | 2.85× | 2.70× | 2.04× |

**The hybrid never beats pure OpenMP at an equal worker count.**

> **Notes (25s):** State the conclusion plainly — on a single shared-memory
> host, adding MPI to OpenMP adds cost and removes nothing. The hybrid's case
> is multi-node, where OpenMP simply cannot reach the second machine.

---

# (c) Task 3 — Performance evaluation · 2 minutes

---

## Slide 11 — The two laws need two *different* experiments

| | **Amdahl** | **Gustafson** |
|---|---|---|
| Holds fixed | problem size *n* | execution time |
| S(W) | 1 / (s + p/W) | s + p·W |
| *s* is the serial share of… | the **serial** run | the **parallel** run as executed |
| So we measure it on… | serial task1 at that n | the P×T run being scored |

**Why this matters:** feeding Amdahl the parallel run's fraction is circular.
As W grows the parallel part shrinks, so that fraction *rises* — 0.028 at W=1
to 0.152 at W=8 — and the "ceiling" would sag as W grows. A bound that moves
with the thing it bounds is not a bound.

> **Notes (40s):** This is the slide the markers are looking for. Make the
> circularity argument explicitly — it is the difference between having done
> the experiment and having quoted the formula.

---

## Slide 12 — Measured fractions, and the Karp–Flatt check

**Amdahl's s, from the serial run — falls as n grows:**

| n | s | ceiling 1/s |
|---|---|---|
| 20M | 0.0288 | 34.8× |
| 100M | 0.0140 | **71.7×** |

Search is ≈O(n^1.5/log n); allocation, sort and I/O scale with π(n).
The parallel part **outgrows** the serial part, so the ceiling *rises* with n.

**Karp–Flatt** — invert Amdahl on the *measured* speedup: e = (1/S − 1/W)/(1 − 1/W)

| W | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| OpenMP e | 0.053 | 0.057 | 0.168 | 0.220 |
| MPI e | **1.127** | 0.338 | 0.325 | 0.456 |

True s is 0.024. **Every e is far larger and all of them climb with W → the
shortfall is overhead, not serial code.**

> **Notes (40s):** Two findings. First, bigger n doesn't make it faster, it
> raises the limit on how fast it could ever get. Second, Karp–Flatt tells us
> *which* of the two possible causes we're looking at: flat e would mean
> genuine serial work, rising e means overhead. Ours rises everywhere, and
> MPI's is an order of magnitude above the program's real serial fraction —
> that's the gather.

---

## Slide 13 — Measured vs theoretical → `fig6.svg` + `fig7.svg`  *(required graphs c-1, c-2)*

**Task 1, n = 30M (Amdahl s = 0.0236, ceiling 42.4×):**

| W | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| Measured | 1.00× | 1.99× | 2.44× | 2.03× |
| Amdahl | 1.00× | 3.74× | 6.87× | 11.82× |
| Gustafson | 1.00× | 3.67× | 6.94× | 14.20× |

- The gap **widens** with W — neither law charges for broadcast or gather
- Gustafson exceeds Amdahl past W=8: it assumes the workload grows with the
  machine. Ours doesn't, so **for this fixed-n experiment Amdahl is the honest
  model**; Gustafson shows what the same code would promise under weak scaling
- Amdahl saturates anyway: W=40 buys 20.8×, and 90% of the 42.4× ceiling would
  need ≈380 workers

> **Notes (40s):** Don't read the table. Say: both laws over-predict, the gap
> grows with W, and Karp–Flatt on the previous slide already told us why. Then
> the Amdahl/Gustafson crossover — not a contradiction, they answer different
> questions and only one of them matches our experiment.

---

## Slide 14 — Conclusions

1. **Partitioning is the cheapest win** — `blockcyclic` over `block` is +35% for
   no hardware and no algorithm change
2. **More processes is not monotonically better** — MPI peaks at 9 and declines;
   at W=2 it is *slower than serial*
3. **On one shared-memory host, MPI's gather is the binding constraint** —
   Karp–Flatt e ≈ 0.33–0.46 against a true serial fraction of 0.024
4. **The hybrid loses to pure OpenMP at every matched worker count** here; its
   case is multi-node, which is the axis this host cannot test
5. **Theory brackets nothing useful without an overhead term** — Amdahl and
   Gustafson both model the division of work, neither models its re-collection

> **Notes (20s):** Land on 5 — it shows you understand the *limits* of the two
> laws, not just how to apply them.

---

## Slide 15 — Q&A

**Anticipated questions — have these ready:**

- *Would you recommend Open MPI for prime search over OpenMP?*
  Not on a single machine — same peak speedup, far more overhead, and it loses
  badly at low process counts. Yes across machines, where OpenMP cannot go at
  all. The pro is memory scale and multi-node reach; the con is that every
  result must cross a process boundary, which for prime search means gathering
  millions of integers to one rank.

- *Why does empirical differ from theoretical?*
  Both laws model only how work is divided. They assign zero cost to broadcast,
  gather, and memory bandwidth. Karp–Flatt separates the two candidate causes,
  and ours rises with W, so it is overhead rather than serial code.

- *Why `MPI_MAX` for compute time?*
  A parallel phase ends when its slowest rank ends. An average would hide the
  imbalance the partitioning experiment exists to measure.

- *Why median of 3 rather than mean?*
  One scheduling hiccup moves a mean of three; a median is unaffected. These
  runs share the host with the OS.

- *Why does `block` need no sort?*
  Each rank owns one contiguous ascending range, so the ranks' results
  concatenate in order at the root. Interleaving schemes scatter each rank's
  primes across [2, n), so they must be sorted after the gather.

---

## Appendix A — AI declaration

⟨Required. Declare Generative-AI use during the preparation period and attach
the prompt records as PDF, or state that none was used. AI tools were not used
during the presentation or Q&A.⟩

---

## Appendix B — Reproducing the results

```sh
bench/sweep.sh          # serial, pthreads, OpenMP baselines
bench/sweep-mpi.sh      # Task 1
bench/sweep-hybrid.sh   # Task 2
python3 bench/analyse.py --out bench/analysis   # both laws + Karp-Flatt
python3 bench/report.py  --out bench/analysis   # figures 1-8
```

Full derivations and experimental design: `lab-2/task3-performance-evaluation.md`

---

## Appendix C — Figure index

| Figure | Shows | Spec requirement |
|---|---|---|
| `fig1.svg` | runtime vs n | (a) 1 |
| `fig2.svg` | speedup vs n | (a) 2 |
| `fig3.svg` | speedup vs worker count | (a) 3, (b) 2 |
| `fig4.svg` | parallel efficiency | supporting |
| `fig5.svg` | partitioning scheme comparison | supporting |
| `fig6.svg` | Task 1 measured vs theoretical | (c) 1 |
| `fig7.svg` | Task 2 measured vs theoretical | (c) 2 |
| `fig8.svg` | threads per rank at fixed P | (b) 1 |
