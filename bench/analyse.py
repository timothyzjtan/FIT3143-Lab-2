#!/usr/bin/env python3
"""
FIT3143 Lab #2 - turn raw sweep CSVs into the figures the specification asks for.

    python3 bench/analyse.py                  # newest CSVs in bench/results
    python3 bench/analyse.py --out analysis   # where the tidy CSVs go

Reads the raw per-repetition rows written by bench/sweep.sh and
bench/sweep-mpi.sh, reduces them to medians, and derives:

  * empirical speedup against the Week 4 SERIAL baseline (the denominator the
    specification mandates), matched on n;
  * theoretical speedup from Amdahl's Law and Gustafson's Law, using the
    measured phase breakdown rather than an assumed serial fraction.

Median, not mean: a single scheduling hiccup skews a mean of three but leaves
a median untouched, and these runs share a laptop with the OS.

ON THE TWO LAWS, AND WHERE THEIR SERIAL FRACTIONS COME FROM
  The two laws make DIFFERENT measurement assumptions, so they must not be fed
  the same measured fraction. Task 3 turns on getting this right.

  Amdahl fixes the PROBLEM SIZE and asks how much faster one fixed workload
  gets with more workers:            S(W) = 1 / (s + p/W)
  Its s is a property of the workload itself, so it is measured on the SERIAL
  run at the same n -- the one-worker execution whose time Amdahl is dividing.
  Taking s from a P-way run instead would be circular: the parallel run's
  serial share already grew because the parallel part shrank, so the ceiling
  would sag as W rose and would no longer be a bound on anything.

  Gustafson fixes the TIME and asks how much more work a bigger machine gets
  through:                           S(W) = s + p*W
  Its s is by construction the serial share OF THE PARALLEL RUN -- the scaled
  workload as actually executed -- so it is measured on the run being scored,
  at that (n, P, T).

  Both are reported per configuration, each from its own experiment:

    serial_fraction_amdahl     from serial task1 at the same n
    serial_fraction_gustafson  from this configuration's own phase breakdown

  KARP-FLATT is reported alongside them as a check on both. Inverting Amdahl
  on the MEASURED speedup gives the experimentally determined serial fraction
        e = (1/S - 1/W) / (1 - 1/W)
  If e stays flat as W grows, the gap to the theoretical curve really is the
  program's serial part. If e climbs with W, the loss is parallel OVERHEAD --
  broadcast, gather, memory bandwidth -- which neither law models, and which
  is the actual reason the empirical curves fall away from theory here.
"""

import argparse
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "bench", "results")


def newest(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    return files[-1] if files else None


def load(path_or_pattern):
    """Load one CSV, or every CSV matching a glob.

    A sweep can be re-measured on its own (SWEEPS=A bench/sweep-mpi.sh), which
    leaves several CSVs per implementation, each holding a different subset of
    the sweeps. Reading only the newest would silently drop the sweeps it does
    not contain, so all matching files are concatenated. Rows stay distinct
    because a re-measurement under a different scheme carries a different
    impl name.
    """
    if not path_or_pattern:
        return []
    if os.path.exists(path_or_pattern):
        paths = [path_or_pattern]
    else:
        paths = sorted(glob.glob(os.path.join(RESULTS, path_or_pattern)))
    rows = []
    for path in paths:
        with open(path, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def fnum(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def median_by(rows, keyfields, valuefield):
    """Group rows by keyfields and return {key_tuple: median(value)}."""
    buckets = defaultdict(list)
    for r in rows:
        key = tuple(r.get(f, "") for f in keyfields)
        v = fnum(r, valuefield, None)
        if v is not None:
            buckets[key].append(v)
    return {k: statistics.median(v) for k, v in buckets.items() if v}


# Phases that do NOT scale with worker count: allocation, dissemination,
# gathering, the root's global sort and writing the file all happen on one
# worker, or are pure communication.
SERIAL_PHASES = ("t_alloc", "t_bcast", "t_merge", "t_comm", "t_sort", "t_io")

# The serial baseline reports a smaller set of phases (it has nothing to
# broadcast and nothing to gather), so it is split with its own tuples. These
# feed Amdahl's fraction; the tuples above feed Gustafson's.
SERIAL_REF_SERIAL_PHASES = ("t_alloc", "t_merge", "t_io")
SERIAL_REF_PARALLEL_PHASES = ("t_compute",)

# Phases that DO scale. t_lsort is the hybrid's per-rank sort of its own
# results: every rank runs it at the same time on disjoint data, so it is
# parallel work, not an added serial cost. Counting it as serial would inflate
# s and depress the Amdahl ceiling for the hybrid alone, making the three
# implementations incomparable.
PARALLEL_PHASES = ("t_compute", "t_lsort")


def sum_phases(row, keys):
    return sum(fnum(row, k) for k in keys if k in row)


def serial_phases(row):
    return sum_phases(row, SERIAL_PHASES)


def parallel_phases(row):
    return sum_phases(row, PARALLEL_PHASES)


def karp_flatt(speedup, workers):
    """Experimentally determined serial fraction, e = (1/S - 1/W)/(1 - 1/W).

    Undefined at W = 1 (Amdahl's law is vacuous there, and the formula divides
    by zero), so one-worker configurations return None and print blank.
    """
    if workers <= 1 or speedup <= 0:
        return None
    return (1.0 / speedup - 1.0 / workers) / (1.0 - 1.0 / workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "analysis"))
    ap.add_argument("--baselines", default=None)
    ap.add_argument("--mpi", default=None)
    ap.add_argument("--hybrid", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    base = load(args.baselines or "baselines-*.csv")
    mpi = load(args.mpi or "mpi-*.csv")
    hyb = load(args.hybrid or "hybrid-*.csv")

    if not base:
        sys.exit("error: no baseline CSV found in bench/results "
                 "(run bench/sweep.sh first)")

    allrows = base + mpi + hyb

    # ---- serial reference, matched on n --------------------------------
    serial_rows = [r for r in base if r["impl"] == "serial"]
    serial = {}
    for (impl, n), t in median_by(serial_rows, ["impl", "n"], "t_total").items():
        serial[int(n)] = t

    if not serial:
        sys.exit("error: no serial rows found - speedup has no denominator")

    # ---- Amdahl's serial fraction, measured on the SERIAL run ------------
    # This is the fixed-problem-size experiment the law assumes: one worker,
    # the whole of n, phases timed in place. It depends only on n, never on
    # how many workers the configuration being scored happens to use.
    serial_by_n = defaultdict(list)
    for r in serial_rows:
        serial_by_n[int(r["n"])].append(r)

    amdahl_s = {}
    for n, group in serial_by_n.items():
        s_abs = statistics.median(
            [sum_phases(r, SERIAL_REF_SERIAL_PHASES) for r in group])
        p_abs = statistics.median(
            [sum_phases(r, SERIAL_REF_PARALLEL_PHASES) for r in group])
        total = s_abs + p_abs
        amdahl_s[n] = (s_abs / total) if total > 0 else 0.0

    # ---- medians for every configuration -------------------------------
    med_total = median_by(allrows, ["impl", "n", "procs", "threads", "sweep"],
                          "t_total")
    med_compute = median_by(allrows, ["impl", "n", "procs", "threads", "sweep"],
                            "t_compute")

    # phase medians, for the theoretical-speedup calculation
    phase_rows = defaultdict(list)
    for r in allrows:
        phase_rows[(r["impl"], r["n"], r["procs"], r["threads"],
                    r.get("sweep", ""))].append(r)

    rows_out = []
    for key, t_total in sorted(med_total.items()):
        impl, n_s, procs_s, threads_s, sweep = key
        n = int(n_s)
        procs, threads = int(procs_s), int(threads_s)
        workers = max(procs * threads, 1)

        ref = serial.get(n)
        if ref is None:
            continue  # no matched serial run at this n; skip rather than guess

        group = phase_rows[key]
        # Phase medians across the repetitions of this configuration.
        s_abs = statistics.median([serial_phases(r) for r in group])
        c_abs = statistics.median(
            [parallel_phases(r) for r in group]) if group else 0.0

        # Gustafson's fraction: the serial share of THIS run, as executed.
        denom = s_abs + c_abs
        s_gus = (s_abs / denom) if denom > 0 else 0.0
        p_gus = 1.0 - s_gus
        gustafson = s_gus + p_gus * workers

        # Amdahl's fraction: the serial share of the SERIAL run at this n.
        # Falls back to the parallel run's fraction only if the baseline sweep
        # never measured this n, which the speedup guard above already skips.
        s_amd = amdahl_s.get(n, s_gus)
        p_amd = 1.0 - s_amd
        amdahl = 1.0 / (s_amd + p_amd / workers) if workers > 0 else 1.0

        emp = (ref / t_total) if t_total > 0 else 0.0
        e_kf = karp_flatt(emp, workers)

        rows_out.append({
            "impl": impl,
            "sweep": sweep,
            "n": n,
            "procs": procs,
            "threads": threads,
            "workers": workers,
            "t_serial_ref": round(ref, 6),
            "t_total_median": round(t_total, 6),
            "t_compute_median": round(med_compute.get(key, 0.0), 6),
            "speedup_empirical": round(emp, 4),
            # Kept under the old name so existing consumers do not break; it is
            # Gustafson's fraction, the serial share of this parallel run.
            "serial_fraction": round(s_gus, 6),
            "parallel_fraction": round(p_gus, 6),
            "serial_fraction_amdahl": round(s_amd, 6),
            "serial_fraction_gustafson": round(s_gus, 6),
            "speedup_amdahl": round(amdahl, 4),
            "speedup_gustafson": round(gustafson, 4),
            "karp_flatt": round(e_kf, 6) if e_kf is not None else "",
            "efficiency": round(emp / workers, 4) if t_total > 0 else 0.0,
        })

    out_path = os.path.join(args.out, "summary.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    print(f"wrote {out_path}  ({len(rows_out)} configurations)")

    # ---- console summary ------------------------------------------------
    def show(title, rows, xlabel, xkey):
        if not rows:
            return
        print(f"\n{title}")
        # PxT is printed because the hybrid reaches one worker count by several
        # splits (8 = 8x1 = 4x2 = 2x4 = 1x8); without it those rows are
        # indistinguishable and look like duplicates.
        print(f"  {xlabel:>10} {'PxT':>6} {'impl':<26} {'total(s)':>9} "
              f"{'speedup':>8} {'amdahl':>8} {'gustaf':>8} {'karp-e':>7} "
              f"{'eff':>6}")
        for r in rows:
            split = f"{r['procs']}x{r['threads']}"
            kf = r["karp_flatt"]
            print(f"  {r[xkey]:>10} {split:>6} {r['impl']:<26} "
                  f"{r['t_total_median']:>9.3f} "
                  f"{r['speedup_empirical']:>8.2f} {r['speedup_amdahl']:>8.2f} "
                  f"{r['speedup_gustafson']:>8.2f} "
                  f"{(f'{kf:.4f}' if kf != '' else '-'):>7} "
                  f"{r['efficiency']:>6.2f}")

    b = sorted([r for r in rows_out if r["sweep"] == "B" and r["impl"] != "serial"],
               key=lambda r: (r["impl"], r["workers"], r["procs"]))
    show("Scaling with worker count (sweep B)", b, "workers", "workers")

    c = sorted([r for r in rows_out if r["sweep"] == "C"],
               key=lambda r: r["impl"])
    show("Partitioning / schedule comparison (sweep C)", c, "workers", "workers")

    a = [r for r in rows_out if r["sweep"] == "A" and r["impl"] != "serial"]
    if a:
        biggest = max(r["n"] for r in a)
        show(f"Largest problem size (sweep A, n={biggest})",
             sorted([r for r in a if r["n"] == biggest], key=lambda r: r["impl"]),
             "n", "n")

    # ---- how the serial fraction itself moves with n --------------------
    # Amdahl's ceiling is set by s, so the interesting question for the
    # "increasing problem size" axis is whether s shrinks as n grows. It does:
    # the search is O(n^1.5)-ish while allocation, sorting and file I/O are
    # roughly O(pi(n)), so the parallel part outgrows the serial part.
    if amdahl_s:
        ns = sorted(amdahl_s)
        show_n = [ns[0], ns[len(ns) // 2], ns[-1]] if len(ns) >= 3 else ns
        print("\nAmdahl's serial fraction s, measured on the serial run")
        print(f"  {'n':>12} {'s':>10} {'p':>10} {'ceiling S(inf)':>15}")
        for n in show_n:
            s = amdahl_s[n]
            ceiling = (1.0 / s) if s > 0 else float("inf")
            print(f"  {n:>12} {s:>10.5f} {1-s:>10.5f} {ceiling:>15.1f}x")

    n_vals = sorted({r["n"] for r in rows_out if r["sweep"] == "A"})
    print(f"\nDistinct values of n in sweep A: {len(n_vals)}"
          f"{'  [OK, >= 30 required]' if len(n_vals) >= 30 else '  [SHORT of the 30 required]'}")


if __name__ == "__main__":
    main()
