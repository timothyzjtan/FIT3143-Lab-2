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

ON THE TWO LAWS
  Amdahl fixes the PROBLEM SIZE and asks how much faster a fixed workload gets
  with more workers:      S(P) = 1 / (s + p/P)
  Gustafson fixes the TIME and asks how much more work gets done:
                          S(P) = s + p*P
  They answer different questions, so both are reported. s is the measured
  serial fraction (allocation + broadcast + gather + sort + file I/O) and
  p = 1 - s the parallel fraction (the search itself).
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


def load(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


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


def serial_phases(row):
    """Sum of the phases that do NOT scale with worker count.

    Whatever columns a given implementation reports, everything except
    t_compute is serial: allocation, dissemination, gathering, sorting and
    writing all happen on one worker (or are pure communication).
    """
    total = 0.0
    for k in ("t_alloc", "t_bcast", "t_merge", "t_lsort", "t_comm",
              "t_sort", "t_io"):
        if k in row:
            total += fnum(row, k)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "analysis"))
    ap.add_argument("--baselines", default=None)
    ap.add_argument("--mpi", default=None)
    ap.add_argument("--hybrid", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    base = load(args.baselines or newest("baselines-*.csv"))
    mpi = load(args.mpi or newest("mpi-*.csv"))
    hyb = load(args.hybrid or newest("hybrid-*.csv"))

    if not base:
        sys.exit("error: no baseline CSV found in bench/results "
                 "(run bench/sweep.sh first)")

    allrows = base + mpi + hyb

    # ---- serial reference, matched on n --------------------------------
    serial = {}
    for (impl, n), t in median_by(
            [r for r in base if r["impl"] == "serial"],
            ["impl", "n"], "t_total").items():
        serial[int(n)] = t

    if not serial:
        sys.exit("error: no serial rows found - speedup has no denominator")

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
            [fnum(r, "t_compute") for r in group]) if group else 0.0

        denom = s_abs + c_abs
        s_frac = (s_abs / denom) if denom > 0 else 0.0
        p_frac = 1.0 - s_frac

        amdahl = 1.0 / (s_frac + p_frac / workers) if workers > 0 else 1.0
        gustafson = s_frac + p_frac * workers

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
            "speedup_empirical": round(ref / t_total, 4) if t_total > 0 else 0.0,
            "serial_fraction": round(s_frac, 6),
            "parallel_fraction": round(p_frac, 6),
            "speedup_amdahl": round(amdahl, 4),
            "speedup_gustafson": round(gustafson, 4),
            "efficiency": round((ref / t_total) / workers, 4) if t_total > 0 else 0.0,
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
        print(f"  {xlabel:>10} {'impl':<26} {'total(s)':>9} {'speedup':>8} "
              f"{'amdahl':>8} {'eff':>6}")
        for r in rows:
            print(f"  {r[xkey]:>10} {r['impl']:<26} {r['t_total_median']:>9.3f} "
                  f"{r['speedup_empirical']:>8.2f} {r['speedup_amdahl']:>8.2f} "
                  f"{r['efficiency']:>6.2f}")

    b = sorted([r for r in rows_out if r["sweep"] == "B" and r["impl"] != "serial"],
               key=lambda r: (r["impl"], r["workers"]))
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

    n_vals = sorted({r["n"] for r in rows_out if r["sweep"] == "A"})
    print(f"\nDistinct values of n in sweep A: {len(n_vals)}"
          f"{'  [OK, >= 30 required]' if len(n_vals) >= 30 else '  [SHORT of the 30 required]'}")


if __name__ == "__main__":
    main()
