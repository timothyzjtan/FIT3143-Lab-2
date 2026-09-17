#!/usr/bin/env python3
"""
FIT3143 Lab #2 - render bench/analysis/summary.csv as figures.

    python3 bench/analyse.py --out bench/analysis     # first
    python3 bench/report.py  --out bench/analysis     # then

Emits one self-contained HTML page with inline SVG, plus each figure as a
standalone .svg that can be dropped straight into the slides. SVG is generated
by hand rather than with matplotlib so the report builds with nothing but the
Python standard library - the benchmark container has no plotting stack, and
installing one mid-sweep would have perturbed the timings being measured.

Every figure carries a table view beneath it: three of the light-mode series
colours sit below 3:1 contrast on the page surface, and the palette rule is
that such a palette ships visible labels or a table.
"""

import argparse
import csv
import html
import math
import os
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Categorical slots 1-5 of the validated reference palette, in fixed order.
# Verified with the palette validator for both surfaces: light worst adjacent
# CVD dE 9.1, dark 8.4; both clear the >=8 target.
SERIES_STYLE = [
    ("serial",              "Serial",            "#2a78d6", "#3987e5"),
    ("pthreads",            "Pthreads",          "#eb6834", "#d95926"),
    ("openmp",              "OpenMP",            "#1baf7a", "#199e70"),
    ("mpi",                 "MPI (Task 1)",      "#eda100", "#c98500"),
    ("hybrid",              "Hybrid (Task 2)",   "#e87ba4", "#d55181"),
]
LABEL = {k: lab for k, lab, _, _ in SERIES_STYLE}
ORDER = [k for k, _, _, _ in SERIES_STYLE]

# Figures 6 and 7 plot three MODELS of one implementation rather than several
# implementations, so their series are not families and cannot be coloured by
# ORDER position. They borrow slots 3-5, which the palette validator already
# cleared as mutually distinguishable, and register their own labels.
MODEL_SERIES = [("empirical", "Measured", 3),
                ("amdahl",    "Amdahl",   4),
                ("gustafson", "Gustafson", 5)]
LABEL.update({k: lab for k, lab, _ in MODEL_SERIES})
SLOT = {k: i + 1 for i, k in enumerate(ORDER)}
SLOT.update({k: s for k, _, s in MODEL_SERIES})


def family(impl):
    """Collapse an impl name to the series it belongs to.

    The P=1 rows carry the fallback suffix `dynamic(->block@P1)`, and Sweep A
    was re-measured under blockcyclic, so one series spans several impl names.
    """
    for fam in ("hybrid", "mpi", "openmp", "pthreads"):
        if impl.startswith(fam):
            return fam
    return impl


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("n", "procs", "threads", "workers"):
            r[k] = int(r[k])
        for k in ("t_total_median", "speedup_empirical", "efficiency",
                  "speedup_amdahl", "speedup_gustafson", "serial_fraction",
                  "serial_fraction_amdahl", "serial_fraction_gustafson"):
            if k in r:
                r[k] = float(r[k])
        # Karp-Flatt is blank at one worker, where the formula is undefined.
        r["karp_flatt"] = (float(r["karp_flatt"])
                           if r.get("karp_flatt") not in (None, "") else None)
        r["family"] = family(r["impl"])
    return rows


# --------------------------------------------------------------------------
# SVG primitives
# --------------------------------------------------------------------------

W, H = 720, 400
PAD = {"l": 62, "r": 132, "t": 18, "b": 46}


def esc(s):
    return html.escape(str(s), quote=True)


class Scale:
    def __init__(self, lo, hi, a, b, log=False):
        self.log, self.a, self.b = log, a, b
        if log:
            lo, hi = math.log10(max(lo, 1e-12)), math.log10(max(hi, 1e-12))
        if hi == lo:
            hi = lo + 1.0
        self.lo, self.hi = lo, hi

    def __call__(self, v):
        if self.log:
            v = math.log10(max(v, 1e-12))
        return self.a + (v - self.lo) / (self.hi - self.lo) * (self.b - self.a)

    def ticks(self, count=5):
        if self.log:
            out, e = [], math.floor(self.lo)
            while e <= math.ceil(self.hi):
                for m in (1, 2, 5):
                    v = m * 10 ** e
                    if self.lo - 1e-9 <= math.log10(v) <= self.hi + 1e-9:
                        out.append(v)
                e += 1
            return out
        step = (self.hi - self.lo) / count
        mag = 10 ** math.floor(math.log10(step)) if step > 0 else 1
        for m in (1, 2, 2.5, 5, 10):
            if mag * m >= step:
                step = mag * m
                break
        out, v = [], math.ceil(self.lo / step) * step
        while v <= self.hi + 1e-9:
            out.append(round(v, 10))
            v += step
        return out


def fmt_n(v):
    if v >= 1e9:
        return f"{v/1e9:g}B"
    if v >= 1e6:
        return f"{v/1e6:g}M"
    if v >= 1e3:
        return f"{v/1e3:g}k"
    return f"{v:g}"


def line_chart(fid, series, xlab, ylab, xlog=False, ylog=False,
               xfmt=fmt_n, yfmt=lambda v: f"{v:g}", hline=None):
    """series: [(family_key, [(x, y, meta), ...]), ...] in ORDER."""
    xs = [p[0] for _, pts in series for p in pts]
    ys = [p[1] for _, pts in series for p in pts]
    if hline is not None:
        ys.append(hline)
    if not xs:
        return ""
    x0, x1 = PAD["l"], W - PAD["r"]
    y0, y1 = H - PAD["b"], PAD["t"]
    ylo = min(ys) if ylog else 0.0
    if ylog:
        ylo = min(ys) * 0.85
    sx = Scale(min(xs), max(xs), x0, x1, xlog)
    sy = Scale(ylo, max(ys) * 1.06, y0, y1, ylog)

    o = [f'<svg class="fig" id="{esc(fid)}" viewBox="0 0 {W} {H}" '
         f'role="img" preserveAspectRatio="xMidYMid meet">']
    # recessive grid
    for t in sy.ticks():
        y = sy(t)
        o.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        o.append(f'<text class="tick" x="{x0-8}" y="{y+3.5:.1f}" text-anchor="end">'
                 f'{esc(yfmt(t))}</text>')
    for t in sx.ticks():
        x = sx(t)
        o.append(f'<text class="tick" x="{x:.1f}" y="{y0+18}" text-anchor="middle">'
                 f'{esc(xfmt(t))}</text>')
    o.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}"/>')
    if hline is not None:
        y = sy(hline)
        o.append(f'<line class="ref" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        o.append(f'<text class="reflabel" x="{x1-4}" y="{y-6:.1f}" text-anchor="end">'
                 f'ideal (linear)</text>')
    o.append(f'<text class="axlabel" x="{(x0+x1)/2:.0f}" y="{H-8}" '
             f'text-anchor="middle">{esc(xlab)}</text>')
    o.append(f'<text class="axlabel" transform="translate(14,{(y0+y1)/2:.0f}) '
             f'rotate(-90)" text-anchor="middle">{esc(ylab)}</text>')

    pts_json = []
    for key, pts in series:
        if not pts:
            continue
        i = SLOT[key] - 1
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(p[0]):.1f},{sy(p[1]):.1f}"
                     for j, p in enumerate(pts))
        o.append(f'<path class="ln s{i+1}" d="{d}"/>')
        # 2px surface ring keeps overlapping markers separable
        for p in pts:
            o.append(f'<circle class="mk s{i+1}" cx="{sx(p[0]):.1f}" '
                     f'cy="{sy(p[1]):.1f}" r="4"/>')
            pts_json.append({"x": round(sx(p[0]), 1), "y": round(sy(p[1]), 1),
                             "s": LABEL[key], "i": i + 1,
                             "t": f"{xfmt(p[0])} · {yfmt(p[1])}"
                                  + (f" · {p[2]}" if len(p) > 2 and p[2] else "")})
        # direct label at the line end - the light-mode relief rule
        lx, ly = sx(pts[-1][0]), sy(pts[-1][1])
        o.append(f'<text class="dlabel s{i+1}t" x="{lx+9:.1f}" y="{ly+4:.1f}">'
                 f'{esc(LABEL[key])}</text>')
    o.append('</svg>')
    import json
    return "".join(o), json.dumps(pts_json)


def bar_chart(fid, rows, valkey, vallab, fmt=lambda v: f"{v:.2f}"):
    """Horizontal bars: one row per configuration, sorted best first."""
    rows = sorted(rows, key=lambda r: -r[valkey])
    bh, gap = 26, 10
    h = PAD["t"] + len(rows) * (bh + gap) + 42
    x0, x1 = 210, W - 70
    vmax = max(r[valkey] for r in rows) * 1.08
    o = [f'<svg class="fig" id="{esc(fid)}" viewBox="0 0 {W} {h}" role="img" '
         f'preserveAspectRatio="xMidYMid meet">']
    for i, r in enumerate(rows):
        y = PAD["t"] + i * (bh + gap)
        w = (r[valkey] / vmax) * (x1 - x0)
        si = SLOT[r["family"]]
        label = f'{r["impl"]}  ({r["procs"]}×{r["threads"]})'
        o.append(f'<text class="blabel" x="{x0-10}" y="{y+bh/2+4:.0f}" '
                 f'text-anchor="end">{esc(label)}</text>')
        # 4px rounded data-end, anchored to the baseline
        o.append(f'<rect class="bar s{si}f" x="{x0}" y="{y}" width="{max(w,2):.1f}" '
                 f'height="{bh}" rx="4"><title>{esc(label)}: '
                 f'{esc(fmt(r[valkey]))}</title></rect>')
        o.append(f'<text class="bval" x="{x0+w+8:.1f}" y="{y+bh/2+4:.0f}">'
                 f'{esc(fmt(r[valkey]))}</text>')
    o.append(f'<line class="axis" x1="{x0}" y1="{PAD["t"]}" x2="{x0}" '
             f'y2="{PAD["t"]+len(rows)*(bh+gap)}"/>')
    o.append(f'<text class="axlabel" x="{(x0+x1)/2:.0f}" y="{h-10}" '
             f'text-anchor="middle">{esc(vallab)}</text>')
    o.append('</svg>')
    return "".join(o)


def table(headers, rows):
    o = ['<div class="tablewrap"><table><thead><tr>']
    o += [f'<th>{esc(h)}</th>' for h in headers]
    o.append('</tr></thead><tbody>')
    for r in rows:
        o.append('<tr>' + "".join(f'<td>{esc(c)}</td>' for c in r) + '</tr>')
    o.append('</tbody></table></div>')
    return "".join(o)


CSS = """
<style>
:root {
  --surface-0:#f6f5f2; --surface-1:#fcfcfb; --border:#e3e1db;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#77756f;
  --grid:#e8e6e0; --axis:#c9c6bf; --ref:#a8a59d;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface-0:#121211; --surface-1:#1a1a19; --border:#33332f;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#96958c;
    --grid:#2b2b28; --axis:#4a4944; --ref:#6b6a63;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
  }
}
:root[data-theme="dark"] {
  --surface-0:#121211; --surface-1:#1a1a19; --border:#33332f;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#96958c;
  --grid:#2b2b28; --axis:#4a4944; --ref:#6b6a63;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--surface-0); color:var(--text-primary);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:860px; margin:0 auto; padding:40px 20px 72px; }
h1 { font-size:26px; line-height:1.2; margin:0 0 6px; letter-spacing:-.02em; }
h2 { font-size:19px; margin:44px 0 4px; letter-spacing:-.01em; }
.sub { color:var(--text-secondary); margin:0 0 28px; }
.note { color:var(--text-secondary); font-size:13.5px; margin:6px 0 14px; }
figure { margin:0 0 8px; background:var(--surface-1); border:1px solid var(--border);
  border-radius:10px; padding:14px 12px 6px; overflow-x:auto; }
.fig { width:100%; height:auto; display:block; min-width:520px; }
.grid { stroke:var(--grid); stroke-width:1; }
.axis { stroke:var(--axis); stroke-width:1; }
.ref  { stroke:var(--ref); stroke-width:1.5; stroke-dasharray:5 4; }
.reflabel { fill:var(--text-muted); font-size:11px; }
.tick { fill:var(--text-muted); font-size:11px; }
.axlabel { fill:var(--text-secondary); font-size:12.5px; }
.dlabel { font-size:12px; font-weight:600; }
.blabel { fill:var(--text-secondary); font-size:12px; font-family:ui-monospace,monospace; }
.bval   { fill:var(--text-primary); font-size:12px; font-weight:600; }
.ln { fill:none; stroke-width:2; }
.mk { stroke:var(--surface-1); stroke-width:2; }
.s1{stroke:var(--s1)} .s2{stroke:var(--s2)} .s3{stroke:var(--s3)}
.s4{stroke:var(--s4)} .s5{stroke:var(--s5)}
circle.s1{fill:var(--s1)} circle.s2{fill:var(--s2)} circle.s3{fill:var(--s3)}
circle.s4{fill:var(--s4)} circle.s5{fill:var(--s5)}
.s1t{fill:var(--s1)} .s2t{fill:var(--s2)} .s3t{fill:var(--s3)}
.s4t{fill:var(--s4)} .s5t{fill:var(--s5)}
.s1f{fill:var(--s1)} .s2f{fill:var(--s2)} .s3f{fill:var(--s3)}
.s4f{fill:var(--s4)} .s5f{fill:var(--s5)}
.bar { stroke:var(--surface-1); stroke-width:2; }
.legend { display:flex; flex-wrap:wrap; gap:14px; margin:2px 0 16px; font-size:13px;
  color:var(--text-secondary); }
.legend span { display:inline-flex; align-items:center; gap:6px; }
.swatch { width:11px; height:11px; border-radius:3px; display:inline-block; }
details { margin:0 0 10px; }
summary { cursor:pointer; color:var(--text-secondary); font-size:13px; padding:6px 0; }
.tablewrap { overflow-x:auto; border:1px solid var(--border); border-radius:8px;
  background:var(--surface-1); }
table { border-collapse:collapse; width:100%; font-size:13px;
  font-variant-numeric:tabular-nums; }
th,td { padding:7px 11px; text-align:right; border-bottom:1px solid var(--border);
  white-space:nowrap; }
th:first-child,td:first-child { text-align:left; }
th { color:var(--text-secondary); font-weight:600; }
tbody tr:last-child td { border-bottom:none; }
.tip { position:fixed; pointer-events:none; opacity:0; transition:opacity .1s;
  background:var(--surface-1); color:var(--text-primary); border:1px solid var(--border);
  border-radius:7px; padding:6px 9px; font-size:12.5px; box-shadow:0 4px 14px rgba(0,0,0,.16);
  z-index:9; font-variant-numeric:tabular-nums; }
.callout { border-left:3px solid var(--s2); background:var(--surface-1);
  padding:12px 16px; border-radius:0 8px 8px 0; margin:16px 0; font-size:14px; }
code { font-family:ui-monospace,monospace; font-size:.92em;
  background:var(--surface-0); padding:1px 5px; border-radius:4px; }
</style>
"""

JS = """
<script>
(function(){
  var tip=document.createElement('div'); tip.className='tip'; document.body.appendChild(tip);
  document.querySelectorAll('svg.fig[data-pts]').forEach(function(svg){
    var pts=JSON.parse(svg.getAttribute('data-pts'));
    svg.addEventListener('mousemove', function(e){
      var r=svg.getBoundingClientRect(), vb=svg.viewBox.baseVal,
          mx=(e.clientX-r.left)*vb.width/r.width, my=(e.clientY-r.top)*vb.height/r.height,
          best=null, bd=1e9;
      pts.forEach(function(p){ var d=(p.x-mx)*(p.x-mx)+(p.y-my)*(p.y-my);
        if(d<bd){bd=d;best=p;} });
      if(best && bd<900){   // generous hit target, larger than the 4px mark
        tip.innerHTML='<b>'+best.s+'</b><br>'+best.t;
        tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY-10)+'px';
        tip.style.opacity=1;
      } else { tip.style.opacity=0; }
    });
    svg.addEventListener('mouseleave', function(){ tip.style.opacity=0; });
  });
})();
</script>
"""


def legend(keys):
    o = ['<div class="legend">']
    for k in keys:
        i = SLOT[k]
        o.append(f'<span><i class="swatch s{i}f" style="background:var(--s{i})"></i>'
                 f'{esc(LABEL[k])}</span>')
    o.append('</div>')
    return "".join(o)


def best_by(rows, keyfn, valkey="t_total_median"):
    """Keep the fastest configuration for each key (the hybrid reaches one
    worker count by several P x T splits; the best one represents it)."""
    best = {}
    for r in rows:
        k = keyfn(r)
        if k not in best or r[valkey] < best[k][valkey]:
            best[k] = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "bench", "analysis"))
    args = ap.parse_args()
    summary = os.path.join(args.out, "summary.csv")
    rows = load(summary)
    os.makedirs(args.out, exist_ok=True)

    figs = {}
    body = []

    A = [r for r in rows if r["sweep"] == "A"]
    B = [r for r in rows if r["sweep"] == "B"]
    C = [r for r in rows if r["sweep"] == "C"]

    # Sweep A uses whichever scheme was measured across the whole ladder; pick
    # per family the impl with the most n-values so a partial re-measurement
    # never silently replaces the full one.
    def pick_A(fam):
        cand = defaultdict(list)
        for r in A:
            if r["family"] == fam:
                cand[r["impl"]].append(r)
        if not cand:
            return []
        impl = max(cand, key=lambda i: (len({x["n"] for x in cand[i]}),
                                        -statistics_mean(cand[i])))
        return sorted(cand[impl], key=lambda r: r["n"])

    def statistics_mean(rs):
        return sum(r["t_total_median"] for r in rs) / len(rs)

    a_series = [(f, pick_A(f)) for f in ORDER]
    a_series = [(f, rs) for f, rs in a_series if rs]
    a_impl = {f: rs[0]["impl"] for f, rs in a_series}

    # ---- Figure 1: runtime vs n ----
    svg, pts = line_chart(
        "fig1", [(f, [(r["n"], r["t_total_median"], f'{r["procs"]}×{r["threads"]}')
                      for r in rs]) for f, rs in a_series],
        "n (upper bound of the prime search)", "total runtime (s), median of 3",
        xlog=True, ylog=True, yfmt=lambda v: f"{v:g}s")
    figs["fig1"] = svg
    body.append("<h2>1. Runtime against problem size</h2>")
    body.append('<p class="note">Log-log. A straight line means runtime is a '
                'power of n; the parallel implementations sit below serial by a '
                'roughly constant factor, which is what a fixed speedup looks '
                'like on these axes.</p>')
    body.append(legend([f for f, _ in a_series]))
    body.append(f'<figure>{svg.replace("<svg ", f"<svg data-pts={chr(34)}{esc(pts)}{chr(34)} ", 1)}</figure>')
    body.append('<details><summary>Table view</summary>' + table(
        ["n"] + [LABEL[f] for f, _ in a_series],
        [[fmt_n(n)] + [next((f'{r["t_total_median"]:.3f}s' for r in rs if r["n"] == n), "—")
                       for _, rs in a_series]
         for n in sorted({r["n"] for _, rs in a_series for r in rs})]) + '</details>')

    # ---- Figure 2: speedup vs n ----
    s2 = [(f, [(r["n"], r["speedup_empirical"], f'{r["procs"]}×{r["threads"]}')
               for r in rs]) for f, rs in a_series if f != "serial"]
    svg, pts = line_chart("fig2", s2, "n (upper bound of the prime search)",
                          "speedup vs serial", xlog=True,
                          yfmt=lambda v: f"{v:g}×", hline=1.0)
    figs["fig2"] = svg
    body.append("<h2>2. Speedup against problem size</h2>")
    body.append('<p class="note">Speedup is measured against the Week 4 serial '
                'baseline at the same n, which is the denominator the '
                'specification mandates. The dashed line is parity with serial.</p>')
    body.append(legend([f for f, _ in s2]))
    body.append(f'<figure>{svg.replace("<svg ", f"<svg data-pts={chr(34)}{esc(pts)}{chr(34)} ", 1)}</figure>')

    # ---- Figure 3: speedup vs workers ----
    fams = [f for f in ORDER if f != "serial"]
    b_best = best_by([r for r in B if r["family"] != "serial"],
                     lambda r: (r["family"], r["workers"]))
    s3 = []
    for f in fams:
        pts_f = sorted([(k[1], v["speedup_empirical"], f'{v["procs"]}×{v["threads"]}')
                        for k, v in b_best.items() if k[0] == f])
        if pts_f:
            s3.append((f, pts_f))
    svg, pts = line_chart("fig3", s3, "workers (MPI ranks × OpenMP threads)",
                          "speedup vs serial", yfmt=lambda v: f"{v:g}×")
    figs["fig3"] = svg
    body.append("<h2>3. Speedup against degree of parallelism</h2>")
    n_b = fmt_n(B[0]["n"]) if B else "?"
    w_max = max((r["workers"] for r in B), default=0)
    body.append(f'<p class="note">n fixed at {n_b}. The hybrid reaches one worker '
                'count by several P×T splits; the fastest split is plotted and '
                f'named in the tooltip. Worker counts run to {w_max}.</p>')
    body.append(legend([f for f, _ in s3]))
    body.append(f'<figure>{svg.replace("<svg ", f"<svg data-pts={chr(34)}{esc(pts)}{chr(34)} ", 1)}</figure>')
    body.append('<details><summary>Table view — every P×T split measured</summary>' + table(
        ["impl", "P×T", "workers", "total (s)", "speedup", "Amdahl", "Gustafson", "efficiency"],
        [[r["impl"], f'{r["procs"]}×{r["threads"]}', r["workers"],
          f'{r["t_total_median"]:.3f}', f'{r["speedup_empirical"]:.2f}×',
          f'{r["speedup_amdahl"]:.2f}×', f'{r["speedup_gustafson"]:.2f}×',
          f'{r["efficiency"]:.2f}']
         for r in sorted(B, key=lambda r: (r["family"], r["workers"], r["procs"]))
         if r["family"] != "serial"]) + '</details>')

    # ---- Figure 4: efficiency ----
    s4 = []
    for f in fams:
        pts_f = sorted([(k[1], v["efficiency"], f'{v["procs"]}×{v["threads"]}')
                        for k, v in b_best.items() if k[0] == f])
        if pts_f:
            s4.append((f, pts_f))
    svg, pts = line_chart("fig4", s4, "workers (MPI ranks × OpenMP threads)",
                          "efficiency (speedup / workers)",
                          yfmt=lambda v: f"{v:g}")
    figs["fig4"] = svg
    body.append("<h2>4. Parallel efficiency</h2>")
    body.append('<p class="note">Efficiency is speedup divided by worker count. '
                'It falls away from 1 as coordination and memory bandwidth start '
                'to cost more than the work they distribute.</p>')
    body.append(legend([f for f, _ in s4]))
    body.append(f'<figure>{svg.replace("<svg ", f"<svg data-pts={chr(34)}{esc(pts)}{chr(34)} ", 1)}</figure>')

    # ---- Figure 5: scheme comparison ----
    if C:
        n_c = fmt_n(C[0]["n"])
        w_c = sorted({r["workers"] for r in C})
        w_lab = "/".join(str(w) for w in w_c) + " workers"
        figs["fig5"] = bar_chart("fig5", C, "speedup_empirical",
                                 f"speedup vs serial (n = {n_c}, {w_lab})",
                                 fmt=lambda v: f"{v:.2f}×")
        body.append("<h2>5. Partitioning scheme and schedule</h2>")
        body.append(f'<p class="note">All at n = {n_c}; MPI schemes at '
                    f'{w_c[0]} workers, hybrid schemes at {w_c[-1]}. Within each '
                    'family the only variable is how the candidate range is divided.</p>')
        body.append(f'<figure>{figs["fig5"]}</figure>')

    # ---- Figures 6 & 7: measured against theoretical (Task 3) ----------
    def theory_figure(fid, fam, heading, xlab, note, xkey):
        """Plot measured / Amdahl / Gustafson for one implementation.

        Sweep B holds n fixed and varies the degree of parallelism, which is
        exactly the experiment both laws describe, so the theory curves are
        only drawn over sweep B. The hybrid reaches a worker count by several
        P x T splits; the fastest is plotted and every split is tabled.
        """
        fam_rows = [r for r in B if r["family"] == fam]
        if not fam_rows:
            return
        best = best_by(fam_rows, lambda r: r[xkey])
        xs = sorted(best)
        s = [(mk, [(x, best[x][col], f'{best[x]["procs"]}x{best[x]["threads"]}')
                   for x in xs])
             for mk, col in (("empirical", "speedup_empirical"),
                             ("amdahl", "speedup_amdahl"),
                             ("gustafson", "speedup_gustafson"))]
        svg, pts = line_chart(fid, s, xlab, "speedup vs serial",
                              xfmt=lambda v: f"{v:g}",
                              yfmt=lambda v: f"{v:g}x", hline=None)
        figs[fid] = svg
        body.append(f"<h2>{esc(heading)}</h2>")
        body.append(f'<p class="note">{note}</p>')
        body.append(legend([k for k, _ in s]))
        body.append('<figure>' + svg.replace(
            "<svg ", f"<svg data-pts={chr(34)}{esc(pts)}{chr(34)} ", 1) + '</figure>')
        body.append('<details><summary>Table view - fractions and Karp-Flatt'
                    '</summary>' + table(
            ["impl", "PxT", "workers", "measured", "Amdahl", "Gustafson",
             "s (Amdahl)", "s (Gustafson)", "Karp-Flatt e"],
            [[r["impl"], f'{r["procs"]}x{r["threads"]}', r["workers"],
              f'{r["speedup_empirical"]:.2f}x', f'{r["speedup_amdahl"]:.2f}x',
              f'{r["speedup_gustafson"]:.2f}x',
              f'{r.get("serial_fraction_amdahl", 0):.4f}',
              f'{r.get("serial_fraction_gustafson", 0):.4f}',
              "-" if r["karp_flatt"] is None else f'{r["karp_flatt"]:.4f}']
             for r in sorted(fam_rows, key=lambda r: (r["workers"], r["procs"]))])
            + '</details>')

    theory_figure(
        "fig6", "mpi",
        "6. Task 1 (Open MPI): measured against theoretical speedup",
        "MPI processes",
        f"n fixed at {fmt_n(B[0]['n']) if B else '?'}. Amdahl's serial fraction is measured on the serial "
        "run at the same n - the fixed-workload experiment the law assumes - "
        "so its curve is a ceiling that does not move as processes are added. "
        "Gustafson's fraction is measured on each parallel run itself, the "
        "scaled-workload assumption, which is why it rises almost linearly. "
        "The measured curve sits below both because neither law charges for "
        "broadcast, gather or memory bandwidth; the Karp-Flatt column in the "
        "table separates those two causes - a rising e is overhead, a flat e "
        "is genuine serial work.",
        "procs")

    theory_figure(
        "fig7", "hybrid",
        "7. Task 2 (hybrid MPI + OpenMP): measured against theoretical speedup",
        "workers (MPI ranks x OpenMP threads)",
        "Same construction as figure 6, with the worker count now the product "
        "of ranks and threads. The two levels are not equivalent: threads "
        "inside a rank share the address space and skip the gather entirely, "
        "while ranks pay for it, so two splits with the same worker count can "
        "land far apart. Both laws see only the product and predict one value "
        "for both, which is precisely the modelling gap this figure shows.",
        "workers")

    # ---- Figure 8: threads per rank at a fixed process count -----------
    # Specification section (b) graph 1: the hybrid against Task 1 with an
    # increasing number of threads and THE SAME number of MPI processes. Task 1
    # has no threads, so it is a flat reference at that process count - which is
    # the point of the comparison: whether a second level of parallelism inside
    # each rank buys anything the ranks alone did not.
    FIXED_P = 4
    hyb_p = sorted([r for r in B if r["family"] == "hybrid"
                    and r["procs"] == FIXED_P], key=lambda r: r["threads"])
    mpi_p = [r for r in B if r["family"] == "mpi" and r["procs"] == FIXED_P]
    if hyb_p and mpi_p:
        ts = [r["threads"] for r in hyb_p]
        s8 = [("hybrid", [(r["threads"], r["speedup_empirical"],
                           f'{FIXED_P}x{r["threads"]}') for r in hyb_p]),
              ("mpi", [(t, mpi_p[0]["speedup_empirical"], f'{FIXED_P}x1') 
                       for t in ts])]
        svg, pts = line_chart("fig8", s8, f"OpenMP threads per rank (MPI ranks fixed at {FIXED_P})",
                              "speedup vs serial", xfmt=lambda v: f"{v:g}",
                              yfmt=lambda v: f"{v:g}x")
        figs["fig8"] = svg
        body.append("<h2>8. Threads per rank at a fixed process count</h2>")
        body.append(f'<p class="note">n fixed at {fmt_n(hyb_p[0]["n"])}, MPI ranks fixed at '
                    f'{FIXED_P}. Task 1 has no threads, so it is flat at its '
                    f'{FIXED_P}-process speedup; the hybrid adds OpenMP threads '
                    'inside each of those same ranks. Total workers is ranks x '
                    'threads.</p>')
        body.append(legend([k for k, _ in s8]))
        body.append('<figure>' + svg.replace(
            "<svg ", f"<svg data-pts={chr(34)}{esc(pts)}{chr(34)} ", 1) + '</figure>')
        body.append(f'<details><summary>Table view - every P x T split at n = {fmt_n(hyb_p[0]["n"])}'
                    '</summary>' + table(
            ["P", "T", "workers", "impl", "total (s)", "speedup", "efficiency"],
            [[r["procs"], r["threads"], r["workers"], r["impl"],
              f'{r["t_total_median"]:.3f}', f'{r["speedup_empirical"]:.2f}x',
              f'{r["efficiency"]:.2f}']
             for r in sorted([x for x in B if x["family"] == "hybrid"],
                             key=lambda r: (r["procs"], r["threads"]))])
            + '</details>')

    title = "Lab 2 Prime Search Benchmarks"
    page = (f"<title>{title}</title>{CSS}<div class=\"wrap\">"
            f"<h1>{title}</h1>"
            f'<p class="sub">Trial-division prime search under five execution '
            f'models. The algorithm is identical in '
            f'every implementation, so the differences are parallelism, not '
            f'arithmetic.</p>'
            + "".join(body) + "</div>" + JS)

    out_html = os.path.join(args.out, "report.html")
    with open(out_html, "w") as fh:
        fh.write(page)

    svg_dir = os.path.join(args.out, "figures")
    os.makedirs(svg_dir, exist_ok=True)
    # Standalone SVGs for the slides: the page's CSS lives in <style>, so each
    # file gets its own copy with the light-mode values resolved.
    standalone_css = ("<style>"
                      ".grid{stroke:#e8e6e0}.axis{stroke:#c9c6bf}"
                      ".ref{stroke:#a8a59d;stroke-width:1.5;stroke-dasharray:5 4}"
                      ".reflabel{fill:#77756f;font-size:11px}"
                      ".tick{fill:#77756f;font-size:11px}"
                      ".axlabel{fill:#52514e;font-size:12.5px}"
                      ".dlabel{font-size:12px;font-weight:600}"
                      ".blabel{fill:#52514e;font-size:12px;font-family:monospace}"
                      ".bval{fill:#0b0b0b;font-size:12px;font-weight:600}"
                      ".ln{fill:none;stroke-width:2}.mk{stroke:#fcfcfb;stroke-width:2}"
                      ".bar{stroke:#fcfcfb;stroke-width:2}"
                      ".s1{stroke:#2a78d6}.s2{stroke:#eb6834}.s3{stroke:#1baf7a}"
                      ".s4{stroke:#eda100}.s5{stroke:#e87ba4}"
                      "circle.s1{fill:#2a78d6}circle.s2{fill:#eb6834}"
                      "circle.s3{fill:#1baf7a}circle.s4{fill:#eda100}circle.s5{fill:#e87ba4}"
                      ".s1t{fill:#2a78d6}.s2t{fill:#eb6834}.s3t{fill:#1baf7a}"
                      ".s4t{fill:#eda100}.s5t{fill:#e87ba4}"
                      ".s1f{fill:#2a78d6}.s2f{fill:#eb6834}.s3f{fill:#1baf7a}"
                      ".s4f{fill:#eda100}.s5f{fill:#e87ba4}"
                      "</style>")
    for fid, svg in figs.items():
        one = svg.replace('<svg class="fig"',
                          '<svg xmlns="http://www.w3.org/2000/svg" class="fig"', 1)
        one = one.replace('>', '>' + standalone_css, 1)
        with open(os.path.join(svg_dir, f"{fid}.svg"), "w") as fh:
            fh.write(one)

    print(f"wrote {out_html}")
    print(f"wrote {len(figs)} standalone SVGs -> {svg_dir}")
    for f, rs in a_series:
        print(f"  sweep A series: {LABEL[f]:<16} impl={a_impl[f]:<22} "
              f"{len(rs)} points")


if __name__ == "__main__":
    main()
