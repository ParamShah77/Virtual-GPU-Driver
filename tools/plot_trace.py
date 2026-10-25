#!/usr/bin/env python3
"""Plot a vgpu scheduler trace (CSV from `vgpu_trace` or `vgpu_bench --trace`).

Produces a Gantt chart per engine (one row per channel, one bar per timeslice)
plus a text summary of how engine time was shared.

    python3 tools/plot_trace.py trace.csv [-o gantt.png] [--window-ms 20]

Without matplotlib it falls back to the text summary only.
"""
import argparse
import csv
import sys
from collections import defaultdict

ENGINE_NAMES = {0: "copy", 1: "compute"}
PRIO_NAMES = {0: "low", 1: "normal", 2: "high"}


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["ts_ns"] = int(r["ts_ns"])
        r["chan"] = int(r["chan"])
        r["engine"] = int(r["engine"])
        r["arg"] = int(r["arg"])
        r["arg2"] = int(r["arg2"])
    return rows


def slices(rows):
    """(engine, chan, prio, start_ns, end_ns, virtual_ns) per scheduled slice."""
    open_ = {}
    out = []
    for r in rows:
        key = r["engine"]
        if r["type"] == "slice_begin":
            open_[key] = (r["chan"], r["arg"], r["ts_ns"])
        elif r["type"] == "slice_end" and key in open_:
            chan, prio, t0 = open_.pop(key)
            out.append((key, chan, prio, t0, r["ts_ns"], r["arg2"]))
    return out


def summarize(rows):
    cost = defaultdict(int)
    cmds = defaultdict(int)
    prio = {}
    for e, c, p, *_ in slices(rows):
        prio[(e, c)] = p
    # Shares are only meaningful while the channels were actually competing:
    # stop counting at the moment the first channel on an engine ran its last
    # command.
    last = {}
    for r in rows:
        if r["type"] == "cmd":
            last[(r["engine"], r["chan"])] = r["ts_ns"]
    window_end = {}
    for (e, c), ts in last.items():
        window_end[e] = min(window_end.get(e, ts), ts)
    for r in rows:
        if r["type"] == "cmd" and r["ts_ns"] <= window_end[r["engine"]]:
            cost[(r["engine"], r["chan"])] += r["arg2"]
            cmds[(r["engine"], r["chan"])] += 1
    for eng in sorted({k[0] for k in cost}):
        total = sum(v for k, v in cost.items() if k[0] == eng) or 1
        print(f"engine {ENGINE_NAMES.get(eng, eng)} (until first channel drained):")
        for (e, c), v in sorted(cost.items()):
            if e != eng:
                continue
            p = PRIO_NAMES.get(prio.get((e, c)), "?")
            print(f"  chan {c:<3} prio={p:<6} cmds={cmds[(e, c)]:<6} "
                  f"modelled={v / 1e6:9.3f} ms  share={100.0 * v / total:5.1f}%")
    faults = [r for r in rows if r["type"] == "fault"]
    for r in faults:
        print(f"fault: chan {r['chan']} code={r['arg']} va=0x{r['arg2']:x}")


def plot(rows, out, window_ms):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot "
              "(pip install matplotlib)", file=sys.stderr)
        return
    sl = slices(rows)
    if not sl:
        print("no slices in trace", file=sys.stderr)
        return
    t0 = min(s[3] for s in sl)
    engines = sorted({s[0] for s in sl})
    fig, axes = plt.subplots(len(engines), 1, squeeze=False,
                             figsize=(11, 2.6 * len(engines)), sharex=True)
    colors = {0: "#6b8fb5", 1: "#d9a441", 2: "#c8553d"}
    for ax, eng in zip(axes[:, 0], engines):
        chans = sorted({s[1] for s in sl if s[0] == eng})
        for s in sl:
            if s[0] != eng:
                continue
            start = (s[3] - t0) / 1e6
            width = max((s[4] - s[3]) / 1e6, 0.005)
            if window_ms and start > window_ms:
                continue
            ax.broken_barh([(start, width)], (chans.index(s[1]) - 0.4, 0.8),
                           facecolors=colors.get(s[2], "gray"))
        ax.set_yticks(range(len(chans)))
        ax.set_yticklabels([f"chan {c}" for c in chans])
        ax.set_title(f"{ENGINE_NAMES.get(eng, eng)} engine timeslices")
        ax.set_xlabel("wall time (ms)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in (0, 1, 2)]
    axes[0, 0].legend(handles, ["low", "normal", "high"], loc="upper right",
                      title="priority")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="gantt.png")
    ap.add_argument("--window-ms", type=float, default=0.0,
                    help="only plot the first N ms (0 = all)")
    a = ap.parse_args()
    rows = load(a.csv)
    summarize(rows)
    plot(rows, a.out, a.window_ms)


if __name__ == "__main__":
    main()
