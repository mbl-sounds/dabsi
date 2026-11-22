"""Paper figures and spread tables of the room experiments:  python plot_room_results.py --snr 30

Convergence examples (per experiment, signal and median/mean over runs, at --snr) go to ``plots/`` as
conv_<exp>_<SNR>_<signal>_<agg>.pdf, and the 2x2 grids (experiment x signal) as conv_grid_<SNR>_<agg>.pdf.
Table goes to ``tables/`` as rooms_spread_table.tex: per experiment, mean +- std over runs of the final NPM [dB]
(non-diverged runs) and the convergence time [s] of the mean per-run NPM [dB] curve as plotted: first time it is at or below -10 dB.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import plot_results as pr
from dbsi import analysis as an

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ["simulated_rooms", "measured_rooms"]
EXPERIMENT_LABELS = {"simulated_rooms": "Simulated", "measured_rooms": "Measured"}
SIGNALS = ["white", "speech"]
AGGREGATES = ["median", "mean"]
CONV_DB = -10.0
EXCLUDE = {"dsaawet"}


def stats(g, name, tail):
    """Final NPM [dB] of the non-diverged runs (non-finite NPM in the last ``tail`` frames counts as diverged)."""
    db = an.curves(g, "npm", db=True, alg=name)
    tail_db = db[:, -tail:]
    per_run = np.nanmean(np.where(np.isfinite(tail_db), tail_db, np.nan), axis=1)
    return per_run[np.isfinite(per_run)]


def crossing(g, name, hop, every, window):
    """Time [s] at which the mean per-run NPM [dB] curve, smoothed as in the plots, first reaches CONV_DB; inf if never."""
    db = an.curves(g, "npm", db=True, alg=name)[:, ::every]
    if db.size == 0:
        return np.inf
    curve = pr.smooth(np.nanmean(db, axis=0), window)
    hit = np.flatnonzero(curve <= CONV_DB)
    return hit[0] * every * hop if hit.size else np.inf


def cell(v, digits=1, bold=False):
    if v.size == 0:
        return "--"
    sd = v.std(ddof=1) if v.size > 1 else 0.0
    text = f"{v.mean():.{digits}f}$\\pm${sd:.{digits}f}"
    return rf"\textbf{{{text}}}" if bold else text


def time_cell(t, bold=False):
    if not np.isfinite(t):
        return "--"
    text = f"{t:.1f}"
    return rf"\textbf{{{text}}}" if bold else text


def best(values):
    """Index of the smallest finite value, None if none is finite."""
    k = int(np.argmin(values))
    return k if np.isfinite(values[k]) else None


def block_rows(snrs, sets, order, tail, every, window):
    """Rows of one table half for the given SNR values, as (cells, rule after the row) pairs."""
    names = [o[4] for o in order]
    rows = []
    for snr in snrs:
        for i, sig in enumerate(SIGNALS):
            finals, convs = [], []
            for _, df, hop in sets:
                g = df.filter((pl.col("SNR") == snr) & (pl.col("signal") == sig))
                finals.append([stats(g, name, tail) for name in names])
                convs.append([crossing(g, name, hop, every, window) for name in names])
            cells = [[] for _ in names]
            for k in range(len(sets)):
                best_npm = best([f.mean() if f.size else np.inf for f in finals[k]])
                best_conv = best(convs[k])
                for j in range(len(names)):
                    cells[j] += [
                        cell(finals[k][j], bold=j == best_npm),
                        time_cell(convs[k][j], bold=j == best_conv),
                    ]
            for j, (_, _, tex, _, _) in enumerate(order):
                snr_cell = (
                    rf"\multirow{{{len(SIGNALS) * len(order)}}}{{*}}{{{snr:g}}}"
                    if i == 0 and j == 0
                    else ""
                )
                sig_cell = rf"\multirow{{{len(order)}}}{{*}}{{{sig}}}" if j == 0 else ""
                rule = None
                if j == len(order) - 1:
                    rule = "snr" if i == len(SIGNALS) - 1 else "sig"
                rows.append(([snr_cell, sig_cell, tex, *cells[j]], rule))
    return rows


def table(sets, order, tail, every, window, caption):
    """Two column halves side by side, each with one column pair per (label, df, hop) in ``sets``."""
    snrs = sorted({s for _, df, _ in sets for s in df["SNR"].unique().to_list()})
    if len(snrs) % 2:
        raise ValueError("need an even number of SNR values")
    half = len(snrs) // 2
    left = block_rows(snrs[:half], sets, order, tail, every, window)
    right = block_rows(snrs[half:], sets, order, tail, every, window)
    width = 3 + 2 * len(sets)
    ncol = 2 * width + 1  # two halves plus an empty spacer column between them
    spec = "c" + "l" * (width - 1)
    head1 = [rf"\multirow{{2}}{{*}}{{{t}}}" for t in ("SNR [dB]", "Signal", "Algorithm")]
    head1 += [rf"\multicolumn{{2}}{{c}}{{{label}}}" for label, _, _ in sets]
    head2 = ["", "", ""] + ["NPM [dB]", "Conv. [s]"] * len(sets)
    # rules run only over the data columns of each half, never into the spacer
    cline = "".join(
        rf"\cline{{{s + 4 + 2 * k}-{s + 5 + 2 * k}}}"
        for s in (0, width + 1)
        for k in range(len(sets))
    )
    rules = {
        "snr": rf"\cline{{1-{width}}}\cline{{{width + 2}-{ncol}}}",
        "sig": rf"\cline{{2-{width}}}\cline{{{width + 3}-{ncol}}}",
        "end": r"\hline",
    }
    lines = [
        f"% {caption}",
        rf"\begin{{tabular}}{{{spec}p{{0.5em}}{spec}}}",
        r"\hline",
        " & ".join(head1 + [""] + head1) + r" \\",
        cline + r"\noalign{\vskip 2pt}",
        " & ".join(head2 + [""] + head2) + r" \\",
        r"\hline",
    ]
    for n, ((lc, rule), (rc, _)) in enumerate(zip(left, right)):
        lines.append(" & ".join(lc + [""] + rc) + r" \\")
        if n == len(left) - 1:
            rule = "end"
        if rule:
            lines.append(rules[rule])
    lines += [r"\end{tabular}", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--snr", type=float, default=30, help="SNR of the convergence examples [dB]"
    )
    p.add_argument(
        "--lambd",
        type=float,
        default=0.0,
        help="regularisation of the tagged DABSI variant (dabsi_l<lambd>)",
    )
    p.add_argument(
        "--tail", type=int, default=50, help="frames averaged for the final NPM"
    )
    p.add_argument("--every", type=int, default=10, help="plot every n-th frame")
    p.add_argument(
        "--smooth", type=int, default=21, help="Savitzky-Golay window (odd, 0 = off)"
    )
    p.add_argument("--plots", default=str(ROOT / "plots"))
    p.add_argument("--tables", default=str(ROOT / "tables"))
    a = p.parse_args()
    if a.smooth and a.smooth % 2 == 0:
        p.error("--smooth must be odd")

    plots, tables = Path(a.plots), Path(a.tables)
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for name in EXPERIMENTS:
        exp_dir = ROOT / ".data" / name
        df = an.load(str(exp_dir))
        hop = pr.hop_seconds(exp_dir)
        order = [
            o
            for o in pr.ordered(df["alg"].unique().to_list(), a.lambd)
            if o[4] not in EXCLUDE
        ]
        if not order:
            p.error(f"{name}: no algorithm at lambd={a.lambd:g}")
        for sig in SIGNALS:
            g = df.filter((pl.col("SNR") == a.snr) & (pl.col("signal") == sig))
            for agg in AGGREGATES:
                fig = pr.conv_figure(
                    g, order, a.every, a.smooth, hop, agg, info=f"SNR {a.snr:g} dB, {sig}"
                )
                fig.savefig(
                    plots / f"conv_{name}_{a.snr:g}_{sig}_{agg}.pdf", bbox_inches="tight"
                )
                plt.close(fig)
        loaded[name] = (df, hop, order)
        print(f"{name}: algorithms {', '.join(o[4] for o in order)}")

    names = [o[4] for o in loaded[EXPERIMENTS[0]][2]]
    for name in EXPERIMENTS[1:]:
        if [o[4] for o in loaded[name][2]] != names:
            p.error(f"{name}: algorithms differ from {EXPERIMENTS[0]}")
    sets = [
        (EXPERIMENT_LABELS[name], loaded[name][0], loaded[name][1])
        for name in EXPERIMENTS
    ]
    caption = "final NPM [dB] mean $\\pm$ std over runs, convergence time [s] of the mean per-run NPM [dB] curve"
    text = table(sets, loaded[EXPERIMENTS[0]][2], a.tail, a.every, a.smooth, caption)
    (tables / "rooms_spread_table.tex").write_text(text)

    for agg in AGGREGATES:
        cells = [
            [
                (
                    loaded[name][0].filter(
                        (pl.col("SNR") == a.snr) & (pl.col("signal") == sig)
                    ),
                    loaded[name][1],
                    f"{EXPERIMENT_LABELS[name]}, {sig}",
                )
                for sig in SIGNALS
            ]
            for name in EXPERIMENTS
        ]
        fig = pr.conv_grid(cells, loaded[EXPERIMENTS[0]][2], a.every, a.smooth, agg)
        fig.savefig(plots / f"conv_grid_{a.snr:g}_{agg}.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
