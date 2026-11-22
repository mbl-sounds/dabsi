"""Scalability: mean NPM [dB] over time, colour = network size M, linestyle = neighbourhood size N_i.

Usage: python plot_scalability.py .data/scalability

Left panel: mean NPM over time. Right panel: final NPM mean and std over runs versus M.

Figure goes to ``--plots`` as conv_scalability.pdf. Curves are smoothed and subsampled as in plot_results.py.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dbsi import analysis as an
from plot_results import (
    FIG_RATIO_ONE_ROW,
    FIG_WIDTH_PT,
    STYLE,
    _style,
    hop_seconds,
    smooth,
)

ROOT = Path(__file__).resolve().parent
ALG = "dabsi"
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
MARKERS = ["o", "s", "^", "D", "v"]
LINESTYLES = ["-", "--", ":", "-."]
LEGEND_GAP_PT = 4


def curves(df, every, window, hop):
    """(M, N_i) -> (time [s], NPM [dB]) of the per-run NPM [dB] averaged over runs."""
    out = {}
    for key, g in df.filter(df["alg"] == ALG).group_by(["M", "N_i"], maintain_order=True):
        db = an.curves(g, "npm", db=True)[:, ::every]
        mean = smooth(np.nanmean(db, axis=0), window)
        out[(key[0], key[1])] = (np.arange(mean.size) * every * hop, mean)
    return out


def final(df, tail):
    """N_i -> (M, mean [dB], std [dB]) of the final NPM over runs (mean of the last ``tail`` frames)."""
    s = an.summary(df.filter(df["alg"] == ALG), tail=tail)
    out = {}
    for n in sorted(s["N_i"].unique().to_list()):
        t = s.filter(s["N_i"] == n).sort("M")
        out[n] = (t["M"].to_numpy(), t["mean"].to_numpy(), t["std"].to_numpy())
    return out


def figure(data, finals, ms, ns):
    size_in = (FIG_WIDTH_PT / 72.27, FIG_WIDTH_PT / 72.27 * FIG_RATIO_ONE_ROW)
    with plt.rc_context(STYLE):
        fig, (ax, axf) = plt.subplots(1, 2, figsize=size_in, sharey=True)
        for (m, n), (t, db) in data.items():
            ax.plot(
                t,
                db,
                color=COLORS[ms.index(m)],
                ls=LINESTYLES[ns.index(n)],
            )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"NPM$^{(\tau)}$ [dB]", labelpad=0)
        ax.set_ylim(top=0)
        _style(ax)
        outline = [pe.withStroke(linewidth=2.2, foreground="white")]
        for i, n in enumerate(ns):
            m_vals, mean, std = finals[n]
            x = m_vals + 0.15 * (i - 1)
            axf.plot(
                x, mean, color="k", ls=LINESTYLES[i], lw=1, path_effects=outline, zorder=1
            )
            for k, m in enumerate(m_vals):
                eb = axf.errorbar(
                    x[k],
                    mean[k],
                    yerr=std[k],
                    color=COLORS[ms.index(int(m))],
                    marker=MARKERS[i],
                    ms=5,
                    mfc="white",
                    capsize=2,
                    elinewidth=0.6,
                )
                for art in (eb.lines[0], *eb.lines[1]):
                    art.set_path_effects(outline)
        axf.set_xlabel("Network size $M$")
        _style(axf)
        axf.margins(x=0.08)
        axf.set_xticks(ms)
        size = [
            Line2D([0], [0], color=COLORS[i], label=rf"$M={m}$") for i, m in enumerate(ms)
        ]
        nbhd = [
            Line2D(
                [0],
                [0],
                color="k",
                ls=LINESTYLES[i],
                marker=MARKERS[i],
                ms=5,
                mfc="white",
                label=rf"$N_i={n}$",
            )
            for i, n in enumerate(ns)
        ]
        leg = fig.legend(
            handles=size + nbhd,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(size) + len(nbhd),
            frameon=False,
            fontsize=8,
            handlelength=1.5,
            handletextpad=0.4,
            columnspacing=0.8,
        )
        fig.subplots_adjust(left=0.14, right=0.985, bottom=0.2, wspace=0.12)
        fig.canvas.draw()
        legend_bottom = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
        fig.subplots_adjust(top=legend_bottom - LEGEND_GAP_PT / (size_in[1] * 72.27))
    return fig


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("exp_dir", help="scalability experiment folder")
    p.add_argument("--plots", default=str(ROOT / "plots"))
    p.add_argument("--every", type=int, default=10, help="plot every n-th frame")
    p.add_argument("--smooth", type=int, default=21, help="Savitzky-Golay window (odd)")
    p.add_argument(
        "--tail", type=int, default=50, help="frames averaged for the final NPM"
    )
    a = p.parse_args()
    if a.smooth % 2 == 0:
        p.error("--smooth must be odd")

    df = an.load(a.exp_dir)
    hop = hop_seconds(a.exp_dir)
    data = curves(df, a.every, a.smooth, hop)
    finals = final(df, a.tail)
    ms = sorted({m for m, _ in data})
    ns = sorted({n for _, n in data})
    if len(ms) > len(COLORS) or len(ns) > min(len(LINESTYLES), len(MARKERS)):
        raise ValueError("more grid values than colours or linestyles")

    plots = Path(a.plots)
    plots.mkdir(parents=True, exist_ok=True)
    fig = figure(data, finals, ms, ns)
    fig.savefig(plots / "conv_scalability.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"M {ms}, N_i {ns}, {len(data)} curves -> {plots / 'conv_scalability.pdf'}")


if __name__ == "__main__":
    main()
