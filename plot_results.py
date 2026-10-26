"""Figures and result tables of experiments:  python plot_results.py .data/wgn_rooms .data/simulated_rooms

Figures (names as cited in the paper) go to ``--plots`` (default: repository ``plots/``):
  conv_<exp>[_<varying params>]_<SNR>_<signal>.pdf   median NPM (solid) and NPMA (dashed) over time [s]
  npm_<exp>[_<varying params>]_<signal>.pdf          final NPM over SNR, median and IQR over runs

Tables go to ``--tables`` (default: repository ``tables/``):
  <exp>_summary.csv    all statistics per grid point and algorithm (analysis.summary)
  <exp>_tables.tex     tabular blocks over SNR per signal: final NPM [dB] and convergence time [s]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

from dbsi import analysis as an

ROOT = Path(__file__).resolve().parent
FIGSIZE = (3.4, 2.6)  # one IEEE column (245 pt)
# base name, letter, plot label, LaTeX label, colour (letters as in the figure captions)
ALGS = [
    ("dabsi", "A", "DABSI", r"\gls{dabsi}", "tab:blue"),
    ("mcqn", "B", "MCQN", r"\gls{mcqn}", "tab:orange"),
    ("spmcqn", "C", "S-MCQN", r"S-\gls{mcqn}", "tab:green"),
    ("dgd", "D", "DGD", r"\gls{dgd}", "tab:red"),
    ("dsaawet", "E", "DSAAWET", r"\gls{dsaawet}", "tab:purple"),
]
# markers as in plot_complexity_scaling.py; S-MCQN has no entry there
MARKERS = {"A": "o", "B": "s", "C": "^", "D": "d", "E": "+"}


def variant(names, base, lambd):
    """Algorithm name of ``base`` at regularisation ``lambd`` (tagged ``<base>_l<lambd>``), else untagged."""
    for name in (f"{base}_l{lambd:g}", base):
        if name in names:
            return name
    return None


def ordered(names, lambd):
    """(letter, plot label, LaTeX label, colour, alg name) of the algorithms present, in paper order."""
    out = []
    for base, letter, label, tex, color in ALGS:
        name = variant(names, base, lambd)
        if name is not None:
            out.append((letter, label, tex, color, name))
    return out


def fmt(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def tag(point, vary, snr=True):
    """File tag: varying grid parameters (except SNR and signal), then SNR (optional) and signal."""
    parts = [f"{k}{fmt(point[k])}" for k in vary if k not in ("SNR", "signal")]
    if snr and "SNR" in point:
        parts.append(fmt(point["SNR"]))
    if "signal" in point:
        parts.append(str(point["signal"]))
    return "_".join(parts)


def smooth(y, window):
    return savgol_filter(y, window, 1) if window >= 3 and len(y) > window else y


def hop_seconds(exp_dir):
    """Duration of one signal frame (hop L at sampling rate fs), from the experiment manifest."""
    cfg = json.loads((Path(exp_dir) / "manifest.json").read_text())["config"]
    return cfg["L"] / cfg["scenario"]["fs"]


def conv_figure(g, order, every, window, hop):
    """Median NPM (solid) and NPMA (dashed) in dB over time, all algorithms of one grid point."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for letter, _, _, color, name in order:
        for metric, ls in (("npm", "-"), ("npma", "--")):
            lin = an.curves(g, metric, db=False, alg=name)[:, ::every]
            if lin.size == 0:
                continue
            med = smooth(np.nanmedian(lin, axis=0), window)
            ax.plot(
                np.arange(med.size) * every * hop,
                20 * np.log10(med),
                ls,
                color=color,
                lw=1,
                marker=MARKERS[letter],
                markevery=max(1, med.size // 10),
                markersize=5,
                markerfacecolor="white",
            )
    handles = [
        Line2D(
            [0],
            [0],
            color=c,
            marker=MARKERS[letter],
            markerfacecolor="white",
            label=f"{letter} {label}",
        )
        for letter, label, _, c, _ in order
    ]
    handles += [
        Line2D([0], [0], color="k", ls="-", label="NPM"),
        Line2D([0], [0], color="k", ls="--", label=r"NPM$_{\mathrm{avg}}$"),
    ]
    ax.set(xlabel="Time [s]", ylabel=r"NPM$^{(\tau)}$ [dB]")
    ax.set_ylim(top=0)
    ax.grid(alpha=0.3)
    ax.legend(handles=handles, fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def final_figure(g, order):
    """Final NPM [dB] (median and IQR over runs) over SNR, all algorithms of one signal and setting."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for letter, label, _, color, name in order:
        s = g.filter(g["alg"] == name).sort("SNR")
        if s.height == 0:
            continue
        med, q1, q3 = s["median"].to_numpy(), s["q1"].to_numpy(), s["q3"].to_numpy()
        ax.errorbar(
            s["SNR"].to_numpy(),
            med,
            yerr=[med - q1, q3 - med],
            color=color,
            marker=MARKERS[letter],
            ms=3,
            mfc="white",
            capsize=2,
            lw=1,
            label=f"{letter} {label}",
        )
    ax.set(xlabel="SNR [dB]", ylabel="Final NPM [dB]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return fig


def latex_block(g, order, col, digits, snrs, caption, scale=1.0):
    """Bare ``tabular`` (rows: algorithms, columns: SNR) of column ``col`` of the summary, times ``scale``."""
    lines = [
        f"% {caption}",
        rf"\begin{{tabular}}{{l{'c' * len(snrs)}}}",
        r"\hline",
        "Algorithm & " + " & ".join(f"{s:g}" for s in snrs) + r" \\",
        r"\hline",
    ]
    for _, _, tex, _, name in order:
        cells = []
        for s in snrs:
            r = g.filter((g["alg"] == name) & (g["SNR"] == s))
            v = float(r[col][0]) * scale if r.height else math.nan
            cells.append(f"{v:.{digits}f}" if math.isfinite(v) else "--")
        lines.append(f"{tex} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def process(exp_dir, a, plots, tables):
    name = Path(exp_dir).name
    df = an.load(exp_dir)
    keys = an.params(df)
    vary = [k for k in keys if df[k].n_unique() > 1]
    order = ordered(df["alg"].unique().to_list(), a.lambd)
    if not order:
        print(f"{name}: no algorithm at lambd={a.lambd:g}, skipped")
        return

    summ = an.summary(df, "npm", tail=a.tail)
    summ.write_csv(tables / f"{name}_summary.csv")

    hop = hop_seconds(exp_dir)
    for key, g in df.group_by(keys, maintain_order=True):
        point = dict(zip(keys, key))
        fig = conv_figure(g, order, a.every, a.smooth, hop)
        fig.savefig(plots / f"conv_{name}_{tag(point, vary)}.pdf", bbox_inches="tight")
        plt.close(fig)

    if "SNR" not in keys:
        print(f"{name}: no SNR grid parameter, final-NPM plots and tables skipped")
        return
    by = [k for k in keys if k != "SNR" and (k in vary or k == "signal")]
    groups = summ.group_by(by, maintain_order=True) if by else [((), summ)]
    snrs = sorted(summ["SNR"].unique().to_list())
    blocks = []
    for key, g in groups:
        t = tag(dict(zip(by, key)), vary, snr=False)
        fig = final_figure(g, order)
        fig.savefig(plots / f"npm_{name}_{t}.pdf", bbox_inches="tight")
        plt.close(fig)
        blocks.append(
            latex_block(
                g,
                order,
                "median",
                1,
                snrs,
                f"{name} {t}: final NPM [dB], median over runs",
            )
        )
        blocks.append(
            latex_block(
                g,
                order,
                "conv_frame",
                2,
                snrs,
                f"{name} {t}: convergence time [s], first time below -10 dB",
                scale=hop,
            )
        )
    (tables / f"{name}_tables.tex").write_text("\n".join(blocks))
    print(f"{name}: algorithms {', '.join(o[4] for o in order)}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "exp_dirs", nargs="+", help="experiment folders written by run_experiments.py"
    )
    p.add_argument(
        "--lambd",
        type=float,
        default=1e-5,
        help="regularisation of the tagged variants (<alg>_l<lambd>)",
    )
    p.add_argument("--plots", default=str(ROOT / "plots"))
    p.add_argument("--tables", default=str(ROOT / "tables"))
    p.add_argument("--every", type=int, default=10, help="plot every n-th frame")
    p.add_argument(
        "--smooth",
        type=int,
        default=21,
        help="Savitzky-Golay window on the plotted samples (odd, 0 = off)",
    )
    p.add_argument(
        "--tail", type=int, default=50, help="frames averaged for the final NPM"
    )
    a = p.parse_args()
    if a.smooth and a.smooth % 2 == 0:
        p.error("--smooth must be odd")

    plots, tables = Path(a.plots), Path(a.tables)
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for exp_dir in a.exp_dirs:
        process(exp_dir, a, plots, tables)


if __name__ == "__main__":
    main()
