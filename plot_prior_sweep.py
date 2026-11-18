"""Prior mismatch sweep: final NPM over lambda.  Usage: python plot_prior_sweep.py .data/prior_mismatch_sweep

Figures go to ``--plots`` as prior_mismatch_sweep_<grid>.pdf (one panel per SNR).
Tables go to ``--tables``: prior_mismatch_sweep_<grid>.tex (median final NPM [dB], IQR in brackets)
and prior_mismatch_sweep_summary.csv (all statistics per grid point and algorithm).
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from dbsi import analysis as an

ROOT = Path(__file__).resolve().parent
ALG = re.compile(r"dabsi_(noprior|matched|slight|distinct|lowpass)(?:_l([0-9.e+-]+))?$")
# prior, label, colour, marker; the no-prior run is a reference line
PRIORS = [
    ("matched", "Matched", "tab:blue", "o"),
    ("slight", "Slight", "tab:green", "s"),
    ("distinct", "Distinct", "tab:red", "^"),
    ("lowpass", "Low-pass 250 Hz", "tab:purple", "d"),
]
TABLE_ROWS = [("noprior", "No prior")] + [(p, n) for p, n, _, _ in PRIORS]


def tagged(summ: pl.DataFrame) -> pl.DataFrame:
    """Summary rows with ``prior`` and ``lambd`` parsed from the algorithm name (no prior: lambd 0)."""
    rows = []
    for r in summ.to_dicts():
        m = ALG.match(r["alg"])
        if m:
            rows.append({**r, "prior": m.group(1), "lambd": float(m.group(2) or 0.0)})
    return pl.DataFrame(rows)


def point(g: pl.DataFrame, prior: str, snr) -> pl.DataFrame:
    return g.filter((pl.col("prior") == prior) & (pl.col("SNR") == snr)).sort("lambd")


def fmt(k, v) -> str:
    return f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}"


def figure(g: pl.DataFrame, snrs):
    fig, axes = plt.subplots(
        1, len(snrs), figsize=(400 / 72.27, 400 / 72.27 * 0.4), sharey=True, squeeze=False
    )
    axes = axes[0]
    for i, (ax, snr) in enumerate(zip(axes, snrs)):
        ref = point(g, "noprior", snr)
        if ref.height:
            ax.axhline(ref["median"][0], color="k", ls=":", lw=1, label="No prior")
        for prior, name, color, marker in PRIORS:
            s = point(g, prior, snr)
            if s.height == 0:
                continue
            med = s["median"].to_numpy()
            ax.errorbar(
                s["lambd"].to_numpy(),
                med,
                yerr=[med - s["q1"].to_numpy(), s["q3"].to_numpy() - med],
                color=color,
                linestyle="-",
                marker=marker,
                markersize=5,
                markerfacecolor="white",
                capsize=2,
                lw=1,
                label=name,
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"Regularization $\lambda_0$")
        ax.text(
            0.03,
            0.95,
            rf"({chr(97 + i)}) SNR ${snr:g}$ dB",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        ax.tick_params(axis="both", which="major", labelsize=8, length=2.5, pad=2)
        ax.tick_params(axis="both", which="minor", length=1.5)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.45)
        ax.grid(True, which="minor", alpha=0.10, linewidth=0.35)
        ax.margins(x=0.02)
    axes[0].set_ylabel("Final NPM [dB]", labelpad=1)
    for ax in axes[1:]:
        ax.tick_params(axis="y", which="both", labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.12, top=0.92, wspace=0.05)
    return fig


def cell(g: pl.DataFrame, prior: str, snr, lam: float) -> str:
    lam = 0.0 if prior == "noprior" else lam
    r = g.filter(
        (pl.col("prior") == prior) & (pl.col("SNR") == snr) & (pl.col("lambd") == lam)
    )
    if r.height == 0:
        return "--"
    med, q1, q3 = r["median"][0], r["q1"][0], r["q3"][0]
    return f"{med:.1f} [{(q3 - q1) / 2:.1f}]"


def table(g: pl.DataFrame, snrs, lambdas, title: str) -> str:
    blocks = []
    for snr in snrs:
        lines = [
            f"% {title}, SNR {snr:g} dB: median final NPM [dB], IQR half-width in brackets",
            rf"\begin{{tabular}}{{l{'c' * len(lambdas)}}}",
            r"\hline",
            r"Prior & " + " & ".join(f"{lam:g}" for lam in lambdas) + r" \\",
            r"\hline",
        ]
        for prior, name in TABLE_ROWS:
            cells = [cell(g, prior, snr, lam) for lam in lambdas]
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        lines += [r"\hline", r"\end{tabular}", ""]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("exp_dir", help="experiment folder written by run_experiments.py")
    p.add_argument(
        "--tail", type=int, default=50, help="frames averaged for the final NPM"
    )
    p.add_argument("--plots", default=str(ROOT / "plots"))
    p.add_argument("--tables", default=str(ROOT / "tables"))
    a = p.parse_args()

    name = Path(a.exp_dir).name
    plots, tables = Path(a.plots), Path(a.tables)
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,
            "pgf.rcfonts": False,
            "font.size": 9,
        }
    )
    df = an.load(a.exp_dir)
    summ = tagged(an.summary(df, "npm", tail=a.tail))
    if summ.height == 0:
        raise SystemExit("no prior_mismatch algorithm names found in the results")
    summ.write_csv(tables / f"{name}_summary.csv")

    failed = int((summ["n_total"] - summ["n_valid"]).sum())
    print(
        f"runs per algorithm and SNR: {summ['n_total'].min()}..{summ['n_total'].max()}, "
        f"invalid runs: {failed}, algorithms: {summ['alg'].n_unique()}"
    )

    lambdas = sorted(v for v in summ["lambd"].unique().to_list() if v > 0)
    snrs = sorted(summ["SNR"].unique().to_list())
    keys = [k for k in an.params(df) if k != "SNR"]
    for key, g in summ.group_by(keys, maintain_order=True):
        point_name = "_".join(fmt(k, v) for k, v in zip(keys, key))
        title = " ".join(f"{k}={v}" for k, v in zip(keys, key))
        fig = figure(g, snrs)
        fig.savefig(plots / f"{name}_{point_name}.pdf", bbox_inches="tight")
        plt.close(fig)
        (tables / f"{name}_{point_name}.tex").write_text(table(g, snrs, lambdas, title))
        print(f"wrote {name}_{point_name}: lambda {lambdas}, SNR {snrs}")


if __name__ == "__main__":
    main()
